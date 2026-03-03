"""
Denoising mixin and helpers for SVI sampling results.
"""

from typing import Dict, Optional, Union

import jax.numpy as jnp
from jax import random

from ..sampling import denoise_counts


def _slice_param_for_dataset(
    param: Optional[jnp.ndarray],
    dataset_idx: int,
    n_datasets: int,
) -> Optional[jnp.ndarray]:
    """Extract a single dataset's slice from a per-dataset parameter.

    If the parameter's leading axis matches ``n_datasets``, returns
    ``param[dataset_idx]``; otherwise returns the parameter unchanged
    (it is shared across datasets).

    Handles both gene-specific per-dataset params with shape
    ``(n_datasets, n_genes)`` and scalar per-dataset params with
    shape ``(n_datasets,)``.

    Parameters
    ----------
    param : jnp.ndarray or None
        Parameter array, e.g. ``(n_datasets, n_genes)``, ``(n_datasets,)``,
        or ``(n_genes,)``.
    dataset_idx : int
        Which dataset to select.
    n_datasets : int
        Total number of datasets (used to detect the dataset axis).

    Returns
    -------
    jnp.ndarray or None
        The single-dataset slice, or ``None`` if input was ``None``.
    """
    if param is None:
        return None
    # Matches both 2D gene-specific per-dataset params (n_datasets, n_genes)
    # and 1D scalar per-dataset params (n_datasets,).  Safe because
    # n_datasets << n_genes in any realistic single-cell dataset.
    if param.ndim >= 1 and param.shape[0] == n_datasets:
        return param[dataset_idx]
    return param


def _denoise_per_dataset(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    gate: Optional[jnp.ndarray],
    p_capture: Optional[jnp.ndarray],
    dataset_indices: jnp.ndarray,
    n_datasets: int,
    method,
    rng_key,
    return_variance: bool,
    mixing_weights: Optional[jnp.ndarray],
    cell_batch_size: Optional[int],
) -> "Union[jnp.ndarray, Dict[str, jnp.ndarray]]":
    """Denoise a multi-dataset model by processing each dataset separately.

    For each dataset ``d``, extracts the cells belonging to it (via
    ``dataset_indices``) and the corresponding single-dataset parameters,
    calls :func:`~scribe.sampling.denoise_counts`, and reassembles the
    results into the original cell order.

    This avoids the shape ambiguity in ``denoise_counts`` where
    ``(n_cells, n_genes)`` per-cell parameters would be misinterpreted
    as a multi-sample leading dimension.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion - ``(n_datasets, n_genes)`` or ``(n_genes,)``.
    p : jnp.ndarray
        Success probability - ``(n_datasets, n_genes)``, ``(n_genes,)``,
        or scalar.
    gate : jnp.ndarray or None
        Gate - ``(n_datasets, n_genes)``, ``(n_genes,)``, or ``None``.
    p_capture : jnp.ndarray or None
        Per-cell capture probability ``(n_cells,)`` or ``None``.
    dataset_indices : jnp.ndarray
        Per-cell dataset assignment ``(n_cells,)``.
    n_datasets : int
        Number of datasets.
    method : str or tuple
        Denoising method forwarded to ``denoise_counts``.
    rng_key : random.PRNGKey or None
        PRNG key.
    return_variance : bool
        Whether to return variance alongside denoised counts.
    mixing_weights : jnp.ndarray or None
        Mixture weights (forwarded unchanged to ``denoise_counts``).
    cell_batch_size : int or None
        Cell batching for memory control.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts ``(n_cells, n_genes)`` in original cell order.
    """
    n_cells, n_genes = counts.shape
    denoised_out = jnp.empty((n_cells, n_genes), dtype=counts.dtype)
    variance_out = (
        jnp.empty((n_cells, n_genes), dtype=jnp.float32)
        if return_variance else None
    )

    for d in range(n_datasets):
        mask = dataset_indices == d
        idx = jnp.where(mask)[0]
        if idx.shape[0] == 0:
            continue

        counts_d = counts[idx]
        r_d = _slice_param_for_dataset(r, d, n_datasets)
        p_d = _slice_param_for_dataset(p, d, n_datasets)
        gate_d = _slice_param_for_dataset(gate, d, n_datasets)
        pc_d = p_capture[idx] if p_capture is not None else None

        if rng_key is not None:
            rng_key, d_key = random.split(rng_key)
        else:
            d_key = None

        result_d = denoise_counts(
            counts=counts_d,
            r=r_d,
            p=p_d,
            p_capture=pc_d,
            gate=gate_d,
            method=method,
            rng_key=d_key,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if return_variance:
            denoised_out = denoised_out.at[idx].set(
                result_d["denoised_counts"]
            )
            variance_out = variance_out.at[idx].set(result_d["variance"])
        else:
            denoised_out = denoised_out.at[idx].set(result_d)

    if return_variance:
        return {"denoised_counts": denoised_out, "variance": variance_out}
    return denoised_out


class DenoisingSamplingMixin:
    """Mixin providing Bayesian denoising methods for observed counts."""

    def denoise_counts_map(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        use_mean: bool = True,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using MAP parameter estimates.

        Computes the posterior of true (pre-capture, pre-dropout)
        transcript counts for each cell and gene, using point estimates
        of the model parameters.  For VCP models this accounts for the
        per-cell capture probability; for ZINB variants it additionally
        corrects zero observations for dropout.  For NBDM the result is
        the identity (``denoised == counts``).

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix of shape ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, optional
            Summary of the denoised posterior to return.

            * ``'mean'``: closed-form posterior mean (shrinkage estimator).
            * ``'mode'``: posterior mode (MAP denoised count).
            * ``'sample'``: one stochastic draw per cell/gene.

            Default: ``'mean'``.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Required when ``method='sample'``.
            Defaults to ``random.PRNGKey(42)`` when ``None``.
        return_variance : bool, optional
            If ``True``, return a dictionary with ``'denoised_counts'``
            and ``'variance'`` keys.  Default: ``False``.
        cell_batch_size : int or None, optional
            Process cells in batches of this size to limit memory.
            ``None`` processes all cells at once.
        use_mean : bool, optional
            If ``True``, replaces undefined MAP values (NaN) with
            posterior means.  Default: ``True``.
        store_result : bool, optional
            If ``True``, stores the denoised counts in
            ``self.denoised_counts``.  Default: ``True``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised count matrix of shape ``(n_cells, n_genes)`` (or
            dict with variance when ``return_variance=True``).

        See Also
        --------
        denoise_counts_posterior : Full-posterior Bayesian denoising.
        get_map_ppc_samples_biological : MAP-based biological PPC.
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        if verbose:
            print("Getting MAP estimates for denoising...")

        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=False,
            counts=counts,
        )

        r = map_estimates.get("r")
        p = map_estimates.get("p")
        if r is None or p is None:
            raise ValueError(
                "Could not extract r and p from MAP estimates. "
                f"Available keys: {list(map_estimates.keys())}"
            )

        p_capture = map_estimates.get("p_capture")
        gate = map_estimates.get("gate")
        is_mixture = self.n_components is not None and self.n_components > 1
        mixing_weights = (
            map_estimates.get("mixing_weights") if is_mixture else None
        )

        if verbose:
            model_desc = (
                f"mixture ({self.n_components} components)"
                if is_mixture
                else "standard"
            )
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            print(
                f"Denoising {model_desc} model "
                f"({self.model_type}){extra_str}, method='{method}'..."
            )

        # Multi-dataset models: per-dataset parameters have shape
        # (n_datasets, n_genes).  We denoise each dataset's cells
        # separately to avoid shape ambiguity in denoise_counts, which
        # would misinterpret (n_cells, n_genes) as a sample dimension.
        n_ds = getattr(self.model_config, "n_datasets", None)
        ds_idx = getattr(self, "_dataset_indices", None)
        if n_ds is not None and ds_idx is not None:
            result = _denoise_per_dataset(
                counts=counts,
                r=r,
                p=p,
                gate=gate,
                p_capture=p_capture,
                dataset_indices=ds_idx,
                n_datasets=n_ds,
                method=method,
                rng_key=rng_key,
                return_variance=return_variance,
                mixing_weights=mixing_weights,
                cell_batch_size=cell_batch_size,
            )
        else:
            result = denoise_counts(
                counts=counts,
                r=r,
                p=p,
                p_capture=p_capture,
                gate=gate,
                method=method,
                rng_key=rng_key,
                return_variance=return_variance,
                mixing_weights=mixing_weights,
                cell_batch_size=cell_batch_size,
            )

        if verbose:
            shape = (
                result["denoised_counts"].shape
                if return_variance
                else result.shape
            )
            print(f"Denoised counts shape: {shape}")

        if store_result:
            self.denoised_counts = (
                result["denoised_counts"] if return_variance else result
            )

        return result

    def denoise_counts_posterior(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using full posterior samples.

        Propagates parameter uncertainty by repeating the denoising
        computation for each draw from the variational posterior.  The
        result has a leading ``n_samples`` dimension that can be
        summarised (e.g. ``.mean(axis=0)`` for a fully Bayesian point
        estimate, or quantiles for credible intervals).

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, optional
            Summary of the per-sample denoised posterior.
            Default: ``'mean'``.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        n_samples : int, optional
            Number of posterior samples to draw from the guide if
            ``self.posterior_samples`` is ``None``.  Default: 100.
        batch_size : int or None, optional
            Batch size for posterior sampling (passed to
            :meth:`get_posterior_samples`).
        return_variance : bool, optional
            If ``True``, return dict with ``'denoised_counts'`` and
            ``'variance'``.  Default: ``False``.
        cell_batch_size : int or None, optional
            Cell batching inside each posterior draw.
        store_result : bool, optional
            Store result in ``self.denoised_counts``.  Default: ``True``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised counts with shape ``(n_samples, n_cells, n_genes)``
            (or dict with variance when ``return_variance=True``).

        See Also
        --------
        denoise_counts_map : MAP-based denoising (single point estimate).
        get_ppc_samples_biological : Full-posterior biological PPC.
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Ensure posterior samples exist
        if self.posterior_samples is None:
            key_post, rng_key = random.split(rng_key)
            if verbose:
                print("Drawing posterior samples...")
            self.get_posterior_samples(
                rng_key=key_post,
                n_samples=n_samples,
                batch_size=batch_size,
                store_samples=True,
                counts=counts,
            )

        r = self.posterior_samples["r"]
        p = self.posterior_samples["p"]
        p_capture = self.posterior_samples.get("p_capture")
        gate = self.posterior_samples.get("gate")
        is_mixture = self.n_components is not None and self.n_components > 1
        mixing_weights = (
            self.posterior_samples.get("mixing_weights")
            if is_mixture
            else None
        )

        if verbose:
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            n_post = r.shape[0]
            print(
                f"Denoising with {n_post} posterior samples"
                f" ({self.model_type}){extra_str}, method='{method}'..."
            )

        _, key_denoise = random.split(rng_key)
        result = denoise_counts(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=key_denoise,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if verbose:
            shape = (
                result["denoised_counts"].shape
                if return_variance
                else result.shape
            )
            print(f"Denoised counts shape: {shape}")

        if store_result:
            self.denoised_counts = (
                result["denoised_counts"] if return_variance else result
            )

        return result

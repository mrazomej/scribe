"""
Sampling mixin for MCMC results.

Provides posterior predictive checks (standard and biological), Bayesian
denoising, and prior predictive sampling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import random

from ..models.config import ModelConfig
from ..sampling import (
    generate_predictive_samples,
    generate_prior_predictive_samples,
    sample_biological_nb,
    denoise_counts as _denoise_counts_util,
)

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


# ==============================================================================
# Sampling Mixin
# ==============================================================================


class SamplingMixin:
    """Mixin providing predictive sampling and denoising methods."""

    # --------------------------------------------------------------------------
    # Posterior predictive checks
    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate posterior predictive check (PPC) samples.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        batch_size : int or None, optional
            Batch size for sample generation.
        store_samples : bool, default=True
            Store result in ``self.predictive_samples``.

        Returns
        -------
        jnp.ndarray
            Posterior predictive samples.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        predictive_samples = _generate_ppc_samples(
            self.get_posterior_samples(),
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    def get_map_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 1,
        cell_batch_size: Optional[int] = None,
        use_mean: bool = True,
        store_samples: bool = True,
        verbose: bool = True,
        counts: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Generate MAP-based posterior predictive samples.

        This method builds a synthetic one-point posterior from the MAP estimate
        and feeds it through the same predictive sampler used by
        :meth:`get_ppc_samples`. The MAP point is repeated ``n_samples`` times so
        callers receive an array with shape ``(n_samples, n_cells, n_genes)``
        that mirrors the SVI API.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        n_samples : int, default=1
            Number of predictive draws to return from the MAP point.
        cell_batch_size : int or None, optional
            Batch size for predictive generation over cells.
        use_mean : bool, default=True
            Included for API parity with SVI. MCMC MAP extraction already
            includes a posterior-mean fallback when potential energy is
            unavailable, so this flag is currently informational.
        store_samples : bool, default=True
            Store generated samples in ``self.predictive_samples``.
        verbose : bool, default=True
            Print lightweight progress messages.
        counts : jnp.ndarray or None, optional
            Included for API parity with SVI. Not required for MCMC MAP PPC.

        Returns
        -------
        jnp.ndarray
            Predictive count samples with shape
            ``(n_samples, n_cells, n_genes)``.

        Notes
        -----
        Unlike full posterior PPC, this method does not integrate over all
        posterior draws; it conditions on a single MAP point estimate.
        """
        _ = use_mean
        _ = counts
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1 for MAP PPC sampling.")

        if verbose:
            print("Generating MCMC MAP PPC samples...")

        # Convert MAP parameter values to a pseudo-posterior with sample axis.
        map_estimates = self.get_map()
        map_parameter_samples = {}
        for key, value in map_estimates.items():
            value_array = jnp.asarray(value)
            value_with_sample_axis = jnp.expand_dims(value_array, axis=0)
            map_parameter_samples[key] = jnp.repeat(
                value_with_sample_axis, repeats=n_samples, axis=0
            )

        predictive_samples = _generate_ppc_samples(
            map_parameter_samples,
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=cell_batch_size,
        )

        if store_samples:
            self.predictive_samples = predictive_samples
        return predictive_samples

    # --------------------------------------------------------------------------
    # Biological (denoised) PPC
    # --------------------------------------------------------------------------

    def get_ppc_samples_biological(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        cell_batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """Generate biological PPC samples from base NB(r, p).

        Strips technical noise parameters (capture probability,
        zero-inflation gate) to yield counts reflecting the latent
        biology.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        batch_size : int or None, optional
            Kept for API symmetry; use ``cell_batch_size`` instead.
        store_samples : bool, default=True
            Store in ``self.predictive_samples_biological``.
        cell_batch_size : int or None, optional
            Process cells in batches of this size.

        Returns
        -------
        jnp.ndarray
            Biological count samples of shape
            ``(n_posterior_samples, n_cells, n_genes)``.

        See Also
        --------
        get_ppc_samples : Standard PPC including technical noise.
        scribe.sampling.sample_biological_nb : Core sampling utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        samples = self.get_posterior_samples()

        bio_samples = sample_biological_nb(
            r=samples["r"],
            p=samples["p"],
            n_cells=self.n_cells,
            rng_key=rng_key,
            mixing_weights=samples.get("mixing_weights"),
            cell_batch_size=cell_batch_size,
        )

        if store_samples:
            self.predictive_samples_biological = bio_samples

        return bio_samples

    # --------------------------------------------------------------------------
    # Bayesian denoising
    # --------------------------------------------------------------------------

    def denoise_counts(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using posterior samples.

        Propagates parameter uncertainty by computing the denoised
        posterior for each MCMC draw.

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, default='mean'
            Summary of the per-sample denoised posterior.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        return_variance : bool, default=False
            Return dict with ``'denoised_counts'`` and ``'variance'``.
        cell_batch_size : int or None, optional
            Cell batching inside each posterior draw.
        store_result : bool, default=True
            Store result in ``self.denoised_counts``.
        verbose : bool, default=True
            Print progress messages.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised counts ``(n_posterior_samples, n_cells, n_genes)``.

        See Also
        --------
        get_ppc_samples_biological : Biological PPC (unconditional).
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        samples = self.get_posterior_samples()

        r = samples["r"]
        p = samples["p"]
        p_capture = samples.get("p_capture")
        gate = samples.get("gate")
        mixing_weights = samples.get("mixing_weights")

        if verbose:
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            n_post = r.shape[0]
            print(
                f"Denoising with {n_post} MCMC samples"
                f" ({self.model_type}){extra_str}, method='{method}'..."
            )

        result = _denoise_counts_util(
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

    # --------------------------------------------------------------------------
    # Denoised AnnData export
    # --------------------------------------------------------------------------

    def get_denoised_anndata(
        self,
        counts: Optional[jnp.ndarray] = None,
        adata: Optional["AnnData"] = None,
        method: Union[str, Tuple[str, str]] = ("mean", "sample"),
        n_datasets: int = 1,
        rng_key: Optional[random.PRNGKey] = None,
        cell_batch_size: Optional[int] = None,
        include_original_counts: bool = True,
        path: Optional[str] = None,
        verbose: bool = True,
    ) -> Union["AnnData", List["AnnData"]]:
        """Export denoised counts as an AnnData object (optionally to h5ad).

        Runs Bayesian denoising on the observed counts and packages the
        result into an :class:`~anndata.AnnData` object with the original
        cell/gene metadata.  Supports generating multiple denoised
        realisations: dataset 1 uses the posterior mean of MCMC samples
        and subsequent datasets each use a different MCMC draw.

        Parameters
        ----------
        counts : jnp.ndarray or None, optional
            Observed UMI count matrix ``(n_cells, n_genes)``.  If ``None``,
            extracted from ``adata.X`` when ``adata`` is provided; otherwise
            an error is raised.
        adata : AnnData or None, optional
            Template AnnData whose ``.obs``, ``.var``, and ``.uns`` are
            copied into the output.  When provided and ``counts`` is
            ``None``, counts are extracted from ``adata.X``.  Takes
            priority over metadata stored on ``self``.
        method : str or tuple of (str, str), optional
            Denoising method.  A single string applies uniformly; a tuple
            ``(general_method, zi_zero_method)`` allows the ZINB zero
            correction to use a different method from the rest.  Default:
            ``("mean", "sample")`` — posterior mean for non-zero
            positions, stochastic sample at ZINB zeros.
        n_datasets : int, optional
            Number of denoised datasets to generate.  Dataset 1 uses the
            posterior mean of MCMC parameters; datasets 2..N each use a
            different MCMC draw.  Default: 1.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        cell_batch_size : int or None, optional
            Process cells in batches of this size to limit memory.
        include_original_counts : bool, optional
            If ``True``, store the input counts in
            ``.layers["original_counts"]``.  Default: ``True``.
        path : str or None, optional
            If provided, write the AnnData to this h5ad path.  For
            multiple datasets, files are named
            ``{stem}_{i}{suffix}`` (0-indexed).
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        AnnData or list of AnnData
            A single AnnData when ``n_datasets=1``, or a list when
            ``n_datasets > 1``.

        Raises
        ------
        ImportError
            If ``anndata`` is not installed.
        ValueError
            If neither ``counts`` nor ``adata`` is provided.
        """
        if AnnData is None:
            raise ImportError(
                "anndata is required for get_denoised_anndata(). "
                "Install it with: pip install anndata"
            )

        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Resolve counts from explicit argument or adata template
        counts = _resolve_counts_mcmc(counts, adata)

        # Resolve metadata: prefer adata template, fall back to self
        obs, var, uns = _resolve_metadata_mcmc(self, adata)

        samples = self.get_posterior_samples()

        r_all = samples["r"]
        p_all = samples["p"]
        pc_all = samples.get("p_capture")
        gate_all = samples.get("gate")
        mw_all = samples.get("mixing_weights")

        n_mcmc = r_all.shape[0]
        results: List["AnnData"] = []

        # --- Dataset 1: posterior-mean denoising ---
        if verbose:
            print(
                f"Generating denoised dataset 1/{n_datasets} "
                f"(posterior mean of {n_mcmc} MCMC samples)..."
            )

        # Average parameters across MCMC draws for a "MAP-like" estimate
        r_mean = jnp.mean(r_all, axis=0)
        p_mean = jnp.mean(p_all, axis=0)
        pc_mean = (
            jnp.mean(pc_all, axis=0) if pc_all is not None else None
        )
        gate_mean = (
            jnp.mean(gate_all, axis=0) if gate_all is not None else None
        )
        mw_mean = (
            jnp.mean(mw_all, axis=0) if mw_all is not None else None
        )

        rng_key, map_key = random.split(rng_key)
        denoised_mean = _denoise_counts_util(
            counts=counts,
            r=r_mean,
            p=p_mean,
            p_capture=pc_mean,
            gate=gate_mean,
            method=method,
            rng_key=map_key,
            mixing_weights=mw_mean,
            cell_batch_size=cell_batch_size,
        )

        results.append(
            _build_denoised_adata_mcmc(
                denoised=denoised_mean,
                counts=counts,
                obs=obs,
                var=var,
                uns=uns,
                method=method,
                dataset_index=0,
                parameter_source="posterior_mean",
                include_original_counts=include_original_counts,
            )
        )

        # --- Datasets 2..N: individual MCMC draw denoising ---
        is_mix = mw_all is not None
        for i in range(n_datasets - 1):
            idx = i % n_mcmc
            if verbose:
                print(
                    f"Generating denoised dataset {i + 2}/{n_datasets} "
                    f"(MCMC sample {idx})..."
                )

            r_s = r_all[idx]
            p_s = (
                p_all[idx]
                if p_all.ndim >= 1 and p_all.shape[0] == n_mcmc
                else p_all
            )
            pc_s = (
                pc_all[idx]
                if pc_all is not None and pc_all.ndim == 2
                else pc_all
            )
            g_s = (
                gate_all[idx]
                if gate_all is not None
                and gate_all.ndim > (1 if not is_mix else 2)
                else gate_all
            )
            mw_s = (
                mw_all[idx]
                if mw_all is not None and mw_all.ndim == 2
                else mw_all
            )

            rng_key, sample_key = random.split(rng_key)
            denoised_s = _denoise_counts_util(
                counts=counts,
                r=r_s,
                p=p_s,
                p_capture=pc_s,
                gate=g_s,
                method=method,
                rng_key=sample_key,
                mixing_weights=mw_s,
                cell_batch_size=cell_batch_size,
            )

            results.append(
                _build_denoised_adata_mcmc(
                    denoised=denoised_s,
                    counts=counts,
                    obs=obs,
                    var=var,
                    uns=uns,
                    method=method,
                    dataset_index=i + 1,
                    parameter_source=f"mcmc_sample_{idx}",
                    include_original_counts=include_original_counts,
                )
            )

        # Write to disk if requested
        if path is not None:
            _write_denoised_h5ad_mcmc(results, path, verbose)

        if n_datasets == 1:
            return results[0]
        return results

    # --------------------------------------------------------------------------
    # Prior predictive sampling
    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate samples from the prior predictive distribution.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key. Defaults to ``PRNGKey(42)``.
        n_samples : int, default=100
            Number of prior predictive samples.
        batch_size : int or None, optional
            Batch size for sample generation.
        store_samples : bool, default=True
            Store in ``self.prior_predictive_samples``.

        Returns
        -------
        jnp.ndarray
            Prior predictive samples.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        prior_predictive_samples = _generate_prior_predictive_samples(
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        if store_samples:
            self.prior_predictive_samples = prior_predictive_samples

        return prior_predictive_samples


# ==============================================================================
# Module-level helpers
# ==============================================================================


def _generate_ppc_samples(
    samples: Dict,
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: Optional[random.PRNGKey] = None,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples."""
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    from ._model_helpers import _get_model_fn

    model = _get_model_fn(model_config)
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    return generate_predictive_samples(
        model,
        samples,
        model_args,
        rng_key=rng_key,
        batch_size=batch_size,
    )


def _generate_prior_predictive_samples(
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: Optional[random.PRNGKey] = None,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate prior predictive samples using the model."""
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    from ._model_helpers import _get_model_fn

    model = _get_model_fn(model_config)
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    return generate_prior_predictive_samples(
        model,
        model_args,
        rng_key=rng_key,
        n_samples=n_samples,
        batch_size=batch_size,
    )


# ==============================================================================
# Helpers for get_denoised_anndata (module-level to avoid mixin complexity)
# ==============================================================================


def _resolve_counts_mcmc(
    counts: Optional[jnp.ndarray],
    adata: Optional["AnnData"],
) -> jnp.ndarray:
    """Determine count matrix from user arguments (MCMC variant)."""
    if counts is not None:
        return jnp.asarray(counts)
    if adata is not None:
        import scipy.sparse

        x = adata.X
        if scipy.sparse.issparse(x):
            x = x.toarray()
        return jnp.asarray(x)
    raise ValueError(
        "Either 'counts' or 'adata' must be provided. Pass the "
        "observed count matrix directly or an AnnData object."
    )


def _resolve_metadata_mcmc(results_obj, adata: Optional["AnnData"]) -> tuple:
    """Resolve obs/var/uns metadata for MCMC results."""
    if adata is not None:
        return adata.obs.copy(), adata.var.copy(), dict(adata.uns)
    obs = (
        results_obj.obs.copy() if getattr(results_obj, "obs", None) is not None
        else None
    )
    var = (
        results_obj.var.copy() if getattr(results_obj, "var", None) is not None
        else None
    )
    uns = (
        dict(results_obj.uns)
        if getattr(results_obj, "uns", None) is not None
        else {}
    )
    return obs, var, uns


def _build_denoised_adata_mcmc(
    denoised: jnp.ndarray,
    counts: jnp.ndarray,
    obs,
    var,
    uns,
    method: Union[str, Tuple[str, str]],
    dataset_index: int,
    parameter_source: str,
    include_original_counts: bool,
) -> "AnnData":
    """Construct an AnnData from a denoised count matrix (MCMC variant)."""
    denoised_np = np.asarray(denoised)
    kwargs: Dict = {}

    if obs is not None:
        kwargs["obs"] = obs.copy()
    if var is not None:
        kwargs["var"] = var.copy()

    adata_out = AnnData(X=denoised_np, **kwargs)

    if include_original_counts:
        adata_out.layers["original_counts"] = np.asarray(counts)

    out_uns = dict(uns) if uns else {}
    out_uns["scribe_denoising"] = {
        "method": list(method) if isinstance(method, tuple) else method,
        "dataset_index": dataset_index,
        "parameter_source": parameter_source,
    }
    adata_out.uns = out_uns

    return adata_out


def _write_denoised_h5ad_mcmc(
    results: List["AnnData"],
    path: str,
    verbose: bool,
) -> None:
    """Write denoised AnnData object(s) to h5ad files."""
    p = Path(path)
    if len(results) == 1:
        if verbose:
            print(f"Writing denoised h5ad to {p}...")
        results[0].write_h5ad(p)
    else:
        for i, adata_i in enumerate(results):
            out_path = p.parent / f"{p.stem}_{i}{p.suffix}"
            if verbose:
                print(f"Writing denoised h5ad to {out_path}...")
            adata_i.write_h5ad(out_path)

"""
AnnData export mixin for denoised SVI sampling results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import random

from ..sampling import denoise_counts
from ._sampling_denoising import _denoise_per_dataset

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


class DenoisedAnnDataMixin:
    """Mixin providing AnnData export methods for denoised counts."""

    def get_denoised_anndata(
        self,
        counts: Optional[jnp.ndarray] = None,
        adata: Optional["AnnData"] = None,
        method: Union[str, Tuple[str, str]] = ("mean", "sample"),
        n_datasets: int = 1,
        rng_key: Optional[random.PRNGKey] = None,
        cell_batch_size: Optional[int] = None,
        use_mean: bool = True,
        n_posterior_samples: Optional[int] = None,
        include_original_counts: bool = True,
        path: Optional[str] = None,
        verbose: bool = True,
    ) -> Union["AnnData", List["AnnData"]]:
        """Export denoised counts as an AnnData object (optionally to h5ad).

        Runs Bayesian denoising on the observed counts and packages the
        result into an :class:`~anndata.AnnData` object with the original
        cell/gene metadata.  Supports generating multiple denoised
        realisations: the first dataset uses MAP parameter estimates and
        subsequent datasets each use a different draw from the variational
        posterior.

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
            ``("mean", "sample")`` - posterior mean for non-zero
            positions, stochastic sample at ZINB zeros.
        n_datasets : int, optional
            Number of denoised datasets to generate.  Dataset 1 uses MAP
            estimates; datasets 2..N each use a different posterior sample.
            Default: 1.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)`` when
            ``None``.
        cell_batch_size : int or None, optional
            Process cells in batches of this size to limit memory.
        use_mean : bool, optional
            If ``True``, replace undefined MAP values (NaN) with posterior
            means.  Default: ``True``.
        n_posterior_samples : int or None, optional
            Number of posterior samples to draw from the guide when
            generating datasets 2..N.  Defaults to ``n_datasets - 1``.
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
        counts = self._resolve_counts(counts, adata)

        # Resolve metadata: prefer adata template, fall back to self
        obs, var, uns = self._resolve_metadata(adata)

        results: List["AnnData"] = []

        # Multi-dataset bookkeeping for expanding per-dataset params
        n_ds = getattr(self.model_config, "n_datasets", None)
        ds_idx = getattr(self, "_dataset_indices", None)

        # --- Dataset 1: MAP-based denoising ---
        if verbose:
            print(f"Generating denoised dataset 1/{n_datasets} (MAP)...")
        rng_key, map_key = random.split(rng_key)
        denoised_map = self.denoise_counts_map(
            counts=counts,
            method=method,
            rng_key=map_key,
            cell_batch_size=cell_batch_size,
            use_mean=use_mean,
            store_result=False,
            verbose=verbose,
        )

        results.append(
            self._build_denoised_adata(
                denoised=denoised_map,
                counts=counts,
                obs=obs,
                var=var,
                uns=uns,
                method=method,
                dataset_index=0,
                parameter_source="map",
                include_original_counts=include_original_counts,
            )
        )

        # --- Datasets 2..N: posterior-sample-based denoising ---
        if n_datasets > 1:
            n_post = (
                n_posterior_samples
                if n_posterior_samples is not None
                else n_datasets - 1
            )

            # Draw posterior samples if not already available
            if self.posterior_samples is None:
                rng_key, post_key = random.split(rng_key)
                if verbose:
                    print(
                        f"Drawing {n_post} posterior samples for "
                        f"datasets 2..{n_datasets}..."
                    )
                self.get_posterior_samples(
                    rng_key=post_key,
                    n_samples=n_post,
                    store_samples=True,
                    counts=counts,
                )

            r_post = self.posterior_samples["r"]
            p_post = self.posterior_samples["p"]
            pc_post = self.posterior_samples.get("p_capture")
            gate_post = self.posterior_samples.get("gate")
            is_mix = (
                self.n_components is not None and self.n_components > 1
            )
            mw_post = (
                self.posterior_samples.get("mixing_weights")
                if is_mix
                else None
            )

            n_available = r_post.shape[0]
            for i in range(n_datasets - 1):
                idx = i % n_available
                if verbose:
                    print(
                        f"Generating denoised dataset "
                        f"{i + 2}/{n_datasets} "
                        f"(posterior sample {idx})..."
                    )

                r_s = r_post[idx]
                p_s = (
                    p_post[idx]
                    if p_post.ndim >= 1
                    and p_post.shape[0] == n_available
                    else p_post
                )
                pc_s = (
                    pc_post[idx]
                    if pc_post is not None and pc_post.ndim == 2
                    else pc_post
                )
                g_s = (
                    gate_post[idx]
                    if gate_post is not None
                    and gate_post.ndim > (1 if not is_mix else 2)
                    else gate_post
                )
                mw_s = (
                    mw_post[idx]
                    if mw_post is not None and mw_post.ndim == 2
                    else mw_post
                )

                rng_key, sample_key = random.split(rng_key)

                # Multi-dataset: after slicing the sample dimension,
                # per-dataset params still have shape (n_datasets, ...).
                # Denoise each dataset's cells separately.
                if n_ds is not None and ds_idx is not None:
                    denoised_s = _denoise_per_dataset(
                        counts=counts,
                        r=r_s,
                        p=p_s,
                        gate=g_s,
                        p_capture=pc_s,
                        dataset_indices=ds_idx,
                        n_datasets=n_ds,
                        method=method,
                        rng_key=sample_key,
                        return_variance=False,
                        mixing_weights=mw_s,
                        cell_batch_size=cell_batch_size,
                    )
                else:
                    denoised_s = denoise_counts(
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
                    self._build_denoised_adata(
                        denoised=denoised_s,
                        counts=counts,
                        obs=obs,
                        var=var,
                        uns=uns,
                        method=method,
                        dataset_index=i + 1,
                        parameter_source=f"posterior_sample_{idx}",
                        include_original_counts=include_original_counts,
                    )
                )

        # Write to disk if requested
        if path is not None:
            self._write_denoised_h5ad(results, path, verbose)

        if n_datasets == 1:
            return results[0]
        return results

    @staticmethod
    def _resolve_counts(
        counts: Optional[jnp.ndarray],
        adata: Optional["AnnData"],
    ) -> jnp.ndarray:
        """Determine the count matrix from user-provided arguments.

        Parameters
        ----------
        counts : jnp.ndarray or None
            Explicit count matrix.
        adata : AnnData or None
            AnnData to extract counts from if ``counts`` is None.

        Returns
        -------
        jnp.ndarray
            The resolved count matrix.

        Raises
        ------
        ValueError
            If neither ``counts`` nor ``adata`` is provided.
        """
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

    def _resolve_metadata(
        self,
        adata: Optional["AnnData"],
    ) -> tuple:
        """Resolve obs/var/uns metadata, preferring the AnnData template.

        Parameters
        ----------
        adata : AnnData or None
            Optional template whose metadata takes priority.

        Returns
        -------
        obs : pd.DataFrame or None
        var : pd.DataFrame or None
        uns : dict or None
        """
        if adata is not None:
            return adata.obs.copy(), adata.var.copy(), dict(adata.uns)
        obs = self.obs.copy() if self.obs is not None else None
        var = self.var.copy() if self.var is not None else None
        uns = dict(self.uns) if self.uns is not None else {}
        return obs, var, uns

    @staticmethod
    def _build_denoised_adata(
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
        """Construct an AnnData from a denoised count matrix.

        Parameters
        ----------
        denoised : jnp.ndarray
            Denoised counts ``(n_cells, n_genes)``.
        counts : jnp.ndarray
            Original observed counts.
        obs : pd.DataFrame or None
            Cell metadata.
        var : pd.DataFrame or None
            Gene metadata.
        uns : dict or None
            Unstructured metadata (copied, then augmented).
        method : str or tuple
            The denoising method used.
        dataset_index : int
            Index of this dataset (0 = MAP, 1+ = posterior).
        parameter_source : str
            How parameters were obtained (e.g. ``"map"``,
            ``"posterior_sample_3"``).
        include_original_counts : bool
            Whether to include original counts as a layer.

        Returns
        -------
        AnnData
            The assembled AnnData object.
        """
        denoised_np = np.asarray(denoised)
        kwargs: Dict = {}

        if obs is not None:
            kwargs["obs"] = obs.copy()
        if var is not None:
            kwargs["var"] = var.copy()

        adata_out = AnnData(X=denoised_np, **kwargs)

        if include_original_counts:
            adata_out.layers["original_counts"] = np.asarray(counts)

        # Store denoising provenance metadata
        out_uns = dict(uns) if uns else {}
        out_uns["scribe_denoising"] = {
            "method": list(method) if isinstance(method, tuple) else method,
            "dataset_index": dataset_index,
            "parameter_source": parameter_source,
        }
        adata_out.uns = out_uns

        return adata_out

    @staticmethod
    def _write_denoised_h5ad(
        results: List["AnnData"],
        path: str,
        verbose: bool,
    ) -> None:
        """Write denoised AnnData object(s) to h5ad files.

        Parameters
        ----------
        results : list of AnnData
            The denoised datasets to write.
        path : str
            Target file path.  For multiple datasets, files are named
            ``{stem}_{i}{suffix}``.
        verbose : bool
            Print progress messages.
        """
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

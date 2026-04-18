"""
Likelihood mixin for SVI results.

This mixin provides methods for computing log-likelihoods using posterior
samples or MAP estimates.
"""

from typing import Optional
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np


class LikelihoodMixin:
    """Mixin providing log-likelihood computation methods."""

    # --------------------------------------------------------------------------
    # Compute log likelihood methods
    # --------------------------------------------------------------------------

    def log_likelihood(
        self,
        counts: jnp.ndarray,
        sample_chunk_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        r_floor: float = 1e-6,
        p_floor: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data under posterior samples.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate likelihood on
        sample_chunk_size : Optional[int], default=None
            Number of posterior samples evaluated per chunk. When set, the
            method computes log-likelihoods in sequential chunks to bound peak
            memory usage; this is slower but avoids OOM for large
            ``(n_samples, n_cells, n_genes)`` workloads. ``None`` keeps the
            previous all-samples-at-once ``vmap`` behavior.
        return_by : str, default='cell'
            Specifies how to return the log probabilities. Must be one of:
                - 'cell': returns log probabilities summed over genes
                - 'gene': returns log probabilities summed over cells
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        ignore_nans : bool, default=False
            If True, removes any samples that contain NaNs.
        split_components : bool, default=False
            If True, returns log likelihoods for each mixture component
            separately. Only applicable for mixture models.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods (for mixture models).
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        r_floor : float, default=1e-6
            Minimum value clamped onto the NB dispersion parameter ``r``
            before evaluating the log-likelihood.  Posterior samples from a
            wide variational guide can occasionally produce ``r`` values that
            underflow to zero (or become negative in float32), causing
            ``lgamma(r)`` to return NaN.  Setting a small positive floor
            (e.g. ``1e-6``) neutralises those degenerate samples without
            meaningfully changing the likelihood for well-behaved ones.
            Set to ``0.0`` to disable the floor entirely.
        p_floor : float, default=1e-6
            Epsilon applied to the success probability ``p`` (or effective
            probability ``p_hat`` for VCP models), clipping it to the open
            interval ``(p_floor, 1 - p_floor)``.

            Two float32 degenerate cases this guards against:

            1. ``phi_g → 0`` in hierarchical parameterisations
               → ``p_g = 1/(1+0) = 1.0`` exactly in float32
               → ``r * log(1 - p) = r * log(0)`` → NaN/−∞.
            2. ``phi_capture → ∞`` in VCP models
               → ``p_capture = 0`` → ``p_hat = 0``
               → ``NB(r, 0).log_prob(0) = 0 * log(0)`` → NaN.

            Set to ``0.0`` to disable.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on model type, return_by and
            split_components parameters. For standard models:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            For mixture models with split_components=False:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            For mixture models with split_components=True:
                - 'cell': shape (n_samples, n_cells, n_components)
                - 'gene': shape (n_samples, n_genes, n_components)

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Get parameter samples
        parameter_samples = self.posterior_samples

        # Get number of samples from first parameter
        n_samples = parameter_samples[next(iter(parameter_samples))].shape[0]

        # Get likelihood function
        likelihood_fn = self._log_likelihood_fn()

        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1

        # Build per-draw ``AxisLayout`` metadata for ``Likelihood.log_prob``.
        # ``self.layouts`` is keyed by *variational*-parameter names
        # (``alpha_p``, ``beta_p``, ...) whereas ``posterior_samples`` uses
        # canonical names (``p``, ``r``, ``gate``, ...).  We therefore
        # rebuild canonical layouts directly from the posterior samples,
        # then strip the sample axis for per-draw evaluation.
        from ..sampling import _build_canonical_layouts

        _post_layouts = _build_canonical_layouts(
            parameter_samples,
            self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=self.n_components,
            has_sample_dim=True,
        )
        draw_layouts = {
            k: v.without_sample_dim() for k, v in _post_layouts.items()
        }

        # Define function to compute likelihood for a single sample
        @jit
        def compute_sample_lik(i):
            # Extract parameters for this sample
            params_i = {k: v[i] for k, v in parameter_samples.items()}
            # For mixture models we need to pass split_components and weights
            if is_mixture:
                return likelihood_fn(
                    counts,
                    params_i,
                    draw_layouts,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    r_floor=r_floor,
                    p_floor=p_floor,
                    dtype=dtype,
                )
            else:
                return likelihood_fn(
                    counts,
                    params_i,
                    draw_layouts,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    r_floor=r_floor,
                    p_floor=p_floor,
                    dtype=dtype,
                )

        # Use chunked vmap to reduce peak memory when requested.
        if (
            sample_chunk_size is None
            or sample_chunk_size <= 0
            or sample_chunk_size >= n_samples
        ):
            log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
        else:
            chunks = []
            for start in range(0, n_samples, sample_chunk_size):
                end = min(start + sample_chunk_size, n_samples)
                chunks.append(vmap(compute_sample_lik)(jnp.arange(start, end)))
            log_liks = jnp.concatenate(chunks, axis=0)

        # Handle NaNs if requested
        if ignore_nans:
            # Check for NaNs appropriately based on dimensions
            if is_mixture and split_components:
                # Handle case with component dimension
                valid_samples = ~jnp.any(
                    jnp.any(jnp.isnan(log_liks), axis=-1), axis=-1
                )
            else:
                # Standard case
                valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=-1)

            # Filter out samples with NaNs
            if jnp.any(~valid_samples):
                print(
                    f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}"
                )
                return log_liks[valid_samples]

        return log_liks

    # --------------------------------------------------------------------------

    def log_likelihood_map(
        self,
        counts: jnp.ndarray,
        gene_batch_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        use_mean: bool = True,
        verbose: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data using MAP parameter estimates.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate likelihood on
        gene_batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation by gene
        return_by : str, default='cell'
            Specifies how to return the log probabilities. Must be one of:
                - 'cell': returns log probabilities summed over genes
                - 'gene': returns log probabilities summed over cells
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        split_components : bool, default=False
            If True, returns log likelihoods for each mixture component separately.
            Only applicable for mixture models.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods (for mixture models).
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        use_mean : bool, default=False
            If True, replaces undefined MAP values (NaN) with posterior means
        verbose : bool, default=True
            If True, prints a warning if NaNs were replaced with means
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on model type, return_by and
            split_components parameters.
        """
        # Multi-dataset models: MAP estimates carry a dataset dimension
        # (e.g. r has shape (K, D, G)) that the per-cell likelihood
        # functions cannot broadcast against.  Dispatch per-dataset,
        # computing likelihoods on each dataset's cells separately via
        # get_dataset(d), then reassemble in original cell order.
        _n_ds = getattr(getattr(self, "model_config", None), "n_datasets", None)
        _ds_idx = getattr(self, "_dataset_indices", None)
        if _n_ds is not None and _ds_idx is not None:
            return self._log_likelihood_map_per_dataset(
                counts=counts,
                gene_batch_size=gene_batch_size,
                return_by=return_by,
                cells_axis=cells_axis,
                split_components=split_components,
                weights=weights,
                weight_type=weight_type,
                use_mean=use_mean,
                verbose=verbose,
                dtype=dtype,
            )

        # Get the log likelihood function
        likelihood_fn = self._log_likelihood_fn()

        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1

        # Get the MAP estimates with canonical parameters included
        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=verbose
        )

        # Build canonical ``AxisLayout`` metadata from the MAP-estimate
        # shapes.  ``self.layouts`` is keyed by variational parameter
        # names, but ``log_prob`` consumes canonical names (``p``, ``r``,
        # ``gate``, ...); we reuse the sampling helper for a consistent
        # lookup.
        from ..sampling import _build_canonical_layouts

        map_layouts = _build_canonical_layouts(
            map_estimates,
            self.model_config,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=self.n_components,
            has_sample_dim=False,
        )

        # If computing by gene and gene_batch_size is provided, use batched computation
        if return_by == "gene" and gene_batch_size is not None:
            # Accumulate results as JAX arrays on-device, then do a
            # single D2H transfer at the end.  The previous approach
            # called np.array() per batch, forcing a blocking
            # device-to-host sync on every iteration.
            jax_chunks = []

            # Process genes in batches
            for i in range(0, self.n_genes, gene_batch_size):
                if verbose and i > 0:
                    print(
                        f"Processing genes {i}-{min(i+gene_batch_size, self.n_genes)} of {self.n_genes}"
                    )

                # Get gene indices for this batch
                end_idx = min(i + gene_batch_size, self.n_genes)
                gene_indices = list(range(i, end_idx))

                # Get subset of results for these genes
                results_subset = self[gene_indices]
                # Get the MAP estimates for this subset (with canonical parameters)
                subset_map_estimates = results_subset.get_map(
                    use_mean=use_mean, canonical=True, verbose=False
                )
                # Rebuild canonical layouts for the gene-subsetted MAP
                # estimates so ``log_prob`` receives layouts keyed by
                # canonical names.
                subset_layouts = _build_canonical_layouts(
                    subset_map_estimates,
                    results_subset.model_config,
                    n_genes=results_subset.n_genes,
                    n_cells=results_subset.n_cells,
                    n_components=results_subset.n_components,
                    has_sample_dim=False,
                )

                # Get subset of counts for these genes
                if cells_axis == 0:
                    counts_subset = counts[:, gene_indices]
                else:
                    counts_subset = counts[gene_indices, :]

                # Get subset of weights if provided
                weights_subset = None
                if weights is not None:
                    if weights.ndim == 1:  # Shape: (n_genes,)
                        weights_subset = weights[gene_indices]
                    else:
                        weights_subset = weights

                # Compute log likelihood for this gene batch
                if is_mixture:
                    batch_log_liks = likelihood_fn(
                        counts_subset,
                        subset_map_estimates,
                        subset_layouts,
                        cells_axis=cells_axis,
                        return_by=return_by,
                        split_components=split_components,
                        weights=weights_subset,
                        weight_type=weight_type,
                        dtype=dtype,
                    )
                else:
                    batch_log_liks = likelihood_fn(
                        counts_subset,
                        subset_map_estimates,
                        subset_layouts,
                        cells_axis=cells_axis,
                        return_by=return_by,
                        dtype=dtype,
                    )

                jax_chunks.append(batch_log_liks)

            # Single concatenation + single D2H transfer
            return jnp.concatenate(jax_chunks, axis=0)

        # Standard computation (no gene batching)
        else:
            # Compute log-likelihood for mixture model
            if is_mixture:
                log_liks = likelihood_fn(
                    counts,
                    map_estimates,
                    map_layouts,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    dtype=dtype,
                )
            # Compute log-likelihood for non-mixture model
            else:
                log_liks = likelihood_fn(
                    counts,
                    map_estimates,
                    map_layouts,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    dtype=dtype,
                )

            return log_liks

    # ------------------------------------------------------------------
    # Multi-dataset dispatch for log_likelihood_map
    # ------------------------------------------------------------------

    def _log_likelihood_map_per_dataset(
        self,
        counts: jnp.ndarray,
        gene_batch_size=None,
        return_by: str = "cell",
        cells_axis: int = 0,
        split_components: bool = False,
        weights=None,
        weight_type=None,
        use_mean: bool = True,
        verbose: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Compute MAP log-likelihoods by iterating over datasets.

        For multi-dataset models the MAP estimates carry a dataset
        dimension that the per-cell likelihood functions cannot
        broadcast against.  This helper iterates over datasets using
        ``get_dataset(d)`` (which strips the dataset dimension) and
        reassembles the per-dataset results.

        Parameters
        ----------
        counts : jnp.ndarray
            Full count matrix, shape ``(n_cells, n_genes)``.
        gene_batch_size, return_by, cells_axis, split_components,
        weights, weight_type, use_mean, verbose, dtype
            Forwarded verbatim to the single-dataset
            ``log_likelihood_map`` call.

        Returns
        -------
        jnp.ndarray
            Same shape contract as ``log_likelihood_map``.
        """
        dataset_indices = np.asarray(self._dataset_indices).reshape(-1)
        unique_ds = np.unique(dataset_indices)

        if return_by == "cell":
            # Collect per-dataset results and stitch back in cell order.
            # Probe shape from the first dataset to pre-allocate.
            first_mask = dataset_indices == unique_ds[0]
            first_counts = (
                counts[first_mask] if cells_axis == 0
                else counts[:, first_mask]
            )
            ds0 = self.get_dataset(int(unique_ds[0]))
            first_liks = ds0.log_likelihood_map(
                first_counts,
                gene_batch_size=gene_batch_size,
                return_by=return_by,
                cells_axis=cells_axis,
                split_components=split_components,
                weights=weights,
                weight_type=weight_type,
                use_mean=use_mean,
                verbose=verbose,
                dtype=dtype,
            )

            n_cells = counts.shape[0] if cells_axis == 0 else counts.shape[1]
            if first_liks.ndim == 1:
                result = np.zeros(n_cells, dtype=dtype)
            else:
                result = np.zeros(
                    (n_cells, first_liks.shape[-1]), dtype=dtype
                )
            result[first_mask] = np.asarray(first_liks)

            for d in unique_ds[1:]:
                mask = dataset_indices == d
                ds_counts = (
                    counts[mask] if cells_axis == 0
                    else counts[:, mask]
                )
                ds_results = self.get_dataset(int(d))
                ds_liks = ds_results.log_likelihood_map(
                    ds_counts,
                    gene_batch_size=gene_batch_size,
                    return_by=return_by,
                    cells_axis=cells_axis,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    use_mean=use_mean,
                    verbose=verbose and d == unique_ds[1],
                    dtype=dtype,
                )
                result[mask] = np.asarray(ds_liks)

            return jnp.array(result)

        else:
            # return_by="gene": each dataset contributes a partial
            # gene-level sum; add them together.
            accumulated = None
            for d in unique_ds:
                mask = dataset_indices == d
                ds_counts = (
                    counts[mask] if cells_axis == 0
                    else counts[:, mask]
                )
                ds_results = self.get_dataset(int(d))
                ds_liks = ds_results.log_likelihood_map(
                    ds_counts,
                    gene_batch_size=gene_batch_size,
                    return_by=return_by,
                    cells_axis=cells_axis,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    use_mean=use_mean,
                    verbose=verbose and d == unique_ds[0],
                    dtype=dtype,
                )
                if accumulated is None:
                    accumulated = ds_liks
                else:
                    accumulated = accumulated + ds_liks

            return accumulated

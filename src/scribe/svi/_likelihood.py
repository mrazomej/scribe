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
        batch_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data under posterior samples.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate likelihood on
        batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation
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

        # Convert posterior samples to canonical form
        self._convert_to_canonical()

        # Get parameter samples
        parameter_samples = self.posterior_samples

        # Get number of samples from first parameter
        n_samples = parameter_samples[next(iter(parameter_samples))].shape[0]

        # Get likelihood function
        likelihood_fn = self._log_likelihood_fn()

        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1

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
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    dtype=dtype,
                )
            else:
                return likelihood_fn(
                    counts,
                    params_i,
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    dtype=dtype,
                )

        # Use vmap for parallel computation (more memory intensive)
        log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))

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
        batch_size: Optional[int] = None,
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
        batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation
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
        # Get the log likelihood function
        likelihood_fn = self._log_likelihood_fn()

        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1

        # Get the MAP estimates with canonical parameters included
        map_estimates = self.get_map(
            use_mean=use_mean, canonical=True, verbose=verbose
        )

        # If computing by gene and gene_batch_size is provided, use batched computation
        if return_by == "gene" and gene_batch_size is not None:
            # Determine output shape
            if (
                is_mixture
                and split_components
                and self.n_components is not None
            ):
                result_shape = (self.n_genes, self.n_components)
            else:
                result_shape = (self.n_genes,)

            # Initialize result array
            log_liks = np.zeros(result_shape, dtype=dtype)

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
                        batch_size=batch_size,
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
                        batch_size=batch_size,
                        cells_axis=cells_axis,
                        return_by=return_by,
                        dtype=dtype,
                    )

                # Store results
                log_liks[i:end_idx] = np.array(batch_log_liks)

            # Convert to JAX array for consistency
            return jnp.array(log_liks)

        # Standard computation (no gene batching)
        else:
            # Compute log-likelihood for mixture model
            if is_mixture:
                log_liks = likelihood_fn(
                    counts,
                    map_estimates,
                    batch_size=batch_size,
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
                    batch_size=batch_size,
                    cells_axis=cells_axis,
                    return_by=return_by,
                    dtype=dtype,
                )

            return log_liks

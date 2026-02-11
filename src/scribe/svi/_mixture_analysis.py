"""
Mixture analysis mixin for SVI results.

This mixin provides methods for analyzing mixture models, including entropy
calculations and cell type probability assignments.
"""

from typing import Dict, Optional
import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import softmax

# ==============================================================================
# Mixture Analysis Mixin
# ==============================================================================


class MixtureAnalysisMixin:
    """Mixin providing mixture model analysis methods."""

    # --------------------------------------------------------------------------
    # Compute entropy of component assignments
    # --------------------------------------------------------------------------

    def mixture_component_entropy(
        self,
        counts: jnp.ndarray,
        return_by: str = "gene",
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        temperature: Optional[float] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute the entropy of mixture component assignment probabilities.

        This method calculates the Shannon entropy of the posterior component
        assignment probabilities for each observation (cell or gene), providing
        a quantitative measure of assignment uncertainty in mixture models.

        The entropy quantifies how uncertain the model is about which component
        each observation belongs to:
            - Low entropy (≈0): High confidence in component assignment
            - High entropy (≈log(n_components)): High uncertainty in assignment
            - Maximum entropy: Uniform assignment probabilities across all
              components

        The entropy is calculated as:
            H = -∑(p_i * log(p_i))
        where p_i are the posterior probabilities of assignment to component i.

        Parameters
        ----------
        counts : jnp.ndarray
            Input count data to evaluate component assignments for. Shape should
            be (n_cells, n_genes) if cells_axis=0, or (n_genes, n_cells) if
            cells_axis=1.

        return_by : str, default='gene'
            Specifies how to compute and return the entropy. Must be one of:
                - 'cell': Compute entropy of component assignments for each cell
                - 'gene': Compute entropy of component assignments for each gene

        batch_size : Optional[int], default=None
            If provided, processes the data in batches of this size to reduce
            memory usage. Useful for large datasets.

        cells_axis : int, default=0
            Specifies which axis in the input counts contains the cells:
                - 0: cells are rows (shape: n_cells × n_genes)
                - 1: cells are columns (shape: n_genes × n_cells)

        ignore_nans : bool, default=False
            If True, excludes any samples containing NaN values from the entropy
            calculation.

        temperature : Optional[float], default=None
            If provided, applies temperature scaling to the log-likelihoods
            before computing entropy. Temperature scaling modifies the sharpness
            of probability distributions by dividing log probabilities by a
            temperature parameter T.

        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations.

        Returns
        -------
        jnp.ndarray
            Array of entropy values. Shape depends on return_by:
                - If return_by='cell': shape is (n_samples, n_cells)
                - If return_by='gene': shape is (n_samples, n_genes)
            Higher values indicate more uncertainty in component assignments.

        Raises
        ------
        ValueError
            If the model is not a mixture model or if posterior samples haven't
            been generated.

        Notes
        -----
        - This method requires posterior samples to be available. Call
          get_posterior_samples() first if they haven't been generated.
        - The entropy is computed using the full posterior predictive
          distribution, accounting for uncertainty in the model parameters.
        - For a mixture with K components, the maximum possible entropy is
          log(K).
        - Entropy values can be used to identify observations that are difficult
          to classify or that may belong to multiple components.
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Mixture component entropy calculation only applies to mixture "
                "models with multiple components"
            )

        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        print("Computing log-likelihoods...")

        # Compute log-likelihoods for each component
        log_liks = self.log_likelihood(
            counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            return_by=return_by,
            ignore_nans=ignore_nans,
            dtype=dtype,
            split_components=True,  # Ensure we get per-component likelihoods
        )

        # Apply temperature scaling if requested
        if temperature is not None:
            from ..core.cell_type_assignment import temperature_scaling

            log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

        print("Converting log-likelihoods to probabilities...")

        # Convert log-likelihoods to probabilities
        probs = softmax(log_liks, axis=-1)

        print("Computing entropy...")

        # Compute entropy: -∑(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        eps = jnp.finfo(dtype).eps
        entropy = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)

        return entropy

    # --------------------------------------------------------------------------

    def assignment_entropy_map(
        self,
        counts: jnp.ndarray,
        return_by: str = "gene",
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        temperature: Optional[float] = None,
        use_mean: bool = True,
        verbose: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute the entropy of component assignments for each cell evaluated at
        the MAP.

        This method calculates the entropy of the posterior component assignment
        probabilities for each cell or gene, providing a measure of assignment
        uncertainty. Higher entropy values indicate more uncertainty in the
        component assignments, while lower values indicate more confident
        assignments.

        The entropy is calculated as:
            H = -∑(p_i * log(p_i))
        where p_i are the normalized probabilities for each component.

        Parameters
        ----------
        counts : jnp.ndarray
            The count matrix with shape (n_cells, n_genes).
        return_by : str, default='gene'
            Whether to return the entropy by cell or gene.
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        temperature : Optional[float], default=None
            If provided, applies temperature scaling to the log-likelihoods
            before computing entropy.
        use_mean : bool, default=True
            If True, uses the mean of the posterior component probabilities
            instead of the MAP.
        verbose : bool, default=True
            If True, prints a warning if NaNs were replaced with means
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            The component entropy for each cell evaluated at the MAP. Shape:
            (n_cells,).

        Raises
        ------
        ValueError
            - If the model is not a mixture model
            - If posterior samples have not been generated yet
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Component entropy calculation only applies to mixture models "
                "with multiple components"
            )

        # Compute log-likelihood at the MAP
        log_liks = self.log_likelihood_map(
            counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            use_mean=use_mean,
            verbose=verbose,
            dtype=dtype,
            return_by=return_by,
            split_components=True,
        )

        # Apply temperature scaling if requested
        if temperature is not None:
            from ..core.cell_type_assignment import temperature_scaling

            log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

        # Compute log-sum-exp for normalization
        log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)

        # Compute probabilities (avoiding log space for final entropy calculation)
        probs = jnp.exp(log_liks - log_sum_exp)

        # Compute entropy: -∑(p_i * log(p_i))
        # Add small epsilon to avoid log(0)
        eps = jnp.finfo(dtype).eps
        entropy = -jnp.sum(probs * jnp.log(probs + eps), axis=-1)

        return entropy

    # --------------------------------------------------------------------------
    # Cell type assignment method for mixture models
    # --------------------------------------------------------------------------

    def cell_type_probabilities(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32,
        fit_distribution: bool = True,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute probabilistic cell type assignments and fit Dirichlet
        distributions to characterize assignment uncertainty.

        For each cell, this method:
            1. Computes component-specific log-likelihoods using posterior
               samples
            2. Converts these to probability distributions over cell types
            3. Fits a Dirichlet distribution to characterize the uncertainty in
               these assignments

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        ignore_nans : bool, default=False
            If True, removes any samples that contain NaNs.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the assignment
            probabilities
        temperature : Optional[float], default=None
            If provided, apply temperature scaling to log probabilities
        weights : Optional[jnp.ndarray], default=None
            Array used to weight genes when computing log likelihoods
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'concentration': Dirichlet concentration parameters for each
                  cell. Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'mean_probabilities': Mean assignment probabilities for each
                  cell. Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'sample_probabilities': Assignment probabilities for each
                  posterior sample. Shape: (n_samples, n_cells, n_components)

        Raises
        ------
        ValueError
            - If the model is not a mixture model
            - If posterior samples have not been generated yet

        Note
        ----
        Most of the log-likelihood value differences between cell types are
        extremely large. Thus, the computation usually returns either 0 or 1.
        This computation is therefore not very useful, but it is included for
        completeness.
        """
        from ..core.cell_type_assignment import compute_cell_type_probabilities

        return compute_cell_type_probabilities(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            dtype=dtype,
            fit_distribution=fit_distribution,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------

    def cell_type_probabilities_map(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        dtype: jnp.dtype = jnp.float32,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        use_mean: bool = False,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute probabilistic cell type assignments using MAP estimates of
        parameters.

        For each cell, this method:
            1. Computes component-specific log-likelihoods using MAP parameter
            estimates
            2. Converts these to probability distributions over cell types

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
        temperature : Optional[float], default=None
            If provided, apply temperature scaling to log probabilities
        weights : Optional[jnp.ndarray], default=None
            Array used to weight genes when computing log likelihoods
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        use_mean : bool, default=False
            If True, replaces undefined MAP values (NaN) with posterior means
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'probabilities': Assignment probabilities for each cell.
                Shape: (n_cells, n_components)

        Raises
        ------
        ValueError
            If the model is not a mixture model
        """
        from ..core.cell_type_assignment import (
            compute_cell_type_probabilities_map,
        )

        return compute_cell_type_probabilities_map(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            dtype=dtype,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            use_mean=use_mean,
            verbose=verbose,
        )

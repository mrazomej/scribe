"""
Normalization mixin for SVI results.

This mixin provides methods for normalizing counts using posterior samples,
including Dirichlet-based normalization and Logistic-Normal distribution fitting.
"""

from typing import Dict, Optional, Union
import jax.numpy as jnp
from jax import random

from ..core.normalization import normalize_counts_from_posterior

# ==============================================================================
# Normalization Mixin
# ==============================================================================


class NormalizationMixin:
    """Mixin providing count normalization methods."""

    # --------------------------------------------------------------------------
    # Count normalization methods
    # --------------------------------------------------------------------------

    def normalize_counts(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples_dirichlet: int = 1,
        fit_distribution: bool = False,
        store_samples: bool = True,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        batch_size: int = 256,
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Normalize counts using posterior samples of the r parameter.

        This method takes posterior samples of the dispersion parameter (r)
        and uses them as concentration parameters for Dirichlet distributions
        to generate normalized expression profiles.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key. Defaults to random.PRNGKey(42).
        n_samples_dirichlet : int, default=1
            Number of samples to draw from each Dirichlet distribution.
        fit_distribution : bool, default=False
            If True, fits a Dirichlet distribution to the generated samples.
        store_samples : bool, default=True
            If True, includes the raw Dirichlet samples in the output.
        sample_axis : int, default=0
            Axis containing samples in the Dirichlet fitting.
        return_concentrations : bool, default=False
            If True, returns the original r parameter samples.
        backend : str, default="numpyro"
            ``"numpyro"`` or ``"scipy"`` for distribution objects.
        batch_size : int, default=256
            Number of posterior samples per batched Dirichlet sampling call.
            Larger values use more GPU memory but fewer dispatches.
        verbose : bool, default=True
            If True, prints progress messages.

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Normalized expression results.

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet.

        Examples
        --------
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=1, batch_size=512,
        ... )
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Use the shared normalization function
        return normalize_counts_from_posterior(
            posterior_samples=self.posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            fit_distribution=fit_distribution,
            store_samples=store_samples,
            sample_axis=sample_axis,
            return_concentrations=return_concentrations,
            backend=backend,
            batch_size=batch_size,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------

    def fit_logistic_normal(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples_dirichlet: int = 1,
        rank: Optional[int] = None,
        distribution_type: str = "softmax",
        remove_mean: bool = False,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Fit a low-rank Logistic-Normal distribution to normalized expression.

        This method takes posterior samples of the dispersion parameter (r) and
        fits a low-rank Logistic-Normal distribution to capture the correlation
        structure in normalized expression profiles.  This preserves the
        low-rank correlation structure discovered during inference, which is
        lost when fitting a single Dirichlet distribution.

        Mathematically, this fits:
            log(rho) ~ MVN(loc, W W^T + D)
        where rho are the normalized expression proportions on the simplex.

        Parameters
        ----------
        rng_key : random.PRNGKey, default=random.PRNGKey(42)
            JAX random number generator key.
        n_samples_dirichlet : int, default=1
            Number of Dirichlet samples to draw per posterior sample for
            fitting.  More samples give better estimates but use more memory.
        rank : Optional[int], default=None
            Rank of the low-rank covariance approximation. If None,
            automatically uses min(n_genes, 50).
        distribution_type : str, default="softmax"
            Type of compositional distribution to fit.  Must be one of:

            - ``"softmax"``: SoftmaxNormal (symmetric, all genes treated
              equally, cannot evaluate log_prob but good for
              sampling/visualization)
            - ``"alr"``: LowRankLogisticNormal (ALR-based, uses last gene as
              reference, can evaluate log_prob for Bayesian inference)
        remove_mean : bool, default=False
            If True, removes the grand mean from ALR-transformed samples
            before fitting, focusing on co-variation patterns rather than
            mean composition.
        batch_size : int, default=256
            Number of posterior samples to process in each batched Dirichlet
            sampling call.  Larger values use more GPU memory but require
            fewer Python-to-JAX dispatches.  At D=20 000 and
            ``n_samples_dirichlet=1``, each batch of 256 requires ~20 MB.
            Reduce if you encounter OOM errors; increase on large-memory
            GPUs for maximum throughput.
        verbose : bool, default=True
            If True, prints progress messages and shows progress bars.

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Dictionary containing fitted Logistic-Normal parameters:

            - 'loc': Location parameter (mean in log-space)
            - 'cov_factor': Low-rank covariance factor W
            - 'cov_diag': Diagonal component of covariance D
            - 'mean_probabilities': Mean probabilities on the simplex
            - 'distributions': Distribution objects

            For non-mixture models:

            - loc: shape (n_genes,)
            - cov_factor: shape (n_genes, rank)
            - cov_diag: shape (n_genes,)
            - mean_probabilities: shape (n_genes,)
            - distributions: single TransformedDistribution object

            For mixture models:

            - loc: shape (n_components, n_genes)
            - cov_factor: shape (n_components, n_genes, rank)
            - cov_diag: shape (n_components, n_genes)
            - mean_probabilities: shape (n_components, n_genes)
            - distributions: list of n_components TransformedDistribution
              objects

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet, or if 'r'
            parameter is not found in posterior samples.

        See Also
        --------
        normalize_counts : Alternative method that fits Dirichlet
            distributions.

        Notes
        -----
        The Logistic-Normal distribution naturally captures correlation
        between genes, which is important for co-regulated gene modules.  The
        correlation structure is inherited from the low-rank structure in the
        posterior over r parameters, making it particularly suitable for
        models that use low-rank Gaussian guides.

        Examples
        --------
        >>> fitted = results.fit_logistic_normal(
        ...     rank=20, batch_size=512, verbose=True,
        ... )
        >>> print(fitted['loc'].shape)  # (n_genes,)
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Validate distribution_type parameter
        from ..stats import SoftmaxNormal, LowRankLogisticNormal

        valid_types = ["softmax", "alr"]
        if distribution_type not in valid_types:
            raise ValueError(
                f"Invalid distribution_type: {distribution_type}. "
                f"Must be one of: {valid_types}"
            )

        # Map string to distribution class
        dist_class_map = {
            "softmax": SoftmaxNormal,
            "alr": LowRankLogisticNormal,
        }

        distribution_class = dist_class_map[distribution_type]

        # Import the fitting function
        from ..core.normalization_logistic import (
            fit_logistic_normal_from_posterior,
        )

        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Use the shared fitting function with distribution class
        return fit_logistic_normal_from_posterior(
            posterior_samples=self.posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            rank=rank,
            distribution_class=distribution_class,
            remove_mean=remove_mean,
            batch_size=batch_size,
            verbose=verbose,
        )

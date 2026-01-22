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
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples_dirichlet: int = 1,
        fit_distribution: bool = False,
        store_samples: bool = True,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Normalize counts using posterior samples of the r parameter.

        This method takes posterior samples of the dispersion parameter (r) and
        uses them as concentration parameters for Dirichlet distributions to
        generate normalized expression profiles. For mixture models,
        normalization is performed per component, resulting in an extra
        dimension in the output.

        Based on the insights from the Dirichlet-multinomial model derivation,
        the r parameters represent the concentration parameters of a Dirichlet
        distribution that can be used to generate normalized expression
        profiles.

        The method generates Dirichlet samples using all posterior samples of r,
        then fits a single Dirichlet distribution to all these samples (or one
        per component for mixture models).

        Parameters
        ----------
        rng_key : random.PRNGKey, default=random.PRNGKey(42)
            JAX random number generator key
        n_samples_dirichlet : int, default=1000
            Number of samples to draw from each Dirichlet distribution
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the generated samples
            using fit_dirichlet_minka from stats.py
        store_samples : bool, default=False
            If True, includes the raw Dirichlet samples in the output
        sample_axis : int, default=0
            Axis containing samples in the Dirichlet fitting (passed to
            fit_dirichlet_minka)
        return_concentrations : bool, default=False
            If True, returns the original r parameter samples used as
            concentrations
        backend : str, default="numpyro"
            Statistical package to use for distributions when
            fit_distribution=True. Must be one of: - "numpyro": Returns
            numpyro.distributions.Dirichlet objects - "scipy": Returns
            scipy.stats distributions via numpyro_to_scipy conversion
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Dictionary containing normalized expression profiles. Keys depend on
            input arguments:
                - 'samples': Raw Dirichlet samples (if store_samples=True)
                - 'concentrations': Fitted concentration parameters (if
                  fit_distribution=True)
                - 'mean_probabilities': Mean probabilities from fitted
                  distribution (if fit_distribution=True)
                - 'distributions': Dirichlet distribution objects (if
                  fit_distribution=True)
                - 'original_concentrations': Original r parameter samples (if
                  return_concentrations=True)

            For non-mixture models:
                - samples: shape (n_posterior_samples, n_genes,
                  n_samples_dirichlet) or (n_posterior_samples, n_genes) if
                  n_samples_dirichlet=1
                - concentrations: shape (n_genes,) - single fitted distribution
                - mean_probabilities: shape (n_genes,) - single fitted
                  distribution
                - distributions: single Dirichlet distribution object

            For mixture models:
                - samples: shape (n_posterior_samples, n_components, n_genes,
                  n_samples_dirichlet) or (n_posterior_samples, n_components,
                  n_genes) if n_samples_dirichlet=1
                - concentrations: shape (n_components, n_genes) - one fitted
                distribution per component - mean_probabilities: shape
                (n_components, n_genes) - one fitted distribution per component
                - distributions: list of n_components Dirichlet distribution
                objects

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet, or if 'r'
            parameter is not found in posterior samples

        Examples
        --------
        >>> # For a non-mixture model
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=100,
        ...     fit_distribution=True
        ... )
        >>> print(normalized['mean_probabilities'].shape)  # (n_genes,)
        >>> print(type(normalized['distributions']))  # Single Dirichlet distribution

        >>> # For a mixture model
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=100,
        ...     fit_distribution=True
        ... )
        >>> print(normalized['mean_probabilities'].shape)  # (n_components, n_genes)
        >>> print(len(normalized['distributions']))  # n_components
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Convert to canonical form to ensure r parameter is available
        self._convert_to_canonical()

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
            verbose=verbose,
        )

    # --------------------------------------------------------------------------

    def fit_logistic_normal(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples_dirichlet: int = 1,
        rank: Optional[int] = None,
        distribution_type: str = "softmax",
        remove_mean: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Fit a low-rank Logistic-Normal distribution to normalized expression.

        This method takes posterior samples of the dispersion parameter (r) and
        fits a low-rank Logistic-Normal distribution to capture the correlation
        structure in normalized expression profiles. This preserves the low-rank
        correlation structure discovered during inference, which is lost when
        fitting a single Dirichlet distribution.

        The Logistic-Normal distribution is particularly useful when the
        posterior over r parameters exhibits correlation structure (e.g., from
        low-rank Gaussian guides). Unlike fitting a single Dirichlet
        distribution, the Logistic-Normal can capture correlations between genes
        that arise from the posterior uncertainty in r.

        Mathematically, this fits:
            log(ρ) ~ MVN(loc, W W^T + D)
        where ρ are the normalized expression proportions on the simplex.

        Parameters
        ----------
        rng_key : random.PRNGKey, default=random.PRNGKey(42)
            JAX random number generator key
        n_samples_dirichlet : int, default=100
            Number of Dirichlet samples to draw per posterior sample for fitting
            the Logistic-Normal distribution. More samples give better estimates
            but use more memory.
        rank : Optional[int], default=None
            Rank of the low-rank covariance approximation. If None,
            automatically uses min(n_genes, 50). Lower rank values use less
            memory and may improve generalization.
        distribution_type : str, default="softmax"
            Type of compositional distribution to fit. Must be one of:
                - "softmax": SoftmaxNormal (symmetric, all genes treated
                  equally, cannot evaluate log_prob but good for
                  sampling/visualization)
                - "alr": LowRankLogisticNormal (ALR-based, uses last gene as
                  reference, can evaluate log_prob for Bayesian inference)
        remove_mean : bool, default=False
            If True, removes the grand mean from ALR-transformed samples before
            fitting the low-rank covariance structure. This is useful when:
                - Data comes from a single cell type (homogeneous population)
                - You want to focus on gene-gene co-expression patterns
                - The first PC captures a dominant mean effect (>10× larger than
                  PC2)

            When False (default), the fitted distribution captures both the mean
            composition and the covariance structure. When True, PC1 will
            represent the largest source of *variation* rather than the mean
            effect.

            **Biological Interpretation**:
                - remove_mean=False: Captures "what is the average expression +
                  variability"
                - remove_mean=True: Captures "how do genes co-vary relative to
                  baseline"
        verbose : bool, default=True
            If True, prints progress messages and shows progress bars during the
            fitting process

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
            parameter is not found in posterior samples

        See Also
        --------
        normalize_counts : Alternative method that fits Dirichlet distributions

        Notes
        -----
        The Logistic-Normal distribution naturally captures correlation between
        genes, which is important for co-regulated gene modules. This
        correlation structure is inherited from the low-rank structure in the
        posterior over r parameters, making it particularly suitable for models
        that use low-rank Gaussian guides.

        When using the scipy backend, only the base multivariate normal
        distribution in log-space is returned. You'll need to apply a softmax
        transformation to samples to get them on the simplex.

        Examples
        --------
        >>> # For a non-mixture model
        >>> fitted = results.fit_logistic_normal(
        ...     n_samples_dirichlet=100,
        ...     rank=20,
        ...     verbose=True
        ... )
        >>> print(fitted['loc'].shape)  # (n_genes,)
        >>> print(fitted['cov_factor'].shape)  # (n_genes, 20)
        >>> # Sample from the distribution
        >>> samples = fitted['distributions'].sample(key, (1000,))

        >>> # For a mixture model
        >>> fitted = results.fit_logistic_normal(
        ...     n_samples_dirichlet=100,
        ...     rank=20
        ... )
        >>> print(fitted['loc'].shape)  # (n_components, n_genes)
        >>> print(len(fitted['distributions']))  # n_components
        """
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        # Convert to canonical form to ensure r parameter is available
        self._convert_to_canonical()

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

        # Use the shared fitting function with distribution class
        return fit_logistic_normal_from_posterior(
            posterior_samples=self.posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            rank=rank,
            distribution_class=distribution_class,
            remove_mean=remove_mean,
            verbose=verbose,
        )

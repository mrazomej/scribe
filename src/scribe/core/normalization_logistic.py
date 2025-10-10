"""
Low-rank Logistic-Normal fitting for normalized gene expression.

This module provides functionality to fit low-rank Logistic-Normal distributions
to normalized expression profiles, preserving correlation structure from the
posterior.
"""

from typing import Dict, Optional, Union, Tuple
import jax.numpy as jnp
from jax import random

from ..stats import (
    sample_dirichlet_from_parameters,
    SoftmaxNormal,
    LowRankLogisticNormal,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------------------------
# Distribution Factory Functions
# ------------------------------------------------------------------------------


def _create_softmax_distribution(loc, cov_factor, cov_diag):
    """Create SoftmaxNormal distribution (symmetric, D-dimensional)."""
    return SoftmaxNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag)


def _create_alr_distribution(loc, cov_factor, cov_diag):
    """Create LowRankLogisticNormal distribution (ALR-based, (D-1)-dimensional)."""
    alr_loc = loc[:-1]
    alr_cov_factor = cov_factor[:-1, :]
    alr_cov_diag = cov_diag[:-1]
    return LowRankLogisticNormal(
        loc=alr_loc, cov_factor=alr_cov_factor, cov_diag=alr_cov_diag
    )


# Factory dispatch dictionary
DISTRIBUTION_FACTORY = {
    SoftmaxNormal: _create_softmax_distribution,
    LowRankLogisticNormal: _create_alr_distribution,
}


# ------------------------------------------------------------------------------
# Logistic-Normal Fitting from Posterior Samples
# ------------------------------------------------------------------------------


def fit_logistic_normal_from_posterior(
    posterior_samples: Dict[str, jnp.ndarray],
    n_components: Optional[int] = None,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples_dirichlet: int = 1,
    rank: Optional[int] = None,
    distribution_class: type = SoftmaxNormal,
    verbose: bool = True,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Fit a low-rank Logistic-Normal distribution to normalized expression
    samples.

    This function takes posterior samples of the dispersion parameter (r) and
    fits a low-rank Logistic-Normal distribution to capture the correlation
    structure in normalized expression profiles. This preserves the low-rank
    correlation structure discovered during inference, which is lost when
    fitting a single Dirichlet distribution.

    The Logistic-Normal distribution is defined as:
        log(ρ) ~ MVN(loc, W W^T + D)
    where ρ are the normalized expression proportions (on the simplex).

    Parameters
    ----------
    posterior_samples : Dict[str, jnp.ndarray]
        Dictionary containing posterior samples, must include 'r' parameter
    n_components : Optional[int], default=None
        Number of mixture components. If None, assumes non-mixture model.
    rng_key : random.PRNGKey, default=random.PRNGKey(42)
        JAX random number generator key
    n_samples_dirichlet : int, default=100
        Number of Dirichlet samples to draw per posterior sample for fitting
    rank : Optional[int], default=None
        Rank of the low-rank covariance approximation. If None, uses
        min(n_genes, 50)
    distribution_class : type, default=SoftmaxNormal
        Type of compositional distribution to fit. Can be:
        - SoftmaxNormal: Symmetric distribution for sampling
        - LowRankLogisticNormal: ALR-based for log_prob evaluation
    verbose : bool, default=True
        If True, prints progress messages and shows progress bars

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Dictionary containing fitted Logistic-Normal parameters:
            - 'loc': Location parameter (mean in log-space)
            - 'cov_factor': Low-rank covariance factor W
            - 'cov_diag': Diagonal component of covariance D
            - 'mean_probabilities': Mean probabilities on the simplex
            - 'distribution': SoftmaxNormal distribution (symmetric, for sampling)
            - 'distribution_alr': LowRankLogisticNormal distribution (ALR-based,
              for log_prob evaluation)
            - 'base_distribution': Underlying LowRankMultivariateNormal in log-space
              (for backward compatibility)

        For non-mixture models:
            - loc: shape (n_genes,)
            - cov_factor: shape (n_genes, rank)
            - cov_diag: shape (n_genes,)
            - mean_probabilities: shape (n_genes,)
            - distribution: SoftmaxNormal object (symmetric, D-dimensional)
            - distribution_alr: LowRankLogisticNormal object (ALR, (D-1)-dimensional)

        For mixture models:
            - loc: shape (n_components, n_genes)
            - cov_factor: shape (n_components, n_genes, rank)
            - cov_diag: shape (n_components, n_genes)
            - mean_probabilities: shape (n_components, n_genes)
            - distributions: list of n_components SoftmaxNormal objects
            - distributions_alr: list of n_components LowRankLogisticNormal objects

    Raises
    ------
    ValueError
        If 'r' parameter is not found in posterior_samples

    Notes
    -----
    The Logistic-Normal distribution naturally captures correlation between
    genes, which is important for co-regulated gene modules. This correlation
    structure is inherited from the low-rank structure in the posterior over r
    parameters.

    Two Types of Distributions Returned
    ------------------------------------
    1. **SoftmaxNormal** (symmetric):
       - All genes treated equally (no reference gene)
       - Use for sampling and visualization
       - Cannot evaluate log_prob (softmax transform is singular)

    2. **LowRankLogisticNormal** (ALR-based):
       - Uses last gene as reference (asymmetric)
       - Can evaluate log_prob for observed data
       - Use for Bayesian inference or likelihood evaluation

    Examples
    --------
    Sampling from the distribution:

        >>> from jax import random
        >>> result = fit_logistic_normal_from_posterior(...)
        >>> # Sample from SoftmaxNormal (symmetric)
        >>> samples = result['distribution'].sample(random.PRNGKey(0), (100,))
        >>> # Samples are on the simplex (sum to 1)
        >>> assert samples.sum(axis=-1).allclose(1.0)

    Evaluating log probability:

        >>> # Use LowRankLogisticNormal for log_prob
        >>> log_prob = result['distribution_alr'].log_prob(samples[0])

    Note: For backward compatibility, `base_distribution` provides access to
    the underlying LowRankMultivariateNormal in log-space.
    """
    # Validate inputs
    if "r" not in posterior_samples:
        raise ValueError(
            "'r' parameter not found in posterior_samples. "
            "This method requires posterior samples of the dispersion "
            "parameter. Please run get_posterior_samples() first."
        )

    # Get r parameter samples
    r_samples = posterior_samples["r"]

    if verbose:
        print(f"Using r parameter samples with shape: {r_samples.shape}")

    # Determine if this is a mixture model
    is_mixture = n_components is not None and n_components > 1

    # Process mixture model
    if is_mixture:
        if verbose:
            print(f"Processing mixture model with {n_components} components")
        # n_components is guaranteed to be not None here due to the is_mixture
        # condition
        assert n_components is not None  # Type assertion for linter
        return _fit_logistic_normal_mixture(
            r_samples,
            n_components,
            rng_key,
            n_samples_dirichlet,
            rank,
            distribution_class,
            verbose,
        )
    # Process non-mixture model
    else:
        if verbose:
            print("Processing non-mixture model")
        return _fit_logistic_normal_non_mixture(
            r_samples,
            rng_key,
            n_samples_dirichlet,
            rank,
            distribution_class,
            verbose,
        )


# ------------------------------------------------------------------------------


def _fit_logistic_normal_non_mixture(
    r_samples: jnp.ndarray,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    rank: Optional[int],
    distribution_class: type,
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """Fit Logistic-Normal distribution for non-mixture models."""
    # r_samples shape: (n_posterior_samples, n_genes)
    n_posterior_samples, n_genes = r_samples.shape

    # Set default rank if not provided
    if rank is None:
        rank = min(n_genes, 50)

    if verbose:
        print(f"Using rank {rank} for low-rank covariance approximation")
        print(
            f"Generating {n_samples_dirichlet} Dirichlet samples for each of "
            f"{n_posterior_samples} posterior samples"
        )

    # Step 1: Sample from Dirichlet for each posterior sample
    all_log_samples = []

    iterator = range(n_posterior_samples)
    if verbose:
        iterator = tqdm(iterator, desc="Sampling from Dirichlet", unit="sample")

    for i in iterator:
        # Use r values as concentration parameters
        key_i = random.fold_in(rng_key, i)
        dirichlet_samples = sample_dirichlet_from_parameters(
            r_samples[i : i + 1],
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=key_i,
        )
        # Debug: check shape
        if i == 0 and verbose:
            print(f"  dirichlet_samples shape: {dirichlet_samples.shape}")

        # dirichlet_samples shape: (1, n_genes, n_samples_dirichlet)
        # Take log and reshape
        log_samples = jnp.log(
            dirichlet_samples[0]
        )  # (n_genes, n_samples_dirichlet)

        if i == 0 and verbose:
            print(f"  log_samples shape after indexing: {log_samples.shape}")
            print(f"  log_samples.T shape: {log_samples.T.shape}")

        # When n_samples_dirichlet=1, need to ensure we keep the sample dimension
        if log_samples.ndim == 1:
            log_samples = log_samples.reshape(-1, 1)  # (n_genes, 1)

        all_log_samples.append(log_samples.T)  # (n_samples_dirichlet, n_genes)

    # Concatenate all samples:
    # (n_posterior_samples * n_samples_dirichlet, n_genes)
    all_log_samples = jnp.concatenate(all_log_samples, axis=0)

    # Ensure 2D shape (handle case when n_samples_dirichlet=1)
    if all_log_samples.ndim == 1:
        all_log_samples = all_log_samples.reshape(1, -1)

    if verbose:
        print(
            f"Fitting low-rank Logistic-Normal to {all_log_samples.shape[0]} samples "
            f"with {all_log_samples.shape[1]} features"
        )

    # Step 2: Fit low-rank MVN to log samples
    loc, cov_factor, cov_diag = _fit_low_rank_mvn(
        all_log_samples, rank=rank, verbose=verbose
    )

    # Step 3: Compute mean probabilities on the simplex
    # Use softmax(loc) as the mode of the Logistic-Normal
    exp_loc = jnp.exp(loc)
    mean_probabilities = exp_loc / jnp.sum(exp_loc)

    # Step 4: Create distribution objects
    results = {
        "loc": loc,
        "cov_factor": cov_factor,
        "cov_diag": cov_diag,
        "mean_probabilities": mean_probabilities,
    }

    if verbose:
        print(f"Creating {distribution_class.__name__} distribution")

    # Create distribution using factory
    distribution = DISTRIBUTION_FACTORY[distribution_class](
        loc, cov_factor, cov_diag
    )

    # Store in results with appropriate key
    if distribution_class == SoftmaxNormal:
        results["distribution"] = distribution
        results["base_distribution"] = (
            distribution.base_dist
        )  # Backward compatibility
    else:  # LowRankLogisticNormal
        results["distribution_alr"] = distribution

    return results


# ------------------------------------------------------------------------------


def _fit_logistic_normal_mixture(
    r_samples: jnp.ndarray,
    n_components: int,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    rank: Optional[int],
    distribution_class: type,
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """Fit Logistic-Normal distribution for mixture models."""
    # r_samples shape: (n_posterior_samples, n_components, n_genes)
    n_posterior_samples, n_components_check, n_genes = r_samples.shape

    if n_components_check != n_components:
        raise ValueError(
            f"Mismatch between n_components ({n_components}) and "
            f"r_samples shape ({r_samples.shape})"
        )

    # Set default rank if not provided
    if rank is None:
        rank = min(n_genes, 50)

    if verbose:
        print(f"Using rank {rank} for low-rank covariance approximation")
        print(
            f"Generating {n_samples_dirichlet} Dirichlet samples for each of "
            f"{n_posterior_samples} posterior samples and {n_components} "
            f"components"
        )

    # Initialize storage for fitted parameters
    locs = []
    cov_factors = []
    cov_diags = []
    mean_probs = []
    distributions = []
    distributions_alr = []

    # Fit one Logistic-Normal per component
    for c in range(n_components):
        if verbose:
            print(
                f"\nFitting Logistic-Normal for component {c + 1}/{n_components}"
            )

        # Step 1: Sample from Dirichlet for this component
        all_log_samples = []

        iterator = range(n_posterior_samples)
        if verbose:
            iterator = tqdm(
                iterator, desc=f"Component {c + 1} - Sampling", unit="sample"
            )

        for i in iterator:
            # Use r values for this component as concentration parameters
            key_i_c = random.fold_in(rng_key, i * n_components + c)
            dirichlet_samples = sample_dirichlet_from_parameters(
                r_samples[i, c : c + 1],
                n_samples_dirichlet=n_samples_dirichlet,
                rng_key=key_i_c,
            )
            # dirichlet_samples shape: (1, n_genes, n_samples_dirichlet)
            # Take log and reshape
            log_samples = jnp.log(
                dirichlet_samples[0]
            )  # (n_genes, n_samples_dirichlet)

            # When n_samples_dirichlet=1, need to ensure we keep the sample dimension
            if log_samples.ndim == 1:
                log_samples = log_samples.reshape(-1, 1)  # (n_genes, 1)

            all_log_samples.append(
                log_samples.T
            )  # (n_samples_dirichlet, n_genes)

        # Concatenate: (n_posterior_samples * n_samples_dirichlet, n_genes)
        all_log_samples = jnp.concatenate(all_log_samples, axis=0)

        # Ensure 2D shape (handle case when n_samples_dirichlet=1)
        if all_log_samples.ndim == 1:
            all_log_samples = all_log_samples.reshape(1, -1)

        if verbose:
            print(f"  Fitting to {all_log_samples.shape[0]} samples")

        # Step 2: Fit low-rank MVN to log samples
        loc, cov_factor, cov_diag = _fit_low_rank_mvn(
            all_log_samples, rank=rank, verbose=False
        )

        locs.append(loc)
        cov_factors.append(cov_factor)
        cov_diags.append(cov_diag)

        # Step 3: Compute mean probabilities on the simplex
        exp_loc = jnp.exp(loc)
        mean_prob = exp_loc / jnp.sum(exp_loc)
        mean_probs.append(mean_prob)

        # Step 4: Create distribution for this component using factory
        distribution = DISTRIBUTION_FACTORY[distribution_class](
            loc, cov_factor, cov_diag
        )

        # Append to appropriate list
        if distribution_class == SoftmaxNormal:
            distributions.append(distribution)
        else:  # LowRankLogisticNormal
            distributions_alr.append(distribution)

    # Stack results
    results = {
        "loc": jnp.stack(locs, axis=0),  # (n_components, n_genes)
        "cov_factor": jnp.stack(
            cov_factors, axis=0
        ),  # (n_components, n_genes, rank)
        "cov_diag": jnp.stack(cov_diags, axis=0),  # (n_components, n_genes)
        "mean_probabilities": jnp.stack(
            mean_probs, axis=0
        ),  # (n_components, n_genes)
    }

    # Add appropriate distribution key
    if distribution_class == SoftmaxNormal:
        results["distributions"] = distributions
    else:  # LowRankLogisticNormal
        results["distributions_alr"] = distributions_alr

    if verbose:
        dist_list = (
            distributions
            if distribution_class == SoftmaxNormal
            else distributions_alr
        )
        print(
            f"\nCreated {len(dist_list)} {distribution_class.__name__} distributions"
        )

    return results


# ------------------------------------------------------------------------------


def _fit_low_rank_mvn(
    samples: jnp.ndarray,
    rank: int,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fit a low-rank multivariate normal distribution to samples using PCA.

    Parameters
    ----------
    samples : jnp.ndarray
        Samples to fit, shape (n_samples, n_features)
    rank : int
        Rank of the low-rank approximation
    verbose : bool
        Whether to print progress

    Returns
    -------
    loc : jnp.ndarray
        Mean vector, shape (n_features,)
    cov_factor : jnp.ndarray
        Low-rank covariance factor W, shape (n_features, rank)
    cov_diag : jnp.ndarray
        Diagonal component D, shape (n_features,)
    """
    n_samples, n_features = samples.shape

    # Check if we have enough samples
    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples to fit low-rank MVN, but got {n_samples}. "
            f"Make sure you have multiple posterior samples of r, or increase "
            f"n_samples_dirichlet to > 1."
        )

    # Compute mean
    loc = jnp.mean(samples, axis=0)

    # Center the data
    centered = samples - loc

    if verbose:
        print(
            f"  Computing SVD of centered data ({n_samples} x {n_features})..."
        )

    # Use SVD on the centered data matrix directly (more memory efficient)
    # For X (n_samples x n_features): X = U @ S @ V.T
    # Then: Cov(X) = (1/(n-1)) * X.T @ X = V @ (S^2/(n-1)) @ V.T
    # So V contains the eigenvectors of the covariance, and S^2/(n-1) are the
    # eigenvalues

    # This is much more memory efficient when n_samples << n_features (e.g., 100
    # << 32K) because we only compute SVD of (n_samples x n_features) matrix
    # rather than eigendecomposition of (n_features x n_features) covariance
    # matrix

    U, singular_values, Vt = jnp.linalg.svd(centered, full_matrices=False)
    # U: (n_samples, min(n_samples, n_features))
    # singular_values: (min(n_samples, n_features),)
    # Vt: (min(n_samples, n_features), n_features)

    # Convert singular values to eigenvalues of covariance
    eigenvalues = (singular_values**2) / (n_samples - 1)
    # Eigenvectors are rows of Vt, we want columns, so transpose
    eigenvectors = Vt.T  # (n_features, min(n_samples, n_features))

    # Sort in descending order (SVD already returns in descending order, but be
    # explicit)
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp rank to available components
    # (can't extract more components than min(n_samples, n_features))
    effective_rank = min(rank, len(eigenvalues))
    if verbose and effective_rank < rank:
        print(
            f"  Clamping rank from {rank} to {effective_rank} (max available)"
        )

    # Take top k eigenvalues/eigenvectors for low-rank part
    top_k_eigenvalues = eigenvalues[:effective_rank]
    top_k_eigenvectors = eigenvectors[:, :effective_rank]

    # Construct W = U_k @ sqrt(S_k)
    # This gives W @ W.T = U_k @ S_k @ U_k.T
    cov_factor = top_k_eigenvectors * jnp.sqrt(
        jnp.maximum(top_k_eigenvalues, 0.0)
    )

    # Construct D as residual variance (remaining eigenvalues averaged)
    # Plus a small constant for numerical stability
    residual_eigenvalues = eigenvalues[effective_rank:]
    if len(residual_eigenvalues) > 0:
        # Use mean of residual eigenvalues for diagonal
        diag_value = jnp.mean(residual_eigenvalues)
    else:
        diag_value = 0.0

    # Add small constant for numerical stability
    cov_diag = jnp.full(n_features, diag_value) + 1e-4

    if verbose:
        total_var = jnp.sum(eigenvalues)
        explained_var = jnp.sum(top_k_eigenvalues)
        print(
            f"  Low-rank approximation explains "
            f"{100 * explained_var / total_var:.1f}% of variance"
        )

    return loc, cov_factor, cov_diag

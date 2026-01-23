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
# ALR (Additive Log-Ratio) Transform Helpers
# ------------------------------------------------------------------------------


def _apply_alr_transform(
    log_samples: jnp.ndarray, reference_index: Optional[int] = None
) -> jnp.ndarray:
    """
    Apply ALR (additive log-ratio) transformation to log-simplex samples.

    This computes ALR coordinates without materializing the full transformation
    matrix, making it memory-efficient for high-dimensional data.

    If the reference is r ∈ {0, ..., D-1}, the ALR coordinates are:
        z_i = log(ρ_i) - log(ρ_r)  for i ≠ r

    Parameters
    ----------
    log_samples : jnp.ndarray, shape (..., D)
        Log of simplex samples (can be batched)
    reference_index : Optional[int]
        Which component to use as the denominator in ALR. If None, use D-1.

    Returns
    -------
    alr_samples : jnp.ndarray, shape (..., D-1)
        ALR coordinates (reference component removed)
    """
    if log_samples.ndim < 1:
        raise ValueError("log_samples must have at least 1 dimension")

    D = log_samples.shape[-1]
    if D < 2:
        raise ValueError("ALR requires D >= 2.")

    r = D - 1 if reference_index is None else int(reference_index)
    if not (0 <= r < D):
        raise ValueError(f"reference_index must be in [0, {D-1}]")

    # Extract reference column: (..., 1)
    log_ref = jnp.expand_dims(log_samples[..., r], axis=-1)

    # Compute differences: log(ρ_i) - log(ρ_r) for all i
    log_diff = log_samples - log_ref  # (..., D)

    # Remove reference column to get ALR coordinates
    if r == D - 1:
        alr_samples = log_diff[..., :-1]
    elif r == 0:
        alr_samples = log_diff[..., 1:]
    else:
        alr_samples = jnp.concatenate(
            [log_diff[..., :r], log_diff[..., r + 1 :]], axis=-1
        )

    return alr_samples


# ------------------------------------------------------------------------------


def _inverse_alr(
    z: jnp.ndarray, reference_index: Optional[int] = None
) -> jnp.ndarray:
    """
    Map ALR coordinates back to the simplex via softmax.

    The inverse ALR is implemented by embedding z ∈ R^{D-1} to R^D by
    inserting a 0 for the reference log-coordinate, then applying softmax.

    Parameters
    ----------
    z : jnp.ndarray, shape (..., D-1)
        ALR coordinates.
    reference_index : Optional[int]
        Index of the reference component. If None, use the last component.

    Returns
    -------
    rho : jnp.ndarray, shape (..., D)
        Simplex points, each row sums to 1 and is in (0,1)^D.
    """
    *batch, d_minus_1 = z.shape
    D = d_minus_1 + 1
    r = D - 1 if reference_index is None else int(reference_index)
    if not (0 <= r < D):
        raise ValueError(f"reference_index must be in [0, {D-1}]")

    # Build logits z_full ∈ R^D by inserting 0 at the reference index
    zeros = jnp.zeros((*batch, 1))
    if r == D - 1:
        z_full = jnp.concatenate([z, zeros], axis=-1)
    elif r == 0:
        z_full = jnp.concatenate([zeros, z], axis=-1)
    else:
        z_full = jnp.concatenate([z[..., :r], zeros, z[..., r:]], axis=-1)

    # Softmax—invariant to constant shift—yields the inverse ALR map
    z_full = z_full - jnp.max(z_full, axis=-1, keepdims=True)  # stability
    exp = jnp.exp(z_full)
    rho = exp / jnp.sum(exp, axis=-1, keepdims=True)
    return rho


# ------------------------------------------------------------------------------
# Distribution Factory Functions
# ------------------------------------------------------------------------------


def _create_softmax_distribution(loc, cov_factor, cov_diag):
    """Create SoftmaxNormal distribution (symmetric, D-dimensional)."""
    return SoftmaxNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag)


def _create_alr_distribution(loc, cov_factor, cov_diag):
    """
    Create LowRankLogisticNormal distribution (ALR-based, (D-1)-dimensional).
    """
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
    rng_key: Optional[random.PRNGKey] = None,
    n_samples_dirichlet: int = 1,
    rank: Optional[int] = None,
    distribution_class: type = SoftmaxNormal,
    remove_mean: bool = False,
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
    rng_key : random.PRNGKey, optional
        JAX random number generator key. Defaults to random.PRNGKey(42) if None
    n_samples_dirichlet : int, default=100
        Number of Dirichlet samples to draw per posterior sample for fitting
    rank : Optional[int], default=None
        Rank of the low-rank covariance approximation. If None, uses
        min(n_genes, 50)
    distribution_class : type, default=SoftmaxNormal
        Type of compositional distribution to fit. Can be: - SoftmaxNormal:
        Symmetric distribution for sampling - LowRankLogisticNormal: ALR-based
        for log_prob evaluation
    remove_mean : bool, default=False
        If True, removes the grand mean from ALR-transformed samples before
        fitting, focusing on co-variation patterns rather than mean composition.
        Recommended for single cell type data where PC1 >> PC2 (dominant mean
        effect).
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
            - 'distribution': SoftmaxNormal distribution (symmetric, for
              sampling)
            - 'distribution_alr': LowRankLogisticNormal distribution (ALR-based,
              for log_prob evaluation)
            - 'base_distribution': Underlying LowRankMultivariateNormal in
              log-space (for backward compatibility)

        For non-mixture models:
            - loc: shape (n_genes,)
            - cov_factor: shape (n_genes, rank)
            - cov_diag: shape (n_genes,)
            - mean_probabilities: shape (n_genes,)
            - distribution: SoftmaxNormal object (symmetric, D-dimensional)
            - distribution_alr: LowRankLogisticNormal object (ALR,
              (D-1)-dimensional)

        For mixture models:
            - loc: shape (n_components, n_genes)
            - cov_factor: shape (n_components, n_genes, rank)
            - cov_diag: shape (n_components, n_genes)
            - mean_probabilities: shape (n_components, n_genes)
            - distributions: list of n_components SoftmaxNormal objects
            - distributions_alr: list of n_components LowRankLogisticNormal
              objects

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
    1. **SoftmaxNormal** (symmetric): - All genes treated equally (no reference
       gene) - Use for sampling and visualization - Cannot evaluate log_prob
       (softmax transform is singular)

    2. **LowRankLogisticNormal** (ALR-based): - Uses last gene as reference
       (asymmetric) - Can evaluate log_prob for observed data - Use for Bayesian
       inference or likelihood evaluation

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

    Note: For backward compatibility, `base_distribution` provides access to the
    underlying LowRankMultivariateNormal in log-space.
    """
    # Create default RNG key if not provided (lazy initialization)
    if rng_key is None:
        rng_key = random.PRNGKey(42)

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
            remove_mean,
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
            remove_mean,
            verbose,
        )


# ------------------------------------------------------------------------------


def _fit_logistic_normal_non_mixture(
    r_samples: jnp.ndarray,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    rank: Optional[int],
    distribution_class: type,
    remove_mean: bool,
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

        # When n_samples_dirichlet=1, need to ensure we keep the sample
        # dimension
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
            f"Collected {all_log_samples.shape[0]} log-Dirichlet samples "
            f"with {all_log_samples.shape[1]} features"
        )

    # Step 2: Apply ALR transformation before fitting
    # This is critical! We must transform to ALR space BEFORE fitting the MVN
    # Otherwise we're fitting in the wrong space (D-dimensional vs
    # (D-1)-dimensional)
    if verbose:
        print(
            f"Applying ALR transformation to map to (D-1)-dimensional space..."
        )

    # Use memory-efficient ALR transform (no matrix materialization)
    alr_samples = _apply_alr_transform(
        all_log_samples, reference_index=None
    )  # (n_samples, n_genes-1)

    # Step 2b: Optionally remove grand mean to focus on co-variation
    if remove_mean:
        alr_grand_mean = jnp.mean(alr_samples, axis=0, keepdims=True)
        alr_samples_centered = alr_samples - alr_grand_mean
        if verbose:
            print(
                f"Removed grand mean from ALR samples "
                f"(focusing on co-variation patterns)"
            )
    else:
        alr_samples_centered = alr_samples
        alr_grand_mean = None

    if verbose:
        print(
            f"Fitting low-rank Logistic-Normal in ALR space: "
            f"{alr_samples_centered.shape[0]} samples, "
            f"{alr_samples_centered.shape[1]} ALR dimensions"
        )

    # Step 3: Fit low-rank MVN in ALR space
    alr_loc, alr_cov_factor, alr_cov_diag = _fit_low_rank_mvn(
        alr_samples_centered, rank=rank, verbose=verbose
    )

    # Step 3b: Add back the grand mean if it was removed
    # (so the returned distribution still represents the full data)
    if remove_mean and alr_grand_mean is not None:
        alr_loc = alr_loc + alr_grand_mean.squeeze()

    # Step 4: Embed ALR parameters back to D dimensions for SoftmaxNormal
    # Insert 0 at reference position (last component) to get D-dimensional params
    loc = jnp.concatenate([alr_loc, jnp.array([0.0])], axis=0)
    cov_factor = jnp.concatenate(
        [alr_cov_factor, jnp.zeros((1, alr_cov_factor.shape[1]))], axis=0
    )
    cov_diag = jnp.concatenate([alr_cov_diag, jnp.array([0.0])], axis=0)

    # Step 5: Compute mean probabilities on the simplex using inverse ALR
    mean_probabilities = _inverse_alr(alr_loc, reference_index=None)

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
    remove_mean: bool,
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
                f"\nFitting Logistic-Normal for component "
                f"{c + 1}/{n_components}"
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

            # When n_samples_dirichlet=1, need to ensure we keep the sample
            # dimension
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
            print(
                f"  Collected {all_log_samples.shape[0]} log-Dirichlet samples"
            )

        # Step 2: Apply ALR transformation before fitting
        alr_samples = _apply_alr_transform(
            all_log_samples, reference_index=None
        )  # (n_samples, n_genes-1)

        if verbose:
            print(
                f"  Applying ALR transformation: "
                f"{alr_samples.shape[1]} ALR dimensions"
            )

        # Step 2b: Optionally remove grand mean
        if remove_mean:
            alr_grand_mean = jnp.mean(alr_samples, axis=0, keepdims=True)
            alr_samples_centered = alr_samples - alr_grand_mean
            if verbose:
                print(f"  Removed grand mean (focusing on co-variation)")
        else:
            alr_samples_centered = alr_samples
            alr_grand_mean = None

        # Step 3: Fit low-rank MVN in ALR space
        alr_loc, alr_cov_factor, alr_cov_diag = _fit_low_rank_mvn(
            alr_samples_centered, rank=rank, verbose=False
        )

        # Step 3b: Add back grand mean if removed
        if remove_mean and alr_grand_mean is not None:
            alr_loc = alr_loc + alr_grand_mean.squeeze()

        # Step 4: Embed ALR parameters back to D dimensions
        loc = jnp.concatenate([alr_loc, jnp.array([0.0])], axis=0)
        cov_factor = jnp.concatenate(
            [alr_cov_factor, jnp.zeros((1, alr_cov_factor.shape[1]))], axis=0
        )
        cov_diag = jnp.concatenate([alr_cov_diag, jnp.array([0.0])], axis=0)

        locs.append(loc)
        cov_factors.append(cov_factor)
        cov_diags.append(cov_diag)

        # Step 5: Compute mean probabilities on the simplex using inverse ALR
        mean_prob = _inverse_alr(alr_loc, reference_index=None)
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
            f"\nCreated {len(dist_list)} {distribution_class.__name__} "
            "distributions"
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
        print(f"  Top 5 eigenvalues: {eigenvalues[:5].tolist()}")

        # Check for dominant first eigenvalue (may indicate mean effect)
        if len(eigenvalues) >= 2 and eigenvalues[0] > 10 * eigenvalues[1]:
            ratio = float(eigenvalues[0] / eigenvalues[1])
            print(
                f"  ⚠️  First eigenvalue is {ratio:.1f}× larger than second "
                f"(may indicate strong mean effect)"
            )

        # Detailed variance breakdown
        ev1 = 100 * eigenvalues[0] / total_var
        print(
            f"  Variance distribution: "
            f"PC1={ev1:.1f}%, "
            f"top 10={100 * jnp.sum(eigenvalues[:10]) / total_var:.1f}%, "
            f"top 50={100 * jnp.sum(eigenvalues[:50]) / total_var:.1f}%, "
            f"top 100={100 * jnp.sum(eigenvalues[:100]) / total_var:.1f}%"
        )

    return loc, cov_factor, cov_diag

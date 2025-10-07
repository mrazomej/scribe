"""
Low-rank Logistic-Normal fitting for normalized gene expression.

This module provides functionality to fit low-rank Logistic-Normal distributions
to normalized expression profiles, preserving correlation structure from the
posterior.
"""

from typing import Dict, Optional, Union, Tuple
import jax.numpy as jnp
from jax import random
import warnings

import numpyro.distributions as dist
from ..stats import sample_dirichlet_from_parameters
from ..utils import numpyro_to_scipy

try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------------------------
# Logistic-Normal Fitting from Posterior Samples
# ------------------------------------------------------------------------------


def fit_logistic_normal_from_posterior(
    posterior_samples: Dict[str, jnp.ndarray],
    n_components: Optional[int] = None,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples_dirichlet: int = 1,
    rank: Optional[int] = None,
    backend: str = "numpyro",
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
    backend : str, default="numpyro"
        Statistical package to use for distributions. Must be one of: -
        "numpyro": Returns numpyro.distributions - "scipy": Returns scipy.stats
        distributions
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
        If 'r' parameter is not found in posterior_samples

    Notes
    -----
    The Logistic-Normal distribution naturally captures correlation between
    genes, which is important for co-regulated gene modules. This correlation
    structure is inherited from the low-rank structure in the posterior over r
    parameters.
    """
    # Validate inputs
    if "r" not in posterior_samples:
        raise ValueError(
            "'r' parameter not found in posterior_samples. "
            "This method requires posterior samples of the dispersion "
            "parameter. Please run get_posterior_samples() first."
        )

    if backend not in ["scipy", "numpyro"]:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'scipy' or 'numpyro'"
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
            backend,
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
            backend,
            verbose,
        )


# ------------------------------------------------------------------------------


def _fit_logistic_normal_non_mixture(
    r_samples: jnp.ndarray,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    rank: Optional[int],
    backend: str,
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
        print(f"Creating Logistic-Normal distribution with {backend} backend")

    # Create numpyro distribution
    base = dist.LowRankMultivariateNormal(
        loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
    )
    # Transform to simplex using softmax
    softmax_transform = dist.transforms.SoftmaxTransform()
    logistic_normal_dist = dist.TransformedDistribution(base, softmax_transform)

    if backend == "scipy":
        # Note: scipy doesn't have a direct Logistic-Normal distribution
        # We'll return the base MVN and note that it needs softmax transform
        warnings.warn(
            "scipy backend returns the base multivariate normal in log-space. "
            "Apply softmax transformation to get samples on the simplex.",
            UserWarning,
        )
        logistic_normal_dist = numpyro_to_scipy(base)

    results["distributions"] = logistic_normal_dist

    return results


# ------------------------------------------------------------------------------


def _fit_logistic_normal_mixture(
    r_samples: jnp.ndarray,
    n_components: int,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    rank: Optional[int],
    backend: str,
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

        # Step 4: Create distribution for this component
        base = dist.LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )
        softmax_transform = dist.transforms.SoftmaxTransform()
        logistic_normal_dist = dist.TransformedDistribution(
            base, softmax_transform
        )

        if backend == "scipy":
            warnings.warn(
                "scipy backend returns the base multivariate normal in log-space. "
                "Apply softmax transformation to get samples on the simplex.",
                UserWarning,
            )
            logistic_normal_dist = numpyro_to_scipy(base)

        distributions.append(logistic_normal_dist)

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
        "distributions": distributions,  # list of n_components distributions
    }

    if verbose:
        print(f"\nCreated {len(distributions)} Logistic-Normal distributions")

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

    # Compute empirical covariance
    # Using (1/(n-1)) for unbiased estimate
    cov = jnp.cov(centered, rowvar=False, bias=False)

    # Ensure cov is 2D (can be 0D or 1D for edge cases)
    if cov.ndim == 0:
        # Single feature case - make it a 1x1 matrix
        cov = cov.reshape(1, 1)
    elif cov.ndim == 1:
        # This shouldn't happen with rowvar=False, but just in case
        cov = jnp.diag(cov)

    # Use SVD for low-rank approximation
    # cov = U @ S @ U.T where S is diagonal
    # We want W @ W.T + D ≈ cov

    # Compute eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    # Sort in descending order
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top k eigenvalues/eigenvectors for low-rank part
    top_k_eigenvalues = eigenvalues[:rank]
    top_k_eigenvectors = eigenvectors[:, :rank]

    # Construct W = U_k @ sqrt(S_k)
    # This gives W @ W.T = U_k @ S_k @ U_k.T
    cov_factor = top_k_eigenvectors * jnp.sqrt(
        jnp.maximum(top_k_eigenvalues, 0.0)
    )

    # Construct D as residual variance (remaining eigenvalues averaged)
    # Plus a small constant for numerical stability
    residual_eigenvalues = eigenvalues[rank:]
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

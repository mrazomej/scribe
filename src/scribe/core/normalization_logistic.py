"""
Low-rank Logistic-Normal fitting for normalized gene expression.

This module provides functionality to fit low-rank Logistic-Normal distributions
to normalized expression profiles, preserving correlation structure from the
posterior.

Performance
-----------
Dirichlet sampling is performed in batches of ``batch_size`` posterior samples
(default 256) to balance GPU throughput against memory usage.  The batch size is
user-configurable all the way from the public API
(``ScribeSVIResults.fit_logistic_normal``) down to the internal helpers.

Future work
~~~~~~~~~~~
The pure-JAX helper ``_fit_low_rank_mvn_core`` is deliberately free of Python
side-effects so that it can be wrapped in ``jax.vmap`` across mixture
components.  A follow-up optimisation could vmap the entire per-component
pipeline (Dirichlet sampling → ALR → SVD → embedding) to run all components
in parallel on GPU.
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


# Default batch size for Dirichlet sampling.  2048 posterior samples × 20 000
# genes × 4 bytes ≈ 160 MB per batch — comfortable on any modern GPU and
# reduces 10 000 posterior samples to just 5 batched JAX dispatches.
_DEFAULT_BATCH_SIZE: int = 2048


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
# Batched Dirichlet Sampling
# ------------------------------------------------------------------------------


def _batched_dirichlet_sample(
    r_samples: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    verbose: bool = False,
) -> jnp.ndarray:
    """
    Sample from Dirichlet distributions in batches to balance speed and memory.

    Instead of calling ``sample_dirichlet_from_parameters`` once per posterior
    sample (O(N) Python→JAX round-trips), this helper processes posterior
    samples in chunks of ``batch_size``, reducing dispatches to
    O(ceil(N / batch_size)).

    Parameters
    ----------
    r_samples : jnp.ndarray, shape (N, D)
        Concentration parameters for N Dirichlet distributions over D genes.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int, default=2048
        Number of posterior samples to process in a single batched JAX call.
        Larger values use more memory but fewer dispatches.  At D=20 000 and
        ``n_samples_dirichlet=1``, each batch of 2048 requires ~160 MB.
    verbose : bool, default=False
        If True, show a tqdm progress bar over batches.

    Returns
    -------
    jnp.ndarray
        If ``n_samples_dirichlet == 1``:
            shape (N, D) — one simplex sample per posterior sample.
        If ``n_samples_dirichlet > 1``:
            shape (N * n_samples_dirichlet, D) — all draws flattened along the
            first axis so the output is ready for ALR → SVD fitting.
    """
    N, D = r_samples.shape

    # Collect chunks of Dirichlet samples
    chunks = []

    # Build an iterator over batch start indices
    starts = range(0, N, batch_size)
    if verbose:
        starts = tqdm(
            starts,
            desc="Batched Dirichlet sampling",
            unit="batch",
            total=(N + batch_size - 1) // batch_size,
        )

    for start in starts:
        # Slice out the current batch of concentration parameters
        end = min(start + batch_size, N)
        r_batch = r_samples[start:end]  # (B, D)

        # Derive a deterministic sub-key for this batch
        key_batch = random.fold_in(rng_key, start)

        # Single batched JAX call: Dirichlet(r_batch).sample(...)
        batch_samples = sample_dirichlet_from_parameters(
            r_batch,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=key_batch,
        )
        # batch_samples shape:
        #   n_samples_dirichlet == 1  →  (B, D)
        #   n_samples_dirichlet  > 1  →  (B, D, S)

        if n_samples_dirichlet == 1:
            # Already (B, D) — append directly
            chunks.append(batch_samples)
        else:
            # (B, D, S) → transpose to (B, S, D) → flatten to (B*S, D)
            chunks.append(
                batch_samples.transpose(0, 2, 1).reshape(-1, D)
            )

    # Concatenate all batches along the sample axis
    return jnp.concatenate(chunks, axis=0)


def _batched_dirichlet_sample_raw(
    r_samples: jnp.ndarray,
    n_samples_dirichlet: int,
    rng_key: random.PRNGKey,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    verbose: bool = False,
) -> jnp.ndarray:
    """
    Sample from Dirichlet distributions in batches, preserving original shape.

    Unlike ``_batched_dirichlet_sample`` (which flattens the
    ``n_samples_dirichlet`` axis for fitting), this variant keeps the
    per-posterior-sample structure intact so that raw samples can be returned
    to the caller (e.g. for ``store_samples=True`` in ``normalize_counts``).

    Parameters
    ----------
    r_samples : jnp.ndarray, shape (N, D)
        Concentration parameters for N Dirichlet distributions over D genes.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rng_key : random.PRNGKey
        JAX PRNG key.
    batch_size : int, default=2048
        Number of posterior samples per batched JAX call.
    verbose : bool, default=False
        If True, show a tqdm progress bar over batches.

    Returns
    -------
    jnp.ndarray
        If ``n_samples_dirichlet == 1``:
            shape (N, D).
        If ``n_samples_dirichlet > 1``:
            shape (N, D, n_samples_dirichlet).
    """
    N, D = r_samples.shape

    # Collect chunks of Dirichlet samples
    chunks = []

    # Build an iterator over batch start indices
    starts = range(0, N, batch_size)
    if verbose:
        starts = tqdm(
            starts,
            desc="Batched Dirichlet sampling",
            unit="batch",
            total=(N + batch_size - 1) // batch_size,
        )

    for start in starts:
        # Slice out the current batch of concentration parameters
        end = min(start + batch_size, N)
        r_batch = r_samples[start:end]  # (B, D)

        # Derive a deterministic sub-key for this batch
        key_batch = random.fold_in(rng_key, start)

        # Single batched JAX call
        batch_samples = sample_dirichlet_from_parameters(
            r_batch,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=key_batch,
        )
        # batch_samples shape:
        #   n_samples_dirichlet == 1  →  (B, D)
        #   n_samples_dirichlet  > 1  →  (B, D, S)
        chunks.append(batch_samples)

    # Concatenate all batches along the first (posterior-sample) axis
    return jnp.concatenate(chunks, axis=0)


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
    batch_size: int = _DEFAULT_BATCH_SIZE,
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
        Dictionary containing posterior samples, must include 'r' parameter.
    n_components : Optional[int], default=None
        Number of mixture components. If None, assumes non-mixture model.
    rng_key : random.PRNGKey, optional
        JAX random number generator key. Defaults to random.PRNGKey(42) if
        None.
    n_samples_dirichlet : int, default=1
        Number of Dirichlet samples to draw per posterior sample for fitting.
    rank : Optional[int], default=None
        Rank of the low-rank covariance approximation. If None, uses
        min(n_genes, 50).
    distribution_class : type, default=SoftmaxNormal
        Type of compositional distribution to fit. Can be:

        - SoftmaxNormal: Symmetric distribution for sampling
        - LowRankLogisticNormal: ALR-based for log_prob evaluation
    remove_mean : bool, default=False
        If True, removes the grand mean from ALR-transformed samples before
        fitting, focusing on co-variation patterns rather than mean
        composition. Recommended for single cell type data where PC1 >> PC2
        (dominant mean effect).
    batch_size : int, default=2048
        Number of posterior samples to process in each batched Dirichlet
        sampling call.  Larger values use more GPU memory but require fewer
        Python-to-JAX dispatches.  At D=20 000 and ``n_samples_dirichlet=1``,
        each batch of 2048 requires ~160 MB.  Reduce if you encounter OOM
        errors; increase on large-memory GPUs for maximum throughput.
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
        If 'r' parameter is not found in posterior_samples.

    Notes
    -----
    The Logistic-Normal distribution naturally captures correlation between
    genes, which is important for co-regulated gene modules. This correlation
    structure is inherited from the low-rank structure in the posterior over r
    parameters.

    Two Types of Distributions Returned
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. **SoftmaxNormal** (symmetric): All genes treated equally (no reference
       gene). Use for sampling and visualization. Cannot evaluate log_prob
       (softmax transform is singular).

    2. **LowRankLogisticNormal** (ALR-based): Uses last gene as reference
       (asymmetric). Can evaluate log_prob for observed data. Use for Bayesian
       inference or likelihood evaluation.

    Examples
    --------
    Sampling from the distribution:

        >>> from jax import random
        >>> result = fit_logistic_normal_from_posterior(
        ...     posterior_samples, batch_size=512
        ... )
        >>> # Sample from SoftmaxNormal (symmetric)
        >>> samples = result['distribution'].sample(random.PRNGKey(0), (100,))

    Evaluating log probability:

        >>> # Use LowRankLogisticNormal for log_prob
        >>> log_prob = result['distribution_alr'].log_prob(samples[0])
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
            batch_size,
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
            batch_size,
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
    batch_size: int,
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Fit Logistic-Normal distribution for non-mixture models.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape (n_posterior_samples, n_genes)
        Posterior samples of the Dirichlet concentration parameter.
    rng_key : random.PRNGKey
        JAX PRNG key.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rank : Optional[int]
        Rank of the low-rank covariance approximation.
    distribution_class : type
        SoftmaxNormal or LowRankLogisticNormal.
    remove_mean : bool
        Whether to centre ALR samples before fitting.
    batch_size : int
        Posterior samples processed per batched Dirichlet call.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Fitted Logistic-Normal parameters and distribution object(s).
    """
    # r_samples shape: (n_posterior_samples, n_genes)
    n_posterior_samples, n_genes = r_samples.shape

    # Set default rank if not provided
    if rank is None:
        rank = min(n_genes, 50)

    if verbose:
        print(f"Using rank {rank} for low-rank covariance approximation")
        n_batches = (n_posterior_samples + batch_size - 1) // batch_size
        print(
            f"Generating {n_samples_dirichlet} Dirichlet sample(s) for each "
            f"of {n_posterior_samples} posterior samples "
            f"(batch_size={batch_size}, {n_batches} batches)"
        )

    # ------------------------------------------------------------------
    # Step 1: Batched Dirichlet sampling
    # ------------------------------------------------------------------
    # _batched_dirichlet_sample returns:
    #   n_samples_dirichlet == 1  →  (N, D)
    #   n_samples_dirichlet  > 1  →  (N * S, D)   (flattened for fitting)
    all_simplex_samples = _batched_dirichlet_sample(
        r_samples,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Take the log (element-wise) to move to log-simplex space
    all_log_samples = jnp.log(all_simplex_samples)

    if verbose:
        print(
            f"Collected {all_log_samples.shape[0]} log-Dirichlet samples "
            f"with {all_log_samples.shape[1]} features"
        )

    # ------------------------------------------------------------------
    # Step 2: ALR transformation  (D → D-1 dimensions)
    # ------------------------------------------------------------------
    if verbose:
        print(
            "Applying ALR transformation to map to (D-1)-dimensional space..."
        )

    # Memory-efficient ALR transform (no matrix materialisation)
    alr_samples = _apply_alr_transform(
        all_log_samples, reference_index=None
    )  # (n_total, n_genes - 1)

    # Step 2b: Optionally remove grand mean to focus on co-variation
    if remove_mean:
        alr_grand_mean = jnp.mean(alr_samples, axis=0, keepdims=True)
        alr_samples_centered = alr_samples - alr_grand_mean
        if verbose:
            print(
                "Removed grand mean from ALR samples "
                "(focusing on co-variation patterns)"
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

    # ------------------------------------------------------------------
    # Step 3: Fit low-rank MVN in ALR space via SVD
    # ------------------------------------------------------------------
    alr_loc, alr_cov_factor, alr_cov_diag = _fit_low_rank_mvn(
        alr_samples_centered, rank=rank, verbose=verbose
    )

    # Step 3b: Add back the grand mean if it was removed
    # (so the returned distribution still represents the full data)
    if remove_mean and alr_grand_mean is not None:
        alr_loc = alr_loc + alr_grand_mean.squeeze()

    # ------------------------------------------------------------------
    # Step 4: Embed ALR parameters back to D dimensions
    # ------------------------------------------------------------------
    # Insert 0 at reference position (last component) for D-dimensional params
    loc = jnp.concatenate([alr_loc, jnp.array([0.0])], axis=0)
    cov_factor = jnp.concatenate(
        [alr_cov_factor, jnp.zeros((1, alr_cov_factor.shape[1]))], axis=0
    )
    cov_diag = jnp.concatenate([alr_cov_diag, jnp.array([0.0])], axis=0)

    # ------------------------------------------------------------------
    # Step 5: Compute mean probabilities on the simplex via inverse ALR
    # ------------------------------------------------------------------
    mean_probabilities = _inverse_alr(alr_loc, reference_index=None)

    # ------------------------------------------------------------------
    # Step 6: Create distribution objects and package results
    # ------------------------------------------------------------------
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
        # Backward compatibility
        results["base_distribution"] = distribution.base_dist
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
    batch_size: int,
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Fit Logistic-Normal distribution for mixture models.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape (n_posterior_samples, n_components, n_genes)
        Posterior samples of the Dirichlet concentration parameter per
        component.
    n_components : int
        Number of mixture components.
    rng_key : random.PRNGKey
        JAX PRNG key.
    n_samples_dirichlet : int
        Number of Dirichlet draws per posterior sample.
    rank : Optional[int]
        Rank of the low-rank covariance approximation.
    distribution_class : type
        SoftmaxNormal or LowRankLogisticNormal.
    remove_mean : bool
        Whether to centre ALR samples before fitting.
    batch_size : int
        Posterior samples processed per batched Dirichlet call.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Fitted Logistic-Normal parameters and distribution object(s) for
        each component.
    """
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
        n_batches = (n_posterior_samples + batch_size - 1) // batch_size
        print(f"Using rank {rank} for low-rank covariance approximation")
        print(
            f"Generating {n_samples_dirichlet} Dirichlet sample(s) for each "
            f"of {n_posterior_samples} posterior samples and {n_components} "
            f"components (batch_size={batch_size}, {n_batches} batches/comp)"
        )

    # Storage for per-component fitted parameters
    locs = []
    cov_factors = []
    cov_diags = []
    mean_probs = []
    distributions = []
    distributions_alr = []

    # ------------------------------------------------------------------
    # Fit one Logistic-Normal per component
    # ------------------------------------------------------------------
    for c in range(n_components):
        if verbose:
            print(
                f"\nFitting Logistic-Normal for component "
                f"{c + 1}/{n_components}"
            )

        # --- Step 1: Batched Dirichlet sampling for this component --------
        # Slice out concentrations for component c: (n_posterior, n_genes)
        r_component = r_samples[:, c, :]

        # Use a per-component sub-key for reproducibility
        key_c = random.fold_in(rng_key, c)

        all_simplex = _batched_dirichlet_sample(
            r_component,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=key_c,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Move to log-simplex space
        all_log_samples = jnp.log(all_simplex)

        if verbose:
            print(
                f"  Collected {all_log_samples.shape[0]} log-Dirichlet samples"
            )

        # --- Step 2: ALR transformation -----------------------------------
        alr_samples = _apply_alr_transform(
            all_log_samples, reference_index=None
        )  # (n_total, n_genes - 1)

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
                print("  Removed grand mean (focusing on co-variation)")
        else:
            alr_samples_centered = alr_samples
            alr_grand_mean = None

        # --- Step 3: Fit low-rank MVN in ALR space -------------------------
        alr_loc, alr_cov_factor, alr_cov_diag = _fit_low_rank_mvn(
            alr_samples_centered, rank=rank, verbose=False
        )

        # Step 3b: Add back grand mean if removed
        if remove_mean and alr_grand_mean is not None:
            alr_loc = alr_loc + alr_grand_mean.squeeze()

        # --- Step 4: Embed ALR parameters back to D dimensions -------------
        loc = jnp.concatenate([alr_loc, jnp.array([0.0])], axis=0)
        cov_factor = jnp.concatenate(
            [alr_cov_factor, jnp.zeros((1, alr_cov_factor.shape[1]))],
            axis=0,
        )
        cov_diag = jnp.concatenate(
            [alr_cov_diag, jnp.array([0.0])], axis=0
        )

        locs.append(loc)
        cov_factors.append(cov_factor)
        cov_diags.append(cov_diag)

        # --- Step 5: Mean probabilities on the simplex ---------------------
        mean_prob = _inverse_alr(alr_loc, reference_index=None)
        mean_probs.append(mean_prob)

        # --- Step 6: Create distribution for this component ----------------
        distribution = DISTRIBUTION_FACTORY[distribution_class](
            loc, cov_factor, cov_diag
        )

        if distribution_class == SoftmaxNormal:
            distributions.append(distribution)
        else:  # LowRankLogisticNormal
            distributions_alr.append(distribution)

    # ------------------------------------------------------------------
    # Stack per-component results
    # ------------------------------------------------------------------
    results = {
        "loc": jnp.stack(locs, axis=0),  # (K, D)
        "cov_factor": jnp.stack(cov_factors, axis=0),  # (K, D, rank)
        "cov_diag": jnp.stack(cov_diags, axis=0),  # (K, D)
        "mean_probabilities": jnp.stack(mean_probs, axis=0),  # (K, D)
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


def _fit_low_rank_mvn_core(
    samples: jnp.ndarray,
    rank: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure-JAX low-rank MVN fit via truncated SVD.

    This function contains **no** Python side-effects (no prints, no tqdm),
    making it safe to use inside ``jax.jit`` or ``jax.vmap``.  All diagnostics
    are returned as arrays so that a caller can inspect them if needed.

    Parameters
    ----------
    samples : jnp.ndarray, shape (n_samples, n_features)
        Centred (or un-centred) data matrix.  The mean is computed internally.
    rank : int
        Target rank for the low-rank covariance factor.  Clamped to
        ``min(rank, min(n_samples, n_features))``.

    Returns
    -------
    loc : jnp.ndarray, shape (n_features,)
        Sample mean.
    cov_factor : jnp.ndarray, shape (n_features, effective_rank)
        Low-rank covariance factor W such that
        ``Sigma ≈ W @ W.T + diag(cov_diag)``.
    cov_diag : jnp.ndarray, shape (n_features,)
        Diagonal residual variance + 1e-4 stability constant.
    eigenvalues : jnp.ndarray, shape (min(n_samples, n_features),)
        All eigenvalues of the sample covariance (for diagnostics).

    Notes
    -----
    This helper is deliberately kept free of Python control-flow on JAX
    values so that it can later be wrapped in ``jax.vmap`` across mixture
    components for fully-parallel GPU execution.
    """
    n_samples, n_features = samples.shape

    # Compute mean
    loc = jnp.mean(samples, axis=0)

    # Centre the data
    centered = samples - loc

    # SVD on the centred data matrix (memory-efficient when n_samples << D)
    # X = U @ diag(S) @ Vt  →  Cov = V @ diag(S²/(n-1)) @ V.T
    _U, singular_values, Vt = jnp.linalg.svd(
        centered, full_matrices=False
    )
    # singular_values: (min(n_samples, n_features),)
    # Vt: (min(n_samples, n_features), n_features)

    # Convert singular values to eigenvalues of the covariance
    eigenvalues = (singular_values ** 2) / (n_samples - 1)

    # Eigenvectors as columns
    eigenvectors = Vt.T  # (n_features, min(n_samples, n_features))

    # Sort in descending order (SVD already descending, but be explicit)
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp rank to available components
    effective_rank = min(rank, len(eigenvalues))

    # Top-k eigenvalues and eigenvectors for the low-rank factor
    top_k_eigenvalues = eigenvalues[:effective_rank]
    top_k_eigenvectors = eigenvectors[:, :effective_rank]

    # W = V_k @ diag(sqrt(lambda_k))   →   W @ W.T ≈ Sigma_top
    cov_factor = top_k_eigenvectors * jnp.sqrt(
        jnp.maximum(top_k_eigenvalues, 0.0)
    )

    # Residual variance: mean of remaining eigenvalues + stability constant
    residual_eigenvalues = eigenvalues[effective_rank:]
    diag_value = (
        jnp.mean(residual_eigenvalues)
        if len(residual_eigenvalues) > 0
        else 0.0
    )
    cov_diag = jnp.full(n_features, diag_value) + 1e-4

    return loc, cov_factor, cov_diag, eigenvalues


def _fit_low_rank_mvn(
    samples: jnp.ndarray,
    rank: int,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fit a low-rank multivariate normal distribution to samples using PCA.

    This is a thin wrapper around ``_fit_low_rank_mvn_core`` that adds
    validation, progress messages and diagnostic printing.

    Parameters
    ----------
    samples : jnp.ndarray, shape (n_samples, n_features)
        Samples to fit.
    rank : int
        Rank of the low-rank approximation.
    verbose : bool, default=False
        Whether to print progress and diagnostic messages.

    Returns
    -------
    loc : jnp.ndarray, shape (n_features,)
        Mean vector.
    cov_factor : jnp.ndarray, shape (n_features, effective_rank)
        Low-rank covariance factor W.
    cov_diag : jnp.ndarray, shape (n_features,)
        Diagonal component D.
    """
    n_samples, n_features = samples.shape

    # --- Validation (not safe inside jit/vmap, so kept in the wrapper) ---
    if n_samples < 2:
        raise ValueError(
            f"Need at least 2 samples to fit low-rank MVN, but got "
            f"{n_samples}. Make sure you have multiple posterior samples of r, "
            f"or increase n_samples_dirichlet to > 1."
        )

    if verbose:
        print(
            f"  Computing SVD of centered data ({n_samples} x {n_features})..."
        )

    # --- Delegate to the pure-JAX core ---
    loc, cov_factor, cov_diag, eigenvalues = _fit_low_rank_mvn_core(
        samples, rank
    )

    # --- Diagnostics (verbose only) ---
    effective_rank = cov_factor.shape[1]
    if verbose and effective_rank < rank:
        print(
            f"  Clamping rank from {rank} to {effective_rank} (max available)"
        )

    if verbose:
        top_k_eigenvalues = eigenvalues[:effective_rank]
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
                f"  First eigenvalue is {ratio:.1f}x larger than second "
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

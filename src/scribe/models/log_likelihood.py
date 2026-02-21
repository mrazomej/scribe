"""
Log likelihood computation functions for single-cell RNA sequencing models.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.scipy as jsp

# Import Pyro-related libraries
import numpyro.distributions as dist

# Import typing
from typing import Dict, Optional


def _validate_mixture_component_shapes(
    r: jnp.ndarray,
    probs: jnp.ndarray,
    n_components: int,
    context: str,
) -> None:
    """Validate component-axis alignment for mixture NB likelihood tensors.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion tensor after mixture reshaping. Expected trailing axis layout
        is ``(..., n_genes, n_components)``.
    probs : jnp.ndarray
        Success-probability tensor (`p_hat`) after capture adjustment. Expected
        trailing axis layout is ``(..., n_genes or 1, n_components)``.
    n_components : int
        Number of active mixture components implied by ``mixing_weights``.
    context : str
        Human-readable caller context to include in error messages.

    Raises
    ------
    ValueError
        Raised when tensors do not agree on the active component axis.

    Notes
    -----
    This explicit check provides a clear error when component pruning produced
    inconsistent posterior tensors (for example, `mixing_weights` pruned to
    ``K=3`` while canonical `p`/`r` stayed at ``K=4``), instead of surfacing
    a low-level JAX broadcasting failure.
    """
    if r.shape[-1] != n_components:
        raise ValueError(
            f"{context}: dispersion tensor has incompatible component axis. "
            f"Expected r.shape[-1] == {n_components}, got r.shape={tuple(r.shape)}."
        )

    if probs.shape[-1] != n_components:
        hint = ""
        if probs.ndim >= 2 and probs.shape[-2] == n_components:
            hint = (
                " Detected component-like size on probs.shape[-2], suggesting "
                "a swapped or stale component axis after mixture pruning."
            )
        raise ValueError(
            f"{context}: probability tensor has incompatible component axis. "
            f"Expected probs.shape[-1] == {n_components}, got probs.shape="
            f"{tuple(probs.shape)}.{hint}"
        )

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial (NBDM) likelihood
# ------------------------------------------------------------------------------


def nbdm_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for NBDM model using plates.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': success probability parameter (scalar)
            - 'r': dispersion parameters for each gene (vector of length
              n_genes)
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities using the NBDM model (default)
            - 'gene': returns log probabilities using independent NB per gene
    r_floor : float, default=1e-6
        Minimum value for the dispersion parameter ``r`` after casting to
        ``dtype``.  Posterior samples drawn from a wide variational guide can
        occasionally underflow to zero or become negative in the constrained
        space, causing ``lgamma(r)`` to return NaN and discarding the entire
        sample.  Clamping ``r`` to this small positive value neutralises those
        degenerate samples at negligible cost to the likelihood.  Set to
        ``0.0`` to disable the floor entirely.
    p_floor : float, default=1e-6
        Epsilon clipped away from both 0 and 1 for the success probability
        ``p``.  In hierarchical parameterisations ``p`` is gene-specific and
        derived from ``phi_g = exp(log_phi_g)``; if ``log_phi_g`` underflows
        to zero in float32, ``p_g = 1/(1+0) = 1.0`` exactly, causing
        ``r * log(1-p)`` to evaluate as ``r * log(0) = NaN`` (without
        ``r_floor``) or ``-inf`` (with it).  Setting a small cap keeps ``p``
        strictly inside ``(p_floor, 1 - p_floor)``.  Set to ``0.0`` to
        disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    # Check return_by
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary - handle both old and new formats
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Clip p away from 0 and 1 to prevent log(0) or log(1-1) = NaN/-inf
    if p_floor > 0.0:
        p = jnp.clip(p, p_floor, 1.0 - p_floor)

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    if return_by == "cell":
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
            # Return per-cell log probabilities
            return base_dist.log_prob(counts)

        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]

            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                base_dist.log_prob(batch_counts)
            )

        return cell_log_probs

    else:  # return_by == 'gene'
        # For per-gene likelihood, use independent negative binomials
        if batch_size is None:
            # Compute log probabilities for all genes at once
            return jnp.sum(
                dist.NegativeBinomialProbs(r, p).log_prob(counts),
                axis=0,  # Sum over cells
            )

        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]

            # Create NB distribution and compute log probs for each gene
            nb_dist = dist.NegativeBinomialProbs(r, p)
            # Shape of batch_counts is (batch_size, n_genes)
            # We want log probs for each gene summed over the batch
            # Shape: (batch_size, n_genes)
            batch_log_probs = nb_dist.log_prob(batch_counts)

            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)

        return gene_log_probs


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial (ZINB) likelihood
# ------------------------------------------------------------------------------


def zinb_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for Zero-Inflated Negative Binomial model.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene
            - 'gate': dropout probability parameter
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the success probability ``p``, clipping it to the
        open interval ``(p_floor, 1 - p_floor)``.  Prevents NaN from
        gene-specific ``p`` values hitting exactly 0 or 1 in float32.
        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Clip p away from 0 and 1 to prevent log(0) NaN
    if p_floor > 0.0:
        p = jnp.clip(p, p_floor, 1.0 - p_floor)
    gate = jnp.squeeze(params["gate"]).astype(dtype)

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    if return_by == "cell":
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Return per-cell log probabilities
            return zinb.log_prob(counts)

        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]

            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                zinb.log_prob(batch_counts)
            )

        return cell_log_probs

    else:  # return_by == 'gene'
        # For per-gene likelihood
        if batch_size is None:
            # Create base distribution and compute all at once
            base_dist = dist.NegativeBinomialProbs(r, p)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            return jnp.sum(zinb.log_prob(counts), axis=0)

        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]

            # Create distributions and compute log probs
            base_dist = dist.NegativeBinomialProbs(r, p)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            # Shape: (batch_size, n_genes)
            batch_log_probs = zinb.log_prob(batch_counts)

            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)

        return gene_log_probs


# ------------------------------------------------------------------------------
# Negative Binomial with Variable Capture Probability (NBVC) likelihood
# ------------------------------------------------------------------------------


def nbvcp_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for Negative Binomial with Variable Capture
    Probability.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene
            - 'p_capture': cell-specific capture probabilities
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the *effective* probability ``p_hat`` (after
        combining ``p`` and ``p_capture``), clipping it to
        ``(p_floor, 1 - p_floor)``.  Two degenerate cases can make ``p_hat``
        hit 0 or 1 exactly in float32:

        - ``phi_g → 0`` (hierarchical models) → ``p_g = 1/(1+0) = 1.0``
          → ``p_hat = 1.0`` → ``r * log(1 - 1.0) = NaN/−∞``.
        - ``phi_capture → ∞`` → ``p_capture = 0``
          → ``p_hat = 0`` → ``0 * log(0) = NaN`` for zero counts.

        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    # Check return_by
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Handle both p_capture and phi_capture (odds_ratio parameterization)
    if "phi_capture" in params:
        # Convert phi_capture (odds ratio) to p_capture: p = 1 / (1 + phi)
        phi_capture = jnp.squeeze(params["phi_capture"]).astype(dtype)
        p_capture = 1.0 / (1.0 + phi_capture)
    else:
        p_capture = jnp.squeeze(params["p_capture"]).astype(dtype)

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    if return_by == "cell":
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Reshape p_capture to [n_cells, 1] for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute p_hat for all cells
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)
            # Return per-cell log probabilities
            return (
                dist.NegativeBinomialProbs(r, p_hat)
                .to_event(1)
                .log_prob(counts)
            )

        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]

            # Reshape batch_p_capture for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute p_hat for batch
            batch_p_hat = p * batch_p_capture / (1 - p * (1 - batch_p_capture))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                batch_p_hat = jnp.clip(batch_p_hat, p_floor, 1.0 - p_floor)
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                dist.NegativeBinomialProbs(r, batch_p_hat)
                .to_event(1)
                .log_prob(batch_counts)
            )

        return cell_log_probs

    else:  # return_by == 'gene'
        # For per-gene likelihood
        if batch_size is None:
            # Reshape p_capture to [n_cells, 1] for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute p_hat for all cells
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)
            # Compute log probabilities for each gene
            return jnp.sum(
                dist.NegativeBinomialProbs(r, p_hat).log_prob(counts),
                axis=0,  # Sum over cells
            )

        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]

            # Reshape batch_p_capture for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute p_hat for batch
            batch_p_hat = p * batch_p_capture / (1 - p * (1 - batch_p_capture))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                batch_p_hat = jnp.clip(batch_p_hat, p_floor, 1.0 - p_floor)
            # Compute log probabilities for batch
            batch_log_probs = dist.NegativeBinomialProbs(
                r, batch_p_hat
            ).log_prob(batch_counts)

            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)

        return gene_log_probs


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVC)
# ------------------------------------------------------------------------------


def zinbvcp_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for Zero-Inflated Negative Binomial with Variable
    Capture Probability.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene
            - 'p_capture': cell-specific capture probabilities
            - 'gate': dropout probability parameter
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the *effective* probability ``p_hat`` (after
        combining ``p`` and ``p_capture``), clipping it to
        ``(p_floor, 1 - p_floor)``.  Two degenerate cases can make ``p_hat``
        hit 0 or 1 exactly in float32:

        - ``phi_g → 0`` (hierarchical models) → ``p_g = 1/(1+0) = 1.0``
          → ``p_hat = 1.0`` → ``r * log(1 - 1.0) = NaN/−∞``.
        - ``phi_capture → ∞`` → ``p_capture = 0``
          → ``p_hat = 0`` → ``0 * log(0) = NaN`` for zero counts.

        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    # Check return_by
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Handle both p_capture and phi_capture (odds_ratio parameterization)
    if "phi_capture" in params:
        # Convert phi_capture (odds ratio) to p_capture: p = 1 / (1 + phi)
        phi_capture = jnp.squeeze(params["phi_capture"]).astype(dtype)
        p_capture = 1.0 / (1.0 + phi_capture)
    else:
        p_capture = jnp.squeeze(params["p_capture"]).astype(dtype)
    gate = jnp.squeeze(params["gate"]).astype(dtype)

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    if return_by == "cell":
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Reshape capture probabilities for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute adjusted success probability
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Return per-cell log probabilities
            return zinb.log_prob(counts)

        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]

            # Reshape capture probabilities for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute adjusted success probability
            batch_p_hat = p * batch_p_capture / (1 - p * (1 - batch_p_capture))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                batch_p_hat = jnp.clip(batch_p_hat, p_floor, 1.0 - p_floor)
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, batch_p_hat)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                zinb.log_prob(batch_counts)
            )

        return cell_log_probs

    else:  # return_by == 'gene'
        # For per-gene likelihood
        if batch_size is None:
            # Reshape capture probabilities for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute adjusted success probability
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)
            # Create base distribution and compute all at once
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            return jnp.sum(zinb.log_prob(counts), axis=0)

        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)

            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]

            # Reshape capture probabilities for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute adjusted success probability
            batch_p_hat = p * batch_p_capture / (1 - p * (1 - batch_p_capture))
            # Clip effective probability away from 0 and 1 to prevent NaN
            if p_floor > 0.0:
                batch_p_hat = jnp.clip(batch_p_hat, p_floor, 1.0 - p_floor)
            # Create distributions and compute log probs
            base_dist = dist.NegativeBinomialProbs(r, batch_p_hat)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            batch_log_probs = zinb.log_prob(batch_counts)

            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)

        return gene_log_probs


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Mixture Models
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Negative Binomial Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------


def nbdm_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for NBDM mixture model using independent negative
    binomials.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene and component
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component. If
        False, returns the log probability of the mixture.
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights to probabilities. Must be one of:
            - 'multiplicative': applies as p^weight (weight * log(p) in log
              space)
            - 'additive': applies as exp(weight)*p (weight + log(p) in log space)
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the success probability ``p``, clipping it to the
        open interval ``(p_floor, 1 - p_floor)``.  In hierarchical
        parameterisations ``p`` is gene-specific; if ``phi_g`` underflows to
        zero in float32, ``p_g = 1.0`` exactly, causing ``r * log(0) = NaN``.
        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells,
              n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes,
              n_components)
    """
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in [
        "multiplicative",
        "additive",
    ]:
        raise ValueError(
            "weight_type must be one of " "['multiplicative', 'additive']"
        )

    # Extract parameters
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)  # shape (n_components, n_genes)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Clip p away from 0 and 1 to prevent log(0) NaN (relevant for hierarchical p)
    if p_floor > 0.0:
        p = jnp.clip(p, p_floor, 1.0 - p_floor)
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Broadcasting layout: (n_cells, n_genes, n_components)
    counts = jnp.expand_dims(counts, axis=-1)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)

    # Handle p: scalar, (n_components,), or (n_components, n_genes)
    p_is_gene_specific = p.ndim == 2 and p.shape[0] == n_components and p.shape[1] > 1
    p_is_component_specific = (
        not p_is_gene_specific and p.ndim >= 1 and p.shape[0] == n_components
    )

    if p_is_gene_specific:
        p = jnp.expand_dims(jnp.transpose(p), axis=0)
    elif p_is_component_specific:
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        p = jnp.array(p)[None, None, None]

    nb_dist = dist.NegativeBinomialProbs(r, p)

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == "cell" else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == "cell":
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))

            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=1) + jnp.log(
                mixing_weights
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)

                # Compute log probs for batch
                # Shape: (batch_size, n_components, n_genes)
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))

                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            # Shape: (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))

            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_genes, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)

                # Compute log probs for batch
                # Shape: (batch_size, n_components, n_genes)
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))

                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)

            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T

    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------


def zinb_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for ZINB mixture model.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene and component
            - 'gate': dropout probabilities for each gene and component
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component.
        If False, returns the log probability of the mixture.
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights to probabilities. Must be one of:
            - 'multiplicative': applies as p^weight (weight * log(p) in log space)
            - 'additive': applies as exp(weight)*p (weight + log(p) in log space)
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the success probability ``p``, clipping it to the
        open interval ``(p_floor, 1 - p_floor)``.  In hierarchical
        parameterisations ``p`` is gene-specific; if ``phi_g`` underflows to
        zero in float32, ``p_g = 1.0`` exactly, causing ``r * log(0) = NaN``.
        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells, n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes, n_components)
    """
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in [
        "multiplicative",
        "additive",
    ]:
        raise ValueError(
            "weight_type must be one of " "['multiplicative', 'additive']"
        )

    # Extract parameters
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)  # shape (n_components, n_genes)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Clip p away from 0 and 1 to prevent log(0) NaN (relevant for hierarchical p)
    if p_floor > 0.0:
        p = jnp.clip(p, p_floor, 1.0 - p_floor)
    gate = jnp.asarray(params["gate"]).astype(dtype)
    if gate.ndim < 2:
        gate = gate[jnp.newaxis, :]  # (n_genes,) -> (1, n_genes)
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Broadcasting layout: (n_cells, n_genes, n_components)
    counts = jnp.expand_dims(counts, axis=-1)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    gate = jnp.expand_dims(jnp.transpose(gate), axis=0)

    # Handle p: scalar, (n_components,), or (n_components, n_genes)
    p_is_gene_specific = p.ndim == 2 and p.shape[0] == n_components and p.shape[1] > 1
    p_is_component_specific = (
        not p_is_gene_specific and p.ndim >= 1 and p.shape[0] == n_components
    )

    if p_is_gene_specific:
        p = jnp.expand_dims(jnp.transpose(p), axis=0)
    elif p_is_component_specific:
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        p = jnp.array(p)[None, None, None]

    # Create base NB distribution vectorized over cells, genes, components
    # r: (1, n_genes, n_components)
    # p: (1, 1, 1) or scalar
    # counts: (n_cells, n_genes, 1)
    # This will broadcast to: (n_cells, n_genes, n_components)
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Create zero-inflated distribution for each component
    # This will broadcast to: (n_cells, n_genes, n_components)
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == "cell" else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == "cell":
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))

            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=1) + jnp.log(
                mixing_weights
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)

                # Compute log probs for batch
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))

                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))

            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_genes, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)

                # Compute log probs for batch
                # Shape: (batch_size, n_components, n_genes)
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))

                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)

            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T

    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)


# ------------------------------------------------------------------------------
# Negative Binomial Mixture Model with Capture Probabilities
# ------------------------------------------------------------------------------


def nbvcp_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for NBVCP mixture model.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': base success probability parameter (shared or component-specific)
            - 'r': dispersion parameters for each gene and component
            - 'p_capture': cell-specific capture probabilities
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component. If
        False, returns the log probability of the mixture.
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights to probabilities. Must be one of:
            - 'multiplicative': applies as p^weight (weight * log(p) in log
              space)
            - 'additive': applies as exp(weight)*p (weight + log(p) in log space)
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the *effective* probability ``p_hat`` (after
        combining ``p`` and ``p_capture``), clipping it to
        ``(p_floor, 1 - p_floor)``.  Two degenerate cases can make ``p_hat``
        hit 0 or 1 exactly in float32:

        - ``phi_g → 0`` (hierarchical models) → ``p_g = 1/(1+0) = 1.0``
          → ``p_hat = 1.0`` → ``r * log(1 - 1.0) = NaN/−∞``.
        - ``phi_capture → ∞`` → ``p_capture = 0``
          → ``p_hat = 0`` → ``0 * log(0) = NaN`` for zero counts.

        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells,
              n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes,
              n_components)
    """
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in [
        "multiplicative",
        "additive",
    ]:
        raise ValueError(
            "weight_type must be one of " "['multiplicative', 'additive']"
        )

    # Extract parameters
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)  # shape (n_components, n_genes)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    # Handle both p_capture and phi_capture (odds_ratio parameterization)
    if "phi_capture" in params:
        # Convert phi_capture (odds ratio) to p_capture: p = 1 / (1 + phi)
        phi_capture = jnp.squeeze(params["phi_capture"]).astype(dtype)
        p_capture = 1.0 / (1.0 + phi_capture)  # shape (n_cells,)
    else:
        p_capture = jnp.squeeze(params["p_capture"]).astype(
            dtype
        )  # shape (n_cells,)
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Broadcasting layout used throughout: (n_cells, n_genes, n_components)
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture = jnp.expand_dims(p_capture, axis=(-1, -2))

    # Handle p parameter: scalar, (n_components,), or (n_components, n_genes)
    p_is_gene_specific = p.ndim == 2 and p.shape[0] == n_components and p.shape[1] > 1
    p_is_component_specific = (
        not p_is_gene_specific and p.ndim >= 1 and p.shape[0] == n_components
    )

    if p_is_gene_specific:
        # Hierarchical: (n_components, n_genes) -> (1, n_genes, n_components)
        p = jnp.expand_dims(jnp.transpose(p), axis=0)
    elif p_is_component_specific:
        # Component-only: (n_components,) -> (1, 1, n_components)
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        # Shared scalar: () -> (1, 1, 1)
        p = jnp.array(p)[None, None, None]

    # Compute effective probability for each cell
    # p_hat = p * p_capture / (1 - p * (1 - p_capture))
    # Broadcasts to (n_cells, n_genes, n_components) or (n_cells, 1, n_components)
    p_hat = p * p_capture / (1 - p * (1 - p_capture))
    _validate_mixture_component_shapes(
        r=r,
        probs=p_hat,
        n_components=n_components,
        context="nbvcp_mixture_log_likelihood",
    )
    # Clip effective probability away from 0 and 1 to prevent NaN
    if p_floor > 0.0:
        p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == "cell" else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == "cell":
        if batch_size is None:
            # Create base NB distribution vectorized over cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, n_components) or (n_cells, 1, 1)
            # counts: (n_cells, n_genes, 1)
            # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)

            # Compute log probs for all cells at once
            # This gives (n_cells, n_genes, n_components)
            gene_log_probs = nb_dist.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))

            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=1) + jnp.log(
                mixing_weights
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)

                # Compute p_hat for this batch
                batch_p_capture = p_capture[start_idx:end_idx]
                batch_p_hat = (
                    p * batch_p_capture / (1 - p * (1 - batch_p_capture))
                )
                _validate_mixture_component_shapes(
                    r=r,
                    probs=batch_p_hat,
                    n_components=n_components,
                    context="nbvcp_mixture_log_likelihood (batched)",
                )

                # Create base NB distribution for batch
                nb_dist = dist.NegativeBinomialProbs(r, batch_p_hat)

                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))

                # Sum over genes (axis=1) to get (batch_size, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )

    else:  # return_by == 'gene'
        if batch_size is None:
            # Create base NB distribution vectorized over cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, n_components) or (n_cells, 1, 1)
            # counts: (n_cells, n_genes, 1)
            # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Compute log probs for each gene
            gene_log_probs = nb_dist.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))

            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)

                # Compute p_hat for this batch
                batch_p_capture = p_capture[start_idx:end_idx]
                batch_p_hat = (
                    p * batch_p_capture / (1 - p * (1 - batch_p_capture))
                )
                _validate_mixture_component_shapes(
                    r=r,
                    probs=batch_p_hat,
                    n_components=n_components,
                    context="nbvcp_mixture_log_likelihood (batched)",
                )

                # Create base NB distribution for batch
                nb_dist = dist.NegativeBinomialProbs(r, batch_p_hat)

                # Compute log probs for batch
                # Shape: (batch_size, n_genes, n_components)
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))

                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)

            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T

    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model with Capture Probabilities
# ------------------------------------------------------------------------------


def zinbvcp_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    r_floor: float = 1e-6,
    p_floor: float = 1e-6,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """
    Compute log likelihood for ZINBVCP mixture model.

    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene and component
            - 'p_capture': cell-specific capture probabilities
            - 'gate': dropout probabilities for each gene and component
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component. If
        False, returns the log probability of the mixture.
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights. Must be one of:
            - 'multiplicative': multiply log probabilities by weights
            - 'additive': add weights to log probabilities
    r_floor : float, default=1e-6
        Minimum value clamped onto the dispersion parameter ``r`` after
        casting to ``dtype``.  Prevents NaN log-likelihoods from degenerate
        posterior samples where ``r`` underflows to zero.  Set to ``0.0`` to
        disable.
    p_floor : float, default=1e-6
        Epsilon applied to the *effective* probability ``p_hat`` (after
        combining ``p`` and ``p_capture``), clipping it to
        ``(p_floor, 1 - p_floor)``.  Two degenerate cases can make ``p_hat``
        hit 0 or 1 exactly in float32:

        - ``phi_g → 0`` (hierarchical models) → ``p_g = 1/(1+0) = 1.0``
          → ``p_hat = 1.0`` → ``r * log(1 - 1.0) = NaN/−∞``.
        - ``phi_capture → ∞`` → ``p_capture = 0``
          → ``p_hat = 0`` → ``0 * log(0) = NaN`` for zero counts.

        Set to ``0.0`` to disable.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells,
              n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes,
              n_components)
    """
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ["cell", "gene"]:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in [
        "multiplicative",
        "additive",
    ]:
        raise ValueError(
            "weight_type must be one of " "['multiplicative', 'additive']"
        )

    # Extract parameters
    p = jnp.squeeze(params["p"]).astype(dtype)
    r = jnp.squeeze(params["r"]).astype(dtype)  # shape (n_components, n_genes)
    # Guard against degenerate posterior samples where r underflows to 0
    if r_floor > 0.0:
        r = jnp.maximum(r, r_floor)
    if "phi_capture" in params:
        phi_capture = jnp.squeeze(params["phi_capture"]).astype(dtype)
        p_capture = 1.0 / (1.0 + phi_capture)
    else:
        p_capture = jnp.squeeze(params["p_capture"]).astype(dtype)
    gate = jnp.asarray(params["gate"]).astype(dtype)
    if gate.ndim < 2:
        gate = gate[jnp.newaxis, :]
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)

    # Broadcasting layout: (n_cells, n_genes, n_components)
    counts = jnp.expand_dims(counts, axis=-1)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    gate = jnp.expand_dims(jnp.transpose(gate), axis=0)
    p_capture = jnp.expand_dims(p_capture, axis=(-1, -2))

    # Handle p: scalar, (n_components,), or (n_components, n_genes)
    p_is_gene_specific = p.ndim == 2 and p.shape[0] == n_components and p.shape[1] > 1
    p_is_component_specific = (
        not p_is_gene_specific and p.ndim >= 1 and p.shape[0] == n_components
    )

    if p_is_gene_specific:
        p = jnp.expand_dims(jnp.transpose(p), axis=0)
    elif p_is_component_specific:
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        p = jnp.array(p)[None, None, None]

    # p_hat broadcasts to (n_cells, n_genes, n_components) or (n_cells, 1, n_components)
    p_hat = p * p_capture / (1 - p * (1 - p_capture))
    _validate_mixture_component_shapes(
        r=r,
        probs=p_hat,
        n_components=n_components,
        context="zinbvcp_mixture_log_likelihood",
    )
    # Clip effective probability away from 0 and 1 to prevent NaN
    if p_floor > 0.0:
        p_hat = jnp.clip(p_hat, p_floor, 1.0 - p_floor)

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == "cell" else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == "cell":
        if batch_size is None:
            # Create base NB distribution vectorized over
            # cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, n_components) or (n_cells, 1, 1)
            # counts: (n_cells, n_genes, 1)
            # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution for each component
            zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))

            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=1) + jnp.log(
                mixing_weights
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over
                # cells, genes, components
                # r: (1, n_genes, n_components)
                # p_hat: (n_cells, 1, n_components) or (n_cells, 1, 1)
                # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (batch_size, n_genes, n_components)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx]
                )
                # Create zero-inflated distribution for each component
                zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
                # Compute log probs for batch
                # Shape: (batch_size, n_genes, n_components)
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))

                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )

    else:  # return_by == 'gene'
        if batch_size is None:
            # Create base NB distribution vectorized over
            # cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, n_components) or (n_cells, 1, 1)
            # counts: (n_cells, n_genes, 1)
            # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution for each component
            zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)

            # Apply weights based on weight_type
            if weight_type == "multiplicative":
                gene_log_probs *= weights
            elif weight_type == "additive":
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))

            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))

            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over
                # cells, genes, components
                # r: (1, n_genes, n_components)
                # p_hat: (n_cells, 1, n_components) or (n_cells, 1, 1)
                # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (batch_size, n_genes, n_components)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx]
                )
                # Create zero-inflated distribution for each component
                zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
                # Compute log probs for batch
                # Shape: (batch_size, n_genes, n_components)
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == "multiplicative":
                    batch_log_probs *= weights
                elif weight_type == "additive":
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))

                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)

            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T

    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)

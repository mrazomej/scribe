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

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial (NBDM) likelihood
# ------------------------------------------------------------------------------


def nbdm_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = "cell",
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
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Determine if p is component-specific or shared
    p_is_component_specific = len(p.shape) > 0 and p.shape[0] == n_components

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)

    # Handle p parameter based on whether it's component-specific or shared
    if p_is_component_specific:
        # Component-specific p: shape (n_components,) -> (1, 1, n_components)
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        # Shared p: scalar -> (1, 1, 1) for broadcasting
        p = jnp.array(p)[None, None, None]

    # Create base NB distribution vectorized over cells, components, genes
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
    gate = jnp.asarray(params["gate"]).astype(dtype)
    if gate.ndim < 2:
        gate = gate[jnp.newaxis, :]  # (n_genes,) -> (1, n_genes)
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Determine if p is component-specific or shared
    p_is_component_specific = len(p.shape) > 0 and p.shape[0] == n_components

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # gate: (K, n_genes) -> (1, n_genes, K) where K = n_components or 1
    gate = jnp.expand_dims(jnp.transpose(gate), axis=0)

    # Handle p parameter based on whether it's component-specific or shared
    if p_is_component_specific:
        # Component-specific p: shape (n_components,) -> (1, 1, n_components)
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        # Shared p: scalar -> (1, 1, 1) for broadcasting
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

    # Determine if p is component-specific or shared
    p_is_component_specific = len(p.shape) > 0 and p.shape[0] == n_components

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture = jnp.expand_dims(p_capture, axis=(-1, -2))

    # Handle p parameter based on whether it's component-specific or shared
    if p_is_component_specific:
        # Component-specific p: shape (n_components,) -> (1, 1, n_components)
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        # Shared p: scalar -> (1, 1, 1) for broadcasting
        p = jnp.array(p)[None, None, None]

    # Compute effective probability for each cell using the correct formula
    # p_hat = p * p_capture / (1 - p * (1 - p_capture))
    # This will broadcast to shape (n_cells, 1, n_components) or (n_cells, 1, 1)
    p_hat = p * p_capture / (1 - p * (1 - p_capture))

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
    # Handle both p_capture and phi_capture (odds_ratio parameterization)
    if "phi_capture" in params:
        # Convert phi_capture (odds ratio) to p_capture: p = 1 / (1 + phi)
        phi_capture = jnp.squeeze(params["phi_capture"]).astype(dtype)
        p_capture = 1.0 / (1.0 + phi_capture)  # shape (n_cells,)
    else:
        p_capture = jnp.squeeze(params["p_capture"]).astype(
            dtype
        )  # shape (n_cells,)
    gate = jnp.asarray(params["gate"]).astype(dtype)
    if gate.ndim < 2:
        gate = gate[jnp.newaxis, :]  # (n_genes,) -> (1, n_genes)
    mixing_weights = jnp.squeeze(params["mixing_weights"]).astype(dtype)
    n_components = mixing_weights.shape[0]

    # Determine if p is component-specific or shared
    p_is_component_specific = len(p.shape) > 0 and p.shape[0] == n_components

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # gate: (K, n_genes) -> (1, n_genes, K) where K = n_components or 1
    # When K=1 (shared gate), the trailing dim broadcasts with n_components.
    gate = jnp.expand_dims(jnp.transpose(gate), axis=0)
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture = jnp.expand_dims(p_capture, axis=(-1, -2))

    # Handle p parameter based on whether it's component-specific or shared
    if p_is_component_specific:
        # Component-specific p: shape (n_components,) -> (1, 1, n_components)
        p = jnp.expand_dims(p, axis=(0, 1))
    else:
        # Shared p: scalar -> (1, 1, 1) for broadcasting
        p = jnp.array(p)[None, None, None]

    # Compute effective probability for each cell using the correct formula
    # p_hat = p * p_capture / (1 - p * (1 - p_capture))
    # This will broadcast to shape (n_cells, 1, n_components) or (n_cells, 1, 1)
    p_hat = p * p_capture / (1 - p * (1 - p_capture))

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

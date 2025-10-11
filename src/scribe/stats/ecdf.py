"""ECDF (Empirical Cumulative Distribution Function) computation functions."""

import numpy as np
import jax.numpy as jnp
import jax

# ==============================================================================
# ECDF functions
# ==============================================================================


def compute_ecdf_percentiles(
    samples, percentiles=[5, 25, 50, 75, 95], sample_axis=0
):
    """
    Compute percentiles of ECDF values across multiple samples of integers.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1, containing raw data samples of positive
        integers
    percentiles : list-like, optional
        List of percentiles to compute (default: [5, 25, 50, 75, 95])
    sample_axis : int, optional
        Axis containing samples (default: 0)

    Returns
    -------
    bin_edges : array
        Array of integer points at which ECDFs were evaluated (from min to max)
    ecdf_percentiles : array
        Array of shape (len(percentiles), len(bin_edges)) containing the
        percentiles of ECDF values at each integer point
    """
    # Convert to JAX arrays
    samples = jnp.asarray(samples)

    # Ensure samples are in the right orientation (n_samples, n_points)
    if sample_axis == 1:
        samples = samples.T

    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())

    # Create evaluation points for each integer from min to max
    bin_edges = jnp.arange(global_min, global_max + 1)

    # Define function to compute ECDF for a single sample
    def compute_ecdf(sample):
        # For each integer value, count proportion of sample values <= that value
        return jax.vmap(lambda x: jnp.mean(sample <= x))(bin_edges)

    # Use vmap to apply compute_ecdf across all samples
    all_ecdfs = jax.vmap(compute_ecdf)(samples)

    # Convert to numpy for percentile calculations
    all_ecdfs_np = np.array(all_ecdfs)
    bin_edges_np = np.array(bin_edges)

    # Compute percentiles across samples for each evaluation point
    ecdf_percentiles = np.percentile(all_ecdfs_np, percentiles, axis=0)

    return bin_edges_np, ecdf_percentiles


# ==============================================================================
# Credible regions functions
# ==============================================================================


def compute_ecdf_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    sample_axis=0,
    batch_size=1000,
    max_bin=None,
):
    """
    Compute credible regions of ECDF values across multiple samples.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1, containing raw data samples
    credible_regions : list-like, optional
        List of credible region percentages to compute (default: [95, 68, 50])
        For example, 95 will compute the 2.5 and 97.5 percentiles
    sample_axis : int, optional
        Axis containing samples (default: 0)
    batch_size : int, optional
        Number of samples to process in each batch (default: 1000)
    max_bin : int, optional
        Maximum value to include in ECDF evaluation (default: None)

    Returns
    -------
    dict
        Dictionary containing:
            - 'bin_edges': array of points at which ECDFs were evaluated
            - 'regions': nested dictionary where each key is the credible region
              percentage
            and values are dictionaries containing:
                - 'lower': lower bound of the credible region
                - 'upper': upper bound of the credible region
                - 'median': median (50th percentile)
    """
    # Convert to JAX array if not already
    samples = jnp.asarray(samples, dtype=jnp.float32)
    samples_2d = samples if sample_axis == 0 else samples.T

    # Find global min and max across all samples
    global_min = int(samples.min())

    # Define global max if max_bin is not None, else use global max
    global_max = (
        min(int(samples.max()), max_bin)
        if max_bin is not None
        else int(samples.max())
    )

    # Create evaluation points for each integer from min to max
    bin_edges = jnp.arange(global_min, global_max + 1)

    # Define function to compute ECDF for a single sample
    def compute_single_ecdf(sample):
        # For each evaluation point, count proportion of sample values <= that point
        return jax.vmap(lambda x: jnp.mean(sample <= x))(bin_edges)

    # Use vmap to compute ECDFs for all samples
    # Note: We could use jax.lax.map with batching here instead, but vmap is
    # already vectorized and efficient for this operation
    all_ecdfs = jax.vmap(compute_single_ecdf)(samples_2d)

    # Convert to numpy for percentile calculations
    all_ecdfs_np = np.array(all_ecdfs)
    bin_edges_np = np.array(bin_edges)

    # Compute credible regions
    results = {"bin_edges": bin_edges_np, "regions": {}}

    # Calculate median (50th percentile)
    median = np.percentile(all_ecdfs_np, 50, axis=0)

    # Calculate credible regions
    for cr in credible_regions:
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile

        results["regions"][cr] = {
            "lower": np.percentile(all_ecdfs_np, lower_percentile, axis=0),
            "upper": np.percentile(all_ecdfs_np, upper_percentile, axis=0),
            "median": median,
        }

    return results

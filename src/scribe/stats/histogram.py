"""Histogram functions for computing distribution statistics."""

import numpy as np
import jax.numpy as jnp
import jax

# ==============================================================================
# Histogram functions
# ==============================================================================


def compute_histogram_percentiles(
    samples, percentiles=[5, 25, 50, 75, 95], normalize=True, sample_axis=0
):
    """
    Compute percentiles of histogram frequencies across multiple samples.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1
    percentiles : list-like, optional
        List of percentiles to compute (default: [5, 25, 50, 75, 95])
    normalize : bool, optional
        Whether to normalize histograms (default: True)
    sample_axis : int, optional
        Axis containing samples (default: 0)

    Returns
    -------
    bin_edges : array
        Array of bin edges (integers from min to max value + 1)
    hist_percentiles : array
        Array of shape (len(percentiles), len(bin_edges)-1) containing the
        percentiles of histogram frequencies for each bin
    """
    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())

    # Create bin edges (integers from min to max + 1)
    bin_edges = np.arange(
        global_min, global_max + 2
    )  # +2 because we want right edge

    # Initialize array to store histograms
    n_samples = samples.shape[sample_axis]
    n_bins = len(bin_edges) - 1
    all_hists = np.zeros((n_samples, n_bins))

    # Compute histogram for each sample
    for i in range(n_samples):
        sample = samples[i] if sample_axis == 0 else samples[:, i]
        hist, _ = np.histogram(sample, bins=bin_edges)
        if normalize:
            hist = hist / hist.sum()
        all_hists[i] = hist

    # Compute percentiles across samples for each bin
    hist_percentiles = np.percentile(all_hists, percentiles, axis=0)

    return bin_edges, hist_percentiles


# ==============================================================================
# Credible regions functions
# ==============================================================================


def compute_histogram_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    normalize=True,
    sample_axis=0,
    batch_size=1000,
    max_bin=None,
):
    """
    Compute credible regions of histogram frequencies across multiple samples.

    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1
    credible_regions : list-like, optional
        List of credible region percentages to compute (default: [95, 68, 50])
        For example, 95 will compute the 2.5 and 97.5 percentiles
    normalize : bool, optional
        Whether to normalize histograms (default: True)
    sample_axis : int, optional
        Axis containing samples (default: 0)
    batch_size : int, optional
        Number of samples to process in each batch (default: 100)
    max_bin : int, optional
        Maximum number of bins to process (default: None)

    Returns
    -------
    dict
        Dictionary containing: - 'bin_edges': array of bin edges - 'regions':
        nested dictionary where each key is the credible region percentage
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

    # Create bin edges (integers from min to max + 1)
    bin_edges = jnp.arange(global_min, global_max + 2)

    # Function to compute histogram for a single sample
    def compute_single_hist(sample):
        return jnp.histogram(sample, bins=bin_edges)[0]

    # Use lax.map to compute histograms (with automatic batching)
    all_counts = jax.lax.map(
        compute_single_hist, samples_2d, batch_size=batch_size
    )

    # Normalize if requested
    if normalize:
        all_hists = all_counts / all_counts.sum(axis=1)[:, None]
    else:
        all_hists = all_counts

    # Convert to numpy for percentile calculations
    all_hists = np.array(all_hists)

    # Compute credible regions
    results = {"bin_edges": np.array(bin_edges), "regions": {}}

    median = np.percentile(all_hists, 50, axis=0)

    for cr in credible_regions:
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile

        results["regions"][cr] = {
            "lower": np.percentile(all_hists, lower_percentile, axis=0),
            "upper": np.percentile(all_hists, upper_percentile, axis=0),
            "median": median,
        }

    return results

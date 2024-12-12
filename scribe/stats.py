"""
Statistics functions
"""

import numpy as np
# %% ---------------------------------------------------------------------------

def compute_histogram_percentiles(
    samples,
    percentiles=[5, 25, 50, 75, 95],
    normalize=True,
    sample_axis=0
):
    """
    Compute percentiles of histogram frequencies across multiple samples.
    
    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points, n_samples)
        if sample_axis=1
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
        Array of shape (len(percentiles), len(bin_edges)-1) containing
        the percentiles of histogram frequencies for each bin
    """
    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())
    
    # Create bin edges (integers from min to max + 1)
    bin_edges = np.arange(global_min, global_max + 2)  # +2 because we want right edge
    
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

# %% ---------------------------------------------------------------------------

def compute_histogram_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    normalize=True,
    sample_axis=0
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
    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())
    
    # Create bin edges (integers from min to max + 1)
    bin_edges = np.arange(global_min, global_max + 2)  # +2 because we want right edge
    
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
    
    # Compute credible regions
    results = {
        'bin_edges': bin_edges,
        'regions': {}
    }
    
    # Always compute median
    median = np.percentile(all_hists, 50, axis=0)
    
    # Loop through credible regions
    for cr in credible_regions:
        # Compute lower and upper percentiles
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile
        
        # Store results
        results['regions'][cr] = {
            'lower': np.percentile(all_hists, lower_percentile, axis=0),
            'upper': np.percentile(all_hists, upper_percentile, axis=0),
            'median': median
        }
    
    return results

# %% ---------------------------------------------------------------------------

def compute_ecdf_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    sample_axis=0
):
    """
    Compute credible regions of ECDF across multiple samples.
    
    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) by default, or (n_points,
        n_samples) if sample_axis=1
    credible_regions : list-like, optional
        List of credible region percentages to compute (default: [95, 68, 50])
        For example, 95 will compute the 2.5 and 97.5 percentiles
    sample_axis : int, optional
        Axis containing samples (default: 0)
        
    Returns
    -------
    dict
        Dictionary containing:
            - 'x_values': sorted unique values for ECDF computation
            - 'regions': nested dictionary where each key is the credible region percentage
              and values are dictionaries containing:
                - 'lower': lower bound of the credible region
                - 'upper': upper bound of the credible region
                - 'median': median (50th percentile)
    """
    # Get dimensions
    n_samples = samples.shape[sample_axis]
    
    # Compute unique x values across all samples
    x_values = np.sort(np.unique(samples))
    
    # Initialize array to store ECDFs
    all_ecdfs = np.zeros((n_samples, len(x_values)))
    
    # Compute ECDF for each sample
    for i in range(n_samples):
        sample = samples[i] if sample_axis == 0 else samples[:, i]
        sample_sorted = np.sort(sample)
        ecdf = np.searchsorted(sample_sorted, x_values, side='right') / len(sample)
        all_ecdfs[i] = ecdf
    
    # Compute credible regions
    results = {
        'x_values': x_values,
        'regions': {}
    }
    
    # Always compute median
    median = np.percentile(all_ecdfs, 50, axis=0)
    
    # Loop through credible regions
    for cr in credible_regions:
        # Compute lower and upper percentiles
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile
        
        # Store results
        results['regions'][cr] = {
            'lower': np.percentile(all_ecdfs, lower_percentile, axis=0),
            'upper': np.percentile(all_ecdfs, upper_percentile, axis=0),
            'median': median
        }
    
    return results
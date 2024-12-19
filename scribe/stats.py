"""
Statistics functions
"""

# Import numpy for array manipulation
import numpy as np
# Import typing
from typing import Union
# Import JAX-related libraries
import jax.numpy as jnp
import jax.random as random
from jax import scipy as jsp
from numpyro.distributions import Dirichlet

# ------------------------------------------------------------------------------
# Histogram functions
# ------------------------------------------------------------------------------

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
# Credible regions functions
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
            - 'regions': nested dictionary where each key is the credible region
              percentage and values are dictionaries containing:
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

# %% ---------------------------------------------------------------------------
# Fraction of transcriptome functions
# %% ---------------------------------------------------------------------------

def sample_dirichlet_from_parameters(
    parameter_samples: Union[np.ndarray, jnp.ndarray],
    n_samples_dirichlet: int = 1,
    rng_key: random.PRNGKey = random.PRNGKey(42)
) -> jnp.ndarray:
    """
    Samples from a Dirichlet distribution given an array of parameter samples.
    
    Parameters
    ----------
    parameter_samples : array-like
        Array of shape (n_samples, n_variables) containing parameter samples
        to use as concentration parameters for Dirichlet distributions
    n_samples_dirichlet : int, optional
        Number of samples to draw from each Dirichlet distribution (default: 1)
    rng_key : random.PRNGKey
        JAX random number generator key. Defaults to random.PRNGKey(42)
        
    Returns
    -------
    jnp.ndarray
        If n_samples_dirichlet=1:
            Array of shape (n_samples, n_variables)
        If n_samples_dirichlet>1:
            Array of shape (n_samples, n_variables, n_samples_dirichlet)
    """
    # Get dimensions
    n_samples, n_variables = parameter_samples.shape
    
    # Create Dirichlet distribution
    dirichlet_dist = Dirichlet(parameter_samples)
    
    # Sample from the distribution
    samples = dirichlet_dist.sample(
        rng_key,
        sample_shape=(n_samples_dirichlet,)
    )
    
    if n_samples_dirichlet == 1:
        # Return 2D array if only one sample per distribution
        return jnp.transpose(samples, (1, 2, 0)).squeeze(axis=-1)
    else:
        # Return 3D array if multiple samples per distribution
        return jnp.transpose(samples, (1, 2, 0))

# %% ---------------------------------------------------------------------------


def fit_dirichlet_mle(
    samples: Union[np.ndarray, jnp.ndarray],
    max_iter: int = 1000,
    tol: float = 1e-7,
    sample_axis: int = 0
) -> jnp.ndarray:
    """
    Fit a Dirichlet distribution to samples using Maximum Likelihood Estimation.
    
    This implementation uses Newton's method to find the concentration
    parameters that maximize the likelihood of the observed samples. The
    algorithm iteratively updates the concentration parameters using gradient
    and Hessian information until convergence.
    
    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_variables) by default, or (n_variables,
        n_samples) if sample_axis=1, containing Dirichlet samples. Each
        row/column should sum to 1.
    max_iter : int, optional
        Maximum number of iterations for optimization (default: 1000)
    tol : float, optional
        Tolerance for convergence in parameter updates (default: 1e-7)
    sample_axis : int, optional
        Axis containing samples (default: 0)
        
    Returns
    -------
    jnp.ndarray
        Array of concentration parameters for the fitted Dirichlet distribution.
        Shape is (n_variables,).
    """
    # Convert input samples to JAX array and transpose if needed
    x = jnp.asarray(samples)
    if sample_axis == 1:
        x = x.T
    
    # Extract dimensions of the input data
    n_samples, n_variables = x.shape
    
    # Initialize alpha parameters using method of moments estimator
    # This uses the mean and variance of the samples to get a starting point
    alpha = jnp.mean(x, axis=0) * (
        jnp.mean(x * (1 - x), axis=0) / jnp.var(x, axis=0)
    ).mean()
    
    # Compute mean of log samples - this is a sufficient statistic for the MLE
    mean_log_x = jnp.mean(jnp.log(x), axis=0)
    
    def iteration_step(alpha):
        """Single iteration of Newton's method"""
        # Compute sum of current alpha values
        alpha_sum = jnp.sum(alpha)
        
        # Compute gradient using digamma functions
        grad = mean_log_x - (jsp.special.digamma(alpha) - jsp.special.digamma(alpha_sum))
        
        # Compute diagonal of Hessian matrix
        q = -jsp.special.polygamma(1, alpha)
        
        # Compute sum term for Hessian
        z = 1.0 / jsp.special.polygamma(1, alpha_sum)
        
        # Compute update step using gradient and Hessian information
        b = jnp.sum(grad / q) / (1.0 / z + jnp.sum(1.0 / q))
        
        # Return updated alpha parameters
        return alpha - (grad - b) / q
    
    # Iterate Newton's method until convergence or max iterations reached
    for _ in range(max_iter):
        # Compute new alpha values
        alpha_new = iteration_step(alpha)
        
        # Check for convergence
        if jnp.max(jnp.abs(alpha_new - alpha)) < tol:
            break
            
        # Update alpha for next iteration
        alpha = alpha_new
    
    # Return final concentration parameters
    return alpha
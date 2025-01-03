"""
Model comparison utilities for SCRIBE.
"""

import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Callable
from functools import partial
from .results import BaseScribeResults, NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults
import pandas as pd

# ------------------------------------------------------------------------------

@partial(jit, static_argnums=(1,3))
def _compute_log_liks(
    params_dict: Dict, 
    likelihood_fn: Callable, 
    counts: jnp.ndarray, 
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """JIT-compiled log likelihood computation for multiple parameter samples."""
    # Get number of samples
    n_samples = params_dict[next(iter(params_dict))].shape[0]
    
    # Define function to compute log likelihood for a single sample
    def compute_sample_lik(i):
        # Extract parameters for this sample
        params_i = {k: v[i] for k, v in params_dict.items()}
        # Compute log likelihood
        return likelihood_fn(counts, params_i, batch_size)
    
    # Vectorize over samples
    return vmap(compute_sample_lik)(jnp.arange(n_samples))

# ------------------------------------------------------------------------------

@jit
def _compute_waic_stats(log_liks: jnp.ndarray) -> Tuple[float, float, float]:
    """JIT-compiled WAIC statistics computation."""
    # Compute log predictive density
    lpd = jnp.mean(log_liks, axis=0)

    # Compute variance of log predictive density
    var_lpd = jnp.var(log_liks, axis=0)

    # Compute effective number of parameters
    p_eff = jnp.sum(var_lpd)

    # Compute WAIC
    waic = -2 * (jnp.sum(lpd) - p_eff)
    return waic, p_eff, jnp.sum(lpd)

# ------------------------------------------------------------------------------

def compute_waic(
    results: BaseScribeResults,
    counts: jnp.ndarray,
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0),
    cells_axis: int = 0,
) -> Dict:
    """
    Compute the Widely Applicable Information Criterion (WAIC) for a fitted
    model.
    
    Parameters
    ----------
    results : BaseScribeResults
        A fitted model results object containing the posterior samples or the
        ability to generate them
    counts : jnp.ndarray
        The observed count data matrix used to compute the likelihood
    n_samples : int, default=1000
        Number of posterior samples to use for WAIC computation if samples need
        to be generated
    batch_size : int, default=512
        Size of mini-batches used for likelihood computation to manage memory
        usage
    rng_key : random.PRNGKey, default=random.PRNGKey(0)
        JAX random key for reproducibility when generating posterior samples
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
        
    Returns
    -------
    Dict
        Dictionary containing:
            waic : float
                The computed WAIC value (-2 times the computed log pointwise
                predictive density plus a correction for effective number of
                parameters)
            p_eff : float
                The effective number of parameters (computed from the variance
                of individual terms in the log predictive density)
            lpd : float
                The total log pointwise predictive density
            pointwise : Dict
                Dictionary containing cell-wise statistics:
                    lpd : array
                        Log predictive density for each observation
                    p_eff : array
                        Effective parameters (variance) for each observation
                    waic : array
                        WAIC contribution from each observation
    """
    # Get posterior samples if not already present
    if (results.posterior_samples is None or 
        'parameter_samples' not in results.posterior_samples):
        results.sample_posterior(rng_key, n_samples)
    
    # Get likelihood function and parameters
    likelihood_fn = results.get_log_likelihood_fn()
    params_dict = results.posterior_samples['parameter_samples']
    
    # If cells_axis is 1, transpose counts
    if cells_axis == 1:
        counts = counts.T
    
    # Compute log likelihoods directly using _compute_log_liks
    log_liks = _compute_log_liks(params_dict, likelihood_fn, counts, batch_size)
    
    # Compute WAIC statistics
    waic, p_eff, lpd_sum = _compute_waic_stats(log_liks)
    
    # Compute pointwise statistics
    pointwise_lpd = jnp.mean(log_liks, axis=0)
    pointwise_var = jnp.var(log_liks, axis=0)
    
    return {
        'waic': float(waic),
        'p_eff': float(p_eff),
        'lpd': float(lpd_sum),
        'pointwise': {
            'lpd': pointwise_lpd,
            'p_eff': pointwise_var,
            'waic': -2 * (pointwise_lpd - pointwise_var)
        }
    }

# ------------------------------------------------------------------------------

@jit
def _compute_weights(waic_values: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled weight computation."""
    # Compute minimum WAIC
    min_waic = jnp.min(waic_values)

    # Compute WAIC difference
    waic_diff = waic_values - min_waic

    # Compute weights
    weights = jnp.exp(-0.5 * waic_diff)

    # Normalize weights
    return weights / jnp.sum(weights)

def compare_models(
    results_list: List[BaseScribeResults],
    counts: Union[np.ndarray, jnp.ndarray],
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0),
    cells_axis: int = 0,
) -> pd.DataFrame:
    """
    Compare multiple models using WAIC.
    
    Parameters
    ----------
    results_list : List[BaseScribeResults]
        List of results objects from different model fits
    counts : jnp.ndarray
        Observed count data
    n_samples : int
        Number of posterior samples for WAIC computation
    batch_size : int
        Batch size for likelihood computation
    rng_key : random.PRNGKey
        Random key for reproducibility
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing model comparison metrics:
            - Model type
            - WAIC
            - Effective parameters
            - Delta WAIC (difference from best model)
            - WAIC weights
    """
    # If cells_axis is 1, transpose counts
    if cells_axis == 1:
        counts = counts.T
    
    # Convert counts to jnp.ndarray if necessary
    counts = jnp.array(counts, dtype=jnp.float32)
    
    # Split RNG key for each model
    rng_keys = random.split(rng_key, len(results_list))
    
    # Compute WAIC for each model
    waic_results = []
    for results, key in zip(results_list, rng_keys):
        waic = compute_waic(
            results, counts, n_samples, batch_size, key
        )
        waic_results.append({
            'model': results.model_type,
            'waic': waic['waic'],
            'p_eff': waic['p_eff'],
            'lpd': waic['lpd']
        })
    
    # Create DataFrame
    df = pd.DataFrame(waic_results)
    
    # Compute weights using JIT
    waic_values = jnp.array(df['waic'])
    weights = _compute_weights(waic_values)
    
    # Add delta WAIC and weights to DataFrame
    min_waic = df['waic'].min()
    df['delta_waic'] = df['waic'] - min_waic
    df['weight'] = weights
    
    # Sort by WAIC
    df = df.sort_values('waic')
    
    return df
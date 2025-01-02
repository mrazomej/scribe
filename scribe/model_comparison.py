"""
Model comparison utilities for SCRIBE.
"""

import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
from typing import List, Dict, Union, Tuple
from functools import partial
from .results import BaseScribeResults, NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults
import pandas as pd

@partial(jit, static_argnums=(3,))
def _compute_log_liks(
    params: Dict, 
    model, 
    counts: jnp.ndarray, 
    batch_size: int
) -> jnp.ndarray:
    """JIT-compiled log likelihood computation."""
    return model.log_prob(counts, **params, batch_size=batch_size)

@partial(jit, static_argnums=(0,))
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

def compute_waic(
    results: BaseScribeResults,
    counts: jnp.ndarray,
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0)
) -> Dict:
    """
    Compute WAIC (Widely Applicable Information Criterion) for a model.
    
    Parameters
    ----------
    results : BaseScribeResults
        Results object from model fitting
    counts : jnp.ndarray
        Observed count data
    n_samples : int
        Number of posterior samples to use for estimation
    batch_size : int
        Batch size for likelihood computation
    rng_key : random.PRNGKey
        Random key for reproducibility
    
    Returns
    -------
    Dict
        Dictionary containing:
        - 'waic': WAIC value
        - 'p_eff': Effective number of parameters
        - 'lpd': Expected log predictive density
        - 'pointwise': Point-wise WAIC contributions
    """
    # Get posterior samples if not already present
    if (results.posterior_samples is None or 
        'parameter_samples' not in results.posterior_samples):
        results.sample_posterior(rng_key, n_samples)
    
    # Get model and guide functions
    model, _ = results.get_model_and_guide()
    
    # Initialize array for log likelihoods
    log_liks = jnp.zeros((n_samples, results.n_cells, results.n_genes))
    
    # Vectorize log likelihood computation over samples
    vmap_log_lik = vmap(
        lambda params: _compute_log_liks(params, model, counts, batch_size),
        in_axes=0
    )
    
    # Compute log likelihoods for all samples at once
    params_dict = results.posterior_samples['parameter_samples']
    log_liks = vmap_log_lik(params_dict)
    
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
    counts: jnp.ndarray,
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0)
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
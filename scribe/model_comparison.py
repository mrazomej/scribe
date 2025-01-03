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

def _compute_log_liks(
    params_dict: Dict, 
    likelihood_fn: Callable, 
    counts: jnp.ndarray, 
    batch_size: Optional[int] = None,
    ignore_nans: bool = False,
) -> jnp.ndarray:
    """
    Log likelihood computation for multiple parameter samples.
    
    Parameters
    ----------
    params_dict : Dict
        Dictionary of parameter samples
    likelihood_fn : Callable
        Function to compute log likelihood
    counts : jnp.ndarray
        Observed count data
    batch_size : Optional[int], default=None
        Size of mini-batches used for likelihood computation
    ignore_nans : bool, default=False
        If True, removes any samples that contain NaNs. 
    """
    n_samples = params_dict[next(iter(params_dict))].shape[0]
    
    @partial(jit, static_argnums=(0,))
    def compute_sample_lik(i):
        params_i = {k: v[i] for k, v in params_dict.items()}
        return likelihood_fn(counts, params_i, batch_size)
    
    # Compute log likelihoods for all samples
    log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
    
    if ignore_nans:
        # Keep only samples that have no NaNs
        valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=1)
        # Print fraction of samples removed
        print(f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}")
        return log_liks[valid_samples]
    return log_liks

# ------------------------------------------------------------------------------

@jit
def _compute_lppd(log_liks: jnp.ndarray) -> float:
    """
    JIT-compiled log pointwise predictive density computation. This is computed
    as:

    lppd = ∑ᵢ log(1/S ∑ⱼ exp(log p(yᵢ|θⱼ)))

    where i indexes cells, j indexes samples, S is number of posterior samples, 
    yᵢ is the observed data for cell i, and θⱼ is the j-th parameter sample.
    
    Parameters
    ----------
    log_liks : jnp.ndarray
        Log likelihoods for each sample and cell
    """
    # Use log-sum-exp trick for numerical stability
    # First find the maximum log likelihood for each cell
    max_log_liks = jnp.max(log_liks, axis=0)
    
    # Subtract max and exponentiate (stable)
    exp_centered = jnp.exp(log_liks - max_log_liks)
    
    # Average over samples and take log, adding back the max
    # log(mean(exp(x))) = log(sum(exp(x))/n) = log(sum(exp(x))) - log(n)
    lppd = jnp.sum(
        max_log_liks + jnp.log(jnp.mean(exp_centered, axis=0))
    )
    
    return lppd

# ------------------------------------------------------------------------------

@jit
def _compute_p_waic_1(
    log_liks: jnp.ndarray, lppd: Optional[float] = None
) -> float:
    """
    JIT-compiled effective adjustment for WAIC version 1. This is computed as:

    p_waic_1 = 2 ∑ᵢ [log(1/S ∑ⱼ exp(log p(yᵢ|θⱼ))) - 1/S ∑ⱼ log(p(yᵢ|θⱼ))],

    where i indexes cells, j indexes samples, S is number of posterior samples,
    yᵢ is the observed data for cell i, and θⱼ is the j-th parameter sample.
    
    Parameters
    ----------
    log_liks : jnp.ndarray
        Log likelihoods for each sample and cell
    lppd : Optional[float], default=None
        Pre-computed log pointwise predictive density. If None, will be
        computed.
    """
    # Compute lppd if not provided
    if lppd is None:
        lppd = _compute_lppd(log_liks)

    # Compute mean log likelihood for each cell
    mean_log_liks = jnp.sum(jnp.mean(log_liks, axis=0))
    
    # Compute p_waic_1
    p_waic_1 = 2 * (lppd - mean_log_liks)

    return p_waic_1

# ------------------------------------------------------------------------------

@jit
def _compute_p_waic_2(log_liks: jnp.ndarray) -> float:
    """
    JIT-compiled effective adjustment for WAIC version 2. This is computed as:

    p_waic_2 = ∑ᵢ var(log p(yᵢ|θⱼ))

    where i indexes cells, j indexes samples, and θⱼ is the j-th parameter
    sample.

    Parameters
    ----------
    log_liks : jnp.ndarray
        Log likelihoods for each sample and cell
    """
    # Compute variance of log likelihoods
    var_log_liks = jnp.var(log_liks, axis=0)
    
    # Compute p_waic_2
    return jnp.sum(var_log_liks)

# ------------------------------------------------------------------------------

@jit
def _compute_waic_stats(
    log_liks: jnp.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    JIT-compiled computation of WAIC statistics.
    
    Computes the Widely Applicable Information Criterion (WAIC) statistics
    including WAIC1, WAIC2, and their corresponding effective parameter counts
    p_waic_1 and p_waic_2.
    
    Parameters
    ----------
    log_liks : jnp.ndarray
        Array of shape (n_samples, n_cells) containing log likelihoods for each
        posterior sample and cell
        
    Returns
    -------
    Tuple[float, float, float, float]
        A tuple containing:
            - waic_1: WAIC computed using p_waic_1 penalty
            - waic_2: WAIC computed using p_waic_2 penalty  
            - p_waic_1: Effective parameter count computed via mean log
              likelihood
            - p_waic_2: Effective parameter count computed via variance
            - lppd: Log pointwise predictive density
    """
    # Compute lppd
    lppd = _compute_lppd(log_liks)

    # Compute p_waic_1
    p_waic_1 = _compute_p_waic_1(log_liks, lppd)

    # Compute p_waic_2
    p_waic_2 = _compute_p_waic_2(log_liks)

    # Compute WAIC
    waic_1 = -2 * (lppd - p_waic_1)
    waic_2 = -2 * (lppd - p_waic_2)
    return waic_1, waic_2, p_waic_1, p_waic_2, lppd

# ------------------------------------------------------------------------------

def compute_waic(
    results: BaseScribeResults,
    counts: jnp.ndarray,
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0),
    cells_axis: int = 0,
    ignore_nans: bool = False,
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
    ignore_nans: bool = False,
        If True, removes any samples that contain NaNs when evaluating the log
        likelihood.
        
    Returns
    -------
    Dict
        Dictionary containing:
            waic_1 : float
                WAIC computed using p_waic_1 penalty
            waic_2 : float
                WAIC computed using p_waic_2 penalty
            p_waic_1 : float
                Effective parameter count computed via mean log likelihood
            p_waic_2 : float
                Effective parameter count computed via variance
            lppd : float
                The log pointwise predictive density
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
    log_liks = _compute_log_liks(
        params_dict, likelihood_fn, counts, batch_size, ignore_nans
    )
    
    # Compute WAIC statistics
    waic_1, waic_2, p_waic_1, p_waic_2, lppd = _compute_waic_stats(log_liks)
    
    return {
        'waic_1': float(waic_1),
        'waic_2': float(waic_2), 
        'p_waic_1': float(p_waic_1),
        'p_waic_2': float(p_waic_2),
        'lppd': float(lppd)
    }

# ------------------------------------------------------------------------------

@jit
def _compute_waic_weights(waic_values: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled weight computation for WAIC. This is computed as:

    wᵢ = exp(-0.5 * (WAICᵢ - min(WAIC))) / ∑ⱼ exp(-0.5 * (WAICⱼ - min(WAIC))),

    where i and j index models.
    
    Parameters
    ----------
    waic_values : jnp.ndarray
        Array of WAIC values
    """
    # Compute minimum WAIC
    min_waic = jnp.min(waic_values)

    # Compute WAIC difference
    waic_diff = waic_values - min_waic

    # Compute weights
    weights = jnp.exp(-0.5 * waic_diff)

    # Normalize weights
    return weights / jnp.sum(weights)

# ------------------------------------------------------------------------------

def compare_models(
    results_list: List[BaseScribeResults],
    counts: Union[np.ndarray, jnp.ndarray],
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0),
    cells_axis: int = 0,
    ignore_nans: bool = False,
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
    ignore_nans: bool = False,
        If True, removes any samples that contain NaNs when evaluating the log
        likelihood.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing model comparison metrics:
            - Model type
            - WAIC1 and WAIC2
            - Effective parameters (p_waic_1 and p_waic_2)
            - Delta WAIC (difference from best model) for both versions
            - WAIC weights for both versions
            - Log pointwise predictive density (lppd)
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
        print(f"Computing WAIC for {results.model_type}...")
        waic = compute_waic(
            results, counts, n_samples, batch_size, key, 
            ignore_nans=ignore_nans
        )
        waic_results.append({
            'model': results.model_type,
            'waic_1': waic['waic_1'],
            'waic_2': waic['waic_2'],
            'p_waic_1': waic['p_waic_1'],
            'p_waic_2': waic['p_waic_2'],
            'lppd': waic['lppd']
        })
    
    # Create DataFrame
    df = pd.DataFrame(waic_results)
    
    # Compute weights using JIT for both WAIC versions
    waic1_values = jnp.array(df['waic_1'])
    waic2_values = jnp.array(df['waic_2'])
    weights1 = _compute_waic_weights(waic1_values)
    weights2 = _compute_waic_weights(waic2_values)
    
    # Add delta WAIC and weights to DataFrame for both versions
    min_waic1 = df['waic_1'].min()
    min_waic2 = df['waic_2'].min()
    df['delta_waic_1'] = df['waic_1'] - min_waic1
    df['delta_waic_2'] = df['waic_2'] - min_waic2
    df['weight_1'] = weights1
    df['weight_2'] = weights2
    
    # Sort by WAIC1
    df = df.sort_values('waic_1')
    
    return df
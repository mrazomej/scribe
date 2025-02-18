"""
Model comparison utilities for SCRIBE.
"""

import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Callable
from functools import partial
from .results import ScribeResults
import pandas as pd

# ------------------------------------------------------------------------------

def _compute_log_liks(
    params_dict: Dict, 
    likelihood_fn: Callable, 
    counts: jnp.ndarray, 
    batch_size: Optional[int] = None,
    ignore_nans: bool = False,
    return_by: str = 'cell',
    dtype: jnp.dtype = jnp.float32
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
    return_by: str = 'cell'
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    """
    n_samples = params_dict[next(iter(params_dict))].shape[0]
    
    @partial(jit, static_argnums=(0,))
    def compute_sample_lik(i):
        params_i = {k: v[i] for k, v in params_dict.items()}
        return likelihood_fn(
            counts, 
            params_i, 
            batch_size, 
            return_by=return_by,
            dtype=dtype
        )
    
    # Compute log likelihoods for all samples
    log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
    
    if ignore_nans:
        valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=1)
        print(f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}")
        return log_liks[valid_samples]
    return log_liks

# ------------------------------------------------------------------------------

@partial(jit, static_argnames=['aggregate', 'dtype'])
def _compute_lppd(
    log_liks: jnp.ndarray, 
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> Union[float, jnp.ndarray]:
    """
    JIT-compiled log pointwise predictive density computation. This is computed
    as:

    lppd = ∑ᵢ log(1/S ∑ⱼ exp(log p(yᵢ|θⱼ)))

    where i indexes data points, j indexes posterior samples, S is number of
    posterior samples, yᵢ is the observed data for data point i, and θⱼ is the
    j-th posterior parameter sample.
    
    Parameters
    ----------
    log_liks : jnp.ndarray
        Log likelihoods for each posterior sample and data point
    aggregate: bool, default=True
        If True, returns the mean log likelihood for each data point
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    """
    # Convert input to specified dtype 
    log_liks = log_liks.astype(dtype)
    
    # Use log-sum-exp trick for numerical stability
    max_log_liks = jnp.max(log_liks, axis=0)
    exp_centered = jnp.exp(log_liks - max_log_liks)
    
    if aggregate:
        return jnp.sum(
            max_log_liks + jnp.log(jnp.mean(exp_centered, axis=0))
        )
    else:
        return max_log_liks + jnp.log(jnp.mean(exp_centered, axis=0))

# ------------------------------------------------------------------------------

@partial(jit, static_argnames=['aggregate', 'dtype'])
def _compute_p_waic_1(
    log_liks: jnp.ndarray, 
    lppd: Optional[Union[float, jnp.ndarray]] = None, 
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> Union[float, jnp.ndarray]:
    """
    JIT-compiled effective adjustment for WAIC version 1. This is computed as:

    p_waic_1 = 2 ∑ᵢ [log(1/S ∑ⱼ exp(log p(yᵢ|θⱼ))) - 1/S ∑ⱼ log(p(yᵢ|θⱼ))],

    where i indexes data points, j indexes posterior samples, S is number of
    posterior samples, yᵢ is the observed data for data point i, and θⱼ is the
    j-th posterior parameter sample.
    
    Parameters
    ----------
    log_liks : jnp.ndarray
        Log likelihoods for each posterior sample and data point
    lppd : Optional[float], default=None
        Pre-computed log pointwise predictive density. If None, will be
        computed.
    aggregate: bool, default=True
        If True, returns the evaluation of the effective parameter count for the
        entire model. If False, returns the evaluation of the effective
        parameter count for each data point.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    """
    # Compute lppd if not provided
    if lppd is None:
        lppd = _compute_lppd(log_liks, aggregate=aggregate, dtype=dtype)

    if aggregate:
        # Compute mean log likelihood for each data point
        mean_log_liks = jnp.sum(jnp.mean(log_liks, axis=0))
        # Compute p_waic_1
        return 2 * (lppd - mean_log_liks)
    else:
        # Compute mean log likelihood for each data point
        mean_log_liks = jnp.mean(log_liks, axis=0)
        # Compute p_waic_1
        return 2 * (lppd - mean_log_liks)

# ------------------------------------------------------------------------------

@partial(jit, static_argnames=['aggregate', 'dtype'])
def _compute_p_waic_2(
    log_liks: jnp.ndarray, 
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> Union[float, jnp.ndarray]:
    """
    JIT-compiled effective adjustment for WAIC version 2. This is computed as:

    p_waic_2 = ∑ᵢ var(log p(yᵢ|θⱼ))

    where i indexes data points, j indexes posterior samples, and θⱼ is the
    j-th posterior parameter sample.

    Parameters
    ----------
    log_liks : jnp.ndarray
        Log likelihoods for each posterior sample and data point
    aggregate: bool, default=True
        If True, returns the evaluation of the effective parameter count for the
        entire model. If False, returns the evaluation of the effective
        parameter count for each data point.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    """
    # Convert input to specified dtype only if needed
    if log_liks.dtype != dtype:
        log_liks = log_liks.astype(dtype)

    # Compute variance of log likelihoods
    var_log_liks = jnp.var(log_liks, axis=0)
    
    if aggregate:
        # Compute p_waic_2
        return jnp.sum(var_log_liks)
    else:
        return var_log_liks

# ------------------------------------------------------------------------------

@partial(jit, static_argnames=['aggregate', 'dtype'])
def _compute_waic_stats(
    log_liks: jnp.ndarray, 
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32
) -> Dict:
    """
    JIT-compiled computation of WAIC statistics.
    
    Computes the Widely Applicable Information Criterion (WAIC) statistics
    including WAIC1, WAIC2, and their corresponding effective parameter counts
    p_waic_1 and p_waic_2.
    
    Parameters
    ----------
    log_liks : jnp.ndarray
        Array of shape (n_samples, n_data_points) containing log likelihoods for
        each posterior sample and data point
    aggregate: bool, default=True
        If True, returns the evaluation of the WAIC statistics for the entire
        model. If False, returns the evaluation of the WAIC statistics for each
        data point.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    Dict
        Dictionary containing:
            - lppd: Log pointwise predictive density
            - p_waic_1: Effective parameter count computed via mean log
                likelihood
            - p_waic_2: Effective parameter count computed via variance
            - elppd_waic_1: Expected log pointwise predictive density
                computed using p_waic_1 penalty
            - elppd_waic_2: Expected log pointwise predictive density
                computed using p_waic_2 penalty
            - waic_1: WAIC computed using p_waic_1 penalty
            - waic_2: WAIC computed using p_waic_2 penalty
    """
    # Compute lppd
    lppd = _compute_lppd(log_liks, aggregate=aggregate, dtype=dtype)

    # Compute p_waic_1
    p_waic_1 = _compute_p_waic_1(
        log_liks, lppd, aggregate=aggregate, dtype=dtype
    )

    # Compute p_waic_2
    p_waic_2 = _compute_p_waic_2(
        log_liks, aggregate=aggregate, dtype=dtype
    )

    # Compute WAIC
    elppd_waic_1 = lppd - p_waic_1
    elppd_waic_2 = lppd - p_waic_2
    waic_1 = -2 * elppd_waic_1
    waic_2 = -2 * elppd_waic_2

    # Return results as dictionary
    return {
        'lppd': lppd,
        'p_waic_1': p_waic_1,
        'p_waic_2': p_waic_2,
        'elppd_waic_1': elppd_waic_1,
        'elppd_waic_2': elppd_waic_2,
        'waic_1': waic_1,
        'waic_2': waic_2,
    }

# ------------------------------------------------------------------------------

def compute_waic(
    results: BaseScribeResults,
    counts: jnp.ndarray,
    n_samples: int = 1000,
    batch_size: Optional[int] = None,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    cells_axis: int = 0,
    ignore_nans: bool = False,
    dtype: jnp.dtype = jnp.float32
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
    batch_size : Optional[int], default=None
        Size of mini-batches used for likelihood computation to manage memory
        usage. If None, uses full dataset.
    rng_key : random.PRNGKey, default=random.PRNGKey(42)
        JAX random key for reproducibility when generating posterior samples
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    ignore_nans: bool = False,
        If True, removes any samples that contain NaNs when evaluating the log
        likelihood.
    aggregate: bool = True
        If True, returns the evaluation of the WAIC statistics for the entire
        model. If False, returns the evaluation of the WAIC statistics for each
        data point.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
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
        results.get_posterior_samples(rng_key, n_samples)
    
    # Get likelihood function and parameters
    likelihood_fn = results.get_log_likelihood_fn()
    params_dict = results.posterior_samples['parameter_samples']
    
    # If cells_axis is 1, transpose counts
    if cells_axis == 1:
        counts = counts.T
    
    # Compute log likelihoods directly using _compute_log_liks
    log_liks = _compute_log_liks(
        params_dict, 
        likelihood_fn, 
        counts, 
        batch_size, 
        ignore_nans, 
        dtype=dtype
    )
    
    # Compute WAIC statistics
    return _compute_waic_stats(log_liks, dtype=dtype)
    

# ------------------------------------------------------------------------------

def compute_waic_by_gene(
    results: BaseScribeResults,
    counts: jnp.ndarray,
    n_samples: int = 1000,
    batch_size: Optional[int] = None,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    cells_axis: int = 0,
    ignore_nans: bool = False,
    dtype: jnp.dtype = jnp.float32
) -> Dict:
    """
    Compute WAIC (Widely Applicable Information Criterion) statistics for each
    gene independently.

    This function computes WAIC statistics per gene rather than aggregated
    across all genes. This allows for gene-specific model comparison and
    assessment.

    Parameters
    ----------
    results : BaseScribeResults
        Results object containing model fit information
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    n_samples : int, default=1000
        Number of posterior samples to use for WAIC computation
    batch_size : Optional[int], default=None
        Size of mini-batches for computation. If None, uses full dataset
    rng_key : random.PRNGKey, default=random.PRNGKey(42)
        Random number generator key for reproducibility
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows, 1 means
        cells are columns
    ignore_nans : bool, default=False
        Whether to ignore NaN values in computation
    dtype : jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations

    Returns
    -------
    Dict
        Dictionary containing arrays of length n_genes with WAIC statistics:
            lppd : array
                Log pointwise predictive density for each gene
            p_waic_1 : array
                Effective parameter count via mean log likelihood for each gene
            p_waic_2 : array
                Effective parameter count via variance for each gene
            waic_1 : array
                WAIC computed using p_waic_1 penalty for each gene
            waic_2 : array
                WAIC computed using p_waic_2 penalty for each gene
    """
    # Get posterior samples if not already present
    if (results.posterior_samples is None or 
        'parameter_samples' not in results.posterior_samples):
        results.get_posterior_samples(rng_key, n_samples)
    
    # Get likelihood function and parameters
    likelihood_fn = results.get_log_likelihood_fn()
    params_dict = results.posterior_samples['parameter_samples']
    
    # If cells_axis is 1, transpose counts
    if cells_axis == 1:
        counts = counts.T
    
    # Compute log likelihoods directly using _compute_log_liks
    log_liks = _compute_log_liks(
        params_dict, likelihood_fn, counts, batch_size, ignore_nans, 
        return_by='gene', dtype=dtype
    )
    
    # Compute WAIC statistics
    return _compute_waic_stats(log_liks, aggregate=False, dtype=dtype)
    
# ------------------------------------------------------------------------------

@partial(jit, static_argnames=['dtype'])
def _compute_waic_weights(
    waic_values: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    JIT-compiled weight computation for WAIC. This is computed as:

    wᵢ = exp(-0.5 * (WAICᵢ - min(WAIC))) / ∑ⱼ exp(-0.5 * (WAICⱼ - min(WAIC))),

    where i and j index models.
    
    Parameters
    ----------
    waic_values : jnp.ndarray
        Array of WAIC values
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    """
    # Convert input to specified dtype only if needed
    if waic_values.dtype != dtype:
        waic_values = waic_values.astype(dtype)

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
    dtype: jnp.dtype = jnp.float32
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
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    
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
    counts = jnp.array(counts, dtype=dtype)
    
    # Split RNG key for each model
    rng_keys = random.split(rng_key, len(results_list))
    
    # Compute WAIC for each model
    waic_results = []
    for results, key in zip(results_list, rng_keys):
        print(f"Computing WAIC for {results.model_type}...")
        waic = compute_waic(
            results, counts, n_samples, batch_size, key, 
            ignore_nans=ignore_nans, dtype=dtype
        )
        # Add model type to results
        waic['model'] = results.model_type
        # Append results to list
        waic_results.append(waic)

    # Create DataFrame
    df = pd.DataFrame(waic_results)
    
    # Convert DataFrame columns to numeric arrays
    waic1_values = jnp.array(df["waic_1"].values, dtype=dtype)
    waic2_values = jnp.array(df["waic_2"].values, dtype=dtype)
    
    
    # Compute weights using JIT for both WAIC versions
    weights1 = _compute_waic_weights(waic1_values, dtype=dtype)
    weights2 = _compute_waic_weights(waic2_values, dtype=dtype)
    
    # Add delta WAIC and weights to DataFrame for both versions
    min_waic1 = jnp.min(waic1_values)
    min_waic2 = jnp.min(waic2_values)
    df['delta_waic_1'] = jnp.array(df['waic_1'].values, dtype=dtype) - min_waic1
    df['delta_waic_2'] = jnp.array(df['waic_2'].values, dtype=dtype) - min_waic2
    df['weight_1'] = weights1
    df['weight_2'] = weights2
    
    return df

# ------------------------------------------------------------------------------

def compare_models_by_gene(
    results_list: List[BaseScribeResults],
    counts: Union[np.ndarray, jnp.ndarray],
    n_samples: int = 1000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(0),
    cells_axis: int = 0,
    ignore_nans: bool = False,
    dtype: jnp.dtype = jnp.float32
) -> pd.DataFrame:
    """
    Compare multiple models using WAIC, computed separately for each gene.
    
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
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing model comparison metrics for each gene:
            - Model type
            - Gene index
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
    counts = jnp.array(counts, dtype=dtype)
    n_genes = counts.shape[1]
    
    # Split RNG key for each model
    rng_keys = random.split(rng_key, len(results_list))
    
    # Initialize empty DataFrame
    df_list = []
    
    # Compute WAIC for each model
    for results, key in zip(results_list, rng_keys):
        print(f"Computing WAIC by gene for {results.model_type}...")
        waic = compute_waic_by_gene(
            results, counts, n_samples, batch_size, key, 
            ignore_nans=ignore_nans
        )
        # Add model type to results
        waic['model'] = results.model_type
        # Add gene indices to results
        waic['gene'] = np.arange(n_genes)
        # Append results to list dataframe
        df_list.append(pd.DataFrame(waic))
    
    # Combine all model results
    df = pd.concat(df_list, ignore_index=True)
    
    @partial(jit, static_argnames=['n_models'])
    def _compute_gene_metrics(waic_values, n_models):
        """
        Vectorized computation of delta WAIC and weights for all genes at once
        """
        # Reshape to (n_genes, n_models)
        waic_matrix = waic_values.reshape(-1, n_models)
        
        # Compute min WAIC for each gene
        min_waics = jnp.min(waic_matrix, axis=1, keepdims=True)
        
        # Compute delta WAIC
        delta_waics = waic_matrix - min_waics
        
        # Compute weights
        exp_terms = jnp.exp(-0.5 * delta_waics)
        weights = exp_terms / jnp.sum(exp_terms, axis=1, keepdims=True)
        
        return delta_waics.ravel(), weights.ravel()

    # Get number of models
    n_models = len(results_list)
    
    # Compute metrics for both WAIC versions
    waic1_values = jnp.array(df['waic_1'].values, dtype=dtype)
    waic2_values = jnp.array(df['waic_2'].values, dtype=dtype)
    
    delta_waic1, weights1 = _compute_gene_metrics(waic1_values, n_models)
    delta_waic2, weights2 = _compute_gene_metrics(waic2_values, n_models)
    
    # Add results to dataframe
    df['delta_waic_1'] = delta_waic1
    df['delta_waic_2'] = delta_waic2
    df['weight_1'] = weights1
    df['weight_2'] = weights2
    
    return df

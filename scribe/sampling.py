"""
Sampling utilities for SCRIBE.
"""

from jax import random
import jax.numpy as jnp
from numpyro.infer import Predictive
from typing import Dict, Optional, Union, Callable
from numpyro.infer import SVI

# ------------------------------------------------------------------------------
# Posterior predictive samples
# ------------------------------------------------------------------------------

def sample_variational_posterior(
    guide: Callable,
    params: Dict,
    n_cells: int,
    n_genes: int,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples: int = 100,
) -> Dict:
    """
    Sample parameters from the variational posterior distribution.
    
    Parameters
    ----------
    guide : Callable
        Guide function
    params : Dict
        Dictionary containing optimized variational parameters
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of posterior samples to generate (default: 100)
        
    Returns
    -------
    Dict
        Dictionary containing samples from the variational posterior
    """
    # Create predictive object for posterior parameter samples
    predictive_param = Predictive(
        guide,
        params=params,
        num_samples=n_samples
    )
    
    # Sample parameters from the variational posterior
    return predictive_param(
        rng_key,
        n_cells,
        n_genes,
    )

# ------------------------------------------------------------------------------

def generate_predictive_samples(
    model: Callable,
    posterior_samples: Dict,
    n_cells: int,
    n_genes: int,
    rng_key: random.PRNGKey,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """
    Generate predictive samples using posterior parameter samples.
    
    Parameters
    ----------
    model : Callable
        Model function
    posterior_samples : Dict
        Dictionary containing samples from the variational posterior
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    rng_key : random.PRNGKey
        JAX random number generator key
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.
        
    Returns
    -------
    jnp.ndarray
        Array of predictive samples
    """
    # Create predictive object for generating new data
    predictive = Predictive(
        model,
        posterior_samples,
        # Take the number of samples from the first parameter
        num_samples=next(iter(posterior_samples.values())).shape[0]
    )
    
    # Generate predictive samples
    predictive_samples = predictive(
        rng_key,
        n_cells,
        n_genes,
        batch_size=batch_size
    )
    
    return predictive_samples["counts"]

# ------------------------------------------------------------------------------

def generate_ppc_samples(
    model: Callable,
    guide: Callable,
    params: Dict,
    n_cells: int,
    n_genes: int,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> Dict:
    """
    Generate posterior predictive check samples.
    
    Parameters
    ----------
    model : Callable
        Model function
    guide : Callable
        Guide function
    params : Dict
        Dictionary containing optimized variational parameters
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of posterior samples to generate (default: 100)
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.
        
    Returns
    -------
    Dict
        Dictionary containing: - 'parameter_samples': Samples from the
        variational posterior - 'predictive_samples': Samples from the
        predictive distribution
    """
     # Split RNG key for parameter sampling and predictive sampling
    key_params, key_pred = random.split(rng_key)
    
    # Sample from variational posterior
    posterior_param_samples = sample_variational_posterior(
        guide,
        params,
        n_cells,
        n_genes,
        key_params,
        n_samples
    )
    
    # Generate predictive samples
    predictive_samples = generate_predictive_samples(
        model,
        posterior_param_samples,
        n_cells,
        n_genes,
        key_pred,
        batch_size
    )
    
    return {
        'parameter_samples': posterior_param_samples,
        'predictive_samples': predictive_samples
    }
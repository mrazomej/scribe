"""
Utility functions for SCRIBE.
"""

import os
import jax
import numpy as np
from contextlib import contextmanager

import numpyro.distributions as dist
import scipy.stats as stats

# ------------------------------------------------------------------------------

def git_root(current_path=None):
    """
    Finds the root directory of a Git repository.
    
    Parameters
    ----------
    current_path : str, optional
        The starting path. If None, uses the current working directory.
    
    Returns
    -------
    str
        The path to the Git root directory, or None if not found.
    """
    if current_path is None:
        current_path = os.getcwd()
    
    while current_path and current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)  # Move up one directory level
    
    return None  # Git root not found

# ------------------------------------------------------------------------------

@contextmanager
def use_cpu():
    """
    Context manager to temporarily force JAX computations to run on CPU.
    
    This is useful when you want to ensure specific computations run on CPU
    rather than GPU/TPU, for example when running out of GPU memory.
    
    Returns
    -------
    None
        Yields control back to the context block
        
    Example
    -------
    >>> # Force posterior sampling to run on CPU
    >>> with use_cpu():
    ...     results.get_ppc_samples(n_samples=100)
    """
    # Store the current default device to restore it later
    original_device = jax.default_device()
    
    # Get the first available CPU device
    cpu_device = jax.devices('cpu')[0]
    
    # Set CPU as the default device for JAX computations
    jax.default_device(cpu_device)
    
    try:
        # Yield control to the context block
        yield
    finally:
        # Restore the original default device when exiting the context
        jax.default_device(original_device)

# ------------------------------------------------------------------------------

def numpyro_to_scipy(distribution: dist.Distribution) -> stats.rv_continuous:
    """
    Get the corresponding scipy.stats distribution for a
    numpyro.distributions.Distribution.
    
    Parameters
    ----------
    distribution : numpyro.distributions.Distribution
        The numpyro distribution to convert
        
    Returns
    -------
    scipy.stats.rv_continuous
        The corresponding scipy.stats distribution
    """
    if isinstance(distribution, dist.Beta):
        # Extract parameters from distribution
        a = distribution.concentration1
        b = distribution.concentration0
        return stats.beta(a, b)
    elif isinstance(distribution, dist.Gamma):
        # Extract parameters from distribution
        shape = distribution.concentration
        scale = 1 / distribution.rate
        return stats.gamma(shape, scale)
    elif isinstance(distribution, dist.LogNormal):
        # Extract parameters from distribution
        loc = distribution.loc
        scale = distribution.scale
        return stats.lognorm(scale, loc=0, scale=np.exp(loc))
    elif isinstance(distribution, dist.Dirichlet):
        return stats.dirichlet(distribution.concentration)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

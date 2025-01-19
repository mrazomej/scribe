"""
Utility functions for SCRIBE.
"""

import os
import jax
from contextlib import contextmanager

def git_root(current_path=None):
    """
    Finds the root directory of a Git repository.
    
    Args:
        current_path (str, optional): The starting path. If None, uses the current working directory.
    
    Returns:
        str: The path to the Git root directory, or None if not found.
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
import jax
from contextlib import contextmanager

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
    ...     results.ppc_samples(n_samples=100)
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
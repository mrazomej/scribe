"""
Utility functions for SCRIBE.
"""

import os
import jax
import numpy as np
from contextlib import contextmanager
from pathlib import Path

import numpyro.distributions as dist
import scipy.stats as stats

# Import custom distributions
from ..stats import BetaPrime

# ------------------------------------------------------------------------------
# Configure JAX compilation cache to avoid recompilation
# This can be overridden by setting JAX_COMPILATION_CACHE_DIR environment variable
# before importing jax. For maximum effectiveness, set the environment variable
# before importing any JAX-related modules.
# ------------------------------------------------------------------------------

# Set default cache directory if not already set via environment variable
if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
    # Use a cache directory in the user's home directory
    cache_dir = Path.home() / ".cache" / "jax"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        # Cache all compilations (not just those taking >1 second)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:
        # If config update fails (e.g., older JAX version), silently continue
        pass
else:
    # If environment variable is set, ensure the directory exists
    cache_dir = Path(os.environ["JAX_COMPILATION_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Still update config to ensure caching is enabled
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:
        pass

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
        if os.path.isdir(os.path.join(current_path, ".git")):
            return current_path
        current_path = os.path.dirname(
            current_path
        )  # Move up one directory level

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
    cpu_device = jax.devices("cpu")[0]

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
    elif isinstance(distribution, BetaPrime):
        return stats.betaprime(
            distribution.concentration1, distribution.concentration0
        )
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

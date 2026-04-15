"""Array-backend dispatch helpers for NumPy / JAX interoperability.

Provides lightweight dispatchers that inspect the concrete type of an
input array and return the matching module (``numpy`` vs ``jax.numpy``,
``scipy.stats.norm`` vs ``jax.scipy.stats.norm``, etc.).  This lets
call-sites write backend-agnostic code with a single ``xp = ...``
binding instead of duplicating logic.

When posterior samples have been converted to NumPy via
``convert_to_numpy=True``, all downstream DE computations transparently
use the NumPy/SciPy stack — avoiding JAX's XLA CPU backend overhead
and unnecessary GPU round-trips.  When samples remain as ``jax.Array``
(the default for small sample counts), the JAX/GPU path is used
automatically.
"""

from __future__ import annotations

import math

import numpy as np


def _array_module(x):
    """Return ``numpy`` or ``jax.numpy`` depending on the input type.

    Parameters
    ----------
    x : array-like
        Any array whose concrete type determines the backend.

    Returns
    -------
    module
        ``numpy`` if *x* is a plain ``numpy.ndarray``, otherwise
        ``jax.numpy``.
    """
    if isinstance(x, np.ndarray):
        return np
    import jax.numpy as jnp

    return jnp


def _stats_norm(x):
    """Return ``scipy.stats.norm`` or ``jax.scipy.stats.norm``.

    Parameters
    ----------
    x : array-like
        Any array whose concrete type determines the backend.

    Returns
    -------
    module
        The ``norm`` object from the matching SciPy backend.
    """
    if isinstance(x, np.ndarray):
        from scipy.stats import norm

        return norm
    from jax.scipy.stats import norm

    return norm


def _special_module(x):
    """Return ``scipy.special`` or ``jax.scipy.special``.

    Parameters
    ----------
    x : array-like
        Any array whose concrete type determines the backend.

    Returns
    -------
    module
        ``scipy.special`` or ``jax.scipy.special``.
    """
    if isinstance(x, np.ndarray):
        import scipy.special as special

        return special
    from jax import scipy as jsp

    return jsp.special


def _vmap_chunk_size(n_total: int, per_element_bytes: int) -> int:
    """Compute how many elements to vmap at once given GPU memory limits.

    Used to chunk ``vmap`` calls over the sample axis when the full
    batch would exceed available device memory.

    Parameters
    ----------
    n_total : int
        Total number of elements (e.g. ``n_samples``).
    per_element_bytes : int
        Estimated memory consumption per element in bytes
        (output + intermediates).

    Returns
    -------
    int
        Chunk size in ``[1, n_total]``.  Equal to ``n_total`` when
        there is no GPU or memory stats are unavailable.
    """
    budget = _gpu_memory_budget()
    if budget == math.inf or per_element_bytes <= 0:
        return n_total
    chunk = max(1, int(budget // per_element_bytes))
    return min(chunk, n_total)


def _gpu_memory_budget(fraction: float = 0.8) -> float:
    """Estimate usable GPU memory in bytes.

    Queries the default JAX device for its total memory and returns
    ``fraction`` of that value.  On CPU-only runtimes or when memory
    stats are unavailable, returns ``math.inf`` so that callers never
    chunk unnecessarily.

    Parameters
    ----------
    fraction : float, default=0.8
        Fraction of total device memory to consider usable.  The
        remaining headroom covers JAX runtime allocations, XLA
        temporaries, and other resident tensors.

    Returns
    -------
    float
        Usable bytes on the GPU, or ``math.inf`` when no GPU is
        present or stats cannot be read.
    """
    try:
        import jax

        dev = jax.local_devices()[0]
        # CPU devices have platform == "cpu" and no useful memory stats
        if dev.platform == "cpu":
            return math.inf
        stats = dev.memory_stats()
        if stats and "bytes_limit" in stats:
            return stats["bytes_limit"] * fraction
    except Exception:
        pass
    return math.inf

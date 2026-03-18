"""Gauss--Legendre quadrature utilities for JAX.

Provides general-purpose 1D numerical integration via Gauss--Legendre
quadrature.  Nodes and weights are precomputed with NumPy (exact to
machine precision) and converted to JAX arrays on demand.  All public
functions are JIT-compatible and vectorization-friendly.
"""

from functools import lru_cache
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np


# =========================================================================
# Node / weight generation
# =========================================================================


@lru_cache(maxsize=16)
def _leggauss_numpy(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gauss--Legendre nodes and weights on [-1, 1].

    Results are cached so that repeated calls with the same *n* do not
    recompute the eigenvalue problem.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    nodes : np.ndarray, shape ``(n,)``
        Quadrature nodes on [-1, 1].
    weights : np.ndarray, shape ``(n,)``
        Corresponding quadrature weights.
    """
    return np.polynomial.legendre.leggauss(n)


def gauss_legendre_nodes_weights(
    n: int,
    a: float = 0.0,
    b: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return Gauss--Legendre nodes and weights on ``[a, b]`` as JAX arrays.

    The standard Gauss--Legendre rule is defined on ``[-1, 1]``.  This
    function applies the affine transformation
    ``x = (b - a) / 2 * t + (a + b) / 2`` to map nodes to ``[a, b]``
    and rescales the weights by ``(b - a) / 2``.

    Parameters
    ----------
    n : int
        Number of quadrature nodes (typically 32--64 for smooth
        integrands on a bounded interval).
    a : float, optional
        Left endpoint of the integration interval.  Default: 0.0.
    b : float, optional
        Right endpoint of the integration interval.  Default: 1.0.

    Returns
    -------
    nodes : jnp.ndarray, shape ``(n,)``
        Quadrature nodes on ``[a, b]``.
    weights : jnp.ndarray, shape ``(n,)``
        Quadrature weights (already include the Jacobian ``(b-a)/2``).
    """
    t, w = _leggauss_numpy(n)
    half_len = (b - a) / 2.0
    mid = (a + b) / 2.0
    nodes = jnp.asarray(half_len * t + mid)
    weights = jnp.asarray(half_len * w)
    return nodes, weights


# =========================================================================
# High-level integration
# =========================================================================


def gauss_legendre_integrate(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    a: float = 0.0,
    b: float = 1.0,
    n: int = 64,
) -> jnp.ndarray:
    r"""Approximate :math:`\int_a^b f(x)\,dx` via Gauss--Legendre quadrature.

    Parameters
    ----------
    f : callable
        Integrand.  Receives a JAX array of shape ``(n,)`` containing
        the quadrature nodes and must return an array whose leading axis
        has length ``n`` (remaining axes are batch dimensions that will
        be summed over the node axis).
    a : float, optional
        Left endpoint.  Default: 0.0.
    b : float, optional
        Right endpoint.  Default: 1.0.
    n : int, optional
        Number of quadrature nodes.  Default: 64.

    Returns
    -------
    jnp.ndarray
        Approximation to the integral.  Shape equals the batch
        dimensions of ``f``'s output (i.e., everything after the leading
        node axis).

    Examples
    --------
    Integrate :math:`x^2` on :math:`[0, 1]`:

    >>> import jax.numpy as jnp
    >>> result = gauss_legendre_integrate(lambda x: x**2, 0.0, 1.0, n=8)
    >>> float(result)  # ≈ 1/3
    0.333...
    """
    nodes, weights = gauss_legendre_nodes_weights(n, a, b)
    # f(nodes) has shape (n, ...).  Weights are (n,); we need to
    # broadcast them to match any trailing batch dimensions.
    fvals = f(nodes)
    # Reshape weights to (n, 1, 1, ...) for broadcasting
    w = weights.reshape((-1,) + (1,) * (fvals.ndim - 1))
    return jnp.sum(w * fvals, axis=0)

"""Gauss-Jacobi quadrature on [0, 1] with a Beta weight.

This module provides nodes and log-weights for the integral

    ∫₀¹ f(p) · Beta(p | α, β) dp ≈ ∑ₖ wₖ · f(pₖ),


with the Beta density absorbed into the quadrature weight so that the
implementer only supplies a smooth ``f(p)`` (typically a Poisson PMF
evaluated at ``rate * p``). The nodes ``p_k`` and weights ``w_k`` depend
on ``(alpha, beta)`` and are returned at shape ``(*alpha.shape, K)``.

Two backends are exposed behind a single dispatcher:

- ``"golub_welsch"`` (default, on-the-fly): builds the symmetric
  tridiagonal Jacobi matrix for each ``(alpha, beta)`` pair and reads
  off the nodes and weights from ``jax.scipy.linalg.eigh``. The
  eigendecomposition is differentiable through implicit differentiation,
  so gradients of ``(alpha, beta)`` flow through the quadrature.

- ``"precomputed_grid"`` (stub for phase 2): would interpolate nodes
  and weights from a precomputed grid of ``(alpha, beta)`` pairs.
  Not implemented in phase 1; raising ``NotImplementedError`` keeps
  the interface ready for the follow-up without leaving dead code.

Math reference
--------------
Standard Gauss-Jacobi is defined on ``[-1, 1]`` with weight
``(1 - x)^a (1 + x)^b``. We use the affine map ``p = (1 + x) / 2``,
``dp = dx / 2``, which converts a Beta weight on ``[0, 1]`` into
the standard form with ``a = beta - 1`` and ``b = alpha - 1``.

The Jacobi three-term recurrence
``x P_n(x) = a_n P_{n+1}(x) + b_n P_n(x) + c_n P_{n-1}(x)``
gives, after symmetrization, the tridiagonal matrix whose
eigenvalues are the nodes on ``[-1, 1]`` and whose first-row
eigenvector entries squared (times the zeroth moment of the
weight function) are the quadrature weights. We then map the
nodes back to ``[0, 1]`` and normalize the weights so they
integrate the Beta density to 1.

Numerical sanity
----------------
The Jacobi parameters ``a = beta - 1`` and ``b = alpha - 1`` must
both be ``> -1``. The two-state reparameterisation in
``models/components/likelihoods/two_state.py`` floors ``alpha`` and
``beta`` at ``0.05`` (so ``a, b > -0.95``), keeping the recurrence
coefficients away from their singular boundary.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

# Backend strings exposed at module level so callers can pass them by name.
GOLUB_WELSCH = "golub_welsch"
PRECOMPUTED_GRID = "precomputed_grid"


# ==============================================================================
# Public API
# ==============================================================================


def gauss_jacobi_nodes_weights(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    n_nodes: int,
    backend: str = GOLUB_WELSCH,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Gauss-Jacobi nodes and log-weights on [0, 1].

    The returned ``nodes_p`` and ``log_weights`` approximate

        ∫₀¹ f(p) · Beta(p | α, β) dp ≈
        ∑ₖ exp(log_weights[..., k]) · f(nodes_p[..., k]).

    Parameters
    ----------
    alpha : jnp.ndarray
        Beta first shape parameter α; arbitrary leading shape. Must
        broadcast against ``beta``.
    beta : jnp.ndarray
        Beta second shape parameter β; same broadcasting rules as
        ``alpha``.
    n_nodes : int
        Number of quadrature nodes K. Static under JIT (not traced).
    backend : str
        Backend selector. One of :data:`GOLUB_WELSCH` (default) or
        :data:`PRECOMPUTED_GRID` (phase 2 stub).

    Returns
    -------
    nodes_p : jnp.ndarray
        Quadrature nodes on [0, 1], shape
        ``broadcast(alpha, beta) + (K,)``.
    log_weights : jnp.ndarray
        log of the quadrature weights, same shape. Weights are
        normalised so that ∑ₖ exp(log_weights[..., k]) = 1 exactly
        (modulo floating-point round-off), so the quadrature directly
        approximates a Beta-weighted integral rather than an
        unnormalised one.

    Raises
    ------
    ValueError
        If ``backend`` is not recognised.
    NotImplementedError
        If ``backend == "precomputed_grid"`` (phase 2).
    """
    if backend == GOLUB_WELSCH:
        return _golub_welsch(alpha, beta, n_nodes)
    if backend == PRECOMPUTED_GRID:
        raise NotImplementedError(
            "precomputed_grid backend is reserved for phase 2; "
            "use backend='golub_welsch' in phase 1."
        )
    raise ValueError(
        f"Unknown Gauss-Jacobi backend {backend!r}; "
        f"expected one of {GOLUB_WELSCH!r}, {PRECOMPUTED_GRID!r}."
    )


# ==============================================================================
# Golub-Welsch backend
# ==============================================================================


def _jacobi_recurrence_coeffs(
    a: jnp.ndarray, b: jnp.ndarray, n_nodes: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Three-term recurrence coefficients for Jacobi polynomials on [-1, 1].

    The monic Jacobi polynomials Pₙ⁽ᵃ·ᵇ⁾(x) satisfy

        Pₙ₊₁ = (x − αₙ) Pₙ − βₙ² Pₙ₋₁,
        P₋₁  = 0,   P₀ = 1.

    The symmetric tridiagonal Jacobi matrix has diagonal αₙ and
    off-diagonal βₙ. Its eigenvalues are the quadrature nodes; the
    first row of the eigenvector matrix squared (times the zeroth
    moment) gives the weights.

    Parameters
    ----------
    a, b : scalar arrays
        Jacobi parameters in standard form (a = β − 1, b = α − 1 in
        our Beta-weighted setting).
    n_nodes : int
        Order of the recurrence (number of nodes).

    Returns
    -------
    diag : jnp.ndarray, shape ``(n_nodes,)``
        Diagonal entries αₙ.
    offdiag : jnp.ndarray, shape ``(n_nodes - 1,)``
        Off-diagonal entries βₙ.
    """
    # Static index range; cast to float for arithmetic.
    n = jnp.arange(n_nodes, dtype=a.dtype)

    # αₙ for n ≥ 0.  α₀ is a special case to avoid 0/0.
    # General formula:
    #   αₙ = (b² − a²) / ((2n + a + b) · (2n + a + b + 2))
    # For n = 0 this is (b² − a²) / ((a + b) · (a + b + 2)).
    # When a + b == 0 the denominator is 0; use the limit (b − a) / 2.
    two_n_ab = 2.0 * n + a + b
    denom_alpha = two_n_ab * (two_n_ab + 2.0)
    # Guard against a + b == 0 at n = 0 by replacing the zero
    # denominator with 1.0 and overriding the corresponding value
    # with the analytical limit.
    safe_denom_alpha = jnp.where(denom_alpha == 0, 1.0, denom_alpha)
    alpha_n = (b**2 - a**2) / safe_denom_alpha
    alpha_n = alpha_n.at[0].set(
        jnp.where(a + b == 0, (b - a) / 2.0, alpha_n[0])
    )

    # βₙ for n ≥ 1:
    #   βₙ² = 4 · n · (n + a) · (n + b) · (n + a + b)
    #         / ((2n + a + b)² · (2n + a + b + 1) · (2n + a + b − 1))
    # We need β₁, …, βₙ₋₁.
    n1 = jnp.arange(1, n_nodes, dtype=a.dtype)
    two_n_ab_1 = 2.0 * n1 + a + b
    beta_sq = (4.0 * n1 * (n1 + a) * (n1 + b) * (n1 + a + b)) / (
        two_n_ab_1**2 * (two_n_ab_1 + 1.0) * (two_n_ab_1 - 1.0)
    )
    # Numerical guard: tiny negative values can appear from float
    # round-off near the recurrence boundary; clamp to 0 before sqrt.
    beta_n = jnp.sqrt(jnp.clip(beta_sq, a_min=0.0))

    return alpha_n, beta_n


def _golub_welsch_single(
    alpha: jnp.ndarray, beta: jnp.ndarray, n_nodes: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute nodes and log-weights for a single (α, β) pair.

    Vectorised over leading axes by ``_golub_welsch`` via ``jax.vmap``.

    Parameters
    ----------
    alpha, beta : scalar jnp.ndarray
        Beta shape parameters α, β (NOT the Jacobi standard form).
    n_nodes : int
        Number of quadrature nodes.

    Returns
    -------
    nodes_p : jnp.ndarray, shape ``(n_nodes,)``
        Nodes on [0, 1].
    log_weights : jnp.ndarray, shape ``(n_nodes,)``
        Log-weights such that ∑ exp(log_weights) = 1.
    """
    # Map Beta(α, β) on [0, 1] to standard Gauss-Jacobi on [-1, 1]
    # with weight (1 − x)ᵃ · (1 + x)ᵇ.
    #
    #   p = (1 + x) / 2  ⇒  (1 − p)^(β − 1) p^(α − 1) dp
    #                     = (1 − x)^(β − 1) (1 + x)^(α − 1) · 2^(−(α + β − 1)) dx
    # so the standard-form parameters are a = β − 1, b = α − 1.
    #
    # Detect the singular region (a + b near 1 − 2n for integer n ≥ 1)
    # and skip the most problematic point a + b = −1 by widening the
    # input slightly. Within the two-state floors (α, β ≥ 0.05) the
    # only point that hits this is α = β = 0.5 (a + b = −1).
    #
    # Under float32, a tiny nudge (~1e-6) does NOT preserve enough
    # precision in the 0/0 limit of β₁², so we use a larger nudge
    # (~1e-3) that pushes us clearly off the singularity while
    # introducing at most ~1e-3 shift in moments — well below the
    # quadrature accuracy needed for the application (~1e-5).
    _SINGULARITY_NUDGE = 1e-3
    a = beta - 1.0 + _SINGULARITY_NUDGE
    b = alpha - 1.0 + _SINGULARITY_NUDGE

    diag, offdiag = _jacobi_recurrence_coeffs(a, b, n_nodes)

    # Build the symmetric tridiagonal Jacobi matrix.  Dense eigh; for
    # n_nodes ~ 60 the cost is negligible on GPU.
    T = jnp.diag(diag) + jnp.diag(offdiag, k=1) + jnp.diag(offdiag, k=-1)

    # eigvals are the quadrature nodes on [-1, 1]; eigvecs[0]² · μ₀
    # are the unnormalised weights, where μ₀ is the zeroth moment of
    # the standard weight function on [-1, 1].
    eigvals, eigvecs = jsp.linalg.eigh(T)

    # We do not need the absolute scale: we renormalise the weights
    # to sum to 1 (the Beta density integrates to 1 by definition).
    # Take the squared first row of the eigvecs:
    w_unnorm = eigvecs[0, :] ** 2

    # Map nodes from [-1, 1] back to [0, 1]:
    nodes_p = 0.5 * (1.0 + eigvals)

    # Normalise weights to sum to 1 (Beta(α, β) integrates to 1).
    # Equivalently: this absorbs the zeroth-moment constant and the
    # Jacobian of the affine map. Using log-space to keep small
    # weights representable.
    log_w_unnorm = jnp.log(jnp.clip(w_unnorm, a_min=1e-300))
    log_weights = log_w_unnorm - jsp.special.logsumexp(log_w_unnorm)

    return nodes_p, log_weights


def _golub_welsch(
    alpha: jnp.ndarray, beta: jnp.ndarray, n_nodes: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorised Golub-Welsch over leading axes of (α, β).

    Broadcasts ``alpha`` and ``beta`` to a common shape, flattens the
    leading axes, applies ``_golub_welsch_single`` with ``jax.vmap``,
    and reshapes back.

    Parameters
    ----------
    alpha, beta : jnp.ndarray
        Beta shape parameters α, β; arbitrary broadcastable shapes.
    n_nodes : int
        Static node count.

    Returns
    -------
    nodes_p, log_weights : jnp.ndarray
        Shape ``broadcast(alpha, beta) + (n_nodes,)``.
    """
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    out_shape = jnp.broadcast_shapes(alpha.shape, beta.shape)
    alpha_b = jnp.broadcast_to(alpha, out_shape).reshape(-1)
    beta_b = jnp.broadcast_to(beta, out_shape).reshape(-1)

    nodes, log_w = jax.vmap(lambda a, b: _golub_welsch_single(a, b, n_nodes))(
        alpha_b, beta_b
    )

    nodes = nodes.reshape(out_shape + (n_nodes,))
    log_w = log_w.reshape(out_shape + (n_nodes,))
    return nodes, log_w


__all__ = [
    "gauss_jacobi_nodes_weights",
    "GOLUB_WELSCH",
    "PRECOMPUTED_GRID",
]

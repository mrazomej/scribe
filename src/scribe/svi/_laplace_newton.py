"""Pure-JAX Newton kernel for the Poisson-LogNormal MAP per cell.

Implements the inner Newton iteration of the Laplace-approximation
inference scheme for PLN (and its capture-anchored variant). No
NumPyro dependency — the kernel can be unit-tested by comparing to
``scipy.optimize.minimize`` on a small joint log-density.

Mathematical background
-----------------------
The PLN per-cell joint log-density (with the biology-informed capture
anchor active, dropping additive constants in ``(x, η)``) is

.. math::
    f(x, \\eta)
        = u^\\top (x - \\eta \\mathbf{1})
        - \\mathbf{1}^\\top \\exp(x - \\eta \\mathbf{1})
        - \\tfrac{1}{2}(x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)
        - \\tfrac{1}{2}(\\eta - \\eta_{\\text{anchor}})^2 / \\sigma_M^2

where ``x ∈ R^G`` is the latent log-rate vector, ``η`` is the per-cell
capture offset (in log-rate space), ``μ ∈ R^G`` is the decoder bias,
``Σ = W W^\\top + \\text{diag}(d)`` is the low-rank-plus-diagonal
prior covariance (``W ∈ R^{G \\times k}``, ``d ∈ R^G_{>0}``), and
``η_anchor = log(M_0) - log(L_c)`` is the per-cell anchor with prior
scale ``σ_M``. ``u ∈ R^G`` is the observed count vector for the cell.

Gradient (component-wise):

.. math::
    \\partial_x f &= u - \\exp(x - \\eta) - \\Sigma^{-1}(x - \\mu) \\\\
    \\partial_\\eta f &= -\\mathbf{1}^\\top u + \\mathbf{1}^\\top \\exp(x - \\eta)
                       - (\\eta - \\eta_{\\text{anchor}})/\\sigma_M^2

Hessian blocks:

.. math::
    H_{xx} &= -\\text{diag}(\\exp(x - \\eta)) - \\Sigma^{-1} \\\\
    H_{x\\eta} &= \\exp(x - \\eta) \\quad (\\text{column vector}) \\\\
    H_{\\eta\\eta} &= -\\mathbf{1}^\\top \\exp(x - \\eta) - 1/\\sigma_M^2

The full ``(G+1) × (G+1)`` Hessian is symmetric negative definite
(PLN is globally log-concave, see Chiquet et al. 2018), so ``-H`` is
symmetric positive definite. Letting

.. math::
    A = -H_{xx} = \\text{diag}(\\lambda) + \\Sigma^{-1},
    \\qquad \\lambda_g = \\exp(x_g - \\eta),

and applying Woodbury to ``Σ^{-1}`` and then again to ``A``, both the
linear solve ``A^{-1} y`` and the log-determinant ``log det(A)`` are
computable in ``O(G·k + k^3)`` per cell — the cost of one ``k × k``
Cholesky and a few low-rank multiplies. See :func:`_woodbury_factors`
and :func:`_solve_A` below.

Newton step
-----------
At each iterate, solve ``-H δ = ∇f`` for ``δ = (δ_x, δ_η)`` via block
Schur complement:

.. math::
    s &= -H_{\\eta\\eta} - H_{x\\eta}^\\top A^{-1} H_{x\\eta} \\\\
    \\delta_\\eta &= (g_\\eta - H_{x\\eta}^\\top A^{-1} g_x) / s \\\\
    \\delta_x &= A^{-1}(g_x - H_{x\\eta} \\, \\delta_\\eta)

The same ``A^{-1}`` factorization is reused for all three solves
needed per Newton step.

Damping
-------
A small Tikhonov term ``damping`` is added to the diagonal of ``A``
(``λ_g + 1/d_g + damping``) and to ``-H_{ηη}`` to stabilise Newton
when ``A`` is ill-conditioned. The default ``damping = 1e-4`` is
small enough not to bias the MAP for well-conditioned problems and
large enough to avoid Cholesky failures in pathological corners
(e.g., near-zero ``d`` or extremely sparse cells).

API
---
- :func:`newton_step_joint(x, eta, u, mu, W, d, eta_anchor, sigma_M, damping)`
  — single joint Newton step. Returns ``(x_new, eta_new, grad_inf_norm)``.
- :func:`laplace_newton_loop(...)` — `lax.scan` over a fixed number of
  Newton steps; returns ``(x_map, eta_map, final_grad_norm, log_det_neg_H)``.
- :func:`laplace_newton_batch(...)` — ``jax.vmap`` of the above over
  cells. Inputs are per-cell ``(G,)``-shaped arrays; outputs include
  per-cell ``log_det_neg_H`` for the Laplace correction term in the
  outer ELBO.
- :func:`newton_step_x_only`, :func:`laplace_newton_loop_x_only`,
  :func:`laplace_newton_batch_x_only` — variants for fits without a
  capture anchor (eliminates the ``η`` block). Same ``A`` Woodbury
  structure.

The kernel is JIT-traceable and ``vmap``-compatible: all loops use
``lax.scan`` (data-independent iteration count) so vectorising over
cells is a one-line ``jax.vmap``.

References
----------
- Chiquet, Mariadassou, Robin (2018). *Variational inference for
  probabilistic Poisson PCA*. Annals of Applied Statistics.
- The PLN R package https://github.com/PLN-team/PLNmodels uses the
  same variational-EM scheme: outer SGD on global parameters,
  inner Newton (or quasi-Newton) on per-cell latents.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


# Clamp on (x - eta) before exponentiation. exp(30) ≈ 1e13, so the
# Poisson rate cannot exceed that; in float32 this leaves ~25 bits of
# headroom before overflow at exp(88). Same bound used by the PLN
# likelihood in pln.py so the Newton kernel and the model agree on the
# rate clamp.
_LOG_RATE_MIN = -30.0
_LOG_RATE_MAX = 30.0


# =====================================================================
# Woodbury helpers
# =====================================================================


def _woodbury_factors(
    W: jnp.ndarray,
    d: jnp.ndarray,
    log_rate: jnp.ndarray,
    damping: float,
):
    """Pre-compute Woodbury factors for ``A = diag(λ + 1/d + damping) − V K⁻¹ V'``.

    Caller passes ``log_rate = x - η`` (already centred); we
    exponentiate inside (with the standard clamp) so callers do not
    have to remember the safety net. All factors are reused across
    the three linear solves performed in one Newton step (gradient
    direction + Schur complement).

    Parameters
    ----------
    W : jnp.ndarray, shape (G, k)
        Decoder loadings.
    d : jnp.ndarray, shape (G,)
        Diagonal residual variance (positive).
    log_rate : jnp.ndarray, shape (G,)
        Current ``x - η`` value at the iterate.
    damping : float
        Tikhonov damping added to the diagonal of ``A``.

    Returns
    -------
    dict with keys:
        ``m_inv``: shape (G,), ``1/(λ_g + 1/d_g + damping)``.
        ``V``: shape (G, k), ``(1/d) ⊙ W``.
        ``L_K``: shape (k, k), Cholesky factor of ``K = I + W' diag(1/d) W``.
        ``L_S``: shape (k, k), Cholesky of ``S = K − V' diag(m_inv) V``.
        ``log_det_K``: scalar.
        ``log_det_S``: scalar.
    """
    rate = jnp.exp(jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX))
    inv_d = 1.0 / d
    # M_diag = λ + 1/d + damping (positive). m_inv is its reciprocal.
    m_diag = rate + inv_d + damping
    m_inv = 1.0 / m_diag

    # V = diag(1/d) W (the same matrix that appears in Woodbury for Σ⁻¹).
    V = inv_d[:, None] * W

    # K = I_k + W' diag(1/d) W (symmetric positive definite). Used both
    # for Σ⁻¹ and for the Schur complement S.
    k = W.shape[1]
    K = jnp.eye(k) + W.T @ V

    # S = K − V' diag(m_inv) V. Symmetric positive definite when A is
    # positive definite (Sylvester / Schur complement).
    S = K - V.T @ (m_inv[:, None] * V)

    # Cholesky factors. We compute log-dets here once so subsequent
    # log-det queries are free.
    L_K = jnp.linalg.cholesky(K)
    L_S = jnp.linalg.cholesky(S)
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
    log_det_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L_S)))

    return {
        "m_inv": m_inv,
        "V": V,
        "L_K": L_K,
        "L_S": L_S,
        "log_det_K": log_det_K,
        "log_det_S": log_det_S,
    }


def _solve_A(factors: dict, y: jnp.ndarray) -> jnp.ndarray:
    """Compute ``A⁻¹ y`` using the pre-built Woodbury factors.

    Uses the Woodbury identity for ``A = M̃ - V K⁻¹ V'``:

    .. math::
        A^{-1} y = m_inv \\odot y +
                   m_inv \\odot V (S^{-1} V^\\top (m_inv \\odot y))

    where ``S = K - V' diag(m_inv) V``. Cost: ``O(G·k + k^3)`` per
    solve (one Cholesky-back-substitute on the ``k × k`` matrix
    ``S``). Same algorithm as ``cho_solve`` from SciPy.

    Parameters
    ----------
    factors : dict
        Output of :func:`_woodbury_factors`.
    y : jnp.ndarray, shape (G,)

    Returns
    -------
    jnp.ndarray, shape (G,)
    """
    m_inv = factors["m_inv"]
    V = factors["V"]
    L_S = factors["L_S"]

    # First term: m_inv ⊙ y.
    base = m_inv * y
    # Correction: m_inv ⊙ (V (S⁻¹ V' (m_inv ⊙ y))).
    rhs = V.T @ base
    z = jax.scipy.linalg.cho_solve((L_S, True), rhs)
    return base + m_inv * (V @ z)


def _log_det_A(factors: dict) -> jnp.ndarray:
    """Compute ``log det A`` from the Woodbury factors.

    Using the matrix-determinant lemma on ``A = M̃ - V K⁻¹ V'``:

    .. math::
        \\log \\det A = \\sum_g \\log(λ_g + 1/d_g + \\text{damping})
                      + \\log \\det S - \\log \\det K

    All three terms are pre-computed in :func:`_woodbury_factors`.

    Returns
    -------
    jnp.ndarray, scalar.
    """
    m_inv = factors["m_inv"]
    log_det_K = factors["log_det_K"]
    log_det_S = factors["log_det_S"]
    log_det_M = -jnp.sum(jnp.log(m_inv))  # = sum log(m_diag)
    return log_det_M + log_det_S - log_det_K


# =====================================================================
# Newton step (joint x, eta)
# =====================================================================


def newton_step_joint(
    x: jnp.ndarray,
    eta: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    damping: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One joint Newton step on ``(x, η)``.

    Solves ``-H δ = ∇f`` via block Schur complement, reusing one
    Cholesky factorization of ``S = K - V' diag(m_inv) V`` for all
    three internal solves.

    Parameters
    ----------
    x : jnp.ndarray, shape (G,)
        Current log-rate iterate.
    eta : jnp.ndarray, scalar
        Current capture offset iterate.
    u : jnp.ndarray, shape (G,)
        Observed counts for this cell.
    mu : jnp.ndarray, shape (G,)
        Decoder bias.
    W : jnp.ndarray, shape (G, k)
        Decoder loadings.
    d : jnp.ndarray, shape (G,)
        Diagonal residual variance.
    eta_anchor : jnp.ndarray, scalar
        Anchor value for ``η``: ``log M_0 - log L_c``.
    sigma_M : float
        Anchor scale (``> 0``).
    damping : float, default 1e-4
        Tikhonov damping on ``A`` and on the ``η``-block scalar.

    Returns
    -------
    x_new : jnp.ndarray, shape (G,)
    eta_new : jnp.ndarray, scalar
    grad_inf_norm : jnp.ndarray, scalar
        Pre-step ``L∞`` norm of the gradient ``(∇_x f, ∇_η f)``,
        useful for convergence diagnostics.
    """
    # Centre once and exponentiate (with clamp) to get the Poisson rate.
    log_rate = x - eta
    rate = jnp.exp(jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX))

    # Gradient.
    factors = _woodbury_factors(W, d, log_rate, damping)
    # Σ⁻¹ (x - μ) — use the inner Woodbury for Σ to keep cost low.
    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )
    g_x = u - rate - sigma_inv_diff
    g_eta = -jnp.sum(u) + jnp.sum(rate) - (eta - eta_anchor) / (sigma_M**2)

    # Block Schur complement on -H.
    A_inv_g_x = _solve_A(factors, g_x)
    H_x_eta = rate  # ∂²f/∂x∂η, shape (G,)
    # Schur complement scalar s = -H_ηη - H_xη' A⁻¹ H_xη + damping
    # = (1' rate + 1/σ²) - rate' A⁻¹ rate + damping.
    A_inv_H_xeta = _solve_A(factors, H_x_eta)
    s = (
        jnp.sum(rate)
        + 1.0 / (sigma_M**2)
        - jnp.dot(H_x_eta, A_inv_H_xeta)
        + damping
    )
    delta_eta = (g_eta - jnp.dot(H_x_eta, A_inv_g_x)) / s
    delta_x = A_inv_g_x - A_inv_H_xeta * delta_eta

    grad_inf_norm = jnp.maximum(
        jnp.max(jnp.abs(g_x)), jnp.abs(g_eta)
    )
    return x + delta_x, eta + delta_eta, grad_inf_norm


def laplace_newton_loop(
    x_init: jnp.ndarray,
    eta_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    n_iters: int,
    damping: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps on ``(x, η)`` for one cell.

    Uses :func:`jax.lax.scan` so the iteration count is a static
    integer (compatible with ``vmap`` over cells). Convergence is
    surfaced via the final gradient norm rather than early exit; the
    caller is expected to set ``n_iters`` large enough that
    quadratic convergence has plateau'd (5–10 iterations from a warm
    start is typically more than enough for a log-concave posterior).

    Parameters
    ----------
    x_init, eta_init : jnp.ndarray
        Newton starting point. Shape ``(G,)`` and scalar.
    u, mu, W, d, eta_anchor, sigma_M : see :func:`newton_step_joint`.
    n_iters : int
        Number of Newton iterations to run. Static (Python int).
    damping : float

    Returns
    -------
    x_map : jnp.ndarray, shape (G,)
    eta_map : jnp.ndarray, scalar
    final_grad_norm : jnp.ndarray, scalar
    log_det_neg_H : jnp.ndarray, scalar
        ``log det(-H)`` evaluated at the MAP. The Schur-complement
        decomposition gives ``log det(-H) = log det(A) + log s`` where
        ``s = -H_ηη - H_xη' A⁻¹ H_xη``. Required by the Laplace
        correction term in the outer ELBO.
    """

    def step(carry, _):
        x_, eta_, _grad = carry
        x_new, eta_new, grad_norm = newton_step_joint(
            x_, eta_, u, mu, W, d, eta_anchor, sigma_M, damping
        )
        return (x_new, eta_new, grad_norm), None

    init = (x_init, eta_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, eta_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    # log det(-H) at the MAP. Recompute the Woodbury factors at the
    # final iterate; this is one extra factorization per cell but the
    # caller needs the determinant for the Laplace ELBO, so we cannot
    # skip it.
    final_log_rate = x_final - eta_final
    final_rate = jnp.exp(
        jnp.clip(final_log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    )
    factors = _woodbury_factors(W, d, final_log_rate, damping)
    log_det_A = _log_det_A(factors)
    A_inv_rate = _solve_A(factors, final_rate)
    s = (
        jnp.sum(final_rate)
        + 1.0 / (sigma_M**2)
        - jnp.dot(final_rate, A_inv_rate)
        + damping
    )
    log_det_neg_H = log_det_A + jnp.log(s)

    return x_final, eta_final, final_grad, log_det_neg_H


# Vectorise over cells: each per-cell input has a leading batch axis,
# global parameters (mu, W, d, sigma_M, damping) are shared.
laplace_newton_batch = jax.vmap(
    laplace_newton_loop,
    in_axes=(0, 0, 0, None, None, None, 0, None, None, None),
)


# =====================================================================
# x-only variant (no capture anchor)
# =====================================================================


def newton_step_x_only(
    x: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    damping: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x`` only (no capture anchor).

    The η block of the joint Hessian disappears entirely, so this is
    the standard PLN MAP inference (Chiquet et al. 2018). Newton step
    is just ``δ = A⁻¹ g`` where ``A = diag(exp(x) + 1/d) - V K⁻¹ V'``.
    """
    rate = jnp.exp(jnp.clip(x, _LOG_RATE_MIN, _LOG_RATE_MAX))
    factors = _woodbury_factors(W, d, x, damping)

    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )
    g_x = u - rate - sigma_inv_diff

    delta_x = _solve_A(factors, g_x)
    grad_inf_norm = jnp.max(jnp.abs(g_x))
    return x + delta_x, grad_inf_norm


def laplace_newton_loop_x_only(
    x_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps on ``x`` for one cell (no capture)."""

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only(
            x_, u, mu, W, d, damping
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    factors = _woodbury_factors(W, d, x_final, damping)
    log_det_neg_H = _log_det_A(factors)
    return x_final, final_grad, log_det_neg_H


laplace_newton_batch_x_only = jax.vmap(
    laplace_newton_loop_x_only,
    in_axes=(0, 0, None, None, None, None, None),
)


__all__ = [
    "newton_step_joint",
    "newton_step_x_only",
    "laplace_newton_loop",
    "laplace_newton_loop_x_only",
    "laplace_newton_batch",
    "laplace_newton_batch_x_only",
]

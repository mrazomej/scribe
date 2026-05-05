"""Pure-JAX Newton kernels for LNM Laplace inference.

Two parallel variants, dispatched in the engine on ``d_mode``:

* :func:`newton_step_z` / :func:`laplace_newton_loop_z` —
  ``d_mode='low_rank'``. Newton over the factor scores
  ``z ∈ ℝ^k``. The Hessian is ``−H_z = I + W^T M(z) W`` where
  ``M(z) = N · (diag(ρ) − ρρ^T)`` is the multinomial Fisher
  matrix at ``ρ = softmax_full(μ + W z)`` restricted to ALR
  coordinates. Cost ``O(k³)`` per cell, no Woodbury needed.

* :func:`newton_step_y_alr` / :func:`laplace_newton_loop_y_alr` —
  ``d_mode='learned'``. Newton over ``y ∈ ℝ^{G−1}`` (ALR logits).
  The Hessian is

  .. math::
      -H_y = N \\bigl(\\mathrm{diag}(\\rho) - \\rho\\rho^\\top\\bigr)
             + \\Sigma^{-1},
      \\qquad \\Sigma = W W^\\top + \\mathrm{diag}(d).

  Reuses scribe's Woodbury machinery on
  ``Σ⁻¹ = D − D W K⁻¹ W^T D`` (``D = diag(1/d)``). The diagonal
  ``N\\,\\rho`` part folds straight into PLN's ``m_diag`` slot
  (``N\\,ρ_g + 1/d_g``); the rank-1 ``−N\\,\\rho\\rho^T`` correction
  is handled by one Sherman-Morrison step on top of the Woodbury
  inverse. Cost ``O(G \\cdot k + k³)`` per cell, same scaling as
  PLN Laplace.

Both variants:

* Run as a JIT-compiled ``lax.scan`` over a fixed number of
  Newton iterations (cleaner under ``vmap`` than a ``while_loop``).
* Apply Tikhonov damping (``λ_z = damping`` on the
  ``z``-block; ``λ_d`` on the diagonal of ``A`` in the
  ``y_alr``-branch) for solver stability. The damping is **not**
  applied to the Laplace correction's ``log det(-H)`` (see
  :func:`laplace_log_det_neg_H_z` /
  :func:`laplace_log_det_neg_H_y_alr`).
* Apply a step-size cap (``MAX_STEP``) to prevent runaway
  iterates when the cell's data is degenerate (e.g. a single
  count concentrated in one gene).

The PLN Newton kernel in :mod:`scribe.laplace._newton_pln` is the
template for these — read it first if you want to understand the
shared scaffolding.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


# Numerical safety bounds shared across both variants.
_LOGITS_MIN = -30.0
_LOGITS_MAX = 30.0
_MAX_STEP = 5.0


# =====================================================================
# Helpers shared by both branches
# =====================================================================


def _augment_logits(
    y_alr: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
) -> jnp.ndarray:
    """Insert a zero at the reference position to recover ``G``-dim logits.

    LNM convention: the ALR map drops the reference gene whose
    logit is conventionally fixed at zero. To compute multinomial
    probabilities we need the full ``G``-vector; this helper builds
    one by inserting a zero at ``alr_reference_idx``.

    Parameters
    ----------
    y_alr : jnp.ndarray, shape ``(..., G-1)``
        ALR logits.
    alr_reference_idx : int
        Zero-based index of the reference gene in the *full*
        ``G``-gene order.
    n_genes : int
        Total number of genes ``G``.

    Returns
    -------
    jnp.ndarray, shape ``(..., G)``
        Full logits with a zero at the reference position.
    """
    leading = y_alr.shape[:-1]
    full = jnp.zeros(leading + (n_genes,), dtype=y_alr.dtype)
    other_idx = jnp.asarray(
        [g for g in range(n_genes) if g != int(alr_reference_idx)]
    )
    full = full.at[..., other_idx].set(y_alr)
    return full


def _multinomial_p_alr(
    y_alr: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
) -> jnp.ndarray:
    """Compute ``p_alr = softmax(y_full)[non-ref]`` for the multinomial.

    Both Newton variants need the multinomial probability *vector*
    restricted to the ALR coordinates (i.e., the ``G-1`` non-
    reference probabilities). The reference gene's probability is
    ``1 - sum(p_alr)`` and is not stored separately — it falls out
    of the softmax normalization.

    Parameters
    ----------
    y_alr : jnp.ndarray, shape ``(G-1,)``
        Current ALR logits.
    alr_reference_idx : int
    n_genes : int

    Returns
    -------
    jnp.ndarray, shape ``(G-1,)``
        Softmax probabilities for the non-reference genes.
    """
    full = _augment_logits(
        jnp.clip(y_alr, _LOGITS_MIN, _LOGITS_MAX),
        alr_reference_idx,
        n_genes,
    )
    p_full = jax.nn.softmax(full, axis=-1)
    other_idx = jnp.asarray(
        [g for g in range(n_genes) if g != int(alr_reference_idx)]
    )
    return p_full[..., other_idx]


# =====================================================================
# Variant A: Newton over z (d_mode='low_rank')
# =====================================================================


def newton_step_z(
    z: jnp.ndarray,
    u_alr: jnp.ndarray,
    n_total: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
    damping: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One Newton step on ``z`` for the LNM ``low_rank`` Laplace path.

    The per-cell log-density is

    .. math::
        f(z) = u_{\\text{alr}}^\\top (\\mu + W z)
               - N \\cdot \\mathrm{logsumexp}(y_{\\text{full}})
               - \\tfrac{1}{2} z^\\top z,

    where ``y_full`` is ``mu + W z`` augmented with a zero at the
    reference position. Gradient and Hessian:

    .. math::
        \\nabla_z f = W^\\top (u_{\\text{alr}} - N\\,\\rho_{\\text{alr}}) - z,
        \\qquad
        -H_z = W^\\top M_{\\text{alr}} W + I,

    with ``M_alr = N · (diag(ρ_alr) - ρ_alr ρ_alr^T)``. The
    Newton step solves ``-H_z δ = ∇f`` via a ``k × k`` Cholesky.

    Parameters
    ----------
    z : jnp.ndarray, shape ``(k,)``
        Current iterate.
    u_alr : jnp.ndarray, shape ``(G-1,)``
        Counts in the non-reference genes.
    n_total : jnp.ndarray, scalar
        Total count for the cell (``sum(u_full)``).
    mu : jnp.ndarray, shape ``(G-1,)``
        Decoder bias in ALR coordinates.
    W : jnp.ndarray, shape ``(G-1, k)``
        Decoder loadings.
    alr_reference_idx : int
        Reference gene index in the full ``G``-gene order.
    n_genes : int
        Total ``G`` (so ``G - 1 = mu.shape[0]``).
    damping : float
        Tikhonov damping added to the diagonal of ``-H_z``.
        Stabilises the solve when ``W`` is near-degenerate; does
        **not** flow into the Laplace correction's ``log det``.

    Returns
    -------
    z_new : jnp.ndarray, shape ``(k,)``
    grad_inf_norm : jnp.ndarray, scalar
        Pre-step ``L∞`` norm of ``∇_z f`` for diagnostics.
    """
    y_alr = mu + W @ z
    p_alr = _multinomial_p_alr(y_alr, alr_reference_idx, n_genes)

    # Gradient: ∇f = W^T (u_alr - N ρ_alr) - z.
    grad_z = W.T @ (u_alr - n_total * p_alr) - z

    # Hessian: -H_z = W^T M_alr W + I, with
    #   M_alr v = N (ρ_alr ⊙ v - ρ_alr (ρ_alr^T v))
    # We construct -H_z explicitly because k is small (≤ ~50).
    Wp = W * p_alr[:, None]                                  # (G-1, k)
    rhs = W.T @ Wp                                           # = W^T diag(ρ) W, (k, k)
    Wp_sum = W.T @ p_alr                                     # (k,)
    M_W = n_total * (rhs - jnp.outer(Wp_sum, Wp_sum))        # = W^T M_alr W
    k = z.shape[0]
    neg_H = M_W + (1.0 + damping) * jnp.eye(k, dtype=z.dtype)

    # Solve  neg_H @ delta = grad_z  via Cholesky.
    L = jnp.linalg.cholesky(neg_H)
    delta = jax.scipy.linalg.cho_solve((L, True), grad_z)

    # Step-size cap protects against pathological cells.
    step_norm = jnp.max(jnp.abs(delta))
    scale = jnp.minimum(1.0, _MAX_STEP / jnp.maximum(step_norm, 1e-12))
    delta = delta * scale

    grad_inf_norm = jnp.max(jnp.abs(grad_z))
    return z + delta, grad_inf_norm


def laplace_newton_loop_z(
    z_init: jnp.ndarray,
    u_alr: jnp.ndarray,
    n_total: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
    n_iters: int,
    damping: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps via :func:`lax.scan`.

    ``lax.scan`` over a fixed iteration count keeps the function
    ``vmap``-friendly. Returns the final iterate and gradient norm.
    """
    def body(carry, _x):
        z = carry
        z_new, gn = newton_step_z(
            z, u_alr, n_total, mu, W,
            alr_reference_idx, n_genes, damping,
        )
        return z_new, gn

    z_final, gn_history = jax.lax.scan(
        body, z_init, jnp.arange(n_iters)
    )
    # Final gradient norm = grad norm at the *last* step (not the
    # post-last gradient), which is a slight approximation but
    # sufficient for the diagnostic.
    return z_final, gn_history[-1]


# Vmap over the cell axis for batched processing inside the engine.
#
# The (G,)-dependent integer args (``alr_reference_idx``, ``n_genes``,
# ``n_iters``, ``damping``) are broadcast (axis None); only the per-
# cell arrays (``z``, ``u_alr``, ``n_total``) carry a leading axis.
laplace_newton_batch_z = jax.vmap(
    laplace_newton_loop_z,
    in_axes=(0, 0, 0, None, None, None, None, None, None),
)


def laplace_log_det_neg_H_z(
    z: jnp.ndarray,
    u_alr: jnp.ndarray,
    n_total: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
) -> jnp.ndarray:
    """``log det(-H_z)`` at the MAP, with **no** damping.

    Used by the outer Laplace ELBO so that gradients flow through
    ``W`` (the ``mu`` block doesn't appear in ``-H_z``). The MAP
    iterate ``z`` should be passed in stop-gradient form by the
    caller (envelope theorem on ``z``); the live globals supply the
    direct gradient through ``W``.

    Parameters
    ----------
    z : jnp.ndarray, shape ``(k,)``
        Per-cell MAP (``stop_gradient``ed by the caller).
    u_alr, n_total : data terms (used only via ``ρ_alr``).
    mu, W : live globals.
    alr_reference_idx, n_genes : ALR bookkeeping.

    Returns
    -------
    jnp.ndarray, scalar.
    """
    y_alr = mu + W @ z
    p_alr = _multinomial_p_alr(y_alr, alr_reference_idx, n_genes)
    Wp = W * p_alr[:, None]
    rhs = W.T @ Wp
    Wp_sum = W.T @ p_alr
    M_W = n_total * (rhs - jnp.outer(Wp_sum, Wp_sum))
    k = z.shape[0]
    neg_H = M_W + jnp.eye(k, dtype=z.dtype)
    sign, logdet = jnp.linalg.slogdet(neg_H)
    # ``-H_z = I + W^T M_alr W`` is SPD (proof: I is SPD; M_alr is
    # PSD as a covariance-like matrix). slogdet returns sign=+1.
    return logdet


laplace_log_det_neg_H_batch_z = jax.vmap(
    laplace_log_det_neg_H_z,
    in_axes=(0, 0, 0, None, None, None, None),
)


# =====================================================================
# Variant B: Newton over y_alr (d_mode='learned')
# =====================================================================
#
# This branch directly mirrors the PLN Newton kernel structure. The
# Hessian
#
#     -H_y = N (diag(ρ_alr) - ρ_alr ρ_alr^T) + Σ⁻¹
#
# decomposes as
#
#     -H_y = A_y - rank-1 correction,
#
#     A_y  = diag(N ρ_alr + 1/d)
#            - diag(1/d) W (I + W^T diag(1/d) W)⁻¹ W^T diag(1/d).
#
# A_y has the same low-rank-plus-diagonal structure that PLN's A
# does, just with ``rate`` replaced by ``N ρ_alr``. So we can reuse
# scribe.laplace._newton_pln's Woodbury factorization helpers and
# add one Sherman-Morrison step for the ``-N ρ_alr ρ_alr^T``
# correction.


def _woodbury_factors_lnm(
    W: jnp.ndarray,
    d: jnp.ndarray,
    n_total: jnp.ndarray,
    p_alr: jnp.ndarray,
    damping: float,
) -> dict:
    """Pre-compute Woodbury factors for ``A_y``.

    Mirrors ``_woodbury_factors`` in ``_newton_pln`` but with
    ``m_diag = N · ρ_alr + 1/d + damping`` instead of
    ``λ_g + 1/d_g + damping``. Same returned shape; same downstream
    ``_solve_A`` / ``_log_det_A`` logic applies.
    """
    inv_d = 1.0 / d
    m_diag = n_total * p_alr + inv_d + damping
    m_inv = 1.0 / m_diag
    V = inv_d[:, None] * W                                   # (G-1, k)
    # Inner Woodbury: K = I_k + W^T D W, with D = diag(1/d).
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ V
    L_K = jnp.linalg.cholesky(K)
    # Outer Woodbury: S = K + V^T diag(m_inv) V (?)
    # Actually we want S = K_outer for inverting A. From the PLN
    # derivation:
    #   A = diag(m) - V K⁻¹ V^T,
    #   A⁻¹ = diag(1/m) + diag(1/m) V S⁻¹ V^T diag(1/m),
    #   S   = K - V^T diag(1/m) V.
    # Build S and its Cholesky.
    Vm = V * m_inv[:, None]                                  # (G-1, k)
    S = K - V.T @ Vm
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


def _solve_A_lnm(factors: dict, y: jnp.ndarray) -> jnp.ndarray:
    """Solve ``A_y x = y`` using Woodbury factors. Same as PLN's _solve_A."""
    m_inv = factors["m_inv"]
    V = factors["V"]
    L_S = factors["L_S"]
    base = m_inv * y
    rhs = V.T @ base
    z = jax.scipy.linalg.cho_solve((L_S, True), rhs)
    return base + m_inv * (V @ z)


def _log_det_A_lnm(factors: dict) -> jnp.ndarray:
    """``log det A_y`` from Woodbury factors. Same identity as PLN."""
    m_inv = factors["m_inv"]
    log_det_K = factors["log_det_K"]
    log_det_S = factors["log_det_S"]
    log_det_M = -jnp.sum(jnp.log(m_inv))
    return log_det_M + log_det_S - log_det_K


def newton_step_y_alr(
    y: jnp.ndarray,
    u_alr: jnp.ndarray,
    n_total: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
    damping: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One Newton step on ``y_alr`` for the LNM ``learned`` Laplace path.

    Per-cell log-density (in ALR coordinates):

    .. math::
        f(y) = u_{\\text{alr}}^\\top y - N\\,\\mathrm{logsumexp}(y_{\\text{full}})
               - \\tfrac{1}{2}(y - \\mu)^\\top \\Sigma^{-1}(y - \\mu),

    with ``Σ = WW^T + diag(d)``. Gradient:

    .. math::
        \\nabla_y f = u_{\\text{alr}} - N\\,\\rho_{\\text{alr}}
                     - \\Sigma^{-1}(y - \\mu).

    The Newton step solves ``-H_y δ = ∇f`` where
    ``-H_y = N(diag(ρ) - ρρ^T) + Σ⁻¹``. We rewrite this as
    ``A_y - N\\,ρρ^T`` and solve via Sherman-Morrison on top of
    the Woodbury inverse of ``A_y``:

    .. math::
        (-H_y)^{-1} v = A_y^{-1} v + \\frac{(A_y^{-1} \\rho_{\\text{alr}})\\;
                       (\\rho_{\\text{alr}}^\\top A_y^{-1} v)}
                       {1/N - \\rho_{\\text{alr}}^\\top A_y^{-1} \\rho_{\\text{alr}}}.

    Cost per step: ``O(G\\cdot k + k³)`` — same as PLN.
    """
    p_alr = _multinomial_p_alr(y, alr_reference_idx, n_genes)
    factors = _woodbury_factors_lnm(W, d, n_total, p_alr, damping)

    # Compute Σ⁻¹ (y - mu) via the inner Woodbury (without the
    # multinomial-Fisher diagonal addition). This mirrors PLN's
    # gradient assembly for the prior-pull term.
    inv_d = 1.0 / d
    diff = y - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )

    grad_y = u_alr - n_total * p_alr - sigma_inv_diff

    # Sherman-Morrison: (A - N ρρ^T)^-1 g = A^-1 g + α (A^-1 ρ)
    # where α = (ρ^T A^-1 g) / (1/N - ρ^T A^-1 ρ).
    A_inv_g = _solve_A_lnm(factors, grad_y)
    A_inv_rho = _solve_A_lnm(factors, p_alr)
    rho_A_inv_rho = jnp.dot(p_alr, A_inv_rho)
    rho_A_inv_g = jnp.dot(p_alr, A_inv_g)
    # Floor the SM denominator to avoid numerical issues for cells
    # whose ρ is degenerate (single dominant gene). With the
    # Sherman-Morrison form, ``denom -> 1/N - 0+`` so we always
    # have at least ``1/N - eps`` magnitude.
    denom = jnp.maximum(1.0 / n_total - rho_A_inv_rho, 1e-12)
    alpha = rho_A_inv_g / denom
    delta = A_inv_g + alpha * A_inv_rho

    # Step-size cap.
    step_norm = jnp.max(jnp.abs(delta))
    scale = jnp.minimum(1.0, _MAX_STEP / jnp.maximum(step_norm, 1e-12))
    delta = delta * scale

    grad_inf_norm = jnp.max(jnp.abs(grad_y))
    return y + delta, grad_inf_norm


def laplace_newton_loop_y_alr(
    y_init: jnp.ndarray,
    u_alr: jnp.ndarray,
    n_total: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
    n_iters: int,
    damping: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps via ``lax.scan``."""
    def body(carry, _x):
        y = carry
        y_new, gn = newton_step_y_alr(
            y, u_alr, n_total, mu, W, d,
            alr_reference_idx, n_genes, damping,
        )
        return y_new, gn

    y_final, gn_history = jax.lax.scan(
        body, y_init, jnp.arange(n_iters)
    )
    return y_final, gn_history[-1]


laplace_newton_batch_y_alr = jax.vmap(
    laplace_newton_loop_y_alr,
    in_axes=(0, 0, 0, None, None, None, None, None, None, None),
)


def laplace_log_det_neg_H_y_alr(
    y: jnp.ndarray,
    u_alr: jnp.ndarray,
    n_total: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
) -> jnp.ndarray:
    """``log det(-H_y)`` at the MAP, with **no** damping.

    Uses the matrix-determinant lemma:

    .. math::
        \\log \\det(-H_y) = \\log \\det(A_y)
                          + \\log\\bigl(1 - N\\,\\rho_{\\text{alr}}^\\top
                                       A_y^{-1} \\rho_{\\text{alr}}\\bigr).

    The first term comes from PLN's Woodbury identity (``log det M
    + log det S - log det K``); the second is the rank-1
    correction's contribution.

    Returns
    -------
    jnp.ndarray, scalar.

    Notes
    -----
    The ``1 - N ρ^T A_y⁻¹ ρ`` correction is bounded above by
    ``1 - 0 = 1`` (because the Sherman-Morrison residual cannot
    produce a negative-definite ``-H_y``). It is bounded below by
    a small positive number (Schur-complement positivity); we
    floor at 1e-30 to keep ``log`` finite for pathological cells.
    """
    p_alr = _multinomial_p_alr(y, alr_reference_idx, n_genes)
    factors = _woodbury_factors_lnm(W, d, n_total, p_alr, damping=0.0)
    log_det_A = _log_det_A_lnm(factors)
    A_inv_rho = _solve_A_lnm(factors, p_alr)
    correction = 1.0 - n_total * jnp.dot(p_alr, A_inv_rho)
    return log_det_A + jnp.log(jnp.maximum(correction, 1e-30))


laplace_log_det_neg_H_batch_y_alr = jax.vmap(
    laplace_log_det_neg_H_y_alr,
    in_axes=(0, 0, 0, None, None, None, None, None),
)


__all__ = [
    # Variant A
    "newton_step_z",
    "laplace_newton_loop_z",
    "laplace_newton_batch_z",
    "laplace_log_det_neg_H_z",
    "laplace_log_det_neg_H_batch_z",
    # Variant B
    "newton_step_y_alr",
    "laplace_newton_loop_y_alr",
    "laplace_newton_batch_y_alr",
    "laplace_log_det_neg_H_y_alr",
    "laplace_log_det_neg_H_batch_y_alr",
]

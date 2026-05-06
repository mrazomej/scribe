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
    # whose ρ is degenerate (single dominant gene). The
    # mathematical lower bound (joint log-concavity of the LNM
    # posterior) is `1/N - rho^T A^-1 rho > 0` strictly, but in
    # float32 the subtraction can lose precision when the two
    # terms are individually large and nearly equal. We use a
    # *relative* floor rather than absolute: ``denom`` cannot be
    # less than 1% of ``1/N``. For typical N=10⁴ this floors at
    # ``1e-6``, four orders of magnitude tighter than the
    # original ``1e-12`` and well above float32 catastrophic
    # cancellation. The relative form scales correctly across
    # cells with different totals.
    one_over_N = 1.0 / n_total
    denom = jnp.maximum(one_over_N - rho_A_inv_rho, 1e-2 * one_over_N)
    alpha = rho_A_inv_g / denom
    delta = A_inv_g + alpha * A_inv_rho

    # Step-size cap (matches the value used elsewhere in this
    # module — see ``_MAX_STEP``). Limits a single Newton step's
    # L∞ to 5 in log-odds space; the SM denominator floor above
    # is the primary defence against pathological-cell blow-ups.
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


# =====================================================================
# LNMVCP capture-anchor: scalar Newton on per-cell eta
# =====================================================================
#
# In LNMVCP the per-cell observed total ``u_T = sum(counts)`` is
# conditioned on, and the NB-on-totals likelihood becomes
#
#     log NB(u_T | r_T, mu_T * exp(-eta_c)) - 0.5 (eta_c - eta_anchor_c)^2 / sigma_M^2,
#
# where ``eta_c = -log(p_capture_c)`` is the per-cell capture latent
# and ``eta_anchor_c = log_M_0 - log(L_c)`` is the biology-informed
# prior mean. Crucially this term is independent of ``z``, and the
# multinomial likelihood used by Variants A/B above is independent
# of ``eta``, so the joint Hessian on ``(z, eta)`` is **block
# diagonal**:
#
#     -H_zz : (Variant A or B)            -H_z_eta = 0
#     -H_eta_z = 0                        -H_eta_eta : scalar
#
# Newton over ``eta`` is therefore a scalar Newton per cell that
# can be run in parallel with (independently of) the z- or y_alr-
# Newton. The Laplace correction factorises into the sum of two
# log-determinants.


def _nb_eta_grad_and_hessian(
    eta: jnp.ndarray,
    u_T: jnp.ndarray,
    r_T: jnp.ndarray,
    mu_T: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Closed-form ``∂/∂η`` and ``∂²/∂η²`` of ``log NB(u_T | r_T, mu_T·exp(-η))``.

    Derivation (with ``v = r_T + mu_T·exp(-η)``):

        ∂/∂η  log NB = r_T (mu_T·exp(-η) - u_T) / v
        ∂²/∂η² log NB = -r_T · mu_T·exp(-η) · (r_T + u_T) / v²

    The Hessian is strictly negative on the interior — the term is
    log-concave in η — which makes the joint
    ``(z, eta)`` log-density log-concave whenever the prior on η is
    log-concave (the TruncN prior at ``low=0`` is log-concave on
    its support).

    Returns
    -------
    grad, hess : jnp.ndarray
        Both scalar per cell (same shape as ``eta``).
    """
    # exp(-eta) with the same float32 safety bound the kernel uses
    # for the Poisson rate elsewhere.
    exp_neg_eta = jnp.exp(jnp.clip(-eta, _LOGITS_MIN, _LOGITS_MAX))
    rate_T = mu_T * exp_neg_eta             # mean of NB at this η
    v = r_T + rate_T                         # NB normaliser denominator
    grad = r_T * (rate_T - u_T) / v
    hess = -r_T * rate_T * (r_T + u_T) / (v * v)
    return grad, hess


def newton_step_eta(
    eta: jnp.ndarray,
    u_T: jnp.ndarray,
    r_T: jnp.ndarray,
    mu_T: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    damping: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One Newton step on the per-cell capture-offset ``eta``.

    The per-cell log-density (in η, holding all else fixed) is

        f(η) = log NB(u_T | r_T, mu_T · exp(-η))
               - 0.5 (η - η_anchor)² / σ_M².

    The Newton step is just a scalar division: ``η_new = η - g/H``
    where ``g = ∂f/∂η`` and ``H = ∂²f/∂η²``. With the Tikhonov
    damping convention, the denominator is ``-H + damping`` so the
    step is well-defined even when ``H`` is near zero (which can
    happen for cells with very few counts).

    A floor at ``η_new ≥ 1e-3`` enforces the TruncatedNormal(low=0)
    constraint of the LNMVCP capture prior — analogous to the
    projection used in :func:`newton_step_joint` for PLN.

    Parameters
    ----------
    eta : jnp.ndarray, scalar
        Current iterate.
    u_T : jnp.ndarray, scalar
        Observed total counts for this cell.
    r_T, mu_T : jnp.ndarray, scalar
        Globals (NB shape and mean of total mRNA per cell).
    eta_anchor : jnp.ndarray, scalar
        Per-cell prior mean ``log_M_0 - log(L_c)``.
    sigma_M : float
        Prior scale on ``eta_capture``.
    damping : float
        Tikhonov damping added to the negative Hessian.

    Returns
    -------
    eta_new : jnp.ndarray, scalar
    grad_inf_norm : jnp.ndarray, scalar
        Pre-step ``|∂f/∂η|`` for diagnostics.
    """
    nb_grad, nb_hess = _nb_eta_grad_and_hessian(eta, u_T, r_T, mu_T)
    prior_grad = -(eta - eta_anchor) / (sigma_M * sigma_M)
    prior_hess = -1.0 / (sigma_M * sigma_M)

    grad = nb_grad + prior_grad
    neg_H = -(nb_hess + prior_hess) + damping  # > 0 (data + prior + damping)

    delta = grad / neg_H
    # Step-size cap shared with the other LNM variants.
    delta = jnp.clip(delta, -_MAX_STEP, _MAX_STEP)

    eta_new = eta + delta
    # Project back to TruncN(low=0) feasible region.
    eta_new = jnp.maximum(eta_new, 1e-3)

    return eta_new, jnp.abs(grad)


def laplace_newton_loop_eta(
    eta_init: jnp.ndarray,
    u_T: jnp.ndarray,
    r_T: jnp.ndarray,
    mu_T: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    n_iters: int,
    damping: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` scalar Newton steps on η via ``lax.scan``.

    The η Newton converges quickly (5–10 iterations from any
    sensible warm start, e.g. ``eta_anchor``) because the scalar
    log-density is strictly log-concave and exhibits quadratic
    Newton convergence near the MAP.
    """
    def body(carry, _x):
        eta = carry
        eta_new, gn = newton_step_eta(
            eta, u_T, r_T, mu_T, eta_anchor, sigma_M, damping
        )
        return eta_new, gn

    eta_final, gn_history = jax.lax.scan(
        body, eta_init, jnp.arange(n_iters)
    )
    return eta_final, gn_history[-1]


laplace_newton_batch_eta = jax.vmap(
    laplace_newton_loop_eta,
    in_axes=(0, 0, None, None, 0, None, None, None),
)


def laplace_log_det_neg_H_eta(
    eta: jnp.ndarray,
    u_T: jnp.ndarray,
    r_T: jnp.ndarray,
    mu_T: jnp.ndarray,
    sigma_M: float,
) -> jnp.ndarray:
    """``log det(-H_η) = log(-H_η)`` for the scalar η-block.

    Used by the outer Laplace ELBO. damping=0 is hardcoded so the
    correction reflects the *true* posterior curvature, not the
    inner-Newton solver's regularised one. Live globals (``r_T``,
    ``mu_T``) flow gradients into the outer Adam step so the
    NB-on-totals globals get a direct gradient from the Laplace
    correction.

    Returns
    -------
    jnp.ndarray, scalar
        ``log(neg_H)``.
    """
    _, nb_hess = _nb_eta_grad_and_hessian(eta, u_T, r_T, mu_T)
    neg_H = -(nb_hess - 1.0 / (sigma_M * sigma_M))
    # Floor for numerical safety at very low total counts.
    return jnp.log(jnp.maximum(neg_H, 1e-30))


laplace_log_det_neg_H_batch_eta = jax.vmap(
    laplace_log_det_neg_H_eta,
    in_axes=(0, 0, None, None, None),
)


__all__ = [
    # Variant A (z-space, d_mode='low_rank')
    "newton_step_z",
    "laplace_newton_loop_z",
    "laplace_newton_batch_z",
    "laplace_log_det_neg_H_z",
    "laplace_log_det_neg_H_batch_z",
    # Variant B (y_alr-space, d_mode='learned')
    "newton_step_y_alr",
    "laplace_newton_loop_y_alr",
    "laplace_newton_batch_y_alr",
    "laplace_log_det_neg_H_y_alr",
    "laplace_log_det_neg_H_batch_y_alr",
    # LNMVCP capture-anchor extension (scalar η per cell)
    "newton_step_eta",
    "laplace_newton_loop_eta",
    "laplace_newton_batch_eta",
    "laplace_log_det_neg_H_eta",
    "laplace_log_det_neg_H_batch_eta",
]

"""Pure-JAX Newton kernel for the NB-LogNormal MAP per cell.

Implements the inner Newton iteration of the Laplace-approximation
inference scheme for NB-LogNormal (and its capture-anchored variant).
Mirrors :mod:`scribe.laplace._newton_pln` exactly in structure: same
Woodbury machinery, same Schur block decomposition, same step-size cap
and damping safeguards. The only differences are in the *data-side*
contributions to the gradient and Hessian, which we now derive from
the Negative-Binomial likelihood instead of Poisson.

Mathematical background
-----------------------
The NB-LogNormal per-cell joint log-density (with the biology-informed
capture anchor active, dropping additive constants in (𝑥, η)) is

    f(𝑥, η) = Σ_g log NB(u_g | exp(x_g − η), r_g)
              − ½ (𝑥 − μ)ᵀ Σ⁻¹ (𝑥 − μ)
              − ½ (η − η_anchor)² / σ_M²

where 𝑥 ∈ ℝ^G is the latent log-rate vector (the decoder output
``y_log_rate``), η is the per-cell capture offset (in log-rate space),
μ ∈ ℝ^G is the population mean of 𝑥 (decoder bias),
Σ = 𝑊 𝑊ᵀ + diag(𝑑), η_anchor = log(M₀) − log(L_c), and
𝑟 ∈ ℝ^G_{>0} is the gene-specific NB dispersion. 𝑢 ∈ ℝ^G is the
observed count vector for the cell.

Define the per-gene success probability (NumPyro convention):

    p_g = r_g / (r_g + exp(x_g − η)).

Then the data-side gradient is

    (∂_x f)_g^data = u_g − (u_g + r_g)(1 − p_g)
    ∂_η f^data     = −Σ_g u_g + Σ_g (u_g + r_g)(1 − p_g)

(note that (u_g + r_g)(1 − p_g) is the conditional posterior mean of
the latent Gamma rate given u_g — it reduces to rate_g in the Poisson
limit r_g → ∞).

The data-side Hessian diagonal is

    a_g = (u_g + r_g) p_g (1 − p_g),

and entering the same block structure as PLN:

    H_xx^data = −diag(𝑎)
    H_xη^data = +𝑎          (column vector)
    H_ηη^data = −Σ_g a_g.

Adding the prior contributions (Σ⁻¹ on 𝑥, 1/σ_M² on η), the full
(G+1) × (G+1) Hessian is symmetric negative definite (NB-LogNormal is
per-cell log-concave: see ``paper/_nb_lognormal.qmd`` §8.4 and
@eq-nbln-loglik-curvature for the proof). Letting

    𝐴 = −H_xx = diag(𝑎) + Σ⁻¹,

and applying Woodbury to Σ⁻¹ and then again to 𝐴, both the linear
solve 𝐴⁻¹ 𝑦 and the log-determinant log det(𝐴) are computable in
O(G·k + k³) per cell — identical to PLN. The only difference is that
the diagonal of 𝐴 is built from a_g rather than λ_g = exp(x_g − η).

Stable computation of p and a
-----------------------------
We avoid materialising exp(𝑥 − η) directly by computing p in log-space
via softplus identities:

    δ_g       = (x_g − η) − log r_g
    log p_g   = −softplus(δ_g)
    log(1−p_g) = δ_g − softplus(δ_g)

This matches the parameterisation used by
:class:`scribe.stats.distributions.LogMeanNegativeBinomial`, so the
likelihood and the Newton kernel agree to numerical precision. The
diagonal a_g = (u_g + r_g) p_g (1 − p_g) is bounded by (u_g + r_g)/4
for any log_rate.

Newton step
-----------
Identical to PLN's joint solve, with 𝑎 replacing rate:

    s   = Σ_g a_g + 1/σ_M² − 𝑎ᵀ 𝐴⁻¹ 𝑎
    δ_η = (g_η + 𝑎ᵀ 𝐴⁻¹ g_x) / s
    δ_x = 𝐴⁻¹ (g_x + 𝑎 δ_η)

The same 𝐴⁻¹ factorisation is reused for all three solves needed per
Newton step.

API
---
- :func:`newton_step_joint(x, eta, u, mu, W, d, r, eta_anchor, sigma_M, damping)`
  — single joint Newton step.  Returns ``(x_new, eta_new, grad_inf_norm)``.
- :func:`laplace_newton_loop(...)` — ``lax.scan`` over a fixed number
  of Newton steps; returns ``(x_map, eta_map, final_grad_norm,
  log_det_neg_H)``.
- :func:`laplace_newton_batch(...)` — ``jax.vmap`` of the above over
  cells.
- :func:`newton_step_x_only`, :func:`laplace_newton_loop_x_only`,
  :func:`laplace_newton_batch_x_only` — variants for fits without a
  capture anchor (eliminates the ``η`` block).

Cross-references
----------------
- ``paper/_nb_lognormal.qmd`` §8.4 (log-concavity), §8.6 (Newton +
  Woodbury), §8.7 (algorithm summary).
- ``scribe.laplace._newton_pln`` for the Poisson-LogNormal analogue.
- ``scribe.stats.distributions.LogMeanNegativeBinomial`` for the
  matching observation distribution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


# Sanity bounds on (x - eta).  As in :mod:`_newton_pln`, this clamp
# protects float32 arithmetic; here it is even less load-bearing than
# in PLN because ``a_g = (u+r) p (1-p)`` saturates as |x-η| → ∞ (p or
# 1-p vanishes).  The bound matches the NBLN likelihood's clamp in
# ``models/components/likelihoods/nbln.py``.
_LOG_RATE_MIN = -50.0
_LOG_RATE_MAX = 50.0

# Floor on r_g to avoid log(0) and divide-by-zero.
_R_EPS = 1e-6


# =====================================================================
# Stable NB factor computation
# =====================================================================


def _nb_factors(
    log_rate: jnp.ndarray,
    u: jnp.ndarray,
    r: jnp.ndarray,
) -> dict:
    """Compute NB-likelihood quantities used by gradient and Hessian.

    Given ``log_rate = x - η`` and per-gene NB parameters ``(u, r)``,
    return the success probability ``p_g``, its complement
    ``one_minus_p = 1 - p_g``, and the data-side Hessian diagonal
    ``a_g = (u_g + r_g) p_g (1 - p_g)``.

    All quantities are computed in log-space via softplus identities
    so the result is well-defined for any finite ``log_rate``.

    Parameters
    ----------
    log_rate : jnp.ndarray, shape (G,)
        Effective log-rate ``x - η`` (already centred by capture).
    u : jnp.ndarray, shape (G,)
        Observed counts for this cell.
    r : jnp.ndarray, shape (G,)
        Gene dispersion, strictly positive.

    Returns
    -------
    dict with keys:
        ``p`` : success probability ``r/(r+exp(log_rate))``, shape (G,).
        ``one_minus_p`` : ``1 - p``, shape (G,).
        ``a`` : data-Hessian diagonal ``(u+r)·p·(1-p)``, shape (G,).
    """
    log_rate_clipped = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    log_r = jnp.log(jnp.maximum(r, _R_EPS))
    delta = log_rate_clipped - log_r
    sp = jax.nn.softplus(delta)
    log_p = -sp
    log_1mp = delta - sp
    p = jnp.exp(log_p)
    one_minus_p = jnp.exp(log_1mp)
    a = (u + r) * p * one_minus_p
    return {"p": p, "one_minus_p": one_minus_p, "a": a}


# =====================================================================
# Woodbury helpers
# =====================================================================


def _woodbury_factors(
    W: jnp.ndarray,
    d: jnp.ndarray,
    a_diag: jnp.ndarray,
    damping: float,
) -> dict:
    """Pre-compute Woodbury factors for ``A = diag(a + 1/d + damping) − V K⁻¹ V'``.

    Mirrors :func:`scribe.laplace._newton_pln._woodbury_factors`
    exactly except that the diagonal contribution is ``a_diag``
    (the NB Hessian diagonal) rather than ``exp(log_rate)``.  Caller
    is responsible for computing ``a_diag = (u+r)·p·(1-p)`` upstream.

    All factors are reused across the three linear solves performed
    in one Newton step (gradient direction + Schur complement).

    Parameters
    ----------
    W : jnp.ndarray, shape (G, k)
        Decoder loadings.
    d : jnp.ndarray, shape (G,)
        Diagonal residual variance (positive).
    a_diag : jnp.ndarray, shape (G,)
        Data-Hessian diagonal contribution ``a = (u+r) p (1-p)``.
    damping : float
        Tikhonov damping added to the diagonal of ``A``.

    Returns
    -------
    dict with keys:
        ``m_inv`` : shape (G,), ``1/(a_g + 1/d_g + damping)``.
        ``V`` : shape (G, k), ``(1/d) ⊙ W``.
        ``L_K`` : shape (k, k), Cholesky factor of ``K = I + W' diag(1/d) W``.
        ``L_S`` : shape (k, k), Cholesky of ``S = K − V' diag(m_inv) V``.
        ``log_det_K`` : scalar.
        ``log_det_S`` : scalar.
    """
    inv_d = 1.0 / d
    m_diag = a_diag + inv_d + damping
    m_inv = 1.0 / m_diag

    V = inv_d[:, None] * W

    k = W.shape[1]
    K = jnp.eye(k) + W.T @ V

    S = K - V.T @ (m_inv[:, None] * V)

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

    Identical to :func:`scribe.laplace._newton_pln._solve_A` — the
    Woodbury inversion does not depend on the form of ``a_diag``.
    """
    m_inv = factors["m_inv"]
    V = factors["V"]
    L_S = factors["L_S"]

    base = m_inv * y
    rhs = V.T @ base
    z = jax.scipy.linalg.cho_solve((L_S, True), rhs)
    return base + m_inv * (V @ z)


def _log_det_A(factors: dict) -> jnp.ndarray:
    """Compute ``log det A`` from Woodbury factors via matrix-determinant lemma."""
    m_inv = factors["m_inv"]
    log_det_K = factors["log_det_K"]
    log_det_S = factors["log_det_S"]
    log_det_M = -jnp.sum(jnp.log(m_inv))
    return log_det_M + log_det_S - log_det_K


def laplace_log_det_neg_H(
    x_map: jnp.ndarray,
    eta_map: Optional[jnp.ndarray],
    u: jnp.ndarray,
    r: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    sigma_M: float,
) -> jnp.ndarray:
    """``log det(-H)`` at the MAP, with **no Tikhonov damping**.

    Used by the outer Laplace ELBO so that gradients flow through the
    determinant correction into the globals ``(W, d, r, μ)``.

    Capture-anchor on (``eta_map is not None``):

        log det(−H) = log det 𝐴 + log s,

    where s = Σ_g a_g + 1/σ_M² − 𝑎ᵀ 𝐴⁻¹ 𝑎 is the Schur scalar.

    Parameters
    ----------
    x_map : jnp.ndarray, shape (G,)
        MAP log-rate.
    eta_map : jnp.ndarray, scalar or None
        MAP capture offset (None when capture anchor is off).
    u : jnp.ndarray, shape (G,)
        Counts for this cell.
    r : jnp.ndarray, shape (G,)
        Gene dispersion at which the determinant is evaluated.
    W, d : jnp.ndarray
        Globals at which the determinant is evaluated.
    sigma_M : float
        Capture-anchor prior scale.  Ignored when ``eta_map is None``.

    Returns
    -------
    jnp.ndarray, scalar.
    """
    log_rate = x_map - eta_map if eta_map is not None else x_map
    nb = _nb_factors(log_rate, u, r)
    factors = _woodbury_factors(W, d, nb["a"], damping=0.0)
    log_det_A = _log_det_A(factors)
    if eta_map is None:
        return log_det_A

    a = nb["a"]
    a_inv_A = _solve_A(factors, a)
    s = jnp.maximum(
        jnp.sum(a) + 1.0 / (sigma_M**2) - jnp.dot(a, a_inv_A),
        1e-30,
    )
    return log_det_A + jnp.log(s)


# Vmapped versions: per-cell axes are 0 for x_map/eta_map/u; globals
# shared.  Capture-active path additionally accepts a per-cell
# ``sigma_M`` (axis 0) so each cell can carry its own truncated-normal
# prior scale on ``η`` — used by the SVI-informative-prior cascade
# where each cell's η scale comes from the SVI posterior.  For the
# x-only path (no capture), ``sigma_M`` is unused inside the function
# and the vmap keeps it shared.
laplace_log_det_neg_H_batch = jax.vmap(
    laplace_log_det_neg_H,
    in_axes=(0, 0, 0, None, None, None, 0),
)
laplace_log_det_neg_H_batch_x_only = jax.vmap(
    laplace_log_det_neg_H,
    in_axes=(0, None, 0, None, None, None, None),
)


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
    r: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One joint Newton step on ``(x, η)`` for NB-LogNormal.

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
        Decoder bias (population mean of x under the prior).
    W : jnp.ndarray, shape (G, k)
        Decoder loadings.
    d : jnp.ndarray, shape (G,)
        Diagonal residual variance.
    r : jnp.ndarray, shape (G,)
        Gene dispersion (positive).
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
        Pre-step ``L∞`` norm of the gradient ``(∇_x f, ∇_η f)``.
    """
    log_rate = x - eta
    nb = _nb_factors(log_rate, u, r)
    a = nb["a"]
    one_minus_p = nb["one_minus_p"]

    factors = _woodbury_factors(W, d, a, damping)

    # Σ⁻¹ (x - μ): use the inner Woodbury for Σ to keep cost low.
    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )

    # Data-side gradient.
    # ∂L_data/∂x_g = u_g - (u_g + r_g)(1 - p_g)
    # ∂L_data/∂η   = -Σu + Σ(u + r)(1 - p)
    expected_count = (u + r) * one_minus_p
    g_x_data = u - expected_count
    g_eta_data = -jnp.sum(u) + jnp.sum(expected_count)

    g_x = g_x_data - sigma_inv_diff
    g_eta = g_eta_data - (eta - eta_anchor) / (sigma_M**2)

    # Block Schur complement on -H. Solve  -H @ delta = grad f.
    #
    # In block form  -H = [[A, B], [B^T, C]] with
    #   A = -H_xx  = diag(a) + Σ⁻¹
    #   B = -H_xη  = -a            (column vector; sign from η entering as -η)
    #   C = -H_ηη  = sum(a) + 1/σ_M²
    #
    # Schur back-substitution:
    #   δ_η = (g_η - B^T A⁻¹ g_x) / (C - B^T A⁻¹ B)
    #   δ_x = A⁻¹ (g_x - B δ_η)
    #
    # With B = -a:  B^T A⁻¹ g_x = -a^T A⁻¹ g_x ;  C - B^T A⁻¹ B = C - a^T A⁻¹ a.
    A_inv_g_x = _solve_A(factors, g_x)
    a_inv_A = _solve_A(factors, a)
    s = jnp.sum(a) + 1.0 / (sigma_M**2) - jnp.dot(a, a_inv_A) + damping
    delta_eta = (g_eta + jnp.dot(a, A_inv_g_x)) / s
    delta_x = A_inv_g_x + a_inv_A * delta_eta

    grad_inf_norm = jnp.maximum(
        jnp.max(jnp.abs(g_x)), jnp.abs(g_eta)
    )
    # Step-size cap: if the proposed step is huge in either block,
    # scale the whole step down to ``max_step``.  Default 5.0 matches
    # the PLN-era safeguard.  Exposed as a config knob
    # (``LaplaceConfig.newton_max_step``) — raise it (e.g. 20-50) when
    # the rigid-translation gauge is structurally pinned (cascade
    # freeze on η) and Newton can take larger steps safely.
    step_norm = jnp.maximum(jnp.max(jnp.abs(delta_x)), jnp.abs(delta_eta))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    delta_x = delta_x * scale
    delta_eta = delta_eta * scale
    eta_new = eta + delta_eta
    # TruncatedNormal(low=0) constraint on the capture offset, matching
    # PLN's projection.
    eta_new = jnp.maximum(eta_new, 1e-3)
    return x + delta_x, eta_new, grad_inf_norm


def laplace_newton_loop(
    x_init: jnp.ndarray,
    eta_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps on ``(x, η)`` for one cell.

    Uses :func:`jax.lax.scan` so the iteration count is a static
    integer (compatible with ``vmap`` over cells).  Quadratic
    convergence is guaranteed by §8.4 log-concavity, so 5–10 iterations
    suffice from a warm start (``x_init = mu``, ``eta_init`` modest).

    Returns
    -------
    x_map : jnp.ndarray, shape (G,)
    eta_map : jnp.ndarray, scalar
    final_grad_norm : jnp.ndarray, scalar
    log_det_neg_H : jnp.ndarray, scalar
        ``log det(-H)`` evaluated at the MAP for the Laplace ELBO.
    """

    def step(carry, _):
        x_, eta_, _grad = carry
        x_new, eta_new, grad_norm = newton_step_joint(
            x_, eta_, u, mu, W, d, r, eta_anchor, sigma_M, damping, max_step,
        )
        return (x_new, eta_new, grad_norm), None

    init = (x_init, eta_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, eta_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    # log det(-H) at the MAP. Recompute factors at the converged
    # iterate; this is one extra factorisation per cell but the caller
    # needs the determinant for the Laplace ELBO so we cannot skip it.
    final_log_rate = x_final - eta_final
    nb_final = _nb_factors(final_log_rate, u, r)
    a_final = nb_final["a"]
    factors = _woodbury_factors(W, d, a_final, damping)
    log_det_A = _log_det_A(factors)
    A_inv_a = _solve_A(factors, a_final)
    s = (
        jnp.sum(a_final)
        + 1.0 / (sigma_M**2)
        - jnp.dot(a_final, A_inv_a)
        + damping
    )
    log_det_neg_H = log_det_A + jnp.log(s)

    return x_final, eta_final, final_grad, log_det_neg_H


# Vectorise over cells: each per-cell input has a leading batch axis,
# global parameters (mu, W, d, r, damping) are shared.  ``sigma_M``
# (slot 8) is per-cell so each cell can carry its own ``η`` prior
# scale — used by the SVI-informative-prior cascade.  The scalar-anchor
# code path passes a broadcast ``jnp.full((B,), sigma_M)``.
laplace_newton_batch = jax.vmap(
    laplace_newton_loop,
    in_axes=(0, 0, 0, None, None, None, None, 0, 0, None, None, None),
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
    r: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x`` only (no capture anchor).

    The η block disappears; this is the standard NB-LogNormal MAP
    inference.  Newton step is just ``δ = A⁻¹ g`` where
    ``A = diag(a + 1/d) - V K⁻¹ V'``.
    """
    nb = _nb_factors(x, u, r)
    a = nb["a"]
    one_minus_p = nb["one_minus_p"]

    factors = _woodbury_factors(W, d, a, damping)

    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )
    g_x = u - (u + r) * one_minus_p - sigma_inv_diff

    delta_x = _solve_A(factors, g_x)
    grad_inf_norm = jnp.max(jnp.abs(g_x))
    # Step-size cap (configurable via LaplaceConfig.newton_max_step).
    step_norm = jnp.max(jnp.abs(delta_x))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    return x + delta_x * scale, grad_inf_norm


def laplace_newton_loop_x_only(
    x_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps on ``x`` for one cell (no capture)."""

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only(
            x_, u, mu, W, d, r, damping, max_step,
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    nb_final = _nb_factors(x_final, u, r)
    factors = _woodbury_factors(W, d, nb_final["a"], damping)
    log_det_neg_H = _log_det_A(factors)
    return x_final, final_grad, log_det_neg_H


laplace_newton_batch_x_only = jax.vmap(
    laplace_newton_loop_x_only,
    in_axes=(0, 0, None, None, None, None, None, None, None),
)


# =====================================================================
# x-only-with-offset variant (capture frozen at NBVCP MAP)
# =====================================================================
#
# When the SVI-cascade `informative_priors_freeze` includes "eta", the
# per-cell capture offset is pinned at the SVI source's MAP and is no
# longer a Newton-optimized latent.  The model effectively becomes
# x-only with a fixed per-cell offset:
#
#     log_rate_cg = x_cg − η_offset_c.
#
# The Newton step is identical to the no-capture x-only path EXCEPT the
# data-side NB factors are evaluated at the offset-shifted log_rate.
# No η block in the Hessian (η is constant), so no rigid-translation
# gauge degeneracy — the joint (x, η) inverse collapses to a clean
# x-only inverse.


def newton_step_x_only_offset(
    x: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_offset: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x`` with fixed ``η_offset``.

    The NB factors are evaluated at the shifted log-rate
    ``log_rate = x − η_offset``; the MVN prior on ``x`` is unchanged
    (the prior is on the latent log-rate, not on the observable
    log_rate).  No η Newton block; no rigid-translation gauge.

    Parameters
    ----------
    x : jnp.ndarray, shape (G,)
        Current iterate of the latent log-rate.
    u : jnp.ndarray, shape (G,)
        Observed counts for this cell.
    mu, W, d, r
        Globals (same as the no-offset x-only path).
    eta_offset : jnp.ndarray, scalar
        Fixed per-cell capture offset.  When zero, this path is
        identical to :func:`newton_step_x_only`.
    damping : float
        Tikhonov damping on ``A``.

    Returns
    -------
    x_new : jnp.ndarray, shape (G,)
    grad_inf_norm : jnp.ndarray, scalar
    """
    log_rate = x - eta_offset
    nb = _nb_factors(log_rate, u, r)
    a = nb["a"]
    one_minus_p = nb["one_minus_p"]

    factors = _woodbury_factors(W, d, a, damping)

    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )
    # NB gradient at the shifted log_rate; sign on x is +1 (chain rule
    # through ``log_rate = x − η_offset`` has ∂ log_rate / ∂ x = 1).
    g_x = u - (u + r) * one_minus_p - sigma_inv_diff

    delta_x = _solve_A(factors, g_x)
    grad_inf_norm = jnp.max(jnp.abs(g_x))
    # Step-size cap (configurable via LaplaceConfig.newton_max_step).
    # The x-only-with-offset path has no rigid-translation gauge null
    # direction (η is fixed); raising max_step to 20-50 is safe and
    # often necessary for x to converge in finite outer iterations on
    # high-count datasets.
    step_norm = jnp.max(jnp.abs(delta_x))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    return x + delta_x * scale, grad_inf_norm


def laplace_newton_loop_x_only_offset(
    x_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_offset: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``n_iters`` Newton steps on ``x`` with fixed ``η_offset`` for one cell.

    Returns ``(x_map, final_grad_norm, log_det_neg_H)``.  The log-det
    is the negative-x-Hessian at the converged iterate, with NB factors
    evaluated at the shifted log_rate.
    """

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only_offset(
            x_, u, mu, W, d, r, eta_offset, damping, max_step,
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    log_rate_final = x_final - eta_offset
    nb_final = _nb_factors(log_rate_final, u, r)
    factors = _woodbury_factors(W, d, nb_final["a"], damping)
    log_det_neg_H = _log_det_A(factors)
    return x_final, final_grad, log_det_neg_H


# Vmapped: per-cell axes are 0 for x_init, u, eta_offset; globals
# (mu, W, d, r, n_iters, damping) shared.
laplace_newton_batch_x_only_offset = jax.vmap(
    laplace_newton_loop_x_only_offset,
    in_axes=(0, 0, None, None, None, None, 0, None, None, None),
)


def laplace_log_det_neg_H_x_only_offset(
    x_map: jnp.ndarray,
    eta_offset: jnp.ndarray,
    u: jnp.ndarray,
    r: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """``log det(−H_x)`` at the MAP for the x-only-with-offset path.

    Used by the outer Laplace ELBO so the determinant correction
    propagates gradients through the globals ``(W, d, r, μ)``.  No η
    block — log det is just ``log det A`` where
    ``A = diag(a) + Σ⁻¹`` with ``a = (u+r) p (1−p)`` evaluated at
    ``log_rate = x − η_offset``.
    """
    log_rate = x_map - eta_offset
    nb = _nb_factors(log_rate, u, r)
    factors = _woodbury_factors(W, d, nb["a"], damping=0.0)
    return _log_det_A(factors)


laplace_log_det_neg_H_batch_x_only_offset = jax.vmap(
    laplace_log_det_neg_H_x_only_offset,
    in_axes=(0, 0, 0, None, None, None),
)


# =====================================================================
# Per-block gradient-norm split for diagnostics
# =====================================================================


def nbln_grad_split(
    x: jnp.ndarray,
    eta: Optional[jnp.ndarray],
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_anchor: Optional[jnp.ndarray] = None,
    sigma_M: Optional[float] = None,
) -> dict:
    """Compute the gradient of the per-cell log-density, split per block.

    Useful as a diagnostic: ``norm_x`` should drop to <1e-3 within
    5–10 Newton iterations.  ``norm_eta`` (when capture is on) tells
    you whether the capture anchor block is converging or stuck.

    Returns
    -------
    dict with keys ``g_x`` (shape (G,)), ``g_eta`` (scalar or None),
    ``norm_x`` (scalar), ``norm_eta`` (scalar or None).
    """
    log_rate = x - eta if eta is not None else x
    nb = _nb_factors(log_rate, u, r)
    one_minus_p = nb["one_minus_p"]

    inv_d = 1.0 / d
    diff = x - mu
    # Σ⁻¹ (x - μ) via inner Woodbury.
    k = W.shape[1]
    K = jnp.eye(k) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )

    expected_count = (u + r) * one_minus_p
    g_x = u - expected_count - sigma_inv_diff
    out = {
        "g_x": g_x,
        "norm_x": jnp.max(jnp.abs(g_x)),
    }

    if eta is not None:
        if eta_anchor is None or sigma_M is None:
            raise ValueError(
                "eta_anchor and sigma_M required when eta is provided"
            )
        g_eta = (
            -jnp.sum(u)
            + jnp.sum(expected_count)
            - (eta - eta_anchor) / (sigma_M**2)
        )
        out["g_eta"] = g_eta
        out["norm_eta"] = jnp.abs(g_eta)
    else:
        out["g_eta"] = None
        out["norm_eta"] = None

    return out


# Vmappable helpers analogous to ``_newton_pln.pln_grad_split_batch`` and
# ``_newton_pln.pln_grad_x_only_norm_batch``: same per-block grad-norm
# diagnostic, used by the generic Laplace driver to surface a
# composition-style ``max/p99/med`` triplet per block in the progress
# display.  The dict-returning :func:`nbln_grad_split` above does not
# vmap cleanly because of the optional ``eta`` branch, so these
# fixed-shape variants are provided alongside it.


def _nbln_grad_split_with_eta(
    x_map: jnp.ndarray,
    eta_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-block ``L∞`` grad norms at the joint NBLN Newton MAP.

    Returns ``(grad_inf_norm_x, grad_inf_norm_eta)`` for the per-cell
    joint log-density evaluated at ``(x_map, eta_map)``.  Same role
    as :func:`scribe.laplace._newton_pln.pln_grad_split` but with the
    NB-Hessian-conditional gradient ``u - (u+r)(1-p)`` replacing the
    Poisson form ``u - rate``.
    """
    log_rate = x_map - eta_map
    nb = _nb_factors(log_rate, u, r)
    one_minus_p = nb["one_minus_p"]

    inv_d = 1.0 / d
    diff = x_map - mu
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )

    expected_count = (u + r) * one_minus_p
    g_x = u - expected_count - sigma_inv_diff
    g_eta = (
        -jnp.sum(u)
        + jnp.sum(expected_count)
        - (eta_map - eta_anchor) / (sigma_M ** 2)
    )
    return jnp.max(jnp.abs(g_x)), jnp.abs(g_eta)


nbln_grad_split_batch = jax.vmap(
    _nbln_grad_split_with_eta,
    in_axes=(0, 0, 0, None, None, None, None, 0, 0),
)


def _nbln_grad_x_only_norm(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    """``L∞`` of ``∇_x f`` at the x-only NBLN Newton MAP (no capture)."""
    nb = _nb_factors(x_map, u, r)
    one_minus_p = nb["one_minus_p"]

    inv_d = 1.0 / d
    diff = x_map - mu
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )

    expected_count = (u + r) * one_minus_p
    g_x = u - expected_count - sigma_inv_diff
    return jnp.max(jnp.abs(g_x))


nbln_grad_x_only_norm_batch = jax.vmap(
    _nbln_grad_x_only_norm, in_axes=(0, 0, None, None, None, None)
)


def _nbln_grad_x_only_offset_norm(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_offset: jnp.ndarray,
) -> jnp.ndarray:
    """``L∞`` of ``∇_x f`` at the x-only-with-offset NBLN Newton MAP.

    Mirrors :func:`_nbln_grad_x_only_norm` but evaluates the NB factors at
    the **shifted log-rate** ``log_rate = x − η_offset``.  Required for
    the frozen-eta path: evaluating at the raw ``x`` produces
    misleadingly huge gradients (the NB likelihood saturates because
    ``exp(x)`` is enormous), even when the Newton MAP itself is healthy.
    """
    log_rate = x_map - eta_offset
    nb = _nb_factors(log_rate, u, r)
    one_minus_p = nb["one_minus_p"]

    inv_d = 1.0 / d
    diff = x_map - mu
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )

    expected_count = (u + r) * one_minus_p
    g_x = u - expected_count - sigma_inv_diff
    return jnp.max(jnp.abs(g_x))


nbln_grad_x_only_offset_norm_batch = jax.vmap(
    _nbln_grad_x_only_offset_norm,
    in_axes=(0, 0, None, None, None, None, 0),
)


# =====================================================================
# Per-cell-W legacy kernels (hierarchical gene-gene correlation, Rung 1)
# =====================================================================
#
# For the per-donor program-activity hierarchy (NB-LogNormal Rung 1), each
# cell's latent prior covariance is donor-specific:
#
#     Σ_{σ(c)} = W diag(s_{σ(c)}^2) Wᵀ + diag(d)
#              = W_eff,{σ(c)} W_eff,{σ(c)}ᵀ + diag(d),
#       with   W_eff,d = W · diag(s_d).
#
# Because cells in a mini-batch can come from different donors, the effective
# loadings differ per cell, so ``W`` must be a per-cell ``(N, G, k)`` tensor
# rather than the shared ``(G, k)`` matrix the standard batch kernels assume.
#
# Crucially, the *per-cell* Newton / log-det / grad-norm functions already
# accept a single ``(G, k)`` ``W`` -- they are vmapped over the cell axis. So
# the only change needed is to flip ``W``'s ``in_axes`` entry from ``None``
# (shared/broadcast) to ``0`` (mapped over cells). The underlying per-cell math
# is byte-identical; ``d`` stays shared (the residual diagonal is common across
# donors). These ``_percellW`` twins are used only when the hierarchy is active
# (gated in ``_obs_nbln``); the shared-``W`` kernels above remain the path for
# every non-hierarchical fit, so there is zero regression risk.
#
# (Legacy / absolute-``x`` layout only for now; the ``_decoupled`` family below
# gets its own ``_percellW`` twins in a follow-up.)

# No-capture, x-only Newton. Call:
#   (latent_init, counts, mu, W, d, r, n_newton, damping, max_step)  -> W idx 3
laplace_newton_batch_x_only_percellW = jax.vmap(
    laplace_newton_loop_x_only,
    in_axes=(0, 0, None, 0, None, None, None, None, None),
)

# Frozen-eta, x-only-with-offset Newton. Call:
#   (latent_init, counts, mu, W, d, r, eta_offset, n_newton, damping, max_step)
laplace_newton_batch_x_only_offset_percellW = jax.vmap(
    laplace_newton_loop_x_only_offset,
    in_axes=(0, 0, None, 0, None, None, 0, None, None, None),
)

# Soft-eta, joint (x, η) Newton. Call:
#   (latent_init, eta_init, counts, mu, W, d, r, eta_anchor, sigma_eta,
#    n_newton, damping, max_step)  -> W idx 4
laplace_newton_batch_percellW = jax.vmap(
    laplace_newton_loop,
    in_axes=(0, 0, 0, None, 0, None, None, 0, 0, None, None, None),
)

# log det(−H) twins. Calls:
#   x_only:        (x, None, counts, r, W, d, 1.0)           -> W idx 4
#   x_only_offset: (x, eta_offset, counts, r, W, d)          -> W idx 4
#   joint:         (x, eta, counts, r, W, d, sigma_eta)      -> W idx 4
laplace_log_det_neg_H_batch_x_only_percellW = jax.vmap(
    laplace_log_det_neg_H,
    in_axes=(0, None, 0, None, 0, None, None),
)
laplace_log_det_neg_H_batch_x_only_offset_percellW = jax.vmap(
    laplace_log_det_neg_H_x_only_offset,
    in_axes=(0, 0, 0, None, 0, None),
)
laplace_log_det_neg_H_batch_percellW = jax.vmap(
    laplace_log_det_neg_H,
    in_axes=(0, 0, 0, None, 0, None, 0),
)

# Gradient-norm twins. Calls:
#   x_only:        (x, counts, mu, W, d, r)                   -> W idx 3
#   x_only_offset: (x, counts, mu, W, d, r, eta_offset)       -> W idx 3
#   joint split:   (x, eta, counts, mu, W, d, r, eta_anchor, sigma_eta) W idx 4
nbln_grad_x_only_norm_batch_percellW = jax.vmap(
    _nbln_grad_x_only_norm, in_axes=(0, 0, None, 0, None, None)
)
nbln_grad_x_only_offset_norm_batch_percellW = jax.vmap(
    _nbln_grad_x_only_offset_norm,
    in_axes=(0, 0, None, 0, None, None, 0),
)
nbln_grad_split_batch_percellW = jax.vmap(
    _nbln_grad_split_with_eta,
    in_axes=(0, 0, 0, None, 0, None, None, 0, 0),
)


# =====================================================================
# Decoupled-layout kernels (`_other` excluded from Σ, Commit 2b)
# =====================================================================
#
# Under ``correlate_other_column=False`` with a pooled trailing
# ``_other`` column, the per-cell latent splits into:
#
#   • ``x_dev`` ∈ ℝ^{G_kept} — deviation from μ on the kept-gene axis,
#     with MVN prior  ``x_dev ~ 𝒩(0, Σ_kept)`` where
#     ``Σ_kept = W Wᵀ + diag(d)`` and ``W: (G_kept, k)``, ``d: (G_kept,)``.
#   • ``η`` ∈ ℝ — per-cell capture offset (unchanged across layouts).
#
# The pooled ``_other`` gene at position ``other_idx`` keeps participating
# in the NB observation likelihood but contributes no row to Σ — its
# log-rate is fully determined by ``μ[other_idx] − η`` (no x_dev term).
#
# Effective log-rate fed to the NB likelihood per gene:
#   kept gene ``g`` at kept position ``k_g``:  ``μ[g] + x_dev[k_g] − η``
#   ``_other`` at ``other_idx``:                ``μ[other_idx] − η``
#
# Math contract (``paper/_nb_lognormal.qmd`` §sec-nbln-decorrelate-other):
#
#   −H_{x_dev, x_dev} = diag(a_kept) + Σ_kept⁻¹   on G_kept × G_kept
#       where a_full[g] = (u_g + r_g) p_g (1−p_g) on G_obs (legacy form,
#       evaluated at the FULL G_obs log_rate); a_kept = a_full[kept_idx].
#
#   −H_{x_dev[k], η} = −a_kept[k]
#       (NB cross block; ``_other`` contributes NO coupling to x_dev.)
#
#   −H_{η, η} = Σ_{g ∈ G_obs} a_full[g] + 1/σ_M²
#       (η sees every gene's NB curvature, including ``_other``.)
#
#   Schur scalar s = Σ_g a_full[g] + 1/σ_M² − a_keptᵀ A⁻¹ a_kept
#       where A = diag(a_kept) + Σ_kept⁻¹.
#
# The Woodbury factors for A are computed via the existing
# :func:`_woodbury_factors` helper — it is shape-agnostic and Just Works
# on G_kept-sized inputs (``W: (G_kept, k)``, ``d: (G_kept,)``,
# ``a_kept: (G_kept,)``).  The same ``_solve_A`` and ``_log_det_A``
# helpers are reused without modification.
#
# Per-block gradient on G_obs (sum over all genes for η; kept-slice for
# x_dev, since x_dev[k] only enters log_rate for its own kept gene):
#
#   ∂L_data/∂x_dev[k] = u_{kept[k]} − (u + r)(1 − p)_{kept[k]}
#   ∂L_data/∂η       = − Σ_{g ∈ G_obs} [u_g − (u + r)(1 − p)_g]


def _scatter_log_rate_decoupled(
    x_dev: jnp.ndarray,
    mu: jnp.ndarray,
    eta_subtract: jnp.ndarray,
    kept_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Build full G_obs log-rate from kept-axis x_dev under decoupling.

    Returns ``log_rate`` of shape ``(G_obs,)`` with:
        log_rate[g] = mu[g] + x_dev[k] − eta_subtract  for g == kept_idx[k]
        log_rate[g] = mu[g] − eta_subtract             for g == other_idx

    Implementation: initialise the full vector at the ``_other``-style
    base (μ − η for every gene) then scatter-add ``x_dev`` at kept
    positions.  The ``_other`` entry retains the base value because no
    kept index maps to it (``other_idx ∉ kept_idx`` is enforced by
    ``AxisLayout.__post_init__``).

    Parameters
    ----------
    x_dev : jnp.ndarray, shape (G_kept,)
        Per-cell deviation on the kept-gene axis.
    mu : jnp.ndarray, shape (G_obs,)
        Per-gene log-rate baseline (lives on full obs axis under both
        layouts).
    eta_subtract : jnp.ndarray, scalar
        Per-cell capture offset.  Use ``0.0`` for the no-capture path.
    kept_idx : jnp.ndarray, shape (G_kept,)
        Integer positions of kept genes in the G_obs axis.

    Returns
    -------
    jnp.ndarray, shape (G_obs,)
    """
    base = mu - eta_subtract  # broadcast scalar over (G_obs,)
    return base.at[kept_idx].add(x_dev)


# ---------------------------------------------------------------------
# Joint Newton (soft-η, decoupled)
# ---------------------------------------------------------------------


def newton_step_joint_decoupled(
    x_dev: jnp.ndarray,
    eta: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    kept_idx: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One joint Newton step on ``(x_dev, η)`` under the deviation reparam.

    Mirrors :func:`newton_step_joint` structurally; differences:

    * Latent ``x_dev`` lives on G_kept (length-K-rank low-rank covariance).
    * NB factors are computed at the FULL G_obs log-rate via
      :func:`_scatter_log_rate_decoupled`; ``a_full`` covers every gene
      (including ``_other``) and ``a_kept = a_full[kept_idx]`` is sliced
      to drive the Woodbury factors for ``A``.
    * The MVN-prior gradient term is ``Σ_kept⁻¹ x_dev`` (zero-centred)
      instead of ``Σ⁻¹ (x − μ)`` — μ has moved into the NB likelihood
      under the reparameterisation.
    * The η Schur scalar s uses the FULL G_obs sum of ``a_full``.

    Parameters
    ----------
    x_dev : jnp.ndarray, shape (G_kept,)
    eta : jnp.ndarray, scalar
    u : jnp.ndarray, shape (G_obs,)
    mu : jnp.ndarray, shape (G_obs,)
    W : jnp.ndarray, shape (G_kept, k)
    d : jnp.ndarray, shape (G_kept,)
    r : jnp.ndarray, shape (G_obs,)
    kept_idx : jnp.ndarray, shape (G_kept,) integer
    eta_anchor : jnp.ndarray, scalar
    sigma_M : float
    damping, max_step : float

    Returns
    -------
    x_dev_new : jnp.ndarray, shape (G_kept,)
    eta_new : jnp.ndarray, scalar
    grad_inf_norm : jnp.ndarray, scalar
    """
    log_rate = _scatter_log_rate_decoupled(x_dev, mu, eta, kept_idx)
    nb = _nb_factors(log_rate, u, r)
    a_full = nb["a"]
    one_minus_p_full = nb["one_minus_p"]
    a_kept = a_full[kept_idx]

    factors = _woodbury_factors(W, d, a_kept, damping)

    # Σ_kept⁻¹ x_dev via the inner Woodbury for Σ_kept.  Zero-centred
    # prior: no μ offset to subtract.
    inv_d = 1.0 / d
    sigma_inv_xdev = inv_d * x_dev - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * x_dev)
        )
    )

    # Data-side gradient on G_obs.
    # ∂L_data/∂log_rate_g = u_g − (u_g + r_g)(1 − p_g)
    expected_count_full = (u + r) * one_minus_p_full
    g_lograte_full = u - expected_count_full
    # x_dev[k] only enters log_rate at the kept gene kept_idx[k], so
    # the chain rule gives the kept-slice directly.
    g_x_dev_data = g_lograte_full[kept_idx]
    # η enters every gene's log_rate with coefficient −1, so its data
    # gradient sums over ALL G_obs (including ``_other``).
    g_eta_data = -jnp.sum(g_lograte_full)

    g_x_dev = g_x_dev_data - sigma_inv_xdev
    g_eta = g_eta_data - (eta - eta_anchor) / (sigma_M**2)

    # Schur back-substitution (identical structure to legacy joint solve;
    # only the kept-axis substitution and full-G_obs Schur scalar differ).
    A_inv_g_x = _solve_A(factors, g_x_dev)
    a_inv_A = _solve_A(factors, a_kept)
    s = (
        jnp.sum(a_full)
        + 1.0 / (sigma_M**2)
        - jnp.dot(a_kept, a_inv_A)
        + damping
    )
    delta_eta = (g_eta + jnp.dot(a_kept, A_inv_g_x)) / s
    delta_x_dev = A_inv_g_x + a_inv_A * delta_eta

    grad_inf_norm = jnp.maximum(
        jnp.max(jnp.abs(g_x_dev)), jnp.abs(g_eta)
    )
    step_norm = jnp.maximum(jnp.max(jnp.abs(delta_x_dev)), jnp.abs(delta_eta))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    delta_x_dev = delta_x_dev * scale
    delta_eta = delta_eta * scale
    eta_new = eta + delta_eta
    eta_new = jnp.maximum(eta_new, 1e-3)
    return x_dev + delta_x_dev, eta_new, grad_inf_norm


def laplace_newton_loop_decoupled(
    x_dev_init: jnp.ndarray,
    eta_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    kept_idx: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run ``n_iters`` Newton steps on ``(x_dev, η)`` for one cell."""

    def step(carry, _):
        x_, eta_, _grad = carry
        x_new, eta_new, grad_norm = newton_step_joint_decoupled(
            x_, eta_, u, mu, W, d, r, kept_idx,
            eta_anchor, sigma_M, damping, max_step,
        )
        return (x_new, eta_new, grad_norm), None

    init = (x_dev_init, eta_init, jnp.asarray(jnp.inf, dtype=x_dev_init.dtype))
    (x_final, eta_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    # log det(-H) at the converged MAP.
    log_rate_final = _scatter_log_rate_decoupled(
        x_final, mu, eta_final, kept_idx
    )
    nb_final = _nb_factors(log_rate_final, u, r)
    a_full_final = nb_final["a"]
    a_kept_final = a_full_final[kept_idx]
    factors = _woodbury_factors(W, d, a_kept_final, damping)
    log_det_A = _log_det_A(factors)
    A_inv_a = _solve_A(factors, a_kept_final)
    s = (
        jnp.sum(a_full_final)
        + 1.0 / (sigma_M**2)
        - jnp.dot(a_kept_final, A_inv_a)
        + damping
    )
    log_det_neg_H = log_det_A + jnp.log(s)
    return x_final, eta_final, final_grad, log_det_neg_H


# Vmap over cells: per-cell x_dev_init, eta_init, u, eta_anchor, sigma_M
# carry leading batch axes; globals (mu, W, d, r, kept_idx) shared.
laplace_newton_batch_decoupled = jax.vmap(
    laplace_newton_loop_decoupled,
    in_axes=(0, 0, 0, None, None, None, None, None, 0, 0, None, None, None),
)


# ---------------------------------------------------------------------
# x-only Newton (no capture, decoupled)
# ---------------------------------------------------------------------


def newton_step_x_only_decoupled(
    x_dev: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    kept_idx: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x_dev`` only (no capture, decoupled).

    No η block; the per-cell joint reduces to the x_dev solve with NB
    factors evaluated at ``log_rate = μ + x_dev[k]`` for kept genes and
    ``log_rate = μ`` for ``_other`` (no η subtraction).
    """
    log_rate = _scatter_log_rate_decoupled(
        x_dev, mu, jnp.asarray(0.0, dtype=x_dev.dtype), kept_idx
    )
    nb = _nb_factors(log_rate, u, r)
    a_full = nb["a"]
    one_minus_p_full = nb["one_minus_p"]
    a_kept = a_full[kept_idx]

    factors = _woodbury_factors(W, d, a_kept, damping)

    inv_d = 1.0 / d
    sigma_inv_xdev = inv_d * x_dev - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * x_dev)
        )
    )
    # NB gradient on kept axis (no η contribution).
    expected_count_full = (u + r) * one_minus_p_full
    g_x_dev = (u - expected_count_full)[kept_idx] - sigma_inv_xdev

    delta_x_dev = _solve_A(factors, g_x_dev)
    grad_inf_norm = jnp.max(jnp.abs(g_x_dev))
    step_norm = jnp.max(jnp.abs(delta_x_dev))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    return x_dev + delta_x_dev * scale, grad_inf_norm


def laplace_newton_loop_x_only_decoupled(
    x_dev_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    kept_idx: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``n_iters`` Newton steps on ``x_dev`` (no capture, decoupled)."""

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only_decoupled(
            x_, u, mu, W, d, r, kept_idx, damping, max_step,
        )
        return (x_new, grad_norm), None

    init = (x_dev_init, jnp.asarray(jnp.inf, dtype=x_dev_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    log_rate_final = _scatter_log_rate_decoupled(
        x_final, mu, jnp.asarray(0.0, dtype=x_final.dtype), kept_idx
    )
    nb_final = _nb_factors(log_rate_final, u, r)
    a_kept_final = nb_final["a"][kept_idx]
    factors = _woodbury_factors(W, d, a_kept_final, damping)
    log_det_neg_H = _log_det_A(factors)
    return x_final, final_grad, log_det_neg_H


laplace_newton_batch_x_only_decoupled = jax.vmap(
    laplace_newton_loop_x_only_decoupled,
    in_axes=(0, 0, None, None, None, None, None, None, None, None),
)


# ---------------------------------------------------------------------
# x-only Newton with fixed offset (frozen η, decoupled)
# ---------------------------------------------------------------------


def newton_step_x_only_offset_decoupled(
    x_dev: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_offset: jnp.ndarray,
    kept_idx: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x_dev`` with fixed ``η_offset`` (decoupled).

    Equivalent to the no-capture decoupled path with a constant additive
    shift applied to log_rate.  No η block in the Hessian; the offset
    only changes where NB factors saturate.
    """
    log_rate = _scatter_log_rate_decoupled(x_dev, mu, eta_offset, kept_idx)
    nb = _nb_factors(log_rate, u, r)
    a_full = nb["a"]
    one_minus_p_full = nb["one_minus_p"]
    a_kept = a_full[kept_idx]

    factors = _woodbury_factors(W, d, a_kept, damping)

    inv_d = 1.0 / d
    sigma_inv_xdev = inv_d * x_dev - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * x_dev)
        )
    )
    expected_count_full = (u + r) * one_minus_p_full
    g_x_dev = (u - expected_count_full)[kept_idx] - sigma_inv_xdev

    delta_x_dev = _solve_A(factors, g_x_dev)
    grad_inf_norm = jnp.max(jnp.abs(g_x_dev))
    step_norm = jnp.max(jnp.abs(delta_x_dev))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    return x_dev + delta_x_dev * scale, grad_inf_norm


def laplace_newton_loop_x_only_offset_decoupled(
    x_dev_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_offset: jnp.ndarray,
    kept_idx: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """``n_iters`` Newton steps on ``x_dev`` with fixed ``η_offset``."""

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only_offset_decoupled(
            x_, u, mu, W, d, r, eta_offset, kept_idx, damping, max_step,
        )
        return (x_new, grad_norm), None

    init = (x_dev_init, jnp.asarray(jnp.inf, dtype=x_dev_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    log_rate_final = _scatter_log_rate_decoupled(
        x_final, mu, eta_offset, kept_idx
    )
    nb_final = _nb_factors(log_rate_final, u, r)
    a_kept_final = nb_final["a"][kept_idx]
    factors = _woodbury_factors(W, d, a_kept_final, damping)
    log_det_neg_H = _log_det_A(factors)
    return x_final, final_grad, log_det_neg_H


laplace_newton_batch_x_only_offset_decoupled = jax.vmap(
    laplace_newton_loop_x_only_offset_decoupled,
    in_axes=(0, 0, None, None, None, None, 0, None, None, None, None),
)


# ---------------------------------------------------------------------
# log det(−H) batch helpers (decoupled) — match the legacy
# laplace_log_det_neg_H_* signatures so loss_fn can call them with
# minimal special-casing.
# ---------------------------------------------------------------------


def laplace_log_det_neg_H_decoupled(
    x_dev_map: jnp.ndarray,
    eta_map: Optional[jnp.ndarray],
    u: jnp.ndarray,
    r: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    mu: jnp.ndarray,
    kept_idx: jnp.ndarray,
    sigma_M: float,
) -> jnp.ndarray:
    """``log det(−H)`` at the MAP under the decoupled layout.

    Soft-η: ``log det(−H) = log det A + log s`` where
    ``s = Σ_g a_full + 1/σ_M² − a_keptᵀ A⁻¹ a_kept``.
    No capture / frozen-η (``eta_map is None``): just ``log det A``.
    """
    eta_sub = eta_map if eta_map is not None else jnp.asarray(
        0.0, dtype=x_dev_map.dtype
    )
    log_rate = _scatter_log_rate_decoupled(x_dev_map, mu, eta_sub, kept_idx)
    nb = _nb_factors(log_rate, u, r)
    a_full = nb["a"]
    a_kept = a_full[kept_idx]
    factors = _woodbury_factors(W, d, a_kept, damping=0.0)
    log_det_A = _log_det_A(factors)
    if eta_map is None:
        return log_det_A
    A_inv_a = _solve_A(factors, a_kept)
    s = jnp.maximum(
        jnp.sum(a_full) + 1.0 / (sigma_M**2) - jnp.dot(a_kept, A_inv_a),
        1e-30,
    )
    return log_det_A + jnp.log(s)


laplace_log_det_neg_H_batch_decoupled = jax.vmap(
    laplace_log_det_neg_H_decoupled,
    in_axes=(0, 0, 0, None, None, None, None, None, 0),
)
laplace_log_det_neg_H_batch_x_only_decoupled = jax.vmap(
    laplace_log_det_neg_H_decoupled,
    in_axes=(0, None, 0, None, None, None, None, None, None),
)


def laplace_log_det_neg_H_x_only_offset_decoupled(
    x_dev_map: jnp.ndarray,
    eta_offset: jnp.ndarray,
    u: jnp.ndarray,
    r: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    mu: jnp.ndarray,
    kept_idx: jnp.ndarray,
) -> jnp.ndarray:
    """``log det(−H_x)`` at the decoupled x-only-offset MAP.

    No η block: returns ``log det A`` with NB factors at the offset-
    shifted log_rate.
    """
    log_rate = _scatter_log_rate_decoupled(
        x_dev_map, mu, eta_offset, kept_idx
    )
    nb = _nb_factors(log_rate, u, r)
    a_kept = nb["a"][kept_idx]
    factors = _woodbury_factors(W, d, a_kept, damping=0.0)
    return _log_det_A(factors)


laplace_log_det_neg_H_batch_x_only_offset_decoupled = jax.vmap(
    laplace_log_det_neg_H_x_only_offset_decoupled,
    in_axes=(0, 0, 0, None, None, None, None, None),
)


# ---------------------------------------------------------------------
# Per-block gradient-norm diagnostics (decoupled)
# ---------------------------------------------------------------------


def _nbln_grad_split_decoupled_with_eta(
    x_dev_map: jnp.ndarray,
    eta_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    kept_idx: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-block L∞ grad norms at the joint decoupled Newton MAP."""
    log_rate = _scatter_log_rate_decoupled(x_dev_map, mu, eta_map, kept_idx)
    nb = _nb_factors(log_rate, u, r)
    one_minus_p_full = nb["one_minus_p"]

    inv_d = 1.0 / d
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_xdev = inv_d * x_dev_map - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * x_dev_map))
    )

    expected_count_full = (u + r) * one_minus_p_full
    g_lograte_full = u - expected_count_full
    g_x_dev = g_lograte_full[kept_idx] - sigma_inv_xdev
    g_eta = (
        -jnp.sum(g_lograte_full) - (eta_map - eta_anchor) / (sigma_M**2)
    )
    return jnp.max(jnp.abs(g_x_dev)), jnp.abs(g_eta)


nbln_grad_split_batch_decoupled = jax.vmap(
    _nbln_grad_split_decoupled_with_eta,
    in_axes=(0, 0, 0, None, None, None, None, None, 0, 0),
)


def _nbln_grad_x_only_norm_decoupled(
    x_dev_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    kept_idx: jnp.ndarray,
) -> jnp.ndarray:
    """``L∞`` of ``∇_{x_dev} f`` at the decoupled x-only Newton MAP."""
    log_rate = _scatter_log_rate_decoupled(
        x_dev_map, mu, jnp.asarray(0.0, dtype=x_dev_map.dtype), kept_idx
    )
    nb = _nb_factors(log_rate, u, r)
    one_minus_p_full = nb["one_minus_p"]

    inv_d = 1.0 / d
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_xdev = inv_d * x_dev_map - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * x_dev_map))
    )

    expected_count_full = (u + r) * one_minus_p_full
    g_x_dev = (u - expected_count_full)[kept_idx] - sigma_inv_xdev
    return jnp.max(jnp.abs(g_x_dev))


nbln_grad_x_only_norm_batch_decoupled = jax.vmap(
    _nbln_grad_x_only_norm_decoupled,
    in_axes=(0, 0, None, None, None, None, None),
)


def _nbln_grad_x_only_offset_norm_decoupled(
    x_dev_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_offset: jnp.ndarray,
    kept_idx: jnp.ndarray,
) -> jnp.ndarray:
    """``L∞`` of ``∇_{x_dev} f`` at the decoupled x-only-offset MAP."""
    log_rate = _scatter_log_rate_decoupled(
        x_dev_map, mu, eta_offset, kept_idx
    )
    nb = _nb_factors(log_rate, u, r)
    one_minus_p_full = nb["one_minus_p"]

    inv_d = 1.0 / d
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_xdev = inv_d * x_dev_map - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * x_dev_map))
    )

    expected_count_full = (u + r) * one_minus_p_full
    g_x_dev = (u - expected_count_full)[kept_idx] - sigma_inv_xdev
    return jnp.max(jnp.abs(g_x_dev))


nbln_grad_x_only_offset_norm_batch_decoupled = jax.vmap(
    _nbln_grad_x_only_offset_norm_decoupled,
    in_axes=(0, 0, None, None, None, None, 0, None),
)

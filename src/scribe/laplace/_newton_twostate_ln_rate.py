"""Pure-JAX Newton kernel for the TwoState-LogNormal-Rate MAP per cell.

Implements the inner Newton iteration of the Laplace-approximation
inference scheme for the ``twostate_ln_rate`` base model. Mirrors
:mod:`scribe.laplace._newton_nbln` exactly in structure: same Woodbury
machinery, same Schur block decomposition, same step-size cap and
damping safeguards. The only differences are in the *data-side*
contributions to the gradient and Hessian, which come from a
**Poisson-Beta compound** likelihood evaluated via fixed Gauss-Legendre
quadrature instead of a Negative-Binomial closed form.

Mathematical background
-----------------------
The TSLN-Rate per-cell joint log-density (capture anchor active,
dropping additive constants) is

    f(x, η) = Σ_g log L_PB(u_g | α_g, β_g, exp(x_g − η))
              − ½ (x − μ)ᵀ Σ⁻¹ (x − μ)
              − ½ (η − η_anchor)² / σ_M²

where x ∈ ℝ^G is the latent log-rate (with prior centering
``μ_g = log r_hat_g``), η is the per-cell capture offset
(convention: ``ν_c = exp(-η)``, so ``log_rate_cg = x_g − η``),
μ ∈ ℝ^G is the population log-rate (= log r_hat), Σ = WWᵀ + diag(d),
``η_anchor = log(M_0) − log(L_c)``, and ``(α_g, β_g) ∈ ℝ^G_{>0}`` are
the gene-level Beta-density shape parameters derived from the upstream
TwoState SVI's ``(μ_g, b_g, k^-_g)`` via the existing
``_twostate_reparam`` helper. The Poisson-Beta marginal is

    L_PB(u | α, β, λ) = ∫₀¹ Po(u | λ p) · Beta(p | α, β) dp.

**Important naming clash**: the ``μ`` parameter in this kernel is the
**log r_hat** (the prior centering on the latent log-rate ``x``),
matching the NBLN convention.  This is NOT the same as the TwoState
SVI's ``μ_g`` (positive gene mean expression). The Laplace globals
that get optimized are TwoState's ``(mu, burst_size, k_off)``; the
kernel-internal ``μ`` argument is the derived ``log r_hat_g``
computed via ``_twostate_reparam(mu, burst_size, k_off)``.

Quadrature
----------
The Beta-weighted Poisson posterior over ``p`` is evaluated at K
fixed Gauss-Legendre nodes on [0, 1] (see
:mod:`scribe.stats._jacobi_quad`).  The nodes and weights are
JIT-time constants in ``(α, β)`` — only the Beta-density factor at
each node depends on ``(α, β, x)``.  Autograd is clean through the
Beta log-density via closed-form digamma derivatives; no
eigendecomposition is required (unlike Gauss-Jacobi via Golub-Welsch).

Closed-form factors (no autodiff)
---------------------------------
Define the per-quadrature-node posterior over ``p`` given ``u`` and
the current iterate ``log_rate = x − η``:

    q_k = softmax_k[log w_k + log Beta(p_k | α, β)
                  + u · log(λ p_k) − λ p_k]

with ``λ = exp(log_rate)``. Then

    Ē_p = Σ_k q_k p_k,   Var_p = Σ_k q_k p_k² − Ē_p².

The data-side gradient is

    (∂_x f)_g^data = u_g − λ_g · Ē_p

(observed minus posterior-expected Poisson rate — the direct analog of
NBLN's ``u − (u+r)(1−p)`` with ``(u+r)(1−p)`` replaced by ``λ · E[p]``).

The data-side Hessian diagonal is

    a_g = λ_g · Ē_p − λ_g² · Var_p.

Sign / log-concavity caveat
---------------------------
Unlike NBLN where the per-cell log-likelihood is log-concave in
``x``, the Poisson-Beta marginal is **not** uniformly log-concave when
the Beta is U-shaped (``α < 1`` or ``β < 1``).  The Prékopa argument
fails because the Beta density is not log-concave in that regime.

**Defensive clamp**: we floor ``a_g`` at ``_A_MIN = 1e-3`` before the
Woodbury solve.  Both the clamped ``a`` and the raw ``a_raw`` are
returned by the factor function so the obs model can surface a
"clamp activation rate" diagnostic.  Tests must exercise the
``α, β < 1`` grid; see ``tests/test_twostate_ln_rate_newton.py``.

Capture-active path (joint Newton on x and η)
----------------------------------------------
With the convention ``ν_c = exp(-η)`` (so ``η = -log ν_c``), z and η
enter the same scalar log-rate with opposite signs:

    log_rate_cg = x_g − η_c

Therefore

    g_x_data = u − λ · Ē_p
    g_η_data = -Σ_g g_x_data_g = -Σ u + Σ λ · Ē_p
    H_{x, η} = +a       (column vector)
    H_{η, η} = -Σ a − 1/σ_M²

The Schur block decomposition is identical to NBLN's joint Newton step
in :func:`scribe.laplace._newton_nbln.newton_step_joint`.

NB limit (sanity check)
-----------------------
As ``β → ∞`` with ``α = r`` fixed and ``b = λ/β`` finite, the Beta-
Poisson posterior approaches the Gamma-Poisson conjugate update:

    Ē_p → (r + u) / (β + λ)
    Var_p → (r + u) / (β + λ)²
    a_g → (r + u) · q (1 − q),  q = λ/(β + λ)

which reproduces NBLN's ``a = (u + r) p (1 − p)`` exactly (with
``p_success = 1 − q``).  See plan §3.1 NB-limit derivation.

API
---
- :func:`_twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)`
  — closed-form factor dict ``{a, a_raw, g_data, log_marginal, ...}``.
- :func:`newton_step_joint`, :func:`newton_step_x_only`,
  :func:`newton_step_x_only_offset` — single Newton steps.
- :func:`laplace_newton_loop*` — fixed-iteration ``lax.scan``.
- :func:`laplace_newton_batch*` — ``jax.vmap`` over cells.
- :func:`laplace_log_det_neg_H*` — log-det wrappers for the ELBO.
- ``twostate_ln_rate_grad_*_norm_batch`` — per-block gradient
  diagnostics used by the progress display.

Cross-references
----------------
- ``paper/_two_state_promoter.qmd`` §sec-twostate-tsln-rate (planned).
- ``scribe.laplace._newton_nbln`` — the NB-LogNormal analogue.
- ``scribe.stats._jacobi_quad.gauss_legendre_01`` — fixed quadrature
  rule.
- ``scribe.stats.distributions.PoissonBetaCompound`` — matching SVI
  observation distribution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special

from scribe.stats._jacobi_quad import gauss_legendre_01


# Sanity bounds on (x - eta) to protect float32 arithmetic.  Matches
# the bound in NBLN; ``a_g`` here is bounded by ``λ · Ē_p`` so
# saturation at extreme log-rates is graceful.
_LOG_RATE_MIN = -50.0
_LOG_RATE_MAX = 50.0

# Floor on Beta shape parameters when computing the Beta log-density.
# These should already be enforced upstream by ``_twostate_reparam``
# (with floors at 0.05), but we re-floor here as a defensive check.
_ALPHA_MIN_KERNEL = 1e-3
_BETA_MIN_KERNEL = 1e-3

# Defensive floor on the data-side Hessian diagonal ``a_g`` before
# entering the Woodbury solve.  See module docstring "Sign / log-
# concavity caveat" for the rationale.  Empirically, ``a_raw < _A_MIN``
# is expected to be rare when (α, β) are both ≳ 1 (the regime most
# genes occupy) but can occur for bursty / U-shaped Beta genes.
_A_MIN = 1e-3

# Default number of quadrature nodes.  Matches the SVI-side
# ``PoissonBetaCompound`` default in scribe.stats.distributions.
_DEFAULT_K = 60


# =====================================================================
# Closed-form factor computation
# =====================================================================


def _twostate_ln_rate_factors(
    log_rate: jnp.ndarray,
    u: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> dict:
    """Compute Poisson-Beta data-side quantities for one cell.

    Given ``log_rate = x − η ∈ ℝ^G`` and gene-level Beta shape
    parameters ``(α_g, β_g) ∈ ℝ^G_+``, return the closed-form
    gradient ``g_data_g = u_g − λ_g Ē_p`` and Hessian-diagonal
    ``a_g = λ_g Ē_p − λ_g² Var_p`` per gene, plus the per-gene
    log-marginal ``log L_PB(u_g | α_g, β_g, λ_g)`` for the EM loss.

    The defensive floor ``a = jnp.maximum(a_raw, _A_MIN)`` is applied
    before returning ``a``; ``a_raw`` is returned alongside for
    diagnostics (clamp activation rate).

    Parameters
    ----------
    log_rate : jnp.ndarray, shape ``(G,)``
        Effective log-rate ``x_g − η_c``.
    u : jnp.ndarray, shape ``(G,)``
        Observed counts for this cell.
    alpha : jnp.ndarray, shape ``(G,)``
        Gene-level Beta first shape parameter (= ``mu_g / burst_size_g``
        after ``_twostate_reparam`` floors).
    beta : jnp.ndarray, shape ``(G,)``
        Gene-level Beta second shape parameter (= ``k_off_g``).
    n_quad_nodes : int, default 60
        Number of Gauss-Legendre quadrature nodes on [0, 1].

    Returns
    -------
    dict with keys:
        ``a`` : shape ``(G,)``, Hessian-diagonal entry after clamp.
        ``a_raw`` : shape ``(G,)``, raw Hessian diagonal (pre-clamp).
        ``g_data`` : shape ``(G,)``, data-side gradient ``u - λ E_q[p]``.
        ``lambda_g`` : shape ``(G,)``, Poisson rate ``exp(log_rate)``.
        ``E_p`` : shape ``(G,)``, posterior mean of ``p`` given ``u``.
        ``Var_p`` : shape ``(G,)``, posterior variance of ``p``.
        ``log_marginal`` : shape ``(G,)``, per-gene log Poisson-Beta
            marginal log-likelihood ``log L_PB(u_g | α_g, β_g, λ_g)``.
        ``softmax_node`` : shape ``(G, K)``, posterior softmax weights.
    """
    log_rate_clipped = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    alpha_safe = jnp.maximum(alpha, _ALPHA_MIN_KERNEL)
    beta_safe = jnp.maximum(beta, _BETA_MIN_KERNEL)

    # Fixed Gauss-Legendre nodes on [0, 1].
    p_node, log_w_node = gauss_legendre_01(n_quad_nodes)  # both (K,)
    log_p = jnp.log(p_node)
    log_1mp = jnp.log1p(-p_node)

    # Beta log-density at the fixed nodes: log_Beta(p | α, β) =
    # (α-1) log p + (β-1) log(1-p) − betaln(α, β).
    log_beta_density = (
        (alpha_safe[:, None] - 1.0) * log_p[None, :]
        + (beta_safe[:, None] - 1.0) * log_1mp[None, :]
        - jsp_special.betaln(alpha_safe, beta_safe)[:, None]
    )  # (G, K)
    log_w_eff = log_w_node[None, :] + log_beta_density  # (G, K)

    # Poisson log-PMF kernel in stable log-rate form:
    # log Po(u | λ p) = u log λ + u log p − λ p − gammaln(u + 1).
    log_lambda = log_rate_clipped[:, None] + log_p[None, :]  # (G, K)
    log_poisson = (
        u[:, None] * log_lambda
        - jnp.exp(log_lambda)
        - jsp_special.gammaln(u + 1.0)[:, None]
    )  # (G, K)

    log_q = log_w_eff + log_poisson  # (G, K)
    log_marginal = jsp_special.logsumexp(log_q, axis=-1)  # (G,)
    softmax_node = jax.nn.softmax(log_q, axis=-1)  # (G, K)

    # Conditional moments of p.
    E_p = (softmax_node * p_node[None, :]).sum(axis=-1)  # (G,)
    E_p2 = (softmax_node * (p_node**2)[None, :]).sum(axis=-1)  # (G,)
    Var_p = jnp.maximum(E_p2 - E_p * E_p, 0.0)

    lambda_g = jnp.exp(log_rate_clipped)
    g_data = u - lambda_g * E_p
    a_raw = lambda_g * E_p - lambda_g * lambda_g * Var_p

    # Defensive floor before Woodbury.  See module docstring.
    a = jnp.maximum(a_raw, _A_MIN)

    return {
        "a": a,
        "a_raw": a_raw,
        "g_data": g_data,
        "lambda_g": lambda_g,
        "E_p": E_p,
        "Var_p": Var_p,
        "log_marginal": log_marginal,
        "softmax_node": softmax_node,
    }


# =====================================================================
# Woodbury helpers (inlined; identical to NBLN's structure)
# =====================================================================


def _woodbury_factors(
    W: jnp.ndarray,
    d: jnp.ndarray,
    a_diag: jnp.ndarray,
    damping: float,
) -> dict:
    """Pre-compute Woodbury factors for ``A = diag(a + 1/d + damping) − V K⁻¹ V'``.

    Mirrors :func:`scribe.laplace._newton_nbln._woodbury_factors`
    bit-identically; only the source of ``a_diag`` differs.
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
    """Compute ``A⁻¹ y`` using pre-built Woodbury factors."""
    m_inv = factors["m_inv"]
    V = factors["V"]
    L_S = factors["L_S"]

    base = m_inv * y
    rhs = V.T @ base
    z = jax.scipy.linalg.cho_solve((L_S, True), rhs)
    return base + m_inv * (V @ z)


def _log_det_A(factors: dict) -> jnp.ndarray:
    """``log det A`` from Woodbury factors via matrix-determinant lemma."""
    m_inv = factors["m_inv"]
    log_det_K = factors["log_det_K"]
    log_det_S = factors["log_det_S"]
    log_det_M = -jnp.sum(jnp.log(m_inv))
    return log_det_M + log_det_S - log_det_K


# =====================================================================
# log det(-H) wrappers for the Laplace ELBO
# =====================================================================


def laplace_log_det_neg_H(
    x_map: jnp.ndarray,
    eta_map: Optional[jnp.ndarray],
    u: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    sigma_M: float,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``log det(-H)`` at the MAP for the joint (x, η) Laplace ELBO.

    Used by the outer Laplace ELBO so gradients flow through the
    determinant correction into the globals ``(W, d, μ, α, β)``.

    Capture-anchor on (``eta_map is not None``):

        log det(-H) = log det A + log s,

    where ``s = Σ_g a_g + 1/σ_M² − a^T A⁻¹ a`` is the Schur scalar.
    """
    log_rate = x_map - eta_map if eta_map is not None else x_map
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)
    factors = _woodbury_factors(W, d, fac["a"], damping=0.0)
    log_det_A = _log_det_A(factors)
    if eta_map is None:
        return log_det_A
    a = fac["a"]
    a_inv_A = _solve_A(factors, a)
    s = jnp.maximum(
        jnp.sum(a) + 1.0 / (sigma_M**2) - jnp.dot(a, a_inv_A),
        1e-30,
    )
    return log_det_A + jnp.log(s)


# Vmapped: per-cell axes 0 for x_map/eta_map/u; globals shared.
# ``sigma_M`` (slot 7) is per-cell for the soft-cascade prior;
# ``n_quad_nodes`` (slot 8) is shared.
laplace_log_det_neg_H_batch = jax.vmap(
    laplace_log_det_neg_H,
    in_axes=(0, 0, 0, None, None, None, None, 0, None),
)
laplace_log_det_neg_H_batch_x_only = jax.vmap(
    laplace_log_det_neg_H,
    in_axes=(0, None, 0, None, None, None, None, None, None),
)


def laplace_log_det_neg_H_x_only_offset(
    x_map: jnp.ndarray,
    eta_offset: jnp.ndarray,
    u: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``log det(-H_x)`` at the MAP for the x-only-with-offset path.

    The η block disappears (η is a fixed offset, not a Newton latent);
    the log-det is just ``log det A`` with ``a`` evaluated at the
    shifted log-rate ``x − η_offset``.
    """
    log_rate = x_map - eta_offset
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)
    factors = _woodbury_factors(W, d, fac["a"], damping=0.0)
    return _log_det_A(factors)


laplace_log_det_neg_H_batch_x_only_offset = jax.vmap(
    laplace_log_det_neg_H_x_only_offset,
    in_axes=(0, 0, 0, None, None, None, None, None),
)


# =====================================================================
# Newton step (joint x, eta) — capture-anchor active
# =====================================================================


def newton_step_joint(
    x: jnp.ndarray,
    eta: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One joint Newton step on ``(x, η)`` for TSLN-Rate.

    Solves ``-H δ = ∇f`` via block Schur complement, reusing one
    Cholesky factorisation of ``S = K − V' diag(m_inv) V`` for all
    three internal solves.

    Parameters
    ----------
    x : jnp.ndarray, shape ``(G,)``
        Current latent log-rate iterate (centred at ``μ = log r_hat``).
    eta : jnp.ndarray, scalar
        Current capture offset iterate. ``ν_c = exp(-η)``.
    u : jnp.ndarray, shape ``(G,)``
        Observed counts.
    mu : jnp.ndarray, shape ``(G,)``
        Prior centering on ``x``: ``log r_hat_g`` (NOT TwoState's
        positive ``mu``; see module docstring).
    W : jnp.ndarray, shape ``(G, k)``
        Low-rank loadings.
    d : jnp.ndarray, shape ``(G,)``
        Diagonal residual variance (positive).
    alpha, beta : jnp.ndarray, shape ``(G,)``
        Beta-density shape parameters from ``_twostate_reparam``.
    eta_anchor : jnp.ndarray, scalar
        TruncN anchor for η.
    sigma_M : float
        TruncN scale on η.
    damping : float, default 1e-4
    max_step : float, default 5.0
    n_quad_nodes : int, default 60

    Returns
    -------
    x_new : jnp.ndarray, shape ``(G,)``
    eta_new : jnp.ndarray, scalar
    grad_inf_norm : jnp.ndarray, scalar
        Pre-step ``L∞`` norm of ``(∇_x f, ∇_η f)``.
    """
    log_rate = x - eta
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)
    a = fac["a"]
    g_data = fac["g_data"]

    factors = _woodbury_factors(W, d, a, damping)

    # Σ⁻¹ (x − μ) via inner Woodbury.
    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )

    # Data + prior gradients.  Convention: log_rate = x − η, so
    # ∂/∂η = -∂/∂x on the data side ⇒ g_eta_data = -Σ_g g_x_data_g.
    g_x = g_data - sigma_inv_diff
    g_eta_data = -jnp.sum(g_data)
    g_eta = g_eta_data - (eta - eta_anchor) / (sigma_M**2)

    # Schur block decomposition on -H (identical algebra to NBLN).
    A_inv_g_x = _solve_A(factors, g_x)
    a_inv_A = _solve_A(factors, a)
    s = jnp.sum(a) + 1.0 / (sigma_M**2) - jnp.dot(a, a_inv_A) + damping
    delta_eta = (g_eta + jnp.dot(a, A_inv_g_x)) / s
    delta_x = A_inv_g_x + a_inv_A * delta_eta

    grad_inf_norm = jnp.maximum(jnp.max(jnp.abs(g_x)), jnp.abs(g_eta))

    # Step-size cap.
    step_norm = jnp.maximum(jnp.max(jnp.abs(delta_x)), jnp.abs(delta_eta))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    delta_x = delta_x * scale
    delta_eta = delta_eta * scale

    eta_new = eta + delta_eta
    # TruncN(low=0) projection on η (matches NBLN).
    eta_new = jnp.maximum(eta_new, 1e-3)
    return x + delta_x, eta_new, grad_inf_norm


def laplace_newton_loop(
    x_init: jnp.ndarray,
    eta_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    jnp.ndarray,
]:
    """Run ``n_iters`` Newton steps on ``(x, η)`` for one cell.

    Returns
    -------
    x_map : jnp.ndarray, shape ``(G,)``
    eta_map : jnp.ndarray, scalar
    final_grad_norm : jnp.ndarray, scalar
    log_det_neg_H : jnp.ndarray, scalar
        ``log det(-H)`` at the converged iterate.
    log_marginal_sum : jnp.ndarray, scalar
        Sum-over-genes of the Poisson-Beta marginal log-likelihood at
        the converged iterate.  Used by the obs model's loss_fn to
        assemble the data log-prob without recomputing quadrature.
    a_raw_min : jnp.ndarray, scalar
        Minimum of ``a_raw`` across the gene axis at the converged
        iterate.  Used by ``compute_global_uncertainty`` /
        ``pack_result`` to surface the clamp-activation diagnostic.
    """

    def step(carry, _):
        x_, eta_, _grad = carry
        x_new, eta_new, grad_norm = newton_step_joint(
            x_, eta_, u, mu, W, d, alpha, beta, eta_anchor, sigma_M,
            damping, max_step, n_quad_nodes,
        )
        return (x_new, eta_new, grad_norm), None

    init = (x_init, eta_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, eta_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    # Recompute factors at the MAP for log det(-H), log_marginal, and
    # a_raw diagnostics.  One extra factor call per cell — unavoidable
    # because the ELBO needs the determinant.
    final_log_rate = x_final - eta_final
    fac_final = _twostate_ln_rate_factors(
        final_log_rate, u, alpha, beta, n_quad_nodes
    )
    a_final = fac_final["a"]
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
    log_marginal_sum = jnp.sum(fac_final["log_marginal"])
    a_raw_min = jnp.min(fac_final["a_raw"])

    return (
        x_final, eta_final, final_grad, log_det_neg_H,
        log_marginal_sum, a_raw_min,
    )


laplace_newton_batch = jax.vmap(
    laplace_newton_loop,
    in_axes=(
        0,      # x_init
        0,      # eta_init
        0,      # u
        None,   # mu
        None,   # W
        None,   # d
        None,   # alpha
        None,   # beta
        0,      # eta_anchor (per-cell)
        0,      # sigma_M (per-cell — for SVI cascade)
        None,   # n_iters
        None,   # damping
        None,   # max_step
        None,   # n_quad_nodes
    ),
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
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x`` only (no capture anchor).

    ``A = diag(a + 1/d) − V K⁻¹ V'``; Newton step ``δ = A⁻¹ g_x``.
    """
    fac = _twostate_ln_rate_factors(x, u, alpha, beta, n_quad_nodes)
    a = fac["a"]
    g_data = fac["g_data"]

    factors = _woodbury_factors(W, d, a, damping)

    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )
    g_x = g_data - sigma_inv_diff

    delta_x = _solve_A(factors, g_x)
    grad_inf_norm = jnp.max(jnp.abs(g_x))

    step_norm = jnp.max(jnp.abs(delta_x))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    return x + delta_x * scale, grad_inf_norm


def laplace_newton_loop_x_only(
    x_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """``n_iters`` Newton steps on ``x`` for one cell (no capture).

    Returns ``(x_map, final_grad_norm, log_det_neg_H,
    log_marginal_sum, a_raw_min)``.
    """

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only(
            x_, u, mu, W, d, alpha, beta, damping, max_step, n_quad_nodes,
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    fac_final = _twostate_ln_rate_factors(
        x_final, u, alpha, beta, n_quad_nodes
    )
    factors = _woodbury_factors(W, d, fac_final["a"], damping)
    log_det_neg_H = _log_det_A(factors)
    log_marginal_sum = jnp.sum(fac_final["log_marginal"])
    a_raw_min = jnp.min(fac_final["a_raw"])
    return x_final, final_grad, log_det_neg_H, log_marginal_sum, a_raw_min


laplace_newton_batch_x_only = jax.vmap(
    laplace_newton_loop_x_only,
    in_axes=(0, 0, None, None, None, None, None, None, None, None, None),
)


# =====================================================================
# x-only-with-offset variant (capture frozen at upstream MAP)
# =====================================================================


def newton_step_x_only_offset(
    x: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_offset: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Newton step on ``x`` with fixed ``η_offset``.

    Data factors evaluated at ``log_rate = x − η_offset``; the MVN
    prior on ``x`` is unchanged (prior is on the latent log-rate, not
    on the observable log_rate).  No η block ⇒ no rigid-translation
    gauge in the Hessian; safe to raise ``max_step`` to 20-50.
    """
    log_rate = x - eta_offset
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)
    a = fac["a"]
    g_data = fac["g_data"]

    factors = _woodbury_factors(W, d, a, damping)

    inv_d = 1.0 / d
    diff = x - mu
    sigma_inv_diff = inv_d * diff - inv_d * (
        W
        @ jax.scipy.linalg.cho_solve(
            (factors["L_K"], True), W.T @ (inv_d * diff)
        )
    )
    # ∂ log_rate / ∂ x = 1 (∂ η_offset / ∂ x = 0), so gradient on x
    # is just the data gradient minus the prior gradient.
    g_x = g_data - sigma_inv_diff

    delta_x = _solve_A(factors, g_x)
    grad_inf_norm = jnp.max(jnp.abs(g_x))

    step_norm = jnp.max(jnp.abs(delta_x))
    scale = jnp.minimum(1.0, max_step / jnp.maximum(step_norm, 1e-12))
    return x + delta_x * scale, grad_inf_norm


def laplace_newton_loop_x_only_offset(
    x_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_offset: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """``n_iters`` Newton steps on ``x`` with fixed ``η_offset``.

    Returns ``(x_map, final_grad_norm, log_det_neg_H,
    log_marginal_sum, a_raw_min)``.
    """

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only_offset(
            x_, u, mu, W, d, alpha, beta, eta_offset,
            damping, max_step, n_quad_nodes,
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    log_rate_final = x_final - eta_offset
    fac_final = _twostate_ln_rate_factors(
        log_rate_final, u, alpha, beta, n_quad_nodes
    )
    factors = _woodbury_factors(W, d, fac_final["a"], damping)
    log_det_neg_H = _log_det_A(factors)
    log_marginal_sum = jnp.sum(fac_final["log_marginal"])
    a_raw_min = jnp.min(fac_final["a_raw"])
    return x_final, final_grad, log_det_neg_H, log_marginal_sum, a_raw_min


laplace_newton_batch_x_only_offset = jax.vmap(
    laplace_newton_loop_x_only_offset,
    in_axes=(0, 0, None, None, None, None, None, 0, None, None, None, None),
)


# =====================================================================
# Per-block gradient-norm diagnostics
# =====================================================================
#
# Used by the obs-model's loss_fn to surface a per-block max/p99/med
# triplet in the progress display.  Mirrors the NBLN-side helpers
# (``nbln_grad_split_batch`` etc.) one-for-one.


def _twostate_ln_rate_grad_split_with_eta(
    x_map: jnp.ndarray,
    eta_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    sigma_M: float,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-block ``L∞`` grad norms at the joint TSLN-Rate MAP.

    Returns ``(grad_inf_norm_x, grad_inf_norm_eta)``.
    """
    log_rate = x_map - eta_map
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)
    g_data = fac["g_data"]

    inv_d = 1.0 / d
    diff = x_map - mu
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )

    g_x = g_data - sigma_inv_diff
    g_eta = -jnp.sum(g_data) - (eta_map - eta_anchor) / (sigma_M**2)
    return jnp.max(jnp.abs(g_x)), jnp.abs(g_eta)


twostate_ln_rate_grad_split_batch = jax.vmap(
    _twostate_ln_rate_grad_split_with_eta,
    in_axes=(0, 0, 0, None, None, None, None, None, 0, 0, None),
)


def _twostate_ln_rate_grad_x_only_norm(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``L∞`` of ``∇_x f`` at the x-only TSLN-Rate MAP."""
    fac = _twostate_ln_rate_factors(x_map, u, alpha, beta, n_quad_nodes)
    g_data = fac["g_data"]

    inv_d = 1.0 / d
    diff = x_map - mu
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )
    g_x = g_data - sigma_inv_diff
    return jnp.max(jnp.abs(g_x))


twostate_ln_rate_grad_x_only_norm_batch = jax.vmap(
    _twostate_ln_rate_grad_x_only_norm,
    in_axes=(0, 0, None, None, None, None, None, None),
)


def _twostate_ln_rate_grad_x_only_offset_norm(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_offset: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``L∞`` of ``∇_x f`` at the x-only-with-offset TSLN-Rate MAP."""
    log_rate = x_map - eta_offset
    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes)
    g_data = fac["g_data"]

    inv_d = 1.0 / d
    diff = x_map - mu
    k = W.shape[1]
    K = jnp.eye(k, dtype=W.dtype) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    sigma_inv_diff = inv_d * diff - inv_d * (
        W @ jax.scipy.linalg.cho_solve((L_K, True), W.T @ (inv_d * diff))
    )
    g_x = g_data - sigma_inv_diff
    return jnp.max(jnp.abs(g_x))


twostate_ln_rate_grad_x_only_offset_norm_batch = jax.vmap(
    _twostate_ln_rate_grad_x_only_offset_norm,
    in_axes=(0, 0, None, None, None, None, None, 0, None),
)


# =====================================================================
# Closed-form global-curvature factors (compute_global_uncertainty)
# =====================================================================
#
# These helpers compute per-cell-per-gene contributions to the
# negative-log-posterior gradient and Hessian-diagonal in the
# "natural" basis (α, β, log_rate_g), then the caller chain-rules
# through ``_twostate_reparam`` to the unconstrained ``*_loc`` space.
#
# Sign convention: ``g_*`` and ``H_*`` are gradients / curvatures of
# the *negative* log posterior, so the diagonal posterior precision
# at convergence is ``H_natural + prior_precision`` (positive).
#
# Why hand-derive: ``jnp.diag(jax.hessian(neg_log_post))`` materialises
# transient ``(C, G, G)`` autodiff tensors that OOM at production
# scale (G ~ 10K, C ~ several thousand → hundreds of GB).  The
# chunked-autodiff fallback bounds memory but is ``O(G)`` forward-
# over-reverse passes.  The hand-derived path piggybacks on the
# existing Newton kernel's softmax + adds a handful of K-axis
# moment reductions; same memory footprint as one Newton iterate.


def _global_curvature_factors_rate(
    x: jnp.ndarray,
    u: jnp.ndarray,
    eta_cap: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> dict:
    """Per-cell per-gene closed-form NLP gradient + Hessian-diagonal in
    the *(α, β)* data-side basis.

    The TSLN-Rate data log-likelihood depends on the gene globals
    only through ``(α_g, β_g)`` — ``log_rate_cg = x_cg − η_cap_c`` is
    ``stop_gradient``'d at the per-cell MAP.  Per the log-marginal
    identity (Louis 1982 / log-deriv lemma), the per-cell-per-gene
    derivatives of ``log L_PB(u | α, β, λ)`` w.r.t. ``α, β`` are:

    .. math::
        \\frac{\\partial \\log L_{PB}}{\\partial \\alpha}
            = E_q[\\log p] - \\psi(\\alpha) + \\psi(\\alpha+\\beta) \\\\
        \\frac{\\partial \\log L_{PB}}{\\partial \\beta}
            = E_q[\\log(1-p)] - \\psi(\\beta) + \\psi(\\alpha+\\beta)

    where ``q(p) ∝ Beta(α, β) · Po(u | λ p)`` is the posterior over
    ``p`` for this cell-gene.  The Hessian elements follow:

    .. math::
        \\partial^2_\\alpha
            = -\\psi'(\\alpha) + \\psi'(\\alpha+\\beta)
              + \\mathrm{Var}_q[\\log p] \\\\
        \\partial^2_\\beta
            = -\\psi'(\\beta) + \\psi'(\\alpha+\\beta)
              + \\mathrm{Var}_q[\\log(1-p)] \\\\
        \\partial_\\alpha \\partial_\\beta
            = \\psi'(\\alpha+\\beta) + \\mathrm{Cov}_q[\\log p, \\log(1-p)]

    Sign-flipped to get NLP curvatures; the caller sums across cells
    before adding the MVN prior contribution to the ``log_rate_g``
    diagonal entry.

    Parameters
    ----------
    x : jnp.ndarray, shape ``(G,)``
        Per-cell latent at the MAP (the latent-side coordinate;
        ``log_rate_cg = x − η_cap_c``).
    u : jnp.ndarray, shape ``(G,)``
        Per-cell observed counts.
    eta_cap : jnp.ndarray, scalar
        Per-cell capture offset (use ``0.0`` for the no-capture path).
    alpha, beta : jnp.ndarray, shape ``(G,)``
        Gene-level Beta shape parameters after ``_twostate_reparam``
        floors.
    n_quad_nodes : int, default 60

    Returns
    -------
    dict with per-gene ``(G,)`` arrays (all NLP sign convention):
        ``g_alpha``  : ``ψ(α) − ψ(α+β) − E_q[log p]``
        ``g_beta``   : ``ψ(β) − ψ(α+β) − E_q[log(1-p)]``
        ``H_aa``     : ``ψ'(α) − ψ'(α+β) − Var_q[log p]``
        ``H_bb``     : ``ψ'(β) − ψ'(α+β) − Var_q[log(1-p)]``
        ``H_ab``     : ``−ψ'(α+β) − Cov_q[log p, log(1-p)]``
    """
    # Reuse the kernel factor function for the softmax_node and the
    # Beta/Poisson posterior weights — bit-identical to the Newton
    # kernel's evaluation at the MAP.
    log_rate_at_cell = x - eta_cap
    fac = _twostate_ln_rate_factors(
        log_rate_at_cell, u, alpha, beta, n_quad_nodes,
    )
    softmax_node = fac["softmax_node"]              # (G, K)

    # Reconstruct the K-axis log-coordinate arrays.
    p_node, _ = gauss_legendre_01(n_quad_nodes)
    log_p = jnp.log(p_node)                         # (K,)
    log_1mp = jnp.log1p(-p_node)                    # (K,)

    # Posterior moments under q.
    E_logp = (softmax_node * log_p[None, :]).sum(axis=-1)       # (G,)
    E_log1mp = (softmax_node * log_1mp[None, :]).sum(axis=-1)
    E_logp_sq = (softmax_node * (log_p[None, :] ** 2)).sum(axis=-1)
    E_log1mp_sq = (softmax_node * (log_1mp[None, :] ** 2)).sum(axis=-1)
    E_logp_log1mp = (
        softmax_node * (log_p * log_1mp)[None, :]
    ).sum(axis=-1)
    Var_logp = jnp.maximum(E_logp_sq - E_logp * E_logp, 0.0)
    Var_log1mp = jnp.maximum(E_log1mp_sq - E_log1mp * E_log1mp, 0.0)
    Cov_logp_log1mp = E_logp_log1mp - E_logp * E_log1mp

    # Digamma / trigamma at (α, β, α+β).
    psi_alpha = jsp_special.digamma(alpha)
    psi_beta = jsp_special.digamma(beta)
    psi_alpha_beta = jsp_special.digamma(alpha + beta)
    pp_alpha = jsp_special.polygamma(1, alpha)
    pp_beta = jsp_special.polygamma(1, beta)
    pp_alpha_beta = jsp_special.polygamma(1, alpha + beta)

    # NLP gradient (per cell, per gene) — sign flip applied.
    g_alpha = psi_alpha - psi_alpha_beta - E_logp
    g_beta = psi_beta - psi_alpha_beta - E_log1mp

    # NLP Hessian (per cell, per gene) — sign flip applied.
    H_aa = pp_alpha - pp_alpha_beta - Var_logp
    H_bb = pp_beta - pp_alpha_beta - Var_log1mp
    H_ab = -pp_alpha_beta - Cov_logp_log1mp

    return {
        "g_alpha": g_alpha,
        "g_beta": g_beta,
        "H_aa": H_aa,
        "H_bb": H_bb,
        "H_ab": H_ab,
    }


# Vmapped over cells.  Per-cell arg shapes:
#   x       : (G,)
#   u       : (G,)
#   eta_cap : ()       — scalar offset per cell
# Gene-level (α, β) are broadcast.
_global_curvature_factors_rate_batch = jax.vmap(
    _global_curvature_factors_rate,
    in_axes=(0, 0, 0, None, None, None),
)


def global_curvature_rate_summed(
    x_map: jnp.ndarray,
    counts: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_cap: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> dict:
    """Population-summed data-side NLP gradient + Hessian in the
    ``(α, β)`` basis for the TSLN-Rate global-curvature path.

    Drop-in alternative to ``jnp.diag(jax.hessian(neg_log_post))``
    *for the data-side block* — the MVN prior block (which acts only
    on ``log r_g``) is added separately by the caller.  Closed-form,
    no autodiff, memory bounded by the Newton-kernel softmax.

    Parameters
    ----------
    x_map : jnp.ndarray, shape ``(C, G)``
        Per-cell-per-gene latent at the MAP.
    counts : jnp.ndarray, shape ``(C, G)``
        Observed counts.
    alpha, beta : jnp.ndarray, shape ``(G,)``
        Gene-level Beta shape parameters.
    eta_cap : jnp.ndarray, shape ``(C,)``
        Per-cell capture offset (zeros under no-capture).
    n_quad_nodes : int, default 60

    Returns
    -------
    dict with per-gene ``(G,)`` arrays (NLP sign):
        ``g_alpha, g_beta``     : data-side gradient.
        ``H_aa, H_bb, H_ab``    : data-side Hessian 2×2 block.

    Notes
    -----
    The third row/column of the natural-basis 3×3 Hessian (the
    ``log_rate_g`` axis) is *not* populated here — the data does not
    depend on ``rate_g``, so the (α, log r) and (β, log r) cross
    entries are identically zero, and the (log r, log r) diagonal
    entry comes entirely from the MVN prior centered at ``log r_g``.
    The caller stitches these together before Faà di Bruno chain.
    """
    per_cell = _global_curvature_factors_rate_batch(
        x_map, counts, eta_cap, alpha, beta, n_quad_nodes,
    )
    return {
        "g_alpha":  per_cell["g_alpha"].sum(axis=0),
        "g_beta":   per_cell["g_beta"].sum(axis=0),
        "H_aa":     per_cell["H_aa"].sum(axis=0),
        "H_bb":     per_cell["H_bb"].sum(axis=0),
        "H_ab":     per_cell["H_ab"].sum(axis=0),
    }

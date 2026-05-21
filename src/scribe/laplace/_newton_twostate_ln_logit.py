"""Pure-JAX Newton kernel for the TwoState-LogNormal-Logit MAP per cell.

Implements the inner Newton iteration of the Laplace-approximation
inference scheme for the ``twostate_ln_logit`` base model — Variant B
of the cross-gene TwoState extension.  Mirrors
:mod:`scribe.laplace._newton_twostate_ln_rate` in structure (Woodbury
machinery, Schur block decomposition, step-size cap, damping) but the
data-side ``(a, g_data)`` factors come from a *different* perturbation
of the Poisson-Beta marginal: instead of the latent shifting the
log-rate, it shifts the **activation log-odds**.

Mathematical background
-----------------------
The TSLN-Logit per-cell joint log-density (capture off, dropping
additive constants) is

    f(z) = Σ_g log L_PB(u_g | κ_g σ(θ_g + z_g),
                                κ_g (1 − σ(θ_g + z_g)),
                                rate_g)
           − ½ (z − μ_z)ᵀ Σ⁻¹ (z − μ_z)

where ``z ∈ ℝ^G`` is the latent (with Gaussian prior, typically
centred at zero), ``(rate_g, κ_g, θ_g)`` are the gene-level globals
(positive ``rate``, positive concentration ``κ = α + β``, real
activation log-odds ``θ``), and ``Σ = WWᵀ + diag(d)``.  The
Poisson-Beta marginal is

    L_PB(u | α, β, rate) = ∫₀¹ Po(u | rate · p) · Beta(p | α, β) dp.

For consistency with the rest of the Laplace machinery the kernel's
per-cell argument name is **``x``** (the latent variable Newton
optimizes); the plan's ``z`` ↔ this file's ``x``.  The gene-level
``μ_z`` becomes the kernel argument ``mu`` (typically zeros, but
allowed to carry a cascade-supplied centering offset).

**Critical Rev 4 invariant**: ``rate_g`` is gene-level and does NOT
depend on z.  This is what gives the cross-gene mean covariance the
non-trivial structure ``Cov(σ(θ_g + z_g), σ(θ_h + z_h))`` and makes
the gauge between ``θ_g`` and ``z_g`` exact under
``θ → θ + c, z → z − c``.  The first-round audit flagged a fatal
flaw in Rev 0 where ``rate_g`` was z-dependent and the cross-gene
mean covariance collapsed to zero.

Closed-form factors (no autodiff)
---------------------------------
With ``eta = θ + x``, ``φ = σ(eta)``, ``φ' = φ(1 − φ)``,
``φ'' = φ'(1 − 2φ)``, ``α_cg = κ φ``, ``β_cg = κ(1 − φ)``, the
z-derivative of the per-node log-integrand simplifies (rate is
z-independent) to

    ∂ log T / ∂z = φ' · D(p; κ, φ),
    D = κ [log p − log(1 − p) − (ψ(κφ) − ψ(κ(1 − φ)))]

The gradient and Hessian-diagonal under the posterior over ``p`` are

    (g_data)_g = φ'_g · E_q[D]_g
    a_g = -φ''_g · E_q[D]_g
          + φ'_g² · κ_g² · (ψ'(κ_g φ_g) + ψ'(κ_g(1 − φ_g)))
          - φ'_g² · κ_g² · Var_q(log p − log(1 − p))

The three terms have, respectively, sign-uncertain / strictly-positive
/ strictly-non-positive sign.  Like TSLN-Rate, the joint log-concavity
is not guaranteed; a defensive floor ``a = max(a_raw, _A_MIN)`` is
applied before Woodbury.

Capture-active path (PR-2 Rev 4: restricted)
--------------------------------------------
Per the Rev 4 audit, **PR-2 supports only two capture modes**:

  * ``x_only``         — no capture (``ν_c ≡ 1``).
  * ``x_only_offset``  — capture frozen at upstream MAP
    (``ν_c = exp(-η_cap_c)``); Newton optimizes only ``x``.

The capture offset enters through the Poisson log-rate
``log λ = log rate_g − η_cap`` inside the per-node Poisson log-PMF;
the **gradient and Hessian formulas above are unchanged in form**
because (a) ``log rate_g`` and ``η_cap`` are constants in ``z``, and
(b) the z-derivative of the per-node log-integrand only touches the
Beta-density factor.  Capture affects ``a`` and ``g_data`` indirectly
via the posterior-softmax weights ``q_k`` (the data normalization),
which change with ``log λ``.

The **joint** ``(z, η_cap)`` Newton path with the cross-Hessian

    H_{z, η_cap} = φ' · λ · κ · Cov_q(log p − log(1 − p), p)

is **deferred to phase 3** (plan §3.2, Rev 4).  The
``_obs_twostate_ln_logit`` adapter rejects soft-cascade η at the
config level; only fixed-offset / no-capture configurations are
permitted in PR-2.

API
---
- :func:`_twostate_ln_logit_factors(x, u, rate, kappa, theta, eta_cap,
  n_quad_nodes)` — closed-form factor dict
  ``{a, a_raw, g_data, log_marginal, ...}``.
- :func:`newton_step_x_only`,
  :func:`newton_step_x_only_offset` — single Newton steps.
- :func:`laplace_newton_loop_x_only`,
  :func:`laplace_newton_loop_x_only_offset` — fixed-iteration
  ``lax.scan``.
- :func:`laplace_newton_batch_x_only`,
  :func:`laplace_newton_batch_x_only_offset` — ``jax.vmap`` over cells.
- ``laplace_log_det_neg_H_*`` — log-det wrappers for the ELBO.
- ``twostate_ln_logit_grad_*_norm_batch`` — per-block gradient
  diagnostics.

Cross-references
----------------
- ``paper/_two_state_promoter.qmd`` §sec-twostate-tsln-logit (planned).
- ``scribe.laplace._newton_twostate_ln_rate`` — the Variant A analogue.
- ``scribe.stats._jacobi_quad.gauss_legendre_01`` — fixed quadrature
  rule.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special

from scribe.stats._jacobi_quad import gauss_legendre_01


# Sanity bounds on the Poisson log-rate to protect float32 arithmetic.
_LOG_RATE_MIN = -50.0
_LOG_RATE_MAX = 50.0

# Clamps on the activation probability ``φ`` to keep the digamma /
# trigamma evaluations away from the singular boundaries 0 / 1.  The
# raw sigmoid never exactly hits 0 or 1 but in pathological iterates
# (large |x|) the float32 rounding can land there.
#
# **Pseudo-derivative caveat (auditor R3 follow-up)**: when the raw
# sigmoid lands in the clipped region, ``φ`` is set to the boundary
# value (``_PHI_MIN`` or ``_PHI_MAX``) but the *Newton derivatives*
# ``φ' = φ(1-φ)`` and ``φ'' = φ'(1-2φ)`` continue to be computed from
# the clipped ``φ``.  This is intentional: ``φ' = 0`` at the boundary
# would freeze the Newton step (no first-order information), and the
# correct derivative of the clipped function on the saturated side
# would be zero anyway.  The clipped formulae act as a *Newton
# stabilizer* — they keep the iterate moving and never amplify
# garbage from the singular region.  Outside the clip the formulae
# are mathematically exact; only at extreme ``|θ + x|`` (where the
# raw sigmoid would underflow to 0 or saturate to 1) does the
# pseudo-derivative semantics kick in, and at those points the
# Newton step is dominated by the prior anyway because the data-side
# Hessian saturates.
_PHI_MIN = 1e-6
_PHI_MAX = 1.0 - 1e-6

# Defensive floor on the data-side Hessian diagonal ``a_g`` before
# Woodbury.  See module docstring "Closed-form factors" — log-concavity
# is not guaranteed for the Poisson-Beta marginal in z, particularly in
# the U-shaped Beta regime (small κ with φ near 0.5).  ``_A_MIN``
# matches the TSLN-Rate kernel's value so the obs-model clamp
# diagnostics carry the same semantics across the two variants.
_A_MIN = 1e-3

# Default number of Gauss-Legendre quadrature nodes on [0, 1].  Matches
# the SVI-side ``PoissonBetaCompound`` default.
_DEFAULT_K = 60


# =====================================================================
# Closed-form factor computation
# =====================================================================


def _twostate_ln_logit_factors(
    x: jnp.ndarray,
    u: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    eta_cap: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> dict:
    """Compute Poisson-Beta data-side quantities for one cell — logit variant.

    Given ``x ∈ ℝ^G`` and gene-level globals
    ``(rate_g, κ_g, θ_g) ∈ ℝ_+^G × ℝ_+^G × ℝ^G``, return the
    closed-form gradient and Hessian-diagonal of the per-cell
    log-likelihood with respect to ``x``.  The Poisson rate may be
    additionally shifted by a per-cell scalar ``eta_cap`` (PR-2
    Rev 4: fixed offset only — not optimized by Newton).

    Parameters
    ----------
    x : jnp.ndarray, shape ``(G,)``
        Current latent iterate.  Combined with ``θ`` to form the
        activation log-odds ``η_act = θ + x``.
    u : jnp.ndarray, shape ``(G,)``
        Observed counts for this cell.
    rate : jnp.ndarray, shape ``(G,)``
        Gene-level ON-production rate (positive).  z-independent.
    kappa : jnp.ndarray, shape ``(G,)``
        Gene-level Beta concentration ``α + β`` (positive).
    theta : jnp.ndarray, shape ``(G,)``
        Gene-level activation log-odds (real).  z-independent.
    eta_cap : jnp.ndarray, scalar or shape ``()``
        Per-cell capture offset (``ν_c = exp(-eta_cap)``).  Pass ``0.0``
        for the no-capture path.
    n_quad_nodes : int, default 60
        Number of Gauss-Legendre quadrature nodes on [0, 1].

    Returns
    -------
    dict with keys:
        ``a`` : shape ``(G,)``, Hessian-diagonal entry after the
            ``_A_MIN`` clamp.
        ``a_raw`` : shape ``(G,)``, raw Hessian diagonal (pre-clamp).
        ``g_data`` : shape ``(G,)``, data-side gradient ``φ' · E_q[D]``.
        ``phi`` : shape ``(G,)``, sigmoid of ``θ + x``.
        ``alpha_cg`` : shape ``(G,)``, ``κ φ`` (effective Beta α).
        ``beta_cg`` : shape ``(G,)``, ``κ (1 − φ)`` (effective Beta β).
        ``log_marginal`` : shape ``(G,)``, per-gene Poisson-Beta
            marginal log-likelihood.
        ``softmax_node`` : shape ``(G, K)``, posterior softmax weights.
        ``E_p`` : shape ``(G,)``, posterior mean of ``p`` (used by
            the cross-Hessian when joint capture Newton is added in
            phase 3).
    """
    # ---- Activation-log-odds-side quantities ---------------------------
    eta_act = theta + x
    phi_raw = jax.nn.sigmoid(eta_act)
    phi = jnp.clip(phi_raw, _PHI_MIN, _PHI_MAX)
    phi_prime = phi * (1.0 - phi)                  # σ' = φ(1−φ)
    phi_pp = phi_prime * (1.0 - 2.0 * phi)         # σ'' = φ'(1−2φ)
    kappa_safe = jnp.maximum(kappa, 1e-6)
    alpha_cg = kappa_safe * phi
    beta_cg = kappa_safe * (1.0 - phi)

    # ---- Poisson log-rate ---------------------------------------------
    # ``log λ_g = log(rate_g) − eta_cap`` (capture-on offsets the
    # Poisson scale before promoter activation).  ``rate`` is
    # gene-level; ``eta_cap`` is per-cell scalar.
    rate_safe = jnp.maximum(rate, 1e-30)
    log_rate = jnp.log(rate_safe) - eta_cap        # (G,)
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)

    # ---- Quadrature setup ----------------------------------------------
    p_node, log_w_node = gauss_legendre_01(n_quad_nodes)  # (K,), (K,)
    log_p = jnp.log(p_node)
    log_1mp = jnp.log1p(-p_node)
    M_logodds = log_p - log_1mp                    # (K,)

    # ---- Beta log-density at the nodes (z-dependent through α, β) ------
    log_beta_density = (
        (alpha_cg[:, None] - 1.0) * log_p[None, :]
        + (beta_cg[:, None] - 1.0) * log_1mp[None, :]
        - jsp_special.betaln(alpha_cg, beta_cg)[:, None]
    )                                              # (G, K)
    log_w_eff = log_w_node[None, :] + log_beta_density

    # ---- Poisson log-PMF at the nodes ---------------------------------
    log_lambda_p = log_rate[:, None] + log_p[None, :]   # (G, K)
    log_poisson = (
        u[:, None] * log_lambda_p
        - jnp.exp(log_lambda_p)
        - jsp_special.gammaln(u + 1.0)[:, None]
    )                                              # (G, K)

    # ---- Posterior over p given (z, u) -------------------------------
    log_q = log_w_eff + log_poisson                 # (G, K)
    log_marginal = jsp_special.logsumexp(log_q, axis=-1)  # (G,)
    softmax_node = jax.nn.softmax(log_q, axis=-1)   # (G, K)

    # ---- p-side moments under the posterior ---------------------------
    E_p = (softmax_node * p_node[None, :]).sum(axis=-1)            # (G,)
    E_logodds = (softmax_node * M_logodds[None, :]).sum(axis=-1)   # (G,)
    E_logodds2 = (
        softmax_node * (M_logodds[None, :] ** 2)
    ).sum(axis=-1)                                                 # (G,)
    Var_logodds = jnp.maximum(E_logodds2 - E_logodds * E_logodds, 0.0)

    # ---- φ-side digamma / trigamma evaluations ------------------------
    psi_alpha = jsp_special.digamma(alpha_cg)
    psi_beta = jsp_special.digamma(beta_cg)
    psi_p_alpha = jsp_special.polygamma(1, alpha_cg)  # trigamma
    psi_p_beta = jsp_special.polygamma(1, beta_cg)

    # ---- D, E_q[D], and the gradient ---------------------------------
    # D = κ [log p − log(1−p) − (ψ(κφ) − ψ(κ(1−φ)))]
    # E_q[D] = κ · E_logodds − κ · (ψ(κφ) − ψ(κ(1−φ))).
    E_D = kappa_safe * E_logodds - kappa_safe * (psi_alpha - psi_beta)
    g_data = phi_prime * E_D

    # ---- Hessian-diagonal: three-term decomposition ------------------
    # a_raw = -φ'' · E_q[D]
    #         + φ'² · κ² · (ψ'(κφ) + ψ'(κ(1−φ)))     (always > 0)
    #         - φ'² · κ² · Var_q(log p − log(1−p))   (always ≤ 0)
    psi_prime_sum = psi_p_alpha + psi_p_beta
    a_raw = (
        -phi_pp * E_D
        + (phi_prime ** 2) * (kappa_safe ** 2) * psi_prime_sum
        - (phi_prime ** 2) * (kappa_safe ** 2) * Var_logodds
    )
    a = jnp.maximum(a_raw, _A_MIN)

    return {
        "a": a,
        "a_raw": a_raw,
        "g_data": g_data,
        "phi": phi,
        "phi_prime": phi_prime,
        "alpha_cg": alpha_cg,
        "beta_cg": beta_cg,
        "E_p": E_p,
        "log_marginal": log_marginal,
        "softmax_node": softmax_node,
    }


# =====================================================================
# Woodbury helpers (inlined; identical to TSLN-Rate / NBLN)
# =====================================================================


def _woodbury_factors(
    W: jnp.ndarray,
    d: jnp.ndarray,
    a_diag: jnp.ndarray,
    damping: float,
) -> dict:
    """Pre-compute Woodbury factors for ``A = diag(a + 1/d + damping) − V K⁻¹ V'``.

    Mirrors :func:`scribe.laplace._newton_twostate_ln_rate._woodbury_factors`
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


def laplace_log_det_neg_H_x_only(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``log det(-H_x)`` at the MAP for the x-only (no capture) path."""
    fac = _twostate_ln_logit_factors(
        x_map, u, rate, kappa, theta, jnp.asarray(0.0), n_quad_nodes
    )
    factors = _woodbury_factors(W, d, fac["a"], damping=0.0)
    return _log_det_A(factors)


laplace_log_det_neg_H_batch_x_only = jax.vmap(
    laplace_log_det_neg_H_x_only,
    in_axes=(0, 0, None, None, None, None, None, None),
)


def laplace_log_det_neg_H_x_only_offset(
    x_map: jnp.ndarray,
    eta_offset: jnp.ndarray,
    u: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``log det(-H_x)`` at the MAP for the x-only-with-offset path.

    The η block disappears (η is a fixed offset, not a Newton latent);
    the log-det is just ``log det A`` with ``a`` evaluated at
    ``log λ = log rate − η_offset`` inside the factor function.
    """
    fac = _twostate_ln_logit_factors(
        x_map, u, rate, kappa, theta, eta_offset, n_quad_nodes
    )
    factors = _woodbury_factors(W, d, fac["a"], damping=0.0)
    return _log_det_A(factors)


laplace_log_det_neg_H_batch_x_only_offset = jax.vmap(
    laplace_log_det_neg_H_x_only_offset,
    in_axes=(0, 0, 0, None, None, None, None, None, None),
)


# =====================================================================
# Newton step — x-only (no capture)
# =====================================================================


def newton_step_x_only(
    x: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Newton step on ``x`` only (no capture anchor) — logit variant.

    ``A = diag(a + 1/d) − V K⁻¹ V'``; Newton step ``δ = A⁻¹ (g_data −
    Σ⁻¹(x − μ))``.
    """
    fac = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, jnp.asarray(0.0), n_quad_nodes
    )
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
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """``n_iters`` Newton steps on ``x`` for one cell (no capture).

    Returns
    -------
    x_map, final_grad_norm, log_det_neg_H, log_marginal_sum, a_raw_min
    """

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only(
            x_, u, mu, W, d, rate, kappa, theta,
            damping, max_step, n_quad_nodes,
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    fac_final = _twostate_ln_logit_factors(
        x_final, u, rate, kappa, theta, jnp.asarray(0.0), n_quad_nodes
    )
    factors = _woodbury_factors(W, d, fac_final["a"], damping)
    log_det_neg_H = _log_det_A(factors)
    log_marginal_sum = jnp.sum(fac_final["log_marginal"])
    a_raw_min = jnp.min(fac_final["a_raw"])
    return x_final, final_grad, log_det_neg_H, log_marginal_sum, a_raw_min


laplace_newton_batch_x_only = jax.vmap(
    laplace_newton_loop_x_only,
    in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None),
)


# =====================================================================
# Newton step — x-only-with-offset (capture frozen)
# =====================================================================


def newton_step_x_only_offset(
    x: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    eta_offset: jnp.ndarray,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Newton step on ``x`` with fixed ``η_offset`` — logit variant.

    Data factors evaluated with ``eta_cap = η_offset`` (the Poisson log-
    rate is shifted by ``-η_offset`` inside the factor function); the
    MVN prior on ``x`` is unchanged.  No η block ⇒ no rigid-translation
    gauge in the Hessian; safe to raise ``max_step`` to 20-50.
    """
    fac = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, eta_offset, n_quad_nodes
    )
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


def laplace_newton_loop_x_only_offset(
    x_init: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    eta_offset: jnp.ndarray,
    n_iters: int,
    damping: float = 1e-4,
    max_step: float = 5.0,
    n_quad_nodes: int = _DEFAULT_K,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
]:
    """``n_iters`` Newton steps on ``x`` with fixed ``η_offset``.

    Returns
    -------
    x_map, final_grad_norm, log_det_neg_H, log_marginal_sum, a_raw_min
    """

    def step(carry, _):
        x_, _grad = carry
        x_new, grad_norm = newton_step_x_only_offset(
            x_, u, mu, W, d, rate, kappa, theta, eta_offset,
            damping, max_step, n_quad_nodes,
        )
        return (x_new, grad_norm), None

    init = (x_init, jnp.asarray(jnp.inf, dtype=x_init.dtype))
    (x_final, final_grad), _ = jax.lax.scan(
        step, init, None, length=n_iters
    )

    fac_final = _twostate_ln_logit_factors(
        x_final, u, rate, kappa, theta, eta_offset, n_quad_nodes
    )
    factors = _woodbury_factors(W, d, fac_final["a"], damping)
    log_det_neg_H = _log_det_A(factors)
    log_marginal_sum = jnp.sum(fac_final["log_marginal"])
    a_raw_min = jnp.min(fac_final["a_raw"])
    return x_final, final_grad, log_det_neg_H, log_marginal_sum, a_raw_min


laplace_newton_batch_x_only_offset = jax.vmap(
    laplace_newton_loop_x_only_offset,
    in_axes=(0, 0, None, None, None, None, None, None, 0, None, None, None, None),
)


# =====================================================================
# Per-block gradient-norm diagnostics
# =====================================================================
#
# Used by the obs-model's loss_fn to surface a per-block max/p99/med
# triplet in the progress display.  Mirrors the TSLN-Rate side
# (``twostate_ln_rate_grad_*_norm_batch``) one-for-one.


def _twostate_ln_logit_grad_x_only_norm(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``L∞`` of ``∇_x f`` at the x-only TSLN-Logit MAP."""
    fac = _twostate_ln_logit_factors(
        x_map, u, rate, kappa, theta, jnp.asarray(0.0), n_quad_nodes
    )
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


twostate_ln_logit_grad_x_only_norm_batch = jax.vmap(
    _twostate_ln_logit_grad_x_only_norm,
    in_axes=(0, 0, None, None, None, None, None, None, None),
)


def _twostate_ln_logit_grad_x_only_offset_norm(
    x_map: jnp.ndarray,
    u: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    eta_offset: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> jnp.ndarray:
    """``L∞`` of ``∇_x f`` at the x-only-with-offset TSLN-Logit MAP."""
    fac = _twostate_ln_logit_factors(
        x_map, u, rate, kappa, theta, eta_offset, n_quad_nodes
    )
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


twostate_ln_logit_grad_x_only_offset_norm_batch = jax.vmap(
    _twostate_ln_logit_grad_x_only_offset_norm,
    in_axes=(0, 0, None, None, None, None, None, None, 0, None),
)


# =====================================================================
# Closed-form global-parameter gradient + Hessian-diagonal at MAP
# =====================================================================
#
# Hand-derived per-cell-per-gene curvature for the three gene-level
# globals ``(rate, κ, θ)`` in their NATURAL spaces (log_rate, κ, θ).
# Drop-in replacement for ``jnp.diag(jax.hessian(neg_log_post))`` used
# by ``_obs_twostate_ln_logit.compute_global_uncertainty``.
#
# Math: see ``paper/_two_state_promoter.qmd``
# §sec-twostate-tsln-logit-global-uncertainty.  Key facts:
#
#   * **θ (eta_anchor)** enters through θ + x exactly, so the per-cell
#     curvature wrt θ is identical to the Newton-kernel ``a_raw``.
#     Free piggyback on existing kernel state.
#   * **rate** enters only the Poisson log-rate (z-independent), so the
#     curvature wrt ``log rate`` is identical in form to the TSLN-Rate
#     Newton factor ``a = λE[p] − λ²Var[p]``.
#   * **κ** enters only the Beta shape via α = κφ, β = κ(1−φ); the
#     curvature involves digamma/trigamma at (α, β, κ) plus posterior
#     variances over ``log p`` and ``log(1−p)``.
#
# Returns gradients AND curvatures so the downstream chain-rule into
# unconstrained ``*_loc`` space (via pos_forward) can include the
# off-MAP gradient term (negligible at perfect convergence, kept for
# safety).
#
# Output convention: all returned ``H_*`` values are curvature of the
# **negative** log-likelihood (positive at a local minimum).


def _global_curvature_factors_logit(
    x: jnp.ndarray,
    u: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    eta_cap: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> dict:
    """Per-cell-per-gene closed-form gradient + Hessian-diagonal for
    the three TSLN-Logit gene globals (rate, κ, θ).

    Reuses the existing Newton-kernel factor function to get
    ``softmax_node`` and the digamma/trigamma evaluations, then
    augments with the additional moments needed for the global-
    curvature derivation (``Var_q[log p]``, ``Var_q[log(1-p)]``,
    ``Cov_q[log p, log(1-p)]``, ``Var_q[p]``).

    Returns a dict with per-gene gradients and curvatures.  All
    quantities are for a SINGLE cell (vmap over cells happens at
    the caller).  Sign convention: ``H_*`` is curvature of the
    NEGATIVE log-likelihood, so positive values indicate good
    curvature.
    """
    # Reuse existing factor for softmax_node, α, β, φ, log_marginal.
    fac = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, eta_cap, n_quad_nodes,
    )
    softmax_node = fac["softmax_node"]       # (G, K)
    alpha_cg = fac["alpha_cg"]               # (G,) = κφ
    beta_cg = fac["beta_cg"]                 # (G,) = κ(1−φ)
    phi = fac["phi"]                         # (G,) = σ(θ + x), clipped
    E_p = fac["E_p"]                         # (G,)

    # Re-derive the quadrature node arrays (cheap).
    p_node, _ = gauss_legendre_01(n_quad_nodes)
    log_p = jnp.log(p_node)                  # (K,)
    log_1mp = jnp.log1p(-p_node)             # (K,)

    # ---- Additional moments under the posterior --------------------
    # Var_q[p]
    E_p2 = (softmax_node * (p_node[None, :] ** 2)).sum(axis=-1)
    Var_p = jnp.maximum(E_p2 - E_p * E_p, 0.0)

    # E_q[log p], E_q[log(1−p)]
    E_logp = (softmax_node * log_p[None, :]).sum(axis=-1)
    E_log1mp = (softmax_node * log_1mp[None, :]).sum(axis=-1)
    # E_q[(log p)²], etc.
    E_logp2 = (softmax_node * (log_p[None, :] ** 2)).sum(axis=-1)
    E_log1mp2 = (softmax_node * (log_1mp[None, :] ** 2)).sum(axis=-1)
    # E_q[log p · log(1−p)]
    log_p_outer_log_1mp = log_p * log_1mp    # (K,) elementwise
    E_logp_log1mp = (
        softmax_node * log_p_outer_log_1mp[None, :]
    ).sum(axis=-1)
    Var_logp = jnp.maximum(E_logp2 - E_logp * E_logp, 0.0)
    Var_log1mp = jnp.maximum(E_log1mp2 - E_log1mp * E_log1mp, 0.0)
    Cov_logp_log1mp = E_logp_log1mp - E_logp * E_log1mp

    # ---- Digamma / trigamma at (α, β, κ) ----------------------------
    kappa_safe = jnp.maximum(kappa, 1e-6)
    psi_alpha = jsp_special.digamma(alpha_cg)
    psi_beta = jsp_special.digamma(beta_cg)
    psi_kappa = jsp_special.digamma(kappa_safe)
    psi_p_alpha = jsp_special.polygamma(1, alpha_cg)
    psi_p_beta = jsp_special.polygamma(1, beta_cg)
    psi_p_kappa = jsp_special.polygamma(1, kappa_safe)

    one_minus_phi = 1.0 - phi

    # ---- Eta_anchor: piggyback on Newton kernel `a_raw` -------------
    # ∂/∂θ = ∂/∂x exactly (θ and x enter only via θ + x), so the
    # per-cell curvature wrt θ is the kernel's ``a_raw`` and the
    # gradient is the kernel's ``g_data`` (with sign).
    g_eta_anchor = -fac["g_data"]            # ∂(-log L)/∂θ
    H_eta_anchor = fac["a_raw"]              # curvature of -log L wrt θ

    # ---- Rate (work in log-rate space) ------------------------------
    # ∂ log T / ∂(log r) = u − λ p; only the Poisson piece depends.
    # ∂² log T / ∂(log r)² = -λ p.
    # log-marginal identity:
    #   ∂² log L_PB / ∂(log r)² = E[-λp] + Var[u − λp] = -λE[p] + λ²Var[p]
    # so the curvature of -log L wrt log r is λE[p] − λ²Var[p].
    # Same structural form as the TSLN-Rate Newton factor `a`.
    rate_safe = jnp.maximum(rate, 1e-30)
    log_rate = jnp.log(rate_safe) - eta_cap
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    lambda_cg = jnp.exp(log_rate)            # (G,)
    g_log_rate = -(u - lambda_cg * E_p)      # ∂(-log L)/∂(log r)
    H_log_rate = lambda_cg * E_p - lambda_cg * lambda_cg * Var_p

    # ---- Kappa ------------------------------------------------------
    # ∂ log T / ∂κ = D_κ(p) + C_κ where
    #   D_κ(p) = φ log p + (1-φ) log(1-p)        (p-dependent)
    #   C_κ    = -[φ ψ(α) + (1-φ) ψ(β) - ψ(κ)]   (p-independent)
    # ∂² log T / ∂κ² = -φ² ψ'(α) - (1-φ)² ψ'(β) + ψ'(κ)   (constant in p)
    # log-marginal identity:
    #   ∂² log L_PB / ∂κ² =
    #       (-φ²ψ'(α) - (1-φ)²ψ'(β) + ψ'(κ))
    #     + Var_q[D_κ]
    #   with Var_q[D_κ] = φ² Var[log p] + (1-φ)² Var[log(1-p)]
    #                   + 2 φ(1-φ) Cov[log p, log(1-p)].
    # Curvature of -log L wrt κ flips the sign of the above.
    g_kappa = -(
        phi * (E_logp - psi_alpha)
        + one_minus_phi * (E_log1mp - psi_beta)
        + psi_kappa
    )
    Var_D_kappa = (
        (phi ** 2) * Var_logp
        + (one_minus_phi ** 2) * Var_log1mp
        + 2.0 * phi * one_minus_phi * Cov_logp_log1mp
    )
    H_kappa = (
        (phi ** 2) * psi_p_alpha
        + (one_minus_phi ** 2) * psi_p_beta
        - psi_p_kappa
        - Var_D_kappa
    )

    return {
        "g_eta_anchor": g_eta_anchor,
        "H_eta_anchor": H_eta_anchor,
        "g_log_rate": g_log_rate,
        "H_log_rate": H_log_rate,
        "g_kappa": g_kappa,
        "H_kappa": H_kappa,
        # Diagnostic / verifiable intermediates.
        "lambda_cg": lambda_cg,
        "E_p": E_p,
        "Var_p": Var_p,
    }


# Vmapped over cells; per-cell argument shapes:
#   x       : (G,)   — per-cell latent at MAP
#   u       : (G,)   — per-cell counts
#   eta_cap : ()     — per-cell offset (scalar) or (G,) if per-gene; here per-cell
# Gene-level (rate, κ, θ) are broadcast (None in_axes).
_global_curvature_factors_logit_batch = jax.vmap(
    _global_curvature_factors_logit,
    in_axes=(0, 0, None, None, None, 0, None),
)


def global_curvature_logit_summed(
    x_map: jnp.ndarray,
    counts: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    eta_cap: jnp.ndarray,
    n_quad_nodes: int = _DEFAULT_K,
) -> dict:
    """Population-summed gradients and Hessian-diagonals for the three
    TSLN-Logit gene globals.

    Drop-in alternative to ``jnp.diag(jax.hessian(neg_log_post))`` —
    closed-form (no autodiff), constant memory in ``G``, and
    ~10–50× faster than the chunked-autodiff fallback at production
    scale.

    Parameters
    ----------
    x_map : jnp.ndarray, shape ``(C, G)``
        Per-cell-per-gene latent at the MAP.
    counts : jnp.ndarray, shape ``(C, G)``
        Observed counts.
    rate, kappa : jnp.ndarray, shape ``(G,)``
        Gene-level positive globals.
    theta : jnp.ndarray, shape ``(G,)``
        Gene-level activation log-odds.
    eta_cap : jnp.ndarray, shape ``(C,)``
        Per-cell capture offset (use zeros for no-capture).
    n_quad_nodes : int, default 60.

    Returns
    -------
    dict with per-gene ``(G,)`` arrays:
        ``g_log_rate, H_log_rate`` — curvature wrt log r.
        ``g_kappa, H_kappa`` — curvature wrt κ.
        ``g_eta_anchor, H_eta_anchor`` — curvature wrt θ.
    """
    per_cell = _global_curvature_factors_logit_batch(
        x_map, counts, rate, kappa, theta, eta_cap, n_quad_nodes,
    )
    return {
        "g_log_rate":    per_cell["g_log_rate"].sum(axis=0),
        "H_log_rate":    per_cell["H_log_rate"].sum(axis=0),
        "g_kappa":       per_cell["g_kappa"].sum(axis=0),
        "H_kappa":       per_cell["H_kappa"].sum(axis=0),
        "g_eta_anchor":  per_cell["g_eta_anchor"].sum(axis=0),
        "H_eta_anchor":  per_cell["H_eta_anchor"].sum(axis=0),
    }

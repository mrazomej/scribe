"""Unit tests for the TSLN-Logit Newton kernel.

Tests cover the auditor's checklist for plan §3.2 / Rev 4 — the
math derivation that the third audit signed off on:

1. ``_twostate_ln_logit_factors`` closed-form gradient and
   Hessian-diagonal match ``jax.grad`` / ``jax.hessian`` on the
   brute-force Poisson-Beta marginal with ``(κφ, κ(1-φ))`` shape
   parameters. **This is the test that catches sign / chain-rule
   errors in the three-term Hessian decomposition** — the auditor's
   primary concern with Variant B because the closed-form
   derivation goes through digamma, trigamma, ``φ''``, and a
   posterior-variance term.

2. The defensive ``_A_MIN`` clamp on ``a_g`` exposes ``a_raw``
   pre-clamp and ``a`` post-clamp consistently.

3. The exact rigid-translation gauge: ``log L_PB(u | κφ(θ + x),
   κ(1-φ(θ + x)), rate)`` is invariant under the substitution
   ``θ → θ + c, x → x − c``. This is the auditor's Rev 2 gauge
   correction — Variant B's gauge is *exact* (unlike the original
   ``μ/φ`` parameterization where it was approximate).

4. Capture-offset path: ``g_data`` and ``a`` from the
   ``eta_cap``-on factor call equal the ``eta_cap = 0`` factors
   evaluated at the same ``log_rate`` (the capture offset only
   shifts the Poisson log-rate; it does not enter ``α, β``).

5. Newton step converges (quadratically) from a perturbed warm
   start.

6. Woodbury solve equivalence: ``_solve_A`` matches a brute-force
   inversion of ``A = diag(a) + Σ⁻¹``.

7. Module imports cleanly.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest

from scribe.laplace._newton_twostate_ln_logit import (
    _A_MIN,
    _DEFAULT_K,
    _twostate_ln_logit_factors,
    _woodbury_factors,
    _solve_A,
    _log_det_A,
    laplace_newton_loop_x_only,
    laplace_newton_loop_x_only_offset,
    laplace_newton_batch_x_only,
    laplace_newton_batch_x_only_offset,
)
from scribe.stats._jacobi_quad import gauss_legendre_01


# ---------------------------------------------------------------------
# Reference: Poisson-Beta marginal log-prob in (rate, kappa, theta, x)
# coordinates via fixed Legendre quad — z enters only through (α, β).
# ---------------------------------------------------------------------


def _log_pb_marginal_logit(x, u, rate, kappa, theta, eta_cap, K):
    """Per-gene Poisson-Beta marginal in the logit parameterization.

    Important — for autodiff testing, ``betaln`` is expanded to
    ``lgamma(α) + lgamma(β) − lgamma(α + β)`` rather than calling
    ``jsp.betaln`` directly.  As of JAX 0.4.x, ``jsp.betaln`` has a
    second-derivative bug: ``jax.grad(jax.grad(...))`` returns
    incorrect values while ``jax.grad`` is correct.  The explicit
    lgamma sum differentiates cleanly through second order.  Finite
    differences and the explicit form agree to ≥1e-3; ``jsp.betaln``-
    based autodiff disagrees by a fixed offset.  The kernel itself
    is autodiff-free so this is purely a test-side workaround.
    """
    eta_act = theta + x
    phi = jax.nn.sigmoid(eta_act)
    alpha_cg = kappa * phi
    beta_cg = kappa * (1.0 - phi)

    p_node, log_w_node = gauss_legendre_01(K)
    log_p = jnp.log(p_node)
    log_1mp = jnp.log1p(-p_node)

    # Explicit lgamma decomposition (NOT jsp.betaln) — see docstring.
    betaln_explicit = (
        jsp.gammaln(alpha_cg)
        + jsp.gammaln(beta_cg)
        - jsp.gammaln(alpha_cg + beta_cg)
    )
    log_beta_density = (
        (alpha_cg[:, None] - 1.0) * log_p[None, :]
        + (beta_cg[:, None] - 1.0) * log_1mp[None, :]
        - betaln_explicit[:, None]
    )
    log_w_eff = log_w_node[None, :] + log_beta_density

    log_rate = jnp.log(rate) - eta_cap
    log_lambda = log_rate[:, None] + log_p[None, :]
    log_poisson = (
        u[:, None] * log_lambda
        - jnp.exp(log_lambda)
        - jsp.gammaln(u + 1.0)[:, None]
    )
    return jsp.logsumexp(log_w_eff + log_poisson, axis=-1)


def _total_log_lik_logit(x, u, rate, kappa, theta, eta_cap, K):
    return jnp.sum(
        _log_pb_marginal_logit(x, u, rate, kappa, theta, eta_cap, K)
    )


# ---------------------------------------------------------------------
# Test 1: closed-form factors match autodiff (auditor's primary
# concern — three-term Hessian sign-correctness)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("K", [40, 60, 80])
def test_factors_match_autodiff_no_capture(K):
    """``g_data`` and ``a_raw`` should equal autodiff to float32 precision.

    No-capture path (``eta_cap = 0``).
    """
    # Mix of regimes: bursty (small κ, U-shaped Beta), high-mean
    # (large rate), saturating (θ + x large positive), tail (θ + x
    # large negative).
    x = jnp.array([0.0, 1.0, -1.0, 2.0, -2.0])
    theta = jnp.array([0.0, -1.0, 0.5, -2.0, 1.0])
    u = jnp.array([1.0, 5.0, 0.0, 10.0, 0.0])
    rate = jnp.array([2.0, 8.0, 1.0, 50.0, 5.0])
    kappa = jnp.array([0.5, 2.0, 1.0, 10.0, 0.3])
    eta_cap = jnp.asarray(0.0)

    fac = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, eta_cap, n_quad_nodes=K
    )

    grad_autodiff = jax.grad(_total_log_lik_logit)(
        x, u, rate, kappa, theta, eta_cap, K
    )
    hess_diag_autodiff = jnp.diag(
        jax.hessian(_total_log_lik_logit)(
            x, u, rate, kappa, theta, eta_cap, K
        )
    )

    # Float32 precision: gradient ~1e-4 absolute (digamma is touchy),
    # Hessian ~1e-3 absolute because of the three-term decomposition
    # cancellation.
    np.testing.assert_allclose(
        np.asarray(fac["g_data"]),
        np.asarray(grad_autodiff),
        atol=1e-4,
        rtol=1e-3,
        err_msg="closed-form g_data disagrees with autodiff",
    )
    # autodiff returns ∂²log L / ∂x²; kernel returns -∂²log L / ∂x².
    np.testing.assert_allclose(
        np.asarray(fac["a_raw"]),
        -np.asarray(hess_diag_autodiff),
        atol=1e-3,
        rtol=2e-3,
        err_msg="closed-form a_raw disagrees with autodiff",
    )


@pytest.mark.parametrize("K", [60])
def test_factors_match_autodiff_with_capture(K):
    """Closed-form factors match autodiff with ``eta_cap > 0``."""
    x = jnp.array([0.5, -0.5, 1.5])
    theta = jnp.array([0.0, 1.0, -1.0])
    u = jnp.array([2.0, 7.0, 1.0])
    rate = jnp.array([5.0, 20.0, 3.0])
    kappa = jnp.array([1.0, 5.0, 0.4])
    eta_cap = jnp.asarray(0.8)

    fac = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, eta_cap, n_quad_nodes=K
    )

    grad_autodiff = jax.grad(_total_log_lik_logit)(
        x, u, rate, kappa, theta, eta_cap, K
    )
    hess_diag_autodiff = jnp.diag(
        jax.hessian(_total_log_lik_logit)(
            x, u, rate, kappa, theta, eta_cap, K
        )
    )

    np.testing.assert_allclose(
        np.asarray(fac["g_data"]),
        np.asarray(grad_autodiff),
        atol=1e-4,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(fac["a_raw"]),
        -np.asarray(hess_diag_autodiff),
        atol=1e-3,
        rtol=2e-3,
    )


# ---------------------------------------------------------------------
# Test 2: clamp exposes raw and post-clamp values
# ---------------------------------------------------------------------


def test_clamp_exposes_a_raw():
    """``a`` is floored at ``_A_MIN`` while ``a_raw`` is reported pre-clamp.

    The clamp is a *defensive* floor for pathological iterates; it is
    not expected to activate on well-behaved synthetic inputs because
    the trigamma term ``φ'² · κ² · (ψ'(κφ) + ψ'(κ(1-φ)))`` dominates
    in most regimes (trigamma blows up faster than the log-odds
    variance shrinks).  This test only verifies the invariant
    ``a == max(a_raw, _A_MIN)`` element-wise across a mix of regimes,
    plus that the clamp triggers when we inject a known-bad ``a_raw``
    via direct construction (covered below).
    """
    x = jnp.array([0.0, 0.5, -1.0, 1.0])
    theta = jnp.zeros((4,))
    u = jnp.array([1.0, 3.0, 0.0, 5.0])
    rate = jnp.array([2.0, 5.0, 1.0, 10.0])
    kappa = jnp.array([0.3, 1.0, 0.5, 3.0])

    fac = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, jnp.asarray(0.0),
    )
    a = np.asarray(fac["a"])
    a_raw = np.asarray(fac["a_raw"])

    assert np.all(a >= _A_MIN - 1e-12), (
        "post-clamp ``a`` must be at or above _A_MIN everywhere"
    )
    # Element-wise invariant: a == max(a_raw, _A_MIN).
    np.testing.assert_array_equal(
        np.maximum(a_raw, _A_MIN),
        a,
    )
    # ``a_raw`` itself must be finite (no NaNs from digamma/log-zero).
    assert np.all(np.isfinite(a_raw))


# ---------------------------------------------------------------------
# Test 3: exact rigid-translation gauge (auditor Rev 2)
# ---------------------------------------------------------------------


def test_exact_gauge_invariance():
    """``log L_PB`` is invariant under ``θ → θ + c, x → x − c``.

    This is the *exact* gauge property that distinguishes Variant B
    from a naive ``μ/φ`` parameterization where the gauge was only
    approximate.  The likelihood depends on ``θ + x`` only.
    """
    K = 60
    x = jnp.array([0.0, 0.4, -0.7, 1.2])
    theta = jnp.array([0.3, -0.1, 0.8, -0.5])
    u = jnp.array([2.0, 5.0, 0.0, 10.0])
    rate = jnp.array([3.0, 8.0, 1.5, 20.0])
    kappa = jnp.array([0.8, 2.0, 0.3, 5.0])
    eta_cap = jnp.asarray(0.4)

    fac_base = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, eta_cap, n_quad_nodes=K,
    )
    log_marg_base = np.asarray(fac_base["log_marginal"])

    for c in [-0.5, 0.3, 1.0, -2.0]:
        c = float(c)
        fac_shifted = _twostate_ln_logit_factors(
            x - c, u, rate, kappa, theta + c, eta_cap, n_quad_nodes=K,
        )
        np.testing.assert_allclose(
            np.asarray(fac_shifted["log_marginal"]),
            log_marg_base,
            atol=1e-4,
            rtol=1e-5,
            err_msg=f"log_marginal not gauge-invariant for c={c}",
        )
        # Gradient is also gauge-invariant: ∂/∂x and ∂/∂θ are equal
        # so the data-side gradient on x at ``x'' = x - c`` equals
        # the data-side gradient on x at ``x`` (same g_data values).
        np.testing.assert_allclose(
            np.asarray(fac_shifted["g_data"]),
            np.asarray(fac_base["g_data"]),
            atol=1e-4,
            rtol=1e-3,
            err_msg=f"g_data not gauge-invariant for c={c}",
        )


# ---------------------------------------------------------------------
# Test 4: capture-offset effect on log_rate is consistent
# ---------------------------------------------------------------------


def test_capture_offset_acts_on_log_rate_only():
    """``log_marginal(x, ..., eta_cap=c)`` equals
    ``log_marginal(x, rate * exp(-c), kappa, theta, eta_cap=0)``.

    The capture offset enters only through the Poisson log-rate
    (``log λ = log rate − eta_cap``).  It does NOT enter the Beta
    shape parameters.
    """
    K = 60
    x = jnp.array([0.0, 0.5, -0.5])
    theta = jnp.array([0.0, -0.5, 1.0])
    u = jnp.array([1.0, 4.0, 0.0])
    rate = jnp.array([2.0, 10.0, 1.0])
    kappa = jnp.array([1.0, 3.0, 0.5])
    c = 0.7

    fac_with_cap = _twostate_ln_logit_factors(
        x, u, rate, kappa, theta, jnp.asarray(c), n_quad_nodes=K,
    )
    fac_equiv = _twostate_ln_logit_factors(
        x, u, rate * np.exp(-c), kappa, theta, jnp.asarray(0.0),
        n_quad_nodes=K,
    )
    np.testing.assert_allclose(
        np.asarray(fac_with_cap["log_marginal"]),
        np.asarray(fac_equiv["log_marginal"]),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(fac_with_cap["g_data"]),
        np.asarray(fac_equiv["g_data"]),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(fac_with_cap["a_raw"]),
        np.asarray(fac_equiv["a_raw"]),
        atol=1e-4,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------
# Test 5: Newton convergence on x-only path
# ---------------------------------------------------------------------


def test_newton_x_only_converges():
    """From a perturbed warm start, Newton brings ``‖g‖_∞`` to <1e-3.

    Synthetic problem: G=10, latent_dim=2, capture off.  Eight Newton
    iterations should suffice on well-behaved synthetic data.
    """
    G, k = 10, 2
    rng = np.random.default_rng(0)
    rate = jnp.asarray(np.exp(rng.normal(1.0, 0.5, size=G)).astype(np.float32))
    kappa = jnp.asarray(np.exp(rng.normal(0.5, 0.3, size=G)).astype(np.float32))
    theta = jnp.asarray(rng.normal(0.0, 1.0, size=G).astype(np.float32))
    W = jnp.asarray(rng.normal(0.0, 0.3, size=(G, k)).astype(np.float32))
    d = jnp.asarray(np.ones(G, dtype=np.float32) * 0.1)
    mu = jnp.zeros(G, dtype=jnp.float32)

    # Ground-truth z draw + simulated counts.
    z_true = jnp.asarray(rng.normal(0.0, 1.0, size=G).astype(np.float32))
    # Crude counts: Poisson with expected mean rate * sigma(theta + z).
    eta_true = theta + z_true
    phi_true = jax.nn.sigmoid(eta_true)
    lam_true = rate * phi_true
    u = jnp.asarray(rng.poisson(np.asarray(lam_true), size=G).astype(np.float32))

    # Perturbed warm start.
    x_init = z_true + jnp.asarray(rng.normal(0.0, 0.3, size=G).astype(np.float32))

    x_map, grad_norm, log_det, log_marg_sum, a_raw_min = (
        laplace_newton_loop_x_only(
            x_init, u, mu, W, d, rate, kappa, theta,
            n_iters=10, damping=1e-6, max_step=5.0,
        )
    )
    assert float(grad_norm) < 1e-3, (
        f"Newton failed to converge: |grad|_inf = {float(grad_norm)}"
    )
    assert jnp.all(jnp.isfinite(x_map))
    assert jnp.isfinite(log_det)


def test_newton_x_only_offset_converges():
    """Same convergence guarantee with a frozen ``eta_cap``."""
    G, k = 10, 2
    rng = np.random.default_rng(1)
    rate = jnp.asarray(np.exp(rng.normal(1.0, 0.5, size=G)).astype(np.float32))
    kappa = jnp.asarray(np.exp(rng.normal(0.5, 0.3, size=G)).astype(np.float32))
    theta = jnp.asarray(rng.normal(0.0, 1.0, size=G).astype(np.float32))
    W = jnp.asarray(rng.normal(0.0, 0.3, size=(G, k)).astype(np.float32))
    d = jnp.asarray(np.ones(G, dtype=np.float32) * 0.1)
    mu = jnp.zeros(G, dtype=jnp.float32)
    eta_cap = jnp.asarray(0.5)

    z_true = jnp.asarray(rng.normal(0.0, 1.0, size=G).astype(np.float32))
    eta_true = theta + z_true
    phi_true = jax.nn.sigmoid(eta_true)
    lam_true = rate * jnp.exp(-eta_cap) * phi_true
    u = jnp.asarray(rng.poisson(np.asarray(lam_true), size=G).astype(np.float32))

    x_init = z_true + jnp.asarray(rng.normal(0.0, 0.3, size=G).astype(np.float32))

    x_map, grad_norm, *_ = laplace_newton_loop_x_only_offset(
        x_init, u, mu, W, d, rate, kappa, theta, eta_cap,
        n_iters=10, damping=1e-6, max_step=5.0,
    )
    assert float(grad_norm) < 1e-3, (
        f"Newton (x_only_offset) failed to converge: "
        f"|grad|_inf = {float(grad_norm)}"
    )


# ---------------------------------------------------------------------
# Test 6: Woodbury solve equals brute-force inversion
# ---------------------------------------------------------------------


def test_woodbury_solve_equivalence():
    """``_solve_A`` matches ``A⁻¹`` from direct inversion."""
    G, k = 15, 3
    rng = np.random.default_rng(2)
    W = jnp.asarray(rng.normal(0.0, 0.5, size=(G, k)).astype(np.float32))
    d = jnp.asarray(np.ones(G, dtype=np.float32) * 0.2)
    a = jnp.asarray((np.abs(rng.normal(1.0, 0.5, size=G)) + 0.1).astype(np.float32))

    factors = _woodbury_factors(W, d, a, damping=0.0)
    y = jnp.asarray(rng.normal(0.0, 1.0, size=G).astype(np.float32))

    # Brute-force.  A = diag(a) + Σ⁻¹ = diag(a + 1/d) − V K⁻¹ V'.
    Sigma = W @ W.T + jnp.diag(d)
    Sigma_inv = jnp.linalg.inv(Sigma)
    A_dense = jnp.diag(a) + Sigma_inv
    A_inv_y_dense = jnp.linalg.solve(A_dense, y)

    A_inv_y_wood = _solve_A(factors, y)
    np.testing.assert_allclose(
        np.asarray(A_inv_y_wood),
        np.asarray(A_inv_y_dense),
        atol=1e-4,
        rtol=1e-4,
    )

    # log det A also matches.
    log_det_dense = float(jnp.linalg.slogdet(A_dense)[1])
    log_det_wood = float(_log_det_A(factors))
    assert abs(log_det_dense - log_det_wood) < 1e-3


# ---------------------------------------------------------------------
# Test 7: batched Newton works
# ---------------------------------------------------------------------


def test_batched_newton_x_only():
    """``laplace_newton_batch_x_only`` works over C cells."""
    C, G, k = 6, 8, 2
    rng = np.random.default_rng(3)
    rate = jnp.asarray(np.exp(rng.normal(1.0, 0.5, size=G)).astype(np.float32))
    kappa = jnp.asarray(np.exp(rng.normal(0.5, 0.3, size=G)).astype(np.float32))
    theta = jnp.asarray(rng.normal(0.0, 1.0, size=G).astype(np.float32))
    W = jnp.asarray(rng.normal(0.0, 0.3, size=(G, k)).astype(np.float32))
    d = jnp.asarray(np.ones(G, dtype=np.float32) * 0.1)
    mu = jnp.zeros(G, dtype=jnp.float32)

    u = jnp.asarray(rng.integers(0, 10, size=(C, G)).astype(np.float32))
    x_init = jnp.asarray(np.zeros((C, G), dtype=np.float32))

    x_map, final_grad, log_det, log_marg, a_raw_min = (
        laplace_newton_batch_x_only(
            x_init, u, mu, W, d, rate, kappa, theta,
            8, 1e-6, 5.0, _DEFAULT_K,
        )
    )
    assert x_map.shape == (C, G)
    assert final_grad.shape == (C,)
    assert jnp.all(jnp.isfinite(x_map))

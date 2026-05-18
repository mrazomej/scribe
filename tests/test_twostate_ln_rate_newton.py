"""Unit tests for the TSLN-Rate Newton kernel.

Tests cover:

1. ``_twostate_ln_rate_factors`` closed-form gradient/Hessian-diagonal
   matches ``jax.grad`` / ``jax.hessian`` on the Poisson-Beta marginal.
2. The defensive clamp on ``a_g`` exposes ``a_raw`` correctly.
3. NB-limit recovery: as ``β → ∞`` with ``α = r`` fixed and
   ``λ/β = b`` finite, the factors collapse to NBLN's
   ``a = (u+r) p (1-p)`` form (within float32 precision).
4. Newton step converges from a perturbed warm start (quadratic).
5. Woodbury solve equivalence: ``_solve_A`` matches a brute-force
   inversion of ``A = diag(a) + Σ⁻¹``.
6. Module imports cleanly.

These verify the math derivation in plan §3.1 ("Rev 2 — Variant A
Newton kernel derivation").
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import pytest

from scribe.laplace._newton_twostate_ln_rate import (
    _A_MIN,
    _DEFAULT_K,
    _twostate_ln_rate_factors,
    _woodbury_factors,
    _solve_A,
    _log_det_A,
    laplace_newton_loop_x_only,
    laplace_newton_batch_x_only,
)
from scribe.stats._jacobi_quad import gauss_legendre_01


# ---------------------------------------------------------------------
# Reference: Poisson-Beta marginal log-prob via fixed Legendre quad
# ---------------------------------------------------------------------


def _log_pb_marginal(log_rate, u, alpha, beta, K):
    """Per-gene Poisson-Beta marginal log-likelihood via fixed Legendre."""
    p_node, log_w_node = gauss_legendre_01(K)
    log_p = jnp.log(p_node)
    log_1mp = jnp.log1p(-p_node)
    log_beta_density = (
        (alpha[:, None] - 1.0) * log_p[None, :]
        + (beta[:, None] - 1.0) * log_1mp[None, :]
        - jsp.betaln(alpha, beta)[:, None]
    )
    log_w_eff = log_w_node[None, :] + log_beta_density
    log_lambda = log_rate[:, None] + log_p[None, :]
    log_poisson = (
        u[:, None] * log_lambda
        - jnp.exp(log_lambda)
        - jsp.gammaln(u + 1.0)[:, None]
    )
    return jsp.logsumexp(log_w_eff + log_poisson, axis=-1)


def _total_log_lik(log_rate, u, alpha, beta, K):
    return jnp.sum(_log_pb_marginal(log_rate, u, alpha, beta, K=K))


# ---------------------------------------------------------------------
# Test 1: closed-form factors match autodiff
# ---------------------------------------------------------------------


@pytest.mark.parametrize("K", [40, 60, 80])
def test_factors_match_autodiff(K):
    """``g_data`` and ``a_raw`` should equal autodiff to float32 precision."""
    log_rate = jnp.array([0.0, 1.0, -1.0, 2.0, -2.0])
    u = jnp.array([1.0, 5.0, 0.0, 10.0, 0.0])
    alpha = jnp.array([0.5, 1.0, 2.0, 5.0, 50.0])
    beta = jnp.array([10.0, 5.0, 2.0, 1.0, 50.0])

    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes=K)

    grad_autodiff = jax.grad(_total_log_lik)(log_rate, u, alpha, beta, K)
    hess_diag_autodiff = jnp.diag(
        jax.hessian(_total_log_lik)(log_rate, u, alpha, beta, K)
    )

    # Float32 precision: ~1e-5 absolute.  Hessian is more sensitive
    # because it's a second derivative; relax to 5e-5.
    np.testing.assert_allclose(
        np.asarray(fac["g_data"]),
        np.asarray(grad_autodiff),
        atol=5e-5,
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(fac["a_raw"]),
        np.asarray(-hess_diag_autodiff),
        atol=5e-5,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------
# Test 2: clamp diagnostics
# ---------------------------------------------------------------------


def test_a_clamp_floors_at_A_MIN():
    """Clamped ``a`` should be ``maximum(a_raw, _A_MIN)`` element-wise."""
    log_rate = jnp.array([-10.0, 0.0, 10.0])  # extreme range
    u = jnp.array([0.0, 1.0, 100.0])
    alpha = jnp.array([0.1, 1.0, 10.0])
    beta = jnp.array([0.1, 1.0, 10.0])

    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes=60)
    a = np.asarray(fac["a"])
    a_raw = np.asarray(fac["a_raw"])
    # All clamped values must be at least _A_MIN.
    assert np.all(a >= _A_MIN - 1e-7), f"a contains values below _A_MIN: {a}"
    # Where a_raw >= _A_MIN, a == a_raw.
    healthy = a_raw >= _A_MIN
    np.testing.assert_allclose(a[healthy], a_raw[healthy], atol=1e-7)
    # Where a_raw < _A_MIN, a == _A_MIN.
    floored = a_raw < _A_MIN
    if floored.any():
        np.testing.assert_allclose(
            a[floored], np.full(floored.sum(), _A_MIN), atol=1e-7
        )


# ---------------------------------------------------------------------
# Test 3: NB-limit recovery
# ---------------------------------------------------------------------


def test_nb_limit_recovers_nbln_a_factor():
    """In the β → ∞ limit, ``a_g`` should match NBLN's ``(u+r) p (1-p)``.

    Setup: α = r (= 2.0 say), β = large (1000), λ moderate (5.0).
    Then ``q = λ/(β + λ) ≈ 5/1005`` and
    ``a_nb = (u+r) q (1-q) ≈ (u+r) * q`` for small q.
    """
    r = 2.0
    beta_large = 1000.0
    lam = 5.0
    u_vals = np.array([0.0, 1.0, 5.0, 20.0], dtype=np.float32)

    log_rate = jnp.full((4,), float(np.log(lam)))
    u = jnp.asarray(u_vals)
    alpha = jnp.full((4,), float(r))
    beta = jnp.full((4,), float(beta_large))

    fac = _twostate_ln_rate_factors(log_rate, u, alpha, beta, n_quad_nodes=80)

    # NBLN reference: q = λ/(β+λ), p = β/(β+λ) = 1 - q.
    q = lam / (beta_large + lam)
    a_nb = (u_vals + r) * q * (1.0 - q)
    # Tolerance loosened because K=80 isn't infinite; relax to ~5% relative.
    np.testing.assert_allclose(
        np.asarray(fac["a_raw"]),
        a_nb,
        rtol=5e-2,
        atol=1e-3,
        err_msg=(
            "TSLN-Rate a_g should collapse to NBLN's (u+r) p (1-p) in the "
            "β → ∞ limit. See plan §3.1 NB-limit derivation."
        ),
    )

    # And g_data should match u - (u+r)*q:
    g_nb = u_vals - (u_vals + r) * q
    np.testing.assert_allclose(
        np.asarray(fac["g_data"]),
        g_nb,
        rtol=5e-2,
        atol=5e-3,
    )


# ---------------------------------------------------------------------
# Test 4: Newton step convergence
# ---------------------------------------------------------------------


def test_newton_x_only_converges_quadratically():
    """From a perturbed warm start, Newton should converge in <10 iters."""
    rng = np.random.default_rng(0)
    G, k = 8, 2
    # Reasonable problem
    mu = jnp.asarray(rng.normal(size=G).astype(np.float32))
    W = jnp.asarray(rng.normal(size=(G, k)).astype(np.float32) * 0.3)
    d = jnp.asarray(jnp.full((G,), 0.5))
    alpha = jnp.asarray(jnp.full((G,), 2.0))
    beta = jnp.asarray(jnp.full((G,), 3.0))
    u = jnp.asarray(rng.integers(low=0, high=10, size=G).astype(np.float32))

    # Warm start at mu; should converge from there
    x_init = mu
    (
        x_map, final_grad, log_det, log_marginal, a_raw_min,
    ) = laplace_newton_loop_x_only(
        x_init, u, mu, W, d, alpha, beta,
        n_iters=12, damping=1e-4, max_step=20.0,
    )
    assert float(final_grad) < 1e-3, (
        f"Newton did not converge: final_grad={float(final_grad):.3e}"
    )
    assert jnp.isfinite(log_det), "log_det_neg_H should be finite at MAP"
    assert jnp.isfinite(log_marginal), "log_marginal should be finite"


# ---------------------------------------------------------------------
# Test 5: Woodbury solve equivalence
# ---------------------------------------------------------------------


def test_woodbury_solve_matches_brute_force():
    """``_solve_A(factors, y)`` must equal ``inv(A) @ y`` directly."""
    rng = np.random.default_rng(1)
    G, k = 20, 3
    W = jnp.asarray(rng.normal(size=(G, k)).astype(np.float32) * 0.3)
    d = jnp.asarray(jnp.full((G,), 0.5))
    a_diag = jnp.asarray(jnp.full((G,), 1.5))
    y = jnp.asarray(rng.normal(size=G).astype(np.float32))

    factors = _woodbury_factors(W, d, a_diag, damping=0.0)
    x_woodbury = _solve_A(factors, y)

    # Brute force: A = diag(a + 1/d) - V K^{-1} V^T  ⇒ same as
    # A = diag(a) + Σ⁻¹.
    Sigma = jnp.asarray(np.asarray(W) @ np.asarray(W).T + np.diag(np.asarray(d)))
    Sigma_inv = jnp.asarray(np.linalg.inv(np.asarray(Sigma)))
    A = jnp.diag(a_diag) + Sigma_inv
    x_brute = jnp.asarray(np.linalg.solve(np.asarray(A), np.asarray(y)))

    np.testing.assert_allclose(
        np.asarray(x_woodbury), np.asarray(x_brute), atol=1e-4, rtol=1e-4,
    )

    # log det A check
    log_det_brute = float(np.linalg.slogdet(np.asarray(A))[1])
    log_det_wb = float(_log_det_A(factors))
    np.testing.assert_allclose(log_det_wb, log_det_brute, atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------
# Test 6: vmapped batch Newton runs cleanly
# ---------------------------------------------------------------------


def test_batch_newton_x_only_runs():
    """``laplace_newton_batch_x_only`` should vmap over cells without NaNs."""
    rng = np.random.default_rng(2)
    C, G, k = 4, 8, 2
    mu = jnp.asarray(rng.normal(size=G).astype(np.float32))
    W = jnp.asarray(rng.normal(size=(G, k)).astype(np.float32) * 0.3)
    d = jnp.asarray(jnp.full((G,), 0.5))
    alpha = jnp.asarray(jnp.full((G,), 2.0))
    beta = jnp.asarray(jnp.full((G,), 3.0))
    counts = jnp.asarray(
        rng.integers(low=0, high=10, size=(C, G)).astype(np.float32)
    )
    x_init = jnp.log(counts + 1.0)

    (
        x_map, final_grad, log_det, log_marginal, a_raw_min,
    ) = laplace_newton_batch_x_only(
        x_init, counts, mu, W, d, alpha, beta,
        8, 1e-4, 10.0, _DEFAULT_K,
    )
    assert x_map.shape == (C, G)
    assert final_grad.shape == (C,)
    assert log_det.shape == (C,)
    assert log_marginal.shape == (C,)
    assert a_raw_min.shape == (C,)
    assert jnp.all(jnp.isfinite(x_map))
    assert jnp.all(final_grad < 1.0), (
        f"At least one cell's final grad too large: {final_grad}"
    )

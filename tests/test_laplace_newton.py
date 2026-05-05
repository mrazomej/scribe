"""Tests for the pure-JAX Newton kernel ``svi/_laplace_newton.py``.

The kernel is the load-bearing math of the Laplace inference path
for PLN. We verify three things on small problems:

1. **MAP correctness** — the Newton iterate matches the true MAP
   computed by ``scipy.optimize.minimize`` (Newton-CG) on the same
   joint log-density. Both x-only and joint ``(x, η)`` variants.
2. **Hessian determinant correctness** — the Woodbury-based
   ``log det(-H)`` agrees with a dense ``jnp.linalg.slogdet`` on the
   full Hessian.
3. **Convergence and `vmap` compatibility** — the batched kernel
   produces the same per-cell MAP as running the loop one cell at
   a time.

The kernel is JIT-traceable; all tests run on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize as spopt
from jax import random

from scribe.svi._laplace_newton import (
    _log_det_A,
    _solve_A,
    _woodbury_factors,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
    laplace_newton_loop,
    laplace_newton_loop_x_only,
    newton_step_joint,
    newton_step_x_only,
)


# =====================================================================
# Helpers — closed-form references
# =====================================================================


def _build_random_pln_problem(G=5, k=2, seed=0):
    """Construct a small, well-conditioned PLN per-cell problem.

    Returns numpy arrays. We keep the problem deliberately small so
    a dense ``scipy.optimize`` reference can run in milliseconds.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=G).astype(np.float32)
    W = (0.3 * rng.normal(size=(G, k))).astype(np.float32)
    d = (0.05 + 0.5 * rng.uniform(size=G)).astype(np.float32)  # > 0
    # Generate a count vector consistent with the prior + no capture.
    x_true = mu + W @ rng.normal(size=k).astype(np.float32) + np.sqrt(
        d
    ) * rng.normal(size=G).astype(np.float32)
    rate = np.exp(np.clip(x_true, -10, 10))
    u = rng.poisson(rate).astype(np.float32)
    return mu, W, d, u


def _sigma_dense(W: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Build the dense ``Σ = W W' + diag(d)`` for reference computations."""
    return W @ W.T + np.diag(d)


def _neg_log_density_x_only(x, u, mu, W, d):
    """Closed-form ``-log p(x, u)`` for PLN without capture (NumPy)."""
    rate = np.exp(np.clip(x, -30, 30))
    poisson_term = u @ x - np.sum(rate)  # drops u! constant
    Sigma = _sigma_dense(W, d)
    diff = x - mu
    prior_term = -0.5 * diff @ np.linalg.solve(Sigma, diff)
    return -(poisson_term + prior_term)


def _neg_log_density_x_only_jax(x, u, mu, W, d):
    """JAX version of ``_neg_log_density_x_only`` for autodiff references.

    Used inside ``jax.grad`` to give scipy's optimizer an analytic
    gradient that matches our kernel's gradient up to float
    precision. Avoids the ~5e-2 finite-difference noise that
    ``approx_fprime`` introduces.
    """
    rate = jnp.exp(jnp.clip(x, -30, 30))
    poisson_term = u @ x - jnp.sum(rate)
    Sigma = W @ W.T + jnp.diag(d)
    diff = x - mu
    prior_term = -0.5 * diff @ jnp.linalg.solve(Sigma, diff)
    return -(poisson_term + prior_term)


def _neg_log_density_joint(theta, u, mu, W, d, eta_anchor, sigma_M):
    """Closed-form ``-log p(x, η, u)`` (NumPy). ``theta = [x; η]``."""
    G = mu.shape[0]
    x = theta[:G]
    eta = theta[G]
    rate = np.exp(np.clip(x - eta, -30, 30))
    poisson_term = u @ (x - eta) - np.sum(rate)
    Sigma = _sigma_dense(W, d)
    diff = x - mu
    prior_x = -0.5 * diff @ np.linalg.solve(Sigma, diff)
    prior_eta = -0.5 * (eta - eta_anchor) ** 2 / sigma_M ** 2
    return -(poisson_term + prior_x + prior_eta)


def _neg_log_density_joint_jax(theta, u, mu, W, d, eta_anchor, sigma_M):
    """JAX version of the joint negative log-density."""
    G = mu.shape[0]
    x = theta[:G]
    eta = theta[G]
    rate = jnp.exp(jnp.clip(x - eta, -30, 30))
    poisson_term = u @ (x - eta) - jnp.sum(rate)
    Sigma = W @ W.T + jnp.diag(d)
    diff = x - mu
    prior_x = -0.5 * diff @ jnp.linalg.solve(Sigma, diff)
    prior_eta = -0.5 * (eta - eta_anchor) ** 2 / sigma_M ** 2
    return -(poisson_term + prior_x + prior_eta)


# =====================================================================
# 1. Woodbury solve / log-det
# =====================================================================


class TestWoodburyHelpers:
    """Cross-check the Woodbury solve and log-det against dense linear algebra."""

    def test_solve_matches_dense_inverse(self):
        """``_solve_A(y) ≈ A⁻¹ y`` for a random PD ``A``."""
        mu, W, d, u = _build_random_pln_problem(G=8, k=3)
        log_rate = mu  # arbitrary feasible iterate
        damping = 1e-4
        factors = _woodbury_factors(
            jnp.asarray(W), jnp.asarray(d), jnp.asarray(log_rate), damping
        )
        # Dense reference: A = diag(λ + 1/d + damping) + Σ⁻¹.
        rate = np.exp(np.clip(log_rate, -30, 30))
        Sigma = _sigma_dense(W, d)
        Sigma_inv = np.linalg.inv(Sigma)
        A_dense = np.diag(rate + damping) + Sigma_inv
        rng = np.random.default_rng(2)
        y = rng.normal(size=mu.shape[0]).astype(np.float32)
        ref = np.linalg.solve(A_dense, y)
        out = _solve_A(factors, jnp.asarray(y))
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)

    def test_log_det_matches_dense(self):
        """``_log_det_A`` matches ``slogdet`` of the dense matrix."""
        mu, W, d, u = _build_random_pln_problem(G=6, k=2)
        log_rate = mu - 0.5
        damping = 1e-3
        factors = _woodbury_factors(
            jnp.asarray(W), jnp.asarray(d), jnp.asarray(log_rate), damping
        )
        rate = np.exp(np.clip(log_rate, -30, 30))
        Sigma = _sigma_dense(W, d)
        Sigma_inv = np.linalg.inv(Sigma)
        A_dense = np.diag(rate + damping) + Sigma_inv
        sign, logdet_ref = np.linalg.slogdet(A_dense)
        assert sign == 1.0
        out = float(_log_det_A(factors))
        np.testing.assert_allclose(out, logdet_ref, rtol=1e-4, atol=1e-3)


# =====================================================================
# 2. MAP correctness — x only
# =====================================================================


class TestXOnlyMAP:
    """Newton MAP matches scipy's reference on a small no-capture problem."""

    def test_matches_scipy_newton_cg(self):
        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=5, k=2, seed=3)
        # Run scipy from a deliberately bad init so both methods have to
        # do real work; convergence to a global maximum is guaranteed
        # because the joint is log-concave. Use an *analytic* gradient
        # via jax.grad — finite differences (``approx_fprime``) are not
        # accurate enough at float32 to drive Newton-CG down to the
        # 1e-3 tolerance we want for the cross-check.
        x_init = np.zeros_like(mu_np).astype(np.float64)

        def f_neg(x_np):
            return float(_neg_log_density_x_only(x_np, u_np, mu_np, W_np, d_np))

        # Use scipy's CG on the analytic gradient (computed by JAX in
        # float64 for tightness).
        def grad_neg(x_np):
            x_j = jnp.asarray(x_np, dtype=jnp.float64)

            def loss(x):
                return _neg_log_density_x_only_jax(
                    x,
                    jnp.asarray(u_np, dtype=jnp.float64),
                    jnp.asarray(mu_np, dtype=jnp.float64),
                    jnp.asarray(W_np, dtype=jnp.float64),
                    jnp.asarray(d_np, dtype=jnp.float64),
                )

            return np.asarray(jax.grad(loss)(x_j), dtype=np.float64)

        result = spopt.minimize(
            f_neg,
            x_init,
            method="L-BFGS-B",
            jac=grad_neg,
            options={"gtol": 1e-9, "ftol": 1e-12},
        )
        x_ref = result.x.astype(np.float32)

        # Run our kernel from the same init.
        x_jax, grad_norm, _ = laplace_newton_loop_x_only(
            jnp.asarray(np.zeros_like(mu_np), dtype=jnp.float32),
            jnp.asarray(u_np, dtype=jnp.float32),
            jnp.asarray(mu_np, dtype=jnp.float32),
            jnp.asarray(W_np, dtype=jnp.float32),
            jnp.asarray(d_np, dtype=jnp.float32),
            n_iters=30,
            damping=0.0,  # exact comparison against scipy
        )
        # 5e-3 tolerance is plenty given float32 precision in the
        # kernel and float64 reference in scipy.
        np.testing.assert_allclose(
            np.asarray(x_jax), x_ref, atol=5e-3, rtol=5e-3
        )
        # Gradient norm at the MAP should be tiny.
        assert float(grad_norm) < 1e-3

    def test_one_step_makes_progress(self):
        """A single Newton step from a bad init should decrease ``-log p``."""
        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=5, k=2, seed=5)
        x_init = np.zeros_like(mu_np)
        nll_init = _neg_log_density_x_only(x_init, u_np, mu_np, W_np, d_np)
        x_new, _ = newton_step_x_only(
            jnp.asarray(x_init, dtype=jnp.float32),
            jnp.asarray(u_np, dtype=jnp.float32),
            jnp.asarray(mu_np, dtype=jnp.float32),
            jnp.asarray(W_np, dtype=jnp.float32),
            jnp.asarray(d_np, dtype=jnp.float32),
        )
        nll_new = _neg_log_density_x_only(
            np.asarray(x_new), u_np, mu_np, W_np, d_np
        )
        assert nll_new < nll_init


# =====================================================================
# 3. MAP correctness — joint (x, η)
# =====================================================================


class TestJointMAP:
    """Joint Newton MAP matches scipy on a small capture-anchor problem."""

    def test_matches_scipy_newton_cg(self):
        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=5, k=2, seed=7)
        # Pretend the cell has a typical library size; the anchor pulls
        # η to log(M_0/L_c).
        log_M0 = float(np.log(1e5))
        log_L_c = float(np.log(u_np.sum() + 1.0))
        eta_anchor = log_M0 - log_L_c
        sigma_M = 0.3

        theta_init = np.concatenate([mu_np, [eta_anchor]]).astype(np.float64)

        def f_neg(theta_np):
            return float(
                _neg_log_density_joint(
                    theta_np, u_np, mu_np, W_np, d_np, eta_anchor, sigma_M
                )
            )

        def grad_neg(theta_np):
            theta_j = jnp.asarray(theta_np, dtype=jnp.float64)
            return np.asarray(
                jax.grad(_neg_log_density_joint_jax)(
                    theta_j,
                    jnp.asarray(u_np, dtype=jnp.float64),
                    jnp.asarray(mu_np, dtype=jnp.float64),
                    jnp.asarray(W_np, dtype=jnp.float64),
                    jnp.asarray(d_np, dtype=jnp.float64),
                    jnp.asarray(eta_anchor, dtype=jnp.float64),
                    sigma_M,
                ),
                dtype=np.float64,
            )

        result = spopt.minimize(
            f_neg,
            theta_init,
            method="L-BFGS-B",
            jac=grad_neg,
            options={"gtol": 1e-9, "ftol": 1e-12},
        )
        x_ref, eta_ref = (
            result.x[:5].astype(np.float32),
            float(result.x[5]),
        )

        x_jax, eta_jax, grad_norm, log_det_neg_H = laplace_newton_loop(
            jnp.asarray(mu_np, dtype=jnp.float32),
            jnp.asarray(eta_anchor, dtype=jnp.float32),
            jnp.asarray(u_np, dtype=jnp.float32),
            jnp.asarray(mu_np, dtype=jnp.float32),
            jnp.asarray(W_np, dtype=jnp.float32),
            jnp.asarray(d_np, dtype=jnp.float32),
            jnp.asarray(eta_anchor, dtype=jnp.float32),
            sigma_M,
            n_iters=30,
            damping=0.0,
        )
        # 5e-3 tolerance: float32 kernel vs float64 scipy reference.
        np.testing.assert_allclose(
            np.asarray(x_jax), x_ref, atol=5e-3, rtol=5e-3
        )
        assert float(eta_jax) == pytest.approx(eta_ref, abs=5e-3)
        assert float(grad_norm) < 1e-3
        # log det(-H) is finite and positive (negative-definite Hessian
        # → positive determinant of -H).
        assert float(log_det_neg_H) > 0
        assert np.isfinite(float(log_det_neg_H))

    def test_single_step_matches_dense(self):
        """One Newton step on a NON-MAP iterate must match the dense
        block solve of ``-H δ = ∇f``.

        Regression test for a sign bug in the Schur back-substitution
        that previously survived because the converged-MAP test only
        checks the fixed point, not per-step direction. A wrong-sign
        Newton step can still drift toward the MAP under damping +
        step-size cap, so we verify the step itself, off the fixed
        point.
        """
        from scribe.svi._laplace_newton import newton_step_joint

        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=5, k=2, seed=11)
        sigma_M = 0.3
        eta_anchor = 0.5

        # Iterate intentionally NOT at the MAP — pick a perturbation
        # so the cross-coupling term is non-trivial. ``rate^T A^-1 g_x``
        # is the term whose sign was wrong; we want it to be the same
        # order of magnitude as ``g_eta`` so the bug is observable.
        rng = np.random.default_rng(11)
        x_np = mu_np + 0.4 * rng.standard_normal(size=mu_np.shape).astype(np.float32)
        eta_np = np.float32(0.7)

        x = jnp.asarray(x_np)
        eta = jnp.asarray(eta_np)
        mu = jnp.asarray(mu_np)
        W = jnp.asarray(W_np)
        d = jnp.asarray(d_np)
        u = jnp.asarray(u_np)
        eta_anch = jnp.asarray(np.float32(eta_anchor))

        x_new, eta_new, _gn = newton_step_joint(
            x, eta, u, mu, W, d, eta_anch, sigma_M, damping=0.0
        )
        delta_x_kernel = np.asarray(x_new - x)
        delta_eta_kernel = float(eta_new - eta)

        # Build the dense (G+1)x(G+1) -H and solve. We use float64 to
        # avoid float32 cancellation in the comparison; the kernel
        # itself runs in float32, so the agreement is bounded by float32
        # epsilon (~1e-6 relative).
        rate = np.exp(np.asarray(x - eta), dtype=np.float64)
        Sigma = (
            W_np.astype(np.float64) @ W_np.astype(np.float64).T
            + np.diag(d_np.astype(np.float64))
        )
        Sigma_inv = np.linalg.inv(Sigma)
        diff = (x_np - mu_np).astype(np.float64)
        g_x = u_np.astype(np.float64) - rate - Sigma_inv @ diff
        g_eta = (
            -float(np.sum(u_np))
            + float(np.sum(rate))
            - (float(eta_np) - eta_anchor) / sigma_M**2
        )
        G = mu_np.shape[0]
        neg_H = np.zeros((G + 1, G + 1), dtype=np.float64)
        neg_H[:G, :G] = np.diag(rate) + Sigma_inv
        # Off-diagonal block of -H is -rate (because H_xη = +rate).
        neg_H[:G, G] = -rate
        neg_H[G, :G] = -rate
        neg_H[G, G] = float(np.sum(rate)) + 1.0 / sigma_M**2

        grad = np.empty(G + 1, dtype=np.float64)
        grad[:G] = g_x
        grad[G] = g_eta
        delta_dense = np.linalg.solve(neg_H, grad)

        # Float32 epsilon means we can't ask for 1e-9 — but 1e-4
        # relative is comfortably tighter than the wrong-sign step,
        # which differs from the dense solve by O(1) in this setup.
        np.testing.assert_allclose(
            delta_x_kernel, delta_dense[:G], rtol=1e-4, atol=1e-4
        )
        assert delta_eta_kernel == pytest.approx(
            float(delta_dense[G]), rel=1e-4, abs=1e-4
        )


# =====================================================================
# 4. Hessian determinant on full dense Hessian
# =====================================================================


class TestHessianDeterminant:
    """Woodbury-based ``log det(-H)`` matches the dense ``slogdet``."""

    def test_log_det_neg_H_matches_dense(self):
        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=6, k=2, seed=9)
        log_M0 = float(np.log(1e5))
        log_L_c = float(np.log(u_np.sum() + 1.0))
        eta_anchor = log_M0 - log_L_c
        sigma_M = 0.3

        # Run Newton to convergence; then compare ``log det(-H)`` at
        # the MAP from the kernel to a dense computation.
        x_jax, eta_jax, _, log_det_kernel = laplace_newton_loop(
            jnp.asarray(mu_np, dtype=jnp.float32),
            jnp.asarray(eta_anchor, dtype=jnp.float32),
            jnp.asarray(u_np, dtype=jnp.float32),
            jnp.asarray(mu_np, dtype=jnp.float32),
            jnp.asarray(W_np, dtype=jnp.float32),
            jnp.asarray(d_np, dtype=jnp.float32),
            jnp.asarray(eta_anchor, dtype=jnp.float32),
            sigma_M,
            n_iters=20,
            damping=0.0,  # exact comparison against the dense Hessian
        )

        # Build dense -H at the MAP.
        x_map = np.asarray(x_jax)
        eta_map = float(eta_jax)
        rate = np.exp(np.clip(x_map - eta_map, -30, 30))
        Sigma = _sigma_dense(W_np, d_np)
        Sigma_inv = np.linalg.inv(Sigma)
        # H_xx = -diag(rate) - Σ⁻¹  →  -H_xx = diag(rate) + Σ⁻¹
        neg_H_xx = np.diag(rate) + Sigma_inv
        # H_xη = rate (column)  →  -H_xη = -rate
        neg_H_xeta = -rate
        # H_ηη = -1' rate - 1/σ²  →  -H_ηη = 1' rate + 1/σ²
        neg_H_eta = float(rate.sum() + 1.0 / sigma_M ** 2)
        # Block-assemble.
        neg_H = np.zeros((mu_np.shape[0] + 1, mu_np.shape[0] + 1))
        neg_H[:-1, :-1] = neg_H_xx
        neg_H[:-1, -1] = neg_H_xeta
        neg_H[-1, :-1] = neg_H_xeta
        neg_H[-1, -1] = neg_H_eta
        sign, logdet_ref = np.linalg.slogdet(neg_H)
        assert sign == 1.0
        np.testing.assert_allclose(
            float(log_det_kernel), logdet_ref, rtol=1e-3, atol=1e-3
        )


# =====================================================================
# 5. vmap correctness
# =====================================================================


class TestVmapBatch:
    """Batched kernel matches per-cell loops."""

    def test_x_only_batch_matches_serial(self):
        n_cells = 4
        mu_np, W_np, d_np, _ = _build_random_pln_problem(G=5, k=2, seed=11)
        rng = np.random.default_rng(11)
        # Per-cell counts.
        x_seed = mu_np[None, :] + 0.3 * rng.normal(size=(n_cells, 5)).astype(
            np.float32
        )
        rates = np.exp(np.clip(x_seed, -10, 10))
        u = rng.poisson(rates).astype(np.float32)
        x_init = np.zeros_like(u)

        # Batched.
        x_batch, gn_batch, ld_batch = laplace_newton_batch_x_only(
            jnp.asarray(x_init),
            jnp.asarray(u),
            jnp.asarray(mu_np),
            jnp.asarray(W_np),
            jnp.asarray(d_np),
            10,
            1e-6,
        )
        # Per-cell.
        for c in range(n_cells):
            x_serial, gn_serial, ld_serial = laplace_newton_loop_x_only(
                jnp.asarray(x_init[c]),
                jnp.asarray(u[c]),
                jnp.asarray(mu_np),
                jnp.asarray(W_np),
                jnp.asarray(d_np),
                10,
                1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(x_batch[c]),
                np.asarray(x_serial),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                float(gn_batch[c]), float(gn_serial), rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                float(ld_batch[c]), float(ld_serial), rtol=1e-5, atol=1e-5
            )

    def test_joint_batch_runs(self):
        """Smoke test: the joint batched kernel runs end-to-end without
        shape errors and produces finite outputs.
        """
        n_cells = 3
        mu_np, W_np, d_np, _ = _build_random_pln_problem(G=4, k=2, seed=13)
        rng = np.random.default_rng(13)
        u = rng.poisson(5.0, size=(n_cells, 4)).astype(np.float32)
        x_init = jnp.zeros((n_cells, 4))
        eta_init = jnp.zeros((n_cells,))
        eta_anchor = jnp.full((n_cells,), 1.5)
        x, eta, gn, ld = laplace_newton_batch(
            x_init,
            eta_init,
            jnp.asarray(u),
            jnp.asarray(mu_np),
            jnp.asarray(W_np),
            jnp.asarray(d_np),
            eta_anchor,
            0.3,
            10,
            1e-6,
        )
        assert x.shape == (n_cells, 4)
        assert eta.shape == (n_cells,)
        assert gn.shape == (n_cells,)
        assert ld.shape == (n_cells,)
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(jnp.isfinite(eta))


# =====================================================================
# 6. JIT compatibility
# =====================================================================


class TestJIT:
    """The kernel must be JIT-compilable."""

    def test_loop_x_only_jits(self):
        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=5, k=2, seed=17)
        f = jax.jit(
            lambda x: laplace_newton_loop_x_only(
                x,
                jnp.asarray(u_np),
                jnp.asarray(mu_np),
                jnp.asarray(W_np),
                jnp.asarray(d_np),
                10,
                1e-4,
            )
        )
        x_jax, _, _ = f(jnp.zeros_like(mu_np))
        assert x_jax.shape == mu_np.shape

    def test_loop_joint_jits(self):
        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=5, k=2, seed=19)
        f = jax.jit(
            lambda x, eta: laplace_newton_loop(
                x,
                eta,
                jnp.asarray(u_np),
                jnp.asarray(mu_np),
                jnp.asarray(W_np),
                jnp.asarray(d_np),
                jnp.asarray(1.5),
                0.3,
                10,
                1e-4,
            )
        )
        x_jax, eta_jax, _, _ = f(jnp.zeros_like(mu_np), jnp.asarray(1.5))
        assert x_jax.shape == mu_np.shape
        assert eta_jax.shape == ()

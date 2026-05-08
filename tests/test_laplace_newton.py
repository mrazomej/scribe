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

from scribe.laplace._newton_pln import (
    _log_det_A,
    _solve_A,
    _woodbury_factors,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
    laplace_newton_loop,
    laplace_newton_loop_x_only,
    newton_step_joint,
    newton_step_x_only,
    sample_x_posterior,
    sample_x_posterior_batch,
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
        from scribe.laplace._newton_pln import newton_step_joint

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

    def test_log_det_helper_has_live_gradient(self):
        """``laplace_log_det_neg_H`` must have non-zero gradient
        through ``W`` and ``d``.

        Regression for a bug where the loss path stop_gradient'd the
        kernel's returned ``log_det`` and then reused it inside the
        ELBO — so Adam was not optimizing the Laplace correction
        term at all. The fix recomputes ``log det(-H)`` outside the
        Newton kernel against live globals, with damping=0.
        """
        from scribe.laplace._newton_pln import laplace_log_det_neg_H

        mu_np, W_np, d_np, u_np = _build_random_pln_problem(G=4, k=2, seed=3)
        rng = np.random.default_rng(3)
        x_map = jnp.asarray(
            mu_np + 0.1 * rng.standard_normal(size=mu_np.shape).astype(np.float32)
        )
        eta_map = jnp.asarray(np.float32(0.4))
        W = jnp.asarray(W_np)
        d = jnp.asarray(d_np)

        # Joint case (capture anchor on)
        grad_W = jax.grad(
            lambda W_: laplace_log_det_neg_H(x_map, eta_map, W_, d, 0.3)
        )(W)
        grad_d = jax.grad(
            lambda d_: laplace_log_det_neg_H(x_map, eta_map, W, d_, 0.3)
        )(d)
        assert float(jnp.max(jnp.abs(grad_W))) > 1e-4
        assert float(jnp.max(jnp.abs(grad_d))) > 1e-4

        # x-only case (capture off)
        grad_W_x = jax.grad(
            lambda W_: laplace_log_det_neg_H(x_map, None, W_, d, 0.3)
        )(W)
        grad_d_x = jax.grad(
            lambda d_: laplace_log_det_neg_H(x_map, None, W, d_, 0.3)
        )(d)
        assert float(jnp.max(jnp.abs(grad_W_x))) > 1e-4
        assert float(jnp.max(jnp.abs(grad_d_x))) > 1e-4


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


class TestPosteriorSampler:
    """Verify the per-cell Laplace-posterior sampler matches the analytic
    posterior covariance.

    The sampler produces ``x_c ~ N(x_map, (-H_xx)^{-1})`` via a
    Woodbury-derived square-root factor (see
    :func:`scribe.laplace._newton_pln.sample_x_posterior`). The
    empirical covariance from many samples should match the dense
    inverse of ``-H_xx`` evaluated at the MAP.
    """

    def test_empirical_cov_matches_analytic_x_only(self):
        np.random.seed(101)
        G, k = 6, 2
        mu = np.random.normal(0, 0.5, G).astype(np.float32)
        W = (0.3 * np.random.normal(size=(G, k))).astype(np.float32)
        d = (0.05 + 0.5 * np.random.uniform(size=G)).astype(np.float32)
        x_map = jnp.asarray(np.random.normal(0, 1, G).astype(np.float32))
        eta_map = jnp.asarray(0.0, dtype=jnp.float32)

        # Analytic posterior covariance: (diag(λ) + Σ⁻¹)^{-1}
        Sigma_inv = np.linalg.inv(W @ W.T + np.diag(d))
        lam = np.exp(np.asarray(x_map - eta_map))
        neg_H_xx = np.diag(lam) + Sigma_inv
        post_cov_analytic = np.linalg.inv(neg_H_xx)

        n_samples = 30_000
        samples = np.asarray(
            sample_x_posterior(
                random.PRNGKey(0),
                x_map,
                eta_map,
                jnp.asarray(W),
                jnp.asarray(d),
                n_samples,
            )
        )

        emp_mean = samples.mean(axis=0)
        emp_cov = np.cov(samples, rowvar=False)

        # Mean error should be O(1/sqrt(N)) of the posterior std.
        post_std = np.sqrt(np.diag(post_cov_analytic))
        np.testing.assert_array_less(
            np.abs(emp_mean - np.asarray(x_map)), 4.0 * post_std / np.sqrt(n_samples)
        )

        # Covariance: relative Frobenius error < 3% at this sample size.
        rel_fro = np.linalg.norm(emp_cov - post_cov_analytic) / np.linalg.norm(
            post_cov_analytic
        )
        assert rel_fro < 0.03, f"emp/analytic Frobenius error {rel_fro:.3f}"

    def test_batch_sampler_has_correct_shape(self):
        n_cells, G, k, n_samples = 4, 5, 2, 7
        rng_keys = random.split(random.PRNGKey(0), n_cells)
        x_loc = jnp.zeros((n_cells, G))
        eta_loc = jnp.zeros((n_cells,))
        W = jnp.ones((G, k)) * 0.1
        d = jnp.full((G,), 0.1)
        out = sample_x_posterior_batch(
            rng_keys, x_loc, eta_loc, W, d, n_samples, 0.0
        )
        assert out.shape == (n_cells, n_samples, G)


class TestLNMPosteriorSamplers:
    """Verify the LNM Laplace samplers (z-mode, y_alr-mode, eta-block) match
    their analytic posterior covariances.

    The math being tested:

    * **z-mode (low_rank)**: sample from
      ``N(z_map, (W^T M_alr W + I)^{-1})`` via a ``k × k`` Cholesky.
    * **y_alr-mode (learned)**: sample from
      ``N(y_map, (-H_y)^{-1})`` where
      ``-H_y = A_y - N ρρ^T``. Tests both the Woodbury part of
      ``A_y`` and the Sherman-Morrison rank-1 correction together.
    * **eta-block**: sample from the 1-D Gaussian on the LNMVCP
      capture-offset latent.
    """

    def test_z_posterior_cov_matches_analytic(self):
        from scribe.laplace._newton_lnm import (
            _multinomial_p_alr,
            sample_z_posterior,
        )

        np.random.seed(201)
        G_full, k, ref_idx = 8, 3, 0
        G_minus1 = G_full - 1
        mu = jnp.asarray(np.random.normal(0, 0.5, G_minus1).astype(np.float32))
        W = jnp.asarray((0.3 * np.random.normal(size=(G_minus1, k))).astype(np.float32))
        N = jnp.asarray(1000.0, dtype=jnp.float32)
        z_map = jnp.asarray(np.random.normal(0, 0.5, k).astype(np.float32))
        u_alr = jnp.asarray(np.random.poisson(50, G_minus1).astype(np.float32))

        # Analytic: -H_z = W^T M_alr W + I where M_alr = N(diag(ρ) - ρρ^T).
        y_alr = mu + W @ z_map
        p_alr = _multinomial_p_alr(y_alr, ref_idx, G_full)
        Wp = W * p_alr[:, None]
        rhs = W.T @ Wp
        Wp_sum = W.T @ p_alr
        M_W = N * (rhs - jnp.outer(Wp_sum, Wp_sum))
        neg_H_z = M_W + jnp.eye(k, dtype=z_map.dtype)
        post_cov = np.linalg.inv(np.asarray(neg_H_z))

        n_samples = 30_000
        samples = np.asarray(
            sample_z_posterior(
                random.PRNGKey(0), z_map, u_alr, N, mu, W,
                ref_idx, G_full, n_samples,
            )
        )
        emp_cov = np.cov(samples, rowvar=False)
        rel_fro = np.linalg.norm(emp_cov - post_cov) / np.linalg.norm(post_cov)
        assert rel_fro < 0.05, f"z-mode emp/analytic Frobenius error {rel_fro:.3f}"

    def test_y_alr_posterior_cov_matches_analytic(self):
        from scribe.laplace._newton_lnm import (
            _multinomial_p_alr,
            sample_y_alr_posterior,
        )

        np.random.seed(202)
        G_full, k, ref_idx = 8, 3, 0
        G_minus1 = G_full - 1
        mu = jnp.asarray(np.random.normal(0, 0.5, G_minus1).astype(np.float32))
        W = jnp.asarray((0.3 * np.random.normal(size=(G_minus1, k))).astype(np.float32))
        d = jnp.asarray(np.full(G_minus1, 0.05, dtype=np.float32))
        N = jnp.asarray(1000.0, dtype=jnp.float32)
        y_map = jnp.asarray(np.random.normal(0, 0.5, G_minus1).astype(np.float32))
        u_alr = jnp.asarray(np.random.poisson(50, G_minus1).astype(np.float32))

        # Analytic: -H_y = N(diag(ρ) - ρρ^T) + Σ⁻¹.
        Sigma = np.asarray(W) @ np.asarray(W).T + np.diag(np.asarray(d))
        Sigma_inv = np.linalg.inv(Sigma)
        p_alr = np.asarray(_multinomial_p_alr(y_map, ref_idx, G_full))
        M_alr = float(N) * (np.diag(p_alr) - np.outer(p_alr, p_alr))
        neg_H_y = M_alr + Sigma_inv
        post_cov = np.linalg.inv(neg_H_y)

        n_samples = 30_000
        samples = np.asarray(
            sample_y_alr_posterior(
                random.PRNGKey(0), y_map, u_alr, N, mu, W, d,
                ref_idx, G_full, n_samples,
            )
        )
        emp_cov = np.cov(samples, rowvar=False)
        rel_fro = np.linalg.norm(emp_cov - post_cov) / np.linalg.norm(post_cov)
        assert rel_fro < 0.05, f"y_alr emp/analytic Frobenius error {rel_fro:.3f}"

    def test_eta_posterior_is_one_dim_gaussian(self):
        """LNMVCP eta posterior is 1-D Gaussian; check empirical mean/variance."""
        from scribe.laplace._newton_lnm import (
            _nb_eta_grad_and_hessian,
            sample_eta_posterior,
        )

        np.random.seed(203)
        eta_map = jnp.asarray(0.7, dtype=jnp.float32)
        u_T = jnp.asarray(15_000.0, dtype=jnp.float32)
        r_T = jnp.asarray(8.0, dtype=jnp.float32)
        mu_T = jnp.asarray(60_000.0, dtype=jnp.float32)
        sigma_M = 0.5

        _, nb_hess = _nb_eta_grad_and_hessian(eta_map, u_T, r_T, mu_T)
        neg_H = float(-(nb_hess - 1.0 / (sigma_M * sigma_M)))
        post_var = 1.0 / neg_H

        n_samples = 30_000
        samples = np.asarray(
            sample_eta_posterior(
                random.PRNGKey(0), eta_map, u_T, r_T, mu_T,
                sigma_M, n_samples,
            )
        )
        emp_mean = samples.mean()
        emp_var = samples.var()
        # Mean within a few SE; variance within ~5%.
        assert abs(emp_mean - float(eta_map)) < 4.0 * np.sqrt(post_var / n_samples)
        rel_var_err = abs(emp_var - post_var) / post_var
        assert rel_var_err < 0.05, f"eta emp/analytic var error {rel_var_err:.3f}"

    def test_y_alr_sherman_morrison_floor_keeps_finite(self):
        """Degenerate-rho cell hits the SM denominator floor — must stay finite."""
        from scribe.laplace._newton_lnm import sample_y_alr_posterior

        # Build a y_map that gives a near-one-hot p_alr (ρ_alr → e_g),
        # which drives ρ^T A^-1 ρ → 1/N from below — the regime the
        # SM denominator floor protects.
        G_full, k, ref_idx = 6, 2, 0
        G_minus1 = G_full - 1
        mu = jnp.zeros(G_minus1, dtype=jnp.float32)
        W = jnp.zeros((G_minus1, k), dtype=jnp.float32)
        d = jnp.full((G_minus1,), 0.1, dtype=jnp.float32)
        N = jnp.asarray(100.0, dtype=jnp.float32)
        y_map = jnp.asarray([20.0, -20.0, -20.0, -20.0, -20.0], dtype=jnp.float32)
        u_alr = jnp.zeros(G_minus1, dtype=jnp.float32)

        samples = np.asarray(
            sample_y_alr_posterior(
                random.PRNGKey(0), y_map, u_alr, N, mu, W, d,
                ref_idx, G_full, 64,
            )
        )
        assert np.all(np.isfinite(samples)), "SM-floor must keep samples finite"
        assert samples.shape == (64, G_minus1)


class TestPPCMethods:
    """Public PPC API: shape parity, MAP-vs-Laplace variance ordering, and
    NB-fitted totals draw independence (the bug the auditor caught).
    """

    def _build_pln_results(self):
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(301)
        G, C, k = 25, 12, 3
        mu = jnp.asarray(np.random.normal(0, 0.5, G).astype(np.float32))
        W = jnp.asarray((0.3 * np.random.normal(size=(G, k))).astype(np.float32))
        d = jnp.asarray(np.full(G, 0.1, dtype=np.float32))
        x_loc = jnp.asarray(np.random.normal(0, 1, (C, G)).astype(np.float32))
        eta_loc = jnp.asarray(np.random.normal(0.5, 0.2, C).astype(np.float32))
        mc = ModelConfig(
            base_model="pln",
            parameterization=Parameterization.POISSON_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        return ScribeLaplaceResults(
            model_config=mc, mu=mu, W=W, d=d,
            n_genes=G, n_cells=C, x_loc=x_loc, eta_loc=eta_loc,
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

    def _build_lnm_low_rank_results(self):
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(302)
        G_full, C, k = 25, 12, 3
        G_minus1 = G_full - 1
        mu = jnp.asarray(np.random.normal(0, 0.5, G_minus1).astype(np.float32))
        W = jnp.asarray((0.3 * np.random.normal(size=(G_minus1, k))).astype(np.float32))
        d = jnp.asarray(np.full(G_minus1, 0.05, dtype=np.float32))
        z_loc = jnp.asarray(np.random.normal(0, 1, (C, k)).astype(np.float32))
        mc = ModelConfig(
            base_model="lnm",
            parameterization=Parameterization.LOGISTIC_NORMAL_CANONICAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        return ScribeLaplaceResults(
            model_config=mc, mu=mu, W=W, d=d,
            n_genes=G_full, n_cells=C, z_loc=z_loc,
            alr_reference_idx=0,
            mu_T=jnp.asarray(20_000.0, dtype=jnp.float32),
            r_T=jnp.asarray(8.0, dtype=jnp.float32),
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

    def test_pln_get_map_ppc_samples_shape(self):
        res = self._build_pln_results()
        out = res.get_map_ppc_samples(rng_key=random.PRNGKey(0), n_samples=8)
        assert tuple(out.shape) == (8, res.n_cells, res.n_genes)

    def test_pln_get_per_cell_predictive_samples_shape(self):
        res = self._build_pln_results()
        out = res.get_per_cell_predictive_samples(
            rng_key=random.PRNGKey(0), n_samples=8
        )
        assert tuple(out.shape) == (8, res.n_cells, res.n_genes)

    def test_pln_laplace_predictive_variance_exceeds_map_variance(self):
        """Per-cell PPC should have wider tails than MAP-only PPC."""
        res = self._build_pln_results()
        n_samples = 200
        laplace = np.asarray(
            res.get_per_cell_predictive_samples(
                rng_key=random.PRNGKey(0), n_samples=n_samples
            )
        )
        map_only = np.asarray(
            res.get_map_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=n_samples
            )
        )
        # Average per-(cell, gene) variance across samples.
        var_laplace = laplace.var(axis=0).mean()
        var_map = map_only.var(axis=0).mean()
        assert var_laplace > var_map, (
            f"Laplace PPC variance {var_laplace:.2f} should exceed "
            f"MAP-only variance {var_map:.2f}"
        )

    def test_lnm_get_map_ppc_samples_nb_fitted_totals_vary(self):
        """Regression test for the bug where chunked LNM MAP PPC tied
        all PPC samples to identical NB-drawn totals.
        """
        res = self._build_lnm_low_rank_results()
        n_samples = 64
        # No counts kwarg ⇒ NB-fitted totals path.
        ppc = np.asarray(
            res.get_map_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=n_samples
            )
        )
        # Per-(sample, cell) totals must vary across samples.
        per_sample_totals = ppc.sum(axis=-1)         # (n_samples, n_cells)
        std_across_samples = per_sample_totals.std(axis=0)
        # NB(mean ≈ 20K, r=8) has std ≈ sqrt(20K + 20K^2/8) ≈ 7.1K, so
        # we comfortably expect the per-cell std across samples to
        # exceed 1K. The bug would give exactly zero.
        assert std_across_samples.mean() > 1_000.0, (
            f"NB-fitted totals not varying across PPC samples "
            f"(mean std {std_across_samples.mean():.1f})"
        )

    def test_lnm_low_rank_per_cell_predictive_samples_shape(self):
        res = self._build_lnm_low_rank_results()
        # Conditional path with observed counts.
        np.random.seed(0)
        counts = np.random.poisson(50, (res.n_cells, res.n_genes)).astype(np.float32)
        out = res.get_per_cell_predictive_samples(
            rng_key=random.PRNGKey(0), n_samples=8, counts=counts
        )
        assert tuple(out.shape) == (8, res.n_cells, res.n_genes)


class TestPPCLevels:
    """Verify the three-level PPC taxonomy on ``ScribeLaplaceResults``:

    * ``per_cell``: cell-conditioned (Laplace posterior).
    * ``library_anchored``: fresh latent, observed library size.
    * ``marginal``: everything sampled fresh (incl. η / p_capture).

    These tests guard the *plumbing* (shapes, dispatch, level
    keyword); the underlying math is covered by the per-helper
    tests elsewhere in the file.
    """

    def _build_pln(self, *, with_capture: bool = True):
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(701)
        G, C, k = 24, 12, 3
        mu = jnp.asarray(np.random.normal(0, 0.5, G).astype(np.float32))
        W = jnp.asarray((0.3 * np.random.normal(size=(G, k))).astype(np.float32))
        d = jnp.asarray(np.full(G, 0.05, dtype=np.float32))
        x_loc = jnp.asarray(np.random.normal(0, 1, (C, G)).astype(np.float32))
        eta_loc = (
            jnp.asarray(np.random.normal(0.5, 0.2, C).astype(np.float32))
            if with_capture else None
        )
        mc = ModelConfig(
            base_model="pln",
            parameterization=Parameterization.POISSON_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        return ScribeLaplaceResults(
            model_config=mc, mu=mu, W=W, d=d,
            n_genes=G, n_cells=C, x_loc=x_loc, eta_loc=eta_loc,
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

    def _build_lnmvcp(self):
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(702)
        G, C, k = 24, 12, 3
        G_minus1 = G - 1
        mu = jnp.asarray(np.random.normal(0, 0.5, G_minus1).astype(np.float32))
        W = jnp.asarray((0.3 * np.random.normal(size=(G_minus1, k))).astype(np.float32))
        d = jnp.asarray(np.full(G_minus1, 0.05, dtype=np.float32))
        z_loc = jnp.asarray(np.random.normal(0, 1, (C, k)).astype(np.float32))
        p_capture_loc = jnp.asarray(
            np.random.uniform(0.05, 0.5, C).astype(np.float32)
        )
        mc = ModelConfig(
            base_model="lnmvcp",
            parameterization=Parameterization.LOGISTIC_NORMAL_CANONICAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        return ScribeLaplaceResults(
            model_config=mc, mu=mu, W=W, d=d,
            n_genes=G, n_cells=C, z_loc=z_loc,
            alr_reference_idx=0,
            p_capture_loc=p_capture_loc,
            mu_T=jnp.asarray(15_000.0, dtype=jnp.float32),
            r_T=jnp.asarray(8.0, dtype=jnp.float32),
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

    def test_pln_three_levels_shapes(self):
        res = self._build_pln(with_capture=True)
        counts = np.random.poisson(15, (res.n_cells, res.n_genes)).astype(np.float32)
        out_pc = res.get_ppc_samples(
            rng_key=random.PRNGKey(0), n_samples=4,
            level="per_cell", counts=counts,
        )
        out_lib = res.get_ppc_samples(
            rng_key=random.PRNGKey(0), n_samples=4,
            level="library_anchored", counts=counts,
        )
        out_marg = res.get_ppc_samples(
            rng_key=random.PRNGKey(0), n_samples=4, level="marginal",
        )
        assert tuple(out_pc.shape) == (4, res.n_cells, res.n_genes)
        assert tuple(out_lib.shape) == (4, res.n_cells, res.n_genes)
        # Marginal returns (n_samples, G) — one fresh imaginary cell per sample.
        assert tuple(out_marg.shape) == (4, res.n_genes)

    def test_pln_marginal_samples_eta(self):
        """When eta_loc is present, marginal PPC should reflect bootstrapped
        η — totals should *vary* across samples, not all sit at one value
        (which would happen if η were silently held at zero or at the mean).
        """
        res = self._build_pln(with_capture=True)
        n_samples = 200
        out = np.asarray(
            res.get_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=n_samples,
                level="marginal",
            )
        )
        per_sample_totals = out.sum(axis=-1)  # (n_samples,)
        # If η bootstrap is working, totals span a wide range. The
        # silent η ≡ 0 path would give totals concentrated around
        # exp(μ_g + 0.5 σ²_g) summed — this test would catch a
        # regression that drops the η sampling.
        assert per_sample_totals.std() > 1.0, (
            f"marginal PPC totals do not vary: std={per_sample_totals.std():.2f}"
        )

    def test_pln_no_capture_marginal_works(self):
        """Marginal path should also work when there is no η to sample."""
        res = self._build_pln(with_capture=False)
        out = res.get_ppc_samples(
            rng_key=random.PRNGKey(0), n_samples=4, level="marginal",
        )
        assert tuple(out.shape) == (4, res.n_genes)

    def test_lnmvcp_marginal_samples_p_capture(self):
        """LNMVCP marginal PPC should bootstrap-sample p_capture so per-sample
        totals reflect the empirical capture distribution rather than
        collapsing to a single μ_T value (the previous bug).
        """
        res = self._build_lnmvcp()
        n_samples = 200
        out = np.asarray(
            res.get_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=n_samples,
                level="marginal",
            )
        )
        totals = out.sum(axis=-1)  # (n_samples,)
        # With p_capture bootstrapped from U(0.05, 0.5), post-capture
        # μ_T·p_capture spans 750–7500 and NB on top widens further.
        # Without the bootstrap (the old bug), totals would all sit at
        # ~μ_T = 15000.
        assert totals.std() > totals.mean() * 0.1, (
            f"LNMVCP marginal totals not varying with p_capture "
            f"(std={totals.std():.0f}, mean={totals.mean():.0f})"
        )

    def test_library_anchored_requires_counts(self):
        res = self._build_pln(with_capture=True)
        with pytest.raises(ValueError, match="library_anchored.*requires"):
            res.get_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=4,
                level="library_anchored",
            )

    def test_unknown_level_raises(self):
        res = self._build_pln(with_capture=True)
        with pytest.raises(ValueError, match="level must be"):
            res.get_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=4,
                level="not_a_level",
            )

    def test_library_anchored_totals_match_observed(self):
        """Library-anchored PPC must produce per-cell totals that
        exactly match the observed library sizes — that is the
        defining property of this PPC level.
        """
        res = self._build_pln(with_capture=True)
        counts = np.random.poisson(20, (res.n_cells, res.n_genes)).astype(np.float32)
        L_obs = counts.sum(axis=-1).astype(int)
        out = np.asarray(
            res.get_ppc_samples(
                rng_key=random.PRNGKey(0), n_samples=4,
                level="library_anchored", counts=counts,
            )
        )
        L_ppc = out.sum(axis=-1)  # (4, n_cells)
        for s in range(L_ppc.shape[0]):
            np.testing.assert_array_equal(L_ppc[s], L_obs)


class TestCorrelationResidual:
    """Verify the library-size / nuisance-direction projection
    machinery on ``ScribeLaplaceResults``.

    Plants a $W$ where factor 0 is a pure ``1_G`` direction (the
    "library-size" pattern) and factors 1, 2 carry block structure.
    The residual correlation should:
      * keep block within-correlations close to 1 (signal preserved).
      * drop cross-block correlations close to 0 (nuisance removed).
    """

    def _planted_pln_results(self):
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(401)
        G, k = 60, 3
        W = np.zeros((G, k), dtype=np.float32)
        W[:, 0] = 1.5  # library-size axis
        W[:30, 1] = np.random.normal(1.0, 0.05, 30).astype(np.float32)
        W[30:, 2] = np.random.normal(1.0, 0.05, 30).astype(np.float32)
        mu = jnp.zeros(G, dtype=jnp.float32)
        d = jnp.full(G, 0.05, dtype=jnp.float32)
        mc = ModelConfig(
            base_model="pln",
            parameterization=Parameterization.POISSON_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        return ScribeLaplaceResults(
            model_config=mc, mu=mu, W=jnp.asarray(W), d=d,
            n_genes=G, n_cells=10,
            x_loc=jnp.zeros((10, G), dtype=jnp.float32),
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

    def test_library_size_direction_recovers_planted_axis(self):
        res = self._planted_pln_results()
        e = np.asarray(res.get_library_size_direction())
        assert e.shape == (3,)
        # Planted library axis was factor 0; the unit-norm
        # detector should recover it (sign-ambiguous).
        assert abs(e[0]) > 0.99
        assert abs(e[1]) < 1e-3
        assert abs(e[2]) < 1e-3

    def test_correlation_residual_removes_cross_block_signal(self):
        res = self._planted_pln_results()
        C_full = np.asarray(res.get_correlation())
        C_resid = np.asarray(
            res.get_correlation_residual(method="library_size")
        )
        # Within-block correlations stay near 1 (signal preserved).
        assert C_resid[:30, :30].mean() > 0.95
        assert C_resid[30:, 30:].mean() > 0.95
        # Cross-block correlations were inflated by library-size in
        # the full matrix (~0.7); residual should drop them to ~0.
        assert abs(C_full[:30, 30:].mean()) > 0.5
        assert abs(C_resid[:30, 30:].mean()) < 0.05

    def test_pc_method_returns_distinct_residuals(self):
        """``method='pc'`` removes the dominant *variance* direction,
        which need not coincide with the library-size direction
        $\\mathbf{1}_G$ when the planted columns of $W$ are
        correlated. The two methods are *both* valid projections;
        test that they're internally consistent rather than equal.
        """
        res = self._planted_pln_results()
        C_pc1 = np.asarray(
            res.get_correlation_residual(method="pc", n_components=1)
        )
        C_pc2 = np.asarray(
            res.get_correlation_residual(method="pc", n_components=2)
        )
        # Removing more PCs should reduce signal monotonically.
        # Compare absolute mean off-diagonals.
        off_pc1 = np.abs(C_pc1 - np.diag(np.diag(C_pc1))).mean()
        off_pc2 = np.abs(C_pc2 - np.diag(np.diag(C_pc2))).mean()
        assert off_pc2 <= off_pc1 + 1e-6, (
            f"removing more PCs should not increase off-diagonal "
            f"signal (pc1={off_pc1:.4f}, pc2={off_pc2:.4f})"
        )

    def test_pc1_equals_library_size_for_pure_1G_W(self):
        """When ``W`` has *only* the library-size factor (a pure rank-1
        $\\mathbf{1}_G$ column with no other structure), PC1 of
        $W^\\top W$ coincides with the library-size direction. Both
        methods then produce the same residual.
        """
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(402)
        G, k = 30, 2
        # Pure library-size on factor 0; factor 1 is orthogonal noise.
        W = np.zeros((G, k), dtype=np.float32)
        W[:, 0] = 1.0
        # Make factor 1 mean-zero so it has no projection onto 1_G.
        W[:, 1] = np.random.normal(0, 0.5, G).astype(np.float32)
        W[:, 1] -= W[:, 1].mean()

        mc = ModelConfig(
            base_model="pln",
            parameterization=Parameterization.POISSON_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        res = ScribeLaplaceResults(
            model_config=mc, mu=jnp.zeros(G), W=jnp.asarray(W),
            d=jnp.full(G, 0.05), n_genes=G, n_cells=10,
            x_loc=jnp.zeros((10, G)),
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

        C_pc1 = np.asarray(res.get_correlation_residual(method="pc", n_components=1))
        C_lib = np.asarray(res.get_correlation_residual(method="library_size"))
        np.testing.assert_allclose(C_pc1, C_lib, atol=1e-4)

    def test_correlation_residual_invalid_method(self):
        res = self._planted_pln_results()
        with pytest.raises(ValueError, match="method must be"):
            res.get_correlation_residual(method="not_a_real_method")

    def test_residual_correlation_shape_matches_full(self):
        res = self._planted_pln_results()
        C_full = res.get_correlation()
        C_resid = res.get_correlation_residual(method="library_size")
        assert C_resid.shape == C_full.shape


class TestSummarizeCorrelationStructure:
    """Verify ``summarize_correlation_structure`` returns the expected
    diagnostics dict and that the silent (``verbose=False``) path
    matches the verbose computation byte-for-byte (i.e. no
    side-effects from the printing code).
    """

    def _planted_pln_results(self):
        from scribe import ScribeLaplaceResults
        from scribe.models.config import ModelConfig
        from scribe.models.config.enums import (
            InferenceMethod,
            Parameterization,
        )

        np.random.seed(501)
        G, k = 50, 3
        W = np.zeros((G, k), dtype=np.float32)
        W[:, 0] = 1.5
        W[:25, 1] = np.random.normal(1.0, 0.05, 25).astype(np.float32)
        W[25:, 2] = np.random.normal(1.0, 0.05, 25).astype(np.float32)
        mc = ModelConfig(
            base_model="pln",
            parameterization=Parameterization.POISSON_LOGNORMAL,
            inference_method=InferenceMethod.LAPLACE,
        )
        return ScribeLaplaceResults(
            model_config=mc, mu=jnp.zeros(G), W=jnp.asarray(W),
            d=jnp.full(G, 0.05), n_genes=G, n_cells=10,
            x_loc=jnp.zeros((10, G)),
            losses=jnp.zeros(1), final_grad_norms=jnp.zeros(1),
        )

    def test_returned_dict_has_expected_keys(self):
        res = self._planted_pln_results()
        out = res.summarize_correlation_structure(verbose=False)
        expected_keys = {
            "n_genes_effective",
            "n_latent_factors",
            "cos_We_1G",
            "We_concentration",
            "We_rms",
            "library_axis_share",
            "eigenvalues",
            "eigenvalue_fractions",
            "effective_rank",
            "offdiag_quantiles_full",
            "offdiag_quantiles_after_library",
            "offdiag_quantiles_after_pc1",
        }
        assert expected_keys.issubset(out.keys())

    def test_diagnostics_recover_planted_axis(self):
        """On a planted W with factor 0 = 1.5 * 1_G, the library-size
        diagnostics should be near-perfect.
        """
        res = self._planted_pln_results()
        out = res.summarize_correlation_structure(verbose=False)
        # Cosine ≈ 1 (exact-aligned axis was planted).
        assert out["cos_We_1G"] > 0.999
        # We_rms equals the planted column magnitude (1.5).
        assert abs(out["We_rms"] - 1.5) < 1e-3
        # Library-axis share is the dominant fraction (>0.5) since
        # the planted factor 0 has the largest column norm of W.
        assert out["library_axis_share"] > 0.5

    def test_eigenvalues_are_descending(self):
        res = self._planted_pln_results()
        out = res.summarize_correlation_structure(verbose=False)
        eigs = np.asarray(out["eigenvalues"])
        # Strictly descending for non-degenerate W^T W.
        assert np.all(np.diff(eigs) <= 0)
        # Fractions sum to 1.
        assert abs(sum(out["eigenvalue_fractions"]) - 1.0) < 1e-5

    def test_offdiag_quantiles_after_library_drop_signal(self):
        """For the planted W, library-size projection drops the
        off-diagonal *median* correlation substantially relative
        to the full matrix — the user-visible signature in the
        heatmap.
        """
        res = self._planted_pln_results()
        out = res.summarize_correlation_structure(verbose=False)
        full_p50 = abs(out["offdiag_quantiles_full"]["p50"])
        lib_p50 = abs(out["offdiag_quantiles_after_library"]["p50"])
        assert lib_p50 < full_p50, (
            f"library-size projection should reduce |p50| of "
            f"off-diagonals (full={full_p50:.3f}, lib={lib_p50:.3f})"
        )

    def test_verbose_silent_paths_agree(self, capsys):
        """The printing code must not mutate the diagnostics; the
        verbose call should emit text but return the same dict as
        the silent call.
        """
        res = self._planted_pln_results()
        silent = res.summarize_correlation_structure(verbose=False)
        verbose = res.summarize_correlation_structure(verbose=True)
        assert silent.keys() == verbose.keys()
        # Numeric scalars should match exactly.
        for k_ in ("cos_We_1G", "We_rms", "library_axis_share",
                   "effective_rank"):
            assert silent[k_] == verbose[k_]

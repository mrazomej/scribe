"""Pure-kernel tests for the LNM Laplace Newton kernels.

Mirrors the structure of ``tests/test_laplace_newton.py`` (the PLN
counterpart). Two variants are tested:

* **z-space (``d_mode='low_rank'``)**: ``newton_step_z`` and the
  associated ``log det(-H_z)`` helper. Hessian is ``k × k``, no
  Woodbury.
* **y_alr-space (``d_mode='learned'``)**: ``newton_step_y_alr``
  and its ``log det(-H_y)`` helper. Reuses scribe's Woodbury
  scaffolding plus a Sherman-Morrison correction for the
  rank-1 ``-N ρρ^T`` term.

All tests verify the kernel's output against an independent dense
reference (``np.linalg.solve`` on the explicit Hessian, or
``np.linalg.slogdet``) at a non-MAP iterate. The dense reference
catches sign/shape bugs that converged-MAP-only tests cannot.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.laplace._newton_lnm import (
    laplace_log_det_neg_H_y_alr,
    laplace_log_det_neg_H_z,
    newton_step_y_alr,
    newton_step_z,
)


# =====================================================================
# Test fixtures
# =====================================================================


def _build_random_lnm_problem(G=6, k=2, seed=0):
    """Synthetic LNM problem with a non-trivial cell.

    Returns
    -------
    mu : (G-1,)
    W  : (G-1, k)
    d  : (G-1,)  positive
    u_full : (G,)  multinomial counts
    alr_ref : int  reference gene index in 0..G-1
    """
    rng = np.random.default_rng(seed)
    g_alr = G - 1
    mu = rng.normal(size=g_alr).astype(np.float32) * 0.4
    W = (0.4 * rng.normal(size=(g_alr, k))).astype(np.float32)
    d = (0.05 + 0.1 * rng.uniform(size=g_alr)).astype(np.float32)
    # Build a multinomial sample at a random "true" probability.
    p_true = rng.dirichlet(np.ones(G) * 2.0).astype(np.float32)
    n_total = 50
    u_full = rng.multinomial(n_total, p_true).astype(np.float32)
    alr_ref = int(rng.integers(0, G))
    return mu, W, d, u_full, alr_ref


def _split_alr(u_full: np.ndarray, alr_ref: int) -> np.ndarray:
    """Drop the reference-gene count from a full ``u_full`` (G-vector)."""
    mask = np.arange(u_full.shape[-1]) != alr_ref
    return u_full[..., mask]


# =====================================================================
# Variant A: z-space Newton step
# =====================================================================


class TestNewtonStepZ:
    """``newton_step_z`` matches a dense block solve at a non-MAP iterate.

    The wrong-sign-Newton bug we caught in the PLN kernel was
    invisible to the converged-MAP test because damping + step-size
    cap absorbed bad directions. This test does the same single-
    step check for the LNM z-space kernel.
    """

    def test_single_step_matches_dense(self):
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=6, k=2, seed=11
        )
        u_alr_np = _split_alr(u_full_np, alr_ref)
        n_total_np = float(u_full_np.sum())

        # Iterate intentionally NOT at the MAP — pick z so the
        # cross-coupling term ``W^T M W`` is non-trivial.
        rng = np.random.default_rng(11)
        z_np = rng.standard_normal(size=(W_np.shape[1],)).astype(np.float32)

        # ---- Kernel step ----
        z_new, _gn = newton_step_z(
            jnp.asarray(z_np),
            jnp.asarray(u_alr_np),
            jnp.asarray(n_total_np),
            jnp.asarray(mu_np),
            jnp.asarray(W_np),
            alr_ref,
            int(u_full_np.shape[0]),
            damping=0.0,
        )
        delta_kernel = np.asarray(z_new) - z_np

        # ---- Dense reference ----
        # softmax probabilities at the current z (full G then drop ref).
        y_alr = mu_np + W_np @ z_np
        # Augment to G-vector with 0 at ref.
        y_full = np.zeros(u_full_np.shape, dtype=np.float64)
        non_ref = [g for g in range(u_full_np.shape[0]) if g != alr_ref]
        y_full[non_ref] = y_alr.astype(np.float64)
        p_full = np.exp(y_full - y_full.max())
        p_full = p_full / p_full.sum()
        p_alr = p_full[non_ref]
        # Gradient and Hessian.
        grad = (
            W_np.astype(np.float64).T @ (u_alr_np.astype(np.float64) - n_total_np * p_alr)
            - z_np.astype(np.float64)
        )
        # M_alr = N (diag(p_alr) - p_alr p_alr^T)
        # -H_z = W^T M_alr W + I
        M_alr = n_total_np * (np.diag(p_alr) - np.outer(p_alr, p_alr))
        neg_H = W_np.astype(np.float64).T @ M_alr @ W_np.astype(np.float64) + np.eye(W_np.shape[1])
        delta_dense = np.linalg.solve(neg_H, grad)

        np.testing.assert_allclose(
            delta_kernel, delta_dense, rtol=2e-4, atol=2e-4
        )


class TestLogDetNegHZ:
    """``laplace_log_det_neg_H_z`` matches a dense ``slogdet``."""

    def test_matches_dense_slogdet(self):
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=8, k=3, seed=4
        )
        u_alr_np = _split_alr(u_full_np, alr_ref)
        n_total_np = float(u_full_np.sum())
        # Use the MAP-ish iterate z=0 plus a small perturbation.
        z_np = np.array([0.1, -0.2, 0.05], dtype=np.float32)

        # Kernel value.
        ld = float(
            laplace_log_det_neg_H_z(
                jnp.asarray(z_np),
                jnp.asarray(u_alr_np),
                jnp.asarray(n_total_np),
                jnp.asarray(mu_np),
                jnp.asarray(W_np),
                alr_ref,
                int(u_full_np.shape[0]),
            )
        )

        # Dense reference.
        y_alr = mu_np + W_np @ z_np
        y_full = np.zeros(u_full_np.shape, dtype=np.float64)
        non_ref = [g for g in range(u_full_np.shape[0]) if g != alr_ref]
        y_full[non_ref] = y_alr.astype(np.float64)
        p_full = np.exp(y_full - y_full.max())
        p_full = p_full / p_full.sum()
        p_alr = p_full[non_ref]
        M_alr = n_total_np * (np.diag(p_alr) - np.outer(p_alr, p_alr))
        neg_H = W_np.astype(np.float64).T @ M_alr @ W_np.astype(np.float64) + np.eye(W_np.shape[1])
        sign, ld_ref = np.linalg.slogdet(neg_H)
        assert sign > 0
        assert ld == pytest.approx(float(ld_ref), abs=1e-3, rel=1e-3)

    def test_has_live_gradient_through_W(self):
        """``jax.grad`` of ``log det(-H_z)`` w.r.t. ``W`` is non-zero.

        Regression for the PLN-side stop-gradient bug; confirms the
        LNM helper has the same gradient-flow guarantee.
        """
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=6, k=2, seed=2
        )
        u_alr = jnp.asarray(_split_alr(u_full_np, alr_ref))
        n_total = jnp.asarray(float(u_full_np.sum()))
        mu = jnp.asarray(mu_np)
        W = jnp.asarray(W_np)
        z = jnp.zeros(W_np.shape[1], dtype=jnp.float32)

        def f(W_):
            return laplace_log_det_neg_H_z(
                z, u_alr, n_total, mu, W_, alr_ref, int(u_full_np.shape[0])
            )

        grad_W = jax.grad(f)(W)
        assert float(jnp.max(jnp.abs(grad_W))) > 1e-3


# =====================================================================
# Variant B: y_alr-space Newton step
# =====================================================================


class TestNewtonStepYAlr:
    """``newton_step_y_alr`` matches a dense block solve at a non-MAP iterate."""

    def test_single_step_matches_dense(self):
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=8, k=2, seed=23
        )
        u_alr_np = _split_alr(u_full_np, alr_ref)
        n_total_np = float(u_full_np.sum())

        # Iterate intentionally not at the MAP.
        rng = np.random.default_rng(23)
        y_np = (mu_np + 0.3 * rng.standard_normal(size=mu_np.shape)).astype(
            np.float32
        )

        # ---- Kernel step ----
        y_new, _gn = newton_step_y_alr(
            jnp.asarray(y_np),
            jnp.asarray(u_alr_np),
            jnp.asarray(n_total_np),
            jnp.asarray(mu_np),
            jnp.asarray(W_np),
            jnp.asarray(d_np),
            alr_ref,
            int(u_full_np.shape[0]),
            damping=0.0,
        )
        delta_kernel = np.asarray(y_new) - y_np

        # ---- Dense reference ----
        # ρ_alr at y_np.
        y_full = np.zeros(u_full_np.shape, dtype=np.float64)
        non_ref = [g for g in range(u_full_np.shape[0]) if g != alr_ref]
        y_full[non_ref] = y_np.astype(np.float64)
        p_full = np.exp(y_full - y_full.max())
        p_full = p_full / p_full.sum()
        p_alr = p_full[non_ref]
        # Gradient.
        Sigma = (
            W_np.astype(np.float64) @ W_np.astype(np.float64).T
            + np.diag(d_np.astype(np.float64))
        )
        Sigma_inv = np.linalg.inv(Sigma)
        diff = (y_np - mu_np).astype(np.float64)
        grad = u_alr_np.astype(np.float64) - n_total_np * p_alr - Sigma_inv @ diff
        # -H_y = N (diag(p_alr) - p_alr p_alr^T) + Σ⁻¹.
        M_alr = n_total_np * (np.diag(p_alr) - np.outer(p_alr, p_alr))
        neg_H = M_alr + Sigma_inv
        delta_dense = np.linalg.solve(neg_H, grad)

        np.testing.assert_allclose(
            delta_kernel, delta_dense, rtol=2e-3, atol=2e-3
        )


class TestLogDetNegHYAlr:
    """``laplace_log_det_neg_H_y_alr`` matches a dense ``slogdet``."""

    def test_matches_dense_slogdet(self):
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=8, k=3, seed=5
        )
        u_alr_np = _split_alr(u_full_np, alr_ref)
        n_total_np = float(u_full_np.sum())
        y_np = (mu_np + 0.05 * np.random.randn(*mu_np.shape).astype(np.float32))

        # Kernel value.
        ld = float(
            laplace_log_det_neg_H_y_alr(
                jnp.asarray(y_np),
                jnp.asarray(u_alr_np),
                jnp.asarray(n_total_np),
                jnp.asarray(mu_np),
                jnp.asarray(W_np),
                jnp.asarray(d_np),
                alr_ref,
                int(u_full_np.shape[0]),
            )
        )

        # Dense reference.
        y_full = np.zeros(u_full_np.shape, dtype=np.float64)
        non_ref = [g for g in range(u_full_np.shape[0]) if g != alr_ref]
        y_full[non_ref] = y_np.astype(np.float64)
        p_full = np.exp(y_full - y_full.max())
        p_full = p_full / p_full.sum()
        p_alr = p_full[non_ref]
        Sigma = (
            W_np.astype(np.float64) @ W_np.astype(np.float64).T
            + np.diag(d_np.astype(np.float64))
        )
        Sigma_inv = np.linalg.inv(Sigma)
        M_alr = n_total_np * (np.diag(p_alr) - np.outer(p_alr, p_alr))
        neg_H = M_alr + Sigma_inv
        sign, ld_ref = np.linalg.slogdet(neg_H)
        assert sign > 0
        # Tolerance is looser here (1e-2) because the Sherman-
        # Morrison + Woodbury composition involves ~G operations
        # in float32, so cumulative rounding can drift more than
        # a single dense slogdet.
        assert ld == pytest.approx(float(ld_ref), abs=5e-2, rel=5e-3)

    def test_has_live_gradient_through_W_and_d(self):
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=6, k=2, seed=7
        )
        u_alr = jnp.asarray(_split_alr(u_full_np, alr_ref))
        n_total = jnp.asarray(float(u_full_np.sum()))
        mu = jnp.asarray(mu_np)
        W = jnp.asarray(W_np)
        d = jnp.asarray(d_np)
        y = jnp.asarray(mu_np)

        def fW(W_):
            return laplace_log_det_neg_H_y_alr(
                y, u_alr, n_total, mu, W_, d, alr_ref, int(u_full_np.shape[0])
            )

        def fd(d_):
            return laplace_log_det_neg_H_y_alr(
                y, u_alr, n_total, mu, W, d_, alr_ref, int(u_full_np.shape[0])
            )

        assert float(jnp.max(jnp.abs(jax.grad(fW)(W)))) > 1e-3
        assert float(jnp.max(jnp.abs(jax.grad(fd)(d)))) > 1e-3


# =====================================================================
# Vmap / JIT smoke tests
# =====================================================================


class TestVmapJIT:
    """Both kernels work under ``vmap`` and ``jit`` — same pattern as PLN."""

    def test_vmap_z(self):
        from scribe.laplace._newton_lnm import laplace_newton_batch_z

        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=6, k=2, seed=9
        )
        # Build a 4-cell batch.
        u_full_batch = np.tile(u_full_np[None, :], (4, 1)).astype(np.float32)
        u_alr_batch = jnp.asarray(_split_alr(u_full_batch, alr_ref))
        n_total_batch = jnp.asarray(u_full_batch.sum(axis=-1))
        z_init = jnp.zeros((4, W_np.shape[1]), dtype=jnp.float32)

        z_final, gn = laplace_newton_batch_z(
            z_init, u_alr_batch, n_total_batch,
            jnp.asarray(mu_np), jnp.asarray(W_np),
            alr_ref, int(u_full_np.shape[0]),
            8, 1e-4,
        )
        assert z_final.shape == (4, W_np.shape[1])
        # All 4 cells should converge to the same point (same data).
        np.testing.assert_allclose(
            z_final, np.broadcast_to(z_final[0], z_final.shape), atol=1e-4
        )

    def test_jit_z(self):
        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=6, k=2, seed=10
        )
        u_alr = jnp.asarray(_split_alr(u_full_np, alr_ref))
        n_total = jnp.asarray(float(u_full_np.sum()))
        z = jnp.zeros(W_np.shape[1], dtype=jnp.float32)

        @jax.jit
        def step(z_, mu_, W_):
            z_new, _gn = newton_step_z(
                z_, u_alr, n_total, mu_, W_,
                alr_ref, int(u_full_np.shape[0]), 0.0,
            )
            return z_new

        out = step(z, jnp.asarray(mu_np), jnp.asarray(W_np))
        assert out.shape == z.shape

    def test_vmap_y_alr(self):
        from scribe.laplace._newton_lnm import laplace_newton_batch_y_alr

        mu_np, W_np, d_np, u_full_np, alr_ref = _build_random_lnm_problem(
            G=8, k=2, seed=12
        )
        u_full_batch = np.tile(u_full_np[None, :], (4, 1)).astype(np.float32)
        u_alr_batch = jnp.asarray(_split_alr(u_full_batch, alr_ref))
        n_total_batch = jnp.asarray(u_full_batch.sum(axis=-1))
        y_init = jnp.broadcast_to(jnp.asarray(mu_np), (4,) + mu_np.shape)

        y_final, gn = laplace_newton_batch_y_alr(
            y_init, u_alr_batch, n_total_batch,
            jnp.asarray(mu_np), jnp.asarray(W_np), jnp.asarray(d_np),
            alr_ref, int(u_full_np.shape[0]),
            8, 1e-3,
        )
        assert y_final.shape == (4, mu_np.shape[0])
        np.testing.assert_allclose(
            y_final, np.broadcast_to(y_final[0], y_final.shape), atol=1e-3
        )

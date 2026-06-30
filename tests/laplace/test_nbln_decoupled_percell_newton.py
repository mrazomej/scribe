"""Decoupled-layout per-cell-W / per-cell-mu Newton twins (step 6).

The per-donor correlation hierarchy (and its per-donor marginal cascade) on
the DECOUPLED layout. The ``*_decoupled_percellW`` / ``*_decoupled_percellWmu``
twins are the decoupled per-cell functions vmapped with ``W`` (and, for the
``Wmu`` variants, ``mu``) mapped over the cell axis instead of broadcast.

Unlike the legacy twins, the decoupled log-det AND grad kernels both consume
``mu`` (``log_rate = μ + x_dev``), so the ``_percellWmu`` decoupled family
includes log-det twins. These tests prove that broadcasting ``W``/``mu``
identically to every cell reproduces the shared-kernel result bit-for-bit
(float32), the no-regression guarantee for the decoupled hierarchy path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import scribe.laplace._newton_nbln as NB

N, GKEPT, K = 14, 5, 2
GOBS = GKEPT + 1  # one pooled `_other` column
NN, DMP, MS = 10, 1e-2, 5.0


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    mu = jnp.asarray(rng.normal(size=GOBS), dtype=jnp.float32)       # (G_obs,)
    W = jnp.asarray(0.3 * rng.normal(size=(GKEPT, K)), dtype=jnp.float32)
    d = jnp.asarray(0.05 + 0.5 * rng.uniform(size=GKEPT), dtype=jnp.float32)
    r = jnp.asarray(0.5 + rng.uniform(size=GOBS), dtype=jnp.float32)
    u = jnp.asarray(rng.poisson(5.0, size=(N, GOBS)), dtype=jnp.float32)
    kept = jnp.arange(GKEPT, dtype=jnp.int32)
    xdev = jnp.zeros((N, GKEPT), dtype=jnp.float32)
    return mu, W, d, r, u, kept, xdev


def _broadcast(W, mu):
    return jnp.broadcast_to(W, (N, GKEPT, K)), jnp.broadcast_to(mu, (N, GOBS))


def test_decoupled_x_only_percell_matches_shared():
    mu, W, d, r, u, kept, xdev = _problem(0)
    Wpc, mupc = _broadcast(W, mu)
    xs, _, _ = NB.laplace_newton_batch_x_only_decoupled(
        xdev, u, mu, W, d, r, kept, NN, DMP, MS
    )
    xw, _, _ = NB.laplace_newton_batch_x_only_decoupled_percellW(
        xdev, u, mu, Wpc, d, r, kept, NN, DMP, MS
    )
    xm, _, _ = NB.laplace_newton_batch_x_only_decoupled_percellWmu(
        xdev, u, mupc, Wpc, d, r, kept, NN, DMP, MS
    )
    npt.assert_allclose(np.asarray(xw), np.asarray(xs), atol=1e-5)
    npt.assert_allclose(np.asarray(xm), np.asarray(xs), atol=1e-5)


def test_decoupled_offset_percell_matches_shared():
    mu, W, d, r, u, kept, xdev = _problem(1)
    Wpc, mupc = _broadcast(W, mu)
    eoff = jnp.asarray(
        0.1 * np.random.default_rng(1).normal(size=N), dtype=jnp.float32
    )
    xs, _, _ = NB.laplace_newton_batch_x_only_offset_decoupled(
        xdev, u, mu, W, d, r, eoff, kept, NN, DMP, MS
    )
    xw, _, _ = NB.laplace_newton_batch_x_only_offset_decoupled_percellW(
        xdev, u, mu, Wpc, d, r, eoff, kept, NN, DMP, MS
    )
    xm, _, _ = NB.laplace_newton_batch_x_only_offset_decoupled_percellWmu(
        xdev, u, mupc, Wpc, d, r, eoff, kept, NN, DMP, MS
    )
    npt.assert_allclose(np.asarray(xw), np.asarray(xs), atol=1e-5)
    npt.assert_allclose(np.asarray(xm), np.asarray(xs), atol=1e-5)


def test_decoupled_joint_percell_matches_shared():
    mu, W, d, r, u, kept, xdev = _problem(2)
    Wpc, mupc = _broadcast(W, mu)
    rng = np.random.default_rng(2)
    eta = jnp.zeros((N,), dtype=jnp.float32)
    ea = jnp.asarray(0.1 * rng.normal(size=N), dtype=jnp.float32)
    se = jnp.full((N,), 0.3, dtype=jnp.float32)
    xs, es, _, _ = NB.laplace_newton_batch_decoupled(
        xdev, eta, u, mu, W, d, r, kept, ea, se, NN, DMP, MS
    )
    xw, ew, _, _ = NB.laplace_newton_batch_decoupled_percellW(
        xdev, eta, u, mu, Wpc, d, r, kept, ea, se, NN, DMP, MS
    )
    xm, em, _, _ = NB.laplace_newton_batch_decoupled_percellWmu(
        xdev, eta, u, mupc, Wpc, d, r, kept, ea, se, NN, DMP, MS
    )
    npt.assert_allclose(np.asarray(xw), np.asarray(xs), atol=1e-5)
    npt.assert_allclose(np.asarray(xm), np.asarray(xs), atol=1e-5)
    npt.assert_allclose(np.asarray(ew), np.asarray(es), atol=1e-5)
    npt.assert_allclose(np.asarray(em), np.asarray(es), atol=1e-5)


def test_decoupled_logdet_and_grad_percell_match_shared():
    mu, W, d, r, u, kept, xdev = _problem(3)
    Wpc, mupc = _broadcast(W, mu)
    x, _, _ = NB.laplace_newton_batch_x_only_decoupled(
        xdev, u, mu, W, d, r, kept, NN, DMP, MS
    )
    # log det (x-only): shared vs percellW vs percellWmu.
    ld_s = NB.laplace_log_det_neg_H_batch_x_only_decoupled(
        x, None, u, r, W, d, mu, kept, 1.0
    )
    ld_w = NB.laplace_log_det_neg_H_batch_x_only_decoupled_percellW(
        x, None, u, r, Wpc, d, mu, kept, 1.0
    )
    ld_m = NB.laplace_log_det_neg_H_batch_x_only_decoupled_percellWmu(
        x, None, u, r, Wpc, d, mupc, kept, 1.0
    )
    npt.assert_allclose(np.asarray(ld_w), np.asarray(ld_s), atol=1e-4)
    npt.assert_allclose(np.asarray(ld_m), np.asarray(ld_s), atol=1e-4)
    # grad norm (x-only).
    g_s = NB.nbln_grad_x_only_norm_batch_decoupled(x, u, mu, W, d, r, kept)
    g_w = NB.nbln_grad_x_only_norm_batch_decoupled_percellW(
        x, u, mu, Wpc, d, r, kept
    )
    g_m = NB.nbln_grad_x_only_norm_batch_decoupled_percellWmu(
        x, u, mupc, Wpc, d, r, kept
    )
    npt.assert_allclose(np.asarray(g_w), np.asarray(g_s), atol=1e-5)
    npt.assert_allclose(np.asarray(g_m), np.asarray(g_s), atol=1e-5)


def test_decoupled_percellWmu_per_donor_gather():
    """Two donors with distinct mu: per-cell == per-donor shared runs."""
    mu0, W, d, r, u, kept, xdev = _problem(4)
    rng = np.random.default_rng(11)
    mu1 = jnp.asarray(mu0 + rng.normal(size=GOBS) * 0.6, dtype=jnp.float32)
    ds = jnp.arange(N) % 2
    Wpc = jnp.broadcast_to(W, (N, GKEPT, K))
    mupc = jnp.stack([mu0, mu1], axis=0)[ds]  # (N, G_obs)

    x_pc, _, _ = NB.laplace_newton_batch_x_only_decoupled_percellWmu(
        xdev, u, mupc, Wpc, d, r, kept, NN, DMP, MS
    )
    x_ref = np.array(x_pc)
    for donor, mud in [(0, mu0), (1, mu1)]:
        mask = np.asarray(ds) == donor
        x_d, _, _ = NB.laplace_newton_batch_x_only_decoupled(
            xdev[mask], u[mask], mud, W, d, r, kept, NN, DMP, MS
        )
        x_ref[mask] = np.asarray(x_d)
    npt.assert_allclose(np.asarray(x_pc), x_ref, atol=1e-5)

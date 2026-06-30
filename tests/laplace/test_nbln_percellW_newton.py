"""Per-cell-W legacy Newton twins for the per-donor correlation hierarchy.

The hierarchical gene-gene correlation model (NB-LogNormal Rung 1) gives each
cell a donor-specific prior covariance
``Σ_{σ(c)} = W_eff,{σ(c)} W_eff,{σ(c)}ᵀ + diag(d)`` with
``W_eff,d = W·diag(s_d)``. Since a mini-batch mixes donors, the effective
loadings differ per cell, so the Newton kernels must take a per-cell
``(N, G, k)`` ``W`` instead of the shared ``(G, k)`` matrix.

The ``_percellW`` twins are the *same* per-cell functions vmapped with ``W``'s
axis flipped from shared (``None``) to per-cell (``0``). These tests prove:

1. **Consistency** — with ``W`` broadcast identically to every cell, each
   ``_percellW`` kernel reproduces its shared-``W`` counterpart bit-for-bit
   (modulo float32). This is the no-regression guarantee.
2. **Per-donor correctness** — with two distinct donor ``W``s, the per-cell
   kernel's MAP for each cell equals the shared-``W`` kernel run on that
   donor's cells with that donor's ``W`` (the per-cell gather is honored).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from jax import random

from scribe.laplace._newton_nbln import (
    # shared-W (reference)
    laplace_newton_batch_x_only,
    laplace_newton_batch_x_only_offset,
    laplace_newton_batch,
    laplace_log_det_neg_H_batch_x_only,
    nbln_grad_x_only_norm_batch,
    # per-cell-W (under test)
    laplace_newton_batch_x_only_percellW,
    laplace_newton_batch_x_only_offset_percellW,
    laplace_newton_batch_percellW,
    laplace_log_det_neg_H_batch_x_only_percellW,
    nbln_grad_x_only_norm_batch_percellW,
)

N, G, K = 16, 6, 2
N_NEWTON, DAMPING, MAX_STEP = 12, 1e-2, 10.0


def _problem(seed: int = 0):
    """Build a small, valid NBLN per-cell Newton problem."""
    rng = np.random.default_rng(seed)
    mu = jnp.asarray(rng.normal(size=G), dtype=jnp.float32)
    W = jnp.asarray(0.3 * rng.normal(size=(G, K)), dtype=jnp.float32)
    d = jnp.asarray(0.05 + 0.5 * rng.uniform(size=G), dtype=jnp.float32)
    r = jnp.asarray(0.5 + rng.uniform(size=G), dtype=jnp.float32)
    counts = jnp.asarray(rng.poisson(5.0, size=(N, G)), dtype=jnp.float32)
    latent_init = jnp.log(counts + 1.0) - mu[None, :]
    return mu, W, d, r, counts, latent_init


# ---------------------------------------------------------------------------
# Consistency: broadcast W -> identical to shared-W kernels
# ---------------------------------------------------------------------------


def test_x_only_percellW_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(0)
    W_pc = jnp.broadcast_to(W, (N, G, K))

    x_shared, _, _ = laplace_newton_batch_x_only(
        latent_init, counts, mu, W, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    x_pc, _, _ = laplace_newton_batch_x_only_percellW(
        latent_init, counts, mu, W_pc, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    npt.assert_allclose(np.asarray(x_pc), np.asarray(x_shared), atol=1e-5)


def test_x_only_offset_percellW_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(1)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    eta_off = jnp.asarray(
        0.1 * np.random.default_rng(1).normal(size=N), dtype=jnp.float32
    )

    x_shared, _, _ = laplace_newton_batch_x_only_offset(
        latent_init, counts, mu, W, d, r, eta_off, N_NEWTON, DAMPING, MAX_STEP
    )
    x_pc, _, _ = laplace_newton_batch_x_only_offset_percellW(
        latent_init, counts, mu, W_pc, d, r, eta_off, N_NEWTON, DAMPING, MAX_STEP
    )
    npt.assert_allclose(np.asarray(x_pc), np.asarray(x_shared), atol=1e-5)


def test_joint_percellW_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(2)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    rng = np.random.default_rng(2)
    eta_init = jnp.zeros((N,), dtype=jnp.float32)
    eta_anchor = jnp.asarray(0.1 * rng.normal(size=N), dtype=jnp.float32)
    sigma_eta = jnp.full((N,), 0.3, dtype=jnp.float32)

    x_s, eta_s, _, _ = laplace_newton_batch(
        latent_init, eta_init, counts, mu, W, d, r, eta_anchor, sigma_eta,
        N_NEWTON, DAMPING, MAX_STEP,
    )
    x_pc, eta_pc, _, _ = laplace_newton_batch_percellW(
        latent_init, eta_init, counts, mu, W_pc, d, r, eta_anchor, sigma_eta,
        N_NEWTON, DAMPING, MAX_STEP,
    )
    npt.assert_allclose(np.asarray(x_pc), np.asarray(x_s), atol=1e-5)
    npt.assert_allclose(np.asarray(eta_pc), np.asarray(eta_s), atol=1e-5)


def test_logdet_and_gradnorm_percellW_match_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(3)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    x, _, _ = laplace_newton_batch_x_only(
        latent_init, counts, mu, W, d, r, N_NEWTON, DAMPING, MAX_STEP
    )

    ld_s = laplace_log_det_neg_H_batch_x_only(x, None, counts, r, W, d, 1.0)
    ld_pc = laplace_log_det_neg_H_batch_x_only_percellW(
        x, None, counts, r, W_pc, d, 1.0
    )
    npt.assert_allclose(np.asarray(ld_pc), np.asarray(ld_s), atol=1e-5)

    gn_s = nbln_grad_x_only_norm_batch(x, counts, mu, W, d, r)
    gn_pc = nbln_grad_x_only_norm_batch_percellW(x, counts, mu, W_pc, d, r)
    npt.assert_allclose(np.asarray(gn_pc), np.asarray(gn_s), atol=1e-5)


# ---------------------------------------------------------------------------
# Per-donor correctness: distinct donor W's gathered per cell
# ---------------------------------------------------------------------------


def test_x_only_percellW_per_donor_gather():
    """Two donors with distinct W: per-cell result == per-donor shared-W runs."""
    mu, W0, d, r, counts, latent_init = _problem(4)
    # A genuinely different second-donor loadings matrix.
    W1 = jnp.asarray(
        0.3 * np.random.default_rng(99).normal(size=(G, K)), dtype=jnp.float32
    )
    # Alternating donor assignment so both donors are well represented.
    ds = jnp.arange(N) % 2  # 0,1,0,1,...
    W_stack = jnp.stack([W0, W1], axis=0)  # (2, G, K)
    W_pc = W_stack[ds]  # (N, G, K) per-cell gather

    x_pc, _, _ = laplace_newton_batch_x_only_percellW(
        latent_init, counts, mu, W_pc, d, r, N_NEWTON, DAMPING, MAX_STEP
    )

    # Reference: run the shared-W kernel on each donor's cells separately.
    x_ref = np.array(x_pc)  # fill in below
    for donor, Wd in [(0, W0), (1, W1)]:
        mask = np.asarray(ds) == donor
        x_d, _, _ = laplace_newton_batch_x_only(
            latent_init[mask], counts[mask], mu, Wd, d, r,
            N_NEWTON, DAMPING, MAX_STEP,
        )
        x_ref[mask] = np.asarray(x_d)

    npt.assert_allclose(np.asarray(x_pc), x_ref, atol=1e-5)
    # Sanity: the two donors actually produce different MAPs (W's differ),
    # so this isn't a trivially-passing test.
    assert not np.allclose(
        np.asarray(x_pc)[np.asarray(ds) == 0][:1],
        np.asarray(x_pc)[np.asarray(ds) == 1][:1],
        atol=1e-3,
    )

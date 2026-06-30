"""Per-cell-W AND per-cell-mu Newton twins (per-donor marginal cascade).

The hierarchical-marginal cascade (step 4b) freezes a *leaf-level* mean
matrix ``mu`` of shape ``(D, G)``, so each cell's latent prior mean is its
donor's ``mu^{(σ(c))}`` (gathered to a per-cell ``(N, G)`` tensor) on top of
the donor-specific effective loadings ``W_eff,{σ(c)}``.  The ``_percellWmu``
twins are the *same* per-cell Newton/grad functions vmapped with BOTH ``W``
and ``mu`` mapped over cells (``in_axes`` flipped ``None -> 0``).  These tests
prove:

1. **Consistency** — with ``W`` and ``mu`` broadcast identically to every
   cell, each ``_percellWmu`` kernel reproduces its shared-``mu`` ``_percellW``
   counterpart (and hence the shared-``W`` reference) bit-for-bit (float32).
2. **Per-donor correctness** — with two donors carrying distinct ``mu`` (and
   distinct ``W``), the per-cell kernel's MAP for each cell equals the
   shared-kernel run on that donor's cells with that donor's ``(W, mu)``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from scribe.laplace._newton_nbln import (
    # shared-W (reference)
    laplace_newton_batch_x_only,
    laplace_newton_batch_x_only_offset,
    laplace_newton_batch,
    nbln_grad_x_only_norm_batch,
    # per-cell-W, shared-mu
    laplace_newton_batch_x_only_percellW,
    # per-cell-W AND per-cell-mu (under test)
    laplace_newton_batch_x_only_percellWmu,
    laplace_newton_batch_x_only_offset_percellWmu,
    laplace_newton_batch_percellWmu,
    nbln_grad_x_only_norm_batch_percellWmu,
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
    latent_init = jnp.log(counts + 1.0)
    return mu, W, d, r, counts, latent_init


# ---------------------------------------------------------------------------
# Consistency: broadcast W AND mu -> identical to shared kernels
# ---------------------------------------------------------------------------


def test_x_only_percellWmu_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(0)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    mu_pc = jnp.broadcast_to(mu, (N, G))

    x_shared, _, _ = laplace_newton_batch_x_only(
        latent_init, counts, mu, W, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    x_pc, _, _ = laplace_newton_batch_x_only_percellWmu(
        latent_init, counts, mu_pc, W_pc, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    npt.assert_allclose(np.asarray(x_pc), np.asarray(x_shared), atol=1e-5)


def test_x_only_offset_percellWmu_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(1)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    mu_pc = jnp.broadcast_to(mu, (N, G))
    eta_off = jnp.asarray(
        0.1 * np.random.default_rng(1).normal(size=N), dtype=jnp.float32
    )

    x_shared, _, _ = laplace_newton_batch_x_only_offset(
        latent_init, counts, mu, W, d, r, eta_off, N_NEWTON, DAMPING, MAX_STEP
    )
    x_pc, _, _ = laplace_newton_batch_x_only_offset_percellWmu(
        latent_init, counts, mu_pc, W_pc, d, r, eta_off,
        N_NEWTON, DAMPING, MAX_STEP,
    )
    npt.assert_allclose(np.asarray(x_pc), np.asarray(x_shared), atol=1e-5)


def test_joint_percellWmu_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(2)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    mu_pc = jnp.broadcast_to(mu, (N, G))
    rng = np.random.default_rng(2)
    eta_init = jnp.zeros((N,), dtype=jnp.float32)
    eta_anchor = jnp.asarray(0.1 * rng.normal(size=N), dtype=jnp.float32)
    sigma_eta = jnp.full((N,), 0.3, dtype=jnp.float32)

    x_s, eta_s, _, _ = laplace_newton_batch(
        latent_init, eta_init, counts, mu, W, d, r, eta_anchor, sigma_eta,
        N_NEWTON, DAMPING, MAX_STEP,
    )
    x_pc, eta_pc, _, _ = laplace_newton_batch_percellWmu(
        latent_init, eta_init, counts, mu_pc, W_pc, d, r, eta_anchor,
        sigma_eta, N_NEWTON, DAMPING, MAX_STEP,
    )
    npt.assert_allclose(np.asarray(x_pc), np.asarray(x_s), atol=1e-5)
    npt.assert_allclose(np.asarray(eta_pc), np.asarray(eta_s), atol=1e-5)


def test_gradnorm_percellWmu_matches_shared_when_broadcast():
    mu, W, d, r, counts, latent_init = _problem(3)
    W_pc = jnp.broadcast_to(W, (N, G, K))
    mu_pc = jnp.broadcast_to(mu, (N, G))
    x, _, _ = laplace_newton_batch_x_only(
        latent_init, counts, mu, W, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    gn_s = nbln_grad_x_only_norm_batch(x, counts, mu, W, d, r)
    gn_pc = nbln_grad_x_only_norm_batch_percellWmu(x, counts, mu_pc, W_pc, d, r)
    npt.assert_allclose(np.asarray(gn_pc), np.asarray(gn_s), atol=1e-5)


# ---------------------------------------------------------------------------
# Per-donor correctness: distinct donor (W, mu) gathered per cell
# ---------------------------------------------------------------------------


def test_x_only_percellWmu_per_donor_gather():
    """Two donors with distinct mu (and W): per-cell == per-donor shared runs."""
    mu0, W0, d, r, counts, latent_init = _problem(4)
    rng = np.random.default_rng(99)
    W1 = jnp.asarray(0.3 * rng.normal(size=(G, K)), dtype=jnp.float32)
    mu1 = jnp.asarray(mu0 + rng.normal(size=G) * 0.7, dtype=jnp.float32)

    ds = jnp.arange(N) % 2  # alternating donor assignment
    W_pc = jnp.stack([W0, W1], axis=0)[ds]       # (N, G, K)
    mu_pc = jnp.stack([mu0, mu1], axis=0)[ds]    # (N, G)

    x_pc, _, _ = laplace_newton_batch_x_only_percellWmu(
        latent_init, counts, mu_pc, W_pc, d, r, N_NEWTON, DAMPING, MAX_STEP
    )

    # Reference: shared kernel on each donor's cells with that donor's (W, mu).
    x_ref = np.array(x_pc)
    for donor, Wd, mud in [(0, W0, mu0), (1, W1, mu1)]:
        mask = np.asarray(ds) == donor
        x_d, _, _ = laplace_newton_batch_x_only(
            latent_init[mask], counts[mask], mud, Wd, d, r,
            N_NEWTON, DAMPING, MAX_STEP,
        )
        x_ref[mask] = np.asarray(x_d)

    npt.assert_allclose(np.asarray(x_pc), x_ref, atol=1e-5)


def test_percellWmu_mu_gather_changes_map():
    """Per-cell mu actually moves the MAP vs a shared-mu (percellW) run."""
    mu0, W, d, r, counts, latent_init = _problem(5)
    rng = np.random.default_rng(7)
    # Donor 1 has a markedly shifted mean; donor 0 keeps mu0.
    mu1 = jnp.asarray(mu0 + 1.5, dtype=jnp.float32)
    ds = jnp.arange(N) % 2
    W_pc = jnp.broadcast_to(W, (N, G, K))
    mu_pc = jnp.stack([mu0, mu1], axis=0)[ds]

    # Per-cell-mu run (honors the per-donor shift).
    x_mu, _, _ = laplace_newton_batch_x_only_percellWmu(
        latent_init, counts, mu_pc, W_pc, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    # Shared-mu run (everyone gets mu0) — the percellW twin.
    x_shared, _, _ = laplace_newton_batch_x_only_percellW(
        latent_init, counts, mu0, W_pc, d, r, N_NEWTON, DAMPING, MAX_STEP
    )
    # Donor-0 cells unchanged (same mu0); donor-1 cells differ.
    d0 = np.asarray(ds) == 0
    d1 = np.asarray(ds) == 1
    npt.assert_allclose(
        np.asarray(x_mu)[d0], np.asarray(x_shared)[d0], atol=1e-5
    )
    assert not np.allclose(
        np.asarray(x_mu)[d1], np.asarray(x_shared)[d1], atol=1e-2
    )

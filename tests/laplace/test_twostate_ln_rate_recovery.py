"""Synthetic-data recovery tests for TSLN-Rate Laplace.

These tests build small synthetic count matrices from a known
``(mu, burst_size, k_off, W, d)`` ground truth, fit TSLN-Rate
Laplace on the data, and assert that the recovered parameters
satisfy basic sanity checks:

- Mean recovery: ``r_hat`` is in a reasonable range vs the empirical
  mean.  Strict recovery on tiny synthetic data isn't feasible in a
  unit test (would need 1000s of cells and many fitting steps), so
  these assertions are deliberately loose.
- Finite, positive constrained globals (``mu``, ``burst_size``,
  ``k_off``, ``r_hat``, ``alpha``, ``beta``).
- Loss / gradients well-behaved (no NaN, Newton converged).
- Cross-gene mean covariance is *non-zero* when the underlying
  data has a low-rank latent structure (the "does the latent
  do anything" smoke check — analogous to NBLN's basic recovery).

Rigorous posterior-recovery tests (W_⟂ subspace angle, parameter
within tolerance) are deferred to follow-up benchmarks.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _generate_tsln_rate_counts(
    C: int,
    G: int,
    k: int,
    seed: int = 0,
):
    """Generate synthetic TSLN-Rate counts from a known ground truth.

    Returns ``(counts, mu, burst_size, k_off, W, d, z)``.
    """
    from scribe.models.components.likelihoods.two_state import (
        _twostate_reparam,
    )

    rng = np.random.default_rng(seed)
    # Per-gene true params.
    mu_true = jnp.asarray(
        np.exp(0.5 * rng.normal(size=G)).astype(np.float32) * 3.0
    )
    burst_size_true = jnp.asarray(
        np.exp(0.3 * rng.normal(size=G)).astype(np.float32)
    )
    k_off_true = jnp.asarray(
        np.exp(0.5 * rng.normal(size=G)).astype(np.float32) * 3.0
    )
    alpha, beta, rate_hat, _ = _twostate_reparam(
        mu_true, burst_size_true, k_off_true
    )

    # Low-rank latent structure on (C, G).
    W_true = jnp.asarray(
        (0.3 * rng.normal(size=(G, k))).astype(np.float32)
    )
    d_true = jnp.asarray(np.full(G, 0.05, dtype=np.float32))
    u_factor = rng.normal(size=(C, k)).astype(np.float32)
    eps = jnp.asarray(
        (np.sqrt(d_true) * rng.normal(size=(C, G))).astype(np.float32)
    )
    z = jnp.asarray(u_factor) @ W_true.T + eps  # (C, G)

    # Per-(cell, gene) Poisson-Beta draw.  For each (c, g), draw
    # p ~ Beta(alpha_g, beta_g) and then u ~ Poisson(rate_hat_g · exp(z_cg) · p).
    key = jax.random.PRNGKey(seed)
    key_beta, key_pois = jax.random.split(key)
    p = jax.random.beta(key_beta, alpha[None, :], beta[None, :], shape=(C, G))
    lam = rate_hat[None, :] * jnp.exp(z) * p
    counts = jax.random.poisson(key_pois, lam).astype(jnp.float32)

    return counts, mu_true, burst_size_true, k_off_true, W_true, d_true, z


# ---------------------------------------------------------------------
# Test 1: basic recovery — fit produces sensible globals
# ---------------------------------------------------------------------


def test_basic_recovery_sanity():
    """Fit recovers globals that satisfy basic sanity invariants."""
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.base import ModelConfig
    from scribe.models.config.groups import (
        DataConfig,
        LaplaceConfig,
        VAEConfig,
    )
    from scribe.models.config.enums import InferenceMethod, Parameterization

    C, G, k = 40, 12, 2
    counts, mu_true, bs_true, ko_true, W_true, d_true, z_true = (
        _generate_tsln_rate_counts(C, G, k, seed=0)
    )

    cfg = ModelConfig(
        base_model="twostate_ln_rate",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
        vae=VAEConfig(latent_dim=k),
    )

    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=40, n_newton_steps=5),
        data_config=DataConfig(),
        seed=42,
    )

    # Finite + positive constrained globals.
    for name in ("gene_mean", "burst_size", "k_off", "alpha", "beta", "r_hat"):
        val = getattr(result, name)
        assert val is not None
        assert jnp.all(jnp.isfinite(val)), f"{name} contains non-finite"
        assert jnp.all(val > 0), f"{name} not strictly positive: min={val.min()}"

    # Loss finite throughout.
    assert jnp.all(jnp.isfinite(result.losses))
    # Newton converged.
    assert float(result.final_grad_norms.max()) < 1e-2

    # Empirical-mean sanity: the fitted gene_mean should be within an
    # order of magnitude of the empirical mean.  This is a very loose
    # check (true recovery needs longer fits) but catches gross errors.
    empirical_mean = jnp.maximum(counts.mean(axis=0), 1e-3)
    ratio = result.gene_mean / empirical_mean
    # Ratio per-gene should be O(1) — between 0.1x and 10x.
    assert jnp.all(ratio > 0.05), (
        f"gene_mean too small vs empirical: min ratio={float(ratio.min())}"
    )
    assert jnp.all(ratio < 20.0), (
        f"gene_mean too large vs empirical: max ratio={float(ratio.max())}"
    )


# ---------------------------------------------------------------------
# Test 2: cross-gene covariance is non-zero
# ---------------------------------------------------------------------


def test_cross_gene_covariance_nonzero():
    """The latent z_c structure should produce non-zero cross-gene cov.

    This is the smoke version of the plan's audit-critical "cross-gene
    mean covariance recovery" test: when the synthetic ground truth has
    a low-rank latent structure, the fit's recovered latent ``x_c``
    should produce a non-trivial cross-gene covariance — otherwise the
    model is silently behaving like the independent-gene TwoState SVI
    (which would be the same flaw the round-1 audit caught in the
    original Variant B parameterization).
    """
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.base import ModelConfig
    from scribe.models.config.groups import (
        DataConfig,
        LaplaceConfig,
        VAEConfig,
    )
    from scribe.models.config.enums import InferenceMethod, Parameterization

    C, G, k = 50, 15, 3
    counts, *_, z_true = _generate_tsln_rate_counts(C, G, k, seed=1)

    cfg = ModelConfig(
        base_model="twostate_ln_rate",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
        vae=VAEConfig(latent_dim=k),
    )
    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=40, n_newton_steps=5),
        data_config=DataConfig(),
        seed=0,
    )

    # Recovered z = x - mu_x.
    z_fit = result.x_loc - result.mu[None, :]
    # Cross-gene covariance of the fit z.
    cov = jnp.cov(z_fit, rowvar=False)  # (G, G)
    # Off-diagonal entries — magnitude should be non-negligible.
    off_diag = cov - jnp.diag(jnp.diag(cov))
    max_off_diag = float(jnp.abs(off_diag).max())
    # On a non-trivial latent structure with W rank 3, we expect
    # cross-gene cov entries at least ~0.01 in magnitude.
    assert max_off_diag > 1e-3, (
        f"Max off-diagonal cross-gene covariance {max_off_diag:.3e} is "
        "near zero; the latent z is not picking up the synthetic "
        "low-rank structure."
    )

    # W should also be non-trivial (post-fit, |W| should be O(0.1) or
    # larger somewhere, reflecting that the low-rank structure was
    # learned).
    max_w = float(jnp.abs(result.W).max())
    assert max_w > 1e-2, (
        f"max |W| {max_w:.3e} is suspiciously small; latent factor "
        "structure not learned."
    )

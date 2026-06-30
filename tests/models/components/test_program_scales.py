"""
Tests for the per-donor program-activity scale primitive
(:mod:`scribe.models.components.program_scales`).

These cover the *shared* Rung-1 hierarchical-correlation primitive used by both
the SVI/VAE sanity-check (Phase A) and the Laplace production path (Phase B):

- ``effective_loadings``: the ``W_eff,d = W . diag(s_d)`` collapse that keeps
  ``Sigma_d`` low-rank-plus-diagonal (the algebraic identity Phase B relies on).
- ``sample_program_scales``: shapes of every emitted site, the sum-to-zero
  *scale gauge* (``mean_d log s == 0``), strict positivity, and the ``D == 1``
  degenerate case (``s == 1``).
- ``guide_program_scales``: the mean-field guide registers exactly the latent
  sites the model sampler emits (so the mean-field ELBO pairs).

Note on what is intentionally NOT asserted: absolute uniqueness of ``(W, s)``.
Sum-to-zero removes only the *column scale* gauge; column sign/permutation
always remain and the rotation gauge is only generically broken. We therefore
test scale-gauge removal (``mean_d log s == 0``), not uniqueness.
"""

import pytest
import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import numpyro

from scribe.models.components.program_scales import (
    sample_program_scales,
    guide_program_scales,
    effective_loadings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def dims():
    # (G genes, K programs, D donors) -- small but non-degenerate.
    return {"G": 12, "K": 4, "D": 5}


# ---------------------------------------------------------------------------
# effective_loadings: the W_eff collapse
# ---------------------------------------------------------------------------


def test_effective_loadings_single_donor_collapse(rng, dims):
    """1-D ``s`` yields (G, K) loadings with W_eff W_eff^T == W diag(s^2) W^T."""
    G, K = dims["G"], dims["K"]
    kW, ks = jax.random.split(rng)
    W = jax.random.normal(kW, (G, K))
    s = jnp.exp(0.3 * jax.random.normal(ks, (K,)))  # positive scales

    W_eff = effective_loadings(W, s)
    assert W_eff.shape == (G, K)

    # The defining identity: the low-rank part of Sigma is unchanged when the
    # column scaling is folded into the loadings.
    lhs = W_eff @ W_eff.T
    rhs = W @ jnp.diag(s**2) @ W.T
    npt.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-5, atol=1e-5)


def test_effective_loadings_multi_donor_collapse(rng, dims):
    """2-D ``s`` yields (D, G, K) loadings; each slice reproduces Sigma_d."""
    G, K, D = dims["G"], dims["K"], dims["D"]
    kW, ks = jax.random.split(rng)
    W = jax.random.normal(kW, (G, K))
    s = jnp.exp(0.3 * jax.random.normal(ks, (D, K)))

    W_eff = effective_loadings(W, s)
    assert W_eff.shape == (D, G, K)

    # Every donor slice must reconstruct that donor's low-rank covariance.
    for d in range(D):
        lhs = W_eff[d] @ W_eff[d].T
        rhs = W @ jnp.diag(s[d] ** 2) @ W.T
        npt.assert_allclose(
            np.asarray(lhs), np.asarray(rhs), rtol=1e-5, atol=1e-5
        )


def test_effective_loadings_scale_gauge_invariance(rng, dims):
    """Sigma_d is invariant under W[:,k]->a_k W[:,k], s->s/a_k (the scale gauge)."""
    G, K, D = dims["G"], dims["K"], dims["D"]
    kW, ks, ka = jax.random.split(rng, 3)
    W = jax.random.normal(kW, (G, K))
    s = jnp.exp(0.3 * jax.random.normal(ks, (D, K)))
    a = jnp.exp(0.5 * jax.random.normal(ka, (K,)))  # arbitrary positive per-column

    W_g = W * a[None, :]  # rescale columns
    s_g = s / a[None, :]  # compensate in the scales

    for d in range(D):
        base = effective_loadings(W, s)[d]
        gauged = effective_loadings(W_g, s_g)[d]
        npt.assert_allclose(
            np.asarray(base @ base.T),
            np.asarray(gauged @ gauged.T),
            rtol=1e-5,
            atol=1e-5,
        )


def test_effective_loadings_validates_shapes(dims):
    """Mismatched K and bad ndim raise informative ValueErrors."""
    G, K = dims["G"], dims["K"]
    W = jnp.ones((G, K))
    with pytest.raises(ValueError, match="W has K"):
        effective_loadings(W, jnp.ones((K + 1,)))
    with pytest.raises(ValueError, match="W has K"):
        effective_loadings(W, jnp.ones((3, K + 1)))
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        effective_loadings(W, jnp.ones((2, 3, K)))


# ---------------------------------------------------------------------------
# sample_program_scales: sites, gauge, positivity, degenerate D == 1
# ---------------------------------------------------------------------------


def _trace_scales(rng, n_datasets, latent_dim, prefix="program_scale"):
    """Seed + trace the model sampler and return the site->value trace."""
    seeded = numpyro.handlers.seed(sample_program_scales, rng)
    tr = numpyro.handlers.trace(seeded).get_trace(n_datasets, latent_dim)
    return tr


def test_sample_program_scales_site_shapes(rng, dims):
    """All emitted sites exist with the documented shapes."""
    K, D = dims["K"], dims["D"]
    tr = _trace_scales(rng, D, K)

    assert tr["program_scale_tau_raw"]["value"].shape == ()
    assert tr["program_scale_raw"]["value"].shape == (D, K)
    assert tr["program_scale_log"]["value"].shape == (D, K)
    assert tr["program_scale"]["value"].shape == (D, K)


def test_sample_program_scales_sum_to_zero_gauge(rng, dims):
    """The scale gauge is fixed: mean over donors of log s is ~0 per program."""
    K, D = dims["K"], dims["D"]
    tr = _trace_scales(rng, D, K)
    log_s = np.asarray(tr["program_scale_log"]["value"])
    # mean over the donor axis (axis 0) must vanish for every program.
    npt.assert_allclose(log_s.mean(axis=0), np.zeros(K), atol=1e-6)


def test_sample_program_scales_positive(rng, dims):
    """Constrained scales are strictly positive."""
    K, D = dims["K"], dims["D"]
    tr = _trace_scales(rng, D, K)
    s = np.asarray(tr["program_scale"]["value"])
    assert np.all(s > 0.0)


def test_sample_program_scales_degenerate_single_donor(rng):
    """With D == 1 the centering forces s == 1 (no hierarchy)."""
    tr = _trace_scales(rng, 1, 6)
    s = np.asarray(tr["program_scale"]["value"])
    npt.assert_allclose(s, np.ones((1, 6)), atol=1e-6)


def test_sample_program_scales_custom_prefix(rng, dims):
    """A custom site_prefix renames every site consistently."""
    K, D = dims["K"], dims["D"]
    seeded = numpyro.handlers.seed(sample_program_scales, rng)
    tr = numpyro.handlers.trace(seeded).get_trace(
        D, K, site_prefix="act"
    )
    assert "act" in tr and "act_raw" in tr and "act_tau_raw" in tr
    assert "program_scale" not in tr


# ---------------------------------------------------------------------------
# guide_program_scales: latent-site pairing with the model
# ---------------------------------------------------------------------------


def test_guide_pairs_model_latent_sites(rng, dims):
    """The guide registers exactly the latent sites the model emits.

    The mean-field ELBO requires every *latent* (non-deterministic,
    non-observed) model site to have a matching guide site. The deterministic
    sites (``program_scale`` and ``program_scale_log``) are functions of the
    latents and must NOT appear in the guide.
    """
    K, D = dims["K"], dims["D"]

    model_tr = _trace_scales(rng, D, K)
    model_latents = {
        name
        for name, site in model_tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }

    seeded_guide = numpyro.handlers.seed(guide_program_scales, rng)
    guide_tr = numpyro.handlers.trace(seeded_guide).get_trace(D, K)
    guide_latents = {
        name
        for name, site in guide_tr.items()
        if site["type"] == "sample"
    }

    assert model_latents == guide_latents == {
        "program_scale_tau_raw",
        "program_scale_raw",
    }
    # Guide latent shapes must match the model's.
    assert guide_tr["program_scale_raw"]["value"].shape == (D, K)
    assert guide_tr["program_scale_tau_raw"]["value"].shape == ()

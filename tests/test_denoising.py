"""
Tests for Bayesian denoising of observed cell profiles.

These tests verify the ``denoise_counts`` family of functions and methods
that compute the posterior of true (pre-capture, pre-dropout) transcript
counts from observed UMI counts, as derived in ``paper/_denoising.qmd``.

Tests cover:
- Core ``denoise_counts`` utility (shapes, non-negativity, identity for NBDM)
- ZINB zero handling (gate weight computation, mixture posterior)
- Limiting behaviour (perfect capture, weak prior, strong signal)
- Mixture model support (marginalisation and component assignment)
- Posterior variance computation
- SVI integration (MAP and full-posterior methods)
- MCMC integration (full-posterior method)
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.sampling import denoise_counts, _compute_gate_weight
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig
from scribe.inference import run_scribe
from scribe.inference.preset_builder import build_config_from_preset


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def dataset():
    """Small dataset for denoising tests."""
    np.random.seed(456)
    n_cells, n_genes = 10, 5
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    return jnp.array(counts), n_cells, n_genes


@pytest.fixture(scope="module")
def rng():
    """Consistent PRNG key."""
    return random.PRNGKey(77)


# Caches for fitted models (avoid re-fitting per test)
_svi_cache = {}
_mcmc_cache = {}


def _fit_svi(model, counts, parameterization="standard"):
    """Fit an SVI model (cached)."""
    key = (model, parameterization)
    if key not in _svi_cache:
        priors = _get_priors(model, parameterization)
        model_config = build_config_from_preset(
            model=model,
            parameterization=parameterization,
            inference_method="svi",
            unconstrained=False,
            guide_rank=None,
            priors=priors,
        )
        svi_config = SVIConfig(n_steps=5, batch_size=5)
        inference_config = InferenceConfig.from_svi(svi_config)
        _svi_cache[key] = run_scribe(
            counts=counts,
            model_config=model_config,
            inference_config=inference_config,
            seed=42,
        )
    return _svi_cache[key]


def _fit_mcmc(model, counts, parameterization="standard"):
    """Fit an MCMC model (cached)."""
    key = (model, parameterization)
    if key not in _mcmc_cache:
        priors = _get_priors(model, parameterization)
        model_config = build_config_from_preset(
            model=model,
            parameterization=parameterization,
            inference_method="mcmc",
            unconstrained=False,
            priors=priors,
        )
        mcmc_config = MCMCConfig(n_warmup=2, n_samples=3, n_chains=1)
        inference_config = InferenceConfig.from_mcmc(mcmc_config)
        _mcmc_cache[key] = run_scribe(
            counts=counts,
            model_config=model_config,
            inference_config=inference_config,
            seed=42,
        )
    return _mcmc_cache[key]


def _get_priors(model, parameterization):
    """Return appropriate priors for a given model and parameterization."""
    base_priors = {
        "standard": {"r": (2, 0.1), "p": (1, 1)},
        "linked": {"p": (1, 1), "mu": (1, 1)},
        "odds_ratio": {"phi": (3, 2), "mu": (1, 1)},
    }
    priors = dict(base_priors[parameterization])
    if "vcp" in model:
        if parameterization == "odds_ratio":
            priors["phi_capture"] = (3, 2)
        else:
            priors["p_capture"] = (1, 1)
    if "zinb" in model:
        priors["gate"] = (1, 1)
    return priors


# ==============================================================================
# Tests for the core denoise_counts utility
# ==============================================================================


class TestDenoiseCountsCore:
    """Tests for the shared ``denoise_counts`` function in ``sampling.py``."""

    def test_single_params_shape(self, rng):
        """Single-param-set denoising returns (n_cells, n_genes)."""
        n_cells, n_genes = 8, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.float32(0.4)
        nu = jnp.ones(n_cells) * 0.5

        result = denoise_counts(counts, r, p, p_capture=nu)
        assert result.shape == (n_cells, n_genes)

    def test_multi_sample_shape(self, rng):
        """Multi-sample denoising returns (n_samples, n_cells, n_genes)."""
        n_samples, n_cells, n_genes = 3, 6, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 2
        r = jnp.ones((n_samples, n_genes)) * 5.0
        p = jnp.ones(n_samples) * 0.3
        nu = jnp.ones((n_samples, n_cells)) * 0.6

        result = denoise_counts(counts, r, p, p_capture=nu)
        assert result.shape == (n_samples, n_cells, n_genes)

    def test_non_negative(self, rng):
        """Denoised counts must be non-negative for all methods."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])

        for method in ("mean", "mode", "sample"):
            result = denoise_counts(
                counts, r, p, p_capture=nu, method=method, rng_key=rng,
            )
            assert jnp.all(result >= 0), f"method={method}"

    def test_nbdm_identity(self):
        """Without VCP or gate, denoising is identity (m = u)."""
        counts = jnp.array([[5, 0, 3], [1, 2, 7]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)

        result = denoise_counts(counts, r, p)
        np.testing.assert_allclose(result, counts, atol=1e-5)

    def test_nbdm_identity_mode(self):
        """Mode method also gives identity for NBDM."""
        counts = jnp.array([[5, 0, 3]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)

        result = denoise_counts(counts, r, p, method="mode")
        np.testing.assert_allclose(result, counts, atol=1e-5)

    def test_cell_batching_matches_full(self, rng):
        """Cell batching produces the same result as processing all cells."""
        n_cells, n_genes = 15, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.float32(0.3)
        nu = jnp.ones(n_cells) * 0.5

        full = denoise_counts(counts, r, p, p_capture=nu)
        batched = denoise_counts(
            counts, r, p, p_capture=nu, cell_batch_size=4,
        )
        np.testing.assert_allclose(full, batched, atol=1e-5)

    def test_denoised_ge_observed_mean(self):
        """Posterior mean should be >= observed counts (we add uncaptured)."""
        counts = jnp.array([[10, 5, 8], [3, 0, 12]], dtype=jnp.float32)
        r = jnp.array([5.0, 3.0, 7.0])
        p = jnp.float32(0.3)
        nu = jnp.array([0.5, 0.5])

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")
        assert jnp.all(result >= counts - 1e-5)

    def test_invalid_method_raises(self):
        """Invalid method string should raise ValueError."""
        counts = jnp.ones((2, 3), dtype=jnp.float32)
        r = jnp.ones(3)
        p = jnp.float32(0.5)
        with pytest.raises(ValueError, match="method must be"):
            denoise_counts(counts, r, p, method="invalid")

    def test_sample_method_shape(self, rng):
        """Sample method produces correct shape."""
        n_cells, n_genes = 5, 3
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 2
        r = jnp.ones(n_genes) * 4.0
        p = jnp.float32(0.3)
        nu = jnp.ones(n_cells) * 0.6

        result = denoise_counts(
            counts, r, p, p_capture=nu, method="sample", rng_key=rng,
        )
        assert result.shape == (n_cells, n_genes)


# ==============================================================================
# Tests for ZINB zero handling
# ==============================================================================


class TestDenoiseZINB:
    """Tests for zero-inflation gate correction in denoising."""

    def test_gate_weight_basic(self):
        """Gate weight w should be in [0, 1]."""
        gate = jnp.array([0.3, 0.5, 0.8])
        r = jnp.array([2.0, 3.0, 1.0])
        p = jnp.float32(0.4)
        one_minus_pp = jnp.float32(1.0)  # no VCP

        w = _compute_gate_weight(gate, r, p, one_minus_pp)
        assert jnp.all(w >= 0.0)
        assert jnp.all(w <= 1.0)

    def test_gate_weight_zero_gate(self):
        """When gate=0, weight should be 0 (no zero-inflation)."""
        gate = jnp.array([0.0, 0.0])
        r = jnp.array([2.0, 3.0])
        p = jnp.float32(0.4)
        one_minus_pp = jnp.float32(1.0)

        w = _compute_gate_weight(gate, r, p, one_minus_pp)
        np.testing.assert_allclose(w, 0.0, atol=1e-7)

    def test_gate_weight_one_gate(self):
        """When gate=1, weight should be 1 (always dropout)."""
        gate = jnp.array([1.0, 1.0])
        r = jnp.array([2.0, 3.0])
        p = jnp.float32(0.4)
        one_minus_pp = jnp.float32(1.0)

        w = _compute_gate_weight(gate, r, p, one_minus_pp)
        np.testing.assert_allclose(w, 1.0, atol=1e-7)

    def test_nonzero_positions_unaffected_by_gate(self):
        """Gate correction only affects zero positions."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])
        gate = jnp.array([0.3, 0.3, 0.3])

        with_gate = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mean",
        )
        without_gate = denoise_counts(
            counts, r, p, p_capture=nu, method="mean",
        )

        # Non-zero positions should be identical
        nonzero_mask = counts > 0
        np.testing.assert_allclose(
            with_gate[nonzero_mask], without_gate[nonzero_mask], atol=1e-5,
        )

    def test_zinb_zero_mean_positive(self):
        """Denoised mean at zero positions should be positive when gate > 0."""
        counts = jnp.array([[0, 0, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 3.0, 1.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5])
        gate = jnp.array([0.5, 0.5, 0.5])

        result = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mean",
        )
        assert jnp.all(result > 0)

    def test_zinb_zero_larger_than_nb_zero(self):
        """ZINB denoised zero mean should be >= NB denoised zero mean.

        The gate pathway contributes the prior mean, which is always
        larger than the NB-only mean at u=0.
        """
        counts = jnp.array([[0, 0, 0]], dtype=jnp.float32)
        r = jnp.array([5.0, 3.0, 7.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5])
        gate = jnp.array([0.3, 0.3, 0.3])

        nb_only = denoise_counts(
            counts, r, p, p_capture=nu, method="mean",
        )
        zinb = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mean",
        )
        assert jnp.all(zinb >= nb_only - 1e-5)

    def test_zinb_no_vcp_zero_identity_nonzero(self):
        """ZINB without VCP: non-zero positions should be identity."""
        counts = jnp.array([[5, 0, 3], [0, 2, 7]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        gate = jnp.array([0.3, 0.3, 0.3])

        result = denoise_counts(counts, r, p, gate=gate, method="mean")

        nonzero_mask = counts > 0
        np.testing.assert_allclose(
            result[nonzero_mask], counts[nonzero_mask], atol=1e-5,
        )


# ==============================================================================
# Tests for limiting behaviour
# ==============================================================================


class TestDenoiseLimits:
    """Tests for expected behaviour in limiting cases."""

    def test_perfect_capture_identity(self):
        """When nu_c = 1, denoising should be identity (no correction)."""
        counts = jnp.array([[5, 3, 8], [1, 0, 4]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.ones(2)  # perfect capture

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")
        np.testing.assert_allclose(result, counts, atol=1e-5)

    def test_weak_prior_rescaling(self):
        """When r → 0, denoising mean ≈ u / (1 - p*(1-nu))."""
        counts = jnp.array([[10.0, 20.0]], dtype=jnp.float32)
        r = jnp.array([1e-6, 1e-6])
        p = jnp.float32(0.3)
        nu = jnp.array([0.5])

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")

        probs_post = p * (1.0 - nu[:, None])
        expected = counts / (1.0 - probs_post)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_strong_signal_rescaling(self):
        """When u >> r, denoising mean ≈ u / (1 - p*(1-nu))."""
        counts = jnp.array([[1000.0, 2000.0]], dtype=jnp.float32)
        r = jnp.array([1.0, 1.0])
        p = jnp.float32(0.3)
        nu = jnp.array([0.5])

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")

        probs_post = p * (1.0 - nu[:, None])
        naive_rescale = counts / (1.0 - probs_post)
        # Should be close but not exact (prior contributes a small amount)
        np.testing.assert_allclose(result, naive_rescale, rtol=0.01)

    def test_zero_obs_positive_mean(self):
        """When u=0 and nu < 1, denoised mean should still be positive."""
        counts = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
        r = jnp.array([5.0, 3.0])
        p = jnp.float32(0.3)
        nu = jnp.array([0.5])

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")
        assert jnp.all(result > 0)

    def test_monotone_in_capture(self):
        """Better capture (higher nu) → closer to identity."""
        counts = jnp.array([[10.0]], dtype=jnp.float32)
        r = jnp.array([5.0])
        p = jnp.float32(0.3)

        result_low = denoise_counts(
            counts, r, p, p_capture=jnp.array([0.3]), method="mean",
        )
        result_high = denoise_counts(
            counts, r, p, p_capture=jnp.array([0.8]), method="mean",
        )

        # Higher capture → less correction → closer to observed
        assert float(result_high[0, 0]) < float(result_low[0, 0])
        assert float(result_high[0, 0]) > float(counts[0, 0])


# ==============================================================================
# Tests for mixture model denoising
# ==============================================================================


class TestDenoiseMixture:
    """Tests for mixture model support in denoising."""

    def test_marginal_shape(self):
        """Marginalised denoising returns (n_cells, n_genes)."""
        n_cells, n_genes, n_components = 6, 4, 3
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.float32(0.4)
        nu = jnp.ones(n_cells) * 0.5
        mw = jnp.ones(n_components) / n_components

        result = denoise_counts(
            counts, r, p, p_capture=nu, mixing_weights=mw,
        )
        assert result.shape == (n_cells, n_genes)

    def test_component_assignment_shape(self):
        """Component-assigned denoising returns (n_cells, n_genes)."""
        n_cells, n_genes, n_components = 6, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.float32(0.4)
        nu = jnp.ones(n_cells) * 0.5
        mw = jnp.array([0.6, 0.4])
        comp = jnp.array([0, 0, 0, 1, 1, 1])

        result = denoise_counts(
            counts, r, p, p_capture=nu,
            mixing_weights=mw, component_assignment=comp,
        )
        assert result.shape == (n_cells, n_genes)

    def test_component_specific_p(self):
        """Mixture with per-component p produces correct shape."""
        n_cells, n_genes, n_components = 6, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.array([0.3, 0.6])
        nu = jnp.ones(n_cells) * 0.5
        mw = jnp.array([0.5, 0.5])

        result = denoise_counts(
            counts, r, p, p_capture=nu, mixing_weights=mw,
        )
        assert result.shape == (n_cells, n_genes)

    def test_single_component_equals_standard(self):
        """Mixture with one component should match standard denoising."""
        n_cells, n_genes = 5, 3
        counts = jnp.array(
            [[5, 0, 3], [0, 2, 0], [1, 1, 1], [3, 3, 3], [7, 0, 2]],
            dtype=jnp.float32,
        )
        r_flat = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.linspace(0.3, 0.8, n_cells)

        standard = denoise_counts(counts, r_flat, p, p_capture=nu)

        r_mix = r_flat[None, :]  # (1, n_genes)
        mw = jnp.ones(1)
        mixture = denoise_counts(
            counts, r_mix, p, p_capture=nu, mixing_weights=mw,
        )

        np.testing.assert_allclose(mixture, standard, atol=1e-5)

    def test_sample_method_mixture(self, rng):
        """Sample method with mixture model returns correct shape."""
        n_cells, n_genes, n_components = 6, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.float32(0.4)
        nu = jnp.ones(n_cells) * 0.5
        mw = jnp.array([0.5, 0.5])

        result = denoise_counts(
            counts, r, p, p_capture=nu,
            mixing_weights=mw, method="sample", rng_key=rng,
        )
        assert result.shape == (n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_sample_method_mixture_component_p_with_batching(self, rng):
        """Sampled mixture denoising handles per-component p with batching."""
        n_cells, n_genes, n_components = 9, 5, 3
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 2
        r = jnp.ones((n_components, n_genes), dtype=jnp.float32) * 4.0
        p = jnp.array([0.2, 0.4, 0.6], dtype=jnp.float32)
        nu = jnp.linspace(0.2, 0.8, n_cells, dtype=jnp.float32)
        mw = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)

        result = denoise_counts(
            counts,
            r,
            p,
            p_capture=nu,
            mixing_weights=mw,
            method="sample",
            rng_key=rng,
            cell_batch_size=4,
        )
        assert result.shape == (n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_multi_sample_mixture_component_p_shape(self, rng):
        """Multi-sample mixture denoising returns expected 3D shape."""
        n_samples, n_cells, n_genes, n_components = 3, 8, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(
            (n_samples, n_components, n_genes), dtype=jnp.float32
        ) * 5.0
        p = jnp.array(
            [[0.25, 0.45], [0.30, 0.50], [0.35, 0.55]],
            dtype=jnp.float32,
        )
        nu = jnp.ones((n_samples, n_cells), dtype=jnp.float32) * 0.5
        mw = jnp.array(
            [[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]],
            dtype=jnp.float32,
        )

        result = denoise_counts(
            counts,
            r,
            p,
            p_capture=nu,
            mixing_weights=mw,
            method="sample",
            rng_key=rng,
            cell_batch_size=3,
        )
        assert result.shape == (n_samples, n_cells, n_genes)
        assert jnp.all(result >= 0)


# ==============================================================================
# Tests for variance computation
# ==============================================================================


class TestDenoiseVariance:
    """Tests for the ``return_variance=True`` option."""

    def test_variance_dict_keys(self):
        """Return dict should have 'denoised_counts' and 'variance'."""
        counts = jnp.ones((3, 2), dtype=jnp.float32) * 5
        r = jnp.ones(2) * 3.0
        p = jnp.float32(0.4)
        nu = jnp.ones(3) * 0.5

        result = denoise_counts(
            counts, r, p, p_capture=nu, return_variance=True,
        )
        assert isinstance(result, dict)
        assert "denoised_counts" in result
        assert "variance" in result

    def test_variance_shape(self):
        """Variance should have the same shape as denoised counts."""
        n_cells, n_genes = 5, 3
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 5
        r = jnp.ones(n_genes) * 3.0
        p = jnp.float32(0.4)
        nu = jnp.ones(n_cells) * 0.5

        result = denoise_counts(
            counts, r, p, p_capture=nu, return_variance=True,
        )
        assert result["denoised_counts"].shape == (n_cells, n_genes)
        assert result["variance"].shape == (n_cells, n_genes)

    def test_variance_non_negative(self):
        """Variance must be non-negative."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])
        gate = jnp.array([0.3, 0.3, 0.3])

        result = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, return_variance=True,
        )
        assert jnp.all(result["variance"] >= -1e-7)

    def test_variance_zero_at_perfect_capture(self):
        """When nu = 1 (perfect capture), variance should be ~0."""
        counts = jnp.array([[5, 3, 8]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.ones(1)

        result = denoise_counts(
            counts, r, p, p_capture=nu, return_variance=True,
        )
        np.testing.assert_allclose(result["variance"], 0.0, atol=1e-5)

    def test_variance_increases_with_count(self):
        """Larger observed counts → larger posterior variance."""
        r = jnp.array([5.0])
        p = jnp.float32(0.3)
        nu = jnp.array([0.5])

        counts_low = jnp.array([[1.0]])
        counts_high = jnp.array([[100.0]])

        var_low = denoise_counts(
            counts_low, r, p, p_capture=nu, return_variance=True,
        )["variance"]
        var_high = denoise_counts(
            counts_high, r, p, p_capture=nu, return_variance=True,
        )["variance"]

        assert float(var_high[0, 0]) > float(var_low[0, 0])

    def test_multi_sample_variance_shape(self):
        """Multi-sample with variance returns correct shapes."""
        n_samples, n_cells, n_genes = 3, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 5
        r = jnp.ones((n_samples, n_genes)) * 3.0
        p = jnp.ones(n_samples) * 0.4
        nu = jnp.ones((n_samples, n_cells)) * 0.5

        result = denoise_counts(
            counts, r, p, p_capture=nu, return_variance=True,
        )
        assert result["denoised_counts"].shape == (n_samples, n_cells, n_genes)
        assert result["variance"].shape == (n_samples, n_cells, n_genes)


# ==============================================================================
# Tests for SVI denoising integration
# ==============================================================================


class TestSVIDenoising:
    """Tests for SVI ``denoise_counts_map`` and ``denoise_counts_posterior``."""

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_svi_denoise_map_shape(self, dataset, rng, model):
        """MAP denoising returns (n_cells, n_genes) for all model types."""
        counts, n_cells, n_genes = dataset
        results = _fit_svi(model, counts)

        denoised = results.denoise_counts_map(
            counts=counts, rng_key=rng, verbose=False,
        )
        assert denoised.shape == (n_cells, n_genes)

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_svi_denoise_posterior_shape(self, dataset, rng, model):
        """Posterior denoising returns (n_samples, n_cells, n_genes)."""
        counts, n_cells, n_genes = dataset
        results = _fit_svi(model, counts)

        denoised = results.denoise_counts_posterior(
            counts=counts, rng_key=rng, n_samples=3, verbose=False,
        )
        assert denoised.shape[1:] == (n_cells, n_genes)
        assert denoised.shape[0] > 0

    def test_svi_denoise_stored_attribute(self, dataset, rng):
        """Verify result is stored in self.denoised_counts."""
        counts, _, _ = dataset
        results = _fit_svi("nbvcp", counts)

        results.denoise_counts_map(
            counts=counts, rng_key=rng, store_result=True, verbose=False,
        )
        assert results.denoised_counts is not None

    def test_svi_denoise_map_non_negative(self, dataset, rng):
        """MAP denoised counts must be non-negative."""
        counts, _, _ = dataset
        results = _fit_svi("zinbvcp", counts)

        denoised = results.denoise_counts_map(
            counts=counts, rng_key=rng, verbose=False,
        )
        assert jnp.all(denoised >= -1e-5)

    def test_svi_denoise_nbdm_identity(self, dataset, rng):
        """NBDM denoising should approximately equal observed counts."""
        counts, _, _ = dataset
        results = _fit_svi("nbdm", counts)

        denoised = results.denoise_counts_map(
            counts=counts, rng_key=rng, verbose=False,
        )
        np.testing.assert_allclose(denoised, counts, atol=1e-3)

    def test_svi_denoise_with_cell_batching(self, dataset, rng):
        """MAP denoising with cell batching produces correct shape."""
        counts, n_cells, n_genes = dataset
        results = _fit_svi("nbvcp", counts)

        denoised = results.denoise_counts_map(
            counts=counts, rng_key=rng, cell_batch_size=3, verbose=False,
        )
        assert denoised.shape == (n_cells, n_genes)

    def test_svi_denoise_map_variance(self, dataset, rng):
        """MAP denoising with return_variance=True returns dict."""
        counts, _, _ = dataset
        results = _fit_svi("nbvcp", counts)

        result = results.denoise_counts_map(
            counts=counts, rng_key=rng, return_variance=True, verbose=False,
        )
        assert isinstance(result, dict)
        assert "denoised_counts" in result
        assert "variance" in result

    def test_svi_denoise_posterior_variance(self, dataset, rng):
        """Posterior denoising with return_variance=True returns dict."""
        counts, _, _ = dataset
        results = _fit_svi("nbvcp", counts)

        result = results.denoise_counts_posterior(
            counts=counts, rng_key=rng, n_samples=2,
            return_variance=True, verbose=False,
        )
        assert isinstance(result, dict)
        assert result["denoised_counts"].ndim == 3
        assert result["variance"].ndim == 3


# ==============================================================================
# Tests for MCMC denoising integration
# ==============================================================================


class TestMCMCDenoising:
    """Tests for MCMC ``denoise_counts``."""

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_mcmc_denoise_shape(self, dataset, rng, model):
        """MCMC denoising returns (n_posterior_samples, n_cells, n_genes)."""
        counts, n_cells, n_genes = dataset
        results = _fit_mcmc(model, counts)

        denoised = results.denoise_counts(
            counts=counts, rng_key=rng, verbose=False,
        )
        assert denoised.shape[-1] == n_genes
        assert denoised.shape[-2] == n_cells

    def test_mcmc_denoise_stored_attribute(self, dataset, rng):
        """Verify result is stored in self.denoised_counts."""
        counts, _, _ = dataset
        results = _fit_mcmc("nbvcp", counts)

        results.denoise_counts(
            counts=counts, rng_key=rng, store_result=True, verbose=False,
        )
        assert results.denoised_counts is not None

    def test_mcmc_denoise_with_cell_batching(self, dataset, rng):
        """MCMC denoising with cell batching produces correct shape."""
        counts, n_cells, n_genes = dataset
        results = _fit_mcmc("nbvcp", counts)

        denoised = results.denoise_counts(
            counts=counts, rng_key=rng, cell_batch_size=4, verbose=False,
        )
        assert denoised.shape[-1] == n_genes
        assert denoised.shape[-2] == n_cells

    def test_mcmc_denoise_nbdm_identity(self, dataset, rng):
        """NBDM denoising via MCMC should approximately equal observed."""
        counts, _, _ = dataset
        results = _fit_mcmc("nbdm", counts)

        denoised = results.denoise_counts(
            counts=counts, rng_key=rng, verbose=False,
        )
        # Each posterior sample's denoised should ≈ counts
        for s in range(denoised.shape[0]):
            np.testing.assert_allclose(
                denoised[s], counts, atol=1e-3,
            )

    def test_mcmc_denoise_variance(self, dataset, rng):
        """MCMC denoising with return_variance=True returns dict."""
        counts, _, _ = dataset
        results = _fit_mcmc("nbvcp", counts)

        result = results.denoise_counts(
            counts=counts, rng_key=rng,
            return_variance=True, verbose=False,
        )
        assert isinstance(result, dict)
        assert "denoised_counts" in result
        assert "variance" in result


# ==============================================================================
# Tests for gene-specific p denoising (hierarchical model)
# ==============================================================================


class TestDenoiseGeneSpecificP:
    """Tests for denoising when p is gene-specific (hierarchical model).

    Verifies that ``denoise_counts`` correctly handles ``p`` arrays
    with shape ``(n_genes,)`` or ``(n_components, n_genes)`` instead of
    scalar, as produced by hierarchical parameterizations.
    """

    def test_gene_specific_p_shape(self):
        """Gene-specific p produces correct output shape."""
        n_cells, n_genes = 8, 5
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.linspace(0.2, 0.6, n_genes)
        nu = jnp.ones(n_cells) * 0.5

        result = denoise_counts(counts, r, p, p_capture=nu)
        assert result.shape == (n_cells, n_genes)

    def test_gene_specific_p_non_negative(self, rng):
        """Gene-specific p denoised counts are non-negative."""
        n_cells, n_genes = 6, 4
        counts = jnp.array(
            np.random.RandomState(99).negative_binomial(3, 0.4, (n_cells, n_genes)),
            dtype=jnp.float32,
        )
        r = jnp.ones(n_genes) * 3.0
        p = jnp.array([0.2, 0.4, 0.5, 0.7])
        nu = jnp.full(n_cells, 0.4)

        for method in ("mean", "mode", "sample"):
            result = denoise_counts(
                counts, r, p, p_capture=nu, method=method, rng_key=rng,
            )
            assert jnp.all(result >= 0), f"method={method}"

    def test_gene_specific_p_ge_observed(self):
        """Denoised mean with gene-specific p >= observed counts."""
        n_cells, n_genes = 4, 5
        counts = jnp.array(
            np.random.RandomState(88).negative_binomial(5, 0.3, (n_cells, n_genes)),
            dtype=jnp.float32,
        )
        r = jnp.ones(n_genes) * 4.0
        p = jnp.linspace(0.2, 0.7, n_genes)
        nu = jnp.full(n_cells, 0.5)

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")
        assert jnp.all(result >= counts - 1e-5)

    def test_gene_specific_p_matches_scalar_when_uniform(self):
        """When all p_g are equal, gene-specific matches scalar result."""
        n_cells, n_genes = 6, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 5
        r = jnp.array([2.0, 3.0, 4.0, 1.0])
        p_val = 0.4
        p_scalar = jnp.float32(p_val)
        p_vector = jnp.full(n_genes, p_val)
        nu = jnp.full(n_cells, 0.5)

        result_scalar = denoise_counts(counts, r, p_scalar, p_capture=nu)
        result_vector = denoise_counts(counts, r, p_vector, p_capture=nu)
        np.testing.assert_allclose(result_scalar, result_vector, atol=1e-5)

    def test_gene_specific_p_different_genes_differ(self):
        """Different p_g values produce different denoised values per gene."""
        n_cells, n_genes = 4, 3
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 5
        r = jnp.ones(n_genes) * 4.0
        p = jnp.array([0.1, 0.5, 0.9])
        nu = jnp.full(n_cells, 0.5)

        result = denoise_counts(counts, r, p, p_capture=nu, method="mean")
        # Genes with different p should have different denoised values
        assert not jnp.allclose(result[:, 0], result[:, 1])
        assert not jnp.allclose(result[:, 1], result[:, 2])

    def test_gene_specific_p_nbdm_identity(self):
        """Without VCP, gene-specific p still gives identity."""
        counts = jnp.array([[5, 0, 3], [1, 2, 7]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.array([0.3, 0.5, 0.7])

        result = denoise_counts(counts, r, p)
        np.testing.assert_allclose(result, counts, atol=1e-5)

    def test_gene_specific_p_cell_batching(self):
        """Cell batching with gene-specific p matches full computation."""
        n_cells, n_genes = 15, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array([0.2, 0.4, 0.5, 0.7])
        nu = jnp.full(n_cells, 0.5)

        full = denoise_counts(counts, r, p, p_capture=nu)
        batched = denoise_counts(
            counts, r, p, p_capture=nu, cell_batch_size=4,
        )
        np.testing.assert_allclose(full, batched, atol=1e-5)

    def test_gene_specific_p_with_gate(self):
        """Gene-specific p works correctly with zero-inflation gate."""
        n_cells, n_genes = 6, 3
        counts = jnp.array(
            [[5, 0, 3], [0, 0, 0], [2, 1, 0],
             [0, 3, 1], [4, 0, 2], [1, 0, 0]],
            dtype=jnp.float32,
        )
        r = jnp.array([3.0, 2.0, 4.0])
        p = jnp.array([0.3, 0.5, 0.7])
        nu = jnp.full(n_cells, 0.4)
        gate = jnp.array([0.1, 0.2, 0.15])

        result = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mean",
        )
        assert result.shape == (n_cells, n_genes)
        assert jnp.all(result >= 0)
        assert jnp.all(jnp.isfinite(result))

    def test_gene_specific_p_mixture_marginal(self):
        """Gene-specific p with mixture model (marginal path)."""
        n_cells, n_genes, n_comp = 6, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_comp, n_genes)) * jnp.array([[3.0], [5.0]])
        p = jnp.array([[0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8]])
        nu = jnp.full(n_cells, 0.5)
        weights = jnp.array([0.6, 0.4])

        result = denoise_counts(
            counts, r, p, p_capture=nu,
            mixing_weights=weights, method="mean",
        )
        assert result.shape == (n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_gene_specific_p_mixture_assignment(self):
        """Gene-specific p with mixture model (component assignment)."""
        n_cells, n_genes, n_comp = 8, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_comp, n_genes)) * jnp.array([[3.0], [5.0]])
        p = jnp.array([[0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8]])
        nu = jnp.full(n_cells, 0.5)
        weights = jnp.array([0.6, 0.4])
        assignment = jnp.array([0, 0, 1, 1, 0, 1, 0, 1])

        result = denoise_counts(
            counts, r, p, p_capture=nu,
            mixing_weights=weights,
            component_assignment=assignment, method="mean",
        )
        assert result.shape == (n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_gene_specific_p_mixture_assignment_batched(self):
        """Gene-specific p with mixture + assignment + cell_batch_size."""
        n_cells, n_genes, n_comp = 10, 4, 2
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones((n_comp, n_genes)) * jnp.array([[3.0], [5.0]])
        p = jnp.array([[0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8]])
        nu = jnp.full(n_cells, 0.5)
        weights = jnp.array([0.6, 0.4])
        assignment = jnp.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])

        full = denoise_counts(
            counts, r, p, p_capture=nu,
            mixing_weights=weights,
            component_assignment=assignment, method="mean",
        )
        batched = denoise_counts(
            counts, r, p, p_capture=nu,
            mixing_weights=weights,
            component_assignment=assignment,
            method="mean", cell_batch_size=3,
        )
        np.testing.assert_allclose(full, batched, atol=1e-5)

    def test_gene_specific_p_multi_sample(self):
        """Multi-sample denoising with gene-specific p."""
        n_samples, n_cells, n_genes = 3, 6, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 2
        r = jnp.ones((n_samples, n_genes)) * 5.0
        p = jnp.tile(jnp.linspace(0.2, 0.6, n_genes), (n_samples, 1))
        nu = jnp.ones((n_samples, n_cells)) * 0.5

        result = denoise_counts(counts, r, p, p_capture=nu)
        assert result.shape == (n_samples, n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_gene_specific_p_variance(self):
        """return_variance works with gene-specific p."""
        n_cells, n_genes = 6, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array([0.2, 0.4, 0.5, 0.7])
        nu = jnp.full(n_cells, 0.5)

        result = denoise_counts(
            counts, r, p, p_capture=nu, return_variance=True,
        )
        assert isinstance(result, dict)
        assert result["denoised_counts"].shape == (n_cells, n_genes)
        assert result["variance"].shape == (n_cells, n_genes)
        assert jnp.all(result["variance"] >= 0)


# ==============================================================================
# Tests for tuple method (independent control of non-zero and ZI zero methods)
# ==============================================================================


class TestDenoiseTupleMethod:
    """Tests for the ``(general_method, zi_zero_method)`` tuple interface."""

    def test_string_backward_compat(self):
        """Single string method still works exactly as before."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])

        result_str = denoise_counts(counts, r, p, p_capture=nu, method="mean")
        result_tuple = denoise_counts(
            counts, r, p, p_capture=nu, method=("mean", "mean"),
        )
        np.testing.assert_allclose(result_str, result_tuple, atol=1e-5)

    def test_tuple_mean_mean_matches_string_mean(self):
        """Tuple ("mean", "mean") is identical to string "mean" for ZINB."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])
        gate = jnp.array([0.3, 0.3, 0.3])

        result_str = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mean",
        )
        result_tuple = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method=("mean", "mean"),
        )
        np.testing.assert_allclose(result_str, result_tuple, atol=1e-5)

    def test_tuple_nonzero_positions_match_general(self, rng):
        """Non-zero positions use the general method, unaffected by zi_zero."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])
        gate = jnp.array([0.3, 0.3, 0.3])
        nonzero_mask = counts > 0

        # general_method="mean", zi_zero_method="sample"
        result_tuple = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate,
            method=("mean", "sample"), rng_key=rng,
        )
        result_mean = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mean",
        )

        # Non-zero positions should be identical (both use "mean")
        np.testing.assert_allclose(
            result_tuple[nonzero_mask], result_mean[nonzero_mask], atol=1e-5,
        )

    def test_tuple_mode_general_nonzero(self, rng):
        """With ("mode", "sample"), non-zero positions match mode method."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])
        gate = jnp.array([0.3, 0.3, 0.3])
        nonzero_mask = counts > 0

        result_tuple = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate,
            method=("mode", "sample"), rng_key=rng,
        )
        result_mode = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate, method="mode",
        )

        np.testing.assert_allclose(
            result_tuple[nonzero_mask], result_mode[nonzero_mask], atol=1e-5,
        )

    def test_tuple_sample_zi_zeros_stochastic(self, rng):
        """ZINB zeros with zi_zero_method="sample" vary across RNG keys."""
        counts = jnp.array([[0, 0, 0]], dtype=jnp.float32)
        r = jnp.array([3.0, 5.0, 2.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5])
        gate = jnp.array([0.5, 0.5, 0.5])

        key1, key2 = random.split(rng)
        r1 = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate,
            method=("mean", "sample"), rng_key=key1,
        )
        r2 = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate,
            method=("mean", "sample"), rng_key=key2,
        )

        # With high gate probability, at least some zeros should differ
        assert not jnp.allclose(r1, r2)

    def test_tuple_sample_zi_zero_keeps_genuine_zeros(self, rng):
        """Sampled ZI zeros: genuine NB zeros use NB posterior (VCP-aware).

        With gate=1.0 (all from dropout), zeros are replaced by prior
        NB samples.  With gate=0.0 (all genuine NB), the NB posterior
        at u=0 is used: positive when VCP is present (unobserved mRNA),
        zero when there is no capture probability.
        """
        counts = jnp.array([[0, 0, 0]], dtype=jnp.float32)
        r = jnp.array([5.0, 3.0, 7.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5])

        # gate = 0, WITH VCP → genuine NB zeros get the NB posterior
        # (positive because capture loss hides real expression)
        gate_zero = jnp.array([0.0, 0.0, 0.0])
        result_vcp = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate_zero,
            method=("mean", "sample"), rng_key=rng,
        )
        assert jnp.all(result_vcp >= 0)

        # gate = 0, WITHOUT VCP → genuine NB zeros stay at 0
        result_no_vcp = denoise_counts(
            counts, r, p, p_capture=None, gate=gate_zero,
            method=("mean", "sample"), rng_key=rng,
        )
        np.testing.assert_allclose(result_no_vcp, 0.0, atol=1e-5)

        # gate = 1 → all from dropout → should all be positive
        gate_one = jnp.array([1.0, 1.0, 1.0])
        result_all_gate = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate_one,
            method=("mean", "sample"), rng_key=rng,
        )
        assert jnp.all(result_all_gate >= 0)

    def test_tuple_shape_and_non_negative(self, rng):
        """Tuple method returns correct shape and non-negative values."""
        n_cells, n_genes = 8, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        counts = counts.at[0, :].set(0)
        r = jnp.ones(n_genes) * 5.0
        p = jnp.float32(0.4)
        nu = jnp.ones(n_cells) * 0.5
        gate = jnp.ones(n_genes) * 0.3

        for method in [
            ("mean", "sample"),
            ("mode", "sample"),
            ("mean", "mode"),
            ("sample", "sample"),
        ]:
            result = denoise_counts(
                counts, r, p, p_capture=nu, gate=gate,
                method=method, rng_key=rng,
            )
            assert result.shape == (n_cells, n_genes), f"method={method}"
            assert jnp.all(result >= 0), f"method={method}"

    def test_invalid_tuple_raises(self):
        """Invalid tuple elements raise ValueError."""
        counts = jnp.ones((2, 3), dtype=jnp.float32)
        r = jnp.ones(3)
        p = jnp.float32(0.5)

        with pytest.raises(ValueError, match="method"):
            denoise_counts(counts, r, p, method=("mean", "invalid"))
        with pytest.raises(ValueError, match="method"):
            denoise_counts(counts, r, p, method=("bad",))
        with pytest.raises(ValueError, match="method"):
            denoise_counts(counts, r, p, method=123)

    def test_tuple_without_gate_same_as_general(self, rng):
        """Without gate, tuple zi_zero has no effect (no ZI zeros to handle)."""
        counts = jnp.array([[5, 0, 3], [0, 2, 0]], dtype=jnp.float32)
        r = jnp.array([2.0, 1.5, 3.0])
        p = jnp.float32(0.4)
        nu = jnp.array([0.5, 0.7])

        result_mean = denoise_counts(
            counts, r, p, p_capture=nu, method="mean",
        )
        result_tuple = denoise_counts(
            counts, r, p, p_capture=nu,
            method=("mean", "sample"), rng_key=rng,
        )

        np.testing.assert_allclose(result_mean, result_tuple, atol=1e-5)

    def test_tuple_multi_sample(self, rng):
        """Tuple method works with multi-sample (posterior) arrays."""
        n_samples, n_cells, n_genes = 3, 6, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 2
        counts = counts.at[0, :].set(0)
        r = jnp.ones((n_samples, n_genes)) * 5.0
        p = jnp.ones(n_samples) * 0.3
        nu = jnp.ones((n_samples, n_cells)) * 0.6
        gate = jnp.ones(n_genes) * 0.3

        result = denoise_counts(
            counts, r, p, p_capture=nu, gate=gate,
            method=("mean", "sample"), rng_key=rng,
        )
        assert result.shape == (n_samples, n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_tuple_cell_batching_matches_full(self, rng):
        """Cell batching with tuple method matches unbatched result."""
        n_cells, n_genes = 15, 4
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.float32) * 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.float32(0.3)
        nu = jnp.ones(n_cells) * 0.5

        # No gate → deterministic for both components of the tuple
        full = denoise_counts(
            counts, r, p, p_capture=nu, method=("mean", "mean"),
        )
        batched = denoise_counts(
            counts, r, p, p_capture=nu,
            method=("mean", "mean"), cell_batch_size=4,
        )
        np.testing.assert_allclose(full, batched, atol=1e-5)


# ==============================================================================
# Tests for get_denoised_anndata (SVI and MCMC)
# ==============================================================================


class TestGetDenoisedAnnData:
    """Tests for ``get_denoised_anndata()`` on SVI and MCMC results."""

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_svi_single_dataset_structure(self, dataset, rng, model):
        """SVI get_denoised_anndata returns valid AnnData for all models."""
        from anndata import AnnData

        counts, n_cells, n_genes = dataset
        results = _fit_svi(model, counts)

        adata = results.get_denoised_anndata(
            counts=counts, rng_key=rng, verbose=False,
        )

        assert isinstance(adata, AnnData)
        assert adata.X.shape == (n_cells, n_genes)
        assert "original_counts" in adata.layers
        np.testing.assert_allclose(
            adata.layers["original_counts"], np.asarray(counts),
        )
        assert "scribe_denoising" in adata.uns
        meta = adata.uns["scribe_denoising"]
        assert meta["dataset_index"] == 0
        assert meta["parameter_source"] == "map"

    def test_svi_no_original_counts_layer(self, dataset, rng):
        """include_original_counts=False omits the layer."""
        counts, _, _ = dataset
        results = _fit_svi("nbvcp", counts)

        adata = results.get_denoised_anndata(
            counts=counts, rng_key=rng,
            include_original_counts=False, verbose=False,
        )
        assert "original_counts" not in adata.layers

    def test_svi_multi_dataset(self, dataset, rng):
        """n_datasets > 1 returns a list of AnnData objects."""
        counts, n_cells, n_genes = dataset
        results = _fit_svi("nbvcp", counts)

        adatas = results.get_denoised_anndata(
            counts=counts, rng_key=rng, n_datasets=3, verbose=False,
        )

        assert isinstance(adatas, list)
        assert len(adatas) == 3

        # First dataset is MAP-based
        assert adatas[0].uns["scribe_denoising"]["parameter_source"] == "map"

        # Subsequent datasets are posterior-sample-based
        for i in range(1, 3):
            meta = adatas[i].uns["scribe_denoising"]
            assert meta["dataset_index"] == i
            assert "posterior_sample" in meta["parameter_source"]
            assert adatas[i].X.shape == (n_cells, n_genes)

    def test_svi_tuple_method_stored_in_metadata(self, dataset, rng):
        """Tuple method value is recorded in .uns metadata."""
        counts, _, _ = dataset
        results = _fit_svi("zinbvcp", counts)

        adata = results.get_denoised_anndata(
            counts=counts, rng_key=rng,
            method=("mode", "sample"), verbose=False,
        )
        assert adata.uns["scribe_denoising"]["method"] == ["mode", "sample"]

    def test_svi_string_method_works(self, dataset, rng):
        """Plain string method also works for get_denoised_anndata."""
        counts, _, _ = dataset
        results = _fit_svi("nbvcp", counts)

        adata = results.get_denoised_anndata(
            counts=counts, rng_key=rng, method="mean", verbose=False,
        )
        assert isinstance(adata.X, np.ndarray)

    def test_svi_requires_counts_or_adata(self, dataset, rng):
        """Raises ValueError if neither counts nor adata is provided."""
        counts, _, _ = dataset
        results = _fit_svi("nbvcp", counts)

        with pytest.raises(ValueError, match="counts.*adata"):
            results.get_denoised_anndata(rng_key=rng, verbose=False)

    def test_svi_denoised_non_negative(self, dataset, rng):
        """Denoised .X values are non-negative."""
        counts, _, _ = dataset
        results = _fit_svi("zinbvcp", counts)

        adata = results.get_denoised_anndata(
            counts=counts, rng_key=rng, verbose=False,
        )
        assert np.all(adata.X >= -1e-5)

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_mcmc_single_dataset_structure(self, dataset, rng, model):
        """MCMC get_denoised_anndata returns valid AnnData for all models."""
        from anndata import AnnData

        counts, n_cells, n_genes = dataset
        results = _fit_mcmc(model, counts)

        adata = results.get_denoised_anndata(
            counts=counts, rng_key=rng, verbose=False,
        )

        assert isinstance(adata, AnnData)
        assert adata.X.shape == (n_cells, n_genes)
        assert "original_counts" in adata.layers
        assert "scribe_denoising" in adata.uns
        assert adata.uns["scribe_denoising"]["dataset_index"] == 0

    def test_mcmc_multi_dataset(self, dataset, rng):
        """MCMC n_datasets > 1 returns a list."""
        counts, n_cells, n_genes = dataset
        results = _fit_mcmc("nbvcp", counts)

        adatas = results.get_denoised_anndata(
            counts=counts, rng_key=rng, n_datasets=2, verbose=False,
        )

        assert isinstance(adatas, list)
        assert len(adatas) == 2
        assert adatas[0].uns["scribe_denoising"]["parameter_source"] == (
            "posterior_mean"
        )
        assert "mcmc_sample" in adatas[1].uns["scribe_denoising"][
            "parameter_source"
        ]

    def test_svi_adata_template(self, dataset, rng):
        """Passing adata as template copies obs/var metadata."""
        import pandas as pd
        from anndata import AnnData

        counts, n_cells, n_genes = dataset
        results = _fit_svi("nbvcp", counts)

        # Create a template with metadata
        obs = pd.DataFrame(
            {"cell_type": [f"type_{i}" for i in range(n_cells)]},
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        var = pd.DataFrame(
            {"gene_name": [f"gene_{g}" for g in range(n_genes)]},
            index=[f"gene_{g}" for g in range(n_genes)],
        )
        template = AnnData(
            X=np.asarray(counts), obs=obs, var=var,
        )

        out = results.get_denoised_anndata(
            adata=template, rng_key=rng, verbose=False,
        )

        assert list(out.obs.columns) == ["cell_type"]
        assert list(out.var.columns) == ["gene_name"]
        assert out.obs.index.tolist() == obs.index.tolist()

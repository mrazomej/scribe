"""
Tests for biological (denoised) posterior predictive check sampling.

These tests verify that the ``get_ppc_samples_biological`` and
``get_map_ppc_samples_biological`` methods correctly generate count samples
from the base Negative Binomial distribution NB(r, p), stripping technical
noise parameters (capture probability, zero-inflation gate).

Tests cover:
- The shared ``sample_biological_nb`` utility (standard and mixture models)
- SVI biological PPC methods (full posterior and MAP-based)
- MCMC biological PPC methods
- Shape correctness for all model types
- Non-negativity of generated counts
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.sampling import sample_biological_nb
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig
from scribe.inference import run_scribe
from scribe.inference.preset_builder import build_config_from_preset


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def bio_dataset():
    """Small dataset for biological PPC tests."""
    np.random.seed(123)
    n_cells, n_genes = 10, 5
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    return jnp.array(counts), n_cells, n_genes


@pytest.fixture(scope="module")
def rng():
    """Provide a consistent PRNG key."""
    return random.PRNGKey(99)


# Cache fitted results across tests within this module to avoid re-fitting
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

    # Add capture priors for VCP models
    if "vcp" in model:
        if parameterization == "odds_ratio":
            priors["phi_capture"] = (3, 2)
        else:
            priors["p_capture"] = (1, 1)

    # Add gate priors for ZINB variants
    if "zinb" in model:
        priors["gate"] = (1, 1)

    return priors


# ==============================================================================
# Tests for sample_biological_nb utility
# ==============================================================================


class TestSampleBiologicalNB:
    """Tests for the core ``sample_biological_nb`` utility function."""

    def test_standard_model_map_shape(self, rng):
        """MAP-based sampling should return (n_samples, n_cells, n_genes)."""
        n_genes, n_cells, n_samples = 5, 10, 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=n_samples,
        )
        assert result.shape == (n_samples, n_cells, n_genes)

    def test_standard_model_posterior_shape(self, rng):
        """Posterior-based sampling should infer n_samples from r's shape."""
        n_post_samples, n_genes, n_cells = 4, 5, 8
        r = jnp.ones((n_post_samples, n_genes)) * 5.0
        p = jnp.ones(n_post_samples) * 0.3

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
        )
        assert result.shape == (n_post_samples, n_cells, n_genes)

    def test_mixture_model_map_shape(self, rng):
        """Mixture MAP sampling with mixing_weights."""
        n_components, n_genes, n_cells, n_samples = 3, 5, 10, 2
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.array(0.4)
        mw = jnp.ones(n_components) / n_components

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=n_samples, mixing_weights=mw,
        )
        assert result.shape == (n_samples, n_cells, n_genes)

    def test_mixture_model_posterior_shape(self, rng):
        """Mixture posterior sampling with per-sample mixing_weights."""
        n_post, n_components, n_genes, n_cells = 4, 3, 5, 8
        r = jnp.ones((n_post, n_components, n_genes)) * 5.0
        p = jnp.ones(n_post) * 0.3
        mw = jnp.ones((n_post, n_components)) / n_components

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, mixing_weights=mw,
        )
        assert result.shape == (n_post, n_cells, n_genes)

    def test_counts_are_non_negative(self, rng):
        """All generated counts must be non-negative integers."""
        r = jnp.ones(5) * 5.0
        p = jnp.array(0.4)
        result = sample_biological_nb(
            r=r, p=p, n_cells=10, rng_key=rng, n_samples=2,
        )
        assert jnp.all(result >= 0)

    def test_cell_batching_matches_full(self, rng):
        """Cell batching should produce the same shape as full sampling."""
        n_genes, n_cells = 5, 20
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)

        result_full = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=1,
        )
        result_batched = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=1,
            cell_batch_size=7,
        )
        assert result_full.shape == result_batched.shape

    def test_component_specific_p_mixture(self, rng):
        """Mixture model with component-specific p values."""
        n_components, n_genes, n_cells = 2, 5, 10
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.array([0.3, 0.7])  # per-component
        mw = jnp.array([0.6, 0.4])

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=3, mixing_weights=mw,
        )
        assert result.shape == (3, n_cells, n_genes)
        assert jnp.all(result >= 0)


# ==============================================================================
# Tests for SVI biological PPC methods
# ==============================================================================


class TestSVIBiologicalPPC:
    """Tests for SVI ``get_ppc_samples_biological`` and
    ``get_map_ppc_samples_biological``."""

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_svi_ppc_biological_shape(self, bio_dataset, rng, model):
        """Biological PPC produces correct shape for all model types."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_svi(model, counts)

        ppc = results.get_ppc_samples_biological(
            rng_key=rng, n_samples=3, store_samples=True,
        )

        assert "parameter_samples" in ppc
        assert "predictive_samples" in ppc
        pred = ppc["predictive_samples"]
        assert pred.shape[-1] == n_genes
        assert pred.shape[-2] == n_cells
        assert jnp.all(pred >= 0)

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_svi_map_ppc_biological_shape(self, bio_dataset, rng, model):
        """MAP biological PPC produces correct shape for all model types."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_svi(model, counts)

        pred = results.get_map_ppc_samples_biological(
            rng_key=rng, n_samples=2, verbose=False,
        )

        assert pred.shape == (2, n_cells, n_genes)
        assert jnp.all(pred >= 0)

    def test_svi_biological_stored_attribute(self, bio_dataset, rng):
        """Verify samples are stored in predictive_samples_biological."""
        counts, _, _ = bio_dataset
        results = _fit_svi("nbvcp", counts)

        results.get_ppc_samples_biological(
            rng_key=rng, n_samples=2, store_samples=True,
        )
        assert hasattr(results, "predictive_samples_biological")
        assert results.predictive_samples_biological is not None

    def test_svi_map_biological_with_cell_batching(self, bio_dataset, rng):
        """MAP biological PPC with cell batching produces correct shape."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_svi("nbvcp", counts)

        pred = results.get_map_ppc_samples_biological(
            rng_key=rng, n_samples=1, cell_batch_size=3, verbose=False,
        )
        assert pred.shape == (1, n_cells, n_genes)

    def test_svi_biological_reuses_posterior_samples(self, bio_dataset, rng):
        """Should reuse existing posterior_samples if already available."""
        counts, _, _ = bio_dataset
        results = _fit_svi("nbdm", counts)

        # First call generates posterior samples
        results.get_posterior_samples(rng_key=rng, n_samples=3)
        saved_samples = results.posterior_samples

        # Second call should reuse them (not regenerate)
        results.get_ppc_samples_biological(rng_key=rng)
        assert results.posterior_samples is saved_samples


# ==============================================================================
# Tests for MCMC biological PPC methods
# ==============================================================================


class TestMCMCBiologicalPPC:
    """Tests for MCMC ``get_ppc_samples_biological``."""

    @pytest.mark.parametrize("model", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_mcmc_ppc_biological_shape(self, bio_dataset, rng, model):
        """Biological PPC produces correct shape for all MCMC model types."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_mcmc(model, counts)

        pred = results.get_ppc_samples_biological(
            rng_key=rng, store_samples=True,
        )

        assert pred.shape[-1] == n_genes
        assert pred.shape[-2] == n_cells
        assert jnp.all(pred >= 0)

    def test_mcmc_biological_stored_attribute(self, bio_dataset, rng):
        """Verify samples are stored in predictive_samples_biological."""
        counts, _, _ = bio_dataset
        results = _fit_mcmc("nbvcp", counts)

        results.get_ppc_samples_biological(rng_key=rng, store_samples=True)
        assert hasattr(results, "predictive_samples_biological")
        assert results.predictive_samples_biological is not None

    def test_mcmc_biological_with_cell_batching(self, bio_dataset, rng):
        """MCMC biological PPC with cell batching produces correct shape."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_mcmc("nbvcp", counts)

        pred = results.get_ppc_samples_biological(
            rng_key=rng, cell_batch_size=4,
        )
        assert pred.shape[-1] == n_genes
        assert pred.shape[-2] == n_cells


# ==============================================================================
# Tests for NBDM equivalence
# ==============================================================================


class TestNBDMEquivalence:
    """For NBDM models, biological PPC should be functionally equivalent to
    standard PPC (since NBDM has no technical parameters to strip)."""

    def test_nbdm_biological_and_standard_same_shape(self, bio_dataset, rng):
        """NBDM biological PPC should produce same shapes as standard PPC."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_svi("nbdm", counts)

        standard = results.get_ppc_samples(
            rng_key=rng, n_samples=3,
        )
        biological = results.get_ppc_samples_biological(
            rng_key=rng, n_samples=3,
        )

        assert (
            standard["predictive_samples"].shape
            == biological["predictive_samples"].shape
        )

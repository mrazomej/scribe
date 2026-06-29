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

    def test_svi_biological_with_cell_batching(self, bio_dataset, rng):
        """Posterior biological PPC with cell batching produces correct shape."""
        counts, n_cells, n_genes = bio_dataset
        results = _fit_svi("nbvcp", counts)

        ppc = results.get_ppc_samples_biological(
            rng_key=rng, n_samples=3, cell_batch_size=3, store_samples=False,
        )
        pred = ppc["predictive_samples"]
        assert pred.shape == (3, n_cells, n_genes)
        assert jnp.all(pred >= 0)

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


# ==============================================================================
# Unit tests for shared sampling helpers
# ==============================================================================


class TestPPCMapVmap:
    """Tests for the vmap-accelerated MAP path in PPC sampling.

    The MAP branch uses ``jax.vmap`` over independent RNG keys instead
    of a Python for-loop.  These tests verify that the vmap path
    produces correct shapes, reproducible results, and statistically
    sensible outputs.
    """

    def test_biological_map_shape(self, rng):
        """Biological MAP vmap produces (n_samples, n_cells, n_genes)."""
        n_genes, n_cells, n_samples = 5, 10, 4
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=n_samples,
        )
        assert result.shape == (n_samples, n_cells, n_genes)

    def test_biological_map_rng_reproducibility(self, rng):
        """Same RNG key must produce identical MAP vmap results."""
        n_genes, n_cells = 5, 10
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)

        a = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=3,
        )
        b = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=3,
        )
        np.testing.assert_array_equal(a, b)

    def test_biological_map_different_keys_differ(self):
        """Different RNG keys should produce different MAP results."""
        n_genes, n_cells = 5, 10
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)

        a = sample_biological_nb(
            r=r, p=p, n_cells=n_cells,
            rng_key=random.PRNGKey(0), n_samples=3,
        )
        b = sample_biological_nb(
            r=r, p=p, n_cells=n_cells,
            rng_key=random.PRNGKey(1), n_samples=3,
        )
        assert not jnp.allclose(a, b)

    def test_biological_map_non_negative(self, rng):
        """MAP vmap biological counts are non-negative."""
        r = jnp.ones(5) * 5.0
        p = jnp.array(0.4)
        result = sample_biological_nb(
            r=r, p=p, n_cells=10, rng_key=rng, n_samples=5,
        )
        assert jnp.all(result >= 0)

    def test_biological_map_mixture_shape(self, rng):
        """Mixture MAP vmap produces correct shape."""
        n_components, n_genes, n_cells = 3, 5, 10
        r = jnp.ones((n_components, n_genes)) * 5.0
        p = jnp.array(0.4)
        mw = jnp.ones(n_components) / n_components

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=4, mixing_weights=mw,
        )
        assert result.shape == (4, n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_posterior_ppc_map_shape(self, rng):
        """Full-model MAP vmap produces (n_samples, n_cells, n_genes)."""
        from scribe.sampling import sample_posterior_ppc

        n_genes, n_cells, n_samples = 5, 8, 3
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)
        gate = jnp.ones(n_genes) * 0.1
        nu = jnp.ones(n_cells) * 0.5

        result = sample_posterior_ppc(
            r=r, p=p, n_cells=n_cells, rng_key=rng,
            n_samples=n_samples, gate=gate, p_capture=nu,
        )
        assert result.shape == (n_samples, n_cells, n_genes)
        assert jnp.all(result >= 0)

    def test_posterior_ppc_map_rng_reproducibility(self, rng):
        """Same RNG key must produce identical full-model MAP vmap results."""
        from scribe.sampling import sample_posterior_ppc

        n_genes, n_cells = 5, 8
        r = jnp.ones(n_genes) * 5.0
        p = jnp.array(0.4)

        a = sample_posterior_ppc(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=3,
        )
        b = sample_posterior_ppc(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=3,
        )
        np.testing.assert_array_equal(a, b)

    def test_biological_map_statistical_properties(self, rng):
        """MAP vmap NB samples have plausible mean/variance relationship."""
        n_genes, n_cells = 1, 500
        r_val = 5.0
        p_val = 0.3
        r = jnp.ones(n_genes) * r_val
        p = jnp.array(p_val)

        # NB mean = r*p/(1-p), var = r*p/(1-p)^2
        expected_mean = r_val * p_val / (1.0 - p_val)

        result = sample_biological_nb(
            r=r, p=p, n_cells=n_cells, rng_key=rng, n_samples=1,
        )
        empirical_mean = float(jnp.mean(result))
        assert abs(empirical_mean - expected_mean) / expected_mean < 0.3


class TestHasSampleDim:
    """Tests for ``_has_sample_dim`` — reads ``has_sample_dim`` directly
    from the ``param_layouts["r"]`` AxisLayout metadata.
    """

    def test_no_sample_dim(self):
        """Layout with has_sample_dim=False returns False."""
        from scribe.sampling import _has_sample_dim
        from scribe.core.axis_layout import AxisLayout

        layouts = {"r": AxisLayout(("genes",), has_sample_dim=False)}
        assert _has_sample_dim(layouts) is False

    def test_has_sample_dim(self):
        """Layout with has_sample_dim=True returns True."""
        from scribe.sampling import _has_sample_dim
        from scribe.core.axis_layout import AxisLayout

        layouts = {"r": AxisLayout(("genes",), has_sample_dim=True)}
        assert _has_sample_dim(layouts) is True

    def test_mixture_no_sample_dim(self):
        """Mixture layout without sample dim returns False."""
        from scribe.sampling import _has_sample_dim
        from scribe.core.axis_layout import AxisLayout

        layouts = {"r": AxisLayout(("components", "genes"), has_sample_dim=False)}
        assert _has_sample_dim(layouts) is False

    def test_mixture_has_sample_dim(self):
        """Mixture layout with sample dim returns True."""
        from scribe.sampling import _has_sample_dim
        from scribe.core.axis_layout import AxisLayout

        layouts = {"r": AxisLayout(("components", "genes"), has_sample_dim=True)}
        assert _has_sample_dim(layouts) is True


class TestSlicePosteriorDraw:
    """Tests for ``_slice_posterior_draw`` — extracts parameter values for a
    single posterior draw using AxisLayout metadata to determine which
    parameters carry a leading sample axis.
    """

    def _make_layouts(self, axes_map, has_sample=True):
        """Helper to build a param_layouts dict from a simple axes mapping."""
        from scribe.core.axis_layout import AxisLayout

        return {
            k: AxisLayout(axes=v, has_sample_dim=has_sample)
            for k, v in axes_map.items()
        }

    def test_standard_model_slices_all_params(self):
        """All params with a sample axis should be sliced at the given index."""
        from scribe.sampling import _slice_posterior_draw

        n_samples, n_genes, n_cells = 10, 20, 50

        r = jnp.ones((n_samples, n_genes))
        p = jnp.ones((n_samples,)) * 0.3
        p_capture = jnp.ones((n_samples, n_cells)) * 0.9
        gate = jnp.ones((n_samples, n_genes)) * 0.1

        # Build posterior-level layouts (has_sample_dim=True)
        layouts = self._make_layouts({
            "r": ("genes",),
            "p": (),
            "p_capture": ("cells",),
            "gate": ("genes",),
        })

        draw = _slice_posterior_draw(
            3,
            r=r, p=p, p_capture=p_capture, gate=gate,
            mixing_weights=None, param_layouts=layouts,
        )

        assert draw["r"].shape == (n_genes,)
        assert draw["p"].shape == ()
        assert draw["p_capture"].shape == (n_cells,)
        assert draw["gate"].shape == (n_genes,)
        assert draw["mixing_weights"] is None
        assert draw["bnb_concentration"] is None

    def test_mixture_model_slices_all_params(self):
        """Mixture model params with sample axis should be properly sliced."""
        from scribe.sampling import _slice_posterior_draw

        n_samples, K, n_genes = 10, 3, 20

        r = jnp.ones((n_samples, K, n_genes))
        p = jnp.ones((n_samples,)) * 0.3
        gate = jnp.ones((n_samples, K, n_genes)) * 0.1
        mw = jnp.ones((n_samples, K)) / K

        layouts = self._make_layouts({
            "r": ("components", "genes"),
            "p": (),
            "gate": ("components", "genes"),
            "mixing_weights": ("components",),
        })

        draw = _slice_posterior_draw(
            5,
            r=r, p=p, p_capture=None, gate=gate,
            mixing_weights=mw, param_layouts=layouts,
        )

        assert draw["r"].shape == (K, n_genes)
        assert draw["p"].shape == ()
        assert draw["gate"].shape == (K, n_genes)
        assert draw["mixing_weights"].shape == (K,)

    def test_shared_params_not_sliced(self):
        """Params whose layout says has_sample_dim=False pass through."""
        from scribe.sampling import _slice_posterior_draw

        n_samples, n_genes = 10, 20

        r = jnp.ones((n_samples, n_genes))
        p_shared = jnp.array(0.3)
        gate_shared = jnp.ones((n_genes,)) * 0.1

        # r has sample dim; p and gate do not
        layouts = self._make_layouts({"r": ("genes",)}, has_sample=True)
        layouts["p"] = __import__(
            "scribe.core.axis_layout", fromlist=["AxisLayout"]
        ).AxisLayout(axes=(), has_sample_dim=False)
        layouts["gate"] = __import__(
            "scribe.core.axis_layout", fromlist=["AxisLayout"]
        ).AxisLayout(axes=("genes",), has_sample_dim=False)

        draw = _slice_posterior_draw(
            0,
            r=r, p=p_shared, p_capture=None, gate=gate_shared,
            mixing_weights=None, param_layouts=layouts,
        )

        # p is not sliced (no sample dim per layout)
        assert draw["p"].shape == ()
        # gate is not sliced (no sample dim per layout)
        assert draw["gate"].shape == (n_genes,)

    def test_bnb_concentration_sliced_when_has_sample_dim(self):
        """bnb_concentration with a sample axis should be sliced."""
        from scribe.sampling import _slice_posterior_draw

        n_samples, n_genes = 10, 20

        r = jnp.ones((n_samples, n_genes))
        p = jnp.ones((n_samples,)) * 0.3
        bnb = jnp.ones((n_samples, n_genes)) * 0.5

        layouts = self._make_layouts({
            "r": ("genes",),
            "p": (),
            "bnb_concentration": ("genes",),
        })

        draw = _slice_posterior_draw(
            7,
            r=r, p=p, p_capture=None, gate=None,
            mixing_weights=None, param_layouts=layouts,
            bnb_concentration=bnb,
        )

        assert draw["bnb_concentration"].shape == (n_genes,)

    def test_bnb_concentration_mixture_sliced(self):
        """Mixture bnb_concentration (S, K, G) should be sliced to (K, G)."""
        from scribe.sampling import _slice_posterior_draw

        n_samples, K, n_genes = 10, 3, 20

        r = jnp.ones((n_samples, K, n_genes))
        p = jnp.ones((n_samples,)) * 0.3
        mw = jnp.ones((n_samples, K)) / K
        bnb = jnp.ones((n_samples, K, n_genes)) * 0.5

        layouts = self._make_layouts({
            "r": ("components", "genes"),
            "p": (),
            "mixing_weights": ("components",),
            "bnb_concentration": ("components", "genes"),
        })

        draw = _slice_posterior_draw(
            2,
            r=r, p=p, p_capture=None, gate=None,
            mixing_weights=mw, param_layouts=layouts,
            bnb_concentration=bnb,
        )

        assert draw["bnb_concentration"].shape == (K, n_genes)

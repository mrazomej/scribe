"""Tests for the refactored ScribeMCMCResults dataclass and its mixins.

Validates construction, gene subsetting, component subsetting, MCMC
diagnostic delegation, and posterior access -- all using a lightweight
mock MCMC object so no actual inference is required.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import MagicMock

from scribe.mcmc.results import ScribeMCMCResults
from scribe.models.config import ModelConfig
from scribe.models.builders.parameter_specs import LogNormalSpec


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_model_config(
    base_model="nbdm", n_components=None, param_specs=None, **kwargs
):
    """Build a minimal ModelConfig for testing."""
    extra = {}
    if param_specs is not None:
        extra["param_specs"] = param_specs
    return ModelConfig(
        base_model=base_model,
        n_components=n_components,
        **extra,
        **kwargs,
    )


def _make_mock_mcmc(samples, extra_fields=None):
    """Create a mock numpyro.infer.MCMC with .get_samples and .get_extra_fields."""
    mcmc = MagicMock()
    mcmc.get_samples.return_value = samples
    mcmc.get_extra_fields.return_value = extra_fields or {}
    return mcmc


def _make_standard_samples(n_samples=20, n_genes=5):
    """Standard (p, r) canonical samples for an NBDM model."""
    rng = np.random.default_rng(42)
    return {
        "p": jnp.array(
            rng.uniform(0.2, 0.8, size=(n_samples,)), dtype=jnp.float32
        ),
        "r": jnp.array(
            rng.uniform(0.5, 5.0, size=(n_samples, n_genes)), dtype=jnp.float32
        ),
    }


def _make_mixture_samples(n_samples=20, n_genes=5, n_components=3):
    """Mixture model samples with mixing weights."""
    rng = np.random.default_rng(42)
    return {
        "p": jnp.array(
            rng.uniform(0.2, 0.8, size=(n_samples,)), dtype=jnp.float32
        ),
        "r": jnp.array(
            rng.uniform(0.5, 5.0, size=(n_samples, n_components, n_genes)),
            dtype=jnp.float32,
        ),
        "mixing_weights": jnp.array(
            np.full((n_samples, n_components), 1.0 / n_components),
            dtype=jnp.float32,
        ),
    }


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def standard_results():
    """ScribeMCMCResults for a standard (non-mixture) NBDM model."""
    n_genes = 5
    samples = _make_standard_samples(n_genes=n_genes)
    mcmc = _make_mock_mcmc(samples)
    return ScribeMCMCResults.from_mcmc(
        mcmc=mcmc,
        n_cells=10,
        n_genes=n_genes,
        model_type="nbdm",
        model_config=_make_model_config("nbdm"),
        prior_params={},
    )


@pytest.fixture
def mixture_results():
    """ScribeMCMCResults for a 3-component mixture NBDM model."""
    n_genes = 5
    n_components = 3
    samples = _make_mixture_samples(n_genes=n_genes, n_components=n_components)
    mcmc = _make_mock_mcmc(samples)
    # Mark r as mixture so component subsetting detects the axis
    r_spec = LogNormalSpec(
        name="r",
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        is_mixture=True,
        is_gene_specific=True,
    )
    return ScribeMCMCResults.from_mcmc(
        mcmc=mcmc,
        n_cells=10,
        n_genes=n_genes,
        model_type="nbdm_mix",
        model_config=_make_model_config(
            "nbdm", n_components=n_components, param_specs=[r_spec]
        ),
        prior_params={},
    )


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


class TestConstruction:
    """Test ScribeMCMCResults construction paths."""

    def test_from_mcmc_extracts_samples(self, standard_results):
        """from_mcmc should extract samples from the MCMC object."""
        assert "p" in standard_results.samples
        assert "r" in standard_results.samples
        assert standard_results._mcmc is not None

    def test_direct_construction(self):
        """Direct dataclass construction (e.g. for subsets) works."""
        samples = _make_standard_samples(n_genes=3)
        results = ScribeMCMCResults(
            samples=samples,
            n_cells=10,
            n_genes=3,
            model_type="nbdm",
            model_config=_make_model_config("nbdm"),
            prior_params={},
        )
        assert results._mcmc is None
        assert results.n_genes == 3

    def test_n_components_from_config(self):
        """n_components should be inferred from model_config when not set."""
        samples = _make_mixture_samples(n_genes=4, n_components=2)
        mcmc = _make_mock_mcmc(samples)
        results = ScribeMCMCResults.from_mcmc(
            mcmc=mcmc,
            n_cells=5,
            n_genes=4,
            model_type="nbdm_mix",
            model_config=_make_model_config("nbdm", n_components=2),
            prior_params={},
        )
        assert results.n_components == 2

    def test_validation_rejects_mismatched_model_type(self):
        """Mismatched model_type and base_model should raise ValueError."""
        samples = _make_standard_samples()
        with pytest.raises(ValueError, match="does not match"):
            ScribeMCMCResults(
                samples=samples,
                n_cells=10,
                n_genes=5,
                model_type="zinb",
                model_config=_make_model_config("nbdm"),
                prior_params={},
            )


# --------------------------------------------------------------------------
# Posterior access
# --------------------------------------------------------------------------


class TestPosteriorAccess:
    """Test get_posterior_samples and get_samples."""

    def test_get_posterior_samples_returns_samples(self, standard_results):
        """get_posterior_samples() returns the stored samples dict."""
        samples = standard_results.get_posterior_samples()
        assert "p" in samples
        assert "r" in samples
        assert samples is standard_results.samples

    def test_posterior_samples_property(self, standard_results):
        """The .posterior_samples property should return the samples dict."""
        ps = standard_results.posterior_samples
        assert "p" in ps
        assert ps is standard_results.samples

    def test_get_samples_group_by_chain_requires_mcmc(self):
        """group_by_chain on a subset (no _mcmc) should raise."""
        samples = _make_standard_samples(n_genes=3)
        results = ScribeMCMCResults(
            samples=samples,
            n_cells=10,
            n_genes=3,
            model_type="nbdm",
            model_config=_make_model_config("nbdm"),
            prior_params={},
        )
        with pytest.raises(RuntimeError, match="group_by_chain"):
            results.get_samples(group_by_chain=True)


# --------------------------------------------------------------------------
# Gene subsetting
# --------------------------------------------------------------------------


class TestGeneSubsetting:
    """Test __getitem__ gene indexing."""

    def test_slice_indexing(self, standard_results):
        """Slice indexing should return ScribeMCMCResults with fewer genes."""
        subset = standard_results[0:3]
        assert isinstance(subset, ScribeMCMCResults)
        assert subset.n_genes == 3
        assert subset.samples["r"].shape[-1] == 3
        # Subset should not carry the MCMC object
        assert subset._mcmc is None

    def test_int_indexing(self, standard_results):
        """Single int indexing should return 1-gene results."""
        subset = standard_results[2]
        assert subset.n_genes == 1

    def test_bool_mask(self, standard_results):
        """Boolean mask indexing."""
        mask = jnp.array([True, False, True, False, True])
        subset = standard_results[mask]
        assert subset.n_genes == 3

    def test_list_indexing(self, standard_results):
        """List of indices indexing."""
        subset = standard_results[[0, 2, 4]]
        assert subset.n_genes == 3

    def test_subset_preserves_metadata(self, standard_results):
        """Gene subsetting should preserve obs, uns, model_config."""
        subset = standard_results[0:2]
        assert subset.obs is standard_results.obs
        assert subset.model_type == "nbdm"
        assert subset.n_cells == 10


# --------------------------------------------------------------------------
# Component subsetting
# --------------------------------------------------------------------------


class TestComponentSubsetting:
    """Test get_component / get_components on mixture models."""

    def test_single_component(self, mixture_results):
        """Extracting one component should drop mixture semantics."""
        comp0 = mixture_results.get_component(0)
        assert isinstance(comp0, ScribeMCMCResults)
        assert comp0.n_components is None
        assert comp0.model_type == "nbdm"
        # r should be 2D (n_samples, n_genes) after component squeeze
        assert comp0.samples["r"].ndim == 2

    def test_multi_component(self, mixture_results):
        """Extracting 2 of 3 components should reduce n_components."""
        comp = mixture_results.get_components([0, 2])
        assert isinstance(comp, ScribeMCMCResults)
        assert comp.n_components == 2
        assert comp.model_type == "nbdm_mix"

    def test_tuple_indexing(self, mixture_results):
        """Two-axis indexing: (genes, components)."""
        subset = mixture_results[0:3, 1]
        assert isinstance(subset, ScribeMCMCResults)
        assert subset.n_genes == 3
        assert subset.n_components is None


# --------------------------------------------------------------------------
# MCMC diagnostic delegation
# --------------------------------------------------------------------------


class TestMCMCDelegation:
    """Test print_summary and get_extra_fields delegation."""

    def test_print_summary_delegates(self, standard_results):
        """print_summary should call through to _mcmc."""
        standard_results.print_summary()
        standard_results._mcmc.print_summary.assert_called_once()

    def test_print_summary_raises_on_subset(self, standard_results):
        """print_summary on a gene subset should raise RuntimeError."""
        subset = standard_results[0:2]
        with pytest.raises(RuntimeError, match="print_summary"):
            subset.print_summary()

    def test_get_extra_fields_delegates(self, standard_results):
        """get_extra_fields should call through to _mcmc."""
        fields = standard_results.get_extra_fields()
        standard_results._mcmc.get_extra_fields.assert_called()
        assert isinstance(fields, dict)

    def test_get_extra_fields_empty_on_subset(self, standard_results):
        """get_extra_fields on a subset returns empty dict."""
        subset = standard_results[0:2]
        assert subset.get_extra_fields() == {}


# --------------------------------------------------------------------------
# Quantiles and MAP
# --------------------------------------------------------------------------


class TestQuantilesAndMAP:
    """Test get_posterior_quantiles and get_map."""

    def test_quantiles(self, standard_results):
        """Quantiles should return a dict keyed by quantile level."""
        q = standard_results.get_posterior_quantiles("p")
        assert 0.025 in q
        assert 0.5 in q
        assert 0.975 in q

    def test_map_without_potential_energy(self, standard_results):
        """MAP without potential energy should fall back to posterior mean."""
        map_est = standard_results.get_map()
        assert "p" in map_est
        assert "r" in map_est

    def test_map_with_potential_energy(self):
        """MAP should use potential_energy when available."""
        samples = _make_standard_samples(n_samples=10, n_genes=3)
        # Potential energy: minimum at index 5
        pe = jnp.arange(10, dtype=jnp.float32)
        pe = pe.at[5].set(-100.0)
        mcmc = _make_mock_mcmc(samples, extra_fields={"potential_energy": pe})
        results = ScribeMCMCResults.from_mcmc(
            mcmc=mcmc,
            n_cells=10,
            n_genes=3,
            model_type="nbdm",
            model_config=_make_model_config("nbdm"),
            prior_params={},
        )
        map_est = results.get_map()
        # Should pick sample index 5
        np.testing.assert_array_equal(map_est["p"], samples["p"][5])


# --------------------------------------------------------------------------
# MAP PPC sampling
# --------------------------------------------------------------------------


class TestMapPPCSampling:
    """Test MAP-based posterior predictive sampling helpers."""

    def test_get_map_ppc_samples_repeats_map_estimate(
        self, standard_results, monkeypatch
    ):
        """MAP PPC should build a repeated sample-axis pseudo-posterior."""
        captured = {}

        # Patch the low-level PPC generator so this test only validates
        # argument-shaping and storage behavior, not model sampling internals.
        def _fake_generate_ppc_samples(
            samples,
            model_type,
            n_cells,
            n_genes,
            model_config,
            rng_key=None,
            batch_size=None,
        ):
            captured["samples"] = samples
            captured["batch_size"] = batch_size
            return jnp.zeros((3, n_cells, n_genes), dtype=jnp.int32)

        monkeypatch.setattr(
            "scribe.mcmc._sampling._generate_ppc_samples",
            _fake_generate_ppc_samples,
        )

        generated = standard_results.get_map_ppc_samples(
            n_samples=3,
            cell_batch_size=4,
            store_samples=True,
            verbose=False,
        )

        assert generated.shape == (
            3,
            standard_results.n_cells,
            standard_results.n_genes,
        )
        assert standard_results.predictive_samples.shape == generated.shape
        assert captured["samples"]["p"].shape[0] == 3
        assert captured["samples"]["r"].shape[0] == 3
        assert captured["batch_size"] == 4

    def test_get_map_ppc_samples_validates_positive_sample_count(
        self, standard_results
    ):
        """MAP PPC should reject non-positive sample counts."""
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            standard_results.get_map_ppc_samples(n_samples=0)

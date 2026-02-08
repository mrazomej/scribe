"""
Tests for metadata-based gene subsetting.

Verifies that when param_specs are available, subsetting uses the correct
gene axis (avoiding ambiguity when two axes have the same size), and that
fallback behavior works when param_specs is empty.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from scribe.svi._gene_subsetting import build_gene_axis_by_key
from scribe.svi.results import ScribeSVIResults
from scribe.models.config import ModelConfig


# Minimal spec-like object for testing (name, shape_dims, is_gene_specific)
class _MockSpec:
    def __init__(self, name, shape_dims, is_gene_specific=True):
        self.name = name
        self.shape_dims = shape_dims
        self.is_gene_specific = is_gene_specific


class TestBuildGeneAxisByKey:
    """Tests for build_gene_axis_by_key helper."""

    def test_returns_none_when_param_specs_empty(self):
        params = {"r_loc": jnp.ones(5)}
        assert build_gene_axis_by_key([], params, 5) is None

    def test_gene_axis_from_shape_dims_1d(self):
        spec = _MockSpec("r", ("n_genes",))
        params = {"r_loc": jnp.ones(5)}
        out = build_gene_axis_by_key([spec], params, 5)
        assert out is not None
        assert out["r_loc"] == 0

    def test_gene_axis_from_shape_dims_2d_gene_last(self):
        """When shape_dims is (n_components, n_genes), gene axis is 1."""
        spec = _MockSpec("r", ("n_components", "n_genes"))
        params = {"r_loc": jnp.ones((3, 5))}
        out = build_gene_axis_by_key([spec], params, 5)
        assert out is not None
        assert out["r_loc"] == 1

    def test_ambiguous_shape_uses_metadata_axis(self):
        """When two axes have same size (5, 5), metadata gives gene_axis=1."""
        spec = _MockSpec("r", ("n_components", "n_genes"))
        params = {"r_loc": jnp.ones((5, 5))}  # n_components=n_genes=5
        out = build_gene_axis_by_key([spec], params, 5)
        assert out is not None
        assert out["r_loc"] == 1

    def test_matches_prefix_log_spec_name(self):
        spec = _MockSpec("mu", ("n_genes",))
        params = {"log_mu_loc": jnp.ones(5)}
        out = build_gene_axis_by_key([spec], params, 5)
        assert out is not None
        assert out["log_mu_loc"] == 0

    def test_skips_keys_with_dollar(self):
        spec = _MockSpec("r", ("n_genes",))
        params = {"r_loc": jnp.ones(5), "amortizer$params": {}}
        out = build_gene_axis_by_key([spec], params, 5)
        assert "amortizer$params" not in (out or {})


class TestMetadataSubsettingCorrectAxis:
    """Test that subsetting uses the correct axis when metadata is present."""

    @pytest.fixture
    def model_config_with_param_specs(self):
        from scribe.models.builders.parameter_specs import LogNormalSpec
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_components", "n_genes"),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        return ModelConfig(
            base_model="nbdm",
            parameterization="standard",
            unconstrained=False,
            param_specs=[spec],
        )

    def test_subset_uses_gene_axis_from_metadata(self, model_config_with_param_specs):
        """With (n_components, n_genes) = (5, 5), subsetting must use axis 1, not 0."""
        n_genes = 5
        # Param with shape (5, 5) - ambiguous without metadata
        params = {"r_loc": jnp.arange(25).reshape(5, 5).astype(float)}
        gene_axis_by_key = build_gene_axis_by_key(
            model_config_with_param_specs.param_specs, params, n_genes
        )
        assert gene_axis_by_key is not None
        assert gene_axis_by_key["r_loc"] == 1

        # Build a minimal results-like object with _gene_axis_by_key set
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=10,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=model_config_with_param_specs,
            prior_params={},
            _gene_axis_by_key=gene_axis_by_key,
        )
        index = jnp.array([True, False, False, False, False])  # first gene only
        new_params = results._subset_params(params, index)
        subset = new_params["r_loc"]
        # Subset along axis 1 (genes): shape should be (5, 1), not (1, 5)
        assert subset.shape == (5, 1)
        np.testing.assert_array_almost_equal(subset[:, 0], params["r_loc"][:, 0])

    def test_subset_posterior_samples_uses_metadata(self, model_config_with_param_specs):
        """Posterior samples with (n_samples, n_components, n_genes) use gene axis 2."""
        from scribe.models.builders.parameter_specs import LogNormalSpec
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_components", "n_genes"),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        n_genes = 4
        samples = {"r": jnp.ones((10, 3, n_genes))}  # (n_samples, n_components, n_genes)
        gene_axis_by_key = build_gene_axis_by_key([spec], {"r": samples["r"]}, n_genes)
        assert gene_axis_by_key is not None
        assert gene_axis_by_key["r"] == 2

        results = ScribeSVIResults(
            params={},
            loss_history=jnp.array([1.0]),
            n_cells=10,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=model_config_with_param_specs,
            prior_params={},
            _gene_axis_by_key=gene_axis_by_key,
        )
        index = jnp.array([True, False, True, False])
        new_samples = results._subset_posterior_samples(samples, index)
        assert new_samples["r"].shape == (10, 3, 2)


class TestFallbackWhenParamSpecsEmpty:
    """Test that subsetting still runs (heuristic) when param_specs is empty."""

    def test_subset_params_fallback_no_raise(self):
        """When _gene_axis_by_key is None, shape-based heuristic runs and does not raise."""
        config = ModelConfig(base_model="nbdm", parameterization="standard", unconstrained=False)
        assert getattr(config, "param_specs", None) == [] or config.param_specs == []
        params = {"r_loc": jnp.ones(5)}
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=10,
            n_genes=5,
            model_type="nbdm",
            model_config=config,
            prior_params={},
            _gene_axis_by_key=None,
        )
        index = jnp.array([True, False, True, False, False])
        new_params = results._subset_params(params, index)
        assert "r_loc" in new_params
        assert new_params["r_loc"].shape == (2,)

    def test_subset_posterior_samples_fallback_no_raise(self):
        """When _gene_axis_by_key is None, last-axis heuristic runs and does not raise."""
        config = ModelConfig(base_model="nbdm", parameterization="standard", unconstrained=False)
        samples = {"r": jnp.ones((10, 5))}
        results = ScribeSVIResults(
            params={},
            loss_history=jnp.array([1.0]),
            n_cells=10,
            n_genes=5,
            model_type="nbdm",
            model_config=config,
            prior_params={},
            _gene_axis_by_key=None,
        )
        index = jnp.array([True, False, True, False, False])
        new_samples = results._subset_posterior_samples(samples, index)
        assert new_samples["r"].shape == (10, 2)

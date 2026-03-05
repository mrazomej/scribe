"""
Tests for metadata-based gene subsetting.

Verifies that when param_specs are available, subsetting uses the correct
gene axis (avoiding ambiguity when two axes have the same size), and that
fallback behavior works when param_specs is empty.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import pandas as pd

from scribe.svi._gene_subsetting import build_gene_axis_by_key
from scribe.svi.results import ScribeSVIResults
from scribe.models.config import ModelConfig
from scribe.models.builders.parameter_specs import LogNormalSpec, BetaSpec


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


class TestConcatGeneAlignment:
    """Test gene-alignment behavior used by SVI result concatenation."""

    def _make_concat_ready_result(self, var_index, r_values, p_capture_values):
        """Construct a minimal SVI result with gene- and cell-specific params."""
        r_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            unconstrained=True,
        )
        capture_spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            unconstrained=True,
        )
        config = ModelConfig(
            base_model="nbvcp",
            unconstrained=True,
            param_specs=[r_spec, capture_spec],
        )
        return ScribeSVIResults(
            params={
                "r_loc": jnp.array(r_values, dtype=jnp.float32),
                "p_capture_loc": jnp.array(p_capture_values, dtype=jnp.float32),
            },
            loss_history=jnp.array([1.0], dtype=jnp.float32),
            n_cells=len(p_capture_values),
            n_genes=len(var_index),
            model_type="nbvcp",
            model_config=config,
            prior_params={},
            var=pd.DataFrame(index=var_index),
            n_vars=len(var_index),
        )

    def test_concat_reorders_genes_using_var_index(self):
        """Concat should reorder gene axes when ``var.index`` content matches."""
        res_a = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1, 0.2],
        )
        res_b = self._make_concat_ready_result(
            var_index=["g3", "g1", "g2"],
            r_values=[30.0, 10.0, 20.0],
            p_capture_values=[0.3, 0.4, 0.5],
        )

        combined = ScribeSVIResults.concat(
            [res_a, res_b], align_genes="strict"
        )
        assert combined.n_cells == 5
        assert list(combined.var.index) == ["g1", "g2", "g3"]
        # After gene alignment both results have identical r_loc; promotion
        # stacks them along a new dataset axis (2, 3).
        np.testing.assert_allclose(
            combined.params["r_loc"],
            jnp.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
        )
        np.testing.assert_allclose(
            combined.params["p_capture_loc"],
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )

    def test_concat_rejects_different_gene_sets(self):
        """Concat should fail when gene sets differ across inputs."""
        res_a = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1],
        )
        res_b = self._make_concat_ready_result(
            var_index=["g1", "g2", "g4"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.2],
        )

        with pytest.raises(ValueError, match="Gene set mismatch"):
            ScribeSVIResults.concat([res_a, res_b], align_genes="strict")

    def test_concat_var_only_skips_shared_tensor_equality(self):
        """Fast validation should trust shared non-cell-specific tensors."""
        res_a = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1, 0.2],
        )
        res_b = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[100.0, 200.0, 300.0],  # Intentionally different.
            p_capture_values=[0.3],
        )

        with pytest.raises(ValueError, match="Non-cell-specific parameter"):
            ScribeSVIResults.concat([res_a, res_b], validation="strict")

        # In var_only mode, non-cell-specific params are stacked along a
        # dataset axis; cell-specific p_capture_loc is concatenated.
        combined = ScribeSVIResults.concat([res_a, res_b], validation="var_only")
        np.testing.assert_allclose(
            combined.params["r_loc"],
            jnp.array([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]),
        )
        np.testing.assert_allclose(
            combined.params["p_capture_loc"],
            jnp.array([0.1, 0.2, 0.3]),
        )

    def test_concat_assume_aligned_skips_gene_validation(self):
        """Trusted align mode should avoid gene-set/order checks."""
        res_a = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1],
        )
        # Deliberately different gene order/content; trusted mode bypasses checks.
        res_b = self._make_concat_ready_result(
            var_index=["x1", "x2", "x3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.2],
        )
        combined = ScribeSVIResults.concat(
            [res_a, res_b],
            validation="var_only",
            align_genes="assume_aligned",
        )
        assert combined.n_cells == 2

    def test_concat_rejects_single_object_argument(self):
        """Passing a single result instead of a list should fail fast."""
        res = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1],
        )
        with pytest.raises(TypeError, match="sequence of results"):
            ScribeSVIResults.concat(res)

    def test_concat_promotes_to_multi_dataset(self):
        """Concatenating single-dataset results should synthesize dataset metadata."""
        res_a = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1, 0.2],
        )
        res_b = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[100.0, 200.0, 300.0],
            p_capture_values=[0.3, 0.4, 0.5],
        )

        combined = ScribeSVIResults.concat(
            [res_a, res_b], validation="var_only"
        )

        # model_config should now be multi-dataset
        assert combined.model_config.n_datasets == 2

        # _n_cells_per_dataset tracks each input's cell count
        np.testing.assert_array_equal(
            combined._n_cells_per_dataset, jnp.array([2, 3])
        )

        # _dataset_indices assigns cells to their source result
        np.testing.assert_array_equal(
            combined._dataset_indices,
            jnp.array([0, 0, 1, 1, 1]),
        )

        # Gene-specific param should be stacked along dataset axis
        assert combined.params["r_loc"].shape == (2, 3)
        assert "r_loc" in combined._promoted_dataset_keys

    def test_concat_promotes_enables_get_dataset(self):
        """After promotion, ``get_dataset(i)`` should recover per-dataset values."""
        res_a = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1, 0.2],
        )
        res_b = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[100.0, 200.0, 300.0],
            p_capture_values=[0.3, 0.4, 0.5],
        )

        combined = ScribeSVIResults.concat(
            [res_a, res_b], validation="var_only"
        )

        ds0 = combined.get_dataset(0)
        ds1 = combined.get_dataset(1)

        # Cell-specific param should be sliced to correct cells
        assert ds0.n_cells == 2
        assert ds1.n_cells == 3
        np.testing.assert_allclose(
            ds0.params["p_capture_loc"], jnp.array([0.1, 0.2])
        )
        np.testing.assert_allclose(
            ds1.params["p_capture_loc"], jnp.array([0.3, 0.4, 0.5])
        )

        # Gene-specific param should recover each input's original values
        np.testing.assert_allclose(
            ds0.params["r_loc"], jnp.array([10.0, 20.0, 30.0])
        )
        np.testing.assert_allclose(
            ds1.params["r_loc"], jnp.array([100.0, 200.0, 300.0])
        )

    def test_concat_rejects_single_element_list(self):
        """A single-element list is disallowed to prevent the instance-method footgun."""
        res = self._make_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
            p_capture_values=[0.1],
        )
        with pytest.raises(ValueError, match="at least two elements"):
            ScribeSVIResults.concat([res])

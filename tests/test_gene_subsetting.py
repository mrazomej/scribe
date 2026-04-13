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

    def test_detects_joint_prefixed_keys_from_metadata(self):
        """Joint guide keys should be mapped with metadata-derived gene axes."""
        spec = _MockSpec("mu", ("n_components", "n_genes"))
        params = {
            # Joint variational params mirror LowRankMVN tensors and should
            # follow the same axis mapping semantics as non-joint keys.
            "joint_joint_mu_loc": jnp.ones((3, 5)),
            "joint_joint_mu_W": jnp.ones((3, 5, 2)),
        }
        out = build_gene_axis_by_key([spec], params, 5)
        assert out is not None
        assert out["joint_joint_mu_loc"] == 1
        assert out["joint_joint_mu_W"] == 1


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

    def test_subset_uses_gene_axis_from_metadata(
        self, model_config_with_param_specs
    ):
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
        np.testing.assert_array_almost_equal(
            subset[:, 0], params["r_loc"][:, 0]
        )

    def test_subset_posterior_samples_uses_metadata(
        self, model_config_with_param_specs
    ):
        """Posterior samples with (n_samples, n_components, n_genes) use gene axis 2."""
        from scribe.models.builders.parameter_specs import LogNormalSpec

        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_components", "n_genes"),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        n_genes = 4
        samples = {
            "r": jnp.ones((10, 3, n_genes))
        }  # (n_samples, n_components, n_genes)
        gene_axis_by_key = build_gene_axis_by_key(
            [spec], {"r": samples["r"]}, n_genes
        )
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

    def test_subset_joint_params_uses_gene_axis_from_metadata(
        self, model_config_with_param_specs
    ):
        """Joint keys with ambiguous shapes must subset the gene axis, not axis 0."""
        n_genes = 5
        params = {
            # Ambiguous (5, 5): metadata says gene axis is 1 for (K, G).
            "joint_joint_r_loc": jnp.arange(25).reshape(5, 5).astype(float),
            "joint_joint_r_W": jnp.arange(50).reshape(5, 5, 2).astype(float),
        }
        gene_axis_by_key = build_gene_axis_by_key(
            model_config_with_param_specs.param_specs, params, n_genes
        )
        assert gene_axis_by_key is not None
        assert gene_axis_by_key["joint_joint_r_loc"] == 1
        assert gene_axis_by_key["joint_joint_r_W"] == 1

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
        index = jnp.array([True, False, False, False, False])
        new_params = results._subset_params(params, index)

        assert new_params["joint_joint_r_loc"].shape == (5, 1)
        assert new_params["joint_joint_r_W"].shape == (5, 1, 2)
        np.testing.assert_array_almost_equal(
            new_params["joint_joint_r_loc"][:, 0],
            params["joint_joint_r_loc"][:, 0],
        )


class TestIntegerArrayIndexingPreservesOrder:
    """Integer-array indexing must preserve the caller-specified gene order.

    Before the fix, ``__getitem__`` converted integer arrays to a boolean mask
    via ``jnp.isin``, which always returned genes in their original list order
    and silently discarded the requested ordering.  These tests assert the
    corrected behaviour.
    """

    @pytest.fixture
    def ordered_results(self):
        """Results object with five genes whose r_loc values equal their index."""
        r_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        config = ModelConfig(
            base_model="nbdm",
            parameterization="standard",
            unconstrained=False,
            param_specs=[r_spec],
        )
        return ScribeSVIResults(
            params={"r_loc": jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])},
            loss_history=jnp.array([1.0]),
            n_cells=6,
            n_genes=5,
            model_type="nbdm",
            model_config=config,
            prior_params={},
            var=pd.DataFrame(index=["g0", "g1", "g2", "g3", "g4"]),
        )

    def test_integer_array_preserves_param_order(self, ordered_results):
        """results[[4, 1, 3]] should yield r_loc=[50, 20, 40], not [20, 40, 50]."""
        subset = ordered_results[[4, 1, 3]]
        np.testing.assert_array_equal(
            subset.params["r_loc"], jnp.array([50.0, 20.0, 40.0])
        )

    def test_integer_array_preserves_var_index_order(self, ordered_results):
        """var.index of the subset must follow the caller's specified gene order."""
        subset = ordered_results[[4, 1, 3]]
        assert list(subset.var.index) == ["g4", "g1", "g3"]

    def test_integer_array_n_genes_is_element_count(self, ordered_results):
        """n_genes must equal the number of selected genes, not the sum of indices."""
        subset = ordered_results[[4, 1, 3]]
        # Before the fix, index.sum() = 4+1+3 = 8; correct value is 3.
        assert subset.n_genes == 3

    def test_numpy_integer_array_preserves_order(self, ordered_results):
        """numpy integer arrays should behave identically to Python lists."""
        subset = ordered_results[np.array([4, 1, 3])]
        np.testing.assert_array_equal(
            subset.params["r_loc"], jnp.array([50.0, 20.0, 40.0])
        )
        assert list(subset.var.index) == ["g4", "g1", "g3"]

    def test_boolean_mask_still_preserves_original_order(self, ordered_results):
        """Boolean-mask indexing must continue to select genes in original order."""
        mask = jnp.array([False, True, False, True, True])
        subset = ordered_results[mask]
        # Boolean mask selects g1, g3, g4 — always in original list order.
        np.testing.assert_array_equal(
            subset.params["r_loc"], jnp.array([20.0, 40.0, 50.0])
        )
        assert list(subset.var.index) == ["g1", "g3", "g4"]


class TestFallbackWhenParamSpecsEmpty:
    """Test that subsetting still runs (heuristic) when param_specs is empty."""

    def test_subset_params_fallback_no_raise(self):
        """When _gene_axis_by_key is None, shape-based heuristic runs and does not raise."""
        config = ModelConfig(
            base_model="nbdm", parameterization="standard", unconstrained=False
        )
        assert (
            getattr(config, "param_specs", None) == []
            or config.param_specs == []
        )
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
        config = ModelConfig(
            base_model="nbdm", parameterization="standard", unconstrained=False
        )
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

    def _make_joint_concat_ready_result(self, var_index, r_values):
        """Construct a minimal SVI result with joint-guide r variational params."""
        r_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            unconstrained=True,
        )
        config = ModelConfig(
            base_model="nbdm",
            unconstrained=True,
            param_specs=[r_spec],
            joint_params=["r"],
        )
        return ScribeSVIResults(
            params={
                # Joint-key tensors must be reordered by strict concat when
                # var indices are permuted across result objects.
                "joint_joint_r_loc": jnp.array(r_values, dtype=jnp.float32),
                "joint_joint_r_W": jnp.ones(
                    (len(var_index), 2), dtype=jnp.float32
                ),
                "joint_joint_r_raw_diag": jnp.zeros(
                    len(var_index), dtype=jnp.float32
                ),
            },
            loss_history=jnp.array([1.0], dtype=jnp.float32),
            n_cells=2,
            n_genes=len(var_index),
            model_type="nbdm",
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

        combined = ScribeSVIResults.concat([res_a, res_b], align_genes="strict")
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

    def test_concat_reorders_joint_params_using_var_index(self):
        """Strict concat should reorder joint-guide gene axes from var metadata."""
        res_a = self._make_joint_concat_ready_result(
            var_index=["g1", "g2", "g3"],
            r_values=[10.0, 20.0, 30.0],
        )
        res_b = self._make_joint_concat_ready_result(
            var_index=["g3", "g1", "g2"],
            r_values=[30.0, 10.0, 20.0],
        )

        combined = ScribeSVIResults.concat(
            [res_a, res_b], align_genes="strict", validation="var_only"
        )
        assert list(combined.var.index) == ["g1", "g2", "g3"]

        # Promotion stacks non-cell params along a dataset axis after alignment.
        np.testing.assert_allclose(
            combined.params["joint_joint_r_loc"],
            jnp.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]),
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
        combined = ScribeSVIResults.concat(
            [res_a, res_b], validation="var_only"
        )
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


# ===========================================================================
# Flow-guided gene subsetting
# ===========================================================================


class TestFlowGuidedParamNames:
    """Tests for _flow_guided_param_names in posterior.py."""

    def test_detects_per_param_flow(self):
        from scribe.models.builders.posterior import _flow_guided_param_names

        params = {
            "flow_r$params": {"layer_0": {}},
            "eta_capture_loc": jnp.zeros(10),
        }
        assert _flow_guided_param_names(params) == {"r"}

    def test_detects_joint_flow(self):
        from scribe.models.builders.posterior import _flow_guided_param_names

        params = {
            "joint_flow_default_r$params": {"layer_0": {}},
            "joint_flow_default_p$params": {"layer_0": {}},
        }
        assert _flow_guided_param_names(params) == {"r", "p"}

    def test_detects_independent_mixture_flow(self):
        from scribe.models.builders.posterior import _flow_guided_param_names

        params = {
            "flow_r_idx0$params": {"layer_0": {}},
            "flow_r_idx1$params": {"layer_0": {}},
        }
        assert _flow_guided_param_names(params) == {"r"}

    def test_ignores_non_flow_keys(self):
        from scribe.models.builders.posterior import _flow_guided_param_names

        params = {
            "r_loc": jnp.zeros(5),
            "r_scale": jnp.ones(5),
            "amortizer$params": {"encoder": {}},
        }
        assert _flow_guided_param_names(params) == set()


class TestFlowGeneSubsetting:
    """Gene subsetting preserves flow params and stores _subset_gene_index."""

    @staticmethod
    def _make_flow_results(n_genes=20):
        """Create minimal SVI results with a flow param key."""
        config = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
        )
        params = {
            # Flow param (nested dict — not gene-subsettable)
            "joint_flow_default_r$params": {
                "layer_0": {"hidden_0": {"kernel": jnp.ones((10, 64))}}
            },
            "joint_flow_default_p$params": {
                "layer_0": {"hidden_0": {"kernel": jnp.ones((10, 64))}}
            },
            # Cell-specific param (not gene-indexed)
            "eta_capture_loc": jnp.zeros(50),
        }
        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=50,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )

    def test_subset_stores_gene_index(self):
        """__getitem__ stores the gene index on the subsetted result."""
        results = self._make_flow_results(n_genes=20)
        subset = results[np.array([2, 5, 10])]

        assert subset._subset_gene_index is not None
        np.testing.assert_array_equal(
            subset._subset_gene_index, np.array([2, 5, 10])
        )

    def test_subset_preserves_flow_params_intact(self):
        """Flow params (nested dicts) survive gene subsetting unchanged."""
        results = self._make_flow_results(n_genes=20)
        subset = results[np.array([2, 5, 10])]

        # Flow params must be the same objects (not subsetted)
        assert "joint_flow_default_r$params" in subset.params
        assert isinstance(subset.params["joint_flow_default_r$params"], dict)

    def test_subset_n_genes_reflects_selection(self):
        """n_genes on the subset reflects the selected count."""
        results = self._make_flow_results(n_genes=20)
        subset = results[np.array([2, 5, 10])]

        assert subset.n_genes == 3
        assert subset._original_n_genes == 20

    def test_resubset_composes_indices(self):
        """Re-subsetting composes gene indices relative to the original."""
        results = self._make_flow_results(n_genes=20)
        sub1 = results[np.array([2, 5, 10, 15])]
        sub2 = sub1[np.array([1, 3])]

        # sub2 indices should be [5, 15] in original space
        np.testing.assert_array_equal(
            sub2._subset_gene_index, np.array([5, 15])
        )
        assert sub2.n_genes == 2
        assert sub2._original_n_genes == 20


class TestFlowGeneSubsettingOriginalParams:
    """Gene subsetting preserves _original_params for joint flow + nondense."""

    @staticmethod
    def _make_joint_flow_results(n_genes=100):
        """Create SVI results with joint flow (dense r) + nondense p."""
        config = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
        )
        params = {
            # Flow param for dense spec r (nested dict — not subsetted)
            "joint_flow_joint_r$params": {
                "layer_0": {"hidden_0": {"kernel": jnp.ones((n_genes, 64))}}
            },
            # Nondense array params for p (gene-dimensional arrays)
            "joint_flow_joint_p_loc": jnp.zeros(n_genes),
            "joint_flow_joint_p_raw_diag": -3.0 * jnp.ones(n_genes),
            "joint_flow_joint_p_alpha_r": jnp.zeros(n_genes),
        }
        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=50,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )

    def test_subset_stores_original_params(self):
        """Subsetting flow results stores the original unsubsetted params."""
        results = self._make_joint_flow_results(n_genes=100)
        subset = results[np.array([2, 5, 10])]

        assert hasattr(subset, "_original_params")
        assert subset._original_params is results.params

    def test_original_params_has_full_dimension(self):
        """_original_params retains full-dimension nondense arrays."""
        n_genes = 100
        results = self._make_joint_flow_results(n_genes=n_genes)
        idx = np.array([2, 5, 10])
        subset = results[idx]

        # Subsetted params should be sliced
        assert subset.params["joint_flow_joint_p_loc"].shape == (3,)

        # Original params should be full-dimensional
        orig = subset._original_params
        assert orig["joint_flow_joint_p_loc"].shape == (n_genes,)
        assert orig["joint_flow_joint_p_alpha_r"].shape == (n_genes,)

    def test_resubset_preserves_original_params(self):
        """Re-subsetting carries over the first original params reference."""
        n_genes = 100
        results = self._make_joint_flow_results(n_genes=n_genes)
        sub1 = results[np.array([2, 5, 10, 15, 20])]
        sub2 = sub1[np.array([1, 3])]

        # sub2's _original_params should be the same as sub1's
        assert sub2._original_params is results.params
        assert sub2._original_params["joint_flow_joint_p_loc"].shape == (
            n_genes,
        )

    def test_no_original_params_without_flow(self):
        """Results without flow params do not store _original_params."""
        config = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
        )
        params = {
            "r_loc": jnp.zeros(20),
            "p_loc": jnp.zeros(20),
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=50,
            n_genes=20,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        subset = results[np.array([0, 5, 10])]
        assert not hasattr(subset, "_original_params")


class TestHierarchicalFlowSkip:
    """Verify that flow-guided params with hierarchical priors don't crash.

    When ``prob_prior=gaussian`` and ``p`` is flow-guided, Pass 2
    (_apply_gene_level_hierarchy) must still build the hyperparameter
    posteriors (logit_p_loc, logit_p_scale) but skip the gene-level
    ``p`` posterior, deferring it to Pass 10.
    """

    def test_hierarchical_p_with_flow_guide_succeeds(self):
        """get_posterior_distributions succeeds for hierarchical + flow p."""
        from scribe.models.builders.posterior import get_posterior_distributions
        from scribe.models.config.groups import GuideFamilyConfig
        from scribe.models.components.guide_families import (
            JointNormalizingFlowGuide,
        )

        guide = JointNormalizingFlowGuide(
            group="joint",
            flow_type="affine_coupling",
            num_layers=2,
        )
        config = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
            prob_prior="gaussian",
            guide_families=GuideFamilyConfig(r=guide, p=guide),
        )

        # Minimal params: flow keys for r and p, mean-field hypers
        params = {
            "joint_flow_joint_r$params": {
                "layer_0": {"hidden_0": {"kernel": jnp.ones((10, 64))}}
            },
            "joint_flow_joint_p$params": {
                "layer_0": {"hidden_0": {"kernel": jnp.ones((10, 64))}}
            },
            "logit_p_loc_loc": jnp.array(0.0),
            "logit_p_loc_scale": jnp.array(1.0),
            "logit_p_scale_loc": jnp.array(0.0),
            "logit_p_scale_scale": jnp.array(1.0),
        }

        dists = get_posterior_distributions(params, config)

        # Hyperparameters must be present
        assert "logit_p_loc" in dists
        assert "logit_p_scale" in dists
        # Gene-level p should NOT be built by Pass 2 (no p_loc key);
        # the call itself must not raise a KeyError.

    def test_hierarchical_mu_with_flow_guide_succeeds(self):
        """get_posterior_distributions succeeds for hierarchical + flow r."""
        from scribe.models.builders.posterior import get_posterior_distributions
        from scribe.models.config.groups import GuideFamilyConfig
        from scribe.models.components.guide_families import (
            JointNormalizingFlowGuide,
        )

        guide = JointNormalizingFlowGuide(
            group="joint",
            flow_type="affine_coupling",
            num_layers=2,
        )
        config = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
            expression_prior="gaussian",
            n_components=2,
            guide_families=GuideFamilyConfig(r=guide),
        )

        # r is flow-guided, log_r_loc/log_r_scale are mean-field hypers
        params = {
            "joint_flow_joint_r$params": {
                "layer_0": {"hidden_0": {"kernel": jnp.ones((10, 64))}}
            },
            "p_loc": jnp.zeros(5),
            "p_scale": jnp.ones(5),
            "log_r_loc_loc": jnp.array(0.0),
            "log_r_loc_scale": jnp.array(1.0),
            "log_r_scale_loc": jnp.array(0.0),
            "log_r_scale_scale": jnp.array(1.0),
        }

        dists = get_posterior_distributions(params, config)

        # Hyperparameters must be present
        assert "log_r_loc" in dists
        assert "log_r_scale" in dists


class TestFlowSamplingHelpers:
    """Tests for _has_flow_params and _subset_gene_dim_samples."""

    def test_has_flow_params_true(self):
        from scribe.svi._sampling_posterior_predictive import _has_flow_params

        assert _has_flow_params({"flow_r$params": {}})
        assert _has_flow_params({"joint_flow_default_r$params": {}})

    def test_has_flow_params_false(self):
        from scribe.svi._sampling_posterior_predictive import _has_flow_params

        assert not _has_flow_params({"r_loc": jnp.zeros(5)})
        assert not _has_flow_params({"amortizer$params": {}})

    def test_subset_gene_dim_samples(self):
        from scribe.svi._sampling_posterior_predictive import (
            _subset_gene_dim_samples,
        )

        samples = {
            "r": jnp.arange(20).reshape(2, 10),  # (n_samples, n_genes)
            "scalar": jnp.array([1.0, 2.0]),  # (n_samples,)
        }
        idx = np.array([0, 3, 7])
        out = _subset_gene_dim_samples(samples, idx, original_n_genes=10)

        assert out["r"].shape == (2, 3)
        np.testing.assert_array_equal(out["r"][0], jnp.array([0, 3, 7]))
        # Scalar should be unchanged
        np.testing.assert_array_equal(out["scalar"], samples["scalar"])


# =========================================================================
# AxisLayout gene-axis equivalence tests
# =========================================================================

from scribe.core.axis_layout import (
    layout_from_param_spec,
    build_param_layouts,
    GENES,
)


class TestAxisLayoutGeneAxisEquivalence:
    """Verify layout.gene_axis agrees with build_gene_axis_by_key."""

    def test_1d_gene_param(self):
        """Gene-specific 1D param: layout gene_axis should be 0."""
        spec = _MockSpec("r", ("n_genes",))
        params = {"r_loc": jnp.ones(5)}
        old = build_gene_axis_by_key([spec], params, 5)
        assert old is not None

        real_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        layouts = build_param_layouts([real_spec], params)
        assert layouts["r_loc"].gene_axis == old["r_loc"]

    def test_2d_mixture_gene_param(self):
        """(n_components, n_genes) param: layout gene_axis should be 1."""
        real_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_mixture=True,
            is_gene_specific=True,
        )
        params = {"r_loc": jnp.ones((3, 5))}
        old = build_gene_axis_by_key([real_spec], params, 5)
        assert old is not None

        layouts = build_param_layouts([real_spec], params)
        assert layouts["r_loc"].gene_axis == old["r_loc"]

    def test_ambiguous_shape_5x5_layout_is_correct(self):
        """Ambiguous (5,5) shape: layout correctly identifies gene axis as 1.

        ``build_gene_axis_by_key`` falls back to ``shape.index(n_genes)``
        which returns 0 (wrong) because ``shape_dims`` is ``("n_genes",)``
        and doesn't encode the mixture axis.  ``layout_from_param_spec``
        correctly reads ``is_mixture=True`` and places genes at axis 1.
        This is an improvement over the old heuristic.
        """
        real_spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_mixture=True,
            is_gene_specific=True,
        )
        params = {"r_loc": jnp.ones((5, 5))}
        layouts = build_param_layouts([real_spec], params)
        # AxisLayout knows the correct axis via is_mixture
        assert layouts["r_loc"].gene_axis == 1

    def test_scalar_param_no_gene_axis(self):
        """Scalar param: no gene axis in either system."""
        real_spec = BetaSpec(
            name="p", shape_dims=(), default_params=(1.0, 1.0)
        )
        params = {"p_loc": jnp.array(0.5)}
        old = build_gene_axis_by_key([real_spec], params, 5)

        layouts = build_param_layouts([real_spec], params)
        assert layouts["p_loc"].gene_axis is None

    def test_log_prefixed_key(self):
        """log_mu_loc should still get correct gene axis."""
        real_spec = LogNormalSpec(
            name="mu",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        params = {"log_mu_loc": jnp.ones(5)}
        old = build_gene_axis_by_key([real_spec], params, 5)
        assert old is not None

        layouts = build_param_layouts([real_spec], params)
        assert layouts["log_mu_loc"].gene_axis == old["log_mu_loc"]

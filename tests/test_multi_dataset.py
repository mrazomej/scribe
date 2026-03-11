"""Tests for multi-dataset hierarchical modeling.

Covers:
- DatasetHierarchical*Spec creation and sampling
- resolve_shape with is_dataset flag
- ModelConfig with dataset fields and validation
- index_dataset_params likelihood helper
- DatasetMixin.get_dataset and 3-axis __getitem__ on results
- compare_datasets DE helper
- Integration: create_model end-to-end with multi-dataset configs
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from jax import random

from scribe.models.builders.parameter_specs import (
    DatasetHierarchicalExpNormalSpec,
    DatasetHierarchicalSigmoidNormalSpec,
    ExpNormalSpec,
    SigmoidNormalSpec,
    resolve_shape,
)
from scribe.models.config import ModelConfig, ModelConfigBuilder
from scribe.models.components.likelihoods.base import (
    broadcast_p_for_mixture,
    index_dataset_params,
)
from scribe.models.presets.factory import create_model


# ==============================================================================
# resolve_shape with is_dataset
# ==============================================================================


class TestResolveShapeDataset:
    """Test that resolve_shape prepends n_datasets when is_dataset=True."""

    def test_gene_specific_dataset(self):
        """Gene-specific dataset param: (n_datasets, n_genes)."""
        shape = resolve_shape(
            shape_dims=("n_genes",),
            dims={"n_genes": 100, "n_datasets": 3},
            is_dataset=True,
        )
        assert shape == (3, 100)

    def test_global_dataset(self):
        """Global (scalar per dataset) param: (n_datasets,)."""
        shape = resolve_shape(
            shape_dims=(),
            dims={"n_genes": 100, "n_datasets": 3},
            is_dataset=True,
        )
        assert shape == (3,)

    def test_no_dataset(self):
        """Without is_dataset, shape is unchanged."""
        shape = resolve_shape(
            shape_dims=("n_genes",),
            dims={"n_genes": 100, "n_datasets": 3},
            is_dataset=False,
        )
        assert shape == (100,)

    def test_dataset_and_mixture(self):
        """Mixture + dataset param: (n_components, n_datasets, n_genes)."""
        shape = resolve_shape(
            shape_dims=("n_genes",),
            dims={"n_genes": 50, "n_datasets": 2, "n_components": 3},
            is_mixture=True,
            is_dataset=True,
        )
        assert shape == (3, 2, 50)


# ==============================================================================
# DatasetHierarchical*Spec
# ==============================================================================


class TestDatasetHierarchicalSpecs:
    """Test DatasetHierarchicalExpNormalSpec and SigmoidNormal creation."""

    def test_exp_normal_creation(self):
        """DatasetHierarchicalExpNormalSpec with correct fields."""
        spec = DatasetHierarchicalExpNormalSpec(
            name="mu",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_dataset=True,
            loc=0.0,
            scale=1.0,
            hyper_loc_name="log_mu_dataset_loc",
            hyper_scale_name="log_mu_dataset_scale",
        )
        assert spec.name == "mu"
        assert spec.is_dataset is True
        assert spec.is_gene_specific is True
        assert spec.hyper_loc_name == "log_mu_dataset_loc"

    def test_sigmoid_normal_creation(self):
        """DatasetHierarchicalSigmoidNormalSpec with correct fields."""
        spec = DatasetHierarchicalSigmoidNormalSpec(
            name="p",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_dataset=True,
            loc=0.0,
            scale=1.0,
            hyper_loc_name="logit_p_dataset_loc",
            hyper_scale_name="logit_p_dataset_scale",
        )
        assert spec.name == "p"
        assert spec.is_dataset is True

    def test_sample_hierarchical(self):
        """sample_hierarchical produces correctly shaped output."""
        spec = DatasetHierarchicalExpNormalSpec(
            name="mu",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_dataset=True,
            loc=0.0,
            scale=1.0,
            hyper_loc_name="log_mu_dataset_loc",
            hyper_scale_name="log_mu_dataset_scale",
        )

        dims = {"n_genes": 50, "n_datasets": 3}

        # Pre-sampled hyperparameters
        hyper_values = {
            "log_mu_dataset_loc": jnp.zeros(50),
            "log_mu_dataset_scale": jnp.ones(50),
        }

        with numpyro.handlers.seed(rng_seed=0):
            result = spec.sample_hierarchical(dims, hyper_values)

        # Should be (n_datasets, n_genes) = (3, 50)
        assert result.shape == (3, 50)
        # ExpTransform: all values should be positive
        assert jnp.all(result > 0)


# ==============================================================================
# ModelConfig dataset fields and validation
# ==============================================================================


class TestModelConfigDataset:
    """Test dataset-related ModelConfig fields and validation."""

    def test_default_values(self):
        """n_datasets defaults to None, hierarchical_dataset_* to defaults."""
        config = ModelConfig(base_model="nbdm")
        assert config.n_datasets is None
        assert config.hierarchical_dataset_mu is False
        assert config.hierarchical_dataset_p == "none"

    def test_is_multi_dataset_property(self):
        """is_multi_dataset is True only when n_datasets >= 2."""
        config = ModelConfig(
            base_model="nbdm", n_datasets=2, unconstrained=True
        )
        assert config.is_multi_dataset is True

        config2 = ModelConfig(base_model="nbdm")
        assert config2.is_multi_dataset is False

    def test_hierarchical_dataset_mu_requires_n_datasets(self):
        """hierarchical_dataset_mu without n_datasets should raise."""
        with pytest.raises(Exception):
            ModelConfig(
                base_model="nbdm",
                hierarchical_dataset_mu=True,
                unconstrained=True,
            )

    def test_hierarchical_dataset_mu_requires_unconstrained(self):
        """hierarchical_dataset_mu without unconstrained should raise."""
        with pytest.raises(Exception):
            ModelConfig(
                base_model="nbdm",
                n_datasets=2,
                hierarchical_dataset_mu=True,
                unconstrained=False,
            )

    def test_hierarchical_dataset_gate_defaults_false(self):
        """hierarchical_dataset_gate defaults to False."""
        config = ModelConfig(base_model="zinb")
        assert config.hierarchical_dataset_gate is False

    def test_hierarchical_dataset_gate_requires_n_datasets(self):
        """hierarchical_dataset_gate without n_datasets should raise."""
        with pytest.raises(ValueError, match="n_datasets"):
            ModelConfig(
                base_model="zinb",
                hierarchical_dataset_gate=True,
                unconstrained=True,
            )

    def test_hierarchical_dataset_gate_requires_unconstrained(self):
        """hierarchical_dataset_gate without unconstrained should raise."""
        with pytest.raises(ValueError, match="unconstrained"):
            ModelConfig(
                base_model="zinb",
                n_datasets=2,
                hierarchical_dataset_gate=True,
                unconstrained=False,
            )

    def test_hierarchical_dataset_gate_requires_zero_inflated(self):
        """hierarchical_dataset_gate on a non-ZI model should raise."""
        with pytest.raises(ValueError, match="zero-inflated"):
            ModelConfig(
                base_model="nbdm",
                n_datasets=2,
                hierarchical_dataset_gate=True,
                unconstrained=True,
            )

    def test_hierarchical_gate_and_dataset_gate_conflict(self):
        """Cannot set both hierarchical_gate and hierarchical_dataset_gate."""
        with pytest.raises(ValueError, match="cannot be set simultaneously"):
            ModelConfig(
                base_model="zinb",
                n_datasets=2,
                hierarchical_gate=True,
                hierarchical_dataset_gate=True,
                unconstrained=True,
            )


# ==============================================================================
# index_dataset_params
# ==============================================================================


class TestIndexDatasetParams:
    """Test the index_dataset_params likelihood helper."""

    def test_basic_indexing(self):
        """Per-dataset params are indexed to per-cell values."""
        n_datasets = 3
        n_cells = 5
        n_genes = 4

        param_values = {
            # Per-dataset gene-specific: (n_datasets, n_genes)
            "mu": jnp.arange(n_datasets * n_genes)
            .reshape(n_datasets, n_genes)
            .astype(float),
            # Shared across datasets: (n_genes,)
            "gate": jnp.ones(n_genes),
        }

        # Cell assignments: cells 0,1 → ds 0; cells 2,3 → ds 1; cell 4 → ds 2
        dataset_indices = jnp.array([0, 0, 1, 1, 2])

        result = index_dataset_params(param_values, dataset_indices, n_datasets)

        # mu should be indexed: result["mu"][c] == param_values["mu"][ds[c]]
        assert result["mu"].shape == (n_cells, n_genes)
        np.testing.assert_array_equal(result["mu"][0], param_values["mu"][0])
        np.testing.assert_array_equal(result["mu"][2], param_values["mu"][1])
        np.testing.assert_array_equal(result["mu"][4], param_values["mu"][2])

        # gate should be unchanged (not per-dataset)
        np.testing.assert_array_equal(result["gate"], param_values["gate"])

    def test_scalar_per_dataset(self):
        """Scalar per-dataset param (shape (n_datasets,)) is indexed."""
        n_datasets = 2
        param_values = {
            "p": jnp.array([0.3, 0.7]),
        }
        dataset_indices = jnp.array([0, 0, 1, 0, 1])

        result = index_dataset_params(param_values, dataset_indices, n_datasets)
        expected = jnp.array([0.3, 0.3, 0.7, 0.3, 0.7])
        np.testing.assert_allclose(result["p"], expected)

    def test_param_specs_prevent_ambiguous_1d_gene_slicing(self):
        """Gene-specific 1D params stay unsliced when specs mark non-dataset."""
        n_datasets = 3
        n_genes = 3
        dataset_indices = jnp.array([0, 1, 2, 0])

        # n_genes == n_datasets is intentionally ambiguous by shape alone.
        # The param spec metadata should keep this gene-specific vector intact.
        param_values = {"phi": jnp.linspace(0.1, 0.3, n_genes)}
        param_specs = [
            ExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
            )
        ]

        result = index_dataset_params(
            param_values=param_values,
            dataset_indices=dataset_indices,
            n_datasets=n_datasets,
            param_specs=param_specs,
        )
        assert result["phi"].shape == (n_genes,)
        np.testing.assert_allclose(result["phi"], param_values["phi"])


# ==============================================================================
# Helper to build a minimal DatasetHierarchicalExpNormalSpec
# ==============================================================================


def _make_ds_exp_spec(name="r"):
    """Create a DatasetHierarchicalExpNormalSpec for testing."""
    return DatasetHierarchicalExpNormalSpec(
        name=name,
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        is_gene_specific=True,
        is_dataset=True,
        unconstrained=True,
        loc=0.0,
        scale=1.0,
        hyper_loc_name=f"log_{name}_dataset_loc",
        hyper_scale_name=f"log_{name}_dataset_scale",
    )


# ==============================================================================
# SVI DatasetMixin / get_dataset
# ==============================================================================


class TestSVIDatasetMixin:
    """Test get_dataset() and 3-axis indexing on SVI results."""

    @pytest.fixture
    def multi_dataset_svi_results(self):
        """Create a minimal ScribeSVIResults with dataset structure."""
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 10

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[
                _make_ds_exp_spec("r"),
            ],
        )

        # Variational params with dataset dimension
        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            # Population-level hyperparameters (no dataset dim)
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
        }

        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0, 0.5]),
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )

    def test_get_dataset_basic(self, multi_dataset_svi_results):
        """get_dataset() slices the dataset dimension."""
        results = multi_dataset_svi_results
        ds0 = results.get_dataset(0)

        # Per-dataset params should lose the dataset dimension
        assert ds0.params["log_r_loc"].shape == (10,)
        # Population params should remain unchanged
        assert ds0.params["log_r_dataset_loc_loc"].shape == (10,)
        # model_config should have n_datasets=None
        assert ds0.model_config.n_datasets is None

    def test_get_dataset_out_of_range(self, multi_dataset_svi_results):
        """get_dataset() with invalid index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            multi_dataset_svi_results.get_dataset(5)

    def test_get_dataset_no_datasets(self):
        """get_dataset() on non-multi-dataset model raises ValueError."""
        from scribe.svi.results import ScribeSVIResults

        config = ModelConfig(base_model="nbdm")
        results = ScribeSVIResults(
            params={"p_loc": jnp.zeros(5)},
            loss_history=jnp.array([1.0]),
            n_cells=10,
            n_genes=5,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        with pytest.raises(ValueError, match="multi-dataset"):
            results.get_dataset(0)

    def test_three_axis_indexing(self, multi_dataset_svi_results):
        """results[:, :, 0] should call get_dataset(0)."""
        results = multi_dataset_svi_results
        ds0 = results[:, :, 0]
        assert ds0.model_config.n_datasets is None
        assert ds0.params["log_r_loc"].shape == (10,)


# ==============================================================================
# _slice_param_for_dataset broadcasting
# ==============================================================================


class TestSliceParamForDataset:
    """Verify _slice_param_for_dataset handles 1D scalar-per-dataset params."""

    def test_2d_per_dataset_sliced(self):
        """Standard 2D (n_datasets, n_genes) is sliced to (n_genes,)."""
        from scribe.svi._sampling_denoising import _slice_param_for_dataset

        param = jnp.ones((3, 10))
        result = _slice_param_for_dataset(param, dataset_idx=1, n_datasets=3)
        assert result.shape == (10,)

    def test_1d_per_dataset_scalar_sliced(self):
        """1D (n_datasets,) scalar-per-dataset param is sliced to scalar."""
        from scribe.svi._sampling_denoising import _slice_param_for_dataset

        param = jnp.array([0.1, 0.2, 0.3])
        result = _slice_param_for_dataset(param, dataset_idx=2, n_datasets=3)
        assert result.shape == ()
        assert jnp.isclose(result, 0.3)

    def test_1d_per_gene_not_sliced(self):
        """1D (n_genes,) per-gene param is NOT sliced when n_genes != n_datasets."""
        from scribe.svi._sampling_denoising import _slice_param_for_dataset

        param = jnp.ones(500)
        result = _slice_param_for_dataset(param, dataset_idx=0, n_datasets=3)
        # shape[0]=500 != n_datasets=3, so returned unchanged
        assert result.shape == (500,)

    def test_none_returns_none(self):
        """None input returns None."""
        from scribe.svi._sampling_denoising import _slice_param_for_dataset

        assert _slice_param_for_dataset(None, dataset_idx=0, n_datasets=2) is None

    def test_scalar_returned_unchanged(self):
        """0D scalar params pass through unchanged."""
        from scribe.svi._sampling_denoising import _slice_param_for_dataset

        param = jnp.array(0.5)
        result = _slice_param_for_dataset(param, dataset_idx=0, n_datasets=2)
        assert result.shape == ()
        assert jnp.isclose(result, 0.5)


# ==============================================================================
# _compute_canonical_parameters broadcasting with scalar-per-dataset phi/p
# ==============================================================================


class TestCanonicalParamsScalarDataset:
    """Verify _compute_canonical_parameters reshapes scalar-per-dataset phi/p
    for correct broadcasting with (n_datasets, n_genes) mu.
    """

    def _make_results(self, *, parameterization, n_datasets, n_genes):
        """Create a minimal ScribeSVIResults for canonical param testing."""
        from scribe.svi.results import ScribeSVIResults

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            parameterization=parameterization,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            hierarchical_dataset_p="scalar",
        )
        # Dummy variational params (not used directly by
        # _compute_canonical_parameters but needed to construct the object)
        params = {
            "dummy_loc": jnp.zeros(n_genes),
        }
        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=50,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )

    def test_mean_odds_scalar_phi_broadcast(self):
        """r = mu * phi with phi (n_datasets,) and mu (n_datasets, n_genes)."""
        n_ds, n_genes = 3, 20
        results = self._make_results(
            parameterization="mean_odds", n_datasets=n_ds, n_genes=n_genes
        )
        estimates = {
            "phi": jnp.array([0.5, 1.0, 2.0]),
            "mu": jnp.ones((n_ds, n_genes)) * 10.0,
        }
        out = results._compute_canonical_parameters(estimates, verbose=False)
        # r should be (n_datasets, n_genes) after broadcasting
        assert out["r"].shape == (n_ds, n_genes)
        # First dataset: r = 10 * 0.5 = 5
        assert jnp.allclose(out["r"][0], 5.0)
        # Third dataset: r = 10 * 2.0 = 20
        assert jnp.allclose(out["r"][2], 20.0)

    def test_mean_odds_scalar_phi_derives_p(self):
        """p = 1/(1+phi) with scalar-per-dataset phi."""
        n_ds, n_genes = 2, 15
        results = self._make_results(
            parameterization="mean_odds", n_datasets=n_ds, n_genes=n_genes
        )
        estimates = {
            "phi": jnp.array([1.0, 3.0]),
            "mu": jnp.ones((n_ds, n_genes)),
        }
        out = results._compute_canonical_parameters(estimates, verbose=False)
        assert "p" in out
        # p = 1/(1+phi): dataset 0 -> 0.5, dataset 1 -> 0.25
        assert jnp.isclose(out["p"][0], 0.5)
        assert jnp.isclose(out["p"][1], 0.25)

    def test_mean_prob_scalar_p_broadcast(self):
        """r = mu*(1-p)/p with p (n_datasets,) and mu (n_datasets, n_genes)."""
        n_ds, n_genes = 2, 30
        results = self._make_results(
            parameterization="mean_prob", n_datasets=n_ds, n_genes=n_genes
        )
        estimates = {
            "p": jnp.array([0.5, 0.25]),
            "mu": jnp.ones((n_ds, n_genes)) * 10.0,
        }
        out = results._compute_canonical_parameters(estimates, verbose=False)
        # r = mu * (1-p)/p
        # Dataset 0: 10 * (1-0.5)/0.5 = 10
        # Dataset 1: 10 * (1-0.25)/0.25 = 30
        assert out["r"].shape == (n_ds, n_genes)
        assert jnp.allclose(out["r"][0], 10.0)
        assert jnp.allclose(out["r"][1], 30.0)


# ==============================================================================
# MCMC DatasetMixin
# ==============================================================================


class TestMCMCDatasetMixin:
    """Test get_dataset() on MCMC results."""

    @pytest.fixture
    def multi_dataset_mcmc_results(self):
        """Create a minimal ScribeMCMCResults with dataset structure."""
        from scribe.mcmc.results import ScribeMCMCResults

        n_datasets = 2
        n_genes = 8
        n_samples = 50

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[
                _make_ds_exp_spec("r"),
            ],
        )

        # MCMC samples: (n_samples, n_datasets, n_genes)
        samples = {
            "r": jnp.ones((n_samples, n_datasets, n_genes)),
            "log_r_dataset_loc": jnp.zeros((n_samples, n_genes)),
        }

        return ScribeMCMCResults(
            samples=samples,
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )

    def test_get_dataset_mcmc(self, multi_dataset_mcmc_results):
        """get_dataset() slices MCMC samples along dataset axis."""
        ds1 = multi_dataset_mcmc_results.get_dataset(1)

        # Per-dataset: (n_samples, n_datasets, n_genes) → (n_samples, n_genes)
        assert ds1.samples["r"].shape == (50, 8)
        # Population-level: unchanged
        assert ds1.samples["log_r_dataset_loc"].shape == (50, 8)
        assert ds1.model_config.n_datasets is None


# ==============================================================================
# compare_datasets
# ==============================================================================


class TestCompareDatasets:
    """Test the compare_datasets DE helper."""

    def test_requires_multi_dataset(self):
        """compare_datasets raises on non-multi-dataset results."""
        from scribe.de import compare_datasets
        from scribe.svi.results import ScribeSVIResults

        config = ModelConfig(base_model="nbdm")
        results = ScribeSVIResults(
            params={"p_loc": jnp.zeros(5)},
            loss_history=jnp.array([1.0]),
            n_cells=10,
            n_genes=5,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        with pytest.raises(ValueError, match="multi-dataset"):
            compare_datasets(results, 0, 1)


# ==============================================================================
# Mixture + Dataset composition
# ==============================================================================


def _make_ds_exp_spec_mixture(name="r"):
    """Create a DatasetHierarchicalExpNormalSpec that is also mixture-specific."""
    return DatasetHierarchicalExpNormalSpec(
        name=name,
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        is_gene_specific=True,
        is_dataset=True,
        is_mixture=True,
        unconstrained=True,
        loc=0.0,
        scale=1.0,
        hyper_loc_name=f"log_{name}_dataset_loc",
        hyper_scale_name=f"log_{name}_dataset_scale",
    )


def _make_sigmoid_spec_dataset_only(name="p"):
    """Create a DatasetHierarchicalSigmoidNormalSpec (dataset, not mixture)."""
    return DatasetHierarchicalSigmoidNormalSpec(
        name=name,
        shape_dims=("n_genes",),
        default_params=(0.0, 1.0),
        is_gene_specific=True,
        is_dataset=True,
        is_mixture=False,
        unconstrained=True,
        loc=0.0,
        scale=1.0,
        hyper_loc_name=f"logit_{name}_dataset_loc",
        hyper_scale_name=f"logit_{name}_dataset_scale",
    )


class TestMixtureDatasetComposition:
    """Test that mixture (n_components) and multi-dataset (n_datasets) compose."""

    # ------------------------------------------------------------------
    # index_dataset_params with mixture+dataset
    # ------------------------------------------------------------------

    def test_index_dataset_params_mixture_and_dataset(self):
        """Param with is_mixture=True, is_dataset=True: (K, D, G) -> (batch, K, G)."""
        K, D, G = 3, 2, 5
        batch = 4

        r_spec = _make_ds_exp_spec_mixture("r")
        param_values = {
            "r": jnp.arange(K * D * G, dtype=float).reshape(K, D, G),
            "mixing_weights": jnp.ones(K) / K,
        }
        dataset_indices = jnp.array([0, 1, 0, 1])

        result = index_dataset_params(
            param_values,
            dataset_indices,
            D,
            param_specs=[r_spec],
        )

        # r should be (batch, K, G) — batch-first for MixtureSameFamily
        assert result["r"].shape == (batch, K, G)
        # Cell 0 assigned to dataset 0: result["r"][0] == original[:, 0, :]
        np.testing.assert_array_equal(
            result["r"][0], param_values["r"][:, 0, :]
        )
        # Cell 1 assigned to dataset 1: result["r"][1] == original[:, 1, :]
        np.testing.assert_array_equal(
            result["r"][1], param_values["r"][:, 1, :]
        )
        # mixing_weights should be unchanged (no dataset or spec)
        np.testing.assert_array_equal(
            result["mixing_weights"], param_values["mixing_weights"]
        )

    def test_index_dataset_params_dataset_only_in_mixture_model(self):
        """Param with is_dataset=True, is_mixture=False: (D, G) -> (batch, G)."""
        D, G = 2, 5
        batch = 4

        p_spec = _make_sigmoid_spec_dataset_only("p")
        param_values = {
            "p": jnp.arange(D * G, dtype=float).reshape(D, G),
            # r is mixture-only, not dataset — should pass through
            "r": jnp.ones((3, G)),
        }
        dataset_indices = jnp.array([1, 0, 0, 1])

        result = index_dataset_params(
            param_values,
            dataset_indices,
            D,
            param_specs=[p_spec],
        )

        assert result["p"].shape == (batch, G)
        np.testing.assert_array_equal(result["p"][0], param_values["p"][1])
        # r should be unchanged (not in param_specs as dataset)
        np.testing.assert_array_equal(result["r"], param_values["r"])

    def test_index_dataset_params_legacy_fallback(self):
        """Without param_specs, falls back to shape[0] == n_datasets heuristic."""
        D, G = 2, 5
        param_values = {
            "mu": jnp.ones((D, G)),
        }
        dataset_indices = jnp.array([0, 1, 0])

        result = index_dataset_params(
            param_values,
            dataset_indices,
            D,
            param_specs=None,
        )
        assert result["mu"].shape == (3, G)

    # ------------------------------------------------------------------
    # broadcast_p_for_mixture with batch dim
    # ------------------------------------------------------------------

    def test_broadcast_p_batch_with_3d_r(self):
        """p: (batch, G), r: (batch, K, G) -> (batch, 1, G)."""
        batch, K, G = 4, 3, 5
        p = jnp.ones((batch, G))
        r = jnp.ones((batch, K, G))

        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (batch, 1, G)

    def test_broadcast_p_1d_gene_specific_with_3d_r(self):
        """p: (G,), r: (batch, K, G) -> (1, 1, G)."""
        batch, K, G = 4, 3, 5
        p = jnp.ones(G)
        r = jnp.ones((batch, K, G))

        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (1, 1, G)

    def test_broadcast_p_scalar_with_3d_r(self):
        """p: (), r: (batch, K, G) -> (1, 1, 1)."""
        p = jnp.array(0.5)
        r = jnp.ones((4, 3, 5))

        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (1, 1, 1)

    def test_broadcast_p_3d_passthrough(self):
        """p: (batch, K, G) -> unchanged."""
        batch, K, G = 4, 3, 5
        p = jnp.ones((batch, K, G))
        r = jnp.ones((batch, K, G))

        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (batch, K, G)

    def test_broadcast_p_existing_2d_behaviour(self):
        """p: (K, G), r: (K, G) -> (K, G) unchanged (existing behaviour)."""
        K, G = 3, 5
        p = jnp.ones((K, G))
        r = jnp.ones((K, G))

        result = broadcast_p_for_mixture(p, r)
        assert result.shape == (K, G)

    # ------------------------------------------------------------------
    # ModelConfig: hierarchical_p + hierarchical_dataset_p conflict
    # ------------------------------------------------------------------

    def test_model_config_rejects_hierarchical_p_plus_dataset_p(self):
        """Cannot set hierarchical_p=True and hierarchical_dataset_p != 'none'."""
        with pytest.raises(ValueError, match="cannot be set simultaneously"):
            ModelConfig(
                base_model="nbdm",
                n_datasets=2,
                unconstrained=True,
                hierarchical_p=True,
                hierarchical_dataset_p="gene_specific",
            )

    def test_model_config_allows_hierarchical_p_alone(self):
        """hierarchical_p=True alone is fine."""
        config = ModelConfig(
            base_model="nbdm",
            unconstrained=True,
            hierarchical_p=True,
        )
        assert config.hierarchical_p is True

    def test_model_config_allows_dataset_p_alone(self):
        """hierarchical_dataset_p != 'none' alone is fine."""
        config = ModelConfig(
            base_model="nbdm",
            n_datasets=2,
            unconstrained=True,
            hierarchical_dataset_p="gene_specific",
        )
        assert config.hierarchical_dataset_p == "gene_specific"

    # ------------------------------------------------------------------
    # SVI results: mixture + dataset subsetting
    # ------------------------------------------------------------------

    @pytest.fixture
    def mixture_dataset_svi_results(self):
        """ScribeSVIResults with both n_components and n_datasets."""
        from scribe.svi.results import ScribeSVIResults

        K, D, G = 3, 2, 10
        n_samples = 20

        # r is both mixture and dataset: (K, D, G)
        r_spec = _make_ds_exp_spec_mixture("r")
        # p is dataset-only: (D, G)
        p_spec = _make_sigmoid_spec_dataset_only("p")

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=D,
            n_components=K,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[r_spec, p_spec],
        )

        # Variational params
        params = {
            "log_r_loc": jnp.arange(K * D * G, dtype=float).reshape(K, D, G),
            "log_r_scale": jnp.ones((K, D, G)),
            "logit_p_loc": jnp.arange(D * G, dtype=float).reshape(D, G),
            "logit_p_scale": jnp.ones((D, G)),
            "mixing_weights_loc": jnp.ones(K) / K,
        }

        # Posterior samples: sample dim prepended
        posterior_samples = {
            "r": jnp.arange(n_samples * K * D * G, dtype=float).reshape(
                n_samples, K, D, G
            ),
            "p": jnp.arange(n_samples * D * G, dtype=float).reshape(
                n_samples, D, G
            ),
            "mixing_weights": jnp.ones((n_samples, K)) / K,
        }

        return ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0, 0.5]),
            n_cells=100,
            n_genes=G,
            model_type="nbdm_mix",
            model_config=config,
            prior_params={},
            posterior_samples=posterior_samples,
            n_components=K,
        )

    def test_get_dataset_with_mixture(self, mixture_dataset_svi_results):
        """get_dataset slices the correct axis for mixture+dataset params."""
        results = mixture_dataset_svi_results
        K, D, G = 3, 2, 10

        ds0 = results.get_dataset(0)

        # r params: (K, D, G) -> (K, G)
        assert ds0.params["log_r_loc"].shape == (K, G)
        np.testing.assert_array_equal(
            ds0.params["log_r_loc"],
            results.params["log_r_loc"][:, 0, :],
        )
        # p params: (D, G) -> (G,)
        assert ds0.params["logit_p_loc"].shape == (G,)
        np.testing.assert_array_equal(
            ds0.params["logit_p_loc"],
            results.params["logit_p_loc"][0, :],
        )
        # mixing_weights (mixture-only): unchanged
        assert ds0.params["mixing_weights_loc"].shape == (K,)

    def test_get_dataset_posterior_with_mixture(
        self, mixture_dataset_svi_results
    ):
        """get_dataset slices posterior samples at the correct axis."""
        results = mixture_dataset_svi_results
        K, D, G, S = 3, 2, 10, 20

        ds1 = results.get_dataset(1)

        # r posterior: (S, K, D, G) -> (S, K, G)
        assert ds1.posterior_samples["r"].shape == (S, K, G)
        np.testing.assert_array_equal(
            ds1.posterior_samples["r"],
            results.posterior_samples["r"][:, :, 1, :],
        )
        # p posterior: (S, D, G) -> (S, G)
        assert ds1.posterior_samples["p"].shape == (S, G)
        np.testing.assert_array_equal(
            ds1.posterior_samples["p"],
            results.posterior_samples["p"][:, 1, :],
        )
        # mixing_weights: unchanged
        assert ds1.posterior_samples["mixing_weights"].shape == (S, K)

    def test_get_component_then_dataset(self, mixture_dataset_svi_results):
        """Composing get_component(k).get_dataset(d) works correctly."""
        results = mixture_dataset_svi_results
        K, D, G, S = 3, 2, 10, 20

        # First slice component 1, then dataset 0
        comp1 = results.get_component(1)
        # After component slice: r params become (D, G), p stays (D, G)
        assert comp1.params["log_r_loc"].shape == (D, G)

        ds0 = comp1.get_dataset(0)
        # After dataset slice: (D, G) -> (G,)
        assert ds0.params["log_r_loc"].shape == (G,)
        assert ds0.params["logit_p_loc"].shape == (G,)

    def test_get_dataset_then_component(self, mixture_dataset_svi_results):
        """Composing get_dataset(d).get_component(k) works correctly."""
        results = mixture_dataset_svi_results
        K, D, G, S = 3, 2, 10, 20

        # First slice dataset 1, then component 2
        ds1 = results.get_dataset(1)
        # After dataset slice: r params become (K, G)
        assert ds1.params["log_r_loc"].shape == (K, G)

        comp2 = ds1.get_component(2)
        # After component slice: (K, G) -> (G,)
        assert comp2.params["log_r_loc"].shape == (G,)

    # ------------------------------------------------------------------
    # MCMC results: mixture + dataset subsetting
    # ------------------------------------------------------------------

    @pytest.fixture
    def mixture_dataset_mcmc_results(self):
        """ScribeMCMCResults with both n_components and n_datasets."""
        from scribe.mcmc.results import ScribeMCMCResults

        K, D, G = 3, 2, 8
        n_samples = 30

        r_spec = _make_ds_exp_spec_mixture("r")
        p_spec = _make_sigmoid_spec_dataset_only("p")

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=D,
            n_components=K,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[r_spec, p_spec],
        )

        samples = {
            "r": jnp.arange(n_samples * K * D * G, dtype=float).reshape(
                n_samples, K, D, G
            ),
            "p": jnp.arange(n_samples * D * G, dtype=float).reshape(
                n_samples, D, G
            ),
            "mixing_weights": jnp.ones((n_samples, K)) / K,
        }

        return ScribeMCMCResults(
            samples=samples,
            n_cells=100,
            n_genes=G,
            model_type="nbdm_mix",
            model_config=config,
            prior_params={},
            n_components=K,
        )

    def test_mcmc_get_dataset_with_mixture(self, mixture_dataset_mcmc_results):
        """MCMC get_dataset slices axis 2 for mixture+dataset params."""
        results = mixture_dataset_mcmc_results
        K, D, G, S = 3, 2, 8, 30

        ds0 = results.get_dataset(0)

        # r: (S, K, D, G) -> (S, K, G)
        assert ds0.samples["r"].shape == (S, K, G)
        np.testing.assert_array_equal(
            ds0.samples["r"],
            results.samples["r"][:, :, 0, :],
        )
        # p: (S, D, G) -> (S, G)
        assert ds0.samples["p"].shape == (S, G)
        # mixing_weights: unchanged
        assert ds0.samples["mixing_weights"].shape == (S, K)

    # ------------------------------------------------------------------
    # Factory: _datasetify_* preserves is_mixture
    # ------------------------------------------------------------------

    def test_datasetify_mu_preserves_is_mixture(self):
        """_datasetify_mu keeps is_mixture from the original spec."""
        from scribe.models.presets.factory import _datasetify_mu

        # Create a spec list where 'r' is mixture-specific
        original_r = ExpNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_mixture=True,
            unconstrained=True,
        )
        result_specs = _datasetify_mu(
            param_specs=[original_r],
            param_key="canonical",
            guide_families={},
            n_datasets=2,
        )

        # The dataset-hierarchical r spec should have both flags
        r_specs = [s for s in result_specs if s.name == "r"]
        assert len(r_specs) == 1
        assert r_specs[0].is_dataset is True
        assert r_specs[0].is_mixture is True

    def test_datasetify_p_preserves_is_mixture(self):
        """_datasetify_p keeps is_mixture from the original spec."""
        from scribe.models.presets.factory import _datasetify_p

        original_p = SigmoidNormalSpec(
            name="p",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_mixture=True,
            unconstrained=True,
        )
        result_specs = _datasetify_p(
            param_specs=[original_p],
            param_key="mean_prob",
            guide_families={},
            n_datasets=2,
            mode="gene_specific",
        )

        p_specs = [s for s in result_specs if s.name == "p"]
        assert len(p_specs) == 1
        assert p_specs[0].is_dataset is True
        assert p_specs[0].is_mixture is True

    def test_datasetify_mu_non_mixture_stays_non_mixture(self):
        """_datasetify_mu does not add is_mixture when original lacks it."""
        from scribe.models.presets.factory import _datasetify_mu

        original_r = ExpNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_mixture=False,
            unconstrained=True,
        )
        result_specs = _datasetify_mu(
            param_specs=[original_r],
            param_key="canonical",
            guide_families={},
            n_datasets=2,
        )

        r_specs = [s for s in result_specs if s.name == "r"]
        assert len(r_specs) == 1
        assert r_specs[0].is_dataset is True
        assert r_specs[0].is_mixture is False


# ==============================================================================
# Integration: create_model end-to-end with multi-dataset configs
# ==============================================================================


class TestCreateModelMultiDataset:
    """End-to-end tests that call create_model with multi-dataset configs.

    These exercise the full pipeline: factory → model builder → guide builder →
    validation dry run, catching wiring issues like missing dims, mismatched
    event_dim, or missing dataset_indices in the dry run.
    """

    @staticmethod
    def _builder(model_type="zinbvcp", parameterization="mean_odds"):
        """Create a builder pre-configured for multi-dataset testing."""
        b = (
            ModelConfigBuilder()
            .for_model(model_type)
            .with_parameterization(parameterization)
            .unconstrained()
        )
        b._n_datasets = 2
        return b

    def test_hierarchical_dataset_mu_only(self):
        """Model + guide created successfully with hierarchical_dataset_mu."""
        b = self._builder()
        b._hierarchical_dataset_mu = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_hierarchical_dataset_p_gene_specific(self):
        """Model + guide with gene-specific hierarchical dataset p."""
        b = self._builder()
        b._hierarchical_dataset_p = "gene_specific"
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_hierarchical_dataset_mu_and_p(self):
        """Model + guide with both dataset hierarchical mu and p."""
        b = self._builder()
        b._hierarchical_dataset_mu = True
        b._hierarchical_dataset_p = "gene_specific"
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_hierarchical_dataset_with_gate(self):
        """Matches the user's real command: dataset mu + p + hierarchical gate."""
        b = self._builder()
        b._hierarchical_dataset_mu = True
        b._hierarchical_dataset_p = "gene_specific"
        b._hierarchical_gate = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_dataset_mu_all_model_types(self, model_type):
        """hierarchical_dataset_mu works across all model types."""
        b = self._builder(model_type=model_type)
        b._hierarchical_dataset_mu = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize(
        "parameterization", ["canonical", "mean_prob", "mean_odds"]
    )
    def test_dataset_mu_all_parameterizations(self, parameterization):
        """hierarchical_dataset_mu works across all parameterizations."""
        b = self._builder(parameterization=parameterization)
        b._hierarchical_dataset_mu = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_dataset_specs_have_is_dataset_flag(self):
        """Verify that the returned param_specs include dataset-flagged specs."""
        b = self._builder()
        b._hierarchical_dataset_mu = True
        b._hierarchical_dataset_p = "gene_specific"
        config = b.build()
        _, _, specs = create_model(config)
        dataset_specs = [s for s in specs if getattr(s, "is_dataset", False)]
        assert len(dataset_specs) >= 2, (
            "Expected at least mu/r and p/phi dataset specs, "
            f"got {[s.name for s in dataset_specs]}"
        )

    # --- hierarchical_dataset_gate tests ---

    def test_hierarchical_dataset_gate_creates_model(self):
        """Dataset-level hierarchical gate produces a callable model/guide."""
        b = self._builder()
        b._hierarchical_dataset_gate = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)
        # Verify that the gate spec has is_dataset=True
        gate_specs = [s for s in specs if s.name == "gate"]
        assert len(gate_specs) == 1
        assert getattr(gate_specs[0], "is_dataset", False) is True

    def test_hierarchical_dataset_gate_adds_hyperparameters(self):
        """Dataset-level gate hierarchy includes population hyperparameters."""
        b = self._builder()
        b._hierarchical_dataset_gate = True
        config = b.build()
        _, _, specs = create_model(config)
        spec_names = {s.name for s in specs}
        assert "logit_gate_dataset_loc" in spec_names
        assert "logit_gate_dataset_scale" in spec_names

    def test_hierarchical_dataset_gate_combined_with_mu_and_p(self):
        """All three dataset-level hierarchies together produce a valid model."""
        b = self._builder()
        b._hierarchical_dataset_mu = True
        b._hierarchical_dataset_p = "gene_specific"
        b._hierarchical_dataset_gate = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)
        # All three should be dataset-flagged
        dataset_specs = [s for s in specs if getattr(s, "is_dataset", False)]
        dataset_names = {s.name for s in dataset_specs}
        assert "gate" in dataset_names

    @pytest.mark.parametrize("model_type", ["zinb", "zinbvcp"])
    def test_hierarchical_dataset_gate_zi_models(self, model_type):
        """hierarchical_dataset_gate works across ZI model types."""
        b = self._builder(model_type=model_type)
        b._hierarchical_dataset_gate = True
        config = b.build()
        model, guide, specs = create_model(config)
        assert callable(model)
        assert callable(guide)


# ==============================================================================
# _n_cells_per_dataset propagation through get_dataset
# ==============================================================================


class TestNCellsPerDataset:
    """Test that _n_cells_per_dataset is correctly used by get_dataset()."""

    def test_svi_get_dataset_uses_per_dataset_n_cells(self):
        """SVI get_dataset() sets n_cells from _n_cells_per_dataset."""
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 3
        n_genes = 5
        per_ds_counts = jnp.array([40, 30, 30])

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[_make_ds_exp_spec("r")],
        )
        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        results._n_cells_per_dataset = per_ds_counts

        for d in range(n_datasets):
            ds_view = results.get_dataset(d)
            assert ds_view.n_cells == int(per_ds_counts[d])

    def test_svi_get_dataset_falls_back_to_total_without_field(self):
        """Without _n_cells_per_dataset, get_dataset() keeps total n_cells."""
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 5

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[_make_ds_exp_spec("r")],
        )
        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        # _n_cells_per_dataset defaults to None

        ds_view = results.get_dataset(0)
        assert ds_view.n_cells == 100

    def test_mcmc_get_dataset_uses_per_dataset_n_cells(self):
        """MCMC get_dataset() sets n_cells from _n_cells_per_dataset."""
        from scribe.mcmc.results import ScribeMCMCResults

        n_datasets = 2
        n_genes = 5
        n_samples = 10
        per_ds_counts = jnp.array([60, 40])

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[_make_ds_exp_spec("r")],
        )
        samples = {
            "r": jnp.ones((n_samples, n_datasets, n_genes)),
            "log_r_dataset_loc": jnp.zeros((n_samples, n_genes)),
        }
        results = ScribeMCMCResults(
            samples=samples,
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        results._n_cells_per_dataset = per_ds_counts

        for d in range(n_datasets):
            ds_view = results.get_dataset(d)
            assert ds_view.n_cells == int(per_ds_counts[d])

    def test_mcmc_get_dataset_falls_back_to_total_without_field(self):
        """Without _n_cells_per_dataset, MCMC keeps total n_cells."""
        from scribe.mcmc.results import ScribeMCMCResults

        n_datasets = 2
        n_genes = 5
        n_samples = 10

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[_make_ds_exp_spec("r")],
        )
        samples = {
            "r": jnp.ones((n_samples, n_datasets, n_genes)),
            "log_r_dataset_loc": jnp.zeros((n_samples, n_genes)),
        }
        results = ScribeMCMCResults(
            samples=samples,
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )

        ds_view = results.get_dataset(0)
        assert ds_view.n_cells == 100

    def test_three_axis_indexing_preserves_per_dataset_n_cells(self):
        """results[:, :, d] also gets the correct per-dataset n_cells."""
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 5
        per_ds_counts = jnp.array([55, 45])

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[_make_ds_exp_spec("r")],
        )
        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=100,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        results._n_cells_per_dataset = per_ds_counts

        # 3-axis indexing: results[:, :, d] calls get_dataset(d)
        ds_view = results[:, :, 1]
        assert ds_view.n_cells == 45


# ==============================================================================
# Cell-specific param subsetting via _dataset_indices
# ==============================================================================


class TestCellSpecificSubsetting:
    """Test that get_dataset() subsets cell-specific params using _dataset_indices."""

    def test_svi_cell_specific_params_subsetted(self):
        """SVI get_dataset() subsets cell-specific variational params."""
        from scribe.models.builders.parameter_specs import BetaSpec
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 5
        n_cells = 10

        # 6 cells in dataset 0, 4 in dataset 1
        dataset_indices = jnp.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        per_ds_counts = jnp.array([6, 4])

        # Cell-specific BetaSpec — use nbvcp so p_capture is active
        cell_spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            unconstrained=True,
        )
        ds_spec = _make_ds_exp_spec("r")

        config = ModelConfig(
            base_model="nbvcp",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[ds_spec, cell_spec],
        )

        # Cell-specific params: shape (n_cells,)
        phi_capture_loc = jnp.arange(n_cells, dtype=jnp.float32)
        phi_capture_scale = jnp.ones(n_cells)

        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
            "p_capture_loc": phi_capture_loc,
            "p_capture_scale": phi_capture_scale,
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        results._n_cells_per_dataset = per_ds_counts
        results._dataset_indices = dataset_indices

        # Dataset 0: cells at indices [0, 1, 3, 5, 7, 8]
        ds0 = results.get_dataset(0)
        assert ds0.n_cells == 6
        assert ds0.params["p_capture_loc"].shape == (6,)
        np.testing.assert_allclose(
            ds0.params["p_capture_loc"],
            phi_capture_loc[dataset_indices == 0],
        )

        # Dataset 1: cells at indices [2, 4, 6, 9]
        ds1 = results.get_dataset(1)
        assert ds1.n_cells == 4
        assert ds1.params["p_capture_loc"].shape == (4,)
        np.testing.assert_allclose(
            ds1.params["p_capture_loc"],
            phi_capture_loc[dataset_indices == 1],
        )

        # Per-dataset params (log_r_loc) should still be sliced correctly
        assert ds0.params["log_r_loc"].shape == (n_genes,)
        assert ds1.params["log_r_loc"].shape == (n_genes,)

    def test_svi_no_dataset_indices_skips_cell_subsetting(self):
        """Without _dataset_indices, cell-specific params are left untouched."""
        from scribe.models.builders.parameter_specs import BetaSpec
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 5
        n_cells = 10

        cell_spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            unconstrained=True,
        )
        ds_spec = _make_ds_exp_spec("r")

        config = ModelConfig(
            base_model="nbvcp",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[ds_spec, cell_spec],
        )
        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
            "p_capture_loc": jnp.ones(n_cells),
            "p_capture_scale": jnp.ones(n_cells),
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        # No _dataset_indices set — cell-specific params stay full size
        ds0 = results.get_dataset(0)
        assert ds0.params["p_capture_loc"].shape == (n_cells,)

    def test_mcmc_cell_specific_samples_subsetted(self):
        """MCMC get_dataset() subsets cell-specific samples on axis 1."""
        from scribe.models.builders.parameter_specs import BetaSpec
        from scribe.mcmc.results import ScribeMCMCResults

        n_datasets = 2
        n_genes = 5
        n_cells = 10
        n_samples = 8

        dataset_indices = jnp.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1])
        per_ds_counts = jnp.array([6, 4])

        cell_spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            unconstrained=True,
        )
        ds_spec = _make_ds_exp_spec("r")

        config = ModelConfig(
            base_model="nbvcp",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[ds_spec, cell_spec],
        )

        # p_capture samples: (n_samples, n_cells)
        p_capture_samples = jnp.arange(
            n_samples * n_cells, dtype=jnp.float32
        ).reshape(n_samples, n_cells)

        samples = {
            "r": jnp.ones((n_samples, n_datasets, n_genes)),
            "log_r_dataset_loc": jnp.zeros((n_samples, n_genes)),
            "p_capture": p_capture_samples,
        }
        results = ScribeMCMCResults(
            samples=samples,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        results._n_cells_per_dataset = per_ds_counts
        results._dataset_indices = dataset_indices

        # Dataset 0: 6 cells
        ds0 = results.get_dataset(0)
        assert ds0.n_cells == 6
        assert ds0.samples["p_capture"].shape == (n_samples, 6)
        np.testing.assert_allclose(
            ds0.samples["p_capture"],
            p_capture_samples[:, dataset_indices == 0],
        )

        # Dataset 1: 4 cells
        ds1 = results.get_dataset(1)
        assert ds1.n_cells == 4
        assert ds1.samples["p_capture"].shape == (n_samples, 4)

    def test_dataset_indices_propagated_through_gene_subsetting(self):
        """_dataset_indices survives gene subsetting ([:, :, d] pattern)."""
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 5
        n_cells = 10

        dataset_indices = jnp.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 1])

        config = ModelConfig(
            base_model="nbdm",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[_make_ds_exp_spec("r")],
        )
        params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
        }
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.array([1.0]),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        results._dataset_indices = dataset_indices
        results._n_cells_per_dataset = jnp.array([6, 4])

        # Gene subset preserves _dataset_indices
        gene_view = results[:2]
        assert gene_view._dataset_indices is not None
        np.testing.assert_array_equal(
            gene_view._dataset_indices, dataset_indices
        )


# ==============================================================================
# Result concat across cell partitions
# ==============================================================================


class TestResultsConcat:
    """Test SVI/MCMC concat for compatible multi-dataset partitions."""

    def test_svi_concat_cells_preserves_dataset_metadata(self):
        """SVI concat should stack cell-specific params and dataset metadata."""
        from scribe.models.builders.parameter_specs import BetaSpec
        from scribe.svi.results import ScribeSVIResults

        n_datasets = 2
        n_genes = 4
        ds_spec = _make_ds_exp_spec("r")
        cell_spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            unconstrained=True,
        )
        config = ModelConfig(
            base_model="nbvcp",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[ds_spec, cell_spec],
        )

        shared_params = {
            "log_r_loc": jnp.zeros((n_datasets, n_genes)),
            "log_r_scale": jnp.ones((n_datasets, n_genes)),
            "log_r_dataset_loc_loc": jnp.zeros(n_genes),
            "log_r_dataset_loc_scale": jnp.ones(n_genes),
        }

        res_a = ScribeSVIResults(
            params={
                **shared_params,
                "p_capture_loc": jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            },
            loss_history=jnp.array([1.0, 0.8]),
            n_cells=5,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        res_a._dataset_indices = jnp.array([0, 1, 0, 1, 0])
        res_a._n_cells_per_dataset = jnp.array([3, 2])

        res_b = ScribeSVIResults(
            params={
                **shared_params,
                "p_capture_loc": jnp.array([0.6, 0.7, 0.8, 0.9]),
            },
            loss_history=jnp.array([0.9, 0.7]),
            n_cells=4,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        res_b._dataset_indices = jnp.array([1, 0, 1, 1])
        res_b._n_cells_per_dataset = jnp.array([1, 3])

        combined = ScribeSVIResults.concat([res_a, res_b])
        assert combined.n_cells == 9
        np.testing.assert_allclose(
            combined.params["p_capture_loc"],
            jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        )
        np.testing.assert_array_equal(
            combined._dataset_indices,
            jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 1]),
        )
        np.testing.assert_array_equal(
            combined._n_cells_per_dataset,
            jnp.array([4, 5]),
        )
        ds0 = combined.get_dataset(0)
        ds1 = combined.get_dataset(1)
        assert ds0.n_cells == 4
        assert ds1.n_cells == 5

    def test_mcmc_concat_cells_preserves_dataset_metadata(self):
        """MCMC concat should stack cell-specific samples and metadata."""
        from scribe.models.builders.parameter_specs import BetaSpec
        from scribe.mcmc.results import ScribeMCMCResults

        n_datasets = 2
        n_genes = 4
        n_samples = 6
        ds_spec = _make_ds_exp_spec("r")
        cell_spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
            unconstrained=True,
        )
        config = ModelConfig(
            base_model="nbvcp",
            n_datasets=n_datasets,
            unconstrained=True,
            hierarchical_dataset_mu=True,
            param_specs=[ds_spec, cell_spec],
        )

        shared_samples = {
            "r": jnp.ones((n_samples, n_datasets, n_genes)),
            "log_r_dataset_loc": jnp.zeros((n_samples, n_genes)),
        }

        res_a = ScribeMCMCResults(
            samples={
                **shared_samples,
                "p_capture": jnp.arange(n_samples * 5).reshape(n_samples, 5),
            },
            n_cells=5,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        res_a._dataset_indices = jnp.array([0, 1, 0, 1, 0])
        res_a._n_cells_per_dataset = jnp.array([3, 2])

        res_b = ScribeMCMCResults(
            samples={
                **shared_samples,
                "p_capture": jnp.arange(n_samples * 4).reshape(n_samples, 4) + 100,
            },
            n_cells=4,
            n_genes=n_genes,
            model_type="nbvcp",
            model_config=config,
            prior_params={},
        )
        res_b._dataset_indices = jnp.array([1, 0, 1, 1])
        res_b._n_cells_per_dataset = jnp.array([1, 3])

        combined = ScribeMCMCResults.concat([res_a, res_b])
        assert combined.n_cells == 9
        assert combined.samples["p_capture"].shape == (n_samples, 9)
        np.testing.assert_array_equal(
            combined._dataset_indices,
            jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 1]),
        )
        np.testing.assert_array_equal(
            combined._n_cells_per_dataset,
            jnp.array([4, 5]),
        )
        ds0 = combined.get_dataset(0)
        ds1 = combined.get_dataset(1)
        assert ds0.n_cells == 4
        assert ds1.n_cells == 5


# ==============================================================================
# Horseshoe Prior Tests
# ==============================================================================


class TestHorseshoeConfigValidation:
    """Validate horseshoe flags are mutually exclusive with normal hierarchy."""

    def test_horseshoe_p_excludes_hierarchical_p(self):
        """Setting both horseshoe_p and hierarchical_p should fail."""
        with pytest.raises(
            ValueError, match="horseshoe_p.*hierarchical_p.*mutually exclusive"
        ):
            ModelConfig(
                base_model="nbdm",
                unconstrained=True,
                horseshoe_p=True,
                hierarchical_p=True,
            )

    def test_horseshoe_gate_excludes_hierarchical_gate(self):
        with pytest.raises(
            ValueError,
            match="horseshoe_gate.*hierarchical_gate.*mutually exclusive",
        ):
            ModelConfig(
                base_model="zinb",
                unconstrained=True,
                horseshoe_gate=True,
                hierarchical_gate=True,
            )

    def test_horseshoe_dataset_mu_excludes_hierarchical(self):
        with pytest.raises(
            ValueError,
            match="horseshoe_dataset_mu.*hierarchical_dataset_mu.*mutually exclusive",
        ):
            ModelConfig(
                base_model="nbdm",
                n_datasets=2,
                unconstrained=True,
                horseshoe_dataset_mu=True,
                hierarchical_dataset_mu=True,
            )

    def test_horseshoe_dataset_p_excludes_hierarchical(self):
        with pytest.raises(
            ValueError,
            match="horseshoe_dataset_p.*hierarchical_dataset_p.*mutually exclusive",
        ):
            ModelConfig(
                base_model="nbdm",
                n_datasets=2,
                unconstrained=True,
                horseshoe_dataset_p=True,
                hierarchical_dataset_p="gene_specific",
            )

    def test_horseshoe_dataset_gate_excludes_hierarchical(self):
        with pytest.raises(
            ValueError,
            match="horseshoe_dataset_gate.*hierarchical_dataset_gate.*mutually exclusive",
        ):
            ModelConfig(
                base_model="zinb",
                n_datasets=2,
                unconstrained=True,
                horseshoe_dataset_gate=True,
                hierarchical_dataset_gate=True,
            )

    def test_valid_horseshoe_p_standalone(self):
        """horseshoe_p=True alone (no hierarchical_p) should succeed."""
        cfg = ModelConfig(
            base_model="nbdm",
            unconstrained=True,
            horseshoe_p=True,
        )
        assert cfg.horseshoe_p is True

    def test_valid_horseshoe_dataset_mu_standalone(self):
        """horseshoe_dataset_mu=True alone (no hierarchical_dataset_mu)."""
        cfg = ModelConfig(
            base_model="nbdm",
            n_datasets=2,
            unconstrained=True,
            horseshoe_dataset_mu=True,
        )
        assert cfg.horseshoe_dataset_mu is True

    def test_horseshoe_p_requires_unconstrained(self):
        """horseshoe_p=True without unconstrained should fail."""
        with pytest.raises(ValueError, match="horseshoe_p.*unconstrained"):
            ModelConfig(
                base_model="nbdm",
                horseshoe_p=True,
            )

    def test_horseshoe_gate_requires_zero_inflated(self):
        """horseshoe_gate=True on non-ZI model should fail."""
        with pytest.raises(ValueError, match="horseshoe_gate.*zero-inflated"):
            ModelConfig(
                base_model="nbdm",
                unconstrained=True,
                horseshoe_gate=True,
            )

    def test_horseshoe_dataset_mu_requires_n_datasets(self):
        """horseshoe_dataset_mu=True without n_datasets should fail."""
        with pytest.raises(
            ValueError, match="horseshoe_dataset_mu.*n_datasets"
        ):
            ModelConfig(
                base_model="nbdm",
                unconstrained=True,
                horseshoe_dataset_mu=True,
            )


class TestHorseshoeCreateModel:
    """Test create_model with horseshoe flags."""

    @staticmethod
    def _builder(model_type="zinbvcp", parameterization="mean_odds"):
        b = (
            ModelConfigBuilder()
            .for_model(model_type)
            .with_parameterization(parameterization)
            .unconstrained()
        )
        b._n_datasets = 2
        return b

    def test_horseshoe_dataset_mu(self):
        """create_model with horseshoe_dataset_mu produces horseshoe specs."""
        from scribe.models.builders.parameter_specs import (
            HalfCauchySpec,
            HorseshoeDatasetExpNormalSpec,
            InverseGammaSpec,
        )

        b = self._builder()
        b._horseshoe_dataset_mu = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        spec_names = [s.name for s in specs]
        assert "tau_mu_dataset" in spec_names
        assert "lambda_mu_dataset" in spec_names
        assert "c_sq_mu_dataset" in spec_names

        tau_spec = next(s for s in specs if s.name == "tau_mu_dataset")
        lambda_spec = next(s for s in specs if s.name == "lambda_mu_dataset")
        c_sq_spec = next(s for s in specs if s.name == "c_sq_mu_dataset")
        assert isinstance(tau_spec, HalfCauchySpec)
        assert isinstance(lambda_spec, HalfCauchySpec)
        assert isinstance(c_sq_spec, InverseGammaSpec)

        # tau is scalar, lambda is gene-specific
        assert tau_spec.shape_dims == ()
        assert lambda_spec.shape_dims == ("n_genes",)

        mu_spec = next(s for s in specs if s.name == "mu")
        assert isinstance(mu_spec, HorseshoeDatasetExpNormalSpec)
        assert mu_spec.raw_name == "mu_raw"
        assert mu_spec.uses_ncp is True

    def test_horseshoe_dataset_p(self):
        """create_model with horseshoe_dataset_p produces horseshoe specs."""
        from scribe.models.builders.parameter_specs import (
            HalfCauchySpec,
            HorseshoeDatasetSigmoidNormalSpec,
        )

        b = self._builder(parameterization="mean_prob")
        b._horseshoe_dataset_p = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        spec_names = [s.name for s in specs]
        assert "tau_p_dataset" in spec_names
        assert "lambda_p_dataset" in spec_names
        assert "c_sq_p_dataset" in spec_names

        p_spec = next(s for s in specs if s.name == "p")
        assert isinstance(p_spec, HorseshoeDatasetSigmoidNormalSpec)
        assert p_spec.raw_name == "p_raw_dataset"

    def test_horseshoe_dataset_gate(self):
        """create_model with horseshoe_dataset_gate."""
        from scribe.models.builders.parameter_specs import (
            HorseshoeDatasetSigmoidNormalSpec,
        )

        b = self._builder(model_type="zinbvcp")
        b._horseshoe_dataset_gate = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        gate_spec = next(s for s in specs if s.name == "gate")
        assert isinstance(gate_spec, HorseshoeDatasetSigmoidNormalSpec)
        assert gate_spec.raw_name == "gate_raw_dataset"

    def test_horseshoe_gene_level_p(self):
        """create_model with horseshoe_p (gene-level)."""
        from scribe.models.builders.parameter_specs import (
            HorseshoeHierarchicalSigmoidNormalSpec,
        )

        b = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_prob")
            .unconstrained()
        )
        b._horseshoe_p = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        spec_names = [s.name for s in specs]
        assert "tau_p" in spec_names
        assert "lambda_p" in spec_names

        p_spec = next(s for s in specs if s.name == "p")
        assert isinstance(p_spec, HorseshoeHierarchicalSigmoidNormalSpec)
        assert p_spec.raw_name == "p_raw"

    def test_horseshoe_gene_level_phi_mean_odds(self):
        """horseshoe_p with mean_odds should use ExpNormalSpec for phi."""
        from scribe.models.builders.parameter_specs import (
            HorseshoeHierarchicalExpNormalSpec,
        )

        b = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .unconstrained()
        )
        b._horseshoe_p = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        spec_names = [s.name for s in specs]
        assert "tau_phi" in spec_names
        assert "lambda_phi" in spec_names

        phi_spec = next(s for s in specs if s.name == "phi")
        assert isinstance(phi_spec, HorseshoeHierarchicalExpNormalSpec)
        assert phi_spec.raw_name == "phi_raw"

    def test_horseshoe_gene_level_gate(self):
        """create_model with horseshoe_gate (gene-level)."""
        from scribe.models.builders.parameter_specs import (
            HorseshoeHierarchicalSigmoidNormalSpec,
        )

        b = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("mean_prob")
            .unconstrained()
        )
        b._horseshoe_gate = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        gate_spec = next(s for s in specs if s.name == "gate")
        assert isinstance(gate_spec, HorseshoeHierarchicalSigmoidNormalSpec)
        assert gate_spec.raw_name == "gate_raw"

    def test_combined_dataset_horseshoe_mu_p_gate(self):
        """Multiple horseshoe flags enabled simultaneously (standalone)."""
        b = self._builder(model_type="zinbvcp")
        b._horseshoe_dataset_mu = True
        b._horseshoe_dataset_p = True
        b._horseshoe_dataset_gate = True
        config = b.build()
        model, guide, specs = create_model(config)

        assert callable(model)
        assert callable(guide)

        spec_names = [s.name for s in specs]
        # All three horseshoe trios present (mean_odds => phi, not p)
        for prefix in ("mu_dataset", "phi_dataset", "gate_dataset"):
            assert (
                f"tau_{prefix}" in spec_names
            ), f"Expected tau_{prefix} in {spec_names}"
            assert f"lambda_{prefix}" in spec_names
            assert f"c_sq_{prefix}" in spec_names


# ==============================================================================
# API validation: dataset-only hierarchical priors
# ==============================================================================


class TestFitApiDatasetHierarchyValidation:
    """Validate dataset-only hierarchical flags in the public fit API.

    Notes
    -----
    These tests verify the user-facing behavior introduced for early
    configuration validation:

    - Dataset-level hierarchical flags require `dataset_key` so cells can be
      mapped to datasets.
    - Gene-level hierarchical flags remain valid for single-dataset fits.
    """

    @staticmethod
    def _make_adata(
        *, include_dataset_column: bool, single_dataset_column: bool = False
    ):
        """Create a tiny AnnData object for fit API validation tests.

        Parameters
        ----------
        include_dataset_column : bool
            Whether to include an `obs["dataset"]` column used by
            `dataset_key` for multi-dataset indexing.
        single_dataset_column : bool, optional
            When True and ``include_dataset_column=True``, populates
            ``obs["dataset"]`` with a single category across all cells.
            When False, uses two categories (d0/d1). Default is False.

        Returns
        -------
        anndata.AnnData
            Synthetic count data with optional dataset labels.
        """
        import anndata
        import pandas as pd

        # Build a tiny synthetic count matrix to keep fit tests lightweight.
        n_cells, n_genes = 6, 4
        rng = np.random.default_rng(123)
        x = rng.poisson(3, (n_cells, n_genes)).astype(np.float32)

        # Optionally expose dataset labels through adata.obs["dataset"].
        obs = {"cell_type": ["A", "A", "A", "B", "B", "B"]}
        if include_dataset_column:
            if single_dataset_column:
                obs["dataset"] = ["d0", "d0", "d0", "d0", "d0", "d0"]
            else:
                obs["dataset"] = ["d0", "d0", "d0", "d1", "d1", "d1"]
        return anndata.AnnData(X=x, obs=pd.DataFrame(obs))

    def test_dataset_hierarchy_requires_dataset_key(self):
        """Dataset-level hierarchy on single-dataset input raises early."""
        import scribe

        adata = self._make_adata(include_dataset_column=False)

        # Dataset-level priors need dataset_key for per-cell dataset indexing.
        with pytest.raises(ValueError, match="require dataset_key"):
            scribe.fit(
                adata,
                model="nbdm",
                n_steps=1,
                batch_size=3,
                seed=0,
                unconstrained=True,
                hierarchical_dataset_mu=True,
            )

    def test_single_dataset_allows_gene_level_hierarchy(self):
        """Gene-level hierarchy remains valid without dataset splitting."""
        import scribe

        adata = self._make_adata(include_dataset_column=False)

        # hierarchical_p is a gene-level option and should still be valid.
        result = scribe.fit(
            adata,
            model="nbdm",
            n_steps=1,
            batch_size=3,
            seed=0,
            unconstrained=True,
            hierarchical_p=True,
        )
        assert result.n_cells == adata.n_obs

    def test_dataset_hierarchy_allowed_with_dataset_key(self):
        """Dataset-level hierarchy is allowed when dataset_key is provided."""
        import scribe

        adata = self._make_adata(include_dataset_column=True)

        # Providing dataset_key enables multi-dataset indexing and unblocks
        # dataset-level hierarchical priors.
        result = scribe.fit(
            adata,
            model="nbdm",
            n_steps=1,
            batch_size=3,
            seed=0,
            unconstrained=True,
            dataset_key="dataset",
            hierarchical_dataset_mu=True,
        )
        assert result.n_cells == adata.n_obs

    def test_single_dataset_auto_downgrades_dataset_mu(self):
        """Single-dataset columns downgrade hierarchical_dataset_mu safely."""
        import scribe

        adata = self._make_adata(
            include_dataset_column=True, single_dataset_column=True
        )

        # Auto-downgrade should disable dataset-level mu and warn.
        with pytest.warns(UserWarning, match="automatic hierarchy downgrade"):
            result = scribe.fit(
                adata,
                model="nbdm",
                n_steps=1,
                batch_size=3,
                seed=0,
                unconstrained=True,
                dataset_key="dataset",
                hierarchical_dataset_mu=True,
            )
        assert result.model_config.hierarchical_dataset_mu is False
        assert result.model_config.n_datasets is None

    @pytest.mark.parametrize("dataset_p_mode", ["gene_specific", "two_level"])
    def test_single_dataset_auto_downgrades_dataset_p_to_gene_level(
        self, dataset_p_mode
    ):
        """Single-dataset p hierarchy maps to gene-level hierarchical_p."""
        import scribe

        adata = self._make_adata(
            include_dataset_column=True, single_dataset_column=True
        )

        # gene_specific/two_level are downgraded to hierarchical_p in 1-dataset.
        with pytest.warns(UserWarning, match="automatic hierarchy downgrade"):
            result = scribe.fit(
                adata,
                model="nbdm",
                n_steps=1,
                batch_size=3,
                seed=0,
                unconstrained=True,
                dataset_key="dataset",
                hierarchical_dataset_p=dataset_p_mode,
            )
        assert result.model_config.hierarchical_p is True
        assert result.model_config.hierarchical_dataset_p == "none"
        assert result.model_config.n_datasets is None

    def test_single_dataset_auto_downgrades_dataset_p_scalar_to_none(self):
        """Single-dataset scalar p hierarchy downgrades without hierarchical_p."""
        import scribe

        adata = self._make_adata(
            include_dataset_column=True, single_dataset_column=True
        )

        # scalar mode maps to shared p/phi and should not enable hierarchical_p.
        with pytest.warns(UserWarning, match="automatic hierarchy downgrade"):
            result = scribe.fit(
                adata,
                model="nbdm",
                n_steps=1,
                batch_size=3,
                seed=0,
                unconstrained=True,
                dataset_key="dataset",
                hierarchical_dataset_p="scalar",
            )
        assert result.model_config.hierarchical_p is False
        assert result.model_config.hierarchical_dataset_p == "none"
        assert result.model_config.n_datasets is None

    def test_single_dataset_auto_downgrades_dataset_gate_to_gene_level(self):
        """Single-dataset gate hierarchy maps to gene-level hierarchical_gate."""
        import scribe

        adata = self._make_adata(
            include_dataset_column=True, single_dataset_column=True
        )

        # Dataset gate hierarchy should downgrade to gene-level gate in 1-dataset.
        with pytest.warns(UserWarning, match="automatic hierarchy downgrade"):
            result = scribe.fit(
                adata,
                model="zinb",
                n_steps=1,
                batch_size=3,
                seed=0,
                unconstrained=True,
                dataset_key="dataset",
                hierarchical_dataset_gate=True,
            )
        assert result.model_config.hierarchical_gate is True
        assert result.model_config.hierarchical_dataset_gate is False
        assert result.model_config.n_datasets is None

    def test_single_dataset_no_auto_downgrade_preserves_strict_error(self):
        """Opt-out keeps strict dataset-level validation behavior."""
        import scribe

        adata = self._make_adata(
            include_dataset_column=True, single_dataset_column=True
        )

        # Disabling auto-downgrade should preserve n_datasets > 1 validation.
        with pytest.raises(Exception, match="greater than 1"):
            scribe.fit(
                adata,
                model="nbdm",
                n_steps=1,
                batch_size=3,
                seed=0,
                unconstrained=True,
                dataset_key="dataset",
                hierarchical_dataset_mu=True,
                auto_downgrade_single_dataset_hierarchy=False,
            )

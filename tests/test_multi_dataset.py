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

        result = index_dataset_params(
            param_values, dataset_indices, n_datasets
        )

        # mu should be indexed: result["mu"][c] == param_values["mu"][ds[c]]
        assert result["mu"].shape == (n_cells, n_genes)
        np.testing.assert_array_equal(
            result["mu"][0], param_values["mu"][0]
        )
        np.testing.assert_array_equal(
            result["mu"][2], param_values["mu"][1]
        )
        np.testing.assert_array_equal(
            result["mu"][4], param_values["mu"][2]
        )

        # gate should be unchanged (not per-dataset)
        np.testing.assert_array_equal(result["gate"], param_values["gate"])

    def test_scalar_per_dataset(self):
        """Scalar per-dataset param (shape (n_datasets,)) is indexed."""
        n_datasets = 2
        param_values = {
            "p": jnp.array([0.3, 0.7]),
        }
        dataset_indices = jnp.array([0, 0, 1, 0, 1])

        result = index_dataset_params(
            param_values, dataset_indices, n_datasets
        )
        expected = jnp.array([0.3, 0.3, 0.7, 0.3, 0.7])
        np.testing.assert_allclose(result["p"], expected)


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
            param_values, dataset_indices, D, param_specs=[r_spec],
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
            param_values, dataset_indices, D, param_specs=[p_spec],
        )

        assert result["p"].shape == (batch, G)
        np.testing.assert_array_equal(
            result["p"][0], param_values["p"][1]
        )
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
            param_values, dataset_indices, D, param_specs=None,
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

    def test_mcmc_get_dataset_with_mixture(
        self, mixture_dataset_mcmc_results
    ):
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

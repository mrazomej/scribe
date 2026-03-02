"""Tests for multi-dataset hierarchical modeling.

Covers:
- DatasetHierarchical*Spec creation and sampling
- resolve_shape with is_dataset flag
- ModelConfig with dataset fields and validation
- index_dataset_params likelihood helper
- DatasetMixin.get_dataset and 3-axis __getitem__ on results
- compare_datasets DE helper
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
    resolve_shape,
)
from scribe.models.config import ModelConfig
from scribe.models.components.likelihoods.base import index_dataset_params


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

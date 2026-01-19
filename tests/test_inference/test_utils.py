"""Tests for utils module."""

import pytest
import jax.numpy as jnp
from jax import random
from scribe.inference.utils import (
    process_counts_data,
    validate_model_inference_match,
    validate_inference_config_match,
)
from scribe.models.config import (
    ModelConfigBuilder,
    InferenceConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
)


class TestProcessCountsData:
    """Test process_counts_data function."""

    def test_numpy_array(self):
        """Test processing numpy array."""
        rng_key = random.PRNGKey(42)
        counts = random.poisson(rng_key, lam=5.0, shape=(50, 100))
        counts = jnp.array(counts)

        data_config = DataConfig(cells_axis=0, layer=None)
        count_data, adata, n_cells, n_genes = process_counts_data(
            counts, data_config
        )

        assert count_data.shape == (50, 100)
        assert n_cells == 50
        assert n_genes == 100
        assert adata is None

    def test_transpose_cells_axis(self):
        """Test processing with cells_axis=1 (transpose)."""
        rng_key = random.PRNGKey(42)
        counts = random.poisson(
            rng_key, lam=5.0, shape=(100, 50)
        )  # genes x cells
        counts = jnp.array(counts)

        data_config = DataConfig(cells_axis=1, layer=None)
        count_data, adata, n_cells, n_genes = process_counts_data(
            counts, data_config
        )

        # Should be transposed to cells x genes
        assert count_data.shape == (50, 100)
        assert n_cells == 50
        assert n_genes == 100


class TestValidateModelInferenceMatch:
    """Test validate_model_inference_match function."""

    def test_valid_match(self):
        """Test validation with matching methods."""
        model_config = (
            ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
        )

        # Should not raise
        validate_model_inference_match(model_config, "svi")

    def test_mismatch(self):
        """Test validation error with mismatched methods."""
        model_config = (
            ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
        )

        with pytest.raises(ValueError, match="Inference method mismatch"):
            validate_model_inference_match(model_config, "mcmc")


class TestValidateInferenceConfigMatch:
    """Test validate_inference_config_match function."""

    def test_valid_match(self):
        """Test validation with matching methods."""
        model_config = (
            ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
        )

        inference_config = InferenceConfig.from_svi(SVIConfig())

        # Should not raise
        validate_inference_config_match(model_config, inference_config)

    def test_mismatch(self):
        """Test validation error with mismatched methods."""
        model_config = (
            ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
        )

        inference_config = InferenceConfig.from_mcmc(MCMCConfig())

        with pytest.raises(ValueError, match="Inference method mismatch"):
            validate_inference_config_match(model_config, inference_config)

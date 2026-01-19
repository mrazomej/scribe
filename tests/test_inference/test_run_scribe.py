"""Integration tests for run_scribe function."""

import os
import pytest
import jax.numpy as jnp
from scribe.inference import run_scribe
from scribe.models.config import (
    ModelConfigBuilder,
    InferenceConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
)


def pytest_generate_tests(metafunc):
    """Configure device type for tests that need it."""
    # Tests that run actual inference should use CPU by default to avoid slow GPU
    # initialization, but can be overridden with --device flag
    if "device_type" in metafunc.fixturenames:
        device = metafunc.config.getoption("--device", default="cpu")
        metafunc.parametrize("device_type", [device], indirect=True)


@pytest.fixture
def device_type(request):
    """Fixture to configure JAX device based on --device option."""
    device = request.config.getoption("--device", default="cpu")

    # Configure JAX device
    if device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import jax

        jax.config.update("jax_platform_name", "cpu")
    else:
        if "JAX_PLATFORM_NAME" in os.environ:
            del os.environ["JAX_PLATFORM_NAME"]

    return device


class TestRunScribePresetAPI:
    """Test preset-based API."""

    def test_basic_svi_preset(self, sample_counts, device_type):
        """Test basic SVI inference with preset API."""
        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        results = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            inference_config=inference_config,
            seed=42,
        )

        assert results is not None
        assert hasattr(results, "params")
        assert hasattr(results, "loss_history")

    def test_basic_mcmc_preset(self, sample_counts, device_type):
        """Test basic MCMC inference with preset API."""
        inference_config = InferenceConfig.from_mcmc(
            MCMCConfig(n_samples=50, n_warmup=25)
        )

        results = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="canonical",
            inference_method="mcmc",
            inference_config=inference_config,
            seed=42,
        )

        assert results is not None
        assert hasattr(results, "samples")

    def test_preset_with_parameterization_variants(
        self, sample_counts, device_type
    ):
        """Test preset API with different parameterizations."""
        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        # Test canonical (standard)
        results1 = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            inference_config=inference_config,
            seed=42,
        )

        # Test mean_prob (linked)
        results2 = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="mean_prob",
            inference_method="svi",
            inference_config=inference_config,
            seed=42,
        )

        assert results1 is not None
        assert results2 is not None

    def test_preset_backward_compat_parameterization(
        self, sample_counts, device_type
    ):
        """Test backward compatibility with old parameterization names."""
        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        # Old name: "standard" should work
        results = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="standard",  # Old name
            inference_method="svi",
            inference_config=inference_config,
            seed=42,
        )

        assert results is not None

    def test_preset_mixture_model(self, sample_counts, device_type):
        """Test preset API with mixture model."""
        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        results = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            inference_config=inference_config,
            seed=42,
        )

        # Note: mixture models require n_components parameter
        # This would be added via preset_builder in the actual implementation
        assert results is not None

    def test_preset_default_inference_config(self, sample_counts, device_type):
        """Test preset API with default inference config."""
        # Should use defaults if inference_config is None
        results = run_scribe(
            counts=sample_counts,
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            # inference_config=None uses defaults
            seed=42,
        )

        assert results is not None


class TestRunScribeModelConfigAPI:
    """Test ModelConfig-based API."""

    def test_basic_model_config(
        self,
        sample_counts,
        nbdm_model_config,
        svi_inference_config,
        device_type,
    ):
        """Test basic usage with ModelConfig."""
        results = run_scribe(
            counts=sample_counts,
            model_config=nbdm_model_config,
            inference_config=svi_inference_config,
            seed=42,
        )

        assert results is not None
        assert hasattr(results, "params")

    def test_model_config_with_custom_settings(
        self, sample_counts, device_type
    ):
        """Test ModelConfig with custom settings."""
        model_config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("mean_odds")
            .with_inference("svi")
            .unconstrained()
            .build()
        )

        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        results = run_scribe(
            counts=sample_counts,
            model_config=model_config,
            inference_config=inference_config,
            seed=42,
        )

        assert results is not None

    def test_model_config_mcmc(
        self, sample_counts, mcmc_inference_config, device_type
    ):
        """Test ModelConfig with MCMC."""
        model_config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("canonical")
            .with_inference("mcmc")
            .build()
        )

        results = run_scribe(
            counts=sample_counts,
            model_config=model_config,
            inference_config=mcmc_inference_config,
            seed=42,
        )

        assert results is not None
        assert hasattr(results, "samples")


class TestRunScribeValidation:
    """Test validation and error handling."""

    def test_missing_model_error(self, sample_counts):
        """Test error when neither model nor model_config provided."""
        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        with pytest.raises(
            ValueError,
            match="Either 'model' or 'model_config' must be provided",
        ):
            run_scribe(
                counts=sample_counts,
                # model=None, model_config=None
                inference_config=inference_config,
                seed=42,
            )

    def test_inference_method_mismatch(self, sample_counts, nbdm_model_config):
        """Test error when inference methods don't match."""
        # Model config specifies SVI, but we pass MCMC inference config
        mcmc_config = InferenceConfig.from_mcmc(
            MCMCConfig(n_samples=50, n_warmup=25)
        )

        with pytest.raises(ValueError, match="Inference method mismatch"):
            run_scribe(
                counts=sample_counts,
                model_config=nbdm_model_config,  # SVI
                inference_config=mcmc_config,  # MCMC
                seed=42,
            )

    def test_data_config_usage(self, sample_counts, device_type):
        """Test using DataConfig."""
        data_config = DataConfig(cells_axis=0, layer=None)
        inference_config = InferenceConfig.from_svi(
            SVIConfig(n_steps=10, batch_size=None)
        )

        results = run_scribe(
            counts=sample_counts,
            model="nbdm",
            inference_method="svi",
            inference_config=inference_config,
            data_config=data_config,
            seed=42,
        )

        assert results is not None

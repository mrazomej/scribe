"""Tests for inference_config module."""

import pytest
import numpyro
from scribe.inference.inference_config import create_default_inference_config
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig
from scribe.models.config.enums import InferenceMethod
from scribe.inference.optimizer_factory import (
    build_optimizer_from_config,
    resolve_svi_optimizer,
)


class TestCreateDefaultInferenceConfig:
    """Test create_default_inference_config function."""

    def test_default_svi_config(self):
        """Test default SVI config creation."""
        config = create_default_inference_config(InferenceMethod.SVI)

        assert config.method == InferenceMethod.SVI
        assert config.svi is not None
        assert config.mcmc is None
        assert config.svi.n_steps == 50_000
        assert config.svi.batch_size == 512
        assert config.svi.stable_update is True
        assert config.svi.optimizer_config is None

    def test_default_mcmc_config(self):
        """Test default MCMC config creation."""
        config = create_default_inference_config(InferenceMethod.MCMC)

        assert config.method == InferenceMethod.MCMC
        assert config.mcmc is not None
        assert config.svi is None
        assert config.mcmc.n_samples == 2_000
        assert config.mcmc.n_warmup == 1_000
        assert config.mcmc.n_chains == 1

    def test_default_vae_config(self):
        """Test default VAE config creation."""
        config = create_default_inference_config(InferenceMethod.VAE)

        assert config.method == InferenceMethod.VAE
        assert config.svi is not None  # VAE uses SVI config
        assert config.mcmc is None
        assert config.svi.n_steps == 100_000


class TestInferenceConfig:
    """Test InferenceConfig class."""

    def test_from_svi(self):
        """Test InferenceConfig.from_svi factory method."""
        svi_config = SVIConfig(n_steps=50000, batch_size=256)
        inference_config = InferenceConfig.from_svi(svi_config)

        assert inference_config.method == InferenceMethod.SVI
        assert inference_config.svi == svi_config
        assert inference_config.mcmc is None

    def test_from_mcmc(self):
        """Test InferenceConfig.from_mcmc factory method."""
        mcmc_config = MCMCConfig(n_samples=5000, n_chains=4)
        inference_config = InferenceConfig.from_mcmc(mcmc_config)

        assert inference_config.method == InferenceMethod.MCMC
        assert inference_config.mcmc == mcmc_config
        assert inference_config.svi is None

    def test_from_vae(self):
        """Test InferenceConfig.from_vae factory method."""
        svi_config = SVIConfig(n_steps=100000)
        inference_config = InferenceConfig.from_vae(svi_config)

        assert inference_config.method == InferenceMethod.VAE
        assert inference_config.svi == svi_config
        assert inference_config.mcmc is None

    def test_get_config_svi(self):
        """Test get_config method for SVI."""
        svi_config = SVIConfig(n_steps=50000)
        inference_config = InferenceConfig.from_svi(svi_config)

        config = inference_config.get_config()
        assert isinstance(config, SVIConfig)
        assert config == svi_config

    def test_get_config_mcmc(self):
        """Test get_config method for MCMC."""
        mcmc_config = MCMCConfig(n_samples=5000)
        inference_config = InferenceConfig.from_mcmc(mcmc_config)

        config = inference_config.get_config()
        assert isinstance(config, MCMCConfig)
        assert config == mcmc_config

    def test_validation_svi_missing(self):
        """Test validation error when SVI config missing."""
        with pytest.raises(ValueError, match="SVIConfig required"):
            InferenceConfig(
                method=InferenceMethod.SVI,
                svi=None,
                mcmc=None,
            )

    def test_validation_mcmc_missing(self):
        """Test validation error when MCMC config missing."""
        with pytest.raises(ValueError, match="MCMCConfig required"):
            InferenceConfig(
                method=InferenceMethod.MCMC,
                svi=None,
                mcmc=None,
            )

    def test_validation_wrong_config_type(self):
        """Test validation error when wrong config type provided."""
        # This test checks that validation catches missing config
        # The error message will be from Pydantic, not a plain ValueError
        with pytest.raises(Exception, match="MCMCConfig required"):
            InferenceConfig(
                method=InferenceMethod.MCMC,
                svi=None,  # Should have mcmc, not svi
                mcmc=None,  # Missing!
            )


class TestSVIOptimizerConfig:
    """Test structured SVI optimizer configuration behavior."""

    def test_optimizer_config_parses_from_dict(self):
        """SVIConfig accepts dict-based optimizer_config payloads."""
        config = SVIConfig(
            n_steps=100,
            optimizer_config={"name": "adam", "step_size": 1e-3},
        )
        assert config.optimizer_config is not None
        assert config.optimizer_config.name == "adam"
        assert config.optimizer_config.step_size == 1e-3

    def test_backward_compat_old_payload_without_optimizer_config(self):
        """Old SVI payloads without optimizer fields remain valid."""
        config = SVIConfig.model_validate(
            {
                "n_steps": 100,
                "batch_size": 32,
                "stable_update": True,
            }
        )
        assert config.n_steps == 100
        assert config.optimizer is None
        assert config.optimizer_config is None

    def test_optimizer_config_invalid_name(self):
        """Unsupported optimizer names are rejected at validation time."""
        with pytest.raises(ValueError, match="Unsupported optimizer name"):
            SVIConfig(
                n_steps=100,
                optimizer_config={"name": "not_a_real_optimizer"},
            )

    def test_build_optimizer_from_structured_config(self):
        """Structured optimizer configs are converted to NumPyro optimizers."""
        spec = SVIConfig.OptimizerConfig(
            name="adam",
            step_size=5e-4,
        )
        optimizer = build_optimizer_from_config(spec)
        assert optimizer is not None

    def test_explicit_optimizer_takes_precedence(self):
        """Explicit optimizer object overrides optimizer_config resolution."""
        explicit_optimizer = numpyro.optim.Adam(step_size=1e-2)
        config = SVIConfig(
            n_steps=100,
            optimizer=explicit_optimizer,
            optimizer_config={"name": "adam", "step_size": 1e-4},
        )
        resolved = resolve_svi_optimizer(config)
        assert resolved is explicit_optimizer

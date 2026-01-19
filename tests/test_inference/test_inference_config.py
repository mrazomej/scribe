"""Tests for inference_config module."""

import pytest
from scribe.inference.inference_config import create_default_inference_config
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig
from scribe.models.config.enums import InferenceMethod


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

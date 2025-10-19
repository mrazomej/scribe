"""
Tests for the ModelConfig.from_inference_params() factory method.

This module tests the factory method functionality for creating ModelConfig
instances from inference parameters, ensuring proper configuration assembly
and validation.
"""

import pytest
from src.scribe.models.model_config import ModelConfig


class TestModelConfigFactory:
    """Test cases for ModelConfig.from_inference_params() factory method."""

    def test_from_inference_params_svi_basic(self):
        """Test basic SVI configuration creation."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="svi"
        )

        assert config.base_model == "nbdm"
        assert config.inference_method == "svi"
        assert config.parameterization == "standard"
        assert config.unconstrained is False
        assert config.n_components is None
        assert config.guide_rank is None

    def test_from_inference_params_mcmc_basic(self):
        """Test basic MCMC configuration creation."""
        config = ModelConfig.from_inference_params(
            model_type="zinb",
            inference_method="mcmc",
            parameterization="linked",
        )

        assert config.base_model == "zinb"
        assert config.inference_method == "mcmc"
        assert config.parameterization == "linked"
        assert config.unconstrained is False

    def test_from_inference_params_vae_with_config(self):
        """Test VAE configuration with vae_config."""
        vae_config = {
            "vae_latent_dim": 5,
            "vae_hidden_dims": [256, 128],
            "vae_activation": "gelu",
            "vae_prior_type": "decoupled",
        }

        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="vae", vae_config=vae_config
        )

        assert config.base_model == "nbdm"
        assert config.inference_method == "vae"
        assert config.vae_latent_dim == 5
        assert config.vae_hidden_dims == [256, 128]
        assert config.vae_activation == "gelu"
        assert config.vae_prior_type == "decoupled"

    def test_from_inference_params_vae_ignores_none_config(self):
        """Test that None vae_config is handled gracefully."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="vae", vae_config=None
        )

        assert config.base_model == "nbdm"
        assert config.inference_method == "vae"
        # Should use default VAE values
        assert config.vae_latent_dim == 3
        assert config.vae_prior_type == "standard"

    def test_from_inference_params_ignores_vae_config_for_svi(self):
        """Test that vae_config is ignored when inference_method is not 'vae'."""
        vae_config = {
            "vae_latent_dim": 999,  # Should be ignored
            "vae_activation": "relu",
        }

        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="svi", vae_config=vae_config
        )

        assert config.base_model == "nbdm"
        assert config.inference_method == "svi"
        # Should use default values, not the vae_config
        assert config.vae_latent_dim == 3  # Default, not 999
        assert config.vae_activation is None  # Default, not "relu"

    def test_from_inference_params_with_prior_config(self):
        """Test that prior_config is properly merged."""
        prior_config = {
            "p_param_prior": (1.0, 1.0),
            "r_param_prior": (2.0, 0.5),
            "gate_param_prior": (1.5, 0.8),
        }

        config = ModelConfig.from_inference_params(
            model_type="zinb", inference_method="svi", prior_config=prior_config
        )

        assert config.base_model == "zinb"
        assert config.inference_method == "svi"
        assert config.p_param_prior == (1.0, 1.0)
        assert config.r_param_prior == (2.0, 0.5)
        assert config.gate_param_prior == (1.5, 0.8)

    def test_from_inference_params_with_all_params(self):
        """Test with all optional parameters provided."""
        prior_config = {
            "p_param_prior": (1.0, 1.0),
            "r_param_prior": (2.0, 0.5),
        }

        vae_config = {"vae_latent_dim": 5, "vae_hidden_dims": [256, 128]}

        config = ModelConfig.from_inference_params(
            model_type="nbdm",
            inference_method="vae",
            parameterization="linked",
            unconstrained=True,
            prior_config=prior_config,
            vae_config=vae_config,
            n_components=3,
            component_specific_params=True,
            guide_rank=10,
        )

        # Check all parameters
        assert (
            config.base_model == "nbdm_mix"
        )  # Gets modified due to n_components
        assert config.inference_method == "vae"
        assert config.parameterization == "linked"
        assert config.unconstrained is True
        assert config.n_components == 3
        assert config.component_specific_params is True
        assert config.guide_rank == 10

        # Check prior config
        assert config.p_param_prior == (1.0, 1.0)
        assert config.r_param_prior == (2.0, 0.5)

        # Check VAE config
        assert config.vae_latent_dim == 5
        assert config.vae_hidden_dims == [256, 128]

    def test_from_inference_params_auto_validates(self):
        """Test that factory method automatically validates."""
        # This should not raise an exception because validation is automatic
        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="svi"
        )

        # The fact that we got here means validation passed
        assert config is not None
        assert config.base_model == "nbdm"

    def test_from_inference_params_unconstrained(self):
        """Test unconstrained parameterization."""
        prior_config = {
            "p_unconstrained_prior": (0.0, 1.0),
            "r_unconstrained_prior": (0.0, 1.0),
        }

        config = ModelConfig.from_inference_params(
            model_type="nbdm",
            inference_method="svi",
            unconstrained=True,
            prior_config=prior_config,
        )

        assert config.unconstrained is True
        assert config.p_unconstrained_prior == (0.0, 1.0)
        assert config.r_unconstrained_prior == (0.0, 1.0)

    def test_from_inference_params_with_kwargs(self):
        """Test that additional kwargs are passed through."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm",
            inference_method="svi",
            n_components=5,
            component_specific_params=True,
            guide_rank=15,
        )

        assert config.n_components == 5
        assert config.component_specific_params is True
        assert config.guide_rank == 15

    def test_from_inference_params_mixture_model(self):
        """Test mixture model configuration."""
        config = ModelConfig.from_inference_params(
            model_type="zinb_mix",
            inference_method="svi",
            n_components=3,
            component_specific_params=True,
        )

        assert config.base_model == "zinb_mix"
        assert config.n_components == 3
        assert config.component_specific_params is True

    def test_from_inference_params_odds_ratio_parameterization(self):
        """Test odds_ratio parameterization."""
        prior_config = {
            "phi_param_prior": (1.0, 1.0),
            "mu_param_prior": (0.0, 1.0),
        }

        config = ModelConfig.from_inference_params(
            model_type="nbdm",
            inference_method="svi",
            parameterization="odds_ratio",
            prior_config=prior_config,
        )

        assert config.parameterization == "odds_ratio"
        assert config.phi_param_prior == (1.0, 1.0)
        assert config.mu_param_prior == (0.0, 1.0)

    def test_from_inference_params_vae_ignores_vae_config_for_mcmc(self):
        """Test that vae_config is ignored for MCMC inference."""
        vae_config = {"vae_latent_dim": 999, "vae_activation": "relu"}

        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="mcmc", vae_config=vae_config
        )

        assert config.inference_method == "mcmc"
        # Should use default VAE values, not the provided vae_config
        assert config.vae_latent_dim == 3  # Default
        assert config.vae_activation is None  # Default

    def test_from_inference_params_empty_prior_config(self):
        """Test with empty prior_config."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="svi", prior_config={}
        )

        assert config.base_model == "nbdm"
        assert config.inference_method == "svi"
        # Should work fine with empty dict

    def test_from_inference_params_empty_vae_config(self):
        """Test with empty vae_config for VAE inference."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="vae", vae_config={}
        )

        assert config.base_model == "nbdm"
        assert config.inference_method == "vae"
        # Should use default VAE values

    def test_from_inference_params_validation_error(self):
        """Test that validation errors are raised."""
        with pytest.raises((ValueError, TypeError)):
            # This should fail validation due to invalid model type
            ModelConfig.from_inference_params(
                model_type="invalid_model", inference_method="svi"
            )

    def test_from_inference_params_parameterization_consistency(self):
        """Test that parameterization-specific priors are handled correctly."""
        # Test linked parameterization with mu prior
        prior_config = {
            "p_param_prior": (1.0, 1.0),
            "mu_param_prior": (0.0, 1.0),  # Should be valid for linked
        }

        config = ModelConfig.from_inference_params(
            model_type="nbdm",
            inference_method="svi",
            parameterization="linked",
            prior_config=prior_config,
        )

        assert config.parameterization == "linked"
        assert config.p_param_prior == (1.0, 1.0)
        assert config.mu_param_prior == (0.0, 1.0)

    def test_from_inference_params_vae_prior_type_handling(self):
        """Test VAE prior type handling in factory method."""
        vae_config = {
            "vae_prior_type": "decoupled",
            "vae_prior_num_layers": 3,
            "vae_prior_hidden_dims": [64, 64],
        }

        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="vae", vae_config=vae_config
        )

        assert config.vae_prior_type == "decoupled"
        assert config.vae_prior_num_layers == 3
        assert config.vae_prior_hidden_dims == [64, 64]

    def test_from_inference_params_guide_rank_handling(self):
        """Test guide_rank parameter handling."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm", inference_method="svi", guide_rank=20
        )

        assert config.guide_rank == 20

    def test_from_inference_params_component_specific_params(self):
        """Test component_specific_params handling."""
        config = ModelConfig.from_inference_params(
            model_type="nbdm_mix",
            inference_method="svi",
            n_components=4,
            component_specific_params=True,
        )

        assert config.component_specific_params is True
        assert config.n_components == 4

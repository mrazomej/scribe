"""Tests for the new ModelConfig system."""

import pytest
from src.scribe.models.config import (
    ModelConfigBuilder,
    ConstrainedModelConfig,
    UnconstrainedModelConfig,
    ModelType,
    Parameterization,
    InferenceMethod,
    VAEPriorType,
    VAEActivation,
)


class TestModelConfigBuilder:
    """Test the builder pattern."""

    def test_simple_svi_config(self):
        """Test building a simple SVI configuration."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_inference("svi")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert isinstance(config, ConstrainedModelConfig)
        assert config.base_model == "nbdm"
        assert config.inference_method == InferenceMethod.SVI
        assert config.parameterization == Parameterization.STANDARD

    def test_unconstrained_config(self):
        """Test unconstrained configuration."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("linked")
            .unconstrained()
            .with_priors(p=(1.0, 1.0), mu=(1.0, 1.0), gate=(1.0, 1.0))
            .build()
        )

        assert isinstance(config, UnconstrainedModelConfig)
        assert config.parameterization == Parameterization.LINKED

    def test_mixture_config(self):
        """Test mixture model configuration."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .as_mixture(n_components=3)
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), mixing=(1.0, 1.0, 1.0))
            .build()
        )

        assert config.base_model == "nbdm_mix"
        assert config.n_components == 3
        assert config.is_mixture is True

    def test_vae_config(self):
        """Test VAE configuration."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_vae(latent_dim=5, hidden_dims=[256, 128], activation="gelu")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), gate=(1.0, 1.0))
            .build()
        )

        assert config.inference_method == InferenceMethod.VAE
        assert config.vae is not None
        assert config.vae.latent_dim == 5
        assert config.vae.hidden_dims == [256, 128]
        assert config.vae.activation == VAEActivation.GELU

    def test_with_priors(self):
        """Test setting priors."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config.priors.p == (1.0, 1.0)
        assert config.priors.r == (2.0, 0.5)

    def test_fluent_chaining(self):
        """Test fluent method chaining."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("linked")
            .unconstrained()
            .as_mixture(n_components=3, component_specific=True)
            .with_low_rank_guide(10)
            .with_priors(
                p=(1.0, 1.0),
                mu=(1.0, 1.0),
                gate=(1.0, 1.0),
                mixing=(1.0, 1.0, 1.0),
            )
            .with_vae(latent_dim=5)
            .build()
        )

        assert isinstance(config, UnconstrainedModelConfig)
        assert config.base_model == "zinb_mix"
        assert config.parameterization == Parameterization.LINKED
        assert config.n_components == 3
        assert config.component_specific_params is True
        assert config.guide_rank == 10
        assert config.priors.p == (1.0, 1.0)
        assert config.priors.mu == (1.0, 1.0)
        assert config.vae.latent_dim == 5

    def test_immutability(self):
        """Test that configs are immutable."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            config.base_model = "zinb"

    def test_validation_errors(self):
        """Test that invalid configs raise errors."""
        # Missing base_model
        with pytest.raises(ValueError, match="base_model is required"):
            ModelConfigBuilder().build()

        # Invalid n_components
        with pytest.raises(ValueError, match="n_components must be >= 2"):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .as_mixture(n_components=1)
                .build()
            )

        # Invalid guide_rank
        with pytest.raises(ValueError, match="guide_rank must be positive"):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_low_rank_guide(0)
                .build()
            )

    def test_enum_usage(self):
        """Test using enums instead of strings."""
        config = (
            ModelConfigBuilder()
            .for_model(ModelType.NBDM)
            .with_parameterization(Parameterization.LINKED)
            .with_inference(InferenceMethod.VAE)
            .with_vae(prior_type=VAEPriorType.DECOUPLED)
            .with_priors(p=(1.0, 1.0), mu=(1.0, 1.0))
            .build()
        )

        assert config.base_model == "nbdm"
        assert config.parameterization == Parameterization.LINKED
        assert config.inference_method == InferenceMethod.VAE
        assert config.vae.prior_type == VAEPriorType.DECOUPLED


class TestComputedFields:
    """Test computed fields."""

    def test_is_mixture(self):
        """Test is_mixture computed field."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .as_mixture(n_components=3)
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), mixing=(1.0, 1.0, 1.0))
            .build()
        )

        assert config.is_mixture is True

    def test_is_zero_inflated(self):
        """Test is_zero_inflated computed field."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), gate=(1.0, 1.0))
            .build()
        )

        assert config.is_zero_inflated is True

    def test_uses_variable_capture(self):
        """Test uses_variable_capture computed field."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), p_capture=(1.0, 1.0))
            .build()
        )

        assert config.uses_variable_capture is True

    def test_active_parameters(self):
        """Test active_parameters computed field."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("linked")
            .with_priors(p=(1.0, 1.0), mu=(1.0, 1.0), gate=(1.0, 1.0))
            .build()
        )

        params = config.active_parameters
        assert "p" in params
        assert "mu" in params
        assert "gate" in params
        assert "r" not in params  # Not used in linked parameterization


class TestImmutablePatterns:
    """Test immutable update patterns."""

    def test_model_copy_pattern(self):
        """Test using model_copy for updates."""
        config1 = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        # Use model_copy to create a new instance with updated priors
        config2 = config1.model_copy(
            update={
                "priors": config1.priors.model_copy(update={"r": (3.0, 1.0)})
            }
        )

        assert config1.priors.r == (2.0, 0.5)
        assert config2.priors.r == (3.0, 1.0)
        assert config2.priors.p == (1.0, 1.0)
        assert config1 is not config2


class TestValidation:
    """Test validation features."""

    def test_prior_validation(self):
        """Test that priors are validated."""
        # Valid priors
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config.priors.p == (1.0, 1.0)
        assert config.priors.r == (2.0, 0.5)

    def test_invalid_priors(self):
        """Test that invalid priors raise errors."""
        # Negative parameters
        with pytest.raises(
            ValueError, match="Prior parameters must be positive"
        ):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_priors(p=(-1.0, 1.0))
                .build()
            )

        # Wrong tuple length - this now raises a different error
        with pytest.raises(Exception):  # Pydantic validation error
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_priors(p=(1.0, 1.0, 1.0))
                .build()
            )

    def test_vae_validation(self):
        """Test VAE parameter validation."""
        # Valid VAE config
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_vae(latent_dim=5, hidden_dims=[256, 128])
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config.vae.latent_dim == 5
        assert config.vae.hidden_dims == [256, 128]

    def test_invalid_vae_params(self):
        """Test that invalid VAE parameters raise errors."""
        # Negative latent dim
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_vae(latent_dim=0)
                .build()
            )

        # Negative hidden dims
        with pytest.raises(
            ValueError, match="Hidden dimensions must be positive"
        ):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_vae(hidden_dims=[256, -128])
                .build()
            )


class TestModelTypeValidation:
    """Test model type validation."""

    def test_valid_model_types(self):
        """Test that valid model types work."""
        for model_type in ["nbdm", "zinb", "nbvcp", "zinbvcp"]:
            if model_type == "nbdm":
                config = (
                    ModelConfigBuilder()
                    .for_model(model_type)
                    .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
                    .build()
                )
            elif model_type == "zinb":
                config = (
                    ModelConfigBuilder()
                    .for_model(model_type)
                    .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), gate=(1.0, 1.0))
                    .build()
                )
            elif model_type == "nbvcp":
                config = (
                    ModelConfigBuilder()
                    .for_model(model_type)
                    .with_priors(
                        p=(1.0, 1.0), r=(2.0, 0.5), p_capture=(1.0, 1.0)
                    )
                    .build()
                )
            elif model_type == "zinbvcp":
                config = (
                    ModelConfigBuilder()
                    .for_model(model_type)
                    .with_priors(
                        p=(1.0, 1.0),
                        r=(2.0, 0.5),
                        gate=(1.0, 1.0),
                        p_capture=(1.0, 1.0),
                    )
                    .build()
                )
            assert config.base_model == model_type

    def test_invalid_model_type(self):
        """Test that invalid model types raise errors."""
        with pytest.raises(ValueError, match="Invalid model type"):
            (ModelConfigBuilder().for_model("invalid_model").build())

    def test_mixture_model_validation(self):
        """Test mixture model validation."""
        # Valid mixture
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .as_mixture(n_components=3)
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5), mixing=(1.0, 1.0, 1.0))
            .build()
        )

        assert config.base_model == "nbdm_mix"
        assert config.n_components == 3


class TestParameterizationSpecific:
    """Test parameterization-specific behavior."""

    def test_standard_parameterization(self):
        """Test standard parameterization active parameters."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("standard")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        params = config.active_parameters
        assert "p" in params
        assert "r" in params
        assert "mu" not in params
        assert "phi" not in params

    def test_linked_parameterization(self):
        """Test linked parameterization active parameters."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("linked")
            .with_priors(p=(1.0, 1.0), mu=(1.0, 1.0))
            .build()
        )

        params = config.active_parameters
        assert "p" in params
        assert "mu" in params
        assert "r" not in params
        assert "phi" not in params

    def test_odds_ratio_parameterization(self):
        """Test odds_ratio parameterization active parameters."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("odds_ratio")
            .with_priors(phi=(1.0, 1.0), mu=(1.0, 1.0))
            .build()
        )

        params = config.active_parameters
        assert "p" not in params
        assert "mu" in params
        assert "phi" in params
        assert "r" not in params


class TestVAEConfiguration:
    """Test VAE-specific configuration."""

    def test_vae_auto_inference_method(self):
        """Test that VAE config automatically sets inference method."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_vae(latent_dim=5)
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config.inference_method == InferenceMethod.VAE

    def test_vae_default_config(self):
        """Test VAE default configuration."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_inference("vae")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config.vae is not None
        assert config.vae.latent_dim == 3
        assert config.vae.prior_type == VAEPriorType.STANDARD

    def test_vae_prior_types(self):
        """Test different VAE prior types."""
        # Standard prior
        config1 = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_vae(prior_type="standard")
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config1.vae.prior_type == VAEPriorType.STANDARD

        # Decoupled prior
        config2 = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_vae(prior_type="decoupled", prior_num_layers=3)
            .with_priors(p=(1.0, 1.0), r=(2.0, 0.5))
            .build()
        )

        assert config2.vae.prior_type == VAEPriorType.DECOUPLED
        assert config2.vae.prior_num_layers == 3


class TestComplexConfigurations:
    """Test complex configuration scenarios."""

    def test_zinb_mixture_unconstrained(self):
        """Test ZINB mixture model with unconstrained parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("linked")
            .unconstrained()
            .as_mixture(n_components=3, component_specific=True)
            .with_low_rank_guide(10)
            .with_priors(
                p=(1.0, 1.0),
                mu=(1.0, 1.0),
                gate=(2.0, 2.0),
                mixing=(1.0, 1.0, 1.0),
            )
            .build()
        )

        assert isinstance(config, UnconstrainedModelConfig)
        assert config.base_model == "zinb_mix"
        assert config.parameterization == Parameterization.LINKED
        assert config.n_components == 3
        assert config.component_specific_params is True
        assert config.guide_rank == 10
        assert config.priors.p == (1.0, 1.0)
        assert config.priors.mu == (1.0, 1.0)
        assert config.priors.gate == (2.0, 2.0)

    def test_vcp_model_with_odds_ratio(self):
        """Test VCP model with odds_ratio parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("odds_ratio")
            .with_priors(phi=(1.0, 1.0), mu=(1.0, 1.0), phi_capture=(2.0, 0.5))
            .build()
        )

        assert config.base_model == "nbvcp"
        assert config.parameterization == Parameterization.ODDS_RATIO
        assert config.priors.phi == (1.0, 1.0)
        assert config.priors.mu == (1.0, 1.0)
        assert config.priors.phi_capture == (2.0, 0.5)

        # Check active parameters
        params = config.active_parameters
        assert "phi" in params
        assert "mu" in params
        assert "phi_capture" in params
        assert "p" not in params
        assert "r" not in params


# ==============================================================================
# Test Default Priors
# ==============================================================================


class TestDefaultPriors:
    """Test that default priors are set correctly."""

    def test_constrained_defaults(self):
        """Test default priors for constrained models."""
        config = ModelConfigBuilder().for_model("nbdm").build()

        assert config.priors.p == (1.0, 1.0)
        assert config.priors.r == (0.0, 1.0)

    def test_unconstrained_defaults(self):
        """Test default priors for unconstrained models."""
        config = ModelConfigBuilder().for_model("nbdm").unconstrained().build()

        assert config.priors.p == (0.0, 1.0)
        assert config.priors.r == (0.0, 1.0)

    def test_mixture_defaults(self):
        """Test default priors for mixture models."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .as_mixture(n_components=3)
            .build()
        )

        assert config.priors.mixing == (1.0, 1.0, 1.0)

    def test_user_priors_override_defaults(self):
        """Test that user-provided priors override defaults."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_priors(p=(2.0, 3.0))
            .build()
        )

        assert config.priors.p == (2.0, 3.0)  # User value
        assert config.priors.r == (0.0, 1.0)  # Default value

    def test_zinb_defaults(self):
        """Test default priors for zero-inflated models."""
        config = ModelConfigBuilder().for_model("zinb").build()

        assert config.priors.p == (1.0, 1.0)
        assert config.priors.r == (0.0, 1.0)
        assert config.priors.gate == (1.0, 1.0)

    def test_vcp_defaults(self):
        """Test default priors for variable capture models."""
        config = ModelConfigBuilder().for_model("nbvcp").build()

        assert config.priors.p == (1.0, 1.0)
        assert config.priors.r == (0.0, 1.0)
        assert config.priors.p_capture == (1.0, 1.0)

    def test_odds_ratio_defaults(self):
        """Test default priors for odds ratio parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("odds_ratio")
            .build()
        )

        assert config.priors.phi == (1.0, 1.0)
        assert config.priors.mu == (0.0, 1.0)

    def test_unconstrained_mixture_defaults(self):
        """Test default priors for unconstrained mixture models."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .unconstrained()
            .as_mixture(n_components=2)
            .build()
        )

        assert config.priors.p == (0.0, 1.0)
        assert config.priors.r == (0.0, 1.0)
        assert config.priors.mixing == (0.0, 0.0)

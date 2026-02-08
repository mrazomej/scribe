"""Tests for the unified model factory.

This module tests that the unified factory (`create_model`) produces
correct model and guide functions for all model types and configurations.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from scribe.models.config import GuideFamilyConfig, ModelConfigBuilder
from scribe.models.presets import (
    create_model,
    create_model_from_params,
)
from scribe.models.presets.registry import (
    LIKELIHOOD_REGISTRY,
    MODEL_EXTRA_PARAMS,
    apply_prior_guide_overrides,
    build_capture_spec,
    build_gate_spec,
)


# ==============================================================================
# Test Registry Contents
# ==============================================================================


class TestRegistryContents:
    """Test that registries contain expected entries."""

    def test_model_extra_params_contains_all_models(self):
        """Test MODEL_EXTRA_PARAMS has entries for all model types."""
        expected_models = {"nbdm", "zinb", "nbvcp", "zinbvcp"}
        assert set(MODEL_EXTRA_PARAMS.keys()) == expected_models

    def test_model_extra_params_values(self):
        """Test MODEL_EXTRA_PARAMS has correct extra parameters."""
        assert MODEL_EXTRA_PARAMS["nbdm"] == []
        assert MODEL_EXTRA_PARAMS["zinb"] == ["gate"]
        assert MODEL_EXTRA_PARAMS["nbvcp"] == ["p_capture"]
        assert MODEL_EXTRA_PARAMS["zinbvcp"] == ["gate", "p_capture"]

    def test_likelihood_registry_contains_all_models(self):
        """Test LIKELIHOOD_REGISTRY has entries for all model types."""
        expected_models = {"nbdm", "zinb", "nbvcp", "zinbvcp"}
        assert set(LIKELIHOOD_REGISTRY.keys()) == expected_models

    def test_likelihood_registry_classes(self):
        """Test LIKELIHOOD_REGISTRY contains correct likelihood classes."""
        from scribe.models.components.likelihoods import (
            NBWithVCPLikelihood,
            NegativeBinomialLikelihood,
            ZeroInflatedNBLikelihood,
            ZINBWithVCPLikelihood,
        )

        assert LIKELIHOOD_REGISTRY["nbdm"] == NegativeBinomialLikelihood
        assert LIKELIHOOD_REGISTRY["zinb"] == ZeroInflatedNBLikelihood
        assert LIKELIHOOD_REGISTRY["nbvcp"] == NBWithVCPLikelihood
        assert LIKELIHOOD_REGISTRY["zinbvcp"] == ZINBWithVCPLikelihood


# ==============================================================================
# Test Helper Builders
# ==============================================================================


class TestHelperBuilders:
    """Test helper builder functions."""

    def test_build_gate_spec_constrained(self):
        """Test build_gate_spec with constrained parameterization."""
        spec = build_gate_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
        )
        assert spec.name == "gate"
        assert spec.is_gene_specific is True
        assert spec.is_mixture is False

    def test_build_gate_spec_unconstrained(self):
        """Test build_gate_spec with unconstrained parameterization."""
        from scribe.models.builders.parameter_specs import SigmoidNormalSpec

        spec = build_gate_spec(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
        )
        assert spec.name == "gate"
        # When unconstrained=True, we get SigmoidNormalSpec instead of BetaSpec
        assert isinstance(spec, SigmoidNormalSpec)

    def test_build_gate_spec_mixture(self):
        """Test build_gate_spec with mixture model."""
        spec = build_gate_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            n_components=3,
            mixture_params=None,  # Default: gate should be mixture
        )
        assert spec.is_mixture is True

    def test_build_capture_spec_constrained_canonical(self):
        """Test build_capture_spec with canonical parameterization."""
        from scribe.models.parameterizations import PARAMETERIZATIONS

        spec = build_capture_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            param_strategy=PARAMETERIZATIONS["canonical"],
        )
        assert spec.name == "p_capture"
        assert spec.is_cell_specific is True

    def test_build_capture_spec_mean_odds(self):
        """Test build_capture_spec with mean_odds parameterization."""
        from scribe.models.parameterizations import PARAMETERIZATIONS

        spec = build_capture_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            param_strategy=PARAMETERIZATIONS["mean_odds"],
        )
        # mean_odds transforms p_capture to phi_capture
        assert spec.name == "phi_capture"


class TestApplyOverrides:
    """Test apply_prior_guide_overrides function."""

    def test_apply_priors(self):
        """Test applying prior overrides."""
        from scribe.models.builders.parameter_specs import BetaSpec

        specs = [
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0)),
        ]
        updated = apply_prior_guide_overrides(specs, priors={"p": (2.0, 2.0)})
        assert updated[0].prior == (2.0, 2.0)

    def test_no_overrides(self):
        """Test that specs are unchanged when no overrides provided."""
        from scribe.models.builders.parameter_specs import BetaSpec

        specs = [
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0)),
        ]
        updated = apply_prior_guide_overrides(specs, priors=None, guides=None)
        assert updated[0].prior is None  # No override applied


# ==============================================================================
# Test Unified Factory
# ==============================================================================


class TestUnifiedFactory:
    """Test create_model produces callable model and guide functions."""

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_create_model_returns_callables(self, model_type):
        """Test that create_model returns callable functions."""
        config = ModelConfigBuilder().for_model(model_type).build()
        model, guide = create_model(config)

        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize(
        "parameterization", ["canonical", "mean_prob", "mean_odds"]
    )
    def test_create_model_parameterizations(self, parameterization):
        """Test create_model works with all parameterizations."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization(parameterization)
            .build()
        )
        model, guide = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_create_model_unconstrained(self):
        """Test create_model with unconstrained parameterization."""
        config = ModelConfigBuilder().for_model("zinb").unconstrained().build()
        model, guide = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_create_model_mixture(self):
        """Test create_model with mixture model."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .as_mixture(n_components=3)
            .build()
        )
        model, guide = create_model(config)
        assert callable(model)
        assert callable(guide)

    def test_create_model_with_priors(self):
        """Test create_model with custom priors."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        model, guide = create_model(
            config, priors={"p": (2.0, 2.0), "r": (1.0, 0.5)}
        )
        assert callable(model)
        assert callable(guide)


class TestCreateModelFromParams:
    """Test create_model_from_params convenience function."""

    def test_basic_usage(self):
        """Test basic usage of create_model_from_params."""
        model, guide = create_model_from_params(model="nbdm")
        assert callable(model)
        assert callable(guide)

    def test_with_options(self):
        """Test create_model_from_params with options."""
        model, guide = create_model_from_params(
            model="zinb",
            parameterization="linked",
            unconstrained=True,
            n_components=3,
        )
        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_all_model_types(self, model_type):
        """Test create_model_from_params works with all model types."""
        model, guide = create_model_from_params(model=model_type)
        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize(
        "parameterization", ["canonical", "mean_prob", "mean_odds"]
    )
    def test_all_parameterizations(self, parameterization):
        """Test create_model_from_params works with all parameterizations."""
        model, guide = create_model_from_params(
            model="nbdm", parameterization=parameterization
        )
        assert callable(model)
        assert callable(guide)


# ==============================================================================
# Test Factory with Different Configurations
# ==============================================================================


class TestFactoryConfigurations:
    """Test factory with various configuration combinations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing model execution."""
        n_cells = 50
        n_genes = 20
        rng = random.PRNGKey(42)
        # Generate some fake count data
        counts = random.poisson(rng, lam=10.0, shape=(n_cells, n_genes))
        return counts, n_cells, n_genes

    @pytest.fixture
    def mock_model_config(self):
        """Create a basic ModelConfig for testing."""
        return ModelConfigBuilder().for_model("nbdm").build()

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    def test_factory_all_models(self, model_type):
        """Test factory creates models for all model types."""
        config = ModelConfigBuilder().for_model(model_type).build()
        model, guide = create_model(config)

        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize(
        "parameterization", ["canonical", "mean_prob", "mean_odds"]
    )
    def test_factory_all_parameterizations(self, parameterization):
        """Test factory handles all parameterizations."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization(parameterization)
            .build()
        )
        model, guide = create_model(config)

        assert callable(model)
        assert callable(guide)

    def test_factory_mixture_model(self):
        """Test factory handles mixture models."""
        n_components = 3

        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .as_mixture(n_components=n_components)
            .build()
        )
        model, guide = create_model(config)

        assert callable(model)
        assert callable(guide)

    def test_factory_unconstrained(self):
        """Test factory handles unconstrained models."""
        config = ModelConfigBuilder().for_model("nbvcp").unconstrained().build()
        model, guide = create_model(config)

        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize(
        "model_type,parameterization,unconstrained",
        [
            ("nbdm", "canonical", False),
            ("nbdm", "mean_prob", True),
            ("zinb", "mean_odds", False),
            ("nbvcp", "canonical", True),
            ("zinbvcp", "mean_prob", False),
        ],
    )
    def test_factory_combinations(
        self, model_type, parameterization, unconstrained
    ):
        """Test factory handles various combinations of options."""
        builder = (
            ModelConfigBuilder()
            .for_model(model_type)
            .with_parameterization(parameterization)
        )
        if unconstrained:
            builder.unconstrained()

        config = builder.build()
        model, guide = create_model(config)

        assert callable(model)
        assert callable(guide)


# ==============================================================================
# Test Integration with ModelConfig
# ==============================================================================


class TestModelConfigIntegration:
    """Test integration between ModelConfig and unified factory."""

    def test_config_with_priors(self):
        """Test that priors are applied."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        model, guide = create_model(config, priors={"p": (2.0, 2.0)})
        assert callable(model)

    def test_get_prior_overrides_method(self):
        """Test ModelConfig.get_prior_overrides method."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        priors = config.get_prior_overrides()
        assert isinstance(priors, dict)

    def test_get_guide_overrides_method(self):
        """Test ModelConfig.get_guide_overrides method."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        guides = config.get_guide_overrides()
        assert isinstance(guides, dict)

    def test_model_config_properties(self):
        """Test ModelConfig computed properties."""
        # Test ZINB mixture model
        config = (
            ModelConfigBuilder()
            .for_model("zinbvcp")
            .as_mixture(n_components=3)
            .build()
        )

        assert config.is_mixture is True
        assert config.is_zero_inflated is True
        assert config.uses_variable_capture is True

        # Test plain NBDM
        config2 = ModelConfigBuilder().for_model("nbdm").build()
        assert config2.is_mixture is False
        assert config2.is_zero_inflated is False
        assert config2.uses_variable_capture is False


# ==============================================================================
# Test Parameter Name Validation
# ==============================================================================


class TestPriorGuideValidation:
    """Test validation of prior/guide parameter names."""

    def test_unknown_prior_param_raises_error(self):
        """Test that unknown prior parameter names raise ValueError."""
        from scribe.models.builders.parameter_specs import BetaSpec

        specs = [
            BetaSpec(
                name="p",
                shape_dims=("n_genes",),
                default_params=(1.0, 1.0),
                prior=(1.0, 1.0),
            )
        ]
        with pytest.raises(
            ValueError, match="Unknown parameter names in priors"
        ):
            apply_prior_guide_overrides(
                specs,
                priors={"unknown_param": (2.0, 2.0)},
            )

    def test_unknown_guide_param_raises_error(self):
        """Test that unknown guide parameter names raise ValueError."""
        from scribe.models.builders.parameter_specs import BetaSpec

        specs = [
            BetaSpec(
                name="p",
                shape_dims=("n_genes",),
                default_params=(1.0, 1.0),
                prior=(1.0, 1.0),
            )
        ]
        with pytest.raises(
            ValueError, match="Unknown parameter names in guides"
        ):
            apply_prior_guide_overrides(
                specs,
                guides={"invalid_param": (0.5, 0.5)},
            )

    def test_valid_prior_params_succeed(self):
        """Test that valid prior parameter names work."""
        from scribe.models.builders.parameter_specs import (
            BetaSpec,
            LogNormalSpec,
        )

        specs = [
            BetaSpec(
                name="p",
                shape_dims=("n_genes",),
                default_params=(1.0, 1.0),
                prior=(1.0, 1.0),
            ),
            LogNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                prior=(0.0, 1.0),
            ),
        ]
        result = apply_prior_guide_overrides(
            specs,
            priors={"p": (2.0, 2.0), "r": (1.0, 0.5)},
        )
        assert len(result) == 2
        assert result[0].prior == (2.0, 2.0)
        assert result[1].prior == (1.0, 0.5)

    def test_factory_validates_prior_params(self):
        """Test that create_model validates prior parameter names."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        with pytest.raises(ValueError, match="Unknown parameter names"):
            create_model(
                config, priors={"nonexistent": (1.0, 1.0)}, validate=False
            )


# ==============================================================================
# Test Model/Guide Validation
# ==============================================================================


class TestModelGuideValidation:
    """Test the validate_model_guide_compatibility function."""

    def test_validation_runs_successfully(self):
        """Test that validation passes for correct model/guide pairs."""
        from scribe.models.presets.factory import (
            validate_model_guide_compatibility,
        )

        config = ModelConfigBuilder().for_model("nbdm").build()
        model, guide = create_model(config, validate=False)
        # Should not raise
        validate_model_guide_compatibility(model, guide, config)

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        # Should not raise even with validate=False
        model, guide = create_model(config, validate=False)
        assert callable(model)
        assert callable(guide)

    def test_validation_with_different_models(self):
        """Test validation works for different model types."""
        for model_type in ["nbdm", "zinb", "nbvcp", "zinbvcp"]:
            config = ModelConfigBuilder().for_model(model_type).build()
            # Should not raise
            model, guide = create_model(config, validate=True)
            assert callable(model)


# ==============================================================================
# Test Guide Family Registry
# ==============================================================================


class TestGuideFamilyRegistry:
    """Test the guide family registry."""

    def test_registry_contains_families(self):
        """Test GUIDE_FAMILY_REGISTRY contains expected families."""
        from scribe.models.presets.registry import GUIDE_FAMILY_REGISTRY

        assert "mean_field" in GUIDE_FAMILY_REGISTRY
        assert "low_rank" in GUIDE_FAMILY_REGISTRY
        assert "amortized" in GUIDE_FAMILY_REGISTRY

    def test_get_guide_family_mean_field(self):
        """Test get_guide_family returns MeanFieldGuide."""
        from scribe.models.components.guide_families import MeanFieldGuide
        from scribe.models.presets.registry import get_guide_family

        guide = get_guide_family("mean_field")
        assert isinstance(guide, MeanFieldGuide)

    def test_get_guide_family_low_rank(self):
        """Test get_guide_family returns LowRankGuide with rank."""
        from scribe.models.components.guide_families import LowRankGuide
        from scribe.models.presets.registry import get_guide_family

        guide = get_guide_family("low_rank", rank=10)
        assert isinstance(guide, LowRankGuide)
        assert guide.rank == 10

    def test_get_guide_family_invalid(self):
        """Test get_guide_family raises error for invalid name."""
        from scribe.models.presets.registry import get_guide_family

        with pytest.raises(ValueError, match="Unknown guide family"):
            get_guide_family("invalid_family")


# ==============================================================================
# Test Config YAML Serialization
# ==============================================================================


class TestConfigSerialization:
    """Test YAML serialization for config classes."""

    def test_model_config_to_yaml(self):
        """Test ModelConfig.to_yaml produces valid YAML."""
        config = ModelConfigBuilder().for_model("nbdm").build()
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        assert "base_model: nbdm" in yaml_str

    def test_model_config_from_yaml(self):
        """Test ModelConfig.from_yaml loads config correctly."""
        from scribe.models.config import ModelConfig

        yaml_str = """
base_model: zinb
parameterization: linked
unconstrained: false
"""
        config = ModelConfig.from_yaml(yaml_str)
        assert config.base_model == "zinb"
        assert config.parameterization.value == "linked"
        assert config.unconstrained is False

    def test_model_config_roundtrip(self):
        """Test ModelConfig serialization roundtrip."""
        from scribe.models.config import ModelConfig

        original = ModelConfigBuilder().for_model("nbvcp").build()
        yaml_str = original.to_yaml()
        loaded = ModelConfig.from_yaml(yaml_str)
        assert loaded.base_model == original.base_model
        assert loaded.parameterization == original.parameterization
        assert loaded.unconstrained == original.unconstrained

    def test_inference_config_to_yaml(self):
        """Test InferenceConfig.to_yaml produces valid YAML."""
        from scribe.models.config import InferenceConfig, SVIConfig

        config = InferenceConfig.from_svi(SVIConfig(n_steps=50000))
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)
        assert "method: svi" in yaml_str
        assert "n_steps: 50000" in yaml_str

    def test_inference_config_from_yaml(self):
        """Test InferenceConfig.from_yaml loads config correctly."""
        from scribe.models.config import InferenceConfig

        yaml_str = """
method: svi
svi:
  n_steps: 25000
  batch_size: 256
  stable_update: true
mcmc: null
"""
        config = InferenceConfig.from_yaml(yaml_str)
        assert config.method.value == "svi"
        assert config.svi.n_steps == 25000
        assert config.svi.batch_size == 256

    def test_inference_config_roundtrip(self):
        """Test InferenceConfig serialization roundtrip."""
        from scribe.models.config import InferenceConfig, SVIConfig

        original = InferenceConfig.from_svi(
            SVIConfig(n_steps=75000, batch_size=512)
        )
        yaml_str = original.to_yaml()
        loaded = InferenceConfig.from_yaml(yaml_str)
        assert loaded.method == original.method
        assert loaded.svi.n_steps == original.svi.n_steps
        assert loaded.svi.batch_size == original.svi.batch_size

    def test_svi_config_serialization(self):
        """Test SVIConfig standalone serialization."""
        from scribe.models.config import SVIConfig

        original = SVIConfig(n_steps=10000, batch_size=128)
        yaml_str = original.to_yaml()
        loaded = SVIConfig.from_yaml(yaml_str)
        assert loaded.n_steps == 10000
        assert loaded.batch_size == 128

    def test_mcmc_config_serialization(self):
        """Test MCMCConfig standalone serialization."""
        from scribe.models.config import MCMCConfig

        original = MCMCConfig(n_samples=5000, n_warmup=2000, n_chains=4)
        yaml_str = original.to_yaml()
        loaded = MCMCConfig.from_yaml(yaml_str)
        assert loaded.n_samples == 5000
        assert loaded.n_warmup == 2000
        assert loaded.n_chains == 4


# ==============================================================================
# Test Amortization Configuration
# ==============================================================================


class TestAmortizationConfig:
    """Test AmortizationConfig validation and serialization."""

    def test_default_values(self):
        """Test default AmortizationConfig values."""
        from scribe.models.config import AmortizationConfig

        config = AmortizationConfig()
        assert config.enabled is False
        assert config.hidden_dims == [64, 32]
        assert config.activation == "relu"
        assert config.input_transformation == "log1p"
        assert config.output_transform == "softplus"
        assert config.output_clamp_min == 0.1
        assert config.output_clamp_max == 50.0

    def test_custom_values(self):
        """Test AmortizationConfig with custom values."""
        from scribe.models.config import AmortizationConfig

        config = AmortizationConfig(
            enabled=True,
            hidden_dims=[128, 64, 32],
            activation="gelu",
            input_transformation="sqrt",
            output_transform="exp",
            output_clamp_min=0.5,
            output_clamp_max=100.0,
        )
        assert config.enabled is True
        assert config.hidden_dims == [128, 64, 32]
        assert config.activation == "gelu"
        assert config.input_transformation == "sqrt"
        assert config.output_transform == "exp"
        assert config.output_clamp_min == 0.5
        assert config.output_clamp_max == 100.0

    def test_hidden_dims_validation_empty(self):
        """Test that empty hidden_dims raises error."""
        from scribe.models.config import AmortizationConfig

        with pytest.raises(ValueError, match="at least one layer"):
            AmortizationConfig(hidden_dims=[])

    def test_hidden_dims_validation_negative(self):
        """Test that negative hidden_dims raises error."""
        from scribe.models.config import AmortizationConfig

        with pytest.raises(ValueError, match="must be positive"):
            AmortizationConfig(hidden_dims=[64, -32])

    def test_activation_validation(self):
        """Test that invalid activation raises error."""
        from scribe.models.config import AmortizationConfig

        with pytest.raises(ValueError, match="Unknown activation"):
            AmortizationConfig(activation="invalid_activation")

    def test_activation_case_insensitive(self):
        """Test that activation is case-insensitive."""
        from scribe.models.config import AmortizationConfig

        config = AmortizationConfig(activation="GELU")
        assert config.activation == "gelu"

    def test_input_transformation_validation(self):
        """Test that invalid input_transformation raises error."""
        from scribe.models.config import AmortizationConfig

        with pytest.raises(ValueError, match="Unknown transformation"):
            AmortizationConfig(input_transformation="invalid")

    def test_output_transform_validation(self):
        """Test that invalid output_transform raises error."""
        from scribe.models.config import AmortizationConfig

        with pytest.raises(ValueError, match="Unknown output_transform"):
            AmortizationConfig(output_transform="invalid")

    def test_output_transform_case_insensitive(self):
        """Test that output_transform is case-insensitive."""
        from scribe.models.config import AmortizationConfig

        config = AmortizationConfig(output_transform="SOFTPLUS")
        assert config.output_transform == "softplus"

    def test_clamp_range_validation(self):
        """Test that clamp_min >= clamp_max raises error."""
        from scribe.models.config import AmortizationConfig

        with pytest.raises(ValueError, match="must be less than"):
            AmortizationConfig(output_clamp_min=50.0, output_clamp_max=10.0)

        with pytest.raises(ValueError, match="must be less than"):
            AmortizationConfig(output_clamp_min=10.0, output_clamp_max=10.0)

    def test_clamp_none_disables(self):
        """Test that None clamp values are accepted."""
        from scribe.models.config import AmortizationConfig

        config = AmortizationConfig(
            output_clamp_min=None, output_clamp_max=None
        )
        assert config.output_clamp_min is None
        assert config.output_clamp_max is None

    def test_serialization_roundtrip(self):
        """Test AmortizationConfig YAML serialization roundtrip."""
        from scribe.models.config import AmortizationConfig

        original = AmortizationConfig(
            enabled=True,
            hidden_dims=[128, 64],
            activation="silu",
            input_transformation="log",
            output_transform="exp",
            output_clamp_min=0.5,
            output_clamp_max=100.0,
        )
        yaml_str = original.to_yaml()
        loaded = AmortizationConfig.from_yaml(yaml_str)

        assert loaded.enabled == original.enabled
        assert loaded.hidden_dims == original.hidden_dims
        assert loaded.activation == original.activation
        assert loaded.input_transformation == original.input_transformation
        assert loaded.output_transform == original.output_transform
        assert loaded.output_clamp_min == original.output_clamp_min
        assert loaded.output_clamp_max == original.output_clamp_max


class TestGuideFamilyConfigWithAmortization:
    """Test GuideFamilyConfig with capture_amortization field."""

    def test_capture_amortization_none_by_default(self):
        """Test capture_amortization is None by default."""
        config = GuideFamilyConfig()
        assert config.capture_amortization is None

    def test_capture_amortization_can_be_set(self):
        """Test capture_amortization can be set."""
        from scribe.models.config import AmortizationConfig

        amort_config = AmortizationConfig(enabled=True, hidden_dims=[32, 16])
        config = GuideFamilyConfig(capture_amortization=amort_config)

        assert config.capture_amortization is not None
        assert config.capture_amortization.enabled is True
        assert config.capture_amortization.hidden_dims == [32, 16]


# ==============================================================================
# Test Amortizer Factory
# ==============================================================================


class TestAmortizeCaptureFactory:
    """Test create_capture_amortizer factory function."""

    def test_constrained_output_params(self):
        """Test constrained amortizer outputs alpha/beta parameters."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(unconstrained=False)

        assert "alpha" in amortizer.output_params
        assert "beta" in amortizer.output_params
        assert "loc" not in amortizer.output_params

    def test_unconstrained_output_params(self):
        """Test unconstrained amortizer outputs loc/scale parameters."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(unconstrained=True)

        assert "loc" in amortizer.output_params
        assert "log_scale" in amortizer.output_params
        assert "alpha" not in amortizer.output_params

    def test_custom_hidden_dims(self):
        """Test amortizer with custom hidden dimensions."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(hidden_dims=[128, 64, 32])
        assert amortizer.hidden_dims == [128, 64, 32]

        # Verify forward pass works with custom dims via Flax init/apply
        counts = jnp.ones((10, 100))
        params = amortizer.init(jax.random.PRNGKey(0), counts)
        out = amortizer.apply(params, counts)
        outputs = out.params
        assert "alpha" in outputs
        assert outputs["alpha"].shape == (10,)

    def test_invalid_input_transformation(self):
        """Test that invalid input transformation raises error."""
        from scribe.models.presets.registry import create_capture_amortizer

        with pytest.raises(ValueError, match="Unknown input_transformation"):
            create_capture_amortizer(input_transformation="invalid")

    def test_from_config_constrained(self):
        """Test create_capture_amortizer_from_config with constrained."""
        from scribe.models.config import AmortizationConfig
        from scribe.models.presets.registry import (
            create_capture_amortizer_from_config,
        )

        config = AmortizationConfig(
            enabled=True, hidden_dims=[64, 32], activation="relu"
        )
        amortizer = create_capture_amortizer_from_config(
            config, unconstrained=False
        )

        assert "alpha" in amortizer.output_params
        assert amortizer.hidden_dims == [64, 32]

    def test_from_config_unconstrained(self):
        """Test create_capture_amortizer_from_config with unconstrained."""
        from scribe.models.config import AmortizationConfig
        from scribe.models.presets.registry import (
            create_capture_amortizer_from_config,
        )

        config = AmortizationConfig(
            enabled=True, hidden_dims=[128, 64], activation="gelu"
        )
        amortizer = create_capture_amortizer_from_config(
            config, unconstrained=True
        )

        assert "loc" in amortizer.output_params
        assert "log_scale" in amortizer.output_params
        assert amortizer.hidden_dims == [128, 64]

    def test_from_config_with_output_transform_and_clamp(self):
        """Test create_capture_amortizer_from_config forwards output_transform and clamps."""
        from scribe.models.config import AmortizationConfig
        from scribe.models.presets.registry import (
            create_capture_amortizer_from_config,
        )

        config = AmortizationConfig(
            enabled=True,
            hidden_dims=[64, 32],
            output_transform="softplus",
            output_clamp_min=0.2,
            output_clamp_max=25.0,
        )
        amortizer = create_capture_amortizer_from_config(
            config, unconstrained=False
        )

        counts = jnp.ones((10, 50))
        params = amortizer.init(jax.random.PRNGKey(0), counts)
        out = amortizer.apply(params, counts)
        outputs = out.params

        assert jnp.all(outputs["alpha"] >= 0.2)
        assert jnp.all(outputs["alpha"] <= 25.0)
        assert jnp.all(outputs["beta"] >= 0.2)
        assert jnp.all(outputs["beta"] <= 25.0)

    def test_custom_activation(self):
        """Test create_capture_amortizer with custom activation function."""
        from scribe.models.presets.registry import create_capture_amortizer

        # Test with different activations
        activations = ["relu", "gelu", "leaky_relu", "silu", "tanh", "elu"]

        for activation in activations:
            amortizer = create_capture_amortizer(
                hidden_dims=[64, 32],
                activation=activation,
                unconstrained=False,
            )

            assert amortizer.activation == activation

            # Test forward pass works via Flax init/apply
            counts = jnp.ones((10, 100))
            params = amortizer.init(jax.random.PRNGKey(0), counts)
            out = amortizer.apply(params, counts)
            outputs = out.params

            assert "alpha" in outputs
            assert "beta" in outputs
            assert outputs["alpha"].shape == (10,)
            assert outputs["beta"].shape == (10,)

    def test_default_activation(self):
        """Test create_capture_amortizer defaults to relu."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(unconstrained=False)

        assert amortizer.activation == "relu"

    def test_softplus_output_transform(self):
        """Test that softplus output transform produces bounded positive values."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(
            unconstrained=False,
            output_transform="softplus",
            output_clamp_min=0.1,
            output_clamp_max=50.0,
        )

        counts = jnp.ones((10, 100))
        params = amortizer.init(jax.random.PRNGKey(0), counts)
        out = amortizer.apply(params, counts)
        outputs = out.params

        # Output keys should be alpha and beta (already transformed)
        assert "alpha" in outputs
        assert "beta" in outputs

        # Values should be within clamp range (softplus+0.5 is always >= 0.5,
        # and clamped to [0.1, 50.0])
        assert jnp.all(outputs["alpha"] >= 0.1)
        assert jnp.all(outputs["alpha"] <= 50.0)
        assert jnp.all(outputs["beta"] >= 0.1)
        assert jnp.all(outputs["beta"] <= 50.0)

    def test_exp_output_transform_with_clamping(self):
        """Test that exp output transform with clamping respects bounds."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(
            unconstrained=False,
            output_transform="exp",
            output_clamp_min=0.1,
            output_clamp_max=50.0,
        )

        counts = jnp.ones((10, 100))
        params = amortizer.init(jax.random.PRNGKey(0), counts)
        out = amortizer.apply(params, counts)
        outputs = out.params

        assert jnp.all(outputs["alpha"] >= 0.1)
        assert jnp.all(outputs["alpha"] <= 50.0)
        assert jnp.all(outputs["beta"] >= 0.1)
        assert jnp.all(outputs["beta"] <= 50.0)

    def test_invalid_output_transform(self):
        """Test that invalid output_transform raises error."""
        from scribe.models.presets.registry import create_capture_amortizer

        with pytest.raises(ValueError, match="Unknown output_transform"):
            create_capture_amortizer(
                unconstrained=False,
                output_transform="invalid",
            )

    def test_unconstrained_ignores_output_transform(self):
        """Test that unconstrained mode ignores output_transform settings."""
        from scribe.models.presets.registry import create_capture_amortizer

        amortizer = create_capture_amortizer(
            unconstrained=True,
            output_transform="softplus",  # Should be ignored
            output_clamp_min=0.1,  # Should be ignored
        )

        assert "loc" in amortizer.output_params
        assert "log_scale" in amortizer.output_params


# ==============================================================================
# Test Amortization Integration with Factory
# ==============================================================================


class TestAmortizationIntegration:
    """Test amortization integration with model factory."""

    def test_build_capture_spec_with_amortization(self):
        """Test build_capture_spec creates AmortizedGuide when enabled."""
        from scribe.models.components.guide_families import AmortizedGuide
        from scribe.models.config import AmortizationConfig
        from scribe.models.parameterizations import PARAMETERIZATIONS
        from scribe.models.presets.registry import build_capture_spec

        amort_config = AmortizationConfig(enabled=True, hidden_dims=[32])
        guide_families = GuideFamilyConfig(capture_amortization=amort_config)
        param_strategy = PARAMETERIZATIONS["canonical"]

        spec = build_capture_spec(
            unconstrained=False,
            guide_families=guide_families,
            param_strategy=param_strategy,
        )

        assert isinstance(spec.guide_family, AmortizedGuide)
        assert spec.guide_family.amortizer is not None

    def test_build_capture_spec_without_amortization(self):
        """Test build_capture_spec uses default guide when amortization disabled."""
        from scribe.models.components.guide_families import MeanFieldGuide
        from scribe.models.parameterizations import PARAMETERIZATIONS
        from scribe.models.presets.registry import build_capture_spec

        guide_families = GuideFamilyConfig()  # No amortization
        param_strategy = PARAMETERIZATIONS["canonical"]

        spec = build_capture_spec(
            unconstrained=False,
            guide_families=guide_families,
            param_strategy=param_strategy,
        )

        assert isinstance(spec.guide_family, MeanFieldGuide)

    def test_preset_builder_with_amortization(self):
        """Test build_config_from_preset with amortization."""
        from scribe.inference.preset_builder import build_config_from_preset

        config = build_config_from_preset(
            model="nbvcp",
            amortize_capture=True,
            capture_hidden_dims=[64, 32],
            capture_activation="gelu",
        )

        assert config.guide_families is not None
        assert config.guide_families.capture_amortization is not None
        assert config.guide_families.capture_amortization.enabled is True
        assert config.guide_families.capture_amortization.hidden_dims == [
            64,
            32,
        ]
        assert config.guide_families.capture_amortization.activation == "gelu"
        # New fields should have defaults
        assert (
            config.guide_families.capture_amortization.output_transform
            == "softplus"
        )
        assert (
            config.guide_families.capture_amortization.output_clamp_min == 0.1
        )
        assert (
            config.guide_families.capture_amortization.output_clamp_max == 50.0
        )

    def test_preset_builder_with_amortization_custom_transform(self):
        """Test build_config_from_preset with custom output transform settings."""
        from scribe.inference.preset_builder import build_config_from_preset

        config = build_config_from_preset(
            model="nbvcp",
            amortize_capture=True,
            capture_output_transform="exp",
            capture_clamp_min=0.5,
            capture_clamp_max=100.0,
        )

        amort = config.guide_families.capture_amortization
        assert amort.output_transform == "exp"
        assert amort.output_clamp_min == 0.5
        assert amort.output_clamp_max == 100.0

    def test_preset_builder_amortization_requires_vcp_model(self):
        """Test that amortize_capture raises error for non-VCP models."""
        from scribe.inference.preset_builder import build_config_from_preset

        with pytest.raises(ValueError, match="only valid for VCP models"):
            build_config_from_preset(
                model="nbdm",
                amortize_capture=True,
            )

        with pytest.raises(ValueError, match="only valid for VCP models"):
            build_config_from_preset(
                model="zinb",
                amortize_capture=True,
            )

    def test_build_config_from_preset_with_capture_amortization_object(self):
        """Test build_config_from_preset with capture_amortization object (single config)."""
        from scribe.models.config import AmortizationConfig
        from scribe.inference.preset_builder import build_config_from_preset

        # Passing capture_amortization object yields same guide_families as six params
        amort_config = AmortizationConfig(
            enabled=True,
            hidden_dims=[32, 16],
            activation="gelu",
            output_transform="exp",
            output_clamp_min=0.5,
            output_clamp_max=100.0,
        )
        config = build_config_from_preset(
            model="nbvcp",
            capture_amortization=amort_config,
        )
        assert config.guide_families is not None
        assert config.guide_families.capture_amortization is amort_config
        assert (
            config.guide_families.capture_amortization.output_transform == "exp"
        )
        assert (
            config.guide_families.capture_amortization.output_clamp_min == 0.5
        )
        assert (
            config.guide_families.capture_amortization.output_clamp_max == 100.0
        )

        # capture_amortization (object) for non-VCP raises
        with pytest.raises(ValueError, match="only valid for VCP models"):
            build_config_from_preset(
                model="nbdm",
                capture_amortization=amort_config,
            )

    def test_build_config_from_preset_capture_amortization_dict(self):
        """Test build_config_from_preset with capture_amortization as dict (normalized)."""
        from scribe.inference.preset_builder import build_config_from_preset

        config = build_config_from_preset(
            model="nbvcp",
            capture_amortization={
                "enabled": True,
                "hidden_dims": [64, 32],
                "output_transform": "softplus",
                "output_clamp_min": 0.1,
                "output_clamp_max": 50.0,
            },
        )
        assert config.guide_families is not None
        assert config.guide_families.capture_amortization is not None
        assert config.guide_families.capture_amortization.enabled is True
        assert (
            config.guide_families.capture_amortization.output_transform
            == "softplus"
        )

    def test_fit_with_capture_amortization_object(self):
        """Test scribe.fit() with capture_amortization object (single config flow)."""
        import scribe
        from scribe.models.config import AmortizationConfig

        # Minimal run: fit() with capture_amortization=AmortizationConfig(...) runs
        counts = jnp.ones((20, 30))  # tiny data
        capture_amortization = AmortizationConfig(
            enabled=True,
            hidden_dims=[16, 8],
            output_transform="softplus",
        )
        results = scribe.fit(
            counts,
            model="nbvcp",
            capture_amortization=capture_amortization,
            n_steps=2,
        )
        assert results is not None
        # Backward compat: fit() with amortize_capture=True and six params still works
        results2 = scribe.fit(
            counts,
            model="nbvcp",
            amortize_capture=True,
            capture_output_transform="exp",
            n_steps=2,
        )
        assert results2 is not None


# ==============================================================================
# Test Mixture Parameter Validation
# ==============================================================================


class TestMixtureParamsValidation:
    """Test validation of mixture_params against parameterization."""

    def test_invalid_mixture_param_raises_error(self):
        """Test that invalid mixture_params raise ValueError with helpful message."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["r"])  # 'r' is derived in mean_odds, not sampled
            .build()
        )
        with pytest.raises(ValueError, match="Invalid mixture_params"):
            create_model(config)

    def test_error_message_lists_valid_params(self):
        """Test error message includes list of valid parameters."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["invalid_param"])
            .build()
        )
        with pytest.raises(
            ValueError, match=r"Core \(mean_odds\): \['phi', 'mu'\]"
        ):
            create_model(config)

    def test_error_message_mentions_derived_params(self):
        """Test error message explains derived parameters cannot be mixture-specific."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["r"])
            .build()
        )
        with pytest.raises(ValueError, match="Derived parameters"):
            create_model(config)

    def test_valid_core_params_canonical(self):
        """Test valid core params for canonical parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("canonical")
            .as_mixture(2, ["p", "r"])
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

    def test_valid_core_params_mean_prob(self):
        """Test valid core params for mean_prob parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_prob")
            .as_mixture(2, ["p", "mu"])
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

    def test_valid_core_params_mean_odds(self):
        """Test valid core params for mean_odds parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["phi", "mu"])
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

    def test_valid_model_specific_param_gate(self):
        """Test gate is valid mixture param for zinb models."""
        config = (
            ModelConfigBuilder()
            .for_model("zinb")
            .with_parameterization("canonical")
            .as_mixture(2, ["r", "gate"])
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

    def test_valid_model_specific_param_capture(self):
        """Test p_capture is valid mixture param for nbvcp models."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("canonical")
            .as_mixture(2, ["r", "p_capture"])
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

    def test_transformed_capture_param_mean_odds(self):
        """Test phi_capture is valid for mean_odds parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["mu", "phi_capture"])
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

    def test_error_shows_model_specific_params(self):
        """Test error message includes model-specific params for VCP models."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("canonical")
            .as_mixture(2, ["invalid"])
            .build()
        )
        with pytest.raises(ValueError, match=r"Model-specific \(nbvcp\)"):
            create_model(config)

    def test_invalid_param_for_wrong_parameterization(self):
        """Test that params valid in one parameterization fail in another."""
        # 'phi' is valid in mean_odds but not in canonical
        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("canonical")
            .as_mixture(2, ["phi"])  # phi is not in canonical
            .build()
        )
        with pytest.raises(ValueError, match="Invalid mixture_params"):
            create_model(config)

    @pytest.mark.parametrize(
        "model_type,parameterization,valid_params",
        [
            ("nbdm", "canonical", ["p", "r"]),
            ("nbdm", "mean_prob", ["p", "mu"]),
            ("nbdm", "mean_odds", ["phi", "mu"]),
            ("zinb", "canonical", ["p", "r", "gate"]),
            ("zinb", "mean_odds", ["phi", "mu", "gate"]),
            ("nbvcp", "canonical", ["p", "r", "p_capture"]),
            ("nbvcp", "mean_odds", ["phi", "mu", "phi_capture"]),
            ("zinbvcp", "canonical", ["p", "r", "gate", "p_capture"]),
            ("zinbvcp", "mean_odds", ["phi", "mu", "gate", "phi_capture"]),
        ],
    )
    def test_all_valid_params_combinations(
        self, model_type, parameterization, valid_params
    ):
        """Test that all valid param combinations work for each model/parameterization."""
        config = (
            ModelConfigBuilder()
            .for_model(model_type)
            .with_parameterization(parameterization)
            .as_mixture(2, valid_params)
            .build()
        )
        # Should not raise
        model, guide = create_model(config, validate=False)
        assert callable(model)

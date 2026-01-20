"""Tests for the unified model factory.

This module tests that the unified factory (`create_model`) produces
correct model and guide functions for all model types and configurations.
"""

import pytest
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

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

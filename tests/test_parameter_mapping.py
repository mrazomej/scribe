"""Tests for the parameter mapping system."""

import pytest
from src.scribe.models.config.parameter_mapping import (
    get_active_parameters,
    get_required_parameters,
    get_parameterization_mapping,
    validate_parameter_consistency,
    get_parameterization_summary,
    PARAMETERIZATION_MAPPINGS,
    ParameterizationMapping,
)
from src.scribe.models.config.enums import Parameterization


class TestParameterMapping:
    """Test the parameter mapping system."""

    def test_standard_parameterization_mapping(self):
        """Test standard parameterization mapping."""
        mapping = get_parameterization_mapping(Parameterization.STANDARD)

        assert mapping.parameterization == Parameterization.STANDARD
        assert mapping.core_parameters == {"p", "r"}
        assert mapping.optional_parameters == set()
        assert mapping.get_all_parameters() == {"p", "r"}
        assert mapping.is_parameter_supported("p")
        assert mapping.is_parameter_supported("r")
        assert not mapping.is_parameter_supported("mu")
        assert mapping.is_parameter_required("p")
        assert mapping.is_parameter_required("r")

    def test_linked_parameterization_mapping(self):
        """Test linked parameterization mapping."""
        mapping = get_parameterization_mapping(Parameterization.LINKED)

        assert mapping.parameterization == Parameterization.LINKED
        assert mapping.core_parameters == {"p", "mu"}
        assert mapping.optional_parameters == set()
        assert mapping.get_all_parameters() == {"p", "mu"}
        assert mapping.is_parameter_supported("p")
        assert mapping.is_parameter_supported("mu")
        assert not mapping.is_parameter_supported("r")
        assert not mapping.is_parameter_supported("phi")

    def test_odds_ratio_parameterization_mapping(self):
        """Test odds_ratio parameterization mapping."""
        mapping = get_parameterization_mapping(Parameterization.ODDS_RATIO)

        assert mapping.parameterization == Parameterization.ODDS_RATIO
        assert mapping.core_parameters == {"phi", "mu"}
        assert mapping.optional_parameters == set()
        assert mapping.get_all_parameters() == {"phi", "mu"}
        assert mapping.is_parameter_supported("phi")
        assert mapping.is_parameter_supported("mu")
        assert not mapping.is_parameter_supported("p")
        assert not mapping.is_parameter_supported("r")


class TestActiveParameters:
    """Test active parameter computation."""

    def test_standard_nbdm(self):
        """Test standard NBDM model parameters."""
        params = get_active_parameters(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert params == {"p", "r"}

    def test_linked_zinb(self):
        """Test linked ZINB model parameters."""
        params = get_active_parameters(
            parameterization=Parameterization.LINKED,
            model_type="zinb",
            is_mixture=False,
            is_zero_inflated=True,
            uses_variable_capture=False,
        )

        assert params == {"p", "mu", "gate"}

    def test_odds_ratio_nbvcp(self):
        """Test odds_ratio NBVCP model parameters."""
        params = get_active_parameters(
            parameterization=Parameterization.ODDS_RATIO,
            model_type="nbvcp",
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=True,
        )

        assert params == {"phi", "mu", "phi_capture"}

    def test_odds_ratio_zinbvcp(self):
        """Test odds_ratio ZINBVCP model parameters."""
        params = get_active_parameters(
            parameterization=Parameterization.ODDS_RATIO,
            model_type="zinbvcp",
            is_mixture=False,
            is_zero_inflated=True,
            uses_variable_capture=True,
        )

        assert params == {"phi", "mu", "gate", "phi_capture"}

    def test_mixture_model(self):
        """Test mixture model parameters."""
        params = get_active_parameters(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            is_mixture=True,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert params == {"p", "r", "mixing"}

    def test_complex_model(self):
        """Test complex model with all features."""
        params = get_active_parameters(
            parameterization=Parameterization.ODDS_RATIO,
            model_type="zinbvcp",
            is_mixture=True,
            is_zero_inflated=True,
            uses_variable_capture=True,
        )

        assert params == {"phi", "mu", "gate", "phi_capture", "mixing"}


class TestRequiredParameters:
    """Test required parameter computation."""

    def test_standard_nbdm_required(self):
        """Test required parameters for standard NBDM."""
        params = get_required_parameters(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert params == {"p", "r"}

    def test_zinb_required(self):
        """Test required parameters for ZINB."""
        params = get_required_parameters(
            parameterization=Parameterization.STANDARD,
            model_type="zinb",
            is_mixture=False,
            is_zero_inflated=True,
            uses_variable_capture=False,
        )

        assert params == {"p", "r", "gate"}

    def test_vcp_required(self):
        """Test required parameters for VCP models."""
        # Standard parameterization
        params_std = get_required_parameters(
            parameterization=Parameterization.STANDARD,
            model_type="nbvcp",
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=True,
        )
        assert params_std == {"p", "r", "p_capture"}

        # Odds ratio parameterization
        params_or = get_required_parameters(
            parameterization=Parameterization.ODDS_RATIO,
            model_type="nbvcp",
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=True,
        )
        assert params_or == {"phi", "mu", "phi_capture"}

    def test_mixture_required(self):
        """Test required parameters for mixture models."""
        params = get_required_parameters(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            is_mixture=True,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert params == {"p", "r", "mixing"}


class TestParameterValidation:
    """Test parameter validation."""

    def test_valid_configuration(self):
        """Test valid parameter configuration."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            provided_params={"p", "r"},
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert errors == []

    def test_missing_required_parameters(self):
        """Test missing required parameters."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            provided_params={"p"},  # Missing "r"
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert len(errors) == 1
        assert "Missing required parameters" in errors[0]
        assert "r" in errors[0]

    def test_unsupported_parameters(self):
        """Test unsupported parameters."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            provided_params={"p", "r", "mu"},  # "mu" not supported in standard
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert len(errors) == 1
        assert "Unsupported parameters" in errors[0]
        assert "mu" in errors[0]

    def test_zinb_missing_gate(self):
        """Test ZINB model missing gate parameter."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="zinb",
            provided_params={"p", "r"},  # Missing "gate"
            is_mixture=False,
            is_zero_inflated=True,
            uses_variable_capture=False,
        )

        assert len(errors) == 1
        assert "Missing required parameters" in errors[0]
        assert "gate" in errors[0]

    def test_vcp_missing_capture_param(self):
        """Test VCP model missing capture parameter."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbvcp",
            provided_params={"p", "r"},  # Missing "p_capture"
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=True,
        )

        assert len(errors) == 1
        assert "Missing required parameters" in errors[0]
        assert "p_capture" in errors[0]

    def test_odds_ratio_vcp_missing_phi_capture(self):
        """Test odds_ratio VCP model missing phi_capture parameter."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.ODDS_RATIO,
            model_type="nbvcp",
            provided_params={"phi", "mu"},  # Missing "phi_capture"
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=True,
        )

        assert len(errors) == 1
        assert "Missing required parameters" in errors[0]
        assert "phi_capture" in errors[0]

    def test_mixture_missing_mixing(self):
        """Test mixture model missing mixing parameter."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            provided_params={"p", "r"},  # Missing "mixing"
            is_mixture=True,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert len(errors) == 1
        assert "Missing required parameters" in errors[0]
        assert "mixing" in errors[0]


class TestParameterMappingRegistry:
    """Test the parameter mapping registry."""

    def test_all_parameterizations_present(self):
        """Test that all parameterizations have mappings."""
        mappings = PARAMETERIZATION_MAPPINGS

        assert Parameterization.STANDARD in mappings
        assert Parameterization.LINKED in mappings
        assert Parameterization.ODDS_RATIO in mappings

        assert Parameterization.CANONICAL in mappings
        assert Parameterization.MEAN_PROB in mappings
        assert Parameterization.MEAN_ODDS in mappings

        # Hierarchical parameterizations are now handled via boolean flags
        # (hierarchical_p, hierarchical_gate) instead of enum values
        assert len(mappings) == 6

    def test_parameterization_summary(self):
        """Test parameterization summary generation."""
        summary = get_parameterization_summary()

        assert "standard" in summary
        assert "linked" in summary
        assert "odds_ratio" in summary

        # Check structure
        for param_type, details in summary.items():
            assert "core_parameters" in details
            assert "optional_parameters" in details
            assert "all_parameters" in details
            assert "descriptions" in details

            # Check that all_parameters is union of core and optional
            all_params = set(details["all_parameters"])
            core_params = set(details["core_parameters"])
            optional_params = set(details["optional_parameters"])

            assert all_params == core_params | optional_params


class TestParameterMappingEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_parameterization(self):
        """Test invalid parameterization handling."""
        with pytest.raises(KeyError):
            get_parameterization_mapping("invalid")

    def test_empty_provided_params(self):
        """Test validation with empty provided parameters."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            provided_params=set(),
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        assert len(errors) == 1
        assert "Missing required parameters" in errors[0]

    def test_all_parameters_provided(self):
        """Test validation with all possible parameters provided."""
        errors = validate_parameter_consistency(
            parameterization=Parameterization.STANDARD,
            model_type="nbdm",
            provided_params={"p", "r", "gate", "p_capture", "mixing"},
            is_mixture=False,
            is_zero_inflated=False,
            uses_variable_capture=False,
        )

        # Should have errors for unsupported parameters
        assert len(errors) == 1
        assert "Unsupported parameters" in errors[0]
        assert (
            "gate" in errors[0]
            or "p_capture" in errors[0]
            or "mixing" in errors[0]
        )


class TestParameterMappingIntegration:
    """Test integration with model configuration."""

    def test_parameter_mapping_consistency(self):
        """Test that parameter mappings are internally consistent."""
        for param_type, mapping in PARAMETERIZATION_MAPPINGS.items():
            # All core parameters should be in all_parameters
            assert mapping.core_parameters.issubset(
                mapping.get_all_parameters()
            )

            # All optional parameters should be in all_parameters
            assert mapping.optional_parameters.issubset(
                mapping.get_all_parameters()
            )

            # Core and optional should be disjoint
            assert mapping.core_parameters.isdisjoint(
                mapping.optional_parameters
            )

            # All parameters in descriptions should be supported
            for param_name in mapping.parameter_descriptions:
                assert mapping.is_parameter_supported(param_name)

    def test_parameter_descriptions_completeness(self):
        """Test that all parameters have descriptions."""
        for param_type, mapping in PARAMETERIZATION_MAPPINGS.items():
            for param_name in mapping.get_all_parameters():
                assert param_name in mapping.parameter_descriptions
                assert mapping.parameter_descriptions[param_name]  # Not empty

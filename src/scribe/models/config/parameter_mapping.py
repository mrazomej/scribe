"""
Parameter mapping system for SCRIBE model configurations.

This module defines which parameters are active for each parameterization type,
making the system more maintainable and less error-prone than hardcoded if-statements.
"""

from typing import Dict, Set, List, NamedTuple
from dataclasses import dataclass
from .enums import Parameterization

# ==============================================================================
# Parameter Info Class
# ==============================================================================


class ParameterInfo(NamedTuple):
    """
    Holds metadata about a model parameter in a given context.

    This class encapsulates not only the name and description of the parameter,
    but also flags indicating its necessity with respect to specific model
    parameterizations or model types.

    Attributes
    ----------
    name : str
        The name of the parameter (e.g., "p", "r", "mu", etc.).
    description : str
        A human-readable description of what the parameter represents.
    required_for_parameterization : bool, optional
        Whether this parameter is required for the specific parameterization
        (e.g., "linked", "odds_ratio"). Defaults to False.
    required_for_model_type : bool, optional
        Whether this parameter is required for this kind of model
        (e.g., ZINB, NBDM, models with VCP capture probability, etc.).
        Defaults to False.
    """

    name: str
    description: str
    required_for_parameterization: bool = False
    required_for_model_type: bool = False


# ==============================================================================
# Parameterization Mapping Class
# ==============================================================================


@dataclass(frozen=True)
class ParameterizationMapping:
    """
    Represents the mapping from a model parameterization to its supported
    parameters.

    This class encapsulates the relationship between a parameterization type
    (such as "standard", "linked", or "odds_ratio") and the set of model
    parameters that are valid (core and optional) for that parameterization.

    Attributes
    ----------
    parameterization : Parameterization
        The parameterization type this mapping refers to (e.g.
        Parameterization.STANDARD).
    core_parameters : Set[str]
        The set of parameters required for this parameterization; these must
        always be present for the parameterization to be valid. For example,
        {"p", "r"} for "standard".
    optional_parameters : Set[str]
        The set of parameters that are supported by the parameterization but not
        required; these may be present depending on model extensions or specific
        model types.
    parameter_descriptions : Dict[str, str]
        Human-readable descriptions for each parameter in this mapping, keyed by
        parameter name.

    Methods
    -------
    get_all_parameters() -> Set[str]
        Returns the union of core and optional parameters valid for this
        parameterization.
    is_parameter_supported(param_name: str) -> bool
        Returns True if the given parameter name is either required (core) or
        optional for this parameterization.
    is_parameter_required(param_name: str) -> bool
        Returns True if the given parameter name is required for this
        parameterization.

    Examples
    --------
    >>> mapping = ParameterizationMapping(
    ...     parameterization=Parameterization.STANDARD,
    ...     core_parameters={"p", "r"},
    ...     optional_parameters=set(),
    ...     parameter_descriptions={"p": "Success probability", "r": "Dispersion"}
    ... )
    >>> mapping.is_parameter_supported("p")
    True
    >>> mapping.is_parameter_required("p")
    True
    >>> mapping.is_parameter_supported("gate")
    False
    """

    # The parameterization type for this mapping (e.g., STANDARD, LINKED,
    # ODDS_RATIO).
    parameterization: Parameterization
    # The set of core/required parameters for this parameterization.
    core_parameters: Set[str]
    # The set of optional parameters for this parameterization (may or may not
    # be present).
    optional_parameters: Set[str]
    # Human-readable descriptions for each parameter.
    parameter_descriptions: Dict[str, str]

    # --------------------------------------------------------------------------
    # Methods
    # --------------------------------------------------------------------------

    def get_all_parameters(self) -> Set[str]:
        """Get all parameters (core + optional) for this parameterization."""
        return self.core_parameters | self.optional_parameters

    def is_parameter_supported(self, param_name: str) -> bool:
        """Check if a parameter is supported by this parameterization."""
        return param_name in self.get_all_parameters()

    def is_parameter_required(self, param_name: str) -> bool:
        """Check if a parameter is required (core) for this parameterization."""
        return param_name in self.core_parameters


# ==============================================================================
# Parameter Mappings for Each Parameterization
# ==============================================================================

PARAMETERIZATION_MAPPINGS = {
    Parameterization.STANDARD: ParameterizationMapping(
        parameterization=Parameterization.STANDARD,
        core_parameters={"p", "r"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "r": "Dispersion parameter (LogNormal distribution)",
        },
    ),
    Parameterization.LINKED: ParameterizationMapping(
        parameterization=Parameterization.LINKED,
        core_parameters={"p", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
    Parameterization.ODDS_RATIO: ParameterizationMapping(
        parameterization=Parameterization.ODDS_RATIO,
        core_parameters={"phi", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "phi": "Odds ratio parameter (BetaPrime distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
}

# ==============================================================================
# Model-Specific Parameter Mappings
# ==============================================================================

# Parameters that are always optional (not core to any parameterization)
OPTIONAL_MODEL_PARAMETERS = {
    "gate": "Zero-inflation gate parameter (Beta distribution)",
    "p_capture": "Capture probability parameter (Beta distribution)",
    "phi_capture": "Capture phi parameter (BetaPrime distribution)",
    "mixing": "Mixture weights parameter (Dirichlet distribution)",
}

# Parameters that are required for specific model types
MODEL_TYPE_REQUIREMENTS = {
    "zinb": {"gate"},  # Zero-inflated models require gate parameter
    "vcp": {"p_capture"},  # Variable capture models require p_capture
    "vcp_odds_ratio": {
        "phi_capture"
    },  # VCP with odds_ratio requires phi_capture
    "mixture": {"mixing"},  # Mixture models require mixing parameter
}

# ==============================================================================
# Parameter Mapping Utilities
# ==============================================================================


def get_parameterization_mapping(
    parameterization: Parameterization,
) -> ParameterizationMapping:
    """Get the parameter mapping for a specific parameterization."""
    return PARAMETERIZATION_MAPPINGS[parameterization]


# ------------------------------------------------------------------------------


def get_active_parameters(
    parameterization: Parameterization,
    model_type: str,
    is_mixture: bool = False,
    is_zero_inflated: bool = False,
    uses_variable_capture: bool = False,
) -> Set[str]:
    """
    Get the set of active parameters for a given configuration.

    Parameters
    ----------
    parameterization : Parameterization
        The parameterization type
    model_type : str
        The model type (e.g., 'nbdm', 'zinb', 'nbvcp', 'zinbvcp')
    is_mixture : bool
        Whether this is a mixture model
    is_zero_inflated : bool
        Whether this is a zero-inflated model
    uses_variable_capture : bool
        Whether this model uses variable capture

    Returns
    -------
    Set[str]
        Set of active parameter names
    """
    # Start with parameterization-specific parameters
    mapping = get_parameterization_mapping(parameterization)
    active_params = mapping.get_all_parameters().copy()

    # Add model-specific parameters
    if is_zero_inflated:
        active_params.add("gate")

    if uses_variable_capture:
        if parameterization == Parameterization.ODDS_RATIO:
            active_params.add("phi_capture")
        else:
            active_params.add("p_capture")

    if is_mixture:
        active_params.add("mixing")

    return active_params


# ------------------------------------------------------------------------------


def get_required_parameters(
    parameterization: Parameterization,
    model_type: str,
    is_mixture: bool = False,
    is_zero_inflated: bool = False,
    uses_variable_capture: bool = False,
) -> Set[str]:
    """
    Get the set of required parameters for a given configuration.

    Parameters
    ----------
    parameterization : Parameterization
        The parameterization type
    model_type : str
        The model type (e.g., 'nbdm', 'zinb', 'nbvcp', 'zinbvcp')
    is_mixture : bool
        Whether this is a mixture model
    is_zero_inflated : bool
        Whether this is a zero-inflated model
    uses_variable_capture : bool
        Whether this model uses variable capture

    Returns
    -------
    Set[str]
        Set of required parameter names
    """
    # Start with parameterization-specific required parameters
    mapping = get_parameterization_mapping(parameterization)
    required_params = mapping.core_parameters.copy()

    # Add model-specific required parameters
    if is_zero_inflated:
        required_params.add("gate")

    if uses_variable_capture:
        if parameterization == Parameterization.ODDS_RATIO:
            required_params.add("phi_capture")
        else:
            required_params.add("p_capture")

    if is_mixture:
        required_params.add("mixing")

    return required_params


# ------------------------------------------------------------------------------


def get_parameter_description(
    param_name: str, parameterization: Parameterization
) -> str:
    """
    Get the description for a parameter in a specific parameterization context.

    Parameters
    ----------
    param_name : str
        The parameter name
    parameterization : Parameterization
        The parameterization type

    Returns
    -------
    str
        Description of the parameter
    """
    # Check parameterization-specific descriptions first
    mapping = get_parameterization_mapping(parameterization)
    if param_name in mapping.parameter_descriptions:
        return mapping.parameter_descriptions[param_name]

    # Check optional model parameters
    if param_name in OPTIONAL_MODEL_PARAMETERS:
        return OPTIONAL_MODEL_PARAMETERS[param_name]

    # Default description
    return f"Parameter: {param_name}"


# ------------------------------------------------------------------------------


def validate_parameter_consistency(
    parameterization: Parameterization,
    model_type: str,
    provided_params: Set[str],
    is_mixture: bool = False,
    is_zero_inflated: bool = False,
    uses_variable_capture: bool = False,
) -> List[str]:
    """
    Validate that provided parameters are consistent with the configuration.

    Parameters
    ----------
    parameterization : Parameterization
        The parameterization type
    model_type : str
        The model type
    provided_params : Set[str]
        Set of provided parameter names
    is_mixture : bool
        Whether this is a mixture model
    is_zero_inflated : bool
        Whether this is a zero-inflated model
    uses_variable_capture : bool
        Whether this model uses variable capture

    Returns
    -------
    List[str]
        List of validation error messages (empty if valid)
    """
    errors = []

    # Get expected parameters
    expected_params = get_active_parameters(
        parameterization,
        model_type,
        is_mixture,
        is_zero_inflated,
        uses_variable_capture,
    )
    required_params = get_required_parameters(
        parameterization,
        model_type,
        is_mixture,
        is_zero_inflated,
        uses_variable_capture,
    )

    # Check for missing required parameters
    missing_required = required_params - provided_params
    if missing_required:
        errors.append(
            f"Missing required parameters for {parameterization.value} parameterization: "
            f"{', '.join(sorted(missing_required))}"
        )

    # Check for unsupported parameters
    unsupported = provided_params - expected_params
    if unsupported:
        errors.append(
            f"Unsupported parameters for {parameterization.value} parameterization: "
            f"{', '.join(sorted(unsupported))}"
        )

    return errors


# ==============================================================================
# Parameter Mapping Registry
# ==============================================================================


def get_all_parameterization_mappings() -> (
    Dict[Parameterization, ParameterizationMapping]
):
    """Get all parameterization mappings."""
    return PARAMETERIZATION_MAPPINGS.copy()


def get_parameterization_summary() -> Dict[str, Dict[str, any]]:
    """
    Get a summary of all parameterizations and their supported parameters.

    Returns
    -------
    Dict[str, Dict[str, any]]
        Summary dictionary with parameterization details
    """
    summary = {}

    for param_type, mapping in PARAMETERIZATION_MAPPINGS.items():
        summary[param_type.value] = {
            "core_parameters": sorted(mapping.core_parameters),
            "optional_parameters": sorted(mapping.optional_parameters),
            "all_parameters": sorted(mapping.get_all_parameters()),
            "descriptions": mapping.parameter_descriptions,
        }

    return summary

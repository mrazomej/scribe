"""
Parameter mapping system for SCRIBE model configurations.

This module defines which parameters are active for each parameterization type,
making the system more maintainable and less error-prone than hardcoded if-statements.

It also provides the canonical mappings between internal (short/math) parameter
names and user-friendly descriptive names, used by the ``priors`` dict alias
system and the ``descriptive_names`` option on results objects.
"""

from typing import Any, Dict, Set, List, NamedTuple
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
    # Old names (backward compatibility)
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
    # New names (preferred)
    Parameterization.CANONICAL: ParameterizationMapping(
        parameterization=Parameterization.CANONICAL,
        core_parameters={"p", "r"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "r": "Dispersion parameter (LogNormal distribution)",
        },
    ),
    Parameterization.MEAN_PROB: ParameterizationMapping(
        parameterization=Parameterization.MEAN_PROB,
        core_parameters={"p", "mu"},
        optional_parameters=set(),
        parameter_descriptions={
            "p": "Success probability parameter (Beta distribution)",
            "mu": "Mean parameter (LogNormal distribution)",
        },
    ),
    Parameterization.MEAN_ODDS: ParameterizationMapping(
        parameterization=Parameterization.MEAN_ODDS,
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
    hierarchical_mu: bool = False,
    hierarchical_p: bool = False,
    hierarchical_gate: bool = False,
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
    hierarchical_mu : bool
        Whether hierarchical mu/r prior across components is enabled
    hierarchical_p : bool
        Whether hierarchical p/phi prior is enabled (True when
        ``p_prior != "none"`` in ModelConfig).
    hierarchical_gate : bool
        Whether hierarchical gate prior is enabled (True when
        ``gate_prior != "none"`` in ModelConfig).

    Returns
    -------
    Set[str]
        Set of active parameter names
    """
    # Start with parameterization-specific parameters
    mapping = get_parameterization_mapping(parameterization)
    active_params = mapping.get_all_parameters().copy()

    # Add hierarchical mu hyperparameters when flag is set
    if hierarchical_mu:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
            Parameterization.MEAN_PROB,
            Parameterization.LINKED,
        ):
            active_params.update({"log_mu_loc", "log_mu_scale"})
        else:
            active_params.update({"log_r_loc", "log_r_scale"})

    # Add hierarchical hyperparameters when flags are set
    if hierarchical_p:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
            active_params.update({"log_phi_loc", "log_phi_scale"})
        else:
            active_params.update({"logit_p_loc", "logit_p_scale"})

    # Add model-specific parameters
    if is_zero_inflated:
        active_params.add("gate")
        if hierarchical_gate:
            active_params.update({"logit_gate_loc", "logit_gate_scale"})

    if uses_variable_capture:
        if parameterization in (
            Parameterization.MEAN_ODDS,
            Parameterization.ODDS_RATIO,
        ):
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


# ==============================================================================
# Descriptive Name Mapping (internal <-> user-friendly)
# ==============================================================================

# Maps internal (short/math) parameter names to user-friendly descriptive names.
# Used by:
#   - results.get_map(descriptive_names=True)
#   - results.get_distributions(descriptive_names=True)
#   - results.get_posterior_samples(descriptive_names=True)
#
# Hierarchical hyperprior keys (logit_p_loc, log_phi_scale, etc.) are NOT
# included -- they pass through unchanged because the _loc/_scale suffixes
# are correct Normal parameter names in unconstrained space.
DESCRIPTIVE_NAMES: Dict[str, str] = {
    # Core NB parameters
    "r": "dispersion",
    "p": "prob",
    "mu": "expression",
    "phi": "odds",
    "gate": "zero_inflation",
    # Capture parameters
    "p_capture": "capture_prob",
    "phi_capture": "capture_odds",
    "eta_capture": "capture_efficiency",
    # Already descriptive (identity)
    "bnb_concentration": "bnb_concentration",
    "mixing_weights": "mixing_weights",
    "z": "latent_embedding",
}

# Inverse mapping: descriptive name -> internal name
_DESCRIPTIVE_TO_INTERNAL: Dict[str, str] = {
    v: k for k, v in DESCRIPTIVE_NAMES.items()
}


# ==============================================================================
# Prior Key Aliases (descriptive -> internal)
# ==============================================================================

# Maps user-friendly prior dict keys to the internal keys expected by the
# model/builder system. Both the internal key and its descriptive alias are
# accepted in the ``priors`` dict; the normalizer resolves aliases early.
#
# Only core parameter priors and capture-specific keys are aliased.
# Hierarchical hyperprior override keys (logit_p_loc, log_phi_scale, etc.)
# are left as-is -- users who touch those already know the transform space.
PRIOR_KEY_ALIASES: Dict[str, str] = {
    # Core parameter priors (descriptive -> internal)
    "prob": "p",
    "dispersion": "r",
    "expression": "mu",
    "odds": "phi",
    "zero_inflation": "gate",
    "capture_prob": "p_capture",
    "capture_odds": "phi_capture",
    # Capture-specific priors
    "capture_efficiency": "eta_capture",
    "capture_scaling": "mu_eta",
}

# Inverse: internal -> descriptive alias (for documentation / YAML comments)
_INTERNAL_TO_PRIOR_ALIAS: Dict[str, str] = {
    v: k for k, v in PRIOR_KEY_ALIASES.items()
}


# ==============================================================================
# fit() Keyword Argument Rename Mapping
# ==============================================================================

# Maps old (math-notation) fit() kwarg names to new (descriptive) names.
# Used by ModelConfig.__setstate__ for pickle migration and as documentation.
FIT_KWARG_RENAMES: Dict[str, str] = {
    "mu_prior": "expression_prior",
    "mu_dataset_prior": "expression_dataset_prior",
    "mu_eta_prior": "capture_scaling_prior",
    "mu_mean_anchor": "expression_anchor",
    "mu_mean_anchor_sigma": "expression_anchor_sigma",
    "p_prior": "prob_prior",
    "p_dataset_prior": "prob_dataset_prior",
    "p_dataset_mode": "prob_dataset_mode",
    "gate_prior": "zero_inflation_prior",
    "gate_dataset_prior": "zero_inflation_dataset_prior",
}


# ==============================================================================
# Utility Functions
# ==============================================================================


def normalize_prior_keys(priors: Dict[str, Any]) -> Dict[str, Any]:
    """Translate descriptive prior dict keys to their internal equivalents.

    Accepts both internal keys (``"p"``, ``"eta_capture"``) and descriptive
    aliases (``"prob"``, ``"capture_efficiency"``).  Raises ``ValueError`` if
    both an alias and its internal target are present (ambiguous).

    Parameters
    ----------
    priors : dict
        Prior override dictionary as passed by the user.

    Returns
    -------
    dict
        Copy of *priors* with all alias keys replaced by internal names.

    Raises
    ------
    ValueError
        If a descriptive alias and its corresponding internal key are both
        present (e.g. ``{"p": ..., "prob": ...}``).
    """
    if not priors:
        return priors

    result: Dict[str, Any] = {}
    for key, value in priors.items():
        internal_key = PRIOR_KEY_ALIASES.get(key)
        if internal_key is not None:
            # Key is a descriptive alias -- resolve it
            if internal_key in priors:
                raise ValueError(
                    f"Ambiguous priors: both the descriptive alias "
                    f"'{key}' and the internal key '{internal_key}' are "
                    f"present. Use one or the other, not both."
                )
            result[internal_key] = value
        else:
            result[key] = value
    return result


def rename_dict_keys(
    d: Dict[str, Any],
    descriptive: bool = False,
) -> Dict[str, Any]:
    """Optionally rename internal parameter keys to descriptive names.

    Handles suffixed keys such as ``r_0`` (component-indexed) and
    ``r_dataset_0`` (dataset-indexed) by matching the base name before the
    first ``_`` digit boundary.

    Parameters
    ----------
    d : dict
        Dictionary with internal parameter name keys.
    descriptive : bool, default False
        If True, apply the DESCRIPTIVE_NAMES mapping.  If False, return
        *d* unchanged.

    Returns
    -------
    dict
        A new dictionary with renamed keys (if *descriptive* is True),
        or the original dictionary unchanged.
    """
    if not descriptive:
        return d

    renamed: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = DESCRIPTIVE_NAMES.get(key)
        if new_key is not None:
            renamed[new_key] = value
        else:
            # Try to match base_name from suffixed keys like "r_0", "gate_1"
            renamed[_rename_suffixed_key(key)] = value
    return renamed


def _rename_suffixed_key(key: str) -> str:
    """Rename a suffixed key like ``r_0`` -> ``dispersion_0``.

    Tries progressively longer prefixes so that multi-part internal names
    like ``p_capture_0`` are handled correctly (``p_capture`` is in the
    mapping, not ``p``).
    """
    parts = key.split("_")
    # Try longest prefix first (e.g. "p_capture" before "p")
    for i in range(len(parts) - 1, 0, -1):
        prefix = "_".join(parts[:i])
        suffix = "_".join(parts[i:])
        if prefix in DESCRIPTIVE_NAMES:
            return f"{DESCRIPTIVE_NAMES[prefix]}_{suffix}"
    return key


# ==============================================================================
# Parameter Shorthand Resolution
# ==============================================================================

# Recognised string shorthands for mixture_params, joint_params, dense_params.
# These resolve to concrete ``List[str]`` (or ``None``) based on the chosen
# parameterization and model type, so users never need to memorise which
# Greek-letter names belong to which parameterization.
PARAM_SHORTHANDS = frozenset({"all", "biological", "mean", "prob", "gate"})


def resolve_param_shorthand(
    value: "Union[str, List[str], None]",
    param_strategy: "Parameterization",
    base_model: str,
) -> "Optional[List[str]]":
    """Resolve a parameter shorthand to a concrete list of internal names.

    Accepts the semantic shorthands ``"all"``, ``"biological"``, ``"mean"``,
    ``"prob"``, and ``"gate"`` and expands them using the parameterization
    strategy's ``core_parameters`` and ``gene_param_name``.  Explicit
    ``List[str]`` values pass through with descriptive-name aliases resolved
    (e.g. ``"expression"`` -> ``"mu"``).

    Parameters
    ----------
    value : str, List[str], or None
        The user-provided value.  ``None`` passes through unchanged.
    param_strategy : Parameterization
        The active parameterization strategy instance (provides
        ``core_parameters`` and ``gene_param_name``).
    base_model : str
        The base model type (``"nbdm"``, ``"zinb"``, ``"nbvcp"``,
        ``"zinbvcp"``).  Needed to determine whether ``"gate"`` is valid.

    Returns
    -------
    Optional[List[str]]
        Resolved list of internal parameter names, or ``None`` when the
        shorthand ``"all"`` is used (the existing factory logic already
        treats ``None`` as "all parameters are mixture/joint-specific").

    Raises
    ------
    ValueError
        If a shorthand references an unavailable parameter (e.g.
        ``"gate"`` for a non-ZINB model) or if a list element is not a
        recognised internal or descriptive name.

    Examples
    --------
    >>> from scribe.models.parameterizations import PARAMETERIZATIONS
    >>> strat = PARAMETERIZATIONS["mean_odds"]
    >>> resolve_param_shorthand("biological", strat, "zinb")
    ['phi', 'mu']
    >>> resolve_param_shorthand("mean", strat, "zinb")
    ['mu']
    >>> resolve_param_shorthand("all", strat, "zinb")  # returns None
    >>> resolve_param_shorthand(["expression", "odds"], strat, "zinb")
    ['mu', 'phi']
    """
    if value is None:
        return None

    # --- String shorthand resolution ---
    if isinstance(value, str):
        shorthand = value.lower()
        core = list(param_strategy.core_parameters)
        gene_param = param_strategy.gene_param_name
        is_zinb = base_model in ("zinb", "zinbvcp")

        # Derive the "prob" parameter: the core param that is NOT the
        # gene-level (mean/dispersion) parameter.
        prob_param = [p for p in core if p != gene_param]

        if shorthand == "all":
            # None signals "everything" to the existing factory logic
            # (build_param_specs defaults to all core, build_gate_spec
            # defaults to is_mixture=True, etc.)
            return None

        if shorthand == "biological":
            return core

        if shorthand == "mean":
            return [gene_param]

        if shorthand == "prob":
            return prob_param

        if shorthand == "gate":
            if not is_zinb:
                raise ValueError(
                    f"Shorthand 'gate' is only valid for ZINB models "
                    f"(zinb, zinbvcp), but the current model is "
                    f"'{base_model}'."
                )
            return ["gate"]

        raise ValueError(
            f"Unknown parameter shorthand '{value}'. "
            f"Valid shorthands are: {sorted(PARAM_SHORTHANDS)}. "
            f"You can also pass an explicit list of parameter names."
        )

    # --- Explicit list: resolve descriptive aliases ---
    resolved: "List[str]" = []
    for name in value:
        internal = _DESCRIPTIVE_TO_INTERNAL.get(name)
        if internal is not None:
            resolved.append(internal)
        else:
            resolved.append(name)
    return resolved

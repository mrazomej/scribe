"""Model component registries and helper builders.

This module provides registries for model-specific components and helper
functions for building model-specific parameters. This consolidates the
duplicated logic from individual preset factories.

Registries
----------
MODEL_EXTRA_PARAMS : Dict[str, List[str]]
    Maps model types to their extra parameters beyond core parameterization.
LIKELIHOOD_REGISTRY : Dict[str, Type[Likelihood]]
    Maps model types to their likelihood classes.
GUIDE_FAMILY_REGISTRY : Dict[str, Type[GuideFamily]]
    Maps string names to guide family classes for YAML configuration.

Functions
---------
build_gate_spec
    Build gate parameter spec for zero-inflated models.
build_capture_spec
    Build capture probability parameter spec for VCP models.
build_extra_param_spec
    Dispatch to appropriate builder based on parameter name.
apply_prior_guide_overrides
    Apply user-provided prior/guide overrides to param specs.
get_guide_family
    Get a guide family instance by name.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from ..builders.parameter_specs import (
    BetaPrimeSpec,
    BetaSpec,
    ExpNormalSpec,
    ParamSpec,
    SigmoidNormalSpec,
)
from ..components.guide_families import (
    AmortizedGuide,
    GuideFamily,
    LowRankGuide,
    MeanFieldGuide,
)
from ..components.likelihoods import (
    Likelihood,
    NBWithVCPLikelihood,
    NegativeBinomialLikelihood,
    ZeroInflatedNBLikelihood,
    ZINBWithVCPLikelihood,
)
from ..config import GuideFamilyConfig
from ..parameterizations import Parameterization

# ==============================================================================
# Model Component Registries
# ==============================================================================

# Model-specific extra parameters beyond core parameterization
# These are the parameters that differ between model types
MODEL_EXTRA_PARAMS: Dict[str, List[str]] = {
    "nbdm": [],
    "zinb": ["gate"],
    "nbvcp": ["p_capture"],
    "zinbvcp": ["gate", "p_capture"],
}

# Likelihood class registry - maps model type to likelihood class
LIKELIHOOD_REGISTRY: Dict[str, Type[Likelihood]] = {
    "nbdm": NegativeBinomialLikelihood,
    "zinb": ZeroInflatedNBLikelihood,
    "nbvcp": NBWithVCPLikelihood,
    "zinbvcp": ZINBWithVCPLikelihood,
}

# Guide family registry - maps string names to guide family classes
# This allows string-based specification in YAML configs
GUIDE_FAMILY_REGISTRY: Dict[str, Type[GuideFamily]] = {
    "mean_field": MeanFieldGuide,
    "low_rank": LowRankGuide,
    "amortized": AmortizedGuide,
}


def get_guide_family(name: str, **kwargs: Any) -> GuideFamily:
    """Get a guide family instance by name.

    This allows creating guide families from string names, which is useful
    for YAML configuration.

    Parameters
    ----------
    name : str
        Name of the guide family: "mean_field", "low_rank", or "amortized".
    **kwargs
        Additional arguments to pass to the guide family constructor.
        For "low_rank", pass rank=int.
        For "amortized", pass amortizer=Amortizer instance.

    Returns
    -------
    GuideFamily
        Instance of the requested guide family.

    Raises
    ------
    ValueError
        If name is not recognized.

    Examples
    --------
    >>> guide = get_guide_family("mean_field")
    >>> guide = get_guide_family("low_rank", rank=10)
    """
    if name not in GUIDE_FAMILY_REGISTRY:
        raise ValueError(
            f"Unknown guide family: {name}. "
            f"Valid options: {list(GUIDE_FAMILY_REGISTRY.keys())}"
        )
    return GUIDE_FAMILY_REGISTRY[name](**kwargs)


# ==============================================================================
# Extra Parameter Builders
# ==============================================================================


def build_gate_spec(
    unconstrained: bool,
    guide_families: GuideFamilyConfig,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
) -> ParamSpec:
    """Build gate parameter spec for zero-inflated models (ZINB, ZINBVCP).

    The gate parameter controls the probability of structural zeros per gene.
    It is always gene-specific.

    Parameters
    ----------
    unconstrained : bool
        If True, use SigmoidNormalSpec (Normal + sigmoid transform).
        If False, use BetaSpec (constrained Beta distribution).
    guide_families : GuideFamilyConfig
        Guide family configuration for retrieving gate's guide family.
    n_components : int, optional
        Number of mixture components. If provided and gate is in mixture_params,
        the parameter will be marked as mixture-specific.
    mixture_params : List[str], optional
        List of parameters that should be mixture-specific. If None and
        n_components is set, gate defaults to being mixture-specific.

    Returns
    -------
    ParamSpec
        Parameter specification for the gate parameter.
    """
    gate_family = guide_families.get("gate")

    # Determine if gate should be mixture-specific
    is_mixture = False
    if n_components is not None:
        if mixture_params is None:
            # Default: make all gene-specific params mixture-specific
            is_mixture = True
        else:
            is_mixture = "gate" in mixture_params

    if unconstrained:
        return SigmoidNormalSpec(
            name="gate",
            shape_dims=("n_genes",),
            default_params=(-2.0, 1.0),  # Default to low zero-inflation
            is_gene_specific=True,
            guide_family=gate_family,
            is_mixture=is_mixture,
        )
    else:
        return BetaSpec(
            name="gate",
            shape_dims=("n_genes",),
            default_params=(
                1.0,
                9.0,
            ),  # Default to low zero-inflation (mean ~0.1)
            is_gene_specific=True,
            guide_family=gate_family,
            is_mixture=is_mixture,
        )


# ------------------------------------------------------------------------------


def build_capture_spec(
    unconstrained: bool,
    guide_families: GuideFamilyConfig,
    param_strategy: Parameterization,
) -> ParamSpec:
    """Build capture probability parameter spec for VCP models (NBVCP, ZINBVCP).

    The capture parameter models cell-specific technical variation in mRNA
    capture efficiency. It is always cell-specific.

    For mean_odds parameterization, the parameter is transformed from p_capture
    to phi_capture (odds ratio parameterization).

    Parameters
    ----------
    unconstrained : bool
        If True, use transformed Normal distribution.
        If False, use constrained distribution (Beta or BetaPrime).
    guide_families : GuideFamilyConfig
        Guide family configuration for retrieving capture's guide family.
    param_strategy : Parameterization
        Parameterization strategy to determine parameter name transformation
        (p_capture vs phi_capture for mean_odds).

    Returns
    -------
    ParamSpec
        Parameter specification for the capture parameter.
    """
    # Get the appropriate capture parameter name based on parameterization
    # mean_odds uses phi_capture, others use p_capture
    capture_param_name = param_strategy.transform_model_param("p_capture")
    capture_family = guide_families.get(capture_param_name)

    if unconstrained:
        # Use ExpNormalSpec for phi_capture (BetaPrime -> [0, +inf))
        # Use SigmoidNormalSpec for p_capture (Beta -> [0, 1])
        if capture_param_name == "phi_capture":
            return ExpNormalSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(0.0, 1.0),
                is_cell_specific=True,
                guide_family=capture_family,
            )
        else:
            return SigmoidNormalSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(0.0, 1.0),
                is_cell_specific=True,
                guide_family=capture_family,
            )
    else:
        # Use BetaPrime for phi_capture (mean_odds), Beta for p_capture (others)
        if capture_param_name == "phi_capture":
            return BetaPrimeSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(
                    1.0,
                    1.0,
                ),  # Uniform prior on capture odds ratio
                is_cell_specific=True,
                guide_family=capture_family,
            )
        else:
            return BetaSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(
                    1.0,
                    1.0,
                ),  # Uniform prior on capture probability
                is_cell_specific=True,
                guide_family=capture_family,
            )


# ------------------------------------------------------------------------------


def build_extra_param_spec(
    param_name: str,
    unconstrained: bool,
    guide_families: GuideFamilyConfig,
    param_strategy: Parameterization,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
) -> ParamSpec:
    """Build a model-specific extra parameter spec.

    This function dispatches to the appropriate builder based on the parameter
    name. It centralizes the logic for building gate and capture parameters.

    Parameters
    ----------
    param_name : str
        Name of the parameter to build ("gate" or "p_capture").
    unconstrained : bool
        Whether to use unconstrained parameterization.
    guide_families : GuideFamilyConfig
        Guide family configuration.
    param_strategy : Parameterization
        Parameterization strategy (for capture parameter transformation).
    n_components : int, optional
        Number of mixture components.
    mixture_params : List[str], optional
        List of mixture-specific parameters.

    Returns
    -------
    ParamSpec
        Parameter specification for the requested parameter.

    Raises
    ------
    ValueError
        If param_name is not recognized.
    """
    if param_name == "gate":
        return build_gate_spec(
            unconstrained=unconstrained,
            guide_families=guide_families,
            n_components=n_components,
            mixture_params=mixture_params,
        )
    elif param_name == "p_capture":
        return build_capture_spec(
            unconstrained=unconstrained,
            guide_families=guide_families,
            param_strategy=param_strategy,
        )
    else:
        raise ValueError(
            f"Unknown extra parameter: {param_name}. "
            f"Valid parameters are: gate, p_capture"
        )


# ==============================================================================
# Prior/Guide Override Helpers
# ==============================================================================


def apply_prior_guide_overrides(
    param_specs: List[ParamSpec],
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> List[ParamSpec]:
    """Apply user-provided prior and guide overrides to parameter specs.

    This function takes a list of parameter specs and updates them with
    user-provided prior and guide hyperparameters. It creates new spec
    instances (immutable pattern) rather than modifying in place.

    Validation is performed automatically:
    - Parameter names are checked against the available specs
    - Hyperparameter tuple length and values are validated by ParamSpec

    Parameters
    ----------
    param_specs : List[ParamSpec]
        List of parameter specifications to update.
    priors : Dict[str, Tuple[float, ...]], optional
        Dictionary mapping parameter names to prior hyperparameters.
        Example: {"p": (1.0, 1.0), "r": (0.0, 1.0)}
    guides : Dict[str, Tuple[float, ...]], optional
        Dictionary mapping parameter names to guide hyperparameters.

    Returns
    -------
    List[ParamSpec]
        Updated list of parameter specifications with overrides applied.

    Raises
    ------
    ValueError
        If a parameter name in priors/guides is not found in param_specs.
        If hyperparameter values are invalid for the parameter's distribution.

    Examples
    --------
    >>> specs = [BetaSpec(name="p", ...), LogNormalSpec(name="r", ...)]
    >>> updated = apply_prior_guide_overrides(
    ...     specs,
    ...     priors={"p": (2.0, 2.0)},  # Informative Beta prior
    ... )
    """
    if priors is None and guides is None:
        return param_specs

    # Validate parameter names exist in specs
    spec_names = {spec.name for spec in param_specs}

    if priors is not None:
        unknown_priors = set(priors.keys()) - spec_names
        if unknown_priors:
            raise ValueError(
                f"Unknown parameter names in priors: {sorted(unknown_priors)}. "
                f"Valid parameters for this model: {sorted(spec_names)}"
            )

    if guides is not None:
        unknown_guides = set(guides.keys()) - spec_names
        if unknown_guides:
            raise ValueError(
                f"Unknown parameter names in guides: {sorted(unknown_guides)}. "
                f"Valid parameters for this model: {sorted(spec_names)}"
            )

    # Apply overrides (ParamSpec validates hyperparameter values)
    updated_specs = []
    for spec in param_specs:
        updates = {}
        if priors is not None and spec.name in priors:
            updates["prior"] = priors[spec.name]
        if guides is not None and spec.name in guides:
            updates["guide"] = guides[spec.name]

        if updates:
            # Create new spec with updates (immutable pattern)
            # ParamSpec's validate_hyperparameters will validate the values
            updated_spec = spec.model_copy(update=updates)
            updated_specs.append(updated_spec)
        else:
            updated_specs.append(spec)

    return updated_specs


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    # Registries
    "MODEL_EXTRA_PARAMS",
    "LIKELIHOOD_REGISTRY",
    "GUIDE_FAMILY_REGISTRY",
    # Builders
    "build_gate_spec",
    "build_capture_spec",
    "build_extra_param_spec",
    # Helpers
    "apply_prior_guide_overrides",
    "get_guide_family",
]

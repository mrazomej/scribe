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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

if TYPE_CHECKING:
    from ..components.amortizers import Amortizer
    from ..config.groups import AmortizationConfig

import math

from ..builders.parameter_specs import (
    BiologyInformedCaptureSpec,
    BetaPrimeSpec,
    BetaSpec,
    HorseshoeBNBConcentrationSpec,
    NEGBNBConcentrationSpec,
    PositiveNormalSpec,
    HierarchicalSigmoidNormalSpec,
    NormalWithTransformSpec,
    ParamSpec,
    SigmoidNormalSpec,
    SoftplusNormalSpec,
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
    BetaNegativeBinomialLikelihood,
    ZeroInflatedBNBLikelihood,
    BNBWithVCPLikelihood,
    ZIBNBWithVCPLikelihood,
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

# BNB counterparts -- selected when overdispersion='bnb'
BNB_LIKELIHOOD_REGISTRY: Dict[str, Type[Likelihood]] = {
    "nbdm": BetaNegativeBinomialLikelihood,
    "zinb": ZeroInflatedBNBLikelihood,
    "nbvcp": BNBWithVCPLikelihood,
    "zinbvcp": ZIBNBWithVCPLikelihood,
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
    hierarchical: bool = False,
) -> List[ParamSpec]:
    """Build gate parameter spec(s) for zero-inflated models (ZINB, ZINBVCP).

    The gate parameter controls the probability of structural zeros per gene.
    It is always gene-specific.

    When ``hierarchical=True``, returns a list of three specs: two global
    hyperparameters (logit_gate_loc, logit_gate_scale) and one hierarchical
    gene-specific gate (``HierarchicalSigmoidNormalSpec``). The hyperprior
    defaults encode "most genes don't need zero-inflation" via a strongly
    negative location in logit space.

    Parameters
    ----------
    unconstrained : bool
        If True, use SigmoidNormalSpec (Normal + sigmoid transform).
        If False, use BetaSpec (constrained Beta distribution).
        Ignored when hierarchical=True (always unconstrained).
    guide_families : GuideFamilyConfig
        Guide family configuration for retrieving gate's guide family.
    n_components : int, optional
        Number of mixture components. If provided and gate is in mixture_params,
        the parameter will be marked as mixture-specific.
    mixture_params : List[str], optional
        List of parameters that should be mixture-specific. If None and
        n_components is set, gate defaults to being mixture-specific.
    hierarchical : bool, default=False
        If True, return hierarchical triplet (hyperprior loc, hyperprior
        scale, hierarchical gate) instead of a flat gate spec.

    Returns
    -------
    List[ParamSpec]
        One or three parameter specifications for the gate parameter.
    """
    gate_family = guide_families.get("gate")

    # Determine if gate should be mixture-specific
    is_mixture = False
    if n_components is not None:
        if mixture_params is None:
            is_mixture = True
        else:
            is_mixture = "gate" in mixture_params

    if hierarchical:
        # Hierarchical gate: hyperprior loc + hyperprior scale + hierarchical
        # spec
        return [
            NormalWithTransformSpec(
                name="logit_gate_loc",
                shape_dims=(),
                default_params=(-5.0, 1.0),
            ),
            SoftplusNormalSpec(
                name="logit_gate_scale",
                shape_dims=(),
                default_params=(-2.0, 0.5),
            ),
            HierarchicalSigmoidNormalSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name="logit_gate_loc",
                hyper_scale_name="logit_gate_scale",
                is_gene_specific=True,
                guide_family=gate_family,
                is_mixture=is_mixture,
            ),
        ]

    if unconstrained:
        return [
            SigmoidNormalSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(-2.0, 1.0),
                is_gene_specific=True,
                guide_family=gate_family,
                is_mixture=is_mixture,
            )
        ]
    else:
        return [
            BetaSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(
                    1.0,
                    9.0,
                ),
                is_gene_specific=True,
                guide_family=gate_family,
                is_mixture=is_mixture,
            )
        ]


# ------------------------------------------------------------------------------


def build_capture_spec(
    unconstrained: bool,
    guide_families: GuideFamilyConfig,
    param_strategy: Parameterization,
    model_config: Optional[Any] = None,
    positive_transform: Optional[Any] = None,
) -> ParamSpec:
    """Build capture probability parameter spec for VCP models (NBVCP, ZINBVCP).

    The capture parameter models cell-specific technical variation in mRNA
    capture efficiency. It is always cell-specific.

    For mean_odds parameterization, the parameter is transformed from p_capture
    to phi_capture (odds ratio parameterization).

    When ``model_config.capture_prior`` is ``"biology_informed"``, returns a
    ``BiologyInformedCaptureSpec`` that anchors the capture probability to
    library size via total mRNA per cell.  If ``mu_eta_prior`` is non-NONE,
    the spec learns a per-dataset ``mu_eta`` with hierarchical shrinkage.

    If amortization is enabled in guide_families.capture_amortization, the
    guide will use a neural network to predict variational parameters from
    total UMI count instead of learning separate parameters per cell.

    Parameters
    ----------
    unconstrained : bool
        If True, use transformed Normal distribution.
        If False, use constrained distribution (Beta or BetaPrime).
    guide_families : GuideFamilyConfig
        Guide family configuration for retrieving capture's guide family.
        If capture_amortization is set and enabled, amortized inference
        will be used.
    param_strategy : Parameterization
        Parameterization strategy to determine parameter name transformation
        (p_capture vs phi_capture for mean_odds).
    model_config : ModelConfig, optional
        Full model configuration. Used to check ``capture_prior`` mode
        and organism/M_0 settings.
    positive_transform : Transform, optional
        Positive transform to use for unconstrained ``phi_capture``.
        If None, ``PositiveNormalSpec`` default transform is used
        (``ExpTransform`` for backward compatibility).

    Returns
    -------
    ParamSpec
        Parameter specification for the capture parameter.

    Notes
    -----
    When amortization is enabled, the guide family will be an AmortizedGuide
    with an attached Amortizer network, regardless of what was specified in
    guide_families.p_capture or guide_families.phi_capture.
    """
    # Get the appropriate capture parameter name based on parameterization
    # mean_odds uses phi_capture, others use p_capture
    capture_param_name = param_strategy.transform_model_param("p_capture")
    use_phi_capture = capture_param_name == "phi_capture"

    # ---- Biology-informed capture prior path ----
    # Read eta_capture / mu_eta from the priors dict (already resolved
    # by the ModelConfig validator from priors.organism if applicable).
    priors_extra = {}
    if model_config is not None:
        priors_extra = (
            getattr(model_config.priors, "__pydantic_extra__", None) or {}
        )

    eta_capture = priors_extra.get("eta_capture")
    mu_eta = priors_extra.get("mu_eta")

    # Resolve mu_eta_prior from config (HierarchicalPriorType enum)
    from ..config.enums import HierarchicalPriorType

    mu_eta_prior_enum = getattr(
        model_config, "mu_eta_prior", HierarchicalPriorType.NONE
    )
    mu_eta_prior_str = (
        None
        if mu_eta_prior_enum == HierarchicalPriorType.NONE
        else mu_eta_prior_enum.value
    )

    if eta_capture is not None:
        log_M0, sigma_M = eta_capture

        spec_kwargs = dict(
            name=capture_param_name,
            shape_dims=("n_cells",),
            default_params=(log_M0, sigma_M),
            is_cell_specific=True,
            guide_family=guide_families.get(capture_param_name),
            log_M0=log_M0,
            sigma_M=sigma_M,
            mu_eta_prior=mu_eta_prior_str,
            use_phi_capture=use_phi_capture,
        )
        # sigma_mu comes from mu_eta prior config when data-driven
        if mu_eta_prior_str is not None and mu_eta is not None:
            spec_kwargs["sigma_mu"] = mu_eta[1]

        return BiologyInformedCaptureSpec(**spec_kwargs)

    # ---- Standard (flat) capture prior path ----
    # Check if amortization is enabled for capture probability
    amort_config = guide_families.capture_amortization
    if amort_config is not None and amort_config.enabled:
        # Create amortizer from config object (single place unpacks into
        # create_capture_amortizer)
        param_name = (
            param_strategy.name
            if hasattr(param_strategy, "name")
            else "canonical"
        )
        amortizer = create_capture_amortizer_from_config(
            amort_config,
            parameterization=param_name,
            unconstrained=unconstrained,
        )
        capture_family = AmortizedGuide(amortizer=amortizer)
    else:
        # Use the configured guide family or default
        capture_family = guide_families.get(capture_param_name)

    if unconstrained:
        if use_phi_capture:
            # Keep default behavior (ExpTransform) when no transform is
            # provided by caller, but allow factory-resolved
            # positive_transform to drive unconstrained phi_capture.
            _kwargs = {}
            if positive_transform is not None:
                _kwargs["transform"] = positive_transform
            return PositiveNormalSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(0.0, 1.0),
                is_cell_specific=True,
                guide_family=capture_family,
                **_kwargs,
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
        if use_phi_capture:
            return BetaPrimeSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(1.0, 1.0),
                is_cell_specific=True,
                guide_family=capture_family,
            )
        else:
            return BetaSpec(
                name=capture_param_name,
                shape_dims=("n_cells",),
                default_params=(1.0, 1.0),
                is_cell_specific=True,
                guide_family=capture_family,
            )


# ------------------------------------------------------------------------------


def build_bnb_concentration_spec(
    overdispersion_prior: str,
    guide_families: GuideFamilyConfig,
    horseshoe_tau0: float = 1.0,
    horseshoe_slab_df: int = 4,
    horseshoe_slab_scale: float = 2.0,
    neg_u: float = 1.0,
    neg_a: float = 1.0,
    neg_tau: float = 1.0,
) -> List[ParamSpec]:
    """Build parameter specs for the BNB concentration kappa_g.

    Returns the hierarchical spec and its auxiliary sites (global
    shrinkage, per-gene local scales, slab / NEG variance sites, and
    the shared hyper-location).

    The constrained parameter is ``omega_g = softplus(loc + scale * z)``,
    the excess-dispersion fraction.  The likelihood converts this to
    ``kappa_g = 2 + (r_g + 1) / omega_g``.

    Parameters
    ----------
    overdispersion_prior : str
        Prior family: ``"horseshoe"`` or ``"neg"``.
    guide_families : GuideFamilyConfig
        Guide family configuration.
    horseshoe_tau0 : float
        Global shrinkage scale for horseshoe.
    horseshoe_slab_df : int
        Slab degrees of freedom for horseshoe.
    horseshoe_slab_scale : float
        Slab scale for horseshoe.
    neg_u : float
        NEG inner shape parameter.
    neg_a : float
        NEG tail parameter.
    neg_tau : float
        NEG global rate.

    Returns
    -------
    List[ParamSpec]
        Specs for hyper-location, auxiliary sites, and the hierarchical
        ``bnb_concentration`` parameter.
    """
    from ..builders.parameter_specs import (
        NormalWithTransformSpec,
        HalfCauchySpec,
        InverseGammaSpec,
        GammaSpec,
    )

    specs: List[ParamSpec] = []

    # Shared location in unconstrained space (strongly negative so that
    # softplus(loc) ~ 0, defaulting to NB behaviour).
    specs.append(
        NormalWithTransformSpec(
            name="bnb_concentration_loc",
            shape_dims=(),
            default_params=(-5.0, 1.0),
        )
    )

    if overdispersion_prior == "horseshoe":
        # Global shrinkage scale
        specs.append(
            HalfCauchySpec(
                name="bnb_concentration_tau",
                shape_dims=(),
                default_params=(horseshoe_tau0,),
            )
        )
        # Per-gene local scales
        specs.append(
            HalfCauchySpec(
                name="bnb_concentration_lambda",
                shape_dims=("n_genes",),
                default_params=(1.0,),
                is_gene_specific=True,
            )
        )
        # Slab for regularisation
        specs.append(
            InverseGammaSpec(
                name="bnb_concentration_c_sq",
                shape_dims=(),
                concentration=horseshoe_slab_df / 2.0,
                rate=horseshoe_slab_df * horseshoe_slab_scale**2 / 2.0,
            )
        )
        # The hierarchical spec itself
        specs.append(
            HorseshoeBNBConcentrationSpec(
                name="bnb_concentration",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name="bnb_concentration_loc",
                hyper_scale_name="bnb_concentration_loc",
                tau_name="bnb_concentration_tau",
                lambda_name="bnb_concentration_lambda",
                c_sq_name="bnb_concentration_c_sq",
                raw_name="bnb_concentration_raw",
                is_gene_specific=True,
                guide_family=guide_families.get("bnb_concentration"),
            )
        )
    elif overdispersion_prior == "neg":
        # NEG outer rate site: zeta_g ~ Gamma(a, tau)
        specs.append(
            GammaSpec(
                name="bnb_concentration_zeta",
                shape_dims=("n_genes",),
                concentration=neg_a,
                rate=neg_tau,
                is_gene_specific=True,
            )
        )
        # NEG inner variance site: psi_g ~ Gamma(u, zeta_g)
        specs.append(
            GammaSpec(
                name="bnb_concentration_psi",
                shape_dims=("n_genes",),
                concentration=neg_u,
                rate_name="bnb_concentration_zeta",
                is_gene_specific=True,
            )
        )
        # The hierarchical spec itself
        specs.append(
            NEGBNBConcentrationSpec(
                name="bnb_concentration",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name="bnb_concentration_loc",
                hyper_scale_name="bnb_concentration_loc",
                psi_name="bnb_concentration_psi",
                zeta_name="bnb_concentration_zeta",
                raw_name="bnb_concentration_raw",
                is_gene_specific=True,
                guide_family=guide_families.get("bnb_concentration"),
            )
        )
    else:
        raise ValueError(
            f"Unsupported overdispersion_prior: {overdispersion_prior!r}. "
            "Valid options: 'horseshoe', 'neg'."
        )

    return specs


# ------------------------------------------------------------------------------


def build_extra_param_spec(
    param_name: str,
    unconstrained: bool,
    guide_families: GuideFamilyConfig,
    param_strategy: Parameterization,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    hierarchical_gate: bool = False,
    model_config: Optional[Any] = None,
    positive_transform: Optional[Any] = None,
) -> List[ParamSpec]:
    """Build model-specific extra parameter spec(s).

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
    hierarchical_gate : bool, default=False
        If True and param_name is "gate", build hierarchical gate specs.
    model_config : ModelConfig, optional
        Full model configuration (passed to ``build_capture_spec``).
    positive_transform : Transform, optional
        Positive transform for unconstrained ``phi_capture``.

    Returns
    -------
    List[ParamSpec]
        Parameter specification(s) for the requested parameter.

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
            hierarchical=hierarchical_gate,
        )
    elif param_name == "p_capture":
        return [
            build_capture_spec(
                unconstrained=unconstrained,
                guide_families=guide_families,
                param_strategy=param_strategy,
                model_config=model_config,
                positive_transform=positive_transform,
            )
        ]
    elif param_name == "bnb_concentration":
        return build_bnb_concentration_spec(
            overdispersion_prior=(
                model_config.overdispersion_prior.value
                if model_config is not None
                else "horseshoe"
            ),
            guide_families=guide_families,
            horseshoe_tau0=(
                model_config.horseshoe_tau0
                if model_config is not None
                else 1.0
            ),
            horseshoe_slab_df=(
                model_config.horseshoe_slab_df
                if model_config is not None
                else 4
            ),
            horseshoe_slab_scale=(
                model_config.horseshoe_slab_scale
                if model_config is not None
                else 2.0
            ),
            neg_u=(
                model_config.neg_u if model_config is not None else 1.0
            ),
            neg_a=(
                model_config.neg_a if model_config is not None else 1.0
            ),
            neg_tau=(
                model_config.neg_tau if model_config is not None else 1.0
            ),
        )
    else:
        raise ValueError(
            f"Unknown extra parameter: {param_name}. "
            f"Valid parameters are: gate, p_capture, bnb_concentration"
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
# Amortizer Factory
# ==============================================================================


class _SoftplusOffsetTransform:
    """Picklable softplus+offset transform with optional clamping.

    Module-level class (not a nested function) so amortizer output_transforms
    can be pickled. softplus(x) + offset is numerically stable and grows
    linearly for large inputs.
    """

    def __init__(
        self,
        offset: float,
        clamp_min: Optional[float],
        clamp_max: Optional[float],
    ):
        self.offset = offset
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, x):
        import jax
        import jax.numpy as jnp

        val = jax.nn.softplus(x) + self.offset
        if self.clamp_min is not None or self.clamp_max is not None:
            val = jnp.clip(val, self.clamp_min, self.clamp_max)
        return val


# ------------------------------------------------------------------------------


class _ExpTransform:
    """Picklable exp transform with optional clamping.

    Module-level class (not a nested function) so amortizer output_transforms
    can be pickled.
    """

    def __init__(self, clamp_min: Optional[float], clamp_max: Optional[float]):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, x):
        import jax.numpy as jnp

        val = jnp.exp(x)
        if self.clamp_min is not None or self.clamp_max is not None:
            val = jnp.clip(val, self.clamp_min, self.clamp_max)
        return val


# ------------------------------------------------------------------------------


def create_capture_amortizer(
    hidden_dims: List[int] = None,
    activation: str = "relu",
    input_transformation: str = "log1p",
    parameterization: str = "canonical",
    unconstrained: bool = False,
    output_transform: str = "softplus",
    output_clamp_min: Optional[float] = 0.1,
    output_clamp_max: Optional[float] = 50.0,
) -> "Amortizer":
    """Create an amortizer for capture probability (p_capture or phi_capture).

    This factory creates an MLP that maps total UMI count (a sufficient
    statistic for capture probability) to variational posterior parameters.

    The output parameters depend on the parameterization:

    **Constrained (unconstrained=False)**:
        - Output: alpha, beta (already positive via output transform)
        - Transform: softplus+offset (default) or exp, with optional clamping
        - Distribution: Beta(α, β) for p_capture, BetaPrime(α, β) for
          phi_capture

    **Unconstrained (unconstrained=True)**:
        - Output: loc, log_scale → (identity, exp) → loc, scale
        - Distribution: Normal(loc, scale) → transform → parameter
        - Transform: sigmoid for p_capture, exp for phi_capture

    Parameters
    ----------
    hidden_dims : List[int], optional
        Hidden layer dimensions for the MLP. Default: [64, 32].
    activation : str, default="relu"
        Activation function for hidden layers.
    input_transformation : str, default="log1p"
        Transformation applied to counts before computing total.
        Options: "log1p", "log", "sqrt", "identity".
    parameterization : str, default="canonical"
        Model parameterization. Determines output distribution:
        - "canonical" / "mean_prob": Beta/SigmoidNormal for p_capture
        - "mean_odds": BetaPrime/ExpNormal for phi_capture
    unconstrained : bool, default=False
        Whether to use unconstrained parameterization (Normal + transform).
        If True, outputs (loc, scale) for Normal. If False, outputs
        (alpha, beta) for Beta/BetaPrime.
    output_transform : str, default="softplus"
        Transform for positive output parameters in constrained mode.
        "softplus": softplus(x) + 0.5 — bounded away from zero, grows
        linearly. "exp": exponential (original behavior). Only used when
        unconstrained=False.
    output_clamp_min : float or None, default=0.1
        Minimum clamp for positive outputs in constrained mode. Prevents
        extreme BetaPrime/Beta shape parameters. None disables lower clamp.
    output_clamp_max : float or None, default=50.0
        Maximum clamp for positive outputs in constrained mode. None
        disables upper clamp.

    Returns
    -------
    Amortizer
        Configured amortizer network ready for use with AmortizedGuide.

    Examples
    --------
    >>> # Constrained with softplus (default, numerically stable)
    >>> amortizer = create_capture_amortizer(
    ...     hidden_dims=[128, 64],
    ...     activation="gelu",
    ... )

    >>> # Constrained with exp (original behavior)
    >>> amortizer = create_capture_amortizer(
    ...     output_transform="exp",
    ...     output_clamp_min=None,
    ...     output_clamp_max=None,
    ... )

    >>> # Unconstrained (Normal + sigmoid)
    >>> amortizer = create_capture_amortizer(
    ...     unconstrained=True,
    ... )

    Notes
    -----
    The theoretical justification for using total UMI count as a sufficient
    statistic comes from the Dirichlet-Multinomial model derivation. In that
    model, the marginal distribution of total counts T = Σᵢ xᵢ depends only
    on p_capture (through the effective probability p̂), making T sufficient.

    See Also
    --------
    scribe.models.components.amortizers.Amortizer : The MLP class.
    scribe.models.components.amortizers.TOTAL_COUNT : The sufficient statistic.
    """
    from ..components.amortizers import (
        Amortizer,
        SufficientStatistic,
        _compute_total_count,
        _compute_total_count_identity,
        _compute_total_count_log,
        _compute_total_count_sqrt,
    )

    # Default hidden dimensions
    if hidden_dims is None:
        hidden_dims = [64, 32]

    # Create appropriate sufficient statistic based on input transformation.
    # Use module-level functions (not lambdas) so amortizer is picklable.
    INPUT_TRANSFORMS = {
        "log1p": _compute_total_count,
        "log": _compute_total_count_log,
        "sqrt": _compute_total_count_sqrt,
        "identity": _compute_total_count_identity,
    }

    if input_transformation not in INPUT_TRANSFORMS:
        raise ValueError(
            f"Unknown input_transformation: {input_transformation}. "
            f"Valid options: {list(INPUT_TRANSFORMS.keys())}"
        )

    sufficient_statistic = SufficientStatistic(
        name=f"total_count_{input_transformation}",
        compute=INPUT_TRANSFORMS[input_transformation],
    )

    # Determine output parameters based on constrained vs unconstrained
    if unconstrained:
        # Unconstrained: Normal(loc, scale) → transform → parameter
        # Output loc and log_scale as raw values (no amortizer-level transform).
        # The guide_builder applies exp(log_scale) to get the positive scale.
        # No clamping needed — Normal is numerically robust for any params
        output_params = ["loc", "log_scale"]
        output_transforms = None  # Identity for all outputs
    else:
        # Constrained: Beta(alpha, beta) or BetaPrime(alpha, beta)
        # Both use (alpha, beta) parameterization with alpha, beta > 0
        output_params = ["alpha", "beta"]

        if output_transform == "softplus":
            pos_transform = _SoftplusOffsetTransform(
                offset=0.5,
                clamp_min=output_clamp_min,
                clamp_max=output_clamp_max,
            )
        elif output_transform == "exp":
            pos_transform = _ExpTransform(
                clamp_min=output_clamp_min,
                clamp_max=output_clamp_max,
            )
        else:
            raise ValueError(
                f"Unknown output_transform: '{output_transform}'. "
                f"Valid options: 'exp', 'softplus'"
            )

        output_transforms = {
            "alpha": pos_transform,
            "beta": pos_transform,
        }

    # Create and return the amortizer
    return Amortizer(
        sufficient_statistic=sufficient_statistic,
        hidden_dims=hidden_dims,
        output_params=output_params,
        output_transforms=output_transforms,
        input_dim=1,  # Scalar sufficient statistic (total count)
        activation=activation,
    )


# ------------------------------------------------------------------------------


def create_capture_amortizer_from_config(
    config: "AmortizationConfig",
    parameterization: str = "canonical",
    unconstrained: bool = False,
) -> "Amortizer":
    """Create an amortizer from an AmortizationConfig object.

    This is a convenience function that extracts parameters from the config
    and calls create_capture_amortizer.

    Parameters
    ----------
    config : AmortizationConfig
        Configuration object specifying the MLP architecture.
    parameterization : str, default="canonical"
        Model parameterization (affects output distribution interpretation).
    unconstrained : bool, default=False
        Whether to use unconstrained parameterization. Determines output
        parameter types: (loc, scale) for unconstrained, (alpha, beta) for
        constrained.

    Returns
    -------
    Amortizer
        Configured amortizer network.

    Examples
    --------
    >>> from scribe.models.config import AmortizationConfig
    >>> config = AmortizationConfig(
    ...     enabled=True,
    ...     hidden_dims=[128, 64],
    ...     activation="gelu",
    ... )
    >>> amortizer = create_capture_amortizer_from_config(config)

    >>> # Unconstrained version
    >>> amortizer = create_capture_amortizer_from_config(
    ...     config, unconstrained=True
    ... )
    """
    return create_capture_amortizer(
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        input_transformation=config.input_transformation,
        parameterization=parameterization,
        unconstrained=unconstrained,
        output_transform=config.output_transform,
        output_clamp_min=config.output_clamp_min,
        output_clamp_max=config.output_clamp_max,
    )


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    # Registries
    "MODEL_EXTRA_PARAMS",
    "LIKELIHOOD_REGISTRY",
    "BNB_LIKELIHOOD_REGISTRY",
    "GUIDE_FAMILY_REGISTRY",
    # Builders
    "build_gate_spec",
    "build_capture_spec",
    "build_bnb_concentration_spec",
    "build_extra_param_spec",
    # Helpers
    "apply_prior_guide_overrides",
    "get_guide_family",
    # Amortizer factory
    "create_capture_amortizer",
    "create_capture_amortizer_from_config",
]

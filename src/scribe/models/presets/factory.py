"""Unified model factory for creating SCRIBE models.

This module provides a single entry point for creating model and guide
functions from a ModelConfig. It replaces the individual preset factories
(create_nbdm, create_zinb, create_nbvcp, create_zinbvcp) with a unified
approach using registries and helper builders.

Functions
---------
create_model
    Create model and guide functions from a ModelConfig.
create_model_from_params
    Create model and guide functions from individual parameters.

Examples
--------
>>> from scribe.models.config import ModelConfig
>>> from scribe.models.presets.factory import create_model
>>>
>>> # Create model from config
>>> model, guide = create_model(model_config)
>>>
>>> # Or use the convenience function with individual params
>>> model, guide = create_model_from_params(
...     model="zinb",
...     parameterization="linked",
...     unconstrained=True,
... )
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

from ..builders import GuideBuilder, ModelBuilder
from ..config import GuideFamilyConfig, ModelConfig
from ..config.enums import Parameterization as ParamEnum
from ..parameterizations import PARAMETERIZATIONS, Parameterization
from .registry import (
    LIKELIHOOD_REGISTRY,
    MODEL_EXTRA_PARAMS,
    apply_prior_guide_overrides,
    build_extra_param_spec,
)

# ==============================================================================
# Unified Model Factory
# ==============================================================================


def create_model(
    model_config: ModelConfig,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> Tuple[Callable, Callable]:
    """Create model and guide functions from a ModelConfig.

    This is the unified factory that replaces create_nbdm, create_zinb,
    create_nbvcp, and create_zinbvcp. It uses registries to determine
    model-specific components and builds the model/guide using the
    composable builder pattern.

    Parameters
    ----------
    model_config : ModelConfig
        Fully configured model configuration object containing:
        - base_model: Model type (nbdm, zinb, nbvcp, zinbvcp)
        - parameterization: Parameterization scheme
        - unconstrained: Whether to use unconstrained distributions
        - n_components: Number of mixture components (optional)
        - mixture_params: Which params are mixture-specific (optional)
        - guide_families: Per-parameter guide family config (optional)
    priors : Dict[str, Tuple[float, ...]], optional
        User-provided prior hyperparameters keyed by parameter name.
        Overrides default priors. Example: {"p": (2.0, 2.0)}
    guides : Dict[str, Tuple[float, ...]], optional
        User-provided guide hyperparameters keyed by parameter name.

    Returns
    -------
    model : Callable
        NumPyro model function with signature:
        model(n_cells, n_genes, model_config, counts=None, batch_size=None)
    guide : Callable
        NumPyro guide function with the same signature.

    Raises
    ------
    ValueError
        If model type or parameterization is not recognized.

    Examples
    --------
    >>> from scribe.models.config import ModelConfigBuilder
    >>> from scribe.models.presets.factory import create_model
    >>>
    >>> # Build config and create model
    >>> config = (
    ...     ModelConfigBuilder()
    ...     .for_model("zinb")
    ...     .with_parameterization("linked")
    ...     .build()
    ... )
    >>> model, guide = create_model(config)
    >>>
    >>> # With custom priors
    >>> model, guide = create_model(
    ...     config,
    ...     priors={"p": (2.0, 2.0), "mu": (1.0, 0.5)},
    ... )

    See Also
    --------
    create_model_from_params : Convenience function with flat parameters.
    """
    # ==========================================================================
    # Step 1: Validate and get parameterization strategy
    # ==========================================================================
    param_key = _get_parameterization_key(model_config.parameterization)
    if param_key not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {model_config.parameterization}. "
            f"Supported: {list(PARAMETERIZATIONS.keys())}"
        )
    param_strategy = PARAMETERIZATIONS[param_key]

    # Validate model type
    base_model = model_config.base_model
    if base_model not in MODEL_EXTRA_PARAMS:
        raise ValueError(
            f"Unknown model type: {base_model}. "
            f"Supported: {list(MODEL_EXTRA_PARAMS.keys())}"
        )

    # ==========================================================================
    # Step 2: Resolve guide families
    # ==========================================================================
    guide_families = model_config.guide_families or GuideFamilyConfig()

    # ==========================================================================
    # Step 3: Build core parameter specs from parameterization strategy
    # ==========================================================================
    param_specs = param_strategy.build_param_specs(
        unconstrained=model_config.unconstrained,
        guide_families=guide_families,
        n_components=model_config.n_components,
        mixture_params=model_config.mixture_params,
    )

    # ==========================================================================
    # Step 4: Add model-specific extra parameters
    # ==========================================================================
    extra_param_names = MODEL_EXTRA_PARAMS[base_model]
    for param_name in extra_param_names:
        extra_spec = build_extra_param_spec(
            param_name=param_name,
            unconstrained=model_config.unconstrained,
            guide_families=guide_families,
            param_strategy=param_strategy,
            n_components=model_config.n_components,
            mixture_params=model_config.mixture_params,
        )
        param_specs.append(extra_spec)

    # ==========================================================================
    # Step 5: Apply user-provided prior/guide overrides
    # ==========================================================================
    # Merge priors from model_config.param_specs with explicit priors argument
    merged_priors = _extract_priors_from_param_specs(model_config.param_specs)
    if priors:
        merged_priors.update(priors)

    merged_guides = _extract_guides_from_param_specs(model_config.param_specs)
    if guides:
        merged_guides.update(guides)

    if merged_priors or merged_guides:
        param_specs = apply_prior_guide_overrides(
            param_specs,
            priors=merged_priors or None,
            guides=merged_guides or None,
        )

    # ==========================================================================
    # Step 6: Get derived parameters from parameterization strategy
    # ==========================================================================
    derived_params = param_strategy.build_derived_params()

    # ==========================================================================
    # Step 7: Build model using ModelBuilder
    # ==========================================================================
    model_builder = ModelBuilder()
    for spec in param_specs:
        model_builder.add_param(spec)
    for d_param in derived_params:
        model_builder.add_derived(d_param.name, d_param.compute, d_param.deps)

    # Get likelihood from registry
    likelihood_class = LIKELIHOOD_REGISTRY[base_model]
    model_builder.with_likelihood(likelihood_class())

    model = model_builder.build()

    # ==========================================================================
    # Step 8: Build guide using GuideBuilder
    # ==========================================================================
    guide = GuideBuilder().from_specs(param_specs).build()

    return model, guide


# ------------------------------------------------------------------------------


def create_model_from_params(
    model: str,
    parameterization: str = "canonical",
    unconstrained: bool = False,
    guide_families: Optional[GuideFamilyConfig] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> Tuple[Callable, Callable]:
    """Create model and guide functions from individual parameters.

    This is a convenience function that creates a ModelConfig internally and
    calls create_model(). It provides the same interface as the old preset
    factories but routes through the unified factory.

    Parameters
    ----------
    model : str
        Model type: "nbdm", "zinb", "nbvcp", or "zinbvcp".
    parameterization : str, default="canonical"
        Parameterization scheme: "canonical", "mean_prob", "mean_odds"
        (or aliases: "standard", "linked", "odds_ratio").
    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained distributions.
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration.
    n_components : int, optional
        Number of mixture components for mixture models.
    mixture_params : List[str], optional
        List of parameters to make mixture-specific.
    priors : Dict[str, Tuple[float, ...]], optional
        Prior hyperparameters keyed by parameter name.
    guides : Dict[str, Tuple[float, ...]], optional
        Guide hyperparameters keyed by parameter name.

    Returns
    -------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function.

    Examples
    --------
    >>> model, guide = create_model_from_params(
    ...     model="zinb",
    ...     parameterization="linked",
    ...     n_components=3,
    ... )
    """
    from ..config import ModelConfigBuilder

    # Build ModelConfig
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization(parameterization)
    )

    if unconstrained:
        builder.unconstrained()

    if guide_families is not None:
        builder.with_guide_families(guide_families)

    if n_components is not None:
        builder.as_mixture(n_components, mixture_params)

    model_config = builder.build()

    return create_model(model_config, priors=priors, guides=guides)


# ==============================================================================
# Helper Functions
# ==============================================================================


def _get_parameterization_key(param: Union[str, ParamEnum]) -> str:
    """Convert parameterization enum or string to registry key."""
    if isinstance(param, ParamEnum):
        # Map enum values to registry keys
        enum_to_key = {
            ParamEnum.STANDARD: "canonical",
            ParamEnum.LINKED: "mean_prob",
            ParamEnum.ODDS_RATIO: "mean_odds",
        }
        return enum_to_key.get(param, param.value)
    return param


# ------------------------------------------------------------------------------


def _extract_priors_from_param_specs(
    param_specs: List,
) -> Dict[str, Tuple[float, ...]]:
    """Extract prior overrides from a list of ParamSpec objects."""
    priors = {}
    for spec in param_specs:
        if hasattr(spec, "prior") and spec.prior is not None:
            priors[spec.name] = spec.prior
    return priors


# ------------------------------------------------------------------------------


def _extract_guides_from_param_specs(
    param_specs: List,
) -> Dict[str, Tuple[float, ...]]:
    """Extract guide overrides from a list of ParamSpec objects."""
    guides = {}
    for spec in param_specs:
        if hasattr(spec, "guide") and spec.guide is not None:
            guides[spec.name] = spec.guide
    return guides


# ==============================================================================
# Export
# ==============================================================================

__all__ = [
    "create_model",
    "create_model_from_params",
]

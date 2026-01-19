"""
Preset-based model configuration builder.

This module provides functionality to build ModelConfig objects from simple
preset parameters (model name, parameterization, etc.) using the preset factory
system.
"""

from typing import Optional, Dict, Any, List
from ..models.config import ModelConfig, ModelConfigBuilder, GuideFamilyConfig
from ..models.parameterizations import PARAMETERIZATIONS


def build_config_from_preset(
    model: str,
    parameterization: str = "canonical",
    inference_method: str = "svi",
    unconstrained: bool = False,
    guide_rank: Optional[int] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Any]] = None,
) -> ModelConfig:
    """Build ModelConfig using preset factories and parameterization strategies.

    This function creates a ModelConfig by leveraging the preset factory system
    and parameterization strategies. It handles guide family configuration,
    mixture models, and prior specification.

    Parameters
    ----------
    model : str
        Model type: "nbdm", "zinb", "nbvcp", or "zinbvcp".
    parameterization : str, default="canonical"
        Parameterization scheme: "canonical", "mean_prob", "mean_odds"
        (backward compat: "standard", "linked", "odds_ratio").
    inference_method : str, default="svi"
        Inference method: "svi", "mcmc", or "vae".
    unconstrained : bool, default=False
        Whether to use unconstrained parameterization (Normal+transform).
    guide_rank : Optional[int], default=None
        Rank for low-rank guide on gene-specific parameter. If provided,
        creates a LowRankGuide for the appropriate parameter (r or mu).
    n_components : Optional[int], default=None
        Number of mixture components. If provided, creates a mixture model.
    mixture_params : Optional[list], default=None
        List of parameter names to make mixture-specific. If None and
        n_components is set, defaults to all gene-specific parameters.
    priors : Optional[Dict[str, Any]], default=None
        Dictionary of prior parameters keyed by parameter name.
        Values should be tuples of prior hyperparameters.

    Returns
    -------
    ModelConfig
        Fully configured model configuration object.

    Raises
    ------
    ValueError
        If model type is not recognized or parameterization is invalid.

    Examples
    --------
    Basic usage:

    >>> from scribe.inference.preset_builder import build_config_from_preset
    >>>
    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="canonical",
    ...     inference_method="svi"
    ... )

    With low-rank guide:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="mean_prob",
    ...     inference_method="svi",
    ...     guide_rank=15
    ... )

    Mixture model:

    >>> config = build_config_from_preset(
    ...     model="zinb",
    ...     parameterization="mean_odds",
    ...     inference_method="svi",
    ...     n_components=3
    ... )

    With custom priors:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="canonical",
    ...     inference_method="svi",
    ...     priors={"p": (1.0, 1.0), "r": (0.0, 1.0)}
    ... )

    Notes
    -----
    - The preset factories are used to create model/guide functions, but the
      actual ModelConfig is built using ModelConfigBuilder for consistency.
    - Guide rank is applied to the gene-specific parameter (r for canonical,
      mu for mean_prob/mean_odds).
    - Parameterization names support both new ("canonical", "mean_prob", "mean_odds")
      and old ("standard", "linked", "odds_ratio") for backward compatibility.

    See Also
    --------
    scribe.models.presets : Preset factory functions.
    scribe.models.parameterizations : Parameterization strategy classes.
    ModelConfigBuilder : Builder for ModelConfig objects.
    """
    # Validate parameterization is supported
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {parameterization}. "
            f"Supported: {list(PARAMETERIZATIONS.keys())}"
        )

    # Get parameterization strategy to determine gene parameter name
    param_strategy = PARAMETERIZATIONS[parameterization]
    gene_param_name = param_strategy.gene_param_name  # "r" or "mu"

    # Build guide families if guide_rank is specified
    guide_families = None
    if guide_rank is not None:
        from ..models.components import LowRankGuide

        guide_families = GuideFamilyConfig(
            **{gene_param_name: LowRankGuide(rank=guide_rank)}
        )

    # Build ModelConfig using builder
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization(parameterization)
        .with_inference(inference_method)
    )

    # Add unconstrained if needed
    if unconstrained:
        builder.unconstrained()

    # Add mixture configuration
    if n_components is not None:
        builder.as_mixture(n_components, mixture_params)

    # Add guide families if specified
    if guide_families is not None:
        builder.with_guide_families(guide_families)

    # Add priors if specified
    if priors:
        builder.with_priors(**priors)

    return builder.build()

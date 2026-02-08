"""
Preset-based model configuration builder.

This module provides functionality to build ModelConfig objects from simple
preset parameters (model name, parameterization, etc.). The actual model
and guide construction is handled by the unified factory in
`scribe.models.presets.factory`.

Functions
---------
build_config_from_preset
    Build a ModelConfig from simple parameters.

Examples
--------
>>> from scribe.inference.preset_builder import build_config_from_preset
>>>
>>> # Basic usage
>>> config = build_config_from_preset(model="nbdm")
>>>
>>> # With options
>>> config = build_config_from_preset(
...     model="zinb",
...     parameterization="linked",
...     n_components=3,
...     guide_rank=15,
... )

See Also
--------
scribe.models.presets.factory.create_model : Creates model/guide from config.
scribe.models.config.ModelConfigBuilder : Builder for ModelConfig objects.
"""

from typing import Any, Dict, List, Optional, Union

from ..models.config import (
    AmortizationConfig,
    GuideFamilyConfig,
    ModelConfig,
    ModelConfigBuilder,
)
from ..models.parameterizations import PARAMETERIZATIONS


# ==============================================================================
# Build ModelConfig from preset parameters
# ==============================================================================


def build_config_from_preset(
    model: str,
    parameterization: str = "canonical",
    inference_method: str = "svi",
    unconstrained: bool = False,
    guide_rank: Optional[int] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Any]] = None,
    # Amortization options
    amortize_capture: bool = False,
    capture_hidden_dims: Optional[List[int]] = None,
    capture_activation: str = "leaky_relu",
    capture_output_transform: str = "softplus",
    capture_clamp_min: Optional[float] = 0.1,
    capture_clamp_max: Optional[float] = 50.0,
    capture_amortization: Optional[
        Union[AmortizationConfig, Dict[str, Any]]
    ] = None,
) -> ModelConfig:
    """Build ModelConfig from simple preset parameters.

    This function creates a ModelConfig by translating simple parameters
    into the appropriate configuration. It handles:

    - Model type validation
    - Parameterization validation and aliasing
    - Guide family configuration from guide_rank
    - Mixture model configuration

    The actual model/guide construction is done by `create_model()` in
    the factory module.

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
    mixture_params : Optional[List[str]], default=None
        List of parameter names to make mixture-specific. If None and
        n_components is set, defaults to all gene-specific parameters.
    priors : Optional[Dict[str, Any]], default=None
        Dictionary of prior parameters keyed by parameter name.
        Values should be tuples of prior hyperparameters.
        Example: {"p": (2.0, 2.0), "r": (1.0, 0.5)}
    amortize_capture : bool, default=False
        Whether to use amortized inference for capture probability.
        Only applies to VCP models (nbvcp, zinbvcp). When True, uses a
        neural network to predict variational parameters from total UMI.
    capture_hidden_dims : Optional[List[int]], default=None
        Hidden layer dimensions for the capture amortizer MLP.
        Default is [64, 32]. Only used if amortize_capture=True.
    capture_activation : str, default="leaky_relu"
        Activation function for the capture amortizer MLP.
        Only used if amortize_capture=True.
    capture_output_transform : str, default="softplus"
        Transform for positive output parameters in constrained mode.
        "softplus" (default) or "exp". Only used if amortize_capture=True.
    capture_clamp_min : float or None, default=0.1
        Minimum clamp for amortizer positive outputs. Only used if
        amortize_capture=True and unconstrained=False.
    capture_clamp_max : float or None, default=50.0
        Maximum clamp for amortizer positive outputs. Only used if
        amortize_capture=True and unconstrained=False.
    capture_amortization : AmortizationConfig or dict, optional
        Single config object for capture amortization. When provided, it
        is used directly in guide_families and overrides the six
        capture_* parameters above. Can be an AmortizationConfig or a dict
        (converted to AmortizationConfig). When None and amortize_capture=True,
        an AmortizationConfig is built from the six capture_* parameters
        (backward compatible).

    Returns
    -------
    ModelConfig
        Configured model configuration object ready for use with
        `create_model()` or inference functions.

    Raises
    ------
    ValueError
        If model type is not recognized or parameterization is invalid.

    Examples
    --------
    Basic usage:

    >>> config = build_config_from_preset(model="nbdm")

    With low-rank guide:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="mean_prob",
    ...     guide_rank=15,
    ... )

    Mixture model:

    >>> config = build_config_from_preset(
    ...     model="zinb",
    ...     n_components=3,
    ... )

    With custom priors:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     priors={"p": (2.0, 2.0), "r": (1.0, 0.5)},
    ... )

    Notes
    -----
    - Parameterization names support both new ("canonical", "mean_prob",
      "mean_odds") and old ("standard", "linked", "odds_ratio") for
      backward compatibility.
    - Guide rank is applied to the gene-specific parameter (r for canonical,
      mu for mean_prob/mean_odds).
    - The returned ModelConfig can be passed to `create_model()` to get
      the actual model and guide functions.

    See Also
    --------
    scribe.models.presets.factory.create_model : Creates model/guide from config.
    """
    # ==========================================================================
    # Validate parameterization
    # ==========================================================================
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {parameterization}. "
            f"Supported: {list(PARAMETERIZATIONS.keys())}"
        )

    # ==========================================================================
    # Build guide families configuration
    # ==========================================================================
    guide_family_kwargs = {}

    # Handle low-rank guide for gene-specific parameter
    if guide_rank is not None:
        from ..models.components import LowRankGuide

        # Get parameterization strategy to determine gene parameter name
        param_strategy = PARAMETERIZATIONS[parameterization]
        gene_param_name = param_strategy.gene_param_name  # "r" or "mu"
        guide_family_kwargs[gene_param_name] = LowRankGuide(rank=guide_rank)

    # Handle amortized inference for capture probability (VCP models only)
    if capture_amortization is not None:
        if model not in ("nbvcp", "zinbvcp"):
            raise ValueError(
                "capture_amortization is only valid for VCP models "
                f"(nbvcp, zinbvcp), not '{model}'"
            )
        effective = (
            AmortizationConfig(**capture_amortization)
            if isinstance(capture_amortization, dict)
            else capture_amortization
        )
        guide_family_kwargs["capture_amortization"] = effective
    elif amortize_capture:
        if model not in ("nbvcp", "zinbvcp"):
            raise ValueError(
                f"amortize_capture=True is only valid for VCP models "
                f"(nbvcp, zinbvcp), not '{model}'"
            )
        guide_family_kwargs["capture_amortization"] = AmortizationConfig(
            enabled=True,
            hidden_dims=capture_hidden_dims or [64, 32],
            activation=capture_activation,
            output_transform=capture_output_transform,
            output_clamp_min=capture_clamp_min,
            output_clamp_max=capture_clamp_max,
        )

    guide_families = (
        GuideFamilyConfig(**guide_family_kwargs)
        if guide_family_kwargs
        else None
    )

    # ==========================================================================
    # Build ModelConfig using builder
    # ==========================================================================
    builder = (
        ModelConfigBuilder()
        .for_model(model)
        .with_parameterization(parameterization)
        .with_inference(inference_method)
    )

    if unconstrained:
        builder.unconstrained()

    if n_components is not None:
        builder.as_mixture(n_components, mixture_params)

    if guide_families is not None:
        builder.with_guide_families(guide_families)

    if priors:
        builder.with_priors(**priors)

    return builder.build()


# ==============================================================================
# Export
# ==============================================================================

__all__ = ["build_config_from_preset"]

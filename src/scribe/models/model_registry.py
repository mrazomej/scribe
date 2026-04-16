"""Model registry for SCRIBE models.

This module provides functions for retrieving model and guide functions based on
the model type.

The primary API is ``get_model_and_guide()`` which uses the composable builder
system via preset factories. This provides:

- Per-parameter guide families (mean-field, low-rank, amortized, flow, VAE)
- Flexible configuration via GuideFamilyConfig
- Clean, composable architecture

Examples
--------
>>> from scribe.models.config import GuideFamilyConfig
>>> from scribe.models.components import LowRankGuide, AmortizedGuide
>>>
>>> # Simple usage (all mean-field)
>>> model, guide = get_model_and_guide("nbdm")
>>>
>>> # With per-parameter guide families
>>> model, guide = get_model_and_guide(
...     "nbvcp",
...     parameterization="linked",
...     guide_families=GuideFamilyConfig(
...         mu=LowRankGuide(rank=15),
...         p_capture=AmortizedGuide(amortizer=my_amortizer),
...     ),
... )
"""

import importlib
from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from .config import ModelConfig, GuideFamilyConfig


# ------------------------------------------------------------------------------
# Model log likelihood functions
# ------------------------------------------------------------------------------


def get_log_likelihood_fn(model_type: str) -> Callable:
    """
    Get the log likelihood function for a specified model type.

    The log likelihood functions are now located in the dedicated
    log_likelihood module, which is parameterization-independent.

    Parameters
    ----------
    model_type : str
        The type of model to retrieve the log likelihood function for.
        Examples: "nbdm", "zinb_mix".

    Returns
    -------
    Callable
        The log likelihood function for the specified model type.

    Raises
    ------
    ValueError
        If the log likelihood function cannot be found.
    """
    try:
        # Import the dedicated log_likelihood module
        log_likelihood_module = importlib.import_module(
            ".log_likelihood", "scribe.models"
        )
    except ImportError as e:
        raise ImportError(f"Could not import log_likelihood module: {e}")

    # Determine the function name based on convention
    if model_type.endswith("_mix"):
        base_type = model_type.replace("_mix", "")
        ll_name = f"{base_type}_mixture_log_likelihood"
    else:
        ll_name = f"{model_type}_log_likelihood"

    # Retrieve the function from the log_likelihood module
    ll_fn = getattr(log_likelihood_module, ll_name, None)
    if ll_fn is None:
        raise ValueError(
            f"Log likelihood function '{ll_name}' not found in "
            "'log_likelihood' module."
        )

    return ll_fn


# ------------------------------------------------------------------------------
# Main API: Composable builder system
# ------------------------------------------------------------------------------


def get_model_and_guide(
    model_config: "ModelConfig",
    unconstrained: Optional[bool] = None,
    guide_families: Optional["GuideFamilyConfig"] = None,
    n_genes: Optional[int] = None,
) -> Tuple[Callable, Callable, "ModelConfig"]:
    """Create model and guide functions using the unified factory.

    This is the primary API for getting model/guide functions. It uses the
    unified `create_model()` factory which provides a single entry point for
    all model types, eliminating code duplication.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration containing all model parameters:
            - base_model: Type of model ("nbdm", "zinb", "nbvcp", "zinbvcp")
            - parameterization: Parameterization scheme
            - unconstrained: Whether to use unconstrained parameterization
            - guide_families: Per-parameter guide family configuration
            - n_components: Number of mixture components (if mixture model)
            - mixture_params: List of mixture-specific parameter names
            - param_specs: Optional user-provided prior/guide overrides
    unconstrained : bool, optional
        Override the unconstrained setting from model_config. If None, uses
        model_config.unconstrained. Useful for special cases like predictive
        sampling where a constrained model is needed.
    guide_families : GuideFamilyConfig, optional
        Override the guide families from model_config. If None, uses
        model_config.guide_families. Useful for special cases where the guide
        is not needed (e.g., predictive sampling).

    Returns
    -------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function.
    model_config_for_results : ModelConfig
        Config with param_specs set (for use when constructing results).
        Use this when creating ScribeSVIResults so subsetting has metadata.

    Raises
    ------
    ValueError
        If model_config.base_model is not recognized.

    Examples
    --------
    >>> from scribe.models.config import GuideFamilyConfig
    >>> from scribe.models.components import LowRankGuide, AmortizedGuide
    >>> from scribe.inference.preset_builder import build_config_from_preset
    >>>
    >>> # Basic NBDM (all mean-field)
    >>> model_config = build_config_from_preset("nbdm")
    >>> model, guide, model_config_for_results = get_model_and_guide(model_config)
    >>>
    >>> # NBVCP with low-rank for mu and amortized p_capture
    >>> model_config = build_config_from_preset(
    ...     "nbvcp",
    ...     parameterization="linked",
    ...     guide_families=GuideFamilyConfig(
    ...         mu=LowRankGuide(rank=15),
    ...         p_capture=AmortizedGuide(amortizer=my_amortizer),
    ...     ),
    ... )
    >>> model, guide, model_config_for_results = get_model_and_guide(model_config)

    See Also
    --------
    scribe.models.presets.factory.create_model : Unified model factory.
    scribe.inference.preset_builder : Build ModelConfig from presets.
    GuideFamilyConfig : Per-parameter guide family configuration.
    """
    # Import the unified factory
    from .presets.factory import create_model

    # Handle overrides by creating a modified config if needed
    if unconstrained is not None or guide_families is not None:
        # Create a modified config with overrides
        updates = {}
        if unconstrained is not None:
            updates["unconstrained"] = unconstrained
        if guide_families is not None:
            updates["guide_families"] = guide_families

        # Use model_copy for immutable update
        effective_config = model_config.model_copy(update=updates)
    else:
        effective_config = model_config

    # Use the unified factory (n_genes required for VAE)
    model, guide, param_specs = create_model(
        effective_config, n_genes=n_genes
    )
    model_config_for_results = (
        effective_config.model_copy(update={"param_specs": param_specs})
        if param_specs
        else effective_config
    )
    return model, guide, model_config_for_results

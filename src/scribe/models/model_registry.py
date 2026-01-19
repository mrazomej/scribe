"""Model registry for SCRIBE models.

This module provides functions for retrieving model and guide functions based on
the model type.

The primary API is `get_model_and_guide()` which uses the composable builder
system via preset factories. This provides:

- Per-parameter guide families (mean-field, low-rank, amortized)
- Flexible configuration via GuideFamilyConfig
- Clean, composable architecture

For VAE inference and mixture models, `get_model_and_guide_legacy()` uses
the decorator-based registry system.

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
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from ..models.config import ModelConfig, GuideFamilyConfig

if TYPE_CHECKING:
    from .config import ModelConfig, GuideFamilyConfig

# ------------------------------------------------------------------------------
# Model registry - Decorator-based registration system
# ------------------------------------------------------------------------------

# Global registry dictionaries for models and guides
# Keys: (model_type, parameterization, inference_method, prior_type, unconstrained, guide_variant)
# Values: Callable (model or guide function)
_MODEL_REGISTRY = {}
_GUIDE_REGISTRY = {}

# List of supported parameterizations (and their corresponding modules)
SUPPORTED_PARAMETERIZATIONS = [
    "standard",
    "linked",
    "odds_ratio",
]

# List of supported inference methods
SUPPORTED_INFERENCE_METHODS = [
    "svi",
    "mcmc",
    "vae",
]

# List of supported prior types for VAE
SUPPORTED_PRIOR_TYPES = [
    "standard",
    "decoupled",
]

# List of supported guide variants
SUPPORTED_GUIDE_VARIANTS = [
    "mean_field",
    "low_rank",
]

# ------------------------------------------------------------------------------
# Registration decorator
# ------------------------------------------------------------------------------


def register(
    model_type: str,
    parameterization: str = "standard",
    inference_methods: Optional[List[str]] = None,
    prior_type: Optional[str] = None,
    unconstrained: bool = False,
    guide_variant: str = "mean_field",
):
    """
    Decorator to register model and guide functions in the global registry.

    This decorator allows models and guides to self-register, eliminating the
    need for complex if-else logic in get_model_and_guide(). Each function
    declares its own registration metadata through decorator parameters.

    Parameters
    ----------
    model_type : str
        Base model type identifier. Can be:
            - Single models: "nbdm", "zinb", "nbvcp", "zinbvcp"
            - Mixture models: "nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"

        Note: Mixture models use "_mix" suffix to distinguish from single
        models.

    parameterization : str, default="standard"
        Parameterization scheme for the model. Options:
            - "standard": Beta/LogNormal for p/r parameters
            - "linked": Beta/LogNormal for p/mu parameters
            - "odds_ratio": BetaPrime/LogNormal for phi/mu parameters

    inference_methods : Optional[List[str]], default=None
        List of inference methods this model/guide supports. Options:
            - ["svi", "mcmc"]: For standard probabilistic models (default)
            - ["vae"]: For VAE-based models with neural network components

        If None, defaults to ["svi", "mcmc"] for maximum compatibility.

    prior_type : Optional[str], default=None
        VAE prior architecture type (VAE inference only). Options:
            - "standard": Standard Normal prior (default for VAE)
            - "decoupled": Learned decoupled prior (dpVAE)
            - None: Not applicable (for non-VAE models)

    unconstrained : bool, default=False
        Whether this uses unconstrained parameterization. When True:
            - Parameters are sampled in unconstrained space (Real^n)
            - Transformations applied: sigmoid for probabilities, exp for
              positive values
            - Enables more stable MCMC sampling

    guide_variant : str, default="mean_field"
        Guide approximation family. Options:
            - "mean_field": Fully factorized variational family
            - "low_rank": Low-rank multivariate normal approximation

        Note: guide_rank parameter (the actual rank k) is specified at runtime.

    Returns
    -------
    Callable
        The decorated function, unchanged in behavior but registered in global
        dictionaries (_MODEL_REGISTRY or _GUIDE_REGISTRY).

    Raises
    ------
    ValueError
        If function name doesn't end with '_model' or '_guide'.

    Examples
    --------
    Register a standard single model for SVI/MCMC:

    >>> @register(model_type="nbdm", parameterization="standard")
    ... def nbdm_model(n_cells, n_genes, model_config, counts=None, batch_size=None):
    ...     # model implementation

    Register a mixture model:

    >>> @register(model_type="zinb_mix", parameterization="linked")
    ... def zinb_mixture_model(n_cells, n_genes, model_config, counts=None, batch_size=None):
    ...     # mixture model implementation

    Register a VAE model with decoupled prior:

    >>> @register(model_type="nbdm", parameterization="standard",
    ...           inference_methods=["vae"], prior_type="decoupled")
    ... def nbdm_dpvae_model(n_cells, n_genes, model_config, decoder, decoupled_prior, ...):
    ...     # VAE model implementation

    Register an unconstrained low-rank guide:

    >>> @register(model_type="nbdm", parameterization="standard",
    ...           unconstrained=True, guide_variant="low_rank")
    ... def nbdm_guide(n_cells, n_genes, model_config, counts=None, batch_size=None):
    ...     # low-rank guide implementation

    Notes
    -----
    - Registration happens at module import time
    - Functions are registered for all specified inference_methods
    - The decorator does not modify function behavior, only registers it
    - Mixture models are distinguished by "_mix" suffix in model_type
    """
    if inference_methods is None:
        inference_methods = ["svi", "mcmc"]

    def decorator(func):
        # Register for each supported inference method
        for inf_method in inference_methods:
            # Create registry key from all parameters
            key = (
                model_type,
                parameterization,
                inf_method,
                prior_type,
                unconstrained,
                guide_variant,
            )

            # Determine which registry to use based on function name
            if func.__name__.endswith("_model"):
                _MODEL_REGISTRY[key] = func
            elif func.__name__.endswith("_guide"):
                _GUIDE_REGISTRY[key] = func
            else:
                raise ValueError(
                    f"Function {func.__name__} must end with "
                    "'_model' or '_guide' "
                    f"to be registered. Got: {func.__name__}"
                )

        # Return function unchanged
        return func

    return decorator


# ------------------------------------------------------------------------------
# Legacy model and guide retrieval (decorator-based registry)
# ------------------------------------------------------------------------------


def get_model_and_guide_legacy(
    model_type: str,
    parameterization: str = "standard",
    inference_method: str = "svi",
    prior_type: Optional[str] = None,
    unconstrained: bool = False,
    guide_rank: Optional[int] = None,
) -> Tuple[Callable, Optional[Callable]]:
    """
    Retrieve the model and guide functions using the legacy registry system.

    This function is kept for VAE inference and mixture models that haven't
    been migrated to the new builder system. For standard models, use
    `get_model_and_guide()` instead.

    This function looks up the model and guide from the global registry
    populated by @register decorators. For VAE inference, the returned function
    is a factory that produces both model and guide; for SVI/MCMC, both model
    and guide functions are returned separately.

    Parameters
    ----------
    model_type : str
        The type of model to retrieve (e.g., "nbdm", "zinb_mix").
    parameterization : str, default="standard"
        The parameterization module to use (e.g., "standard", "linked",
        "odds_ratio").
    inference_method : str, default="svi"
        The inference method to use ("svi", "mcmc", or "vae").
    prior_type : str, optional
        The prior type to use for VAE inference ("standard" or "decoupled").
    unconstrained : bool, default=False
        Whether to use unconstrained parameterization variants.
    guide_rank : Optional[int], default=None
        If provided, specifies the rank `k` for a low-rank multivariate normal
        guide. If None, a mean-field guide is used.

    Returns
    -------
    Tuple[Callable, Optional[Callable]]
        A tuple containing the model function and the guide function (or None if
        not applicable).

    Raises
    ------
    ValueError
        If the parameterization, inference method, prior type, or required
        functions are not found in the registry.

    See Also
    --------
    get_model_and_guide : Preferred function using the new builder system.
    """
    # Validate inputs
    if parameterization not in SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(
            f"Unsupported parameterization: {parameterization}. "
            f"Supported parameterizations are: {SUPPORTED_PARAMETERIZATIONS}"
        )

    if inference_method not in SUPPORTED_INFERENCE_METHODS:
        raise ValueError(
            f"Unsupported inference method: {inference_method}. "
            f"Supported inference methods are: {SUPPORTED_INFERENCE_METHODS}"
        )

    # Determine guide variant based on guide_rank parameter
    guide_variant = "low_rank" if guide_rank is not None else "mean_field"

    # For VAE inference, wrap in factory
    if inference_method == "vae":
        # Set default prior type for VAE
        vae_prior_type = prior_type or "standard"

        # Create a factory function that returns the actual model and guide
        # The factory uses make_vae_model_and_guide which will dynamically
        # import the appropriate VAE module based on the parameters
        from .vae_core import make_vae_model_and_guide

        def vae_factory(n_genes, model_config):
            return make_vae_model_and_guide(
                model_type=model_type,
                n_genes=n_genes,
                model_config=model_config,
                parameterization=parameterization,
                prior_type=vae_prior_type,
                unconstrained=unconstrained,
            )

        return vae_factory, None

    else:
        # For SVI/MCMC, do simple dictionary lookup
        # prior_type is None for non-VAE methods

        # Models are always looked up with mean_field (they're shared between
        # variants)
        model_key = (
            model_type,
            parameterization,
            inference_method,
            None,
            unconstrained,
            "mean_field",  # Models are always mean_field
        )

        # Guides use the specified guide_variant
        guide_key = (
            model_type,
            parameterization,
            inference_method,
            None,
            unconstrained,
            guide_variant,
        )

        model_fn = _MODEL_REGISTRY.get(model_key)
        guide_fn = _GUIDE_REGISTRY.get(guide_key)

        if model_fn is None:
            raise ValueError(
                f"Model function not found in registry for key: {model_key}. "
                f"Available model keys: "
                f"{[k for k in _MODEL_REGISTRY.keys() if k[0] == model_type]}"
            )

        if guide_fn is None:
            raise ValueError(
                f"Guide function not found in registry for key: {guide_key}. "
                f"Available guide keys: "
                f"{[k for k in _GUIDE_REGISTRY.keys() if k[0] == model_type]}"
            )

        return model_fn, guide_fn


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
) -> Tuple[Callable, Callable]:
    """Create model and guide functions using the composable builder system.

    This is the new, recommended way to get model/guide functions. It uses
    the preset factories which provide more flexibility than the legacy
    decorator-based system.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration containing all model parameters:
            - base_model: Type of model ("nbdm", "zinb", "nbvcp", "zinbvcp")
            - parameterization: Parameterization scheme ("standard", "linked", "odds_ratio")
            - unconstrained: Whether to use unconstrained parameterization
            - guide_families: Per-parameter guide family configuration
            - n_components: Number of mixture components (if mixture model)
            - mixture_params: List of mixture-specific parameter names
            - param_specs: Parameter specifications with priors and guides
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

    Raises
    ------
    ValueError
        If model_config.base_model is not recognized.

    Examples
    --------
    >>> from scribe.models.config import ModelConfig, GuideFamilyConfig
    >>> from scribe.models.components import LowRankGuide, AmortizedGuide
    >>> from scribe.inference.preset_builder import build_config_from_preset
    >>>
    >>> # Basic NBDM (all mean-field)
    >>> model_config = build_config_from_preset("nbdm")
    >>> model, guide = get_model_and_guide(model_config)
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
    >>> model, guide = get_model_and_guide(model_config)

    See Also
    --------
    scribe.models.presets : Individual preset factory functions.
    scribe.inference.preset_builder : Build ModelConfig from presets.
    GuideFamilyConfig : Per-parameter guide family configuration.
    """
    # Import presets here to avoid circular imports
    from .presets import create_nbdm, create_zinb, create_nbvcp, create_zinbvcp

    # Extract all configuration from model_config
    model_type = model_config.base_model
    parameterization = (
        model_config.parameterization.value
        if model_config.parameterization
        else "standard"
    )

    # Use overrides if provided, otherwise use model_config values
    unconstrained_value = (
        unconstrained
        if unconstrained is not None
        else model_config.unconstrained
    )
    guide_families_value = (
        guide_families
        if guide_families is not None
        else model_config.guide_families
    )
    n_components = model_config.n_components
    mixture_params = model_config.mixture_params

    # Extract priors/guides from model_config
    priors = None
    guides = None
    if model_config.param_specs:
        priors = {
            spec.name: spec.prior
            for spec in model_config.param_specs
            if spec.prior is not None
        }
        guides = {
            spec.name: spec.guide
            for spec in model_config.param_specs
            if spec.guide is not None
        }
    # If param_specs is empty, try to extract from builder's _priors/_guides
    # This handles the case where ModelConfig was built but param_specs
    # weren't populated yet
    if not priors and hasattr(model_config, "_priors"):
        priors = getattr(model_config, "_priors", None)
    if not guides and hasattr(model_config, "_guides"):
        guides = getattr(model_config, "_guides", None)

    # Dispatch to appropriate preset
    if model_type == "nbdm":
        return create_nbdm(
            parameterization=parameterization,
            unconstrained=unconstrained_value,
            guide_families=guide_families_value,
            n_components=n_components,
            mixture_params=mixture_params,
            priors=priors,
            guides=guides,
        )
    elif model_type == "zinb":
        return create_zinb(
            parameterization=parameterization,
            unconstrained=unconstrained_value,
            guide_families=guide_families_value,
            n_components=n_components,
            mixture_params=mixture_params,
            priors=priors,
            guides=guides,
        )
    elif model_type == "nbvcp":
        return create_nbvcp(
            parameterization=parameterization,
            unconstrained=unconstrained_value,
            guide_families=guide_families_value,
            n_components=n_components,
            mixture_params=mixture_params,
            priors=priors,
            guides=guides,
        )
    elif model_type == "zinbvcp":
        return create_zinbvcp(
            parameterization=parameterization,
            unconstrained=unconstrained_value,
            guide_families=guide_families_value,
            n_components=n_components,
            mixture_params=mixture_params,
            priors=priors,
            guides=guides,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'nbdm', 'zinb', 'nbvcp', 'zinbvcp'."
        )

"""
This module provides a registry of model functions and their corresponding
guide functions. It allows for easy retrieval of model and guide functions
based on the model type.
"""

import importlib
from typing import Callable, Tuple, Optional

# ------------------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------------------

# List of supported parameterizations (and their corresponding modules)
SUPPORTED_PARAMETERIZATIONS = [
    "standard",
    "linked",
    "odds_ratio",
]

# List of supported inference methods
SUPPORTED_INFERENCE_METHODS = [
    "svi",
    "vae",
]

# List of supported prior types for VAE
SUPPORTED_PRIOR_TYPES = [
    "standard",
    "decoupled",
]

# Dictionary to cache imported model modules
_model_module_cache = {}

# ------------------------------------------------------------------------------


def get_model_and_guide(
    model_type: str,
    parameterization: str = "standard",
    inference_method: str = "svi",
    prior_type: Optional[str] = None,
    unconstrained: bool = False,
    guide_rank: Optional[int] = None,
) -> Tuple[Callable, Optional[Callable]]:
    """
    Retrieve the model and guide functions for a specified model type,
    parameterization, inference method, and (optionally) prior type.

    This function dynamically imports the appropriate parameterization module
    from `scribe.models` and locates the model and guide functions according to
    a naming convention. For VAE inference, the returned function is a factory
    that produces both model and guide; for SVI, both model and guide functions
    are returned separately.

    Parameters
    ----------
    model_type : str
        The type of model to retrieve (e.g., "nbdm", "zinb_mix").
    parameterization : str, default="standard"
        The parameterization module to use (e.g., "standard", "linked", "odds_ratio").
    inference_method : str, default="svi"
        The inference method to use ("svi" or "vae").
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
        functions are not found.
    """
    # Check if parameterization is supported
    if parameterization not in SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(
            f"Unsupported parameterization: {parameterization}. "
            f"Supported parameterizations are: {SUPPORTED_PARAMETERIZATIONS}"
        )

    # Check if inference method is supported
    if inference_method not in SUPPORTED_INFERENCE_METHODS:
        raise ValueError(
            f"Unsupported inference method: {inference_method}. "
            f"Supported inference methods are: {SUPPORTED_INFERENCE_METHODS}"
        )

    # For VAE inference, use the centralized factory
    if inference_method == "vae":
        from .vae_core import make_vae_model_and_guide

        # Create a factory function that calls the centralized factory
        def vae_factory(n_genes, model_config):
            return make_vae_model_and_guide(
                model_type=model_type,
                n_genes=n_genes,
                model_config=model_config,
                parameterization=parameterization,
                prior_type=prior_type or "standard",
                unconstrained=unconstrained,
            )

        return vae_factory, None
    else:
        # For non-VAE models, use the standard registry Determine the module
        # name based on parameterization and unconstrained flag
        if unconstrained:
            model_module_name = f"{parameterization}_unconstrained"
        else:
            model_module_name = f"{parameterization}"

        if guide_rank is not None:
            guide_module_name = f"{model_module_name}_low_rank"
        else:
            guide_module_name = model_module_name

        try:
            model_module = importlib.import_module(
                f".{model_module_name}", "scribe.models"
            )
        except ImportError as e:
            raise ValueError(
                f"Could not import parameterization module '{model_module_name}': {e}"
            )

        try:
            guide_module = importlib.import_module(
                f".{guide_module_name}", "scribe.models"
            )
        except ImportError as e:
            raise ValueError(
                f"Could not import parameterization module '{guide_module_name}': {e}"
            )

        # Determine the function names based on convention
        if model_type.endswith("_mix"):
            base_type = model_type.replace("_mix", "")
            model_name = f"{base_type}_mixture_model"
            guide_name = f"{base_type}_mixture_guide"
        else:
            model_name = f"{model_type}_model"
            guide_name = f"{model_type}_guide"

        # Retrieve the functions from the module
        model_fn = getattr(model_module, model_name, None)
        if model_fn is None:
            raise ValueError(
                f"Model function '{model_name}' "
                f"not found in module '{model_module_name}'"
            )

        # Guide functions exist for all parameterizations including
        # unconstrained
        guide_fn = getattr(guide_module, guide_name, None)
        if guide_fn is None:
            raise ValueError(
                f"Guide function '{guide_name}' "
                f"not found in module '{guide_module_name}'"
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
            f"Log likelihood function '{ll_name}' not found in 'log_likelihood' module."
        )

    return ll_fn

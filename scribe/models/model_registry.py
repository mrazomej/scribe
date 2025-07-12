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
    "unconstrained",
]

# Dictionary to cache imported model modules
_model_module_cache = {}


def get_model_and_guide(
    model_type: str, parameterization: str = "standard"
) -> Tuple[Callable, Optional[Callable]]:
    """
    Get model and guide functions for a specified model type and
    parameterization.

    This function dynamically loads the required parameterization module from
    `scribe.models` and retrieves the corresponding model and guide functions
    based on a naming convention (e.g., `nbdm_model`, `nbdm_guide`).

    Parameters
    ----------
    model_type : str
        The type of model to retrieve functions for. Examples: "nbdm",
        "zinb_mix".
    parameterization : str, default="standard"
        The parameterization module to load from (e.g., "standard",
        "unconstrained").

    Returns
    -------
    Tuple[Callable, Optional[Callable]]
        A tuple containing (model_function, guide_function). The guide function
        is None for unconstrained parameterizations.

    Raises
    ------
    ValueError
        If the parameterization module or the model/guide functions cannot be
        found.
    """
    # Check if parameterization is supported
    if parameterization not in SUPPORTED_PARAMETERIZATIONS:
        raise ValueError(
            f"Unsupported parameterization: {parameterization}. "
            f"Supported parameterizations are: {SUPPORTED_PARAMETERIZATIONS}"
        )

    # Dynamically import the parameterization module (e.g.,
    # scribe.models.standard)
    try:
        module = importlib.import_module(
            f".{parameterization}", "scribe.models"
        )
    except ImportError as e:
        raise ValueError(
            f"Could not import parameterization module '{parameterization}': {e}"
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
    model_fn = getattr(module, model_name, None)
    if model_fn is None:
        raise ValueError(
            f"Model function '{model_name}' not found in module '{parameterization}'"
        )

    # Guide is not used for unconstrained parameterization
    guide_fn = None
    if parameterization != "unconstrained":
        guide_fn = getattr(module, guide_name, None)
        if guide_fn is None:
            raise ValueError(
                f"Guide function '{guide_name}' not found in module '{parameterization}'"
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
        raise ImportError(
            f"Could not import log_likelihood module: {e}"
        )

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

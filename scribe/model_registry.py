"""
This module provides a registry of model functions and their corresponding
guide functions. It allows for easy retrieval of model and guide functions
based on the model type.
"""

# Import necessary modules
from typing import Callable, Tuple, Dict

# ------------------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------------------

def get_model_and_guide(model_type: str) -> Tuple[Callable, Callable]:
    """
    Get model and guide functions for a specified model type.

    This function returns the appropriate model and guide functions based on the
    requested model type. Currently supports:
        - "nbdm": Negative Binomial-Dirichlet Multinomial model
        - "zinb": Zero-Inflated Negative Binomial model
        - "nbvcp": Negative Binomial with variable mRNA capture probability
        - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
          probability
        - "nbdm_mix": Negative Binomial-Dirichlet Multinomial Mixture Model
        - "zinb_mix": Zero-Inflated Negative Binomial Mixture Model
        - "nbvcp_mix": Negative Binomial with variable mRNA capture probability
          Mixture Model
        - "zinbvcp_mix": Zero-Inflated Negative Binomial with variable capture
          probability Mixture Model

    Parameters
    ----------
    model_type : str
        The type of model to retrieve functions for. Must be one of ["nbdm",
        "zinb", "nbvcp", "zinbvcp", "nbdm_mix", "zinb_mix", "nbvcp_mix",
        "zinbvcp_mix"].

    Returns
    -------
    Tuple[Callable, Callable]
        A tuple containing (model_function, guide_function) for the requested
        model type.

    Raises
    ------
    ValueError
        If an unsupported model type is provided.
    """
    # Handle Negative Binomial-Dirichlet Multinomial model
    if model_type == "nbdm":
        # Import model and guide functions locally to avoid circular imports
        from .models import nbdm_model, nbdm_guide
        return nbdm_model, nbdm_guide
    
    # Handle Zero-Inflated Negative Binomial model
    elif model_type == "zinb":
        # Import model and guide functions locally to avoid circular imports
        from .models import zinb_model, zinb_guide
        return zinb_model, zinb_guide
    
    # Handle Negative Binomial with variable mRNA capture probability model
    elif model_type == "nbvcp":
        # Import model and guide functions locally to avoid circular imports
        from .models import nbvcp_model, nbvcp_guide
        return nbvcp_model, nbvcp_guide
    
    # Handle Zero-Inflated Negative Binomial with variable capture probability
    elif model_type == "zinbvcp":
        # Import model and guide functions locally to avoid circular imports
        from .models import zinbvcp_model, zinbvcp_guide
        return zinbvcp_model, zinbvcp_guide
    
    # Handle Negative Binomial-Dirichlet Multinomial Mixture Model
    elif model_type == "nbdm_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import nbdm_mixture_model, nbdm_mixture_guide
        return nbdm_mixture_model, nbdm_mixture_guide

    # Handle Zero-Inflated Negative Binomial Mixture Model
    elif model_type == "zinb_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import zinb_mixture_model, zinb_mixture_guide
        return zinb_mixture_model, zinb_mixture_guide
    
    # Handle Negative Binomial-Variable Capture Probability Mixture Model
    elif model_type == "nbvcp_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import nbvcp_mixture_model, nbvcp_mixture_guide
        return nbvcp_mixture_model, nbvcp_mixture_guide
    
    # Handle Zero-Inflated Negative Binomial-Variable Capture Probability
    # Mixture Model
    elif model_type == "zinbvcp_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import zinbvcp_mixture_model, zinbvcp_mixture_guide
        return zinbvcp_mixture_model, zinbvcp_mixture_guide
    
    # Raise error for unsupported model types
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ------------------------------------------------------------------------------
# Model log likelihood functions
# ------------------------------------------------------------------------------

def get_log_likelihood_fn(model_type: str) -> Callable:
    """Get the log likelihood function for a specified model type."""
    # Standard models
    if model_type == "nbdm":
        from .models import nbdm_log_likelihood
        return nbdm_log_likelihood
    elif model_type == "zinb":
        from .models import zinb_log_likelihood
        return zinb_log_likelihood
    elif model_type == "nbvcp":
        from .models import nbvcp_log_likelihood
        return nbvcp_log_likelihood
    elif model_type == "zinbvcp":
        from .models import zinbvcp_log_likelihood
        return zinbvcp_log_likelihood
    
    # Mixture models
    elif model_type == "nbdm_mix":
        from .models_mix import nbdm_mixture_log_likelihood
        return nbdm_mixture_log_likelihood
    elif model_type == "zinb_mix":
        from .models_mix import zinb_mixture_log_likelihood
        return zinb_mixture_log_likelihood
    elif model_type == "nbvcp_mix":
        from .models_mix import nbvcp_mixture_log_likelihood
        return nbvcp_mixture_log_likelihood
    elif model_type == "zinbvcp_mix":
        from .models_mix import zinbvcp_mixture_log_likelihood
        return zinbvcp_mixture_log_likelihood
    elif model_type == "nbdm_log_mix":
        from .models_mix import nbdm_mixture_log_likelihood
        return nbdm_mixture_log_likelihood
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ------------------------------------------------------------------------------
# Model default priors
# ------------------------------------------------------------------------------

def get_default_priors(model_type: str) -> Dict[str, Tuple[float, float]]:
    """
    Get default prior parameters for a specified model type.

    This function returns a dictionary of default prior parameters based on the
    requested model type. Currently supports:
        - "nbdm": Negative Binomial-Dirichlet Multinomial model
        - "zinb": Zero-Inflated Negative Binomial model
        - "nbvcp": Negative Binomial with variable mRNA capture probability
          model
        - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
          probability model

    Parameters
    ----------
    model_type : str
        The type of model to get default priors for. Must be one of ["nbdm",
        "zinb", "nbvcp", "zinbvcp"]. For custom models, returns an empty
        dictionary.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A dictionary mapping parameter names to prior parameter tuples: - For
        "nbdm":
            - 'p_prior': (alpha, beta) for Beta prior on p parameter
            - 'r_prior': (shape, rate) for Gamma prior on r parameter
        - For "zinb":
            - 'p_prior': (alpha, beta) for Beta prior on p parameter  
            - 'r_prior': (shape, rate) for Gamma prior on r parameter
            - 'gate_prior': (alpha, beta) for Beta prior on gate parameter
        - For "nbvcp":
            - 'p_prior': (alpha, beta) for Beta prior on base success
              probability p
            - 'r_prior': (shape, rate) for Gamma prior on dispersion parameters
            - 'p_capture_prior': (alpha, beta) for Beta prior on capture
              probabilities
        - For "zinbvcp":
            - 'p_prior': (alpha, beta) for Beta prior on base success
              probability p
            - 'r_prior': (shape, rate) for Gamma prior on dispersion parameters
            - 'p_capture_prior': (alpha, beta) for Beta prior on capture
              probabilities
            - 'gate_prior': (alpha, beta) for Beta prior on dropout
              probabilities
        - For custom models: empty dictionary
    """
    if model_type == "nbdm":
        prior_params = {
            'p_prior': (0, 1),
            'r_prior': (1, 0.1)
        }
    elif model_type == "zinb":
        prior_params = {
            'p_prior': (0, 1),
            'r_prior': (1, 0.1),
            'gate_prior': (0, 1)
        }
    elif model_type == "nbvcp":
        prior_params = {
            'p_prior': (0, 1),
            'r_prior': (1, 0.1),
            'p_capture_prior': (0, 1)
        }
    elif model_type == "zinbvcp":
        prior_params = {
            'p_prior': (0, 1),
            'r_prior': (1, 0.1),
            'p_capture_prior': (0, 1),
            'gate_prior': (0, 1)
        }
    elif model_type == "nbdm_mix":
        prior_params = {
            'mixing_prior': (0, 1),
            'p_prior': (0, 1),
            'r_prior': (1, 0.1)
        }
    elif model_type == "zinb_mix":
        prior_params = {
            'mixing_prior': (0, 1),
            'p_prior': (0, 1),
            'r_prior': (1, 0.1),
            'gate_prior': (0, 1)
        }
    elif model_type == "nbvcp_mix":
        prior_params = {
            'mixing_prior': (0, 1),
            'p_prior': (0, 1),
            'r_prior': (1, 0.1),
            'p_capture_prior': (0, 1)
        }
    elif model_type == "zinbvcp_mix":
        prior_params = {
            'mixing_prior': (0, 1),
            'p_prior': (0, 1),
            'r_prior': (1, 0.1),
            'p_capture_prior': (0, 1),
            'gate_prior': (0, 1)
        }
    elif model_type == "nbdm_log_mix":
        prior_params = {
            'mixing_prior': (0, 1),
            'p_prior': (0, 1),
            'r_prior': (0, 1)
        }
    else:
        prior_params = {}  # Empty dict for custom models if none provided

    return prior_params
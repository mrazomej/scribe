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

def get_model_and_guide(
    model_type: str, parameterization: str = "mean_field"
) -> Tuple[Callable, Callable]:
    """
    Get model and guide functions for a specified model type and guide type.

    This function returns the appropriate model and guide functions based on the
    requested model type and guide parameterization. Currently supports:
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
    guide_type : str, default="mean_field"
        The type of variational guide to use:
            - "mean_field": Independent parameters (original)
            - "mean_variance": Correlated r-p parameters via mean-variance
              parameterization
            - "beta_prime": Correlated r-p parameters via beta-prime
              parameterization

    Returns
    -------
    Tuple[Callable, Callable]
        A tuple containing (model_function, guide_function) for the requested
        model type and guide type.

    Raises
    ------
    ValueError
        If an unsupported model type or guide type is provided.
    """
    if parameterization is None:
        parameterization = "mean_field"

    # Validate guide type
    valid_guide_types = ["mean_field", "mean_variance", "beta_prime"]
    if parameterization not in valid_guide_types:
        raise ValueError(f"Unknown guide type: {parameterization}. Must be one of {valid_guide_types}")
    
    # Handle Negative Binomial-Dirichlet Multinomial model
    if model_type == "nbdm":
        # Import model function
        from .models import nbdm_model
        
        # Select guide based on guide_type
        if parameterization == "mean_field":
            from .models import nbdm_guide
            return nbdm_model, nbdm_guide
        elif parameterization == "mean_variance":
            from .models import nbdm_guide_mean_variance
            return nbdm_model, nbdm_guide_mean_variance
        elif parameterization == "beta_prime":
            from .models import nbdm_guide_beta_prime
            return nbdm_model, nbdm_guide_beta_prime
    
    # Handle Zero-Inflated Negative Binomial model
    elif model_type == "zinb":
        # Import model function
        from .models import zinb_model
        
        # Select guide based on guide_type
        if parameterization == "mean_field":
            from .models import zinb_guide
            return zinb_model, zinb_guide
        elif parameterization == "mean_variance":
            from .models import zinb_guide_mean_variance
            return zinb_model, zinb_guide_mean_variance
        elif parameterization == "beta_prime":
            from .models import zinb_guide_beta_prime
            return zinb_model, zinb_guide_beta_prime
    
    # Handle Negative Binomial with variable mRNA capture probability model
    elif model_type == "nbvcp":
        # Import model and guide functions locally to avoid circular imports
        from .models import nbvcp_model, nbvcp_guide
        if parameterization != "mean_field":
            raise ValueError(f"Guide type '{parameterization}' not yet supported for model '{model_type}'")
        return nbvcp_model, nbvcp_guide
    
    # Handle Zero-Inflated Negative Binomial with variable capture probability
    elif model_type == "zinbvcp":
        # Import model and guide functions locally to avoid circular imports
        from .models import zinbvcp_model, zinbvcp_guide
        if parameterization != "mean_field":
            raise ValueError(f"Guide type '{parameterization}' not yet supported for model '{model_type}'")
        return zinbvcp_model, zinbvcp_guide
    
    # Handle Negative Binomial-Dirichlet Multinomial Mixture Model
    elif model_type == "nbdm_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import nbdm_mixture_model, nbdm_mixture_guide
        if parameterization != "mean_field":
            raise ValueError(f"Parameterization '{parameterization}' not yet supported for model '{model_type}'")
        return nbdm_mixture_model, nbdm_mixture_guide

    # Handle Zero-Inflated Negative Binomial Mixture Model
    elif model_type == "zinb_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import zinb_mixture_model, zinb_mixture_guide
        if parameterization != "mean_field":
            raise ValueError(f"Guide type '{parameterization}' not yet supported for model '{model_type}'")
        return zinb_mixture_model, zinb_mixture_guide
    
    # Handle Negative Binomial-Variable Capture Probability Mixture Model
    elif model_type == "nbvcp_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import nbvcp_mixture_model, nbvcp_mixture_guide
        if parameterization != "mean_field":
            raise ValueError(f"Guide type '{parameterization}' not yet supported for model '{model_type}'")
        return nbvcp_mixture_model, nbvcp_mixture_guide
    
    # Handle Zero-Inflated Negative Binomial-Variable Capture Probability
    # Mixture Model
    elif model_type == "zinbvcp_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import zinbvcp_mixture_model, zinbvcp_mixture_guide
        if parameterization != "mean_field":
            raise ValueError(f"Guide type '{parameterization}' not yet supported for model '{model_type}'")
        return zinbvcp_mixture_model, zinbvcp_mixture_guide
    
    # Raise error for unsupported model types
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ------------------------------------------------------------------------------
# Unconstrained model registry
# ------------------------------------------------------------------------------

def get_unconstrained_model(model_type: str) -> Callable:
    """
    Get the unconstrained version of the specified model type.

    Parameters
    ----------
    model_type : str
        Type of model to use. Must be one of:
            - "nbdm": Negative Binomial model
            - "zinb": Zero-Inflated Negative Binomial model
            - "nbvcp": Negative Binomial with variable capture probability
            - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
              probability
            - Mixture variants with "_mix" suffix (e.g. "nbdm_mix")

    Returns
    -------
    Callable
        The unconstrained version of the specified model function.

    Raises
    ------
    ValueError
        If an unsupported model type is specified.
    """
    # Handle Negative Binomial-Dirichlet Multinomial model
    if model_type == "nbdm":
        from .models_unconstrained import nbdm_model_unconstrained
        return nbdm_model_unconstrained
    
    # Handle Zero-Inflated Negative Binomial model
    elif model_type == "zinb":
        from .models_unconstrained import zinb_model_unconstrained
        return zinb_model_unconstrained
    
    # Handle Negative Binomial with variable capture probability model
    elif model_type == "nbvcp":
        from .models_unconstrained import nbvcp_model_unconstrained
        return nbvcp_model_unconstrained
    
    # Handle Zero-Inflated Negative Binomial with variable capture probability
    elif model_type == "zinbvcp":
        from .models_unconstrained import zinbvcp_model_unconstrained
        return zinbvcp_model_unconstrained
    
    # Handle Negative Binomial-Dirichlet Multinomial Mixture Model
    elif model_type == "nbdm_mix":
        from .models_unconstrained_mix import nbdm_mixture_model_unconstrained
        return nbdm_mixture_model_unconstrained
    
    # Handle Zero-Inflated Negative Binomial Mixture Model
    elif model_type == "zinb_mix":
        from .models_unconstrained_mix import zinb_mixture_model_unconstrained
        return zinb_mixture_model_unconstrained
    
    # Handle Negative Binomial-Variable Capture Probability Mixture Model
    elif model_type == "nbvcp_mix":
        from .models_unconstrained_mix import nbvcp_mixture_model_unconstrained
        return nbvcp_mixture_model_unconstrained
    
    # Handle Zero-Inflated Negative Binomial-Variable Capture Probability Mixture Model
    elif model_type == "zinbvcp_mix":
        from .models_unconstrained_mix import zinbvcp_mixture_model_unconstrained
        return zinbvcp_mixture_model_unconstrained
    
    # Raise error for unsupported model types
    else:
        raise ValueError(f"Unknown model type for unconstrained parameterization: {model_type}")

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

# ------------------------------------------------------------------------------
# General model function getter
# ------------------------------------------------------------------------------

def get_model_fn(model_type: str, unconstrained: bool = True) -> Callable:
    """
    Get the model function for a specified model type and parameterization.
    
    Parameters
    ----------
    model_type : str
        Type of model to use
    unconstrained : bool, default=True
        Whether to use unconstrained parameterization
        
    Returns
    -------
    Callable
        The model function
    """
    if unconstrained:
        return get_unconstrained_model(model_type)
    else:
        model_fn, _ = get_model_and_guide(model_type)
        return model_fn
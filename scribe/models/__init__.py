"""
SCRIBE models package.

This package contains all model definitions, configurations, and registry functions.
"""

# Model configuration
from .model_config import ModelConfig

# Model registry
from .model_registry import (
    get_model_and_guide,
    get_log_likelihood_fn,
    get_default_priors,
)

# Model functions - import the main ones that are commonly used
from .models import (
    nbdm_model,
    nbdm_guide,
    zinb_model,
    zinb_guide,
    nbvcp_model,
    nbvcp_guide,
    zinbvcp_model,
    zinbvcp_guide,
)

from .models_mix import (
    nbdm_mixture_model,
    nbdm_mixture_guide,
    zinb_mixture_model,
    zinb_mixture_guide,
    nbvcp_mixture_model,
    nbvcp_mixture_guide,
    zinbvcp_mixture_model,
    zinbvcp_mixture_guide,
)

__all__ = [
    # Configuration
    "ModelConfig",
    # Registry functions
    "get_model_and_guide",
    "get_log_likelihood_fn",
    "get_default_priors",
    # Individual model functions
    "nbdm_model",
    "nbdm_guide",
    "zinb_model",
    "zinb_guide",
    "nbvcp_model",
    "nbvcp_guide",
    "zinbvcp_model",
    "zinbvcp_guide",
    # Mixture model functions
    "nbdm_mixture_model",
    "nbdm_mixture_guide",
    "zinb_mixture_model",
    "zinb_mixture_guide",
    "nbvcp_mixture_model",
    "nbvcp_mixture_guide",
    "zinbvcp_mixture_model",
    "zinbvcp_mixture_guide",
]

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
)

# Parameterization-specific model modules
from . import linked, odds_ratio, standard, vae_standard
from . import (
    linked_unconstrained,
    odds_ratio_unconstrained,
    standard_unconstrained,
)
from . import (
    vae_linked_unconstrained,
    vae_odds_ratio_unconstrained,
    vae_standard_unconstrained,
)

__all__ = [
    # Configuration
    "ModelConfig",
    # Registry functions
    "get_model_and_guide",
    "get_log_likelihood_fn",
    # Model modules
    "standard",
    "vae_standard",
    "linked",
    "odds_ratio",
    "linked_unconstrained",
    "odds_ratio_unconstrained",
    "standard_unconstrained",
    "vae_linked_unconstrained",
    "vae_odds_ratio_unconstrained",
    "vae_standard_unconstrained",
]

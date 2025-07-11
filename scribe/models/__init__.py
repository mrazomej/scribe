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
from . import standard

__all__ = [
    # Configuration
    "ModelConfig",
    # Registry functions
    "get_model_and_guide",
    "get_log_likelihood_fn",
    # Model modules
    "standard",
]

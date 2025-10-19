"""
SCRIBE models package.

This package contains all model definitions, configurations, and registry functions.
"""

# Model registry
from .model_registry import (
    get_model_and_guide,
    get_log_likelihood_fn,
)

# Export new config system from config subdirectory
from .config import (
    ModelConfigBuilder,
    ModelConfig,
    ConstrainedModelConfig,
    UnconstrainedModelConfig,
    PriorConfig,
    GuideConfig,
    VAEConfig,
    ModelType,
    Parameterization,
    InferenceMethod,
    VAEPriorType,
    # Parameter mapping utilities
    get_active_parameters,
    get_required_parameters,
    get_parameterization_mapping,
    validate_parameter_consistency,
    get_parameterization_summary,
)

# Import all model modules to register them via decorators
# Standard parameterizations (constrained)
from . import standard, linked, odds_ratio

# Unconstrained parameterizations
from . import (
    standard_unconstrained,
    linked_unconstrained,
    odds_ratio_unconstrained,
)

# Low-rank guide variants
from . import standard_low_rank, linked_low_rank, odds_ratio_low_rank

# Low-rank unconstrained variants
from . import (
    standard_low_rank_unconstrained,
    linked_low_rank_unconstrained,
    odds_ratio_low_rank_unconstrained,
)

# VAE parameterizations (constrained)
from . import vae_standard, vae_linked, vae_odds_ratio

# VAE unconstrained parameterizations
from . import (
    vae_standard_unconstrained,
    vae_linked_unconstrained,
    vae_odds_ratio_unconstrained,
)

__all__ = [
    # Registry functions
    "get_model_and_guide",
    "get_log_likelihood_fn",
    # New config system
    "ModelConfigBuilder",
    "ModelConfig",
    "ConstrainedModelConfig",
    "UnconstrainedModelConfig",
    "PriorConfig",
    "GuideConfig",
    "VAEConfig",
    "ModelType",
    "Parameterization",
    "InferenceMethod",
    "VAEPriorType",
    # Parameter mapping utilities
    "get_active_parameters",
    "get_required_parameters",
    "get_parameterization_mapping",
    "validate_parameter_consistency",
    "get_parameterization_summary",
    # Model modules (standard parameterizations)
    "standard",
    "linked",
    "odds_ratio",
    # Unconstrained variants
    "standard_unconstrained",
    "linked_unconstrained",
    "odds_ratio_unconstrained",
    # Low-rank guide variants
    "standard_low_rank",
    "linked_low_rank",
    "odds_ratio_low_rank",
    # Low-rank unconstrained variants
    "standard_low_rank_unconstrained",
    "linked_low_rank_unconstrained",
    "odds_ratio_low_rank_unconstrained",
    # VAE parameterizations
    "vae_standard",
    "vae_linked",
    "vae_odds_ratio",
    # VAE unconstrained variants
    "vae_standard_unconstrained",
    "vae_linked_unconstrained",
    "vae_odds_ratio_unconstrained",
]

"""SCRIBE models package.

This package contains all model definitions, configurations, and registry functions.

Use `get_model_and_guide()` to create models with flexible per-parameter guide
families via `GuideFamilyConfig`:

- Mean-field guides (default)
- Low-rank multivariate normal guides
- Amortized neural network guides

Examples
--------
>>> from scribe.models import get_model_and_guide
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
>>>
>>> # Direct preset usage
>>> from scribe.models.presets import create_nbvcp
>>> model, guide = create_nbvcp(
...     guide_families=GuideFamilyConfig(p_capture=AmortizedGuide(amortizer=net))
... )
"""

# Model registry
from .model_registry import (
    get_model_and_guide,
    get_model_and_guide_legacy,
    get_log_likelihood_fn,
)

# Export config system from config subdirectory
# Note: PriorConfig and GuideConfig have been deprecated in favor of ParamSpec
# objects with prior/guide tuples directly on the spec. See
# scribe.models.builders.parameter_specs for the new approach.
from .config import (
    ModelConfigBuilder,
    ModelConfig,
    GuideFamilyConfig,
    VAEConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
    ModelType,
    Parameterization,
    InferenceMethod,
    VAEPriorType,
    VAEMaskType,
    VAEActivation,
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
    "get_model_and_guide_legacy",
    "get_log_likelihood_fn",
    # Config system
    "ModelConfigBuilder",
    "ModelConfig",
    "PriorConfig",
    "GuideConfig",
    "GuideFamilyConfig",
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

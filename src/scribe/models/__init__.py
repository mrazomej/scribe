"""SCRIBE models package.

This package contains all model definitions, configurations, and registry functions.

Use `get_model_and_guide()` or the unified factory `create_model_from_params()`
to create models with flexible per-parameter guide families via
`GuideFamilyConfig`:

- Mean-field guides (default)
- Low-rank multivariate normal guides
- Amortized neural network guides

Examples
--------
>>> from scribe.models import get_model_and_guide
>>> from scribe.inference.preset_builder import build_config_from_preset
>>> from scribe.models.config import GuideFamilyConfig, ModelConfigBuilder
>>> from scribe.models.components import LowRankGuide, AmortizedGuide
>>>
>>> # Simple usage (all mean-field)
>>> config = build_config_from_preset("nbdm")
>>> model, guide = get_model_and_guide(config)
>>>
>>> # With per-parameter guide families
>>> config = build_config_from_preset(
...     model="nbvcp",
...     parameterization="linked",
...     guide_rank=15,  # Creates LowRankGuide for mu
... )
>>> # Or use ModelConfigBuilder for more control
>>> config = (
...     ModelConfigBuilder()
...     .for_model("nbvcp")
...     .with_parameterization("linked")
...     .with_inference("svi")
...     .with_guide_families(
...         GuideFamilyConfig(
...             mu=LowRankGuide(rank=15),
...             p_capture=AmortizedGuide(amortizer=my_amortizer),
...         )
...     )
...     .build()
... )
>>> model, guide = get_model_and_guide(config)
>>>
>>> # Using the unified factory directly
>>> from scribe.models.presets import create_model_from_params
>>> model, guide = create_model_from_params(
...     model="nbvcp",
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

# Note: Legacy model modules (standard.py, linked.py, etc.) are kept in the
# legacy/ subdirectory for reference but are not imported. The new system uses
# the composable builder pattern via presets/factory.py and builders/.

__all__ = [
    # Registry functions
    "get_model_and_guide",
    "get_model_and_guide_legacy",
    "get_log_likelihood_fn",
    # Config system
    "ModelConfigBuilder",
    "ModelConfig",
    "GuideFamilyConfig",
    "VAEConfig",
    "SVIConfig",
    "MCMCConfig",
    "DataConfig",
    "ModelType",
    "Parameterization",
    "InferenceMethod",
    "VAEPriorType",
    "VAEMaskType",
    "VAEActivation",
    # Parameter mapping utilities
    "get_active_parameters",
    "get_required_parameters",
    "get_parameterization_mapping",
    "validate_parameter_consistency",
    "get_parameterization_summary",
]

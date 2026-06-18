"""
Modern configuration system for SCRIBE models.

Uses Pydantic for validation, builder pattern for construction, and
enums for type safety. All configs are immutable by default.
"""

from .enums import (
    ModelType,
    Parameterization,
    InferenceMethod,
    HierarchicalPriorType,
    OverdispersionType,
    VAEPriorType,
    VAEMaskType,
    VAEActivation,
)
from .groups import (
    AmortizationConfig,
    EarlyStoppingConfig,
    GuideFamilyConfig,
    KLAnnealingConfig,
    LaplaceConfig,
    PriorOverrides,
    VAEConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
    InferenceConfig,
)
from .base import ModelConfig
from .builder import ModelConfigBuilder
from .grouping import (
    GroupLevel,
    Factor,
    GroupingSpec,
    normalize_grouping,
    resolve_dataset_prior_dict,
)
from .parameter_mapping import (
    FREEZE_KEY_ALIASES,
    PRIOR_KEY_ALIASES,
    get_active_parameters,
    get_required_parameters,
    get_parameterization_mapping,
    normalize_freeze_keys,
    validate_parameter_consistency,
    get_parameterization_summary,
)

__all__ = [
    # Builder (primary interface)
    "ModelConfigBuilder",
    # Config types
    "ModelConfig",
    # Multi-factor grouping
    "GroupLevel",
    "Factor",
    "GroupingSpec",
    "normalize_grouping",
    "resolve_dataset_prior_dict",
    # Parameter groups
    "AmortizationConfig",
    "EarlyStoppingConfig",
    "GuideFamilyConfig",
    "KLAnnealingConfig",
    "LaplaceConfig",
    "PriorOverrides",
    "VAEConfig",
    "SVIConfig",
    "OptimizerConfig",
    "MCMCConfig",
    "DataConfig",
    "InferenceConfig",
    # Enums
    "ModelType",
    "Parameterization",
    "InferenceMethod",
    "HierarchicalPriorType",
    "OverdispersionType",
    "VAEPriorType",
    "VAEMaskType",
    "VAEActivation",
    # Parameter mapping utilities
    "FREEZE_KEY_ALIASES",
    "PRIOR_KEY_ALIASES",
    "get_active_parameters",
    "get_required_parameters",
    "get_parameterization_mapping",
    "normalize_freeze_keys",
    "validate_parameter_consistency",
    "get_parameterization_summary",
]

# Backward-compatible alias for the nested optimizer config model.
OptimizerConfig = SVIConfig.OptimizerConfig

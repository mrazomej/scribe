"""
Modern configuration system for SCRIBE models.

Uses Pydantic for validation, builder pattern for construction, and
enums for type safety. All configs are immutable by default.
"""

from .enums import (
    ModelType,
    Parameterization,
    InferenceMethod,
    VAEPriorType,
    VAEMaskType,
    VAEActivation,
)
from .groups import (
    GuideFamilyConfig,
    VAEConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
    InferenceConfig,
)
from .base import ModelConfig
from .builder import ModelConfigBuilder
from .parameter_mapping import (
    get_active_parameters,
    get_required_parameters,
    get_parameterization_mapping,
    validate_parameter_consistency,
    get_parameterization_summary,
)

__all__ = [
    # Builder (primary interface)
    "ModelConfigBuilder",
    # Config types
    "ModelConfig",
    # Parameter groups
    "GuideFamilyConfig",
    "VAEConfig",
    "SVIConfig",
    "MCMCConfig",
    "DataConfig",
    "InferenceConfig",
    # Enums
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

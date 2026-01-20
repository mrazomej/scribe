"""Model factory for creating SCRIBE models.

This module provides a unified factory for creating model and guide functions
for all SCRIBE model types. It uses a registry-based design that consolidates
the logic for all model variants.

Factory Functions
-----------------
create_model
    Unified factory that creates any model type from a ModelConfig.
create_model_from_params
    Convenience function with flat parameters.

Registries
----------
MODEL_EXTRA_PARAMS
    Maps model types to their extra parameters (gate, p_capture).
LIKELIHOOD_REGISTRY
    Maps model types to their likelihood classes.

Helper Builders
---------------
build_gate_spec
    Build gate parameter spec for zero-inflated models.
build_capture_spec
    Build capture parameter spec for VCP models.
build_extra_param_spec
    Dispatch to appropriate builder based on parameter name.
apply_prior_guide_overrides
    Apply user-provided prior/guide overrides to param specs.

Supported Models
----------------
- **nbdm**: Negative Binomial Dropout Model
- **zinb**: Zero-Inflated Negative Binomial
- **nbvcp**: NB with Variable Capture Probability
- **zinbvcp**: ZINB with Variable Capture Probability

Configuration Options
---------------------
All models support:

- **parameterization**: "canonical", "mean_prob", "mean_odds"
  (or aliases: "standard", "linked", "odds_ratio")
- **unconstrained**: Use Normal+transform instead of constrained distributions
- **guide_families**: Per-parameter guide family configuration
- **n_components**: Number of mixture components
- **mixture_params**: Which parameters are mixture-specific

Examples
--------
>>> from scribe.models.presets import create_model, create_model_from_params
>>> from scribe.models.config import ModelConfigBuilder
>>>
>>> # Using unified factory with ModelConfig
>>> config = ModelConfigBuilder().for_model("zinb").build()
>>> model, guide = create_model(config)
>>>
>>> # Using convenience function with flat params
>>> model, guide = create_model_from_params(
...     model="zinb",
...     parameterization="linked",
...     n_components=3,
... )
>>>
>>> # With custom priors
>>> model, guide = create_model_from_params(
...     model="nbdm",
...     priors={"p": (2.0, 2.0), "r": (1.0, 0.5)},
... )

See Also
--------
scribe.models.builders : Low-level building blocks.
scribe.models.components : Reusable components.
scribe.models.config : Configuration classes.
"""

# Unified factory
from .factory import (
    create_model,
    create_model_from_params,
    validate_model_guide_compatibility,
)

# Registries and helpers
from .registry import (
    GUIDE_FAMILY_REGISTRY,
    LIKELIHOOD_REGISTRY,
    MODEL_EXTRA_PARAMS,
    apply_prior_guide_overrides,
    build_capture_spec,
    build_extra_param_spec,
    build_gate_spec,
    get_guide_family,
)

__all__ = [
    # Unified factory
    "create_model",
    "create_model_from_params",
    "validate_model_guide_compatibility",
    # Registries
    "MODEL_EXTRA_PARAMS",
    "LIKELIHOOD_REGISTRY",
    "GUIDE_FAMILY_REGISTRY",
    # Registry helpers
    "build_gate_spec",
    "build_capture_spec",
    "build_extra_param_spec",
    "apply_prior_guide_overrides",
    "get_guide_family",
]

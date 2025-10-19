# SCRIBE Model Configuration

## Overview

SCRIBE uses a modern, type-safe configuration system based on `Pydantic` and the
`Builder` pattern. This provides:

- **Automatic validation**: Invalid configurations are impossible
- **Immutability**: Configurations can't be accidentally modified.
- **Type safety**: Full IDE support and type checking.
- **Fluent API**: Readable, chainable configuration.
- **Serialization**: Easy JSON export/import.

## Quick Start

### Simple SVI Model

```python
from scribe.models import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_inference("svi")
    .build())
```

### Unconstrained Linked Model with Priors

```python
config = (ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .unconstrained()
    .with_priors(p=(1.0, 1.0), mu=(0.0, 1.0))
    .build())
```

### VAE Model

```python
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_vae(
        latent_dim=5,
        hidden_dims=[256, 128],
        activation="gelu",
        prior_type="decoupled"
    )
    .build())
```

### Mixture Model

```python
config = (ModelConfigBuilder()
    .for_model("zinb")
    .as_mixture(n_components=3, component_specific=True)
    .build())
```

## Key Concepts

### Immutability

Configurations are immutable by default. To "modify" a config, create a new one:

```python
config1 = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_priors(p=(1.0, 1.0))
    .build())

# Create new config with updated priors
config2 = config1.with_updated_priors(r=(2.0, 0.5))
```

### Computed Fields

Many properties are computed automatically:

```python
config = (ModelConfigBuilder()
    .for_model("zinb")
    .as_mixture(n_components=3)
    .build())

print(config.is_mixture)  # True
print(config.is_zero_inflated)  # True
print(config.active_parameters)  # {'p', 'r', 'gate', 'mixing'}
```

### Type Safety

Use enums for type-safe configuration:

```python
from scribe.models import ModelType, Parameterization, InferenceMethod

config = (ModelConfigBuilder()
    .for_model(ModelType.NBDM)
    .with_parameterization(Parameterization.LINKED)
    .with_inference(InferenceMethod.VAE)
    .build())
```

### Accessing Configuration

Access nested configurations through the object hierarchy:

```python
# Priors
config.priors.p  # Prior for p parameter
config.priors.r  # Prior for r parameter

# VAE configuration
config.vae.latent_dim  # VAE latent dimension
config.vae.hidden_dims  # VAE hidden layer sizes

# Guides
config.guides.p  # Guide for p parameter
```

## API Reference

See individual class docstrings for complete API documentation:

- `ModelConfigBuilder`: Fluent builder for configurations
- `ConstrainedModelConfig`: Configuration for constrained models
- `UnconstrainedModelConfig`: Configuration for unconstrained models
- `PriorConfig`: Prior parameter configuration
- `GuideConfig`: Guide parameter configuration
- `VAEConfig`: VAE-specific configuration

## Configuration Structure

The configuration system is organized in the `config/` subdirectory:

```
src/scribe/models/config/
├── __init__.py          # Export all config classes
├── enums.py             # ModelType, Parameterization, InferenceMethod, etc.
├── groups.py            # PriorConfig, GuideConfig, VAEConfig
├── base.py              # ConstrainedModelConfig, UnconstrainedModelConfig
└── builder.py           # ModelConfigBuilder
```

## Examples

### Complex Configuration

```python
# ZINB mixture model with unconstrained parameterization
config = (ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .unconstrained()
    .as_mixture(n_components=3, component_specific=True)
    .with_low_rank_guide(10)
    .with_priors(p=(1.0, 1.0), mu=(0.0, 1.0), gate=(2.0, 2.0))
    .with_vae(latent_dim=5, hidden_dims=[256, 128])
    .build())
```

### VCP Model with Odds Ratio

```python
config = (ModelConfigBuilder()
    .for_model("nbvcp")
    .with_parameterization("odds_ratio")
    .with_priors(phi=(1.0, 1.0), mu=(0.0, 1.0), phi_capture=(2.0, 0.5))
    .build())
```

### Using Enums

```python
from scribe.models import ModelType, Parameterization, InferenceMethod, VAEPriorType

config = (ModelConfigBuilder()
    .for_model(ModelType.ZINB)
    .with_parameterization(Parameterization.ODDS_RATIO)
    .with_inference(InferenceMethod.VAE)
    .with_vae(prior_type=VAEPriorType.DECOUPLED)
    .build())
```

## Validation

All configurations are automatically validated:

```python
# This raises ValueError: n_components must be >= 2
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .as_mixture(n_components=1)
    .build())

# This raises ValueError: guide_rank must be positive
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_low_rank_guide(0)
    .build())

# This raises ValueError: Prior parameters must be positive
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_priors(p=(-1.0, 1.0))
    .build())
```

## Benefits

1. **Type Safety**: Enums prevent typos and invalid values
2. **Self-Validating**: Impossible to create invalid configurations
3. **Immutable**: No accidental modifications
4. **Fluent API**: Readable, discoverable, chainable
5. **IDE Support**: Full autocomplete and type checking
6. **Serializable**: Easy JSON import/export
7. **Testable**: Builder pattern makes tests cleaner
8. **Organized**: Config system in its own subdirectory
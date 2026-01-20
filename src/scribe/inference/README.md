# SCRIBE Inference Module

This module provides a unified interface for all SCRIBE inference methods (SVI,
MCMC, VAE) with support for both simple preset-based and advanced
ModelConfig-based APIs.

## Overview

The inference module is the main entry point for running SCRIBE inference. It
provides:

1. **Unified API**: Single `run_scribe()` function for all inference methods
2. **Preset Support**: Simple string-based model specification
3. **Advanced Configuration**: Full ModelConfig and InferenceConfig support
4. **Multiple Dispatch**: Type-safe routing to inference handlers
5. **Comprehensive Documentation**: Numpy-style docstrings and examples

## Architecture

```
inference/
├── __init__.py          # Main API: run_scribe()
├── README.md            # This file
├── utils.py             # Shared utilities, data processing
├── preset_builder.py    # Preset → ModelConfig conversion
├── inference_config.py  # Default config creation
├── dispatcher.py        # Multiple dispatch routing
├── svi.py              # SVI inference execution
├── mcmc.py             # MCMC inference execution
└── vae.py              # VAE inference execution
```

### Execution Flow

```
run_scribe(counts, ...)
    │
    ├── Data Processing (utils.py)
    │   └── process_counts_data()
    │
    ├── Config Building (preset_builder.py or direct)
    │   ├── Simple: Preset parameters → build_config_from_preset() → ModelConfig
    │   └── Advanced: Direct ModelConfig input
    │
    ├── Inference Config (inference_config.py)
    │   └── create_default_inference_config() → InferenceConfig
    │
    └── Inference Routing (dispatcher.py)
        ├── @dispatch(SVI, SVIConfig) → svi._run_svi_inference()
        ├── @dispatch(MCMC, MCMCConfig) → mcmc._run_mcmc_inference()
        └── @dispatch(VAE, SVIConfig) → vae._run_vae_inference()
```

## Quick Start

### Simple Preset-Based API

The simplest way to run inference is using preset parameters:

```python
from scribe.inference import run_scribe
from scribe.models.config import InferenceConfig, SVIConfig

# Create inference config
svi_config = SVIConfig(n_steps=50000, batch_size=256)
inference_config = InferenceConfig.from_svi(svi_config)

# Run inference
results = run_scribe(
    counts=adata,
    model="nbdm",
    parameterization="mean_prob",
    inference_method="svi",
    inference_config=inference_config,
)
```

### Advanced ModelConfig API

For more control, use ModelConfigBuilder:

```python
from scribe.inference import run_scribe
from scribe.models.config import (
    ModelConfigBuilder,
    InferenceConfig,
    MCMCConfig,
)

# Build model config
model_config = (
    ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("mean_odds")
    .with_inference("mcmc")
    .as_mixture(n_components=3)
    .build()
)

# Build inference config
mcmc_config = MCMCConfig(n_samples=5000, n_chains=4)
inference_config = InferenceConfig.from_mcmc(mcmc_config)

# Run inference
results = run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=inference_config,
)
```

## API Reference

### `run_scribe()`

Main entry point for all SCRIBE inference methods.

**Parameters:**

- `counts`: Count matrix or AnnData object
- `model`: Model type ("nbdm", "zinb", "nbvcp", "zinbvcp") - required if
  `model_config` is None
- `parameterization`: Parameterization scheme ("canonical", "mean_prob",
  "mean_odds")
- `inference_method`: Inference method ("svi", "mcmc", "vae")
- `model_config`: Fully configured ModelConfig (optional, overrides preset
  params)
- `inference_config`: Unified InferenceConfig (optional, uses defaults if None)
- `data_config`: DataConfig for data processing (optional)
- `cells_axis`: Axis for cells (0=rows, 1=columns) - used if `data_config` is
  None
- `layer`: AnnData layer name - used if `data_config` is None
- `seed`: Random seed (default: 42)

**Returns:**

- `ScribeSVIResults`, `ScribeMCMCResults`, or `ScribeVAEResults` depending on
  inference method

**Examples:**

See Quick Start section above.

### `build_config_from_preset()`

Build ModelConfig from preset parameters.

**Location:** `inference/preset_builder.py`

**Parameters:**

- `model`: Model type string
- `parameterization`: Parameterization string
- `inference_method`: Inference method string
- `unconstrained`: Use unconstrained parameterization
- `guide_rank`: Rank for low-rank guide
- `n_components`: Number of mixture components
- `priors`: Dictionary of prior parameters
- `amortize_capture`: Enable amortized capture probability (VCP models only)
- `capture_hidden_dims`: MLP hidden layer dimensions for amortizer
- `capture_activation`: Activation function for amortizer MLP

**Returns:**

- `ModelConfig` object

### `create_default_inference_config()`

Create default InferenceConfig for an inference method.

**Location:** `inference/inference_config.py`

**Parameters:**

- `inference_method`: InferenceMethod enum

**Returns:**

- `InferenceConfig` with default settings

## Usage Patterns

### Pattern 1: Simple Preset API

Best for quick experiments and standard use cases:

```python
from scribe.inference import run_scribe
from scribe.models.config import InferenceConfig, SVIConfig

results = run_scribe(
    counts=adata,
    model="nbdm",
    parameterization="canonical",
    inference_method="svi",
    inference_config=InferenceConfig.from_svi(
        SVIConfig(n_steps=50000)
    ),
)
```

### Pattern 2: Advanced ModelConfig API

Best for complex configurations and reproducibility:

```python
from scribe.inference import run_scribe
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config import (
    InferenceConfig,
    SVIConfig,
    GuideFamilyConfig,
)
from scribe.models.components import LowRankGuide

# Option A: preset with guide_rank
model_config = build_config_from_preset(
    model="zinb",
    parameterization="mean_prob",
    inference_method="svi",
    guide_rank=15,
)

# Option B: ModelConfigBuilder with explicit guide families
from scribe.models.config import ModelConfigBuilder

model_config = (
    ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("mean_prob")
    .with_inference("svi")
    .with_guide_families(GuideFamilyConfig(mu=LowRankGuide(rank=15)))
    .build()
)

inference_config = InferenceConfig.from_svi(
    SVIConfig(n_steps=100000, batch_size=512)
)

results = run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=inference_config,
)
```

### Pattern 3: Default Configs

Use defaults for quick testing:

```python
from scribe.inference import run_scribe

# Uses default InferenceConfig (100k steps for SVI) when inference_config=None
results = run_scribe(
    counts=adata,
    model="nbdm",
    inference_method="svi",
)
```

### Pattern 4: Amortized Inference

For VCP models with large datasets, use amortized capture probability:

```python
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config import InferenceConfig, SVIConfig

# Using preset builder
model_config = build_config_from_preset(
    model="nbvcp",
    parameterization="canonical",
    inference_method="svi",
    amortize_capture=True,
    capture_hidden_dims=[128, 64],
    capture_activation="gelu",
)

results = run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=InferenceConfig.from_svi(
        SVIConfig(n_steps=100000, batch_size=512)
    ),
)
```

## Parameterization Names

The module supports both new and old parameterization names for backward
compatibility:

| New Name (Preferred) | Old Name (Backward Compat) | Description                         |
|----------------------|----------------------------|-------------------------------------|
| `"canonical"`        | `"standard"`               | Directly samples p and r            |
| `"mean_prob"`        | `"linked"`                 | Samples p and mu, derives r         |
| `"mean_odds"`        | `"odds_ratio"`             | Samples phi and mu, derives p and r |

## Design Decisions

### Why Unified InferenceConfig?

- **Type Safety**: Pydantic validation ensures correct config types for each
  method
- **Consistency**: Single interface for all inference configurations
- **Flexibility**: Factory methods make it easy to create configs
- **Validation**: Automatic validation of method-config compatibility

### Why Multiple Dispatch?

- **Type Safety**: Compile-time routing based on types
- **Extensibility**: Easy to add new inference methods
- **Clarity**: Clear separation of concerns
- **Consistency**: Matches pattern used elsewhere in codebase

### Why Preset Builder?

- **Simplicity**: Easy model specification for common cases
- **Consistency**: Uses same preset factories as rest of codebase
- **Flexibility**: Can still use ModelConfig for advanced cases
- **Documentation**: Clear examples for common use cases

## Migration Guide

### From Old API

**Old way (deprecated: many boolean flags and loose params):**

```python
# Deprecated: zero_inflated, variable_capture, mixture_model, etc.
results = run_scribe(
    counts=adata,
    inference_method="svi",
    zero_inflated=False,
    variable_capture=False,
    parameterization="standard",
    n_steps=50000,
    batch_size=256,
)
```

**New way (preset-based):**

```python
from scribe.inference import run_scribe
from scribe.models.config import InferenceConfig, SVIConfig

results = run_scribe(
    counts=adata,
    model="nbdm",
    parameterization="canonical",
    inference_method="svi",
    inference_config=InferenceConfig.from_svi(
        SVIConfig(n_steps=50000, batch_size=256)
    ),
)
```

**New way (ModelConfig-based):**

```python
from scribe.inference import run_scribe
from scribe.models.config import (
    ModelConfigBuilder,
    InferenceConfig,
    SVIConfig,
)

model_config = (
    ModelConfigBuilder()
    .for_model("nbdm")
    .with_parameterization("canonical")
    .with_inference("svi")
    .build()
)

results = run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=InferenceConfig.from_svi(
        SVIConfig(n_steps=50000, batch_size=256)
    ),
)
```

## See Also

- `scribe.models.config`: Configuration classes and builders
- `scribe.models.presets`: Preset factory functions
- `scribe.models.parameterizations`: Parameterization strategies
- `scribe.svi`: SVI inference engine
- `scribe.mcmc`: MCMC inference engine
- `scribe.vae`: VAE inference engine

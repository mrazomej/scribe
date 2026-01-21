# SCRIBE Configuration System

This directory contains the modern configuration system for SCRIBE models.

## Structure

- `enums.py`: Type-safe enums for all categorical values
- `groups.py`: Grouped configuration classes (priors, guides, VAE)
- `base.py`: Unified ModelConfig class
- `builder.py`: Fluent builder for constructing configurations
- `parameter_mapping.py`: Robust parameter mapping system for parameterizations

## Design Principles

1. **Pydantic for validation**: All configs use Pydantic v2 for automatic
   validation
2. **Immutability**: Configs are frozen by default (use `with_updated_*`
   methods)
3. **Type safety**: Enums prevent invalid categorical values
4. **Computed fields**: Properties like `is_mixture` are computed automatically
5. **Builder pattern**: Fluent, chainable API for construction
6. **Robust parameter mapping**: Declarative system for
   parameter-parameterization relationships

## Usage

Always use the builder pattern to create configurations:

```python
from scribe.models.config import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_inference("svi")
    .with_priors(p=(1.0, 1.0))
    .build())
```

Direct instantiation of config classes is possible but not recommended.

## Classes

### ModelConfigBuilder

The primary interface for creating configurations. Provides fluent methods:

- `.for_model(model_type)`: Set the base model
- `.with_parameterization(param)`: Set parameterization type
- `.with_inference(method)`: Set inference method
- `.unconstrained()`: Use unconstrained parameterization
- `.as_mixture(n_components, mixture_params)`: Configure as mixture
- `.with_guide_families(guide_families)`: Set per-parameter guide families
- `.with_priors(**priors)`: Set prior parameters
- `.with_guides(**guides)`: Set guide parameters
- `.with_vae(**vae_params)`: Configure VAE parameters
- `.build()`: Create and validate the configuration

### ModelConfig

Unified configuration class for all SCRIBE models. Supports both constrained and
unconstrained parameterizations via the `unconstrained` boolean field. Prior and
guide hyperparameters are stored in `param_specs` (list of `ParamSpec`); see
`scribe.models.builders.parameter_specs`.

### PriorConfig / UnconstrainedPriorConfig

Prior parameter configurations with automatic validation:

- `p`: Success probability prior
- `r`: Dispersion prior
- `mu`: Mean prior
- `phi`: Odds ratio prior
- `gate`: Zero-inflation gate prior
- `p_capture`: Capture probability prior
- `phi_capture`: Capture phi prior
- `mixing`: Mixture weights prior

### GuideConfig / UnconstrainedGuideConfig

Guide parameter configurations (same structure as priors).

### VAEConfig

VAE-specific configuration:

- `latent_dim`: Latent space dimensionality
- `hidden_dims`: Encoder/decoder hidden layer sizes
- `activation`: Activation function
- `prior_type`: Prior type (standard/decoupled)
- `prior_num_layers`: Number of coupling layers
- `prior_hidden_dims`: Prior hidden layer sizes
- `prior_activation`: Prior activation function
- `prior_mask_type`: Mask type for decoupled prior
- `standardize`: Whether to standardize input data

### SVIConfig

SVI-specific configuration for Stochastic Variational Inference:

- `optimizer`: Optimizer for variational inference (defaults to Adam)
- `loss`: Loss function (defaults to TraceMeanField_ELBO)
- `n_steps`: Maximum number of optimization steps (must be > 0)
- `batch_size`: Mini-batch size (must be > 0, None uses full dataset)
- `stable_update`: Use numerically stable parameter updates
- `early_stopping`: Optional `EarlyStoppingConfig` for automatic convergence
  detection

### EarlyStoppingConfig

Configuration for early stopping during SVI optimization:

- `enabled`: Whether to enable early stopping (default: True)
- `patience`: Steps without improvement before stopping (default: 500)
- `min_delta`: Minimum change to qualify as improvement (default: 1.0, suitable
  for ELBO values ~10^6-10^7)
- `check_every`: Check convergence every N steps (default: 10)
- `smoothing_window`: Window size for loss smoothing (default: 50)
- `restore_best`: Restore best parameters when stopping (default: True)

```python
from scribe.models.config import SVIConfig, EarlyStoppingConfig

# Configure SVI with early stopping
svi_config = SVIConfig(
    n_steps=100000,
    batch_size=512,
    early_stopping=EarlyStoppingConfig(
        patience=500,
        min_delta=1.0,
    ),
)
```

### MCMCConfig

MCMC-specific configuration for Markov Chain Monte Carlo:

- `n_samples`: Number of MCMC samples (must be > 0)
- `n_warmup`: Number of warmup samples (must be > 0)
- `n_chains`: Number of parallel chains (must be > 0)
- `mcmc_kwargs`: Additional keyword arguments for MCMC kernel

### DataConfig

Data processing configuration:

- `cells_axis`: Axis for cells in count matrix (0=rows, 1=columns, must be 0 or
  1)
- `layer`: Layer in AnnData to use for counts (None uses .X)

### Parameter Mapping System

The parameter mapping system provides a robust, declarative way to define which
parameters are active for each parameterization type:

```python
from scribe.models.config import get_active_parameters, Parameterization

# Get active parameters for a configuration
params = get_active_parameters(
    parameterization=Parameterization.STANDARD,
    model_type="zinb",
    is_mixture=True,
    is_zero_inflated=True,
    uses_variable_capture=False,
)
# Returns: {"p", "r", "gate", "mixing"}

# Validate parameter consistency
from scribe.models.config import validate_parameter_consistency

errors = validate_parameter_consistency(
    parameterization=Parameterization.STANDARD,
    model_type="nbdm",
    provided_params={"p", "r", "invalid_param"},
    is_mixture=False,
    is_zero_inflated=False,
    uses_variable_capture=False,
)
# Returns: ["Unsupported parameters for standard parameterization: invalid_param"]
```

**Key Features:**
- **Declarative**: Parameter-parameterization relationships are clearly defined
- **Maintainable**: Easy to add new parameterizations or modify existing ones
- **Type-safe**: Full IDE support and validation
- **Comprehensive**: Handles all model types, parameterizations, and features
- **Extensible**: Simple to add new parameters or parameterization types

### Enums

- `ModelType`: NBDM, ZINB, NBVCP, ZINBVCP
- `Parameterization`: STANDARD, LINKED, ODDS_RATIO
- `InferenceMethod`: SVI, MCMC, VAE
- `VAEPriorType`: STANDARD, DECOUPLED
- `VAEMaskType`: ALTERNATING, SEQUENTIAL
- `VAEActivation`: RELU, GELU, TANH, SIGMOID

## Validation

All configurations are automatically validated using Pydantic:

- **Type checking**: Ensures correct types
- **Value validation**: Checks parameter ranges
- **Consistency**: Validates parameter combinations
- **Required fields**: Ensures required parameters are present

## Examples

### Basic Configuration

```python
from scribe.models.config import ModelConfigBuilder

# Simple SVI model
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_inference("svi")
    .build())
```

### Complex Configuration

```python
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import LowRankGuide

# ZINB mixture with unconstrained parameterization
config = (ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .unconstrained()
    .as_mixture(n_components=3, mixture_params=["p"])
    .with_guide_families(GuideFamilyConfig(r=LowRankGuide(rank=10)))
    .with_priors(p=(1.0, 1.0), mu=(0.0, 1.0), gate=(2.0, 2.0))
    .build())
```

### Using Enums

```python
from scribe.models.config import ModelType, Parameterization, InferenceMethod

config = (ModelConfigBuilder()
    .for_model(ModelType.NBDM)
    .with_parameterization(Parameterization.LINKED)
    .with_inference(InferenceMethod.VAE)
    .build())
```

### Immutable Updates

```python
# Create initial config
config1 = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_priors(p=(1.0, 1.0))
    .build())

# Create updated config
config2 = config1.with_updated_priors(r=(2.0, 0.5))

# Original config unchanged
assert config1.priors.r is None
assert config2.priors.r == (2.0, 0.5)
```

### Using Configuration Objects with run_scribe

Use `ModelConfig` with `InferenceConfig` and `run_scribe`:

```python
from scribe.inference import run_scribe
from scribe.models.config import (
    ModelConfigBuilder,
    InferenceConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
)

# SVI with explicit configs
model_config = (
    ModelConfigBuilder()
    .for_model("nbdm")
    .with_inference("svi")
    .build()
)
inference_config = InferenceConfig.from_svi(
    SVIConfig(n_steps=50_000, batch_size=256, stable_update=True)
)
results = run_scribe(
    counts,
    model_config=model_config,
    inference_config=inference_config,
    data_config=DataConfig(cells_axis=1, layer="counts"),
)

# MCMC with custom configuration
mcmc_config = MCMCConfig(n_samples=5000, n_warmup=1000, n_chains=4)
results = run_scribe(
    counts,
    model_config=(
        ModelConfigBuilder().for_model("nbdm").with_inference("mcmc").build()
    ),
    inference_config=InferenceConfig.from_mcmc(mcmc_config),
)

# Save and reuse InferenceConfig
import pickle

with open("my_inference_config.pkl", "wb") as f:
    pickle.dump(inference_config, f)

with open("my_inference_config.pkl", "rb") as f:
    loaded = pickle.load(f)

results = run_scribe(counts, model="nbdm", inference_config=loaded)
```

**Benefits of using configuration objects:**

- **Validation**: Automatic parameter validation with helpful error messages
- **Reusability**: Save and reuse configurations across runs
- **Type Safety**: Full IDE autocomplete and type checking
- **Immutability**: Prevents accidental parameter modification
- **Serialization**: Easy JSON/pickle export for reproducibility

## Benefits

1. **Type Safety**: Enums prevent typos and invalid values
2. **Self-Validating**: Impossible to create invalid configurations
3. **Immutable**: No accidental modifications
4. **Fluent API**: Readable, discoverable, chainable
5. **IDE Support**: Full autocomplete and type checking
6. **Serializable**: Easy JSON import/export
7. **Testable**: Builder pattern makes tests cleaner
8. **Organized**: Clear separation of concerns

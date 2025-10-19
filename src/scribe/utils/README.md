# SCRIBE Utils Module

This module contains utility classes and functions used throughout the SCRIBE
codebase for parameter collection, data processing, and other common operations.

## Overview

The utils module provides two main components:

1. **ParameterCollector**: A utility class for collecting and mapping optional
   parameters
2. **Core Utilities**: Essential functions for data processing and distribution
   handling

## ParameterCollector

The `ParameterCollector` class provides static methods to collect non-None
parameters and map them to the appropriate ModelConfig attribute names based on
the model parameterization and constraint settings.

### Key Methods

#### `collect_non_none(**kwargs) -> Dict[str, Any]`

Filters out None values from keyword arguments.

```python
from src.scribe.utils import ParameterCollector

# Filter out None values
params = ParameterCollector.collect_non_none(
    a=1, b=None, c="hello", d=None
)
# Result: {'a': 1, 'c': 'hello'}
```

#### `collect_and_map_priors(...) -> Dict[str, Any]`

Collects and maps prior parameters to ModelConfig attribute names based on
parameterization type and constraint settings.

```python
# Standard parameterization (constrained)
prior_config = ParameterCollector.collect_and_map_priors(
    unconstrained=False,
    parameterization="standard",
    r_prior=(1.0, 1.0),
    p_prior=(2.0, 0.5)
)
# Result: {'r_param_prior': (1.0, 1.0), 'p_param_prior': (2.0, 0.5)}

# Unconstrained parameterization
prior_config = ParameterCollector.collect_and_map_priors(
    unconstrained=True,
    parameterization="standard",
    r_prior=(0.0, 1.0),
    p_prior=(0.0, 1.0)
)
# Result: {'r_unconstrained_prior': (0.0, 1.0), 'p_unconstrained_prior': (0.0, 1.0)}
```

#### `collect_vae_params(...) -> Dict[str, Any]`

Collects VAE-specific parameters for ModelConfig.

```python
vae_config = ParameterCollector.collect_vae_params(
    vae_latent_dim=5,
    vae_hidden_dims=[256, 128],
    vae_activation="gelu",
    vae_prior_type="decoupled"
)
# Result: {'vae_latent_dim': 5, 'vae_hidden_dims': [256, 128], ...}
```

### Parameterization Support

The ParameterCollector supports all SCRIBE parameterization types:

- **Standard**: Beta/LogNormal distributions for p/r parameters
- **Linked**: Beta/LogNormal for p/mu parameters  
- **Odds Ratio**: BetaPrime/LogNormal for phi/mu parameters

For each parameterization, it correctly maps user-provided prior parameters to
the appropriate ModelConfig attribute names, handling both constrained and
unconstrained variants.

## Core Utilities

### `numpyro_to_scipy(distribution) -> scipy.stats.rv_continuous`

Converts NumPyro distributions to their SciPy equivalents for analysis and
visualization.

```python
from src.scribe.utils import numpyro_to_scipy
import numpyro.distributions as dist

# Convert NumPyro Beta to SciPy Beta
numpyro_beta = dist.Beta(1.0, 1.0)
scipy_beta = numpyro_to_scipy(numpyro_beta)
```

**Supported distributions:**
- `Beta` → `scipy.stats.beta`
- `Gamma` → `scipy.stats.gamma`
- `LogNormal` → `scipy.stats.lognorm`
- `Dirichlet` → `scipy.stats.dirichlet`
- `BetaPrime` → `scipy.stats.betaprime`

### `git_root(current_path=None) -> str`

Finds the root directory of a Git repository.

```python
from src.scribe.utils import git_root

# Find Git root from current directory
root = git_root()
print(f"Repository root: {root}")

# Find Git root from specific path
root = git_root("/path/to/some/subdirectory")
```

### `use_cpu()`

Context manager to temporarily force JAX computations to run on CPU.

```python
from src.scribe.utils import use_cpu
import jax.numpy as jnp

# Force computations to run on CPU
with use_cpu():
    # This will run on CPU even if GPU is available
    result = jnp.sum(jnp.array([1, 2, 3, 4, 5]))
```

## Usage in SCRIBE

### In run_scribe()

The ParameterCollector is used in the main `run_scribe()` function to replace
repetitive if-statement parameter collection:

```python
# Before (repetitive if-statements)
user_priors = {}
if r_prior is not None:
    user_priors["r_prior"] = r_prior
if p_prior is not None:
    user_priors["p_prior"] = p_prior
# ... 8 more similar statements

# After (clean utility approach)
prior_config = ParameterCollector.collect_and_map_priors(
    unconstrained=unconstrained,
    parameterization=parameterization,
    r_prior=r_prior,
    p_prior=p_prior,
    # ... all parameters
)
```

### In Results Classes

The core utilities are used throughout the results classes for data processing
and analysis:

```python
from src.scribe.utils import numpyro_to_scipy

# Convert posterior distributions for analysis
scipy_dists = {
    name: numpyro_to_scipy(dist) 
    for name, dist in posterior_samples.items()
}
```

## Benefits

1. **Code Reduction**: Eliminates ~70 lines of repetitive parameter collection
   code
2. **Readability**: Clear intent with named methods and comprehensive
   documentation
3. **Reusability**: Utility functions can be used across the entire codebase
4. **Testability**: Easy to unit test parameter mapping and utility functions
5. **Maintainability**: Single place to update parameter mapping rules and
   utilities
6. **Backward Compatibility**: All existing imports continue to work unchanged

## Testing

The module includes comprehensive test coverage in
`tests/test_parameter_collector.py`:

- Parameter collection and filtering
- Prior mapping for all parameterization types
- VAE parameter collection
- Edge cases and error handling

Run tests with:

```bash
pytest tests/test_parameter_collector.py -v
```

## Dependencies

- **JAX**: For numerical computations and device management
- **NumPyro**: For probabilistic distributions
- **SciPy**: For statistical distributions and analysis
- **Python 3.8+**: For type hints and modern Python features

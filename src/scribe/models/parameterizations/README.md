# Parameterizations Module

This module implements the **Strategy Pattern** for handling different
parameterization schemes in probabilistic models. Each parameterization defines
how model parameters are sampled and derived, eliminating nested conditionals in
preset factories.

## Overview

A parameterization scheme determines:
- **Core parameters**: Which parameters are sampled directly from priors
- **Derived parameters**: Which parameters are computed deterministically from
  core parameters
- **Parameter transformations**: How model-specific parameters (like
  `p_capture`) are transformed

## Parameterization Classes

### `CanonicalParameterization` (formerly "standard")

Directly samples both the success probability `p` and dispersion `r`:

- **Core parameters**: `p`, `r`
- **Derived parameters**: None
- **Gene parameter**: `r` (gene-specific)

**Example**:
```python
from scribe.models.parameterizations import PARAMETERIZATIONS

param_strategy = PARAMETERIZATIONS["canonical"]
param_specs = param_strategy.build_param_specs(
    unconstrained=False,
    guide_families=GuideFamilyConfig(),
)
# Returns: [BetaSpec("p"), LogNormalSpec("r")]
```

### `MeanProbParameterization` (formerly "linked")

Samples `p` and mean expression `mu`, then derives `r`:

- **Core parameters**: `p`, `mu`
- **Derived parameters**: `r = mu * (1 - p) / p`
- **Gene parameter**: `mu` (gene-specific)

This links dispersion to mean expression, which can help capture correlations in
the variational posterior.

**Example**:
```python
param_strategy = PARAMETERIZATIONS["mean_prob"]
param_specs = param_strategy.build_param_specs(...)
derived_params = param_strategy.build_derived_params()
# Returns: [DerivedParam("r", lambda p, mu: mu * (1 - p) / p, ["p", "mu"])]
```

### `MeanOddsParameterization` (formerly "odds_ratio")

Samples odds ratio `phi` and mean expression `mu`, then derives both `p` and
`r`:

- **Core parameters**: `phi`, `mu`
- **Derived parameters**: 
  - `p = 1 / (1 + phi)`
  - `r = mu * phi`
- **Gene parameter**: `mu` (gene-specific)
- **Parameter transformations**: `p_capture` → `phi_capture`

This parameterization is numerically more stable than `mean_prob` when `p` is
close to 0 or 1, as it avoids division by `p` in the computation of `r`.

**Example**:
```python
param_strategy = PARAMETERIZATIONS["mean_odds"]
# Transform model-specific parameter
capture_param = param_strategy.transform_model_param("p_capture")
# Returns: "phi_capture"
```

## Usage in Unified Factory

The unified factory uses parameterizations to eliminate nested conditionals:

```python
from ..parameterizations import PARAMETERIZATIONS

def create_model(model_config):
    # Get parameterization strategy
    param_strategy = PARAMETERIZATIONS[model_config.parameterization]
    
    # Build parameter specs
    param_specs = param_strategy.build_param_specs(
        unconstrained, guide_families
    )
    
    # Get derived parameters
    derived_params = param_strategy.build_derived_params()
    
    # Transform model-specific parameters if needed
    capture_param_name = param_strategy.transform_model_param("p_capture")
    # For mean_odds: returns "phi_capture"
    # For others: returns "p_capture"
```

## Registry

The `PARAMETERIZATIONS` dictionary maps names to parameterization instances:

```python
PARAMETERIZATIONS = {
    # New names (preferred)
    "canonical": CanonicalParameterization(),
    "mean_prob": MeanProbParameterization(),
    "mean_odds": MeanOddsParameterization(),
    # Backward compatibility
    "standard": CanonicalParameterization(),
    "linked": MeanProbParameterization(),
    "odds_ratio": MeanOddsParameterization(),
}
```

Both new and old names are supported for backward compatibility.

## Benefits

1. **Eliminates nested conditionals**: No more `if parameterization ==
   "standard": if unconstrained: ...`
2. **Reduces duplication**: Parameterization logic is centralized
3. **Clearer semantics**: Names like "mean_prob" are more descriptive than
   "linked"
4. **Easy to extend**: Add new parameterizations by creating a new class
5. **Handles transformations**: Model-specific parameter transformations are
   encapsulated

## Extending

To add a new parameterization:

1. Create a new class inheriting from `Parameterization`
2. Implement all abstract methods:
   - `name`: Human-readable name
   - `core_parameters`: List of core parameter names
   - `gene_param_name`: Name of gene-specific parameter
   - `build_param_specs()`: Build parameter specs
   - `build_derived_params()`: Build derived parameter computations
   - `transform_model_param()`: Transform model-specific parameters (optional)
3. Add to `PARAMETERIZATIONS` registry

**Example**:
```python
class MyParameterization(Parameterization):
    @property
    def name(self) -> str:
        return "my_param"
    
    @property
    def core_parameters(self) -> List[str]:
        return ["alpha", "beta"]
    
    # ... implement other methods

PARAMETERIZATIONS["my_param"] = MyParameterization()
```

## See Also

- `scribe.models.presets`: Preset factories that use parameterizations
- `scribe.models.builders.parameter_specs`: Parameter specification classes
- `scribe.models.components.likelihoods`: Likelihoods that handle parameter
  transformations

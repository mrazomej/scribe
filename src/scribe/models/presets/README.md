# Model Presets

Pre-configured model/guide factories for common use cases.

## Available Presets

| Function | Model Type | Description |
|----------|------------|-------------|
| `create_nbdm()` | NBDM | Negative Binomial Dropout Model |
| `create_zinb()` | ZINB | Zero-Inflated Negative Binomial |
| `create_nbvcp()` | NBVCP | NB with Variable Capture Probability |
| `create_zinbvcp()` | ZINBVCP | ZINB with Variable Capture Probability |

## Configuration Options

All presets support:

```python
model, guide = create_nbvcp(
    # Parameterization
    parameterization="standard",     # "standard", "linked", "odds_ratio"
    unconstrained=False,             # Use Normal+transform instead of Beta/LogNormal
    
    # Per-parameter guide families
    r_guide="mean_field",            # "mean_field", "low_rank"
    guide_rank=10,                   # Rank for low_rank guide
    
    # Cell-specific parameter handling (NBVCP/ZINBVCP only)
    p_capture_guide="mean_field",    # "mean_field", "amortized"
    capture_amortizer=None,          # Custom Amortizer instance
)
```

### Parameterization Options

| Name | Parameters | Derived |
|------|------------|---------|
| `"standard"` | p, r | - |
| `"linked"` | p, mu | r = mu*(1-p)/p |
| `"odds_ratio"` | phi, mu | r = mu*phi, p = 1/(1+phi) |

### Guide Families

| Option | Description |
|--------|-------------|
| `"mean_field"` | Independent variational distribution per parameter |
| `"low_rank"` | Low-rank MVN capturing correlations between genes |
| `"amortized"` | Neural network predicts params from sufficient statistics |

## Examples

### Basic Usage

```python
from scribe.models.presets import create_nbdm

model, guide = create_nbdm()

# Use with SVI
svi = numpyro.infer.SVI(model, guide, optimizer, loss)
```

### Linked Parameterization with Low-Rank Guide

```python
from scribe.models.presets import create_nbdm

model, guide = create_nbdm(
    parameterization="linked",
    r_guide="low_rank",
    guide_rank=15,
)
```

### Amortized Capture Probability

For large datasets (100K+ cells), amortize p_capture inference:

```python
from scribe.models.presets import create_nbvcp

model, guide = create_nbvcp(
    p_capture_guide="amortized",
)
```

### Custom Amortizer

```python
from scribe.models.presets import create_nbvcp
from scribe.models.components import Amortizer, TOTAL_COUNT

# Custom amortizer with larger network
amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[128, 64, 32],
    output_params=["log_alpha", "log_beta"],
)

model, guide = create_nbvcp(
    parameterization="linked",
    r_guide="low_rank",
    guide_rank=15,
    p_capture_guide="amortized",
    capture_amortizer=amortizer,
)
```

### Unconstrained Parameterization

Use Normal + transform instead of constrained distributions:

```python
model, guide = create_nbdm(
    parameterization="standard",
    unconstrained=True,  # p ~ sigmoid(Normal), r ~ exp(Normal)
)
```

## Model Equivalence

Each preset creates models equivalent to the old monolithic files:

| Preset Call | Equivalent Old File |
|-------------|---------------------|
| `create_nbdm()` | `standard.py: nbdm_model/guide` |
| `create_nbdm(unconstrained=True)` | `standard_unconstrained.py: nbdm_model/guide` |
| `create_nbdm(r_guide="low_rank")` | `standard_low_rank.py: nbdm_guide` |
| `create_nbdm(parameterization="linked")` | `linked.py: nbdm_model/guide` |
| `create_nbdm(parameterization="odds_ratio")` | `odds_ratio.py: nbdm_model/guide` |

## Architecture

Presets are thin wrappers around the builder pattern:

```
create_nbdm()
    │
    ├── Build param_specs
    │   ├── BetaSpec("p", ...)
    │   └── LogNormalSpec("r", ...)
    │
    ├── ModelBuilder()
    │   ├── .add_param(specs)
    │   ├── .add_derived(...) if needed
    │   └── .with_likelihood(NegativeBinomialLikelihood())
    │
    └── GuideBuilder()
        └── .from_specs(specs)
```

## Adding New Presets

1. Create a new file in `presets/` (e.g., `my_model.py`)
2. Define a factory function following the pattern
3. Add to `__init__.py` exports
4. Add tests comparing output to expected behavior

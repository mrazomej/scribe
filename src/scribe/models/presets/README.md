# Model Presets

Pre-configured model/guide factories for common use cases.

## Available Presets

| Function           | Model Type | Description                            |
|--------------------|------------|----------------------------------------|
| `create_nbdm()`    | NBDM       | Negative Binomial Dropout Model        |
| `create_zinb()`    | ZINB       | Zero-Inflated Negative Binomial        |
| `create_nbvcp()`   | NBVCP      | NB with Variable Capture Probability   |
| `create_zinbvcp()` | ZINBVCP    | ZINB with Variable Capture Probability |

## Configuration Options

All presets support:

```python
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import LowRankGuide, AmortizedGuide, MeanFieldGuide

model, guide = create_nbvcp(
    parameterization="canonical",    # "canonical", "mean_prob", "mean_odds"
    unconstrained=False,             # Use Normal+transform instead of Beta/LogNormal
    guide_families=GuideFamilyConfig(
        mu=LowRankGuide(rank=10),    # or MeanFieldGuide() for mean-field
        p_capture=MeanFieldGuide(),  # or AmortizedGuide(amortizer=...)
    ),
    n_components=None,               # int for mixture
    mixture_params=None,
    priors=None,                     # e.g. {"p": (1,1), "mu": (0,1)}
    guides=None,
)
```

### Parameterization Options

| Name           | Parameters | Derived                   |
|----------------|------------|---------------------------|
| `"standard"`   | p, r       | -                         |
| `"linked"`     | p, mu      | r = mu*(1-p)/p            |
| `"odds_ratio"` | phi, mu    | r = mu*phi, p = 1/(1+phi) |

### Guide Families (via `GuideFamilyConfig`)

| Class                           | Description                                                  |
|---------------------------------|--------------------------------------------------------------|
| `MeanFieldGuide()`              | Independent variational distribution per parameter (default) |
| `LowRankGuide(rank=k)`          | Low-rank MVN capturing correlations between genes            |
| `AmortizedGuide(amortizer=...)` | Neural network predicts params from sufficient statistics    |

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
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import LowRankGuide

model, guide = create_nbdm(
    parameterization="linked",
    guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=15)),
)
```

### Amortized Capture Probability

For large datasets (100K+ cells), amortize p_capture inference:

```python
from scribe.models.presets import create_nbvcp
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import AmortizedGuide, TOTAL_COUNT, Amortizer

amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[64, 32],
    output_params=["log_alpha", "log_beta"],
)
model, guide = create_nbvcp(
    guide_families=GuideFamilyConfig(
        p_capture=AmortizedGuide(amortizer=amortizer),
    ),
)
```

### Custom Amortizer

```python
from scribe.models.presets import create_nbvcp
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import Amortizer, TOTAL_COUNT, LowRankGuide, AmortizedGuide

amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[128, 64, 32],
    output_params=["log_alpha", "log_beta"],
)

model, guide = create_nbvcp(
    parameterization="linked",
    guide_families=GuideFamilyConfig(
        mu=LowRankGuide(rank=15),
        p_capture=AmortizedGuide(amortizer=amortizer),
    ),
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

Each preset produces the same model/guide as the legacy per-file implementations:

| Preset Call                                                             | Legacy equivalent           |
|-------------------------------------------------------------------------|-----------------------------|
| `create_nbdm()`                                                         | `standard.py`               |
| `create_nbdm(unconstrained=True)`                                       | `standard_unconstrained.py` |
| `create_nbdm(guide_families=GuideFamilyConfig(r=LowRankGuide(rank=k)))` | `standard_low_rank.py`      |
| `create_nbdm(parameterization="linked")`                                | `linked.py`                 |
| `create_nbdm(parameterization="odds_ratio")`                            | `odds_ratio.py`             |

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

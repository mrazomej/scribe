# Model Components

Reusable building blocks for constructing probabilistic models.

## Overview

This directory contains the atomic components used by the builders:

- **likelihoods.py**: Likelihood functions (NB, ZINB, VCP variants)
- **guide_families.py**: Variational family implementations (MeanField, LowRank,
  Amortized)
- **amortizers.py**: Neural network amortizers for variational parameters

## Plate Handling

All components handle three plate modes:

| Mode             | `counts` | `batch_size` | Use Case                |
|------------------|----------|--------------|-------------------------|
| Prior Predictive | `None`   | -            | Generate synthetic data |
| Full Sampling    | provided | `None`       | MCMC, small datasets    |
| Batch Sampling   | provided | specified    | SVI on large datasets   |

## Likelihoods

Available likelihood components:

| Class                        | Description                | Parameters            |
|------------------------------|----------------------------|-----------------------|
| `NegativeBinomialLikelihood` | Standard NB                | p, r                  |
| `ZeroInflatedNBLikelihood`   | Zero-inflated NB           | p, r, gate            |
| `NBWithVCPLikelihood`        | NB with variable capture   | p, r, p_capture       |
| `ZINBWithVCPLikelihood`      | ZINB with variable capture | p, r, gate, p_capture |

### Example

```python
from scribe.models.components import NegativeBinomialLikelihood

likelihood = NegativeBinomialLikelihood()
# Use in ModelBuilder
builder.with_likelihood(likelihood)
```

## Guide Families

Guide families are **per-parameter** markers that specify which variational
approximation to use:

| Family                  | Description                     | Use Case          |
|-------------------------|---------------------------------|-------------------|
| `MeanFieldGuide`        | Factorized variational family   | Default, fast     |
| `LowRankGuide(rank)`    | Low-rank MVN covariance         | Gene correlations |
| `AmortizedGuide(net)`   | Neural network amortization     | High-dim params   |
| `GroupedAmortizedGuide` | Joint amortization (future VAE) | Multiple params   |

### Example

```python
from scribe.models.components import MeanFieldGuide, LowRankGuide
from scribe.models.builders import BetaSpec, LogNormalSpec

# Per-parameter guide families
BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide())
LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10))
```

## Amortizers

Amortizers predict variational parameters from sufficient statistics using an
MLP. This is useful for cell-specific parameters like capture probability, where
learning separate parameters for each cell would be prohibitive.

### Using the Factory (Recommended)

```python
from scribe.models.presets import create_capture_amortizer
from scribe.models.components import AmortizedGuide

# Automatically handles constrained vs unconstrained output parameters
amortizer = create_capture_amortizer(
    hidden_dims=[64, 32],
    activation="leaky_relu",
    unconstrained=False,  # Outputs (alpha, beta) for Beta distribution
)
guide_family = AmortizedGuide(amortizer=amortizer)
```

### Manual Construction

```python
from scribe.models.components import Amortizer, TOTAL_COUNT, AmortizedGuide

# Create amortizer for p_capture (constrained)
amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[64, 32],
    output_params=["log_alpha", "log_beta"],  # exp() applied to ensure > 0
)

# For unconstrained p_capture (Normal + sigmoid)
amortizer_unconstrained = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[64, 32],
    output_params=["loc", "log_scale"],  # loc unbounded, exp(log_scale) > 0
)

# Use in guide
BetaSpec(
    "p_capture", ("n_cells",), (1.0, 1.0),
    is_cell_specific=True,
    guide_family=AmortizedGuide(amortizer=amortizer)
)
```

### Built-in Sufficient Statistics

| Name          | Description                        | Formula              |
|---------------|------------------------------------|----------------------|
| `TOTAL_COUNT` | Log-transformed total UMI per cell | `log1p(sum(counts))` |

### Output Parameters

| Parameterization | Output Params           | Distribution                   |
|------------------|-------------------------|--------------------------------|
| Constrained      | `log_alpha`, `log_beta` | Beta(α, β) or BetaPrime(α, β)  |
| Unconstrained    | `loc`, `log_scale`      | Normal(loc, scale) → transform |

## Adding New Components

### New Likelihood

1. Subclass `Likelihood` in `likelihoods.py`
2. Implement `sample()` handling all three plate modes
3. Add to `__init__.py` exports
4. Add tests in `tests/models/test_likelihoods.py`

```python
class MyLikelihood(Likelihood):
    def sample(self, param_values, cell_specs, counts, dims, batch_size, model_config):
        # Handle all three modes: counts=None, batch_size=None, batch_size set
        ...
```

### New Guide Family

1. Create a marker dataclass in `guide_families.py`
2. Add dispatch methods in `builders/guide_builder.py` for each `(SpecType,
   NewGuideFamily)` pair
3. Add to `__init__.py` exports
4. Add tests verifying sampling behavior

```python
@dataclass
class MyGuideFamily(GuideFamily):
    my_param: int = 10
```

### New Sufficient Statistic

```python
MY_STATISTIC = SufficientStatistic(
    name="my_statistic",
    compute=lambda counts: ...  # Return (..., statistic_dim) array
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Model Components                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   Likelihoods   │  │  Guide Families │  │   Amortizers    │     │
│  │                 │  │                 │  │                 │     │
│  │ NegativeBinomial│  │ MeanFieldGuide  │  │ Sufficient-     │     │
│  │ ZeroInflatedNB  │  │ LowRankGuide    │  │   Statistic     │     │
│  │ NBWithVCP       │  │ AmortizedGuide  │  │ Amortizer       │     │
│  │ ZINBWithVCP     │  │ Grouped...      │  │   Network       │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
│                              ▼                                       │
│                     Used by Builders                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

# Model Factory

Unified factory for creating SCRIBE models.

## Factory Functions

| Function                   | Description                               |
|----------------------------|-------------------------------------------|
| `create_model(config)`     | Create model/guide from ModelConfig       |
| `create_model_from_params` | Convenience function with flat parameters |

## Supported Models

| Model Type | Description                            |
|------------|----------------------------------------|
| `nbdm`     | Negative Binomial Dropout Model        |
| `zinb`     | Zero-Inflated Negative Binomial        |
| `nbvcp`    | NB with Variable Capture Probability   |
| `zinbvcp`  | ZINB with Variable Capture Probability |

## Configuration Options

All models support:

```python
from scribe.models.presets import create_model_from_params
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import LowRankGuide, AmortizedGuide, MeanFieldGuide

model, guide = create_model_from_params(
    model="nbvcp",                        # Model type
    parameterization="canonical",         # "canonical", "mean_prob", "mean_odds"
    unconstrained=False,                  # Use Normal+transform instead of Beta/LogNormal
    guide_families=GuideFamilyConfig(
        mu=LowRankGuide(rank=10),         # or MeanFieldGuide() for mean-field
        p_capture=MeanFieldGuide(),       # or AmortizedGuide(amortizer=...)
    ),
    n_components=None,                    # int for mixture models
    mixture_params=None,                  # Which params are mixture-specific
    priors=None,                          # e.g. {"p": (1,1), "mu": (0,1)}
    guides=None,                          # Guide hyperparameters
)
```

### Parameterization Options

| Name          | Parameters | Derived                   |
|---------------|------------|---------------------------|
| `"canonical"` | p, r       | -                         |
| `"mean_prob"` | p, mu      | r = mu*(1-p)/p            |
| `"mean_odds"` | phi, mu    | r = mu*phi, p = 1/(1+phi) |

**Aliases**: `"standard"` = `"canonical"`, `"linked"` = `"mean_prob"`, `"odds_ratio"` = `"mean_odds"`

### Guide Families (via `GuideFamilyConfig`)

| Class                           | Description                                                  |
|---------------------------------|--------------------------------------------------------------|
| `MeanFieldGuide()`              | Independent variational distribution per parameter (default) |
| `LowRankGuide(rank=k)`          | Low-rank MVN capturing correlations between genes            |
| `AmortizedGuide(amortizer=...)` | Neural network predicts params from sufficient statistics    |

## Examples

### Basic Usage

```python
from scribe.models.presets import create_model_from_params

model, guide = create_model_from_params(model="nbdm")

# Use with SVI
svi = numpyro.infer.SVI(model, guide, optimizer, loss)
```

### Using ModelConfig

```python
from scribe.models.presets import create_model
from scribe.models.config import ModelConfigBuilder

config = (
    ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .as_mixture(n_components=3)
    .build()
)
model, guide = create_model(config)
```

### Linked Parameterization with Low-Rank Guide

```python
from scribe.models.presets import create_model_from_params
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import LowRankGuide

model, guide = create_model_from_params(
    model="nbdm",
    parameterization="linked",
    guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=15)),
)
```

### Amortized Capture Probability

For large datasets (100K+ cells), amortize p_capture inference. There are
multiple ways to enable this:

**Option 1: Using the factory function (recommended)**

```python
from scribe.models.presets import create_capture_amortizer, create_model_from_params
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import AmortizedGuide

# Factory handles constrained/unconstrained output params automatically
amortizer = create_capture_amortizer(
    hidden_dims=[64, 32],
    activation="leaky_relu",
    unconstrained=False,  # Outputs (alpha, beta) for Beta distribution
)
model, guide = create_model_from_params(
    model="nbvcp",
    guide_families=GuideFamilyConfig(
        p_capture=AmortizedGuide(amortizer=amortizer),
    ),
)
```

**Option 2: Using AmortizationConfig (integrates with Hydra)**

```python
from scribe.models.presets import create_model_from_params
from scribe.models.config import GuideFamilyConfig, AmortizationConfig

# Config-based approach - works with Hydra YAML files
model, guide = create_model_from_params(
    model="nbvcp",
    guide_families=GuideFamilyConfig(
        capture_amortization=AmortizationConfig(
            enabled=True,
            hidden_dims=[64, 32],
            activation="leaky_relu",
        ),
    ),
)
```

**Option 3: Using scribe.fit() directly (simplest)**

```python
import scribe

# Flat kwargs - no manual config construction needed
results = scribe.fit(
    adata,
    model="nbvcp",
    amortize_capture=True,
    capture_hidden_dims=[64, 32],
    capture_activation="leaky_relu",
)
```

**Option 4: Manual Amortizer construction**

```python
from scribe.models.presets import create_model_from_params
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import AmortizedGuide, TOTAL_COUNT, Amortizer

amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[64, 32],
    output_params=["log_alpha", "log_beta"],  # For constrained (Beta)
    # output_params=["loc", "log_scale"],     # For unconstrained (Normal)
)
model, guide = create_model_from_params(
    model="nbvcp",
    guide_families=GuideFamilyConfig(
        p_capture=AmortizedGuide(amortizer=amortizer),
    ),
)
```

### Custom Amortizer with Mixed Guide Families

```python
from scribe.models.presets import create_model_from_params
from scribe.models.config import GuideFamilyConfig
from scribe.models.components import Amortizer, TOTAL_COUNT, LowRankGuide, AmortizedGuide

amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[128, 64, 32],
    output_params=["log_alpha", "log_beta"],
)

model, guide = create_model_from_params(
    model="nbvcp",
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
model, guide = create_model_from_params(
    model="nbdm",
    parameterization="canonical",
    unconstrained=True,  # p ~ sigmoid(Normal), r ~ exp(Normal)
)
```

### Custom Priors

```python
model, guide = create_model_from_params(
    model="nbdm",
    priors={"p": (2.0, 2.0), "r": (1.0, 0.5)},  # Informative priors
)
```

## Model Equivalence

The unified factory produces the same model/guide as the legacy per-file implementations:

| Factory Call                                                                | Legacy equivalent           |
|-----------------------------------------------------------------------------|-----------------------------|
| `create_model_from_params(model="nbdm")`                                    | `standard.py`               |
| `create_model_from_params(model="nbdm", unconstrained=True)`                | `standard_unconstrained.py` |
| `create_model_from_params(model="nbdm", guide_families=...LowRankGuide...)` | `standard_low_rank.py`      |
| `create_model_from_params(model="nbdm", parameterization="linked")`         | `linked.py`                 |
| `create_model_from_params(model="nbdm", parameterization="odds_ratio")`     | `odds_ratio.py`             |

## Architecture

The unified factory uses registries and composable builders:

```
create_model(config)
    │
    ├── Look up MODEL_EXTRA_PARAMS[model_type]
    │   └── ["gate", "p_capture"] for zinbvcp, etc.
    │
    ├── Look up LIKELIHOOD_REGISTRY[model_type]
    │   └── ZINBWithVCPLikelihood, etc.
    │
    ├── Build param_specs via Parameterization strategy
    │   ├── Core params (p, r or p, mu or phi, mu)
    │   └── Extra params (gate, p_capture as needed)
    │
    ├── Apply prior/guide overrides
    │
    ├── ModelBuilder()
    │   ├── .add_param(specs)
    │   ├── .add_derived(...) if needed
    │   └── .with_likelihood(...)
    │
    └── GuideBuilder()
        └── .from_specs(specs)
```

## Registries

The factory uses two registries defined in `registry.py`:

```python
# Which extra parameters each model needs
MODEL_EXTRA_PARAMS = {
    "nbdm": [],
    "zinb": ["gate"],
    "nbvcp": ["p_capture"],
    "zinbvcp": ["gate", "p_capture"],
}

# Which likelihood class each model uses
LIKELIHOOD_REGISTRY = {
    "nbdm": NegativeBinomialLikelihood,
    "zinb": ZeroInflatedNBLikelihood,
    "nbvcp": NBWithVCPLikelihood,
    "zinbvcp": ZINBWithVCPLikelihood,
}
```

## Adding New Models

1. Add entry to `MODEL_EXTRA_PARAMS` in `registry.py`
2. Add entry to `LIKELIHOOD_REGISTRY` in `registry.py`
3. If needed, add helper builder for new parameters (like `build_gate_spec`)
4. Add tests verifying the new model works correctly

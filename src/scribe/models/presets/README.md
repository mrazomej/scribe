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
    output_transform="softplus",  # or "exp"; optional clamps: output_clamp_min, output_clamp_max
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
            output_transform="softplus",
            output_clamp_min=0.1,
            output_clamp_max=50.0,
        ),
    ),
)
```

**Option 3: Using scribe.fit() directly (simplest)**

You can pass a single config object or the six individual params (backward
compatible):

```python
import scribe
from scribe.models.config import AmortizationConfig

# Preferred: single config object (same object flows infer → fit → build_config)
results = scribe.fit(
    adata,
    model="nbvcp",
    capture_amortization=AmortizationConfig(
        enabled=True,
        hidden_dims=[64, 32],
        activation="leaky_relu",
        output_transform="softplus",
        output_clamp_min=0.1,
        output_clamp_max=50.0,
    ),
)

# Or: flat kwargs (backward compatible; ignored when capture_amortization is set)
results = scribe.fit(
    adata,
    model="nbvcp",
    amortize_capture=True,
    capture_hidden_dims=[64, 32],
    capture_activation="leaky_relu",
    capture_output_transform="softplus",
    capture_clamp_min=0.1,
    capture_clamp_max=50.0,
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
    output_params=["alpha", "beta"],           # For constrained (Beta)
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
    output_params=["alpha", "beta"],
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

## Sparsity-Inducing Prior Factory Functions

**Horseshoe:** Gene-level: `_horseshoe_p`, `_horseshoe_gate`, `_horseshoe_mu`.
Dataset-level: `_horseshoe_dataset_mu`, `_horseshoe_dataset_p`,
`_horseshoe_dataset_gate`. Helper `_make_horseshoe_hypers` creates
HalfCauchy/InverseGamma specs; `_horseshoe_kwargs_from_config` extracts
horseshoe config from ModelConfig.

- `_horseshoe_mu(param_specs, param_key, tau0, slab_df, slab_scale)`: Upgrades
  gene-level hierarchical mu/r to horseshoe. Replaces `log_mu_scale`
  (SoftplusNormalSpec) with horseshoe trio, and `HierarchicalExpNormalSpec`
  with `HorseshoeHierarchicalExpNormalSpec`. Dispatched when
  `model_config.mu_prior == HORSESHOE`.

**NEG:** Gene-level: `_neg_p`, `_neg_gate`, `_neg_mu`. Dataset-level:
`_neg_dataset_mu`, `_neg_dataset_p`, `_neg_dataset_gate`. Helper
`_make_neg_hypers` creates GammaSpec pair (zeta, psi); `_neg_kwargs_from_config`
extracts NEG config from ModelConfig.

- `_neg_mu(param_specs, param_key, u, a, tau)`: Upgrades gene-level
  hierarchical mu/r to NEG. Replaces `log_mu_scale` with zeta/psi pair, and
  `HierarchicalExpNormalSpec` with `NEGHierarchicalExpNormalSpec`. Dispatched
  when `model_config.mu_prior == NEG`.

Both `_horseshoe_mu` and `_neg_mu` are applied only when
`model_config.mu_prior != _NONE` (i.e., when gene-level hierarchical mu is
enabled).

## Multi-Dataset Mixture Hierarchy

When combining mixture models with dataset-level hierarchy (`hierarchical_dataset_mu`
or `hierarchical_dataset_p`), `_datasetify_mu()` and `_datasetify_p()` propagate
`is_mixture=True` to `hyper_loc` specs when the original parameter is
mixture-aware. Each component then gets its own population expression profile
(shape `(K, G)` instead of `(G,)`). Both functions accept a `shared_component_indices`
parameter, passed through to hierarchical specs for scale masking (components
shared across 2+ datasets use learned cross-dataset scale; others use clamped
near-zero scale). The factory reads `model_config.shared_component_indices`,
populated at runtime by `fit()` when `annotation_key` and `dataset_key` are both
provided, and threads this through to the datasetify helpers.

### Dataset-Level Gate (`gate_dataset_prior`)

Unlike mu/p, `_datasetify_gate()` produces **independent** per-dataset gates
pushed toward zero, not a pooling hierarchy. The population location
`logit_gate_dataset_loc` is a **scalar** `N(-5, 1)` shared across all genes,
datasets, and components. This scalar is robust against being overwhelmed by
the likelihood (unlike the per-gene loc used by mu/p). Per-gene adaptive
shrinkage comes from the NEG/horseshoe `psi_g` (or `lambda_g`), and
per-dataset independence comes from the NCP `z_{g,d}` variable:

```
gate_g^(d) = sigmoid(loc + sqrt(psi_g) * z_{g,d})
```

When gate is mixture-aware (`gate in mixture_params`), the shape becomes
`(K, D, G)`, giving each component its own independent gate per dataset.

## Adding New Models

1. Add entry to `MODEL_EXTRA_PARAMS` in `registry.py`
2. Add entry to `LIKELIHOOD_REGISTRY` in `registry.py`
3. If needed, add helper builder for new parameters (like `build_gate_spec`)
4. Add tests verifying the new model works correctly

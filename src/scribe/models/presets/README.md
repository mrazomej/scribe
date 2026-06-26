# Model Factory

Unified factory for creating SCRIBE models.

## Factory Functions

| Function                   | Description                               |
|----------------------------|-------------------------------------------|
| `create_model(config)`     | Create model/guide from ModelConfig       |
| `create_model_from_params` | Convenience function with flat parameters |

## Supported Models

| Model Type | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| `nbdm`     | Negative Binomial Dropout Model                                                             |
| `zinb`     | Zero-Inflated Negative Binomial                                                             |
| `nbvcp`    | NB with Variable Capture Probability                                                        |
| `zinbvcp`  | ZINB with Variable Capture Probability                                                      |
| `lnm`      | Logistic-Normal Multinomial (NB total × multinomial composition via linear-decoder VAE)     |
| `lnmvcp`   | LNM with per-cell variable capture probability on the totals NB submodel                    |
| `pln`      | Poisson-LogNormal (per-gene Poisson on correlated log-normal rates via linear-decoder VAE)  |
| `nbln`     | NB-LogNormal (per-gene NB on log-normal-modulated means; adds gene dispersion `r_g` global) |

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
    mixture_params="all",                 # "all", "biological", "mean", "prob",
                                          # "gate", or explicit list
    priors=None,                          # e.g. {"p": (1,1), "mu": (0,1)}
    guides=None,                          # Guide hyperparameters
)
```

For multi-dataset mixture models, the factory now supports dataset-specific
mixing weights. By default this is enabled when `n_datasets >= 2` via
`ModelConfig.dataset_mixing_enabled`; set `dataset_mixing=False` in config/API
to keep one global mixing vector shared by all datasets.

### Parameterization Options

| Name                | Parameters | Derived                                                                      |
| ------------------- | ---------- | ---------------------------------------------------------------------------- |
| `"canonical"`       | p, r       | -                                                                            |
| `"mean_prob"`       | p, mu      | r = mu*(1-p)/p                                                               |
| `"mean_odds"`       | phi, mu    | r = mu*phi, p = 1/(1+phi)                                                    |
| `"mean_disp"`       | mu, r (both gene-specific) | phi = r/mu, p = mu/(mu+r) — Fisher-orthogonal coords; SVI/MCMC only |
| `"logistic_normal"` | r_T, p | y_alr (via VAE decoder, reference gene configurable via `alr_reference_idx`) |

**Aliases**: `"standard"` = `"canonical"`, `"linked"` = `"mean_prob"`, `"odds_ratio"` = `"mean_odds"`

**`mean_disp` notes**: samples the mean `mu` and dispersion `r` directly (the
orthogonal coordinate; `r` is gene-specific by construction). The factory
targets the `expression_prior` hierarchy at `mu` and **rejects** `prob_prior` /
`prob_dataset_prior` (no scalar success-probability to hierarchicalize). VAE is
rejected (two gene-specific primaries can't share the single-head decoder).

**LNM-only stability defaults** (active when `model in {"lnm", "lnmvcp"}`):

- `vae_input_transform` defaults to `"log1p_prop"` (compositional input).
- `vae_standardize` defaults to `True`; per-feature stats are computed
  from the count matrix in the *same* input-transform space the
  encoder uses, via
  `scribe.core.lnm_data_init.compute_encoder_standardization`.
- The `r_T` LogNormal prior is auto-initialized from the empirical
  total-count moments (method-of-moments NB inversion). User-provided
  `priors={"r_T": ...}` always wins.
- The linear-decoder `y_alr` head bias is anchored to the empirical
  ALR mean of the counts via `DecoderOutputHead.bias_init`, computed
  by `scribe.core.normalization_logistic.empirical_alr_mean_from_counts`.
- The Gaussian encoder is replaced by `LNMGaussianEncoder`, which
  clamps the log-scale head to `[-7, 2]` to short-circuit `σ → 0`/`∞`
  pathologies under the multinomial likelihood.

These changes are gated on the LNM family; non-LNM VAE models are
bit-identical to pre-stability behavior. See the qmd section
"Training stability: practical considerations" in
`paper/_logistic_normal_multinomial.qmd` for the rationale.

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
    unconstrained=True,  # p ~ sigmoid(Normal), r ~ positive transform(Normal)
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
    "lnm": [],
    "lnmvcp": ["p_capture"],
}

# Which likelihood class each model uses
LIKELIHOOD_REGISTRY = {
    "nbdm": NegativeBinomialLikelihood,
    "zinb": ZeroInflatedNBLikelihood,
    "nbvcp": NBWithVCPLikelihood,
    "zinbvcp": ZINBWithVCPLikelihood,
    "lnm": LogisticNormalMultinomialLikelihood,
    "lnmvcp": LNMWithVCPLikelihood,
}
```

## Hierarchical-Prior Cores (descriptor-driven)

Every hierarchical prior — gene-level shrinkage across components and
dataset-level shrinkage across datasets, for `mu`/`r`, `p`/`phi`, `gate`, and
the two-state regime coordinate — is built by **three generic cores** in
`factory.py`, parametrized by a `HierParam` descriptor
(`builders/hier_descriptors.py`). The descriptor declares the exact site names,
spec classes, and construction geometry for a given `(role, param_key)` at a
given level, so the cores carry no per-parameter `if` ladders.

- **`_gaussianize(param_specs, desc, *, guide_families, mode=None, ...)`** —
  builds the Gaussian hierarchy triplet `[hyper_loc, hyper_scale, hier_spec]`.
  Replaces the former `_hierarchicalize_{mu,p}` (gene), `_datasetify_{mu,p,gate}`
  (dataset) and `_datasetify_regime`. The only conditional is regime's
  `inherit_pop_loc_from_flat` (its `hyper_loc` inherits the flat regime prior).
- **`_horseshoe_ncp(param_specs, desc, tau0, slab_df, slab_scale)`** — upgrades a
  Gaussian hierarchy to a regularized horseshoe (replaces `desc.scale` with the
  τ/λ/c² trio and `desc.hier_cls` with `desc.hs_cls`). Replaces the former
  `_horseshoe_{mu,p,gate}` (gene), `_horseshoe_dataset_{mu,p,gate}` and
  `_horseshoe_dataset_regime`. Helper `_make_horseshoe_hypers` creates the
  HalfCauchy/InverseGamma specs; `_horseshoe_kwargs_from_config` reads the
  hyperparameters from `ModelConfig`.
- **`_neg_ncp(param_specs, desc, u, a, tau)`** — NEG counterpart of
  `_horseshoe_ncp` (ζ/ψ Gamma-Gamma pair via `_make_neg_hypers`,
  `desc.neg_cls`). Replaces the former `_neg_{mu,p,gate}` (gene),
  `_neg_dataset_{mu,p,gate}` and `_neg_dataset_regime`.

Descriptors are produced by `gene_hier_param(role, param_key)`,
`dataset_hier_param(role, param_key)`, and
`regime_dataset_hier_param(parameterization, target_override)`. The NCP cores
thread `is_dataset=spec.is_dataset` uniformly, so one implementation serves both
the gene-level (`Hierarchical*` specs) and dataset-level (`DatasetHierarchical*`
specs) families. The horseshoe/NEG upgrades run only when the corresponding
`*_prior == HORSESHOE`/`NEG`.

## Multi-Dataset Mixture Hierarchy

When combining mixture models with dataset-level hierarchy (`hierarchical_dataset_mu`
or `hierarchical_dataset_p`), `_gaussianize()` (driven by
`dataset_hier_param("expression"|"prob", ...)`) propagates `is_mixture=True` to
`hyper_loc` specs when the original parameter is mixture-aware (the descriptor's
`pop_loc_inherits_mixture`). Each component then gets its own population
expression profile (shape `(K, G)` instead of `(G,)`). The core accepts a
`shared_component_indices` parameter, threaded into hierarchical specs for scale
masking when `desc.threads_shared_components` (components shared across 2+
datasets use learned cross-dataset scale; others use clamped near-zero scale).
The factory reads `model_config.shared_component_indices`, populated at runtime
by `fit()` when `annotation_key` and `dataset_key` are both provided.

### Dataset-Level Gate (`zero_inflation_dataset_prior`)

Unlike mu/p, the gate dataset hierarchy (`_gaussianize` with
`dataset_hier_param("gate", ...)`) produces **independent** per-dataset gates
pushed toward zero, not a pooling hierarchy. The population location
`logit_gate_dataset_loc` is a **scalar** `N(-5, 0.01)` shared across all genes,
datasets, and components. The very tight variance (0.01) anchors logit(gate)
deep in the off region so the aggregate likelihood cannot drag it positive.
Per-gene adaptive shrinkage comes from the NEG/horseshoe `psi_g` (or
`lambda_g`) with a **Gamma** variational posterior (mode at zero when
concentration < 1), and per-dataset independence comes from the NCP
`z_{g,d}` variable:

```
gate_g^(d) = sigmoid(loc + sqrt(psi_g) * z_{g,d})
```

When gate is mixture-aware (`gate in mixture_params`), the shape becomes
`(K, D, G)`, giving each component its own independent gate per dataset.

### Two-State Dataset-Level Regime + Free Overdispersion

For `twostate`/`twostatevcp`, the regime/overdispersion coordinates are
*extras* (built in Step 5), so their dataset passes run in **Step 5.6**, after
the extras exist (unlike `mu`/`p`, handled in Step 4.6). `TWOSTATE_REGIME_COORD`
and `TWOSTATE_OVERDISPERSION_COORD` (`config.enums`) map each parameterization
to its coordinates.

- `_datasetify_regime()` replaces the regime coordinate
  (`k_off`/`switching_ratio`/`concentration`/`inv_concentration`) with a
  dataset-hierarchical triplet, choosing **sigmoid** specs + `logit_*` hyper
  names for `inv_concentration` (support `(0, 1)`) and **positive** specs +
  `log_*` names otherwise. The population `hyper_loc` inherits the flat regime
  prior so the "default to NB" tilt carries through.
- `_horseshoe_dataset_regime()` / `_neg_dataset_regime()` upgrade that triplet
  to horseshoe / NEG, mirroring `_horseshoe_dataset_p` / `_neg_dataset_p` with
  the same sigmoid-vs-positive selection. Triggered by
  `regime_dataset_prior == HORSESHOE` / `NEG`.
- `_datasetify_overdispersion_independent()` marks the overdispersion
  coordinate (`burst_size`/`excess_fano`) `is_dataset=True` with **no**
  hyperprior — a free per-`(dataset, gene)` value. Applied when
  `overdispersion_dataset_independent=True` (default for multi-dataset
  two-state).

Note: `_datasetify_mu` / `_horseshoe_dataset_mu` / `_neg_dataset_mu` target
`mu` (not `r`) for two-state parameterizations as well as the NB mean
parameterizations, via the `_expression_target_is_mu()` helper.

## Adding New Models

1. Add entry to `MODEL_EXTRA_PARAMS` in `registry.py`
2. Add entry to `LIKELIHOOD_REGISTRY` in `registry.py`
3. If needed, add helper builder for new parameters (like `build_gate_spec`)
4. Add tests verifying the new model works correctly

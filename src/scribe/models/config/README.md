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
- `.with_hierarchical_mu()`: Set `expression_prior="gaussian"` for gene-level mu
  (or r) shrinkage across mixture components (requires unconstrained, mixture
  model). For horseshoe/neg, pass `expression_prior` when constructing
  ModelConfig directly.
- `.with_hierarchical_p()`: Enable gene-specific p/phi hierarchical prior
  (requires unconstrained)
- `.with_hierarchical_gate()`: Enable gene-specific gate hierarchical prior (ZI
  models only, requires unconstrained)
- `.with_capture_priors(organism, eta_capture, mu_eta, capture_scaling_prior)`:
  Configure biology-informed capture prior (VCP models)
- `.as_mixture(n_components, mixture_params)`: Configure as mixture.
  `mixture_params` accepts shorthands (`"all"`, `"biological"`, `"mean"`,
  `"prob"`, `"gate"`) or an explicit list
- `.with_guide_families(guide_families)`: Set per-parameter guide families
- `.with_joint_params(joint_params)`: Specify parameters to model jointly via
  JointLowRankGuide. Accepts same shorthands as `mixture_params`
- `.with_dense_params(dense_params)`: Subset of `joint_params` for full
  cross-gene coupling (structured joint guide); non-dense params stay gene-local.
  Accepts same shorthands
- `.with_priors(**priors)`: Set prior parameters
- `.with_guides(**guides)`: Set guide parameters
- `.with_vae(**vae_params)`: Configure VAE parameters
- `.build()`: Create and validate the configuration

### ModelConfig

Unified configuration class for all SCRIBE models. Supports both constrained and
unconstrained parameterizations via the `unconstrained` boolean field. Prior and
guide hyperparameters are stored in `param_specs` (list of `ParamSpec`); see
`scribe.models.builders.parameter_specs`.

#### Gene-Level Prior Fields

- `prob_prior: HierarchicalPriorType` — Gene-level prior for p/phi
- `zero_inflation_prior: HierarchicalPriorType` — Gene-level prior for gate (ZI
  models)
- `expression_prior: HierarchicalPriorType` — Gene-level prior for mu (or r)
  across mixture components (replaces deprecated `hierarchical_mu: bool`)

#### Multi-Dataset Hierarchical Model

For joint multi-dataset modeling, the following fields control dataset-level
structure:

- `n_datasets: Optional[int]` — Number of datasets for joint multi-dataset
  modeling
- `dataset_params: Optional[List[str]]` — Which parameters are per-dataset
- `dataset_mixing: Optional[bool]` — Whether mixture weights are per-dataset.
  `None` enables dataset-specific mixing automatically when `n_datasets >= 2`
  and keeps global mixing otherwise.
- `expression_dataset_prior: str` — Prior for hierarchical mu/r across datasets:
  `"none"`, `"gaussian"`, `"horseshoe"`, or `"neg"` (requires `unconstrained`)
- `prob_dataset_prior: str` — Prior for dataset-specific p: `"none"`,
  `"gaussian"`, `"horseshoe"`, or `"neg"`
- `prob_dataset_mode: str` — Structural mode for dataset-level p:
  `"none"`, `"scalar"`, `"gene_specific"`, or `"two_level"`
- `zero_inflation_dataset_prior: str` — Prior for dataset-specific gate:
  `"none"`, `"gaussian"`, `"horseshoe"`, or `"neg"`.  Unlike
  `expression_dataset_prior` and `prob_dataset_prior`, this does **not** pool
  gates toward a shared per-gene mean.  Instead, each (dataset, gene) gate is
  independently shrunk toward zero via a scalar population location `N(-5,
  0.01)` (very tight so the likelihood cannot drag it positive) with per-gene
  adaptive shrinkage from the chosen sparsity prior.  NEG auxiliary variables
  (psi, zeta) use a Gamma variational posterior that can concentrate at zero
- `overdispersion_dataset_prior: str` — Prior for dataset-specific BNB
  concentration (`bnb_concentration`, i.e. `kappa_{d,g}`): `"none"`,
  `"gaussian"`, `"horseshoe"`, or `"neg"`. Requires
  `overdispersion="bnb"` and `unconstrained=True`.
- `is_multi_dataset` (computed property) — `True` when `n_datasets >= 2`
- `dataset_mixing_enabled` (computed property) — Effective switch used by model
  builders to decide whether `mixing_weights` have shape `(K,)` or `(D, K)`.

Dataset-level prior fields (`expression_dataset_prior`, `prob_dataset_prior`,
`zero_inflation_dataset_prior`) are only valid when cells can be mapped to
datasets. In practice, that means passing `dataset_key` to `scribe.fit(...)` (so
`n_datasets` is inferred from `adata.obs[dataset_key]`) or otherwise configuring
explicit multi-dataset mode. Single-dataset fits should use `prob_prior`,
`zero_inflation_prior`, and/or `expression_prior` instead.

When using the public `scribe.fit(...)` API, single-dataset `dataset_key`
columns can be auto-downgraded via
`auto_downgrade_single_dataset_hierarchy=True` (default):

- `expression_dataset_prior != "none"` -> `"none"`
- `prob_dataset_prior != "none"` with `prob_dataset_mode='scalar'` ->
  `prob_dataset_prior='none'`
- `prob_dataset_prior != "none"` with `prob_dataset_mode in {'gene_specific',
  'two_level'}` -> `prob_prior` set from `prob_dataset_prior`, `prob_dataset_prior='none'`
- `zero_inflation_dataset_prior != "none"` -> `zero_inflation_prior` set from
  `zero_inflation_dataset_prior`, `zero_inflation_dataset_prior='none'`
- `overdispersion_dataset_prior != "none"` -> `overdispersion_dataset_prior='none'`

`scribe.fit(...)` emits a `UserWarning` whenever one or more of these
single-dataset downgrades are applied. Setting
`auto_downgrade_single_dataset_hierarchy=False` preserves strict validation.

#### Joint Low-Rank Parameters

For SVI with joint modeling of gene-specific parameters via a low-rank
factorization:

- `joint_params: Optional[List[str]]` — List of gene-specific parameter names
  to model jointly via JointLowRankGuide.
- `dense_params: Optional[List[str]]` — Subset of `joint_params` that receive
  full cross-gene low-rank coupling. Non-dense params get per-gene regression
  on dense params plus per-gene Cholesky among themselves. When `None` or equal
  to `joint_params`, standard JointLowRankGuide is used.

#### Biology-Informed Capture Prior

For VCP models (`nbvcp`, `zinbvcp`), the capture probability can be anchored to
biological knowledge about total cellular mRNA content. The biology-informed
path activates automatically when any of `priors.organism`,
`priors.eta_capture`, or `priors.mu_eta` is set.

| `priors.eta_capture` | `capture_scaling_prior` | Behavior |
|----------------------|-------------------------|----------|
| not set | `"none"` (or omitted) | Standard flat prior (no eta framework) |
| set (directly or via `priors.organism`) | `"none"` | Fixed M_0, no shared parameter |
| set | `"gaussian"`, `"horseshoe"`, or `"neg"` | Learn **per-dataset** `mu_eta` via hierarchical NCP prior, shrunk toward a shared population mean `mu_eta_pop`. `priors.mu_eta` controls `[center, sigma_mu]`. |

- `priors.organism: str` — Shortcut to resolve default `eta_capture` and
  `mu_eta` values: `"human"`, `"mouse"`, `"yeast"`, `"ecoli"` (and aliases).
- `priors.eta_capture: (float, float)` — `(log_M0, sigma_M)` for the per-cell
  TruncatedNormal+ prior on `eta_c`. Overrides organism defaults.
- `priors.mu_eta: (float, float)` — `(center, sigma_mu)` for the population-
  level `mu_eta_pop` Normal prior. When not set but `capture_scaling_prior !=
  "none"`, defaults to `(eta_capture[0], 1.0)`.
- `capture_scaling_prior: str` — Hierarchical prior type for per-dataset
  `mu_eta`: `"none"`, `"gaussian"`, `"horseshoe"`, or `"neg"` (uses
  `HierarchicalPriorType`). When not `"none"`, per-dataset `mu_eta` values are
  learned via a non-centered parameterization, shrunk toward a shared
  `mu_eta_pop` by the chosen shrinkage prior. For single-dataset runs, a scalar
  `mu_eta` fallback is used automatically.

The biology-informed prior samples a latent variable
`eta_c ~ TruncatedNormal+(log M_0 - log L_c, sigma_M^2, low=0)` and derives
capture parameters via exact transformations (`p_capture = exp(-eta)`,
`phi_capture = exp(eta) - 1`). See `paper/_capture_prior.qmd` for derivations.

```python
config = (ModelConfigBuilder()
    .for_model("nbvcp")
    .with_parameterization("mean_odds")
    .with_capture_priors(organism="human")
    .build())
```

#### Data-Informed Mean Anchoring Prior

The mean anchoring prior resolves the mu-phi degeneracy in the negative binomial
by anchoring the per-gene biological mean `mu_g` to the observed sample mean
scaled by the average capture probability. This is an empirical Bayes approach
analogous to DESeq2's dispersion shrinkage.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `expression_anchor` | `bool` | `False` | Enable data-informed anchoring prior on `mu_g` |
| `expression_anchor_sigma` | `float` | `0.3` | Log-scale sigma (tightness: 0.1=tight, 0.3=moderate, 1.0=weak) |

When enabled, per-gene prior centers are computed at fit time from the data:
`log(mu_g) ~ N(log(u_bar_g / nu_bar), sigma^2)`. Requires `unconstrained=True`.
For VCP models (`nbvcp`, `zinbvcp`), `priors.eta_capture` or `priors.organism`
must also be set so that `nu_bar` (average capture probability) can be
estimated; without it, the anchor would silently use `nu_bar=1` and give
incorrect `mu_g` values. Non-VCP models (`nbdm`, `zinb`) correctly use
`nu_bar=1` by default.

```python
# Via builder
config = (ModelConfigBuilder()
    .for_model("nbvcp")
    .with_parameterization("mean_odds")
    .with_capture_priors(organism="human")
    .with_mean_anchor(sigma=0.3)
    .with_hierarchical_p()
    .build())

# Via fit()
results = scribe.fit(
    adata, model="nbvcp", parameterization="mean_odds",
    expression_anchor=True, expression_anchor_sigma=0.3,
    prob_prior="gaussian",
    priors={"organism": "human"},
)
```

See `paper/_mean_anchoring_prior.qmd` for the full mathematical derivation,
including the concentration-of-measure argument and mixture model extensions.

#### Sparsity-Inducing Prior Configuration

Sparsity-inducing priors are configured via enum fields. Each parameter has a
prior type: `"none"`, `"gaussian"`, `"horseshoe"`, or `"neg"`.

**Enum fields:**
- `prob_prior`, `zero_inflation_prior`, `expression_prior` — Gene-level priors
  for p, gate (ZI models), and mu across mixture components. All accept the same
  `HierarchicalPriorType` values: `"none"`, `"gaussian"`, `"horseshoe"`,
  `"neg"`. `expression_prior` requires a mixture model (`n_components >= 2`) and
  `unconstrained=True`; each gene has its own hyperprior because expression
  magnitudes vary by orders of magnitude. `expression_prior` and
  `expression_dataset_prior` are mutually exclusive (component-level vs
  dataset-level shrinkage).
- `expression_dataset_prior`, `prob_dataset_prior`,
  `zero_inflation_dataset_prior`, `overdispersion_dataset_prior` — Dataset-level
  priors for multi-dataset hierarchical models
- `prob_dataset_mode` — Structural mode for dataset-level p: `"none"`,
  `"scalar"`, `"gene_specific"`, or `"two_level"`

**NEG prior** (Normal-Exponential-Gamma): A Gamma-Gamma hierarchy friendlier to
SVI than the horseshoe. Hyperparameters (defaults all 1.0):
- `neg_u` — Shape parameter (u=1 => NEG; u=0.5 => horseshoe-like)
- `neg_a` — Outer Gamma shape
- `neg_tau` — Global rate

**Horseshoe prior**: Regularized horseshoe shrinkage. Hyperparameters:
- `horseshoe_tau0` — Global scale
- `horseshoe_slab_df` — Slab degrees of freedom
- `horseshoe_slab_scale` — Slab scale

Boolean flags like `horseshoe_p` and `hierarchical_mu` are deprecated; use the
enum-based fields instead. The deprecated `hierarchical_mu` property is derived
from `expression_prior` (returns `expression_prior != "none"`).

### PriorConfig / UnconstrainedPriorConfig

Prior parameter configurations with automatic validation:

- `p`: Success probability prior
- `r`: Dispersion prior
- `mu`: Mean prior
- `phi`: Odds ratio prior
- `gate`: Zero-inflation gate prior
- `p_capture`: Capture probability prior
- `phi_capture`: Capture phi prior
- `mixing`: Mixture weights prior (Dirichlet concentrations). Accepts a tuple of
  length `n_components` or a single scalar that is broadcast to all components
  (symmetric Dirichlet).

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
- `optimizer_config`: Serializable optimizer specification (name + kwargs),
  recommended for API/Hydra usage. Example:
  `{"name": "clipped_adam", "step_size": 5e-4, "grad_clip_norm": 1.0}`
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
    optimizer_config={"name": "adam", "step_size": 1e-3},
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

Visualization notes for MCMC runs:

- Posterior samples are already available after inference, so downstream
  visualization should reuse these draws directly instead of re-sampling a
  variational posterior.
- Plot filename run-size tokens are based on
  `"{n_samples}samples_{n_warmup}warmup"` (rather than SVI `"n_steps"` tokens).
- The visualization `loss` panel is replaced by an MCMC diagnostics panel
  (potential energy, divergences, and a representative trace view).

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
- `Parameterization`: CANONICAL, MEAN_PROB, MEAN_ODDS (plus backward-compat
  aliases STANDARD, LINKED, ODDS_RATIO)
- `InferenceMethod`: SVI, MCMC, VAE
- `VAEPriorType`: STANDARD, DECOUPLED
- `VAEMaskType`: ALTERNATING, SEQUENTIAL
- `VAEActivation`: RELU, GELU, TANH, SIGMOID
- `HierarchicalPriorType`: NONE, GAUSSIAN, HORSESHOE, NEG

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

### Hierarchical Configuration

```python
# Hierarchical model with gene-specific p_g (requires unconstrained)
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_parameterization("canonical")
    .unconstrained()
    .with_hierarchical_p()
    .build())

# Hierarchical mean-odds with gene-specific gate (ZI models)
config = (ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("mean_odds")
    .unconstrained()
    .with_hierarchical_p()
    .with_hierarchical_gate()
    .build())

# Hierarchical mu across mixture components — shrinks per-component means
# toward a shared gene-level baseline (requires mixture model, unconstrained)
config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_parameterization("mean_prob")
    .unconstrained()
    .as_mixture(3)
    .with_hierarchical_mu()  # sets expression_prior="gaussian"
    .build())

# Or set expression_prior directly for gaussian/horseshoe/neg:
config = ModelConfig(
    base_model="nbdm",
    unconstrained=True,
    parameterization="mean_prob",
    n_components=3,
    expression_prior="gaussian",  # or "horseshoe", "neg"
)

# NEG prior for SVI-friendly sparsity (alternative to horseshoe)
config = ModelConfig(
    base_model="nbdm",
    unconstrained=True,
    prob_prior="neg",
    parameterization="mean_odds",
    neg_u=1.0,  # shape param (u=1 => NEG; u=0.5 => horseshoe-like)
    neg_a=1.0,  # outer Gamma shape
    neg_tau=1.0,  # global rate
)
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

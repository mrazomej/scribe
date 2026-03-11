# Model Builders

Composable builders for constructing NumPyro models and guides.

## Architecture

```
ParamSpec (defines WHAT to sample)
    │
    ▼
ModelBuilder / GuideBuilder (composes HOW to sample)
    │
    ▼
model_fn / guide_fn (NumPyro callables)
```

## Parameter Specifications

Parameter specs define the distribution and metadata for each parameter:

| Spec Type | Distribution | Constraint | Example Params |
|-----------|--------------|------------|----------------|
| `BetaSpec` | Beta(α, β) | (0, 1) | p, gate, p_capture |
| `LogNormalSpec` | LogNormal(μ, σ) | (0, ∞) | r, mu |
| `BetaPrimeSpec` | BetaPrime(α, β) | (0, ∞) | phi |
| `DirichletSpec` | Dirichlet(α) | simplex | weights |
| `SigmoidNormalSpec` | Normal → sigmoid | (0, 1) | p_unconstrained |
| `ExpNormalSpec` | Normal → exp | (0, ∞) | r_unconstrained |
| `SoftplusNormalSpec` | Normal → softplus | (0, ∞) | r (smooth) |
| `BiologyInformedCaptureSpec` | Normal(η) → exp/exp-1 | cell-specific | p_capture, phi_capture |
| `LatentSpec` | Base for VAE latent z | — | abstract |
| `GaussianLatentSpec` | Normal(loc, scale).to_event(1) from encoder output | — | z (guide only) |

### Hierarchical Parameter Specs

`HierarchicalSigmoidNormalSpec` and `HierarchicalExpNormalSpec` inherit from
`HierarchicalNormalWithTransformSpec`, which extends `NormalWithTransformSpec`.
They define gene-specific parameters whose Normal prior has learnable location
and scale drawn from hyperparameters sampled earlier in the model.

Each hierarchical spec carries `hyper_loc_name` and `hyper_scale_name` fields
identifying which sample sites provide its prior parameters. The model builder
detects these specs via `isinstance` and calls `spec.sample_hierarchical(dims,
param_values)` instead of the standard `sample_prior` dispatch. Because they
inherit from `NormalWithTransformSpec`, all existing guide dispatch (mean-field,
low-rank) works without modification.

#### Dataset-Level Hierarchical Specs

For multi-dataset models, **`DatasetHierarchicalNormalWithTransformSpec`** is
the base class for per-dataset parameters drawn from population-level
hyperparameters. Subclasses: **`DatasetHierarchicalExpNormalSpec`** (positive
params: mu, r, phi; exp transform) and
**`DatasetHierarchicalSigmoidNormalSpec`** ((0,1) params: p, gate; sigmoid
transform). The **`ParamSpec`** base class adds an **`is_dataset`** flag; when
True, `resolve_shape` prepends the `n_datasets` dimension to the parameter
shape. The model builder calls **`sample_hierarchical()`** on these specs, which
draws per-dataset parameters from the population-level loc/scale hyperparameters
already in `param_values`.

#### Horseshoe Prior Specs

Regularized horseshoe shrinkage uses **`HalfCauchySpec`** (τ, λ scales) and
**`InverseGammaSpec`** (slab c²). Gene-level:
**`HorseshoeHierarchicalSigmoidNormalSpec`** (p, gate; sigmoid). Dataset-level:
**`HorseshoeDatasetExpNormalSpec`** (mu; exp),
**`HorseshoeDatasetSigmoidNormalSpec`** (p, gate; sigmoid). All use NCP
(non-centered parameterization) with z ~ Normal(0,1).

#### Biology-Informed Capture Spec

**`BiologyInformedCaptureSpec`** represents the biology-informed capture
probability prior for VCP models. Instead of using a flat prior on `p_capture`
or `phi_capture`, it samples a latent variable `eta_c` from a TruncatedNormal
(low=0) prior anchored to observed library sizes and the expected total mRNA
count (`M_0`). The truncation at zero enforces the physical constraint
`M_c >= L_c` (equivalently, `p_capture <= 1`). The capture parameter is then
derived via exact transformations.

Fields: `log_M0` (log expected total mRNA), `sigma_M` (log-scale std-dev),
`data_driven` (if True, `log_M0` is replaced by a learned shared parameter
`mu_eta`; set by `shared_capture_scaling` config flag), `sigma_mu` (prior
std-dev for `mu_eta`), `use_phi_capture` (selects transformation:
`phi_capture = exp(eta) - 1` vs `p_capture = exp(-eta)`).

The guide dispatch for this spec samples per-cell `eta_capture` variational
parameters and (for data-driven mode) the shared `mu_eta` before the cell plate.

### ParamSpec Attributes

Each spec has the following attributes:

- `name`: Parameter name (used as sample site)
- `shape_dims`: Symbolic dimensions like `()`, `("n_genes",)`, `("n_cells",)`
- `default_params`: Default distribution parameters
- `is_gene_specific`: If True, shape is (n_genes,)
- `is_cell_specific`: If True, sampled inside cell plate
- `is_dataset`: If True, shape expands to include n_datasets dimension
- `guide_family`: Variational family for this parameter
- `support`: Derived from distribution/transform (property)
- `arg_constraints`: Constraints on the distribution's parameters

### LatentSpec (VAE latent z)

For VAE-style models, the **latent variable z** is not a ParamSpec; it uses a
**LatentSpec** that turns encoder output (a params dict) into the guide
distribution. This matches the amortizer pattern: network returns raw params,
spec builds the Distribution.

- **`LatentSpec`** (BaseModel): Base class with `sample_site` (default `"z"`)
  and abstract **`make_guide_dist(var_params)`**. Subclasses implement the
  mapping from encoder output to a NumPyro distribution.
- **`GaussianLatentSpec`**: Expects `var_params` with keys `"loc"` and
  `"log_scale"` (log-variance convention). Returns
  `Normal(loc, exp(0.5*log_scale)).to_event(1)`.

The guide builder uses this when a cell-specific spec has
`VAELatentGuide` with `encoder` and `latent_spec` set: it runs the
encoder, builds `var_params = {"loc": loc, "log_scale": log_scale}`,
calls `latent_spec.make_guide_dist(var_params)`, and samples
`numpyro.sample(latent_spec.sample_site, guide_dist)`. Decoder-driven
parameter names come from `decoder.output_heads` (`VAELatentGuide.param_names`).

```python
from scribe.models.builders import GaussianLatentSpec

latent_spec = GaussianLatentSpec(latent_dim=10, sample_site="z")
# In guide: loc, log_scale = encoder(...); var_params = {"loc": loc, "log_scale": log_scale}
# guide_dist = latent_spec.make_guide_dist(var_params); z = sample("z", guide_dist)
```

## Guide Families

Each parameter can have its own guide family:

| Family | Description | Use Case |
|--------|-------------|----------|
| `MeanFieldGuide` | Factorized variational family | Default, fast |
| `LowRankGuide(rank)` | Low-rank MVN covariance | Gene correlations |
| `JointLowRankGuide(rank, group)` | Joint low-rank MVN across parameter groups | Cross-parameter correlations |
| `AmortizedGuide(net)` | Neural network amortization | High-dim params |
| `VAELatentGuide` | VAE: encoder + latent_spec (z) + decoder | Joint latent z; decoder-driven params (no guide sites) |

## Usage

### Basic Model

```python
from scribe.models.builders import ModelBuilder, BetaSpec, LogNormalSpec
from scribe.models.components import NegativeBinomialLikelihood

model = (ModelBuilder()
    .add_param(BetaSpec("p", (), (1.0, 1.0)))
    .add_param(LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True))
    .with_likelihood(NegativeBinomialLikelihood())
    .build())
```

### Linked Parameterization

```python
model = (ModelBuilder()
    .add_param(BetaSpec("p", (), (1.0, 1.0)))
    .add_param(LogNormalSpec("mu", ("n_genes",), (0.0, 1.0), is_gene_specific=True))
    .add_derived("r", lambda p, mu: mu * (1-p) / p, ["p", "mu"])
    .with_likelihood(NegativeBinomialLikelihood())
    .build())
```

### Building Guides

```python
from scribe.models.builders import GuideBuilder
from scribe.models.components import MeanFieldGuide, LowRankGuide

specs = [
    BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide()),
    LogNormalSpec("r", ("n_genes",), (0.0, 1.0), 
                  is_gene_specific=True, 
                  guide_family=LowRankGuide(rank=10)),
]

guide = GuideBuilder().from_specs(specs).build()
```

### Mixed Guide Families

```python
from scribe.models.components import AmortizedGuide, Amortizer, TOTAL_COUNT

amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[64, 32],
    output_params=["alpha", "beta"],
)

specs = [
    BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide()),
    LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10)),
    BetaSpec("p_capture", ("n_cells",), (1.0, 1.0), 
             is_cell_specific=True,
             guide_family=AmortizedGuide(amortizer=amortizer)),
]
```

### Joint Low-Rank Guide (Cross-Parameter Correlations)

When gene-specific parameters are expected to be correlated (e.g., mean
expression `mu` and dispersion `phi`), assign a `JointLowRankGuide` with the
same `group` string to all parameters that should be jointly modeled. The guide
builder uses a chain rule decomposition with Woodbury-efficient conditioning to
sample from the joint distribution while maintaining separate NumPyro sample
sites.

```python
from scribe.models.components import JointLowRankGuide
from scribe.models.builders import ExpNormalSpec, GuideBuilder

joint = JointLowRankGuide(rank=10, group="nb_params")
specs = [
    ExpNormalSpec("mu", ("n_genes",), (0.0, 1.0),
                  is_gene_specific=True, guide_family=joint, constrained_name="mu"),
    ExpNormalSpec("phi", ("n_genes",), (0.0, 1.0),
                  is_gene_specific=True, guide_family=joint, constrained_name="phi"),
]
guide = GuideBuilder().from_specs(specs).build()
```

This extends naturally to three or more parameters (e.g., ZINB with `gate`).
Parameters not in a joint group are processed independently as usual. See
`paper/_joint_low_rank_guide.qmd` for the full derivation.

## Posterior Extraction

The **`posterior`** module extracts posterior distributions from optimized
variational parameters (e.g., after SVI inference). It handles all
parameterizations and guide families, including **joint-aware** extraction for
`JointLowRankGuide`.

**Joint-aware posterior extraction:**

- `posterior.py` handles `joint_{group}_{name}_*` param keys produced by
  `JointLowRankGuide` (e.g., `joint_joint_mu_loc`, `joint_joint_mu_W`,
  `joint_joint_mu_raw_diag`).
- **`get_posterior_distributions`** returns:
  - **Per-parameter marginals** keyed by parameter name (e.g., `"mu"`, `"phi"`):
    each is a `LowRankMultivariateNormal` + transform, identical in structure
    to standard low-rank posteriors.
  - **Full joint distribution** keyed as `"joint:{group}"` (e.g., `"joint:joint"`):
    a stacked `LowRankMultivariateNormal` over the concatenated parameter
    space, with `param_names` and `param_sizes` for indexing.

**Helper functions:**

- **`_find_joint_prefix`**: Scans params for `joint_*_{name}_loc` and returns
  the prefix (e.g., `"joint_joint_mu"`) or `None`.
- **`_build_joint_low_rank_posterior`**: Builds the per-parameter marginal from
  joint guide params (`{prefix}_loc`, `{prefix}_W`, `{prefix}_raw_diag`).
- **`_build_joint_full_distribution`**: Stacks loc/W/D from each parameter in
  the group into a single joint `LowRankMultivariateNormal`.

### Joint-Parameter Compatibility Checklist

When enabling a new parameter in `joint_params` (for example, adding `gate` to
an existing `mu`/`phi` joint group), ensure the full MAP pipeline remains
consistent end-to-end:

1. **Posterior builder support**:
   - Confirm `get_posterior_distributions()` resolves `joint_*_{name}_*` keys
     for that parameter (not only `{name}_loc/{name}_scale` keys).
2. **MAP extraction compatibility**:
   - Confirm `results.get_map()` succeeds without key errors and returns finite
     constrained values for the joint-modeled parameter.
3. **MAP-dependent consumers**:
   - Validate utilities that internally call `get_map()` (for example
     cell-assignment probabilities, denoising, and MAP-based diagnostics).
4. **Regression coverage**:
   - Add/update tests in:
     - `tests/test_builders.py` (posterior + MAP extraction)
     - `tests/test_visualize_capture_anchor.py` (p-capture scaling path)
     - `tests/test_viz_utils_module_refactor.py` (bio-PPC / denoising path)

Without this checklist, a joint-parameter addition can appear to train
correctly while failing later in diagnostics with missing-key errors.

## Performance Considerations

### Amortized Guides

The guide builder automatically uses `flax_module` to register amortized guides,
which provides optimal JIT performance. Amortizers are implemented as Flax Linen
modules, which use pure functional application (`nn_module.apply`) that doesn't
cause retracing.

**Key optimizations:**

- Linen modules are registered automatically when using `AmortizedGuide`
- Parameters are registered once via `flax_module` (outside the plate)
- Pure functional application inside the plate (JIT-safe, no mutations)
- Works seamlessly with mini-batching (`batch_size` parameter)

**Performance benefits:**

- Faster JIT compilation and execution
- No retracing or progressive slowdown during long SVI runs
- No memory accumulation from repeated mutations
- Correct behavior under JIT compilation
- Compatible with all SVI optimizers

For more details, see the [Amortizers documentation](../components/README.md#performance-considerations).

## Multiple Dispatch

The builders use multiple dispatch to route to the correct implementation:

```python
from multipledispatch import dispatch

@dispatch(BetaSpec, MeanFieldGuide, dict, object)
def setup_guide(spec, guide, dims, model_config, **kwargs):
    # Implementation for Beta + MeanField
    ...

@dispatch(LogNormalSpec, LowRankGuide, dict, object)
def setup_guide(spec, guide, dims, model_config, **kwargs):
    # Implementation for LogNormal + LowRank
    ...
```

Adding new combinations is trivial - just add a new dispatch method.

For `JointLowRankGuide`, parameters are grouped by `group` name and processed
through `setup_joint_guide` using a chain rule decomposition rather than
individual dispatch.

## Plate Handling

The ModelBuilder handles three plate modes:

1. **Prior Predictive** (counts=None): Sample from prior
2. **Full Sampling** (counts provided, batch_size=None): All cells
3. **Batch Sampling** (counts provided, batch_size set): Mini-batch

Cell-specific parameters are sampled inside the cell plate and support
batch indexing for efficient stochastic VI.

### Annotation Prior Logits

The model closure built by `ModelBuilder.build()` accepts an optional
`annotation_prior_logits` keyword argument of shape `(n_cells, n_components)`.
This array is forwarded directly to `likelihood.sample()`.

The **guide** closure built by `GuideBuilder.build()` also accepts
`annotation_prior_logits` for API compatibility (NumPyro passes the same
kwargs to both model and guide), but ignores it.  Annotation priors are
observed data, not latent variables, so no variational approximation is
needed.

```python
# Model function signature (after build):
model(n_cells, n_genes, model_config,
      counts=None, batch_size=None,
      annotation_prior_logits=None)

# Guide function signature (after build):
guide(n_cells, n_genes, model_config,
      counts=None, batch_size=None,
      annotation_prior_logits=None)  # ignored
```

The `annotation_prior_logits` array flows through the inference stack as
follows:

```
scribe.fit(annotation_key=...)  →  build_annotation_prior_logits()
                                        │
                                        ▼
_run_inference(annotation_prior_logits=...)  →  _run_svi_inference(...)
                                                        │
                                                        ▼
SVIInferenceEngine.run_inference(annotation_prior_logits=...)
                                        │
                                        ▼
model_args["annotation_prior_logits"]  →  model(...)  →  likelihood.sample(...)
```

## VAE Path (model builder)

When any cell spec has `guide_family=VAELatentGuide` with `decoder` and
`latent_spec` set, the model builder:

1. **Validates** that no mixture specs are present (VAE and mixture are mutually
   exclusive).
2. Builds a **`vae_cell_fn`** closure that: samples `z` from
   `latent_spec.make_prior_dist()`, runs the decoder via
   `flax_module("vae_decoder", decoder, ...)`, and registers decoder outputs as
   `numpyro.deterministic` sites.
3. **Filters** VAE marker specs out of the specs passed to the likelihood (so
   `z` is not sampled again via `sample_prior`).
4. Passes **`vae_cell_fn`** into the likelihood's `sample()`; the likelihood
   calls it inside the cell plate and merges the returned dict into
   `param_values` before building the observation distribution.

Decoder-driven parameters are thus deterministic given `z`, not sample sites, so
they have no guide counterpart.

## Files

| File | Purpose |
|------|---------|
| `parameter_specs.py` | ParamSpec, LatentSpec, GaussianLatentSpec, Hierarchical*Spec, BiologyInformedCaptureSpec; `sample_prior` dispatch |
| `model_builder.py` | ModelBuilder class |
| `guide_builder.py` | GuideBuilder and `setup_guide` dispatch |
| `posterior.py` | `get_posterior_distributions`; joint-aware extraction from `JointLowRankGuide` params |
| `__init__.py` | Public API exports |

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
| `LatentSpec` | Base for VAE latent z | — | abstract |
| `GaussianLatentSpec` | Normal(loc, scale).to_event(1) from encoder output | — | z (guide only) |

### ParamSpec Attributes

Each spec has the following attributes:

- `name`: Parameter name (used as sample site)
- `shape_dims`: Symbolic dimensions like `()`, `("n_genes",)`, `("n_cells",)`
- `default_params`: Default distribution parameters
- `is_gene_specific`: If True, shape is (n_genes,)
- `is_cell_specific`: If True, sampled inside cell plate
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
`GroupedAmortizedGuide` with `encoder` and `latent_spec` set: it runs the
encoder, builds `var_params = {"loc": loc, "log_scale": log_scale}`,
calls `latent_spec.make_guide_dist(var_params)`, and samples
`numpyro.sample(latent_spec.sample_site, guide_dist)`.

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
| `AmortizedGuide(net)` | Neural network amortization | High-dim params |
| `GroupedAmortizedGuide` | VAE: encoder + latent_spec (z) + decoder | Joint latent + params |

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

## Plate Handling

The ModelBuilder handles three plate modes:

1. **Prior Predictive** (counts=None): Sample from prior
2. **Full Sampling** (counts provided, batch_size=None): All cells
3. **Batch Sampling** (counts provided, batch_size set): Mini-batch

Cell-specific parameters are sampled inside the cell plate and support
batch indexing for efficient stochastic VI.

## Files

| File | Purpose |
|------|---------|
| `parameter_specs.py` | ParamSpec, LatentSpec, GaussianLatentSpec; `sample_prior` dispatch |
| `model_builder.py` | ModelBuilder class |
| `guide_builder.py` | GuideBuilder and `setup_guide` dispatch |
| `__init__.py` | Public API exports |

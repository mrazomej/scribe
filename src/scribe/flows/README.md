# Normalizing Flows Module

Flax Linen-based normalizing flow library for SCRIBE, providing pluggable
flow architectures for use as learned priors and posterior flows in
variational autoencoders.

## Architecture

```
flows/
├── __init__.py          # Public API
├── base.py              # FlowChain (sequential composition)
├── transforms.py        # Rational-quadratic spline primitives (pure JAX)
├── coupling.py          # AffineCoupling, SplineCoupling
├── autoregressive.py    # MADE, MAF, IAF
├── distributions.py     # FlowDistribution (NumPyro wrapper)
└── README.md
```

## Flow Types

| Flow | Forward (data→latent) | Inverse (latent→data) | Best For |
|------|----------------------|----------------------|----------|
| **AffineCoupling** | O(1), parallel | O(1), parallel | Baseline, fast training |
| **SplineCoupling** | O(1), parallel | O(1), parallel | Expressive prior/posterior |
| **MAF** | O(1), parallel | O(D), sequential | Learned priors (fast density) |
| **IAF** | O(D), sequential | O(1), parallel | Posterior flows (fast sampling) |

## Convention

All flows follow the same interface:

```python
z, log_det = flow(x, reverse=False)   # forward:  data → latent
x, log_det = flow(z, reverse=True)    # inverse:  latent → data
```

- `forward` returns `log|det(dz/dx)|` (forward Jacobian)
- `inverse` returns `log|det(dx/dz)|` = `-log|det(dz/dx)|` (inverse Jacobian)

## Usage

### Standalone Flow

```python
import jax
import jax.numpy as jnp
from scribe.flows import FlowChain

# Create a 4-layer spline coupling flow over 10 dimensions
flow = FlowChain(
    features=10,
    num_layers=4,
    flow_type="spline_coupling",
    hidden_dims=[64, 64],
    activation="relu",
    n_bins=8,
)

# Initialize parameters
params = flow.init(jax.random.PRNGKey(0), jnp.zeros(10))

# Forward pass
z, log_det = flow.apply(params, jnp.ones(10))

# Inverse pass (should recover input)
x_recovered, neg_log_det = flow.apply(params, z, reverse=True)
```

### Optional covariate conditioning

`FlowChain` accepts optional `covariate_specs: List[CovariateSpec]`. When
provided, pass a `covariates` dict at init and apply; the chain embeds
covariates and passes them as context to every layer.

```python
from scribe.flows import FlowChain
from scribe.models.components import CovariateSpec

specs = [CovariateSpec("batch", num_categories=4, embedding_dim=8)]
chain = FlowChain(
    features=10, num_layers=4, flow_type="affine_coupling",
    hidden_dims=[64, 64], covariate_specs=specs,
)
x = jnp.ones((3, 10))
covs = {"batch": jnp.array([0, 1, 2])}
params = chain.init(rng, x, covariates=covs)
z, log_det = chain.apply(params, x, covariates=covs)
```

### Continuous context conditioning

`FlowChain` also accepts a `context_dim: int` attribute and a `context` keyword
argument to `__call__`. This enables conditioning on a pre-formed continuous
vector (e.g., previously sampled parameters in a joint guide) without going
through `CovariateEmbedding`. If both `covariates` and `context` are provided,
the embedded covariates and the continuous context are concatenated.

```python
from scribe.flows import FlowChain

chain = FlowChain(
    features=10, num_layers=4, flow_type="spline_coupling",
    hidden_dims=[64, 64], context_dim=5,  # 5-d continuous context
)
x = jnp.ones((3, 10))
ctx = jnp.ones((3, 5))
params = chain.init(rng, x, context=ctx)
z, log_det = chain.apply(params, x, context=ctx)
```

This feature is used by `JointNormalizingFlowGuide` to pass the unconstrained
samples of previously sampled parameters as context to conditional flows in the
chain-rule decomposition.

### As a NumPyro Distribution (Learned Prior)

```python
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from scribe.flows import FlowChain, FlowDistribution

def model(counts, n_cells, n_genes, latent_dim=10):
    # Register flow parameters with NumPyro
    flow_fn = flax_module(
        "prior_flow",
        FlowChain(features=latent_dim, num_layers=4,
                  flow_type="spline_coupling", hidden_dims=[64, 64]),
        input_shape=(latent_dim,),
    )

    # Wrap as a distribution
    base = dist.Normal(jnp.zeros(latent_dim), 1.0).to_event(1)
    prior = FlowDistribution(flow_fn, base)

    # Sample from the learned prior
    with numpyro.plate("cells", n_cells):
        z = numpyro.sample("z", prior)
    ...
```

## Choosing a Flow

- **Start with `affine_coupling`** for fast iteration and debugging.
- **Switch to `spline_coupling`** for production — strictly more expressive
  per layer with moderate overhead.
- **Use `maf`** as a learned prior when fast density evaluation matters
  (e.g., computing KL divergence during training).
- **Use `iaf`** as a posterior flow when fast sampling matters
  (e.g., encoder → sample → decoder in a VAE).

## Spline Details

The rational-quadratic spline (Durkan et al., NeurIPS 2019) provides an
analytically invertible, element-wise monotone transform. Key parameters:

- `n_bins`: Number of spline segments (default 8). More bins = more
  expressive but more parameters per layer.
- `boundary`: The spline is defined on [-B, B]; identity outside.
  Default 3.0 covers ~99.7% of a standard normal.

Each spline dimension requires `3K + 1` parameters (K widths, K heights,
K+1 derivatives), predicted by the conditioner network.

## FlowDistribution Convenience Methods

`FlowDistribution` provides two convenience methods for point estimation:

- **`estimate_mean(key, n_samples=1000)`**: Monte Carlo estimate of the
  distribution mean by averaging independent samples.
- **`find_mode(key, n_init_samples=100, n_steps=300, lr=1e-3)`**: Approximate
  mode via gradient ascent on `log_prob`. Initializes from the best of
  `n_init_samples` candidates and runs Adam optimization. Requires `optax`
  (falls back to `estimate_mean` if not available).

These are useful for standalone analysis but the main `get_map()` pipeline uses
guide-execution-based sampling for uniformity across flow types.

## ComponentFlowDistribution (Mixture / Dataset Support)

`ComponentFlowDistribution` stacks K independent `FlowDistribution` instances
along a leading batch axis, producing a single NumPyro distribution whose
`event_shape` gains an extra leading K dimension.  It is the core primitive for
mixture-aware and dataset-aware normalizing-flow guides.

Two construction strategies are supported:

- **`"independent"`** (default): K separate `FlowChain` instances with
  independent Flax parameters — maximum expressiveness.
- **`"shared"`**: A single `FlowChain` conditioned on a one-hot index vector —
  parameter-efficient, components specialise through the context pathway.

In either case, `get_component(k)` returns a ready-to-use distribution for
index `k`.  For the shared strategy the one-hot covariate is already bound in
the closure, so callers do not need to manage context.

### Basic usage

```python
from scribe.flows import ComponentFlowDistribution, FlowDistribution

# K=3 independent flows over G=20 genes
comp_dist = ComponentFlowDistribution(
    [FlowDistribution(fn_k, base) for fn_k in per_comp_fns],
    axis_name="component",
)

sample = comp_dist.sample(key)           # shape (3, 20)
lp     = comp_dist.log_prob(sample)      # scalar
comp_0 = comp_dist.get_component(0)      # FlowDistribution for component 0
s0     = comp_0.sample(key)              # shape (20,)
```

### Nesting (components x datasets)

For parameters with both `is_mixture=True` and `is_dataset=True`, shapes
resolve to `(K, D, G)`.  The distribution is nested: the outer
`ComponentFlowDistribution` (axis `"component"`) wraps K inner instances
(axis `"dataset"`), each of which wraps D leaf `FlowDistribution` objects.

```python
nested = ComponentFlowDistribution(
    [ComponentFlowDistribution(per_ds, "dataset") for per_ds in per_comp],
    axis_name="component",
)
nested.event_shape                        # (K, D, G)
nested.get_component(0).get_component(1)  # FlowDistribution for comp 0, ds 1
```

### Point estimation

`ComponentFlowDistribution` delegates `estimate_mean` and `find_mode` to each
component independently, stacking the results back into `(K, *inner_event)`.
These are used by `get_map()` when the `flow_map_method` is `"mean"` or
`"optimize"`.

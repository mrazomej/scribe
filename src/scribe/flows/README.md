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

## Conditioner Stability (High-Dimensional Inputs)

Three user-configurable flags stabilize the conditioner MLP in coupling and
autoregressive flows, preventing NaN loss at initialization when input
dimensionality is large (e.g. 28K genes in a joint flow).  All are **on by
default**; disable any of them by setting the flag to `False` on `FlowChain`,
the guide family dataclass, or via the Hydra/API config.

| Flag | Default | What it does |
|---|---|---|
| `zero_init_output` | `True` | Zero-initialize the conditioner's output Dense layer so the flow starts as an **exact identity** (log-det = 0).  For affine coupling this means zero shift / log-scale.  For spline coupling a custom bias initializer sets knot derivatives to 1.0, ensuring identity output *and* zero log-det even at 28K+ dimensions. |
| `use_layer_norm` | `True` | Apply `nn.LayerNorm` after each hidden Dense in the conditioner MLP.  Stabilizes activations when fan-in is large (e.g. 42K inputs funneling into a 64-wide bottleneck). |
| `use_residual` | `True` | Add skip connections between consecutive hidden layers of the same width.  Improves gradient flow during training. |

### Configuration

At the `FlowChain` level:

```python
chain = FlowChain(
    features=28000, num_layers=4, flow_type="spline_coupling",
    hidden_dims=[64, 64],
    zero_init_output=True,   # default
    use_layer_norm=True,     # default
    use_residual=True,       # default
)
```

Via the guide family dataclass:

```python
NormalizingFlowGuide(
    flow_type="spline_coupling",
    zero_init_output=True,
    use_layer_norm=True,
    use_residual=True,
)
```

Via Hydra command line:

```bash
guide_flow_zero_init=false guide_flow_layer_norm=false guide_flow_residual=false
```

## Training Stability (Andrade 2024)

Two additional features from [Andrade 2024 (arXiv:2402.16408)](https://arxiv.org/abs/2402.16408)
bound sample magnitudes during training to prevent NaN gradients in
high-dimensional flows.  Both are **on by default**.

| Flag | Default | What it does |
|---|---|---|
| `soft_clamp` | `True` | Replace hard `jnp.clip(log_scale, -5, 5)` in `AffineCoupling` with a smooth asymmetric `arctan`-based clamp.  `alpha_pos=0.1` caps per-layer expansion to ~10%, while `alpha_neg=2.0` allows contraction.  Preserves gradients at the boundary (unlike hard clipping).  Only affects affine coupling — spline derivatives are already bounded by the RQS construction. |
| `use_loft` | `True` | Append a **LOFT** (Log Soft Extension) layer and a trainable element-wise affine after all coupling layers.  For `|z| < tau` (default 100) the LOFT is identity; beyond that, growth is logarithmic.  The final affine `sigma * z + mu` re-expands the range to match the target posterior's scale. |

### Forward pass with LOFT

```
z_base → [coupling layers (soft-clamped)] → LOFT → final affine → z_out
```

### Float64 log-det accumulation

The log-determinant Jacobian is a running sum across all coupling layers,
LOFT, and the final affine.  In high-dimensional flows (e.g. 28K genes)
this sum can lose significant precision in float32.  Setting
`log_det_f64=True` on `FlowChain` initializes the accumulator in float64;
each per-layer contribution is promoted automatically via JAX type rules.

This requires `jax_enable_x64=True` to be effective (otherwise JAX
silently downcasts float64 to float32).  When set via `fit()` or Hydra
(`guide_flow_log_det_f64=true`), `enable_x64` is **auto-promoted** so the
user does not need to set both flags.

**Off by default** because most consumer GPUs throttle float64 throughput
to 1/32 or 1/64 of float32.  Recommended for datacenter GPUs (A100,
H100, MI250X) with full-rate or half-rate float64.

### Configuration

At the `FlowChain` level:

```python
chain = FlowChain(
    features=28000, num_layers=4, flow_type="affine_coupling",
    hidden_dims=[64, 64],
    soft_clamp=True,      # default
    use_loft=True,        # default
    log_det_f64=False,    # default; set True on datacenter GPUs
)
```

Via Hydra command line:

```bash
guide_flow_soft_clamp=false guide_flow_loft=false
guide_flow_log_det_f64=true   # auto-promotes enable_x64=true
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

### Numerical safety (matching the nflows reference)

The RQS implementation in `transforms.py` includes several safeguards
that match the reference *nflows* library (Durkan et al.):

| Guard | What it prevents |
|---|---|
| **Minimum bin width / height** (`1e-3` proportion) | Bins collapsing → slope `s_k = h_k/w_k` diverging |
| **Boundary pinning** | Float32 cumsum drift pushing knots outside `[-B, B]` |
| **Width/height recomputation** | Inconsistency between cumulative positions and per-bin widths |
| **Log-space log-det** (`log(num) - 2·log(denom)`) | Intermediate overflow in `deriv_num / denom²` |
| **Identity-bias init** (see Conditioner Stability) | Non-zero log-det at initialization |

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

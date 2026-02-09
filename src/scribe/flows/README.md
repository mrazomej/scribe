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

# Model Components

Reusable building blocks for constructing probabilistic models.

## Overview

This directory contains the atomic components used by the builders:

- **likelihoods/**: Likelihood functions (NB, ZINB, VCP variants)
- **guide_families.py**: Variational family implementations (MeanField, LowRank,
  Amortized, GroupedAmortized)
- **vae_components.py**: Encoder/decoder building blocks for VAE-style models
- **covariate_embedding.py**: Categorical covariate embedding for conditioning
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
| `GroupedAmortizedGuide` | VAE: encoder + latent_spec + decoder | Joint latent z + params |

### Example

```python
from scribe.models.components import MeanFieldGuide, LowRankGuide
from scribe.models.builders import BetaSpec, LogNormalSpec

# Per-parameter guide families
BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide())
LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10))
```

## Covariate embedding

Categorical covariates (batch, donor, condition, etc.) are embedded and
concatenated to inputs so encoders, decoders, and flow layers can be
conditioned on them. Used by VAE components and by the flows module.

| Class               | Description                                      |
|---------------------|--------------------------------------------------|
| `CovariateSpec`     | Spec for one covariate: name, num_categories, embedding_dim |
| `CovariateEmbedding`| Flax module: dict of ID arrays → concatenated embedding vector |

### CovariateSpec

```python
from scribe.models.components import CovariateSpec

# name must match the key in the covariates dict passed at call time
spec = CovariateSpec("batch", num_categories=4, embedding_dim=8)
```

### CovariateEmbedding

```python
from scribe.models.components import CovariateEmbedding, CovariateSpec
import jax.numpy as jnp

specs = [
    CovariateSpec("batch", num_categories=4, embedding_dim=8),
    CovariateSpec("donor", num_categories=10, embedding_dim=8),
]
embedder = CovariateEmbedding(covariate_specs=specs)
covariates = {"batch": jnp.array([0, 1, 2]), "donor": jnp.array([3, 5, 7])}
params = embedder.init(jax.random.PRNGKey(0), covariates)
emb = embedder.apply(params, covariates)  # shape (3, 16)
```

Covariates are optional: omit `covariate_specs` (and do not pass `covariates`
at apply time) when not needed.

## VAE encoder/decoder components

Modular encoder and decoder building blocks for VAE-style models. The design
uses **abstract base classes** and a **template method**: shared preprocessing
and MLP backbone are in the base; concrete subclasses implement the output head
via `encode_to_params` (encoder) or `decode_to_output` (decoder).

### Classes

| Class               | Description |
|---------------------|-------------|
| `AbstractEncoder`  | Base: input transform → optional standardize → optional covariate concat → MLP → `encode_to_params(h)` |
| `GaussianEncoder`   | Diagonal-Gaussian posterior: outputs `(loc, log_scale)` |
| `AbstractDecoder`   | Base: optional covariate concat to z → reversed MLP → `decode_to_output(h)` |
| `SimpleDecoder`     | Single continuous vector output; optional destandardization |

Registries for lookup by name: `ENCODER_REGISTRY`, `DECODER_REGISTRY` (e.g.
`ENCODER_REGISTRY["gaussian"]` → `GaussianEncoder`).

When used with **GroupedAmortizedGuide**, the guide builder (see
[builders README](../builders/README.md)) wires encoder output to a
**LatentSpec** (e.g. `GaussianLatentSpec`): encoder → `var_params` →
`latent_spec.make_guide_dist(var_params)` → `sample(site, guide_dist)` for the
latent z.

### Input transforms and standardization

Encoders support configurable **input transformation** (`input_transformation`):
`"log1p"`, `"log"`, `"sqrt"`, `"identity"`. Optional **standardization** via
`standardize_mean` and `standardize_std` (per feature); decoders can apply the
inverse with the same arrays for reconstruction in original scale.

### Optional covariate conditioning

Pass `covariate_specs: List[CovariateSpec]` to encoder or decoder. At init and
apply, pass `covariates: Dict[str, jnp.ndarray]` (e.g. `{"batch": ids}`). The
module embeds and concatenates covariates to the MLP input.

### Example: encoder and decoder

```python
from scribe.models.components import GaussianEncoder, SimpleDecoder
import jax.numpy as jnp

input_dim, latent_dim = 2000, 10
hidden_dims = [256, 128]

encoder = GaussianEncoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dims=hidden_dims,
    activation="relu",
    input_transformation="log1p",
)
decoder = SimpleDecoder(
    output_dim=input_dim,
    latent_dim=latent_dim,
    hidden_dims=hidden_dims,
)

# Without covariates
enc_params = encoder.init(rng, jnp.zeros(input_dim))
dec_params = decoder.init(rng, jnp.zeros(latent_dim))
loc, log_scale = encoder.apply(enc_params, counts)
reconstruction = decoder.apply(dec_params, z)
```

### Example: with covariates

```python
from scribe.models.components import GaussianEncoder, SimpleDecoder, CovariateSpec

covariate_specs = [CovariateSpec("batch", num_categories=4, embedding_dim=8)]
encoder = GaussianEncoder(
    input_dim=100, latent_dim=10, hidden_dims=[64, 32],
    covariate_specs=covariate_specs,
)
decoder = SimpleDecoder(
    output_dim=100, latent_dim=10, hidden_dims=[64, 32],
    covariate_specs=covariate_specs,
)
x = jnp.ones((3, 100))
covs = {"batch": jnp.array([0, 1, 2])}
enc_params = encoder.init(rng, x, covariates=covs)
dec_params = decoder.init(rng, jnp.zeros((3, 10)), covariates=covs)
loc, log_scale = encoder.apply(enc_params, x, covariates=covs)
out = decoder.apply(dec_params, loc, covariates=covs)
```

### Adding new encoder/decoder types

Subclass `AbstractEncoder` and implement `encode_to_params(h)` (return
distribution parameters), or subclass `AbstractDecoder` and implement
`decode_to_output(h)`. Register in `ENCODER_REGISTRY` or `DECODER_REGISTRY` if
you want name-based lookup.

## Amortizers

Amortizers predict variational parameters from sufficient statistics using an
MLP. This is useful for cell-specific parameters like capture probability, where
learning separate parameters for each cell would be prohibitive.

### Using the Factory (Recommended)

```python
from scribe.models.presets import create_capture_amortizer
from scribe.models.components import AmortizedGuide

# Automatically handles constrained vs unconstrained output parameters
# In constrained mode, output_transform ("softplus" or "exp") and
# output_clamp_min / output_clamp_max control positive output stability.
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
    output_params=["alpha", "beta"],  # output_transforms ensure > 0
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
| Constrained      | `alpha`, `beta`         | Beta(α, β) or BetaPrime(α, β)  |
| Unconstrained    | `loc`, `log_scale`      | Normal(loc, scale) → transform |

### Performance Considerations

The amortizer is implemented as a Flax Linen module, which provides optimal JIT
performance:

- **Pure functional design**: Linen modules use `nn_module.apply({"params": params}, ...)`
  which is a pure function that doesn't cause retracing
- **No Python-side mutations**: Unlike NNX modules, Linen modules don't require
  `nnx.merge` operations that can trigger retracing
- **Efficient JIT compilation**: NumPyro's `flax_module` automatically handles
  parameter registration and JIT compilation
- **Stable performance**: No progressive slowdown during long SVI runs

**Usage in NumPyro:**

The guide builder automatically uses `flax_module` to register amortizers. You
don't need to do anything special:

```python
from scribe.models.components import Amortizer, AmortizedGuide, TOTAL_COUNT
from scribe.models.config import GuideFamilyConfig

# Create amortizer (Linen module)
amortizer = Amortizer(
    sufficient_statistic=TOTAL_COUNT,
    hidden_dims=[64, 32],
    output_params=["alpha", "beta"],
)

# Use in guide config - flax_module is used automatically
guide_config = GuideFamilyConfig(
    p_capture=AmortizedGuide(amortizer=amortizer)
)
```

**Manual usage (advanced):**

If you need to use amortizers directly in NumPyro models:

```python
from numpyro.contrib.module import flax_module
import numpyro

# Register Linen module with NumPyro (JIT-safe, no retracing)
module_name = "p_capture_amortizer"
# Input shape is (input_dim,) where input_dim is typically 1 for scalar statistics
net = flax_module(module_name, amortizer, input_shape=(amortizer.input_dim,))

# Use inside plate (JIT-safe)
with numpyro.plate("cells", n_cells):
    var_params = net(data)
```

**Why Linen instead of NNX?**

Flax Linen modules are designed for functional programming and JIT compilation:
- `flax_module` calls `nn_module.apply({"params": params}, ...)` which is pure
- No Python-side object creation or mutations inside traced contexts
- Avoids retracing issues that can cause progressive slowdown
- Matches NumPyro's recommended pattern for neural network integration

**Troubleshooting performance issues:**

If you experience slowdowns during SVI:

1. **Check JAX compilation logs:**
   ```bash
   JAX_LOG_COMPILES=1 python your_script.py
   ```
   Look for repeated compilations after the first few steps - this indicates
   retracing. With Linen, you should see only initial compilation.

2. **Verify parameter stability:**
   - Parameter shapes/dtypes should remain stable during training
   - If shapes change, recompilation is expected (but should be rare)

3. **Common causes of retracing:**
   - Parameter shapes changing during training (unusual)
   - Data-dependent shapes passed to jitted functions
   - Calling `flax_module` inside plates (should never happen - register outside)

**Important notes:**

- The amortizer is a stateless Linen module (no BatchNorm or other mutable state)
- Parameters are automatically registered by `flax_module` when first called
- The module is initialized lazily on first use

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
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Model Components                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐ ┌─────────────────┐ ┌──────────────┐ ┌───────────────┐  │
│  │ Likelihoods  │ │  Guide Families  │ │ Covariate     │ │ VAE Enc/Dec    │  │
│  │              │ │                  │ │ Embedding     │ │               │  │
│  │ NegBinomial  │ │ MeanFieldGuide   │ │ CovariateSpec │ │ GaussianEnc   │  │
│  │ ZINB, VCP    │ │ LowRankGuide     │ │ Covariate     │ │ SimpleDecoder │  │
│  │              │ │ AmortizedGuide   │ │ Embedding     │ │ AbstractEnc   │  │
│  │              │ │ GroupedAmortized │ │               │ │ AbstractDec   │  │
│  └──────────────┘ └─────────────────┘ └──────────────┘ └───────────────┘  │
│                                                                             │
│  ┌─────────────────┐                                                       │
│  │   Amortizers     │   SufficientStatistic, Amortizer, TOTAL_COUNT         │
│  └─────────────────┘                                                       │
│                              ▼                                              │
│                     Used by Builders                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

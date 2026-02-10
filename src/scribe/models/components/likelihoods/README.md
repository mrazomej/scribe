# Likelihoods Submodule

This submodule provides likelihood components for count data models in SCRIBE.

## Module Structure

```
likelihoods/
├── __init__.py          # Re-exports all public classes
├── base.py              # Likelihood ABC + helper functions
├── negative_binomial.py # Standard NB likelihood
├── zero_inflated.py     # Zero-Inflated NB likelihood
├── vcp.py               # VCP variants (NB and ZINB with capture probability)
└── README.md            # This file
```

## Classes

| Class | File | Description |
|-------|------|-------------|
| `Likelihood` | base.py | Abstract base class for all likelihoods |
| `NegativeBinomialLikelihood` | negative_binomial.py | Standard Negative Binomial |
| `ZeroInflatedNBLikelihood` | zero_inflated.py | Zero-Inflated Negative Binomial |
| `NBWithVCPLikelihood` | vcp.py | NB with Variable Capture Probability |
| `ZINBWithVCPLikelihood` | vcp.py | ZINB with Variable Capture Probability |

## Plate Modes

All likelihoods handle three plate modes for different use cases:

| Mode | `counts` | `batch_size` | Use Case |
|------|----------|--------------|----------|
| Prior Predictive | `None` | - | Generate synthetic data |
| Full Sampling | provided | `None` | MCMC, small datasets |
| Batch Sampling | provided | specified | SVI on large datasets |

## VAE Path

When the model uses a **VAE** (encoder/decoder with `VAELatentGuide`), the model
builder passes a **`vae_cell_fn`** into each likelihood's `sample()` method. The
likelihood then:

1. Runs inside the cell plate (per-cell or per minibatch).
2. Calls `vae_cell_fn(batch_idx)` to obtain decoder-driven parameters (e.g. `r`,
   `gate`) and merges them into `param_values`.
3. Builds the observation distribution from the full `param_values` and samples
   or conditions on counts.

Decoder-driven parameters are **not** sample sites; they are produced by the
decoder from the latent `z`. Mixture detection in likelihoods uses
`"mixing_weights" in param_values` (not shape-based checks) so that VAE per-cell
decoder output is not mistaken for a mixture.

## Usage

```python
from scribe.models.components import NegativeBinomialLikelihood

# Create a likelihood
likelihood = NegativeBinomialLikelihood()

# Use in ModelBuilder
builder.with_likelihood(likelihood)
```

## VCP Likelihoods

The Variable Capture Probability (VCP) likelihoods model cell-specific technical
variation in capture efficiency. They support two parameterizations:

- **Canonical/mean-prob**: Uses `p_capture` (Beta distribution)
- **Mean-odds**: Uses `phi_capture` (BetaPrime distribution)

```python
# Explicit parameterization (recommended)
likelihood = NBWithVCPLikelihood(capture_param_name="phi_capture")

# With unconstrained sampling for better optimization
likelihood = NBWithVCPLikelihood(
    capture_param_name="phi_capture",
    is_unconstrained=True,
    transform=ExpTransform(),
    constrained_name="phi_capture",
)
```

## Adding New Likelihoods

To add a new likelihood:

1. Create a new file or add to an existing one
2. Inherit from `Likelihood` (in `base.py`)
3. Implement the `sample` method with signature including `vae_cell_fn: Optional[Callable] = None`, handling:
   - All three plate modes (prior predictive, full, batch)
   - Non-VAE path: `vae_cell_fn is None` — build distribution once from `param_values`
   - VAE path: `vae_cell_fn` provided — call it inside the cell plate, merge into `param_values`, then build distribution
4. Export the class in `__init__.py`

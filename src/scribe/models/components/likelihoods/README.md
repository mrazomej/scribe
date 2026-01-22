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
3. Implement the `sample` method handling all three plate modes
4. Export the class in `__init__.py`

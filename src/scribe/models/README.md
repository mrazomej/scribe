# SCRIBE Models

This directory contains all model definitions, configurations, and registry
functions for the SCRIBE package. SCRIBE provides a comprehensive suite of
probabilistic models for single-cell RNA sequencing data analysis using
variational inference.

## Overview

The models package is organized around three key dimensions:

1. **Model Types**: Different probabilistic models for count data
2. **Parameterizations**: Different ways to parameterize the same underlying
   model
3. **Inference Methods**: SVI (Stochastic Variational Inference) and VAE
   (Variational Autoencoder)

## Model Types

SCRIBE supports four main model types, each available in both single-component
and mixture variants:

### Single-Component Models

- **NBDM** (Negative Binomial-Dirichlet Multinomial): Models gene expression
  counts using a Negative Binomial distribution with cell-specific success
  probabilities and gene-specific dispersion parameters.

- **ZINB** (Zero-Inflated Negative Binomial): Extends NBDM with an additional
  zero-inflation component to handle excess zeros commonly observed in
  single-cell data.

- **NBVCP** (Negative Binomial with Varying Cell Proportions): Models both gene
  expression and cell type proportions simultaneously.

- **ZINBVCP** (Zero-Inflated NBVCP): Combines zero-inflation with varying cell
  proportions modeling.

### Mixture Models

Each model type also supports mixture variants (suffix `_mix`):
- `nbdm_mix`: Mixture of NBDM components
- `zinb_mix`: Mixture of ZINB components  
- `nbvcp_mix`: Mixture of NBVCP components
- `zinbvcp_mix`: Mixture of ZINBVCP components

## Parameterizations

SCRIBE provides three different parameterizations for each model type, allowing
for flexible modeling approaches:

### Standard Parameterization
- **Module**: `standard.py`, `standard_unconstrained.py`
- **Parameters**: Direct parameterization using success probabilities (p) and
  dispersion (r)
- **Distributions**: Beta for probabilities, LogNormal for dispersion
- **Use Case**: Most interpretable, good for exploratory analysis

### Linked Parameterization  
- **Module**: `linked.py`, `linked_unconstrained.py`
- **Parameters**: Links success probabilities to mean expression (μ)
- **Distributions**: Beta for probabilities, LogNormal for means
- **Use Case**: When modeling relationships between probability and expression
  level

### Odds Ratio Parameterization
- **Module**: `odds_ratio.py`, `odds_ratio_unconstrained.py`  
- **Parameters**: Uses odds ratios (φ) instead of probabilities
- **Distributions**: BetaPrime for odds ratios, LogNormal for means
- **Use Case**: When odds ratios provide more natural interpretation

## Constrained vs Unconstrained

Each parameterization comes in two variants:

- **Constrained**: Parameters are sampled from their natural constrained
  distributions (e.g., Beta for probabilities)
- **Unconstrained**: Parameters are transformed to unconstrained space using
  bijective transformations for improved optimization

The unconstrained variants often provide better convergence properties for
variational inference.

## Inference Methods

### SVI (Stochastic Variational Inference)
- **Modules**: All non-VAE modules
- **Approach**: Uses mean-field or low-rank multivariate normal variational
  families
- **Guide Types**: 
  - Mean-field: Independent normal distributions for each parameter
  - Low-rank: Low-rank multivariate normal approximation
- **Use Case**: Fast inference, good for large datasets

### VAE (Variational Autoencoder)
- **Modules**: `vae_*.py` files
- **Approach**: Neural network-based encoder-decoder architecture
- **Prior Types**:
  - Standard: Traditional VAE with standard normal priors
  - Decoupled: Separate modeling of different parameter groups
- **Use Case**: Complex non-linear relationships, representation learning

## File Organization

```
models/
├── __init__.py                     # Package exports and imports
├── model_config.py                 # Unified configuration class
├── model_registry.py               # Dynamic model/guide retrieval
├── model_utils.py                  # Utility functions
├── log_likelihood.py               # Log-likelihood functions
│
├── standard.py                     # Standard parameterization (constrained)
├── standard_unconstrained.py       # Standard parameterization (unconstrained)
├── standard_low_rank.py            # Standard with low-rank guides
├── standard_low_rank_unconstrained.py
│
├── linked.py                       # Linked parameterization (constrained)  
├── linked_unconstrained.py         # Linked parameterization (unconstrained)
├── linked_low_rank.py              # Linked with low-rank guides
├── linked_low_rank_unconstrained.py
│
├── odds_ratio.py                   # Odds ratio parameterization (constrained)
├── odds_ratio_unconstrained.py     # Odds ratio parameterization (unconstrained)
├── odds_ratio_low_rank.py          # Odds ratio with low-rank guides
├── odds_ratio_low_rank_unconstrained.py
│
├── vae_core.py                     # Core VAE functionality
├── vae_standard.py                 # VAE with standard parameterization
├── vae_standard_unconstrained.py   # VAE standard (unconstrained)
├── vae_linked.py                   # VAE with linked parameterization
├── vae_linked_unconstrained.py     # VAE linked (unconstrained)
├── vae_odds_ratio.py               # VAE with odds ratio parameterization
└── vae_odds_ratio_unconstrained.py # VAE odds ratio (unconstrained)
```

## Usage

### Basic Model Retrieval

```python
from scribe.models import get_model_and_guide

# Get a standard NBDM model with SVI inference
model_fn, guide_fn = get_model_and_guide(
    model_type="nbdm",
    parameterization="standard",
    inference_method="svi"
)

# Get an unconstrained linked ZINB model with low-rank guide
model_fn, guide_fn = get_model_and_guide(
    model_type="zinb", 
    parameterization="linked",
    inference_method="svi",
    unconstrained=True,
    guide_rank=10
)

# Get a VAE model factory
vae_factory, _ = get_model_and_guide(
    model_type="nbdm",
    parameterization="standard", 
    inference_method="vae",
    prior_type="standard"
)
```

### Model Configuration

#### Using the Factory Method (Recommended)

The `from_inference_params()` factory method is the recommended way to create
ModelConfig instances:

```python
from scribe.models import ModelConfig
from scribe.utils import ParameterCollector

# Collect parameters
prior_config = ParameterCollector.collect_and_map_priors(
    unconstrained=False,
    parameterization="standard",
    r_prior=(1.0, 1.0),
    p_prior=(2.0, 0.5)
)

vae_config = ParameterCollector.collect_vae_params(
    vae_latent_dim=5,
    vae_hidden_dims=[256, 128],
    vae_activation="gelu"
)

# Create config using factory method
config = ModelConfig.from_inference_params(
    model_type="nbdm",
    inference_method="svi",
    prior_config=prior_config,
    n_components=3,
    guide_rank=10
)

# VAE inference with VAE config
vae_config_obj = ModelConfig.from_inference_params(
    model_type="zinb",
    inference_method="vae",
    vae_config=vae_config,
    prior_config=prior_config
)

# MCMC inference, unconstrained
mcmc_config = ModelConfig.from_inference_params(
    model_type="nbdm",
    inference_method="mcmc",
    unconstrained=True,
    prior_config=prior_config
)
```

#### Direct Instantiation

You can also create ModelConfig directly (for advanced use cases):

```python
from scribe.models import ModelConfig

# Create configuration for a mixture model
config = ModelConfig(
    base_model="zinb_mix",
    parameterization="linked",
    unconstrained=True,
    n_components=3,
    inference_method="svi"
)
config.validate()  # Don't forget to validate!
```

### Log-Likelihood Functions

```python
from scribe.models import get_log_likelihood_fn

# Get log-likelihood function for model evaluation
ll_fn = get_log_likelihood_fn("nbdm")
```

## Key Features

- **Unified Interface**: All models share a common interface through the
  registry system
- **Flexible Parameterization**: Choose the most appropriate parameterization
  for your data
- **Scalable Inference**: Both SVI and VAE methods for different computational
  requirements  
- **Mixture Models**: Support for modeling heterogeneous cell populations
- **Extensible Design**: Easy to add new model types and parameterizations

## Model Selection Guidelines

- **Start with NBDM**: Good baseline for most single-cell datasets
- **Use ZINB**: When your data has excessive zeros beyond what NB can model
- **Consider NBVCP**: When cell type proportions are important
- **Try mixtures**: For datasets with clear subpopulations
- **Standard parameterization**: Most interpretable, good starting point
- **Unconstrained**: Better optimization, use when convergence is difficult
- **VAE**: When you need representation learning or have complex data

## Dependencies

- JAX/NumPyro for probabilistic modeling
- Flax for neural network components (VAE models)
- Standard scientific Python stack (NumPy, SciPy)

# Welcome to SCRIBE

**SCRIBE** (Single-Cell RNA-Seq Inference using Bayesian Estimation) is a
Python package for analyzing single-cell RNA sequencing (scRNA-seq) data using
variational inference based on
[NumPyro](https://num.pyro.ai/en/stable/index.html) — a
[JAX](https://jax.readthedocs.io/en/latest/)-based probabilistic programming
library with GPU acceleration. It provides a collection of probabilistic models
and inference tools specifically designed for scRNA-seq count data.

## Features

- **Multiple probabilistic models** for scRNA-seq data analysis
- **Efficient variational inference** using JAX and NumPyro
- **Full-batch and mini-batch inference** for large-scale data
- **Integration with AnnData** objects
- **Comprehensive visualization tools** for posterior analysis
- **GPU acceleration** support
- **Three inference methods**: SVI, MCMC, and VAE through a unified interface

## Available Models

SCRIBE includes several probabilistic models for scRNA-seq data:

| Model | Description |
|-------|-------------|
| [**NBDM**](models/nbdm.md) | Negative Binomial-Dirichlet Multinomial — models both count magnitudes and proportions, accounts for overdispersion |
| [**ZINB**](models/zinb.md) | Zero-Inflated Negative Binomial — handles excess zeros, models technical and biological dropouts |
| [**NBVCP**](models/nbvcp.md) | NB with Variable Capture Probability — accounts for cell-specific mRNA capture efficiency |
| [**ZINBVCP**](models/zinbvcp.md) | Zero-Inflated NB with Variable Capture Probability — most comprehensive model for technical variation |
| [**Mixture Models**](models/mixture.md) | Any of the above can be extended to mixture models for subpopulation analysis |

## Quick Example

```python
import scribe

# Run inference with the basic NBDM model
results = scribe.run_scribe(counts, inference_method="svi", n_steps=100_000)

# Add zero-inflation handling
results = scribe.run_scribe(
    counts, inference_method="svi", zero_inflated=True
)

# Mixture model for multiple cell populations
results = scribe.run_scribe(
    counts, inference_method="svi", mixture_model=True, n_components=3
)
```

## Getting Started

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install SCRIBE and set up your environment

    [:octicons-arrow-right-24: Installation guide](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } **Quick Overview**

    ---

    Understand the probabilistic approach behind SCRIBE

    [:octicons-arrow-right-24: Quick overview](getting-started/quick-overview.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Run your first inference in minutes

    [:octicons-arrow-right-24: Quickstart tutorial](getting-started/quickstart.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Full reference for all modules and classes

    [:octicons-arrow-right-24: API reference](reference/)

</div>

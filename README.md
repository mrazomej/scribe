# SCRIBE: Single-Cell RNA-seq Inference using Bayesian Estimation

SCRIBE is a `Python` package for analyzing single-cell RNA sequencing
(scRNA-seq) data using variational inference based on `Numpyro`—a `Jax`-based
probabilistic programming library with GPU acceleration. It provides a
collection of probabilistic models and inference tools specifically designed for
scRNA-seq count data.

## Features

- Multiple probabilistic models for scRNA-seq data analysis
- Efficient variational inference using JAX and Numpyro
- Support for both full-batch and mini-batch inference
- Integration with AnnData objects
- Comprehensive visualization tools for posterior analysis
- GPU acceleration support

## Available Models

SCRIBE includes several probabilistic models for scRNA-seq data:

1. **Negative Binomial-Dirichlet Multinomial (NBDM)**
   - Models both count magnitudes and proportions
   - Accounts for overdispersion in count data

2. **Zero-Inflated Negative Binomial (ZINB)**
   - Handles excess zeros in scRNA-seq data
   - Models technical and biological dropouts
   - Includes gene-specific dropout rates

3. **Negative Binomial with Variable Capture Probability (NBVCP)**
   - Accounts for cell-specific mRNA capture efficiency
   - Models technical variation in library preparation
   - Suitable for datasets with varying sequencing depths per cell

4. **Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP)**
   - Combines zero-inflation and variable capture probability
   - Most comprehensive model for technical variation
   - Handles both dropouts and capture efficiency

## Installation

### Using pip

```bash
pip install scribe
```

### Development Installation

For the latest development version:

```bash
git clone https://github.com/mrazomej/scribe.git
cd scribe
pip install -e ".[dev]"
```

### Docker Installation

```bash
# Build the Docker image
docker build -t scribe .

# Run the container
docker run --gpus all -it scribe
```

## Quick Start

Here's a simple example of how to use SCRIBE:

```python
import scribe
from jax import random

# Load your data (using AnnData)
import anndata as ad
adata = ad.read_h5ad("your_data.h5ad")

# Run inference
results = scribe.run_scribe(
    adata,
    model_type="zinb",  # Choose your model
    n_steps=100_000,    # Number of optimization steps
    batch_size=512,     # Mini-batch size
    rng_key=random.PRNGKey(0)
)

# Generate posterior predictive samples
ppc_samples = results.ppc_samples(n_samples=100)

# Visualize results
scribe.viz.plot_parameter_posteriors(results)
```

## Advanced Usage

### Model Selection

Choose the appropriate model based on your data characteristics:

- Use `NBDM` for well-behaved datasets with minimal technical artifacts.
- Use `ZINB` when dropout is a significant concern.
- Use `NBVCP` when capture efficiency varies significantly between cells.
- Use `ZINBVCP` when both dropout and capture efficiency are concerns.

### Customizing Inference

```python
# Custom prior parameters
prior_params = {
    'p_prior': (1.0, 1.0),
    'r_prior': (2.0, 0.1),
    'gate_prior': (1.0, 1.0)
}

# Run inference with custom parameters
results = scribe.run_scribe(
    adata,
    model_type="zinb",
    prior_params=prior_params,
    n_steps=200_000,
    batch_size=1024
)
```

### GPU Acceleration

SCRIBE automatically uses GPU if available. To explicitly control device usage:

```python
with scribe.utils.use_cpu():
    # Force CPU execution
    results = scribe.run_scribe(adata)
```

## Documentation

For detailed documentation, please visit [our documentation
site](https://scribe.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing
Guidelines](CONTRIBUTING.md) for more information.

## Citation

If you use SCRIBE in your research, please cite:

```bibtex
@software{scribe2024,
  author = {Razo, Manuel},
  title = {SCRIBE: Single-Cell RNA-seq Inference using Bayesian Estimation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/mrazomej/scribe}
}
```

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.

## Acknowledgments

SCRIBE builds upon several excellent libraries:

- [JAX](https://github.com/google/jax) for automatic differentiation and GPU acceleration
- [Numpyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [AnnData](https://anndata.readthedocs.io/) for data management
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization

## Support

For questions and support:

- Create an issue in the [GitHub repository](https://github.com/mrazomej/scribe/issues)
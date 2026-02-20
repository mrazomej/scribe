# SCRIBE: Single-Cell RNA-seq Inference with Bayesian Estimation

SCRIBE is a comprehensive Python package for Bayesian analysis of single-cell
RNA sequencing (scRNA-seq) data. Built on JAX and NumPyro, SCRIBE provides a
unified framework for probabilistic modeling, variational inference, and
uncertainty quantification in single-cell genomics.

## Why SCRIBE?

🎯 **Unified Framework**: Single interface for SVI, MCMC, and VAE inference
methods  
🧬 **Specialized Models**: Four probabilistic models designed specifically for
scRNA-seq data  
⚡ **GPU Accelerated**: JAX-based implementation with automatic GPU support  
📊 **Rich Analysis**: Comprehensive posterior analysis and uncertainty
quantification  
🔧 **Flexible**: Multiple parameterizations and constrained/unconstrained
variants  
📈 **Scalable**: From small experiments to large-scale atlases with mini-batch
support

## Key Features

- **Three Inference Methods**: 
  - SVI for speed and scalability
  - MCMC for exact Bayesian inference  
  - VAE for representation learning
- **Specialized scRNA-seq Models**: NBDM, ZINB, NBVCP, ZINBVCP
- **Advanced Analysis**: Mixture models for cell type discovery
- **Seamless Integration**: Works with AnnData and the scanpy ecosystem
- **Professional Visualization**: Comprehensive plotting tools for results
  analysis
- **Split-Safe UMAP Caching**: Visualization cache keys include dataset split
  metadata to avoid cross-split cache reuse

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

4. **Zero-Inflated Negative Binomial with Variable Capture Probability
   (ZINBVCP)**
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

Get started with SCRIBE in just a few lines:

```python
import scribe
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Run SCRIBE with default settings (SVI inference, NBDM model)
results = scribe.fit(adata, model="nbdm")

# Analyze results
posterior_samples = results.get_posterior_samples()
log_likelihood = results.log_likelihood()

# Visualize
scribe.viz.plot_loss_history(results.loss_history)
```

### Customize with Simple Arguments

```python
# Zero-inflated model with more optimization steps
results = scribe.fit(
    adata,
    model="zinb",
    n_steps=100000,
    batch_size=512,
)

# Linked parameterization with low-rank guide
results = scribe.fit(
    adata,
    model="nbdm",
    parameterization="linked",
    guide_rank=15,
)

# Mixture model for cell type discovery
results = scribe.fit(
    adata,
    model="zinb",
    n_components=3,
    n_steps=150000,
)
```

### Choose Your Inference Method

```python
# Fast exploration with SVI (default)
svi_results = scribe.fit(adata, model="zinb", n_steps=75000)

# Exact inference with MCMC
mcmc_results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="mcmc",
    n_samples=3000,
    n_chains=4,
)

# Representation learning with VAE
vae_results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="vae",
    n_steps=50000,
)
```

## Advanced Usage

### Mixture Models for Cell Type Discovery

```python
# Simple: just specify n_components
mixture_results = scribe.fit(
    adata,
    model="nbdm",
    n_components=5,
    n_steps=100000,
)

# Analyze cell type assignments
cell_types = mixture_results.cell_type_probabilities()

# Access individual components
for i in range(5):
    component = mixture_results.get_component(i)
    print(f"Component {i} MAP estimates:", component.get_map())
```

### Model Selection Guide

Choose the right model for your data:

| Model       | Best For                    | Key Features                |
| ----------- | --------------------------- | --------------------------- |
| **NBDM**    | Baseline analysis           | Simple, interpretable, fast |
| **ZINB**    | Data with excess zeros      | Handles dropouts            |
| **NBVCP**   | Variable sequencing depth   | Models capture efficiency   |
| **ZINBVCP** | Complex technical variation | Most comprehensive          |

### Multi-Method Workflow

```python
# Fast exploration with SVI
svi_results = scribe.fit(
    adata,
    model="zinb",
    n_components=3,
    n_steps=50000,
)

# Exact inference for final analysis
mcmc_results = scribe.fit(
    adata,
    model="zinb",
    n_components=3,
    inference_method="mcmc",
    n_samples=2000,
    n_chains=4,
)
```

### Integration with Single-Cell Ecosystem

```python
import scanpy as sc

# Standard scanpy preprocessing
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# SCRIBE analysis with VAE for representation learning
results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="vae",
    n_steps=50000,
)

# Add results back to AnnData
adata.obsm["X_scribe"] = results.get_latent_embeddings(adata.X)
adata.obs["scribe_cluster"] = results.cell_type_probabilities().argmax(axis=1)

# Continue with scanpy
sc.pp.neighbors(adata, use_rep="X_scribe")
sc.tl.umap(adata)
sc.pl.umap(adata, color="scribe_cluster")
```

### Power User: Explicit Configuration Objects

For full control, you can still use explicit configuration objects:

```python
from scribe.models.config import ModelConfigBuilder, InferenceConfig, SVIConfig

# Build detailed model configuration
model_config = (
    ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .unconstrained()
    .as_mixture(n_components=3)
    .build()
)

# Custom inference configuration
inference_config = InferenceConfig.from_svi(
    SVIConfig(n_steps=100000, batch_size=512)
)

# Run with explicit configs
results = scribe.fit(
    adata,
    model_config=model_config,
    inference_config=inference_config,
)
```

## Performance & Scalability

SCRIBE is designed for real-world single-cell datasets:

- **GPU Acceleration**: Automatic GPU detection and usage
- **Memory Efficient**: Mini-batch processing for large datasets
- **Scalable**: Tested on datasets from hundreds to hundreds of thousands of
  cells
- **Fast**: SVI inference typically completes in minutes

```python
# For large datasets - just add batch_size
large_results = scribe.fit(
    large_adata,
    model="nbdm",
    n_steps=150000,
    batch_size=1024,
)
```

## Documentation

Comprehensive documentation is available in each module:

- **[Package Overview](src/scribe/README.md)**: Complete package documentation
- **[Models](src/scribe/models/README.md)**: Probabilistic model details
- **[SVI](src/scribe/svi/README.md)**: Stochastic variational inference
- **[MCMC](src/scribe/mcmc/README.md)**: Markov Chain Monte Carlo
- **[VAE](src/scribe/vae/README.md)**: Variational autoencoders
- **[Core](src/scribe/core/README.md)**: Shared utilities and preprocessing

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

- [JAX](https://github.com/google/jax) for automatic differentiation and GPU
  acceleration
- [Numpyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [AnnData](https://anndata.readthedocs.io/) for data management
- [Matplotlib](https://matplotlib.org/) and
  [Seaborn](https://seaborn.pydata.org/) for visualization

## Support

For questions and support:

- Create an issue in the [GitHub
  repository](https://github.com/mrazomej/scribe/issues)
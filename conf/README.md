# SCRIBE Configuration Guide

This directory contains Hydra configuration files for SCRIBE's inference and
visualization pipelines. The configuration system allows you to easily manage
complex experimental setups, run parameter sweeps, and maintain reproducible
results.

## 🎯 Quick Start

```bash
# Run inference with default settings
python run_hydra.py

# Run inference with specific data and model
python run_hydra.py data=jurkat_cells inference=svi zero_inflated=true

# Generate visualizations from inference results
python viz_hydra.py data=jurkat_cells inference=svi zero_inflated=true

# Override specific parameters
python run_hydra.py data=jurkat_cells inference.n_steps=100000 inference.batch_size=1024
```

## 📁 Configuration Structure

```
conf/
├── config.yaml           # Main configuration file
├── data/                  # Data-specific configurations
│   ├── jurkat_cells.yaml
│   ├── singer.yaml
│   └── 5050mix.yaml
├── inference/             # Inference method configurations
│   ├── svi.yaml          # Stochastic Variational Inference
│   ├── mcmc.yaml         # Markov Chain Monte Carlo
│   └── vae.yaml          # Variational Autoencoder
├── model/                 # Model-specific configurations
├── viz/                   # Visualization configurations
│   └── default.yaml
└── hydra/                 # Hydra framework settings
```

## ⚙️ Main Configuration (`config.yaml`)

The main configuration file defines the complete experimental setup:

### Data Configuration
```yaml
defaults:
  - data: singer          # Choose dataset configuration
  - inference: svi        # Choose inference method
```

### Model Parameters
```yaml
# Model Selection
zero_inflated: false      # Use zero-inflated models (ZINB/ZINBVCP)
variable_capture: false   # Use variable capture probability models
mixture_model: false      # Enable mixture models for cell type discovery
n_components: null        # Number of mixture components (auto if null)

# Parameterization Options
parameterization: "standard"  # "standard", "linked", "odds_ratio"
unconstrained: false          # Use unconstrained parameterization
```

### Prior Configuration
```yaml
# Customize priors for Bayesian inference
r_prior: null            # Negative binomial dispersion prior
p_prior: null            # Success probability prior
gate_prior: null         # Zero-inflation gate prior
p_capture_prior: null    # Capture probability prior
mixing_prior: null       # Mixture component prior
```

### Visualization Settings
```yaml
viz:
  loss: true             # Plot loss/ELBO history
  ecdf: true             # Plot empirical CDFs
  ppc: true              # Plot posterior predictive checks
  format: png            # Output format: png, pdf, svg, eps
  
  ecdf_opts:
    n_genes: 25          # Number of genes for ECDF plots
  
  ppc_opts:
    n_genes: 25          # Number of genes for PPC plots
    n_samples: 1500      # Posterior predictive samples
```

## 📊 Data Configurations (`data/`)

Each dataset has its own configuration file:

### Jurkat Cells (`data/jurkat_cells.yaml`)
```yaml
# @package data
name: "Jurkat_cells"
path: "data/10xGenomics/Jurkat_cells/data.h5ad"
```

### Singer Dataset (`data/singer.yaml`)
```yaml
# @package data
name: "singer"
path: "data/singer/data.h5ad"
preprocessing:
  min_cells: 3
  min_genes: 200
```

### Custom Dataset
Create your own data configuration:
```yaml
# @package data
name: "my_dataset"
path: "path/to/my_data.h5ad"
preprocessing:
  min_cells: 5
  min_genes: 100
  max_genes: 5000
```

## 🔬 Inference Configurations (`inference/`)

### Stochastic Variational Inference (`inference/svi.yaml`)
```yaml
# @package inference
method: svi
n_steps: 50000           # Number of optimization steps
batch_size: null         # Mini-batch size (null = full batch)
stable_update: true      # Use stable parameter updates
learning_rate: 0.001     # Adam optimizer learning rate
```

### MCMC (`inference/mcmc.yaml`)
```yaml
# @package inference
method: mcmc
n_samples: 2000          # Number of MCMC samples
n_chains: 4              # Number of parallel chains
n_warmup: 1000           # Warmup samples
target_accept_prob: 0.8  # Target acceptance probability
```

### VAE (`inference/vae.yaml`)
```yaml
# @package inference
method: vae
n_steps: 50000           # Training steps
latent_dim: 15           # Latent space dimensionality
encoder_layers: [128, 64] # Encoder architecture
decoder_layers: [64, 128] # Decoder architecture
```

## 🎨 Visualization Configuration (`viz/default.yaml`)

Control visualization output:
```yaml
# @package _global_
run_dir: ???             # Path to inference results (auto-detected)

viz:
  loss: true             # Plot training loss
  ecdf: true             # Plot empirical cumulative distribution
  ppc: true              # Plot posterior predictive checks
  format: png            # png, pdf, svg, eps
  
  ecdf_opts:
    n_genes: 25          # Genes to include in ECDF
  
  ppc_opts:
    n_genes: 25          # Genes for posterior predictive checks
    n_samples: 1500      # Number of predictive samples
```

## 🚀 Advanced Usage

### Parameter Sweeps
```bash
# Sweep over multiple parameters
python run_hydra.py -m data=jurkat_cells,singer inference.n_steps=25000,50000,100000

# Bayesian model comparison
python run_hydra.py -m zero_inflated=true,false variable_capture=true,false
```

### Mixture Model Analysis
```bash
# Cell type discovery
python run_hydra.py data=jurkat_cells mixture_model=true n_components=3 inference.n_steps=100000

# Component-specific parameters
python run_hydra.py mixture_model=true component_specific_params=true
```

### Custom Parameterizations
```bash
# Different parameterizations for specialized analysis
python run_hydra.py parameterization=linked zero_inflated=true
python run_hydra.py parameterization=odds_ratio mixture_model=true
```

### High-Performance Computing
```bash
# Large-scale analysis
python run_hydra.py inference.batch_size=2048 inference.n_steps=200000

# Memory-efficient processing
python run_hydra.py cells_axis=0 inference.batch_size=512
```

## 📈 Visualization Workflows

### Basic Visualization
```bash
# Generate all plots in PNG format
python viz_hydra.py data=jurkat_cells inference=svi

# Generate publication-ready PDFs
python viz_hydra.py data=jurkat_cells inference=svi viz.format=pdf
```

### Custom Visualization
```bash
# Focus on specific plots
python viz_hydra.py viz.loss=false viz.ppc=true viz.ppc_opts.n_genes=50

# High-resolution analysis
python viz_hydra.py viz.ppc_opts.n_samples=5000 viz.ecdf_opts.n_genes=100
```

## 🔧 Configuration Tips

### 1. **Reproducibility**
Always specify a seed for reproducible results:
```yaml
seed: 42
```

### 2. **Memory Management**
For large datasets, use batching:
```yaml
inference:
  batch_size: 1024
```

### 3. **Model Selection**
Start simple and add complexity:
```bash
# Baseline
python run_hydra.py

# Add zero-inflation
python run_hydra.py zero_inflated=true

# Add mixture modeling
python run_hydra.py zero_inflated=true mixture_model=true n_components=3
```

### 4. **Convergence**
Monitor convergence and adjust steps:
```yaml
inference:
  n_steps: 100000  # Increase for complex models
```

### 5. **Output Organization**
Hydra automatically organizes outputs:
```
outputs/
├── {dataset}/
│   ├── {inference_method}/
│   │   ├── {parameters}/
│   │   │   ├── scribe_results.pkl
│   │   │   ├── figs/
│   │   │   └── .hydra/
```

## 🐛 Troubleshooting

### Common Issues

**1. Configuration Not Found**
```bash
# Error: Could not override 'data'
# Solution: Use + prefix for new parameters
python run_hydra.py +my_param=value
```

**2. Memory Issues**
```bash
# Reduce batch size or use gradient checkpointing
python run_hydra.py inference.batch_size=256
```

**3. Convergence Problems**
```bash
# Increase steps or adjust learning rate
python run_hydra.py inference.n_steps=100000 inference.learning_rate=0.0005
```

## 📚 Integration Examples

### With Scanpy
```python
import scanpy as sc
import scribe

# Load and preprocess
adata = sc.read_h5ad("data.h5ad")
sc.pp.filter_cells(adata, min_genes=200)

# Run SCRIBE via configuration
# (equivalent to running run_hydra.py)
results = scribe.run_scribe(
    counts=adata.X,
    inference_method="svi",
    zero_inflated=True,
    n_steps=50000
)
```

### Programmatic Configuration
```python
from omegaconf import OmegaConf
import hydra

# Load configuration
with hydra.initialize(config_path="conf"):
    cfg = hydra.compose(config_name="config", 
                       overrides=["data=jurkat_cells", "zero_inflated=true"])

# Use in analysis
results = scribe.run_scribe(counts=data, **OmegaConf.to_container(cfg))
```

## 🎯 Best Practices

1. **Start Simple**: Begin with default configurations and add complexity
   gradually
2. **Version Control**: Track configuration changes alongside code
3. **Document Experiments**: Use descriptive override names
4. **Monitor Resources**: Watch memory and GPU usage for large datasets
5. **Validate Results**: Always check convergence and posterior predictive
   checks

## 📖 Further Reading

- [Hydra Documentation](https://hydra.cc/) for advanced configuration patterns
- [SCRIBE Models](../src/scribe/models/README.md) for model-specific details
- [SCRIBE Core](../src/scribe/core/README.md) for preprocessing options
- Main [README.md](../README.md) for package overview

## 🆘 Support

For configuration-related questions:
- Check the [main documentation](../README.md)
- Create an issue in the [GitHub
  repository](https://github.com/mrazomej/scribe/issues)
- Review example configurations in this directory

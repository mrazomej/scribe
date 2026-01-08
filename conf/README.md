# SCRIBE Configuration Guide

This directory contains Hydra configuration files for SCRIBE's inference and
visualization pipelines. The configuration system allows you to easily manage
complex experimental setups, run parameter sweeps, and maintain reproducible
results.

## 🎯 Quick Start

```bash
# Run inference with default settings (includes visualization)
python infer.py

# Run inference with specific data and model
python infer.py data=jurkat_cells inference=svi zero_inflated=true

# Run inference without visualization
python infer.py data=jurkat_cells inference=svi viz=null

# Generate visualizations from existing inference results
python visualize.py data=jurkat_cells inference=svi zero_inflated=true

# Override specific parameters
python infer.py data=jurkat_cells inference.n_steps=100000 inference.batch_size=1024

# Enable specific plots only
python infer.py data=jurkat_cells viz.loss=true viz.ppc=true viz.ecdf=false
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
  umap: true             # Plot UMAP projection (experimental vs synthetic data)
  heatmap: false         # Plot correlation heatmap (disabled by default, requires posterior sampling)
  mixture_ppc: false     # Plot mixture model PPCs (only for mixture models)
  format: png            # Output format: png, pdf, svg, eps
  
  ecdf_opts:
    n_genes: 25          # Number of genes for ECDF plots
  
  ppc_opts:
    n_rows: 6            # Number of expression bins (logarithmically spaced)
    n_cols: 6            # Number of genes per bin
    n_samples: 1500      # Posterior predictive samples
  
  umap_opts:
    n_neighbors: 15      # Number of neighbors for UMAP
    min_dist: 0.1        # Minimum distance for UMAP
    n_components: 2      # Number of UMAP dimensions
    random_state: 42     # Random seed for reproducibility
    batch_size: 1000     # Batch size for PPC sampling (None = no batching, use for memory efficiency)
    data_color: "dark_blue"      # Color for experimental data
    synthetic_color: "dark_red"  # Color for synthetic data
  
  heatmap_opts:
    n_genes: 500         # Number of genes to display (selected by correlation variance)
    n_samples: 256       # Posterior samples for correlation computation
    figsize: 12          # Figure size (square)
    cmap: "RdBu_r"       # Colormap (red=positive, blue=negative)
  
  mixture_ppc_opts:
    n_rows: 6            # Number of rows in the PPC grid
    n_cols: 6            # Number of columns in the PPC grid
    n_samples: 500       # Number of posterior predictive samples
    n_bins: 6            # Number of expression bins for gene selection
```

**Note**: Visualization is enabled by default when running `infer.py`. Use
`viz=null` to disable all plots, or `viz.loss=false` to disable specific plots.

**PPC Gene Selection**: The PPC plots use a log-spaced binning strategy to
ensure good coverage across the expression range. Genes are divided into
`n_rows` logarithmically-spaced expression bins, and `n_cols` genes are selected
from each bin (also using logarithmic spacing within the bin). This ensures
representation of both low-expression and high-expression genes while avoiding
over-representation of very low-expression genes.

**UMAP Projection**: The UMAP plot provides a 2D visualization comparing
experimental and synthetic data. UMAP is first fitted on the experimental data,
then a single posterior predictive sample (one sample per gene per cell) is
generated and projected onto the same UMAP space. This allows visual comparison
of how well the model captures the structure of the experimental data. Requires
`umap-learn` package (`pip install umap-learn`). The `batch_size` parameter can
be used to process cells in batches during PPC sampling to avoid memory issues
on large datasets.

**Correlation Heatmap**: The heatmap displays pairwise Pearson correlations
between genes computed from posterior samples. Genes are selected by correlation
variance (most informative structure) and displayed with hierarchical clustering
and dendrograms. This visualization helps identify gene co-expression patterns
learned by the model. Disabled by default as it requires posterior sampling.

**Mixture PPC**: For mixture models, this generates specialized posterior
predictive checks that highlight genes with the highest variability between
mixture components. It produces multiple plots: (1) a combined mixture PPC
showing the weighted average across all components (blue), and (2) separate
per-component PPCs using distinct colormaps (green, purple, red, orange, etc.).
Genes are selected using the coefficient of variation (CV) of MAP estimates
across components, with expression-based binning to ensure representation across
the full expression range. This visualization helps assess how well individual
components capture gene expression patterns and identify genes that distinguish
cell types. Only available when `mixture_model=true`.

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
  umap: true             # Plot UMAP projection (experimental vs synthetic)
  heatmap: false         # Plot correlation heatmap (requires posterior sampling)
  mixture_ppc: false     # Plot mixture model PPCs (requires mixture_model=true)
  format: png            # png, pdf, svg, eps
  
  ecdf_opts:
    n_genes: 25          # Genes to include in ECDF
  
  ppc_opts:
    n_rows: 6            # Number of expression bins (logarithmically spaced)
    n_cols: 6            # Number of genes per bin
    n_samples: 1500      # Number of predictive samples
  
  umap_opts:
    n_neighbors: 15      # Number of neighbors for UMAP
    min_dist: 0.1        # Minimum distance for UMAP
    n_components: 2      # Number of UMAP dimensions
    random_state: 42     # Random seed for reproducibility
    batch_size: 1000     # Batch size for PPC sampling
    data_color: "dark_blue"      # Color for experimental data
    synthetic_color: "dark_red"  # Color for synthetic data
  
  heatmap_opts:
    n_genes: 500         # Number of genes to display
    n_samples: 256       # Posterior samples for correlation
    figsize: 12          # Figure size (square)
    cmap: "RdBu_r"       # Colormap
  
  mixture_ppc_opts:
    n_rows: 6            # Number of rows in the PPC grid
    n_cols: 6            # Number of columns in the PPC grid
    n_samples: 500       # Number of posterior predictive samples
    n_bins: 6            # Number of expression bins for gene selection
```

## 🚀 Advanced Usage

### Parameter Sweeps
```bash
# Sweep over multiple parameters
python infer.py -m data=jurkat_cells,singer inference.n_steps=25000,50000,100000

# Bayesian model comparison
python infer.py -m zero_inflated=true,false variable_capture=true,false
```

### Mixture Model Analysis
```bash
# Cell type discovery
python infer.py data=jurkat_cells mixture_model=true n_components=3 inference.n_steps=100000

# Component-specific parameters
python infer.py mixture_model=true component_specific_params=true

# Visualize component-specific PPCs (genes that differ between components)
python visualize.py mixture_model=true n_components=3 viz.mixture_ppc=true
```

### Custom Parameterizations
```bash
# Different parameterizations for specialized analysis
python infer.py parameterization=linked zero_inflated=true
python infer.py parameterization=odds_ratio mixture_model=true
```

### High-Performance Computing
```bash
# Large-scale analysis
python infer.py inference.batch_size=2048 inference.n_steps=200000

# Memory-efficient processing
python infer.py cells_axis=0 inference.batch_size=512
```

## 📈 Visualization Workflows

### Integrated Visualization (Recommended)
```bash
# Run inference with automatic visualization (default)
python infer.py data=jurkat_cells inference=svi

# Generate publication-ready PDFs
python infer.py data=jurkat_cells inference=svi viz.format=pdf

# Enable only specific plots
python infer.py data=jurkat_cells viz.loss=true viz.ppc=true viz.ecdf=false viz.umap=true
```

### Standalone Visualization
```bash
# Generate all plots from existing results
python visualize.py data=jurkat_cells inference=svi

# Generate publication-ready PDFs
python visualize.py data=jurkat_cells inference=svi viz.format=pdf

# Focus on specific plots
python visualize.py viz.loss=false viz.ppc=true viz.ppc_opts.n_rows=6 viz.ppc_opts.n_cols=8

# Enable/disable UMAP projection plot
python visualize.py viz.umap=true

# Customize UMAP parameters
python visualize.py viz.umap=true viz.umap_opts.n_neighbors=30 viz.umap_opts.min_dist=0.2

# Enable correlation heatmap (requires posterior sampling)
python visualize.py viz.heatmap=true viz.heatmap_opts.n_genes=300

# Enable mixture PPC (only for mixture models)
python visualize.py data=5050mix mixture_model=true n_components=2 viz.mixture_ppc=true

# Customize mixture PPC options
python visualize.py mixture_model=true viz.mixture_ppc=true viz.mixture_ppc_opts.n_samples=1000

# High-resolution analysis
python visualize.py viz.ppc_opts.n_samples=5000 viz.ecdf_opts.n_genes=100
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
python infer.py

# Add zero-inflation
python infer.py zero_inflated=true

# Add mixture modeling
python infer.py zero_inflated=true mixture_model=true n_components=3
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
python infer.py +my_param=value
```

**2. Memory Issues**
```bash
# Reduce batch size or use gradient checkpointing
python infer.py inference.batch_size=256
```

**3. Convergence Problems**
```bash
# Increase steps or adjust learning rate
python infer.py inference.n_steps=100000 inference.learning_rate=0.0005
```

**4. Visualization Issues**
```bash
# Disable visualization if causing problems
python infer.py viz=null

# Or run visualization separately
python visualize.py data=your_data inference=your_method
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
# (equivalent to running infer.py)
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

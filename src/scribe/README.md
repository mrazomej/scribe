# SCRIBE: Single-Cell Bayesian Inference Ensemble

SCRIBE is a comprehensive Python package for Bayesian analysis of single-cell
RNA sequencing (scRNA-seq) data. Built on JAX and NumPyro, SCRIBE provides a
unified framework for probabilistic modeling, variational inference, and
uncertainty quantification in single-cell genomics.

## Overview

SCRIBE offers three complementary inference methods—**SVI** (Stochastic
Variational Inference), **MCMC** (Markov Chain Monte Carlo), and **VAE**
(Variational Autoencoder)—across multiple probabilistic models specifically
designed for single-cell count data. The package emphasizes flexibility,
scalability, and rigorous uncertainty quantification.

### Key Features

- **🎯 Simple API**: `scribe.fit()` with flat kwargs and sensible defaults
- **🧬 Specialized Models**: Four probabilistic models designed for scRNA-seq
  data
- **⚡ Multiple Inference**: SVI for speed, MCMC for accuracy, VAE for
  representation learning
- **🔧 Flexible Parameterization**: Three parameterizations with
  constrained/unconstrained variants
- **📊 Rich Analysis**: Comprehensive posterior analysis and visualization tools
- **🚀 GPU Acceleration**: JAX-based implementation with automatic GPU support
- **📈 Scalable**: From small experiments to large-scale atlases

## Quick Start

```python
import scribe
import scanpy as sc

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Run SCRIBE with default settings (SVI inference, NBVCP model)
results = scribe.fit(adata, model="nbdm")

# With customization
results = scribe.fit(
    adata,
    model="zinb",
    n_components=3,
    n_steps=100000,
    batch_size=512,
    optimizer_config={"name": "clipped_adam", "step_size": 5e-4, "grad_clip_norm": 1.0},
)

# Analyze results
posterior_samples = results.get_posterior_samples()
log_likelihood = results.log_likelihood()

# Visualize (CLI-style save to disk)
scribe.viz.plot_loss(
    results=results,
    figs_dir="figs",
    cfg=cfg,
    viz_cfg=viz_cfg,
)

# Visualize interactively in notebooks/scripts.
# All plot functions return a PlotResult which renders a single image
# via _repr_png_ / _repr_html_ — no duplicate display in notebooks.
# figs_dir, cfg, and viz_cfg are always optional.
result = scribe.viz.plot_loss(results=results)
result.fig      # underlying matplotlib Figure
result.axes     # tuple of Axes used
result.n_panels # number of logical panels

# Pre-build a figure and pass it in
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7, 3))
result = scribe.viz.plot_loss(
    results=results,
    fig=fig,      # multi-panel functions prefer `fig`
    save=False,   # do not write files
)

# Single-panel functions can accept ax=
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
result = scribe.viz.plot_ecdf(
    counts=counts_array,
    ax=ax,
    save=False,
)

# PPC-family functions expose n_rows, n_cols, n_genes, n_samples as
# direct keyword arguments — no viz_cfg needed for interactive use.
result = scribe.viz.plot_ppc(results, counts)           # sensible defaults
result = scribe.viz.plot_ppc(results, counts, n_genes=16, n_rows=4)
result = scribe.viz.plot_ppc(results, counts, n_rows=3, n_cols=3, n_samples=256)

# For library / extension authors: PlotContext encapsulates the
# save/show/close policy and filename construction boilerplate.
from scribe.viz import PlotContext
ctx = PlotContext.from_kwargs(
    figs_dir="figs", cfg=cfg, viz_cfg=viz_cfg,
    fig=None, ax=None, axes=None,
    save=None, show=None, close=None,
)
ctx.save           # resolved boolean
ctx.output_format  # e.g. "png"
ctx.build_filename("my_suffix", results=results)  # standardized name
ctx.finalize(fig, [ax], 1, filename=..., save_label="my plot")

# Multi-figure functions (mixture PPC, annotation PPC, correlation
# heatmap) return a PlotResultCollection. It renders all figures
# inline in notebooks and supports indexing/iteration.
collection = scribe.viz.plot_mixture_ppc(results=results, counts=counts)
len(collection)          # number of figures
collection[0].fig        # first figure
collection.output_paths  # list of saved paths (or Nones)
for result in collection:
    display(result)      # render each figure individually

# For new plot functions, use the @plot_function decorator to
# eliminate boilerplate. The decorator handles PlotContext creation,
# filename construction, saving, and finalization automatically.
from scribe.viz import plot_function
from scribe.viz._interactive import _create_or_validate_single_axis

@plot_function(suffix="my_diag", save_label="custom diagnostic",
               save_kwargs={"bbox_inches": "tight"})
def plot_custom(results, counts, *, ctx, viz_cfg=None,
                fig=None, ax=None, axes=None):
    fig, ax = _create_or_validate_single_axis(
        fig=fig, ax=ax, axes=axes, figsize=(6, 4),
    )
    ax.scatter(counts.mean(axis=0), results.some_metric)
    return fig, [ax], 1

# External callers see the standard API (figs_dir, cfg, etc.):
result = plot_custom(results, counts, figs_dir="figs", cfg=cfg)
```

## Unified Inference CLI

For Hydra-driven experiment execution, SCRIBE provides a unified CLI:

```bash
pip install 'scribe[hydra]'
scribe-infer --config-path ./conf data=singer model=zinb
```

The command auto-detects split mode from `data.<dataset>.yaml` (`split_by`) and
dispatches to split orchestration when needed. See `docs/cli_infer.md` for full
usage and expected `conf/` layout.

For SLURM clusters, use interactive submitit launch mode:

```bash
scribe-infer --slurm --config-path ./conf data=singer
```

`--slurm` prompts for cluster-specific resources and requires partition input
(no hardcoded partition default).

For reusable cluster settings across runs, add profiles under `conf/slurm` and
invoke:

```bash
scribe-infer --slurm-profile default --config-path ./conf data=singer
```

Per-run overrides remain available via repeated `--slurm-set key=value`
(and also implicitly enable SLURM mode).

## Visualization CLI

SCRIBE also provides a packaged visualization CLI:

```bash
scribe-visualize outputs/my_run --all
scribe-visualize outputs/my_run/custom_results.pkl --all
scribe-visualize outputs/ --recursive --umap --heatmap
scribe-visualize outputs/ --recursive "*_results.pkl" --all
```

For cluster execution of large recursive runs:

```bash
scribe-visualize --slurm-profile default outputs/ --recursive --all
```

See `docs/cli_visualize.md` for detailed usage.

## Architecture Overview

SCRIBE is organized into specialized modules that work together seamlessly:

```
scribe/
├── models/          # Probabilistic model definitions
├── svi/            # Stochastic Variational Inference
├── mcmc/           # Markov Chain Monte Carlo  
├── vae/            # Variational Autoencoder
├── core/           # Shared utilities and preprocessing
├── stats/          # Statistical analysis functions
├── viz/            # Visualization tools
└── inference.py    # Unified interface
```

### Core Components

#### 🎯 **Simplified API** (`api.py`)
The recommended entry point with flat kwargs and sensible defaults:

```python
import scribe

# SVI inference (fast, scalable) - default
svi_results = scribe.fit(data, model="nbdm", n_steps=75000)

# Zero-inflated model with linked parameterization
svi_results = scribe.fit(
    data,
    model="zinb",
    parameterization="linked",
    unconstrained=True,
    n_steps=75000,
)

# MCMC inference (exact, high-quality)
mcmc_results = scribe.fit(
    data,
    model="nbdm",
    inference_method="mcmc",
    n_samples=3000,
    n_chains=4,
)

# VAE inference (representation learning)
vae_results = scribe.fit(
    data,
    model="nbdm",
    inference_method="vae",
    n_steps=50000,
)

# Mixture model for cell type discovery
mixture_results = scribe.fit(
    data,
    model="zinb",
    n_components=5,
    n_steps=100000,
)
```

#### 🔧 **Power User API** (`inference/`)
For full control, use explicit configuration objects:

```python
from scribe.models.config import ModelConfigBuilder, InferenceConfig, SVIConfig

model_config = (
    ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .as_mixture(n_components=3)
    .build()
)
inference_config = InferenceConfig.from_svi(SVIConfig(n_steps=100000))

results = scribe.fit(
    data,
    model_config=model_config,
    inference_config=inference_config,
)
```

#### 🧬 **Models** (`models/`)
Four specialized probabilistic models for single-cell data:

- **NBDM**: Negative Binomial-Dirichlet Multinomial (baseline model)
- **ZINB**: Zero-Inflated Negative Binomial (handles excess zeros)
- **NBVCP**: NB with Variable Capture Probability (models technical dropout)
- **ZINBVCP**: Combines zero-inflation and variable capture

Each model supports:
- **Parameterizations**: `standard`, `linked`, `odds_ratio`
- **Variants**: Constrained and unconstrained optimization
- **Mixture Models**: Multi-component versions for cell type discovery
- **Guide Types**: Mean-field and low-rank variational families

#### ⚡ **SVI** (`svi/`)
Fast, scalable variational inference:

```python
from scribe.svi import SVIInferenceEngine

# Configure and run SVI
results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=100000,
    batch_size=512  # Mini-batch for large datasets
)

# Rich analysis capabilities
posterior = results.get_posterior_samples()
predictive = results.get_predictive_samples()
log_lik = results.log_likelihood()
```

#### 🎲 **MCMC** (`mcmc/`)
Exact Bayesian inference with full uncertainty quantification:

```python
from scribe.mcmc import MCMCInferenceEngine

# NUTS sampling for exact inference
mcmc_results = MCMCInferenceEngine.run_inference(
    model_config=config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_samples=4000,
    n_warmup=2000,
    n_chains=6
)

# Comprehensive diagnostics
mcmc_results.mcmc.print_summary()  # R-hat, ESS, etc.
```

#### 🧠 **VAE** (`vae/`)
Neural network-based variational inference with representation learning:

```python
from scribe.vae import VAEConfig, create_vae

# Configure VAE architecture
vae_config = VAEConfig(
    input_dim=n_genes,
    latent_dim=20,
    hidden_dims=[1024, 512, 256],
    activation="gelu",
    variable_capture=True,  # For VCP (variable capture) models
)

# Latent space analysis
embeddings = vae_results.get_latent_embeddings(data)
```

#### 🔧 **Core** (`core/`)
Shared preprocessing and analysis utilities:

```python
from scribe.core import InputProcessor, normalize_counts_from_posterior

# Data preprocessing
count_data, adata, n_cells, n_genes = InputProcessor.process_counts_data(
    counts=your_data,
    cells_axis=0,
    layer="counts"
)

# Posterior-based normalization
normalized = normalize_counts_from_posterior(
    posterior_samples=samples,
    n_components=3  # For mixture models
)
```

## Model Selection Guide

### When to Use Each Model

| Model       | Best For                             | Characteristics                       |
|-------------|--------------------------------------|---------------------------------------|
| **NBDM**    | Baseline analysis, well-behaved data | Simple, interpretable, fast           |
| **ZINB**    | Data with excess zeros               | Handles technical/biological dropouts |
| **NBVCP**   | Variable sequencing depth            | Models capture efficiency             |
| **ZINBVCP** | Complex technical variation          | Most comprehensive model              |

### When to Use Each Inference Method

| Method   | Speed  | Accuracy              | Use Case                        |
|----------|--------|-----------------------|---------------------------------|
| **SVI**  | Fast   | Good approximation    | Exploration, large datasets     |
| **MCMC** | Slow   | Exact                 | Final analysis, publication     |
| **VAE**  | Medium | Good + representation | Dimension reduction, clustering |

### Parameterization Guide

| Parameterization | Parameters            | Best For                               |
|------------------|-----------------------|----------------------------------------|
| **Standard**     | Direct p, r           | Most interpretable                     |
| **Linked**       | Links p to expression | When p-expression relationship matters |
| **Odds Ratio**   | Uses odds ratios      | When odds ratios are natural scale     |

## Advanced Usage Examples

### Mixture Model Analysis

```python
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config import InferenceConfig, SVIConfig

model_config = build_config_from_preset(
    model="nbdm",
    inference_method="svi",
    n_components=5,
)
mixture_results = scribe.run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=InferenceConfig.from_svi(SVIConfig(n_steps=100000)),
)

# Analyze cell type assignments
cell_types = mixture_results.cell_type_probabilities()
component_entropy = mixture_results.mixture_component_entropy()

# Access individual components
for i in range(5):
    component = mixture_results.get_component(i)
    print(f"Component {i} MAP estimates:", component.get_map())
```

When using annotation-guided mixtures via `annotation_key` with inferred
`n_components`, SCRIBE now auto-downgrades to non-mixture mode when
`annotation_min_cells` filtering leaves zero or one surviving annotation class.
This avoids invalid single-component mixture construction and also clears
component-only prior flags (for example `expression_prior`) that require
`n_components >= 2`. If `n_components` is set explicitly, strict mixture
behavior is preserved.

### Model Comparison Workflow

```python
from scribe.models.config import InferenceConfig, SVIConfig

# Compare multiple models (nbdm, zinb, nbvcp, zinbvcp)
models = ["nbdm", "zinb", "nbvcp", "zinbvcp"]
results = {}
inference_config = InferenceConfig.from_svi(SVIConfig(n_steps=75000))

for model in models:
    result = scribe.run_scribe(
        counts=adata,
        model=model,
        inference_method="svi",
        inference_config=inference_config,
    )
    results[model] = result

# Compare using log-likelihood
for model, result in results.items():
    ll = result.log_likelihood_map()
    print(f"{model}: {ll:.2f}")
```

### Multi-Method Analysis

```python
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config import InferenceConfig, SVIConfig, MCMCConfig

# Shared model: ZINB mixture with 3 components
model_config = build_config_from_preset(
    model="zinb",
    parameterization="linked",
    n_components=3,
)

# Fast exploration with SVI
svi_results = scribe.run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=InferenceConfig.from_svi(SVIConfig(n_steps=50000)),
)

# Exact inference with MCMC (update model_config.inference_method for mcmc)
mcmc_model = build_config_from_preset(
    model="zinb",
    parameterization="linked",
    inference_method="mcmc",
    n_components=3,
)
mcmc_results = scribe.run_scribe(
    counts=adata,
    model_config=mcmc_model,
    inference_config=InferenceConfig.from_mcmc(
        MCMCConfig(n_samples=3000, n_chains=4)
    ),
)

# Compare results
print("SVI log-likelihood:", svi_results.log_likelihood_map())
print("MCMC log-likelihood:", mcmc_results.log_likelihood())
```

### Biology-Informed Capture Prior

For VCP models, anchor capture probability to biological knowledge about total
cellular mRNA content. This resolves `p_capture` degeneracy by relating capture
efficiency to observed library size and expected total mRNA (`M_0`):

```python
# Biology-informed prior with organism defaults
results = scribe.fit(
    adata,
    model="nbvcp",
    parameterization="mean_odds",
    priors={"organism": "human"},  # Sets M_0 = 200,000 for human cells
)

# Manual capture_efficiency override (alias for eta_capture)
results = scribe.fit(
    adata,
    model="zinbvcp",
    priors={"capture_efficiency": (11.5, 0.3)},  # Custom log_M0 and sigma_M
)

# Hierarchical per-dataset capture scaling (learns total-mRNA scaling)
results = scribe.fit(
    adata,
    model="nbvcp",
    capture_scaling_prior="gaussian",
    priors={"organism": "mouse"},
)

# Hierarchical capture scaling with tight sigma_mu control
results = scribe.fit(
    adata,
    model="nbvcp",
    capture_scaling_prior="gaussian",
    priors={"organism": "human", "capture_scaling": (12.2, 0.3)},
)
```

Using the builder API:

```python
from scribe.models.config import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("nbvcp")
    .with_parameterization("mean_odds")
    .with_capture_priors(organism="human", capture_scaling_prior="gaussian")
    .build())
```

Supported organisms: `human`, `mouse`, `yeast`, `ecoli` (plus aliases like
`homo_sapiens`, `mus_musculus`, `saccharomyces_cerevisiae`,
`escherichia_coli`). See `paper/_capture_prior.qmd` for the full mathematical
derivation.

### Advanced VAE Configuration

```python
from scribe.vae import VAEConfig

# Configure sophisticated VAE architecture (see vae module for full API)
vae_config = VAEConfig(
    input_dim=n_genes,
    latent_dim=25,
    hidden_dims=[2048, 1024, 512, 256],
    activation="gelu",
    input_transformation="log1p",
    variable_capture=True,
    variable_capture_hidden_dims=[128, 64],
    standardize_mean=gene_means,
    standardize_std=gene_stds,
)

# Use with run_scribe: pass model_config and inference_config;
# VAE-specific options go in model_config.vae or via ModelConfigBuilder.
# See scribe.vae and scribe.models.config for full VAE setup.
```

## Integration with Single-Cell Ecosystem

SCRIBE integrates seamlessly with the single-cell Python ecosystem:

```python
import scanpy as sc
import scribe

# Standard scanpy preprocessing
adata = sc.read_h5ad("data.h5ad")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# SCRIBE analysis
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config import InferenceConfig, SVIConfig

model_config = build_config_from_preset(
    model="nbdm",
    inference_method="vae",
    n_components=8,
)
results = scribe.run_scribe(
    counts=adata,
    model_config=model_config,
    inference_config=InferenceConfig.from_vae(SVIConfig(n_steps=50000)),
)

# Add results back to AnnData
adata.obsm["X_scribe_latent"] = results.get_latent_embeddings(adata.X)
adata.obs["scribe_cluster"] = results.cell_type_probabilities().argmax(axis=1)

# Continue with scanpy analysis
sc.pp.neighbors(adata, use_rep="X_scribe_latent")
sc.tl.umap(adata)
sc.pl.umap(adata, color="scribe_cluster")
```

## Performance and Scalability

### Memory Management

```python
# For large datasets, use mini-batching
from scribe.models.config import InferenceConfig, SVIConfig

large_results = scribe.run_scribe(
    counts=large_adata,
    model="nbdm",
    inference_method="svi",
    inference_config=InferenceConfig.from_svi(
        SVIConfig(n_steps=150000, batch_size=1024)
    ),
)
```

### GPU Acceleration

```python
# SCRIBE automatically uses GPU when available
import jax
print("GPU available:", jax.devices("gpu"))

# All inference methods benefit from GPU acceleration
from scribe.models.config import InferenceConfig, SVIConfig

gpu_results = scribe.run_scribe(
    counts=adata,
    model="nbdm",
    inference_method="vae",
    inference_config=InferenceConfig.from_vae(SVIConfig(n_steps=75000)),
)
```

## Data Configuration

`load_and_preprocess_anndata()` accepts `.h5ad`, `.csv`, and 10x Matrix
Exchange (MEX) inputs. For MEX data, set `data.path` to either the directory
containing `matrix.mtx`/`barcodes.tsv`/`features.tsv` (`genes.tsv` also
supported) or directly to `matrix.mtx`.

### Observation Pre-Filtering (`filter_obs`)

Data YAML configs support an optional `filter_obs` field for declarative
observation-level row filtering.  This is applied **before** both the `split_by`
covariate discovery (in split orchestration mode) and per-job subsetting,
ensuring that unwanted categories are excluded at the earliest stage.

Keys are column names in `adata.obs`; values are lists of allowed values.
When multiple columns are specified, the per-column conditions are ANDed
(all must hold).

```yaml
# conf/data/batch_correction/a549.yaml
filter_obs:
  siRNA: ["SCRAMBLE"]

split_by:
  - "treatment"
  - "kit"
```

The same parameter is available on `load_and_preprocess_anndata()` for
programmatic use:

```python
from scribe.data_loader import load_and_preprocess_anndata

adata = load_and_preprocess_anndata(
    "data.h5ad",
    return_jax=False,
    filter_obs={"siRNA": ["SCRAMBLE"], "batch": ["A", "B"]},
)
```

## Module Documentation

Each module has comprehensive documentation with detailed examples:

- **[Models](models/README.md)**: Probabilistic model definitions and
  architectures
- **[SVI](svi/README.md)**: Stochastic variational inference implementation  
- **[MCMC](mcmc/README.md)**: NUTS sampling and exact Bayesian inference
- **[VAE](vae/README.md)**: Neural variational autoencoders and architectures
- **[Core](core/README.md)**: Shared preprocessing and analysis utilities

### Catalog Filtering Note

The experiment catalog in `catalog.py` supports comma-delimited filters for
list-like metadata fields. For example, both
`mixture_params=["phi,mu,gate"]` and `mixture_params="phi,mu,gate"` are
normalized to match metadata stored as `["phi", "mu", "gate"]`.

Dot-key filters are also supported for both metadata layouts:
flattened keys (for example, `inference.enable_x64`) and nested dictionaries
(for example, `inference.batch_size`).

For advanced selection logic, use the callable catalog filter API:
`catalog.filter(lambda run: "annotation_key=cell-class" in run.path)`.
This enables custom path/name filtering or arbitrary metadata predicates.

`ExperimentRun.load_data()` and `ExperimentCatalog.load_data()` replay the run's
configured data pipeline by default (`preprocessing=True`), including
`filter_obs`, `subset_column`/`subset_value`, and `data.preprocessing` steps
such as `filter_cells`. Pass `preprocessing=False` to load directly from
`data.path` without applying these transformations.

## Dependencies

### Core Dependencies
- **JAX**: Automatic differentiation and GPU acceleration
- **NumPyro**: Probabilistic programming framework
- **Flax**: Neural network library (for VAE models)
- **Pandas**: Data manipulation and metadata handling

### Optional Dependencies
- **AnnData**: Single-cell data format integration
- **Scanpy**: Single-cell analysis ecosystem integration
- **Matplotlib/Seaborn**: Visualization (via viz module)
- **scikit-learn**: Dimensionality reduction utilities

## Parameter Name Migration Reference

This section documents the renaming from mathematical/paper notation to
descriptive names across the three API layers.

### `scribe.fit()` Keyword Arguments

| Old name               | New name                       |
|------------------------|--------------------------------|
| `mu_prior`             | `expression_prior`             |
| `mu_dataset_prior`     | `expression_dataset_prior`     |
| `mu_eta_prior`         | `capture_scaling_prior`        |
| `mu_mean_anchor`       | `expression_anchor`            |
| `mu_mean_anchor_sigma` | `expression_anchor_sigma`      |
| `p_prior`              | `prob_prior`                   |
| `p_dataset_prior`      | `prob_dataset_prior`           |
| `p_dataset_mode`       | `prob_dataset_mode`            |
| `gate_prior`           | `zero_inflation_prior`         |
| `gate_dataset_prior`   | `zero_inflation_dataset_prior` |

### `priors` Dict Keys (both accepted, descriptive is preferred)

| Internal key    | Descriptive alias      |
|-----------------|------------------------|
| `p`             | `prob`                 |
| `r`             | `dispersion`           |
| `mu`            | `expression`           |
| `phi`           | `odds`                 |
| `gate`          | `zero_inflation`       |
| `p_capture`     | `capture_prob`         |
| `phi_capture`   | `capture_odds`         |
| `eta_capture`   | `capture_efficiency`   |
| `mu_eta`        | `capture_scaling`      |

### Results Keys (`get_map(descriptive_names=True)`)

| Internal key        | Descriptive name       |
|---------------------|------------------------|
| `r`                 | `dispersion`           |
| `p`                 | `prob`                 |
| `mu`                | `expression`           |
| `phi`               | `odds`                 |
| `gate`              | `zero_inflation`       |
| `p_capture`         | `capture_prob`         |
| `phi_capture`       | `capture_odds`         |
| `eta_capture`       | `capture_efficiency`   |
| `mixing_weights`    | `mixing_weights`       |
| `bnb_concentration` | `bnb_concentration`    |
| `z`                 | `latent_embedding`     |

> **Note:** Hierarchical hyperprior keys (`logit_p_loc`, `log_phi_scale`, etc.)
> and Hydra YAML `priors:` hyperprior overrides are unchanged. Users who
> interact with these already understand the unconstrained transform space.

## Contributing

SCRIBE is actively developed and welcomes contributions. See the main repository
for contribution guidelines and development setup instructions.

## Citation

If you use SCRIBE in your research, please cite our paper:

```bibtex
@article{scribe2024,
  title={SCRIBE: Single-Cell Bayesian Inference Ensemble for RNA-seq Analysis},
  author={Your Name and Others},
  journal={Journal Name},
  year={2024}
}
```

## License

SCRIBE is released under the MIT License. See the LICENSE file for details.

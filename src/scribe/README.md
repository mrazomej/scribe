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

# Run SCRIBE with default settings (SVI inference, NBDM model)
results = scribe.fit(adata, model="nbdm")

# With customization
results = scribe.fit(
    adata,
    model="zinb",
    n_components=3,
    n_steps=100000,
    batch_size=512,
)

# Analyze results
posterior_samples = results.get_posterior_samples()
log_likelihood = results.log_likelihood()

# Visualize
scribe.viz.plot_loss_history(results.loss_history)
```

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

### Observation Pre-Filtering (`filter_obs`)

Data YAML configs support an optional `filter_obs` field for declarative
observation-level row filtering.  This is applied **before** both the
`split_by` covariate discovery (in `infer_split.py`) and per-job subsetting,
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

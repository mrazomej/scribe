# SCRIBE Configuration Guide

This directory contains Hydra configuration files for SCRIBE's inference and
visualization pipelines. The configuration system allows you to easily manage
complex experimental setups, run parameter sweeps, and maintain reproducible
results.

## Quick Start

```bash
# Run inference with default settings (NBDM model, SVI inference)
python infer.py

# Specify model and inference method
python infer.py model=zinb inference=svi

# With model options
python infer.py model=zinb parameterization=linked n_components=3

# Override inference parameters
python infer.py model=nbdm inference.n_steps=100000 inference.batch_size=512

# Use different datasets
python infer.py model=zinb data=jurkat_cells

# Disable visualization
python infer.py model=nbdm viz=null

# NBVCP with amortized capture probability
python infer.py model=nbvcp amortization=capture
```

## Configuration Structure

```
conf/
├── config.yaml           # Main configuration file
├── data/                 # Data-specific configurations
│   ├── jurkat_cells.yaml
│   ├── singer.yaml
│   └── 5050mix.yaml
├── inference/            # Inference method configurations
│   ├── svi.yaml         # Stochastic Variational Inference
│   ├── mcmc.yaml        # Markov Chain Monte Carlo
│   └── vae.yaml         # Variational Autoencoder
├── amortization/        # Amortized inference presets
│   └── capture.yaml     # Amortized capture probability
├── model/               # Model presets
│   ├── nbdm.yaml        # Negative Binomial Dropout Model
│   ├── zinb.yaml        # Zero-Inflated NB
│   ├── nbvcp.yaml       # NB with Variable Capture Probability
│   └── zinbvcp.yaml     # ZINB with Variable Capture
├── viz/                 # Visualization configurations
│   └── default.yaml
└── README.md
```

## Main Configuration (`config.yaml`)

The main configuration file defines the complete experimental setup:

### Model Configuration

```yaml
# Model type: nbdm, zinb, nbvcp, zinbvcp
model: nbdm

# Parameterization scheme
# - "canonical" (or "standard"): Sample p, r directly
# - "linked" (or "mean_prob"): Sample p, mu, derive r
# - "odds_ratio" (or "mean_odds"): Sample phi, mu, derive p and r
parameterization: canonical

# Use unconstrained parameterization (Normal + transform)
unconstrained: false

# Mixture model configuration
n_components: null    # null = single component, int >= 2 for mixture
mixture_params: null  # which params are component-specific

# Guide configuration
guide_rank: null      # null = mean-field, int = low-rank
```

### Prior Configuration

```yaml
# Override default priors (each is [param1, param2])
priors:
  p: null           # Beta(alpha, beta) for success probability
  r: null           # LogNormal(loc, scale) for dispersion
  mu: null          # LogNormal(loc, scale) for mean
  phi: null         # BetaPrime(alpha, beta) for odds ratio
  gate: null        # Beta(alpha, beta) for zero-inflation gate
  p_capture: null   # Beta(alpha, beta) for capture probability
```

### Amortization Configuration

For VCP models (nbvcp, zinbvcp), you can enable amortized inference for capture
probability. This uses a neural network to predict variational parameters from
total UMI count, reducing parameters from O(n_cells) to O(1):

```yaml
# Amortization settings (only for VCP models)
amortization:
  capture:
    enabled: false       # Enable amortized capture probability
    hidden_dims: [64, 32]  # MLP hidden layer dimensions
    activation: leaky_relu     # Activation: relu, gelu, silu, tanh, etc.
    output_transform: softplus  # Constrained outputs: softplus (default) or exp
    output_clamp_min: 0.1       # Min clamp for alpha/beta (null to disable)
    output_clamp_max: 50.0      # Max clamp for alpha/beta (null to disable)
```

### Example Configurations

```yaml
# ZINB mixture model
model: zinb
parameterization: linked
n_components: 3

# NBVCP with low-rank guide
model: nbvcp
parameterization: canonical
guide_rank: 15

# NBVCP with amortized capture probability
model: nbvcp
amortization:
  capture:
    enabled: true
    hidden_dims: [128, 64]
    activation: gelu

# Custom priors
model: nbdm
priors:
  p: [2.0, 2.0]     # Informative Beta prior
  r: [0.0, 0.5]     # Tighter LogNormal prior
```

## Data Configurations (`data/`)

Each dataset has its own configuration file:

### Example: Singer Dataset (`data/singer.yaml`)

```yaml
# @package data
name: "singer"
path: "data/singer/singer_transcript_counts.csv"

# Optional preprocessing
# preprocessing:
#   filter_cells:
#     min_genes: 200
#   filter_genes:
#     min_cells: 3
```

### Creating a Custom Dataset Config

```yaml
# @package data
name: "my_dataset"
path: "path/to/my_data.h5ad"
preprocessing:
  filter_cells:
    min_genes: 200
  filter_genes:
    min_cells: 5
```

### Covariate-Split Inference (`split_by`)

A data config can include optional `split_by` and `n_jobs` fields to enable
**covariate-split parallel inference**.  When `split_by` is set, the
`infer_split.py` orchestrator will fit a separate model for each unique value
of that column in `adata.obs`, running them in parallel across GPUs.

```yaml
# @package data
name: "bleo_study01"
path: "data/lung/bleo_study01.h5ad"
split_by: "treatment"   # .obs column to split on
n_jobs: null             # parallel workers (null = auto-detect from GPUs)
preprocessing:
  filter_cells:
    min_genes: 200
  filter_genes:
    min_cells: 5
```

| Field      | Type          | Description                                                                                         |
|------------|---------------|-----------------------------------------------------------------------------------------------------|
| `split_by` | `str`         | Column name in `adata.obs`. Each unique value defines a data subset that gets its own model fit.    |
| `n_jobs`   | `int \| null` | Number of parallel joblib workers.  Defaults to the number of visible CUDA GPUs, falling back to 1. |

**Usage** --- run `infer_split.py` instead of `infer.py` with the exact same
Hydra overrides:

```bash
# Split by treatment column, auto-detect GPUs
python infer_split.py data=bleo_study01 variable_capture=true annotation_key=subclass-l1

# Explicitly set parallel workers
python infer_split.py data=bleo_study01 data.n_jobs=4 variable_capture=true

# All standard infer.py overrides work
python infer_split.py data=bleo_study01 inference.n_steps=100000 inference.batch_size=512
```

Under the hood, `infer_split.py`:

1. Loads the h5ad to discover the unique values of the `split_by` column.
2. Generates lightweight temporary data YAMLs in `conf/data/_tmp_split/`,
   each pointing to the **same** original h5ad but with `subset_column` /
   `subset_value` fields for in-memory filtering.
3. Launches `python infer.py -m data=_tmp_split/v1,_tmp_split/v2,...` with
   the Hydra joblib launcher for parallelism and round-robin GPU assignment.
4. Cleans up the temporary YAMLs after completion.

Output directories follow the standard layout with the covariate value
appended to the data name:

```
outputs/bleo_study01_bleomycin/nbvcp/svi/...
outputs/bleo_study01_control/nbvcp/svi/...
```

## Inference Configurations (`inference/`)

### SVI (`inference/svi.yaml`)

```yaml
# @package inference
method: svi
n_steps: 50_000
batch_size: null      # null = full batch
stable_update: true
```

### MCMC (`inference/mcmc.yaml`)

```yaml
# @package inference
method: mcmc
n_samples: 2_000
n_warmup: 1_000
n_chains: 1
```

### VAE (`inference/vae.yaml`)

```yaml
# @package inference
method: vae
n_steps: 50_000
batch_size: null
stable_update: true
```

## Model Presets (`model/`)

Model preset files configure the model type and default options:

```bash
# Use ZINB preset
python infer.py model=zinb

# Use NBVCP preset
python infer.py model=nbvcp

# Presets set: model type, parameterization, unconstrained, n_components
# You can override any of these:
python infer.py model=zinb parameterization=linked n_components=3
```

Each preset file (e.g., `model/zinb.yaml`) sets sensible defaults that can be overridden.

## Advanced Usage

### Parameter Sweeps

```bash
# Sweep over models
python infer.py -m model=nbdm,zinb,nbvcp,zinbvcp

# Sweep over mixture components
python infer.py -m model=zinb n_components=2,3,5

# Sweep over inference steps
python infer.py -m inference.n_steps=25000,50000,100000
```

### Mixture Model Analysis

```bash
# Cell type discovery with ZINB mixture
python infer.py model=zinb n_components=5 inference.n_steps=100000

# With linked parameterization for better optimization
python infer.py model=zinb n_components=3 parameterization=linked
```

### Annotation Priors

Use cell-type annotations as soft priors on mixture component assignments:

```bash
# Use annotation_key to bias components toward known cell types
python infer.py model=nbvcp n_components=5 annotation_key=cell_type annotation_confidence=3.0

# Filter rare annotations: labels with < 50 cells get zero bias (treated as unlabeled)
python infer.py model=nbvcp annotation_key=subclass-l1 annotation_min_cells=50

# Explicit component ordering
python infer.py model=nbdm n_components=3 annotation_key=cell_type \
  annotation_component_order='[T,B,Mono]'
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `annotation_key` | `str` or list | `null` | Column(s) in `adata.obs` with cell-type labels |
| `annotation_confidence` | `float` | `3.0` | Prior strength (kappa); 3 ≈ 20x boost |
| `annotation_component_order` | `list` | `null` | Explicit label-to-component index mapping |
| `annotation_min_cells` | `int` | `null` | Min cells per label; rare labels get zero bias |

### Custom Priors

```bash
# Informative priors for specific analysis
python infer.py model=nbdm "priors.p=[2.0,2.0]" "priors.r=[1.0,0.5]"
```

### High-Performance Settings

```bash
# Large dataset with mini-batching
python infer.py model=zinb inference.batch_size=1024 inference.n_steps=200000

# More MCMC samples for publication-quality inference
python infer.py model=nbdm inference=mcmc inference.n_samples=5000 inference.n_chains=4
```

### Covariate-Split Parallel Inference

Fit separate models for each level of a covariate (e.g. treatment) in
parallel across GPUs:

```bash
# Requires split_by in the data config
python infer_split.py data=bleo_study01 variable_capture=true

# Override parallelism
python infer_split.py data=bleo_study01 data.n_jobs=2 model=nbvcp guide_rank=32
```

See the [Covariate-Split Inference](#covariate-split-inference-split_by)
section under Data Configurations for full details.

### Amortized Inference for Large Datasets

For datasets with many cells (100K+), amortized inference reduces parameters:

```bash
# Enable amortized capture probability
python infer.py model=nbvcp amortization.capture.enabled=true

# Use the capture preset (pre-configured amortization)
python infer.py model=nbvcp +amortization=capture

# Custom amortizer architecture (optional: output_transform, output_clamp_*)
python infer.py model=nbvcp \
    amortization.capture.enabled=true \
    amortization.capture.hidden_dims=[128,64,32] \
    amortization.capture.activation=gelu
```

## Output Organization

Hydra automatically organizes outputs:

```
outputs/
├── {dataset}/
│   ├── {model}/
│   │   ├── {inference_method}/
│   │   │   ├── {parameters}/
│   │   │   │   ├── scribe_results.pkl
│   │   │   │   ├── figs/
│   │   │   │   └── .hydra/
```

## Visualization Configuration

```yaml
viz:
  loss: true       # Plot loss/ELBO history
  ecdf: true       # Plot empirical CDFs
  ppc: true        # Plot posterior predictive checks
  format: png      # Output format: png, pdf, svg, eps
```

### Disable Visualization

```bash
# Run inference without any plots
python infer.py model=nbdm viz=null

# Or disable specific plots
python infer.py model=nbdm viz.loss=false viz.ppc=true
```

## Programmatic API

The Hydra configuration maps directly to `scribe.fit()` arguments:

```python
import scribe

# Equivalent to: python infer.py model=zinb n_components=3 inference.n_steps=100000
results = scribe.fit(
    adata,
    model="zinb",
    n_components=3,
    n_steps=100000,
)

# Equivalent to: python infer.py model=nbvcp amortization.capture.enabled=true
# infer.py builds AmortizationConfig from YAML and passes capture_amortization=...
from scribe.models.config import AmortizationConfig

results = scribe.fit(
    adata,
    model="nbvcp",
    capture_amortization=AmortizationConfig(
        enabled=True,
        hidden_dims=[64, 32],
        activation="leaky_relu",
        output_transform="softplus",
        output_clamp_min=0.1,
        output_clamp_max=50.0,
    ),
)
# Backward compatible: you can still pass amortize_capture=True and the six capture_* params.
```

## Best Practices

1. **Start Simple**: Begin with `model=nbdm` and add complexity gradually
2. **Version Control**: Track configuration changes alongside code
3. **Document Experiments**: Use descriptive override names
4. **Monitor Resources**: Watch memory usage for large datasets
5. **Validate Results**: Always check convergence and posterior predictive checks

## Further Reading

- [Hydra Documentation](https://hydra.cc/) for advanced configuration patterns
- [SCRIBE Models](../src/scribe/models/README.md) for model-specific details
- [SCRIBE Package](../src/scribe/README.md) for full API documentation
- Main [README.md](../README.md) for package overview

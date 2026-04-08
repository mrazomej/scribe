# `scribe-infer` CLI Guide

`scribe-infer` is the unified command-line interface for running SCRIBE
inference. It wraps the full `scribe.fit()` pipeline behind
[Hydra](https://hydra.cc/)-managed YAML configs, so you can launch
reproducible runs --- locally or on a SLURM cluster --- without writing
Python scripts.

The CLI automatically detects whether the selected dataset should run as:

- **Standard single-run inference** (`split_by` not set in data config), or
- **Covariate-split orchestration** (`split_by` set), which launches one
  independent fit per unique value of the splitting variable.

---

## Installation

The CLI requires optional Hydra dependencies:

```bash
pip install 'scribe[hydra]'
```

Or, if using `uv`:

```bash
uv pip install 'scribe[hydra]'
```

---

## Quick start

```bash
# 1. Scaffold starter config files
scribe-infer --initialize ./conf

# 2. Copy the example data config and edit it
cp conf/data/example.yaml conf/data/my_dataset.yaml
# edit conf/data/my_dataset.yaml: set name and path

# 3. Run inference
scribe-infer --config-path ./conf data=my_dataset
```

---

## Initialize starter configs

Use `--initialize` to scaffold a documented `conf/` tree:

```bash
# Interactive path selection (or defaults to ./conf in non-interactive mode)
scribe-infer --initialize

# Explicit target
scribe-infer --initialize ./conf
scribe-infer --initialize /path/to/conf
```

The generated tree includes:

```text
conf/
├── config.yaml                          # Global config (model, priors, guide, output layout)
├── data/
│   └── example.yaml                     # Template dataset config
├── inference/
│   ├── svi.yaml                         # SVI defaults (optimizer, early stopping, etc.)
│   ├── mcmc.yaml                        # MCMC defaults (samples, warmup, chains)
│   └── vae.yaml                         # VAE defaults (inherits SVI + architecture)
├── viz/
│   └── default.yaml                     # Visualization defaults for scribe-visualize
├── amortization/
│   └── capture.yaml                     # Capture amortization preset
├── dirname_aliases/
│   └── default.yaml                     # Path aliasing for compact output directories
├── paths/
│   ├── paths.yaml                       # Default output directory
│   └── paths.local.yaml.example         # Machine-local override example
├── slurm/
│   └── default.yaml                     # Reusable SLURM profile
└── README.md
```

If managed files already exist, the CLI prompts before overwriting each file.

---

## Usage

```bash
scribe-infer --config-path ./conf data=<dataset_key> [hydra_overrides...]
```

Everything after the `scribe-infer` flags is forwarded as
[Hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/),
so any config field can be changed from the command line.

### Common examples

```bash
# Standard single-run inference with default model (NBVCP)
scribe-infer --config-path ./conf data=singer

# Override model to ZINB
scribe-infer --config-path ./conf data=singer model=zinb

# Enable variable capture (→ NBVCP) with amortized capture
scribe-infer --config-path ./conf data=singer variable_capture=true \
    amortization.capture.enabled=true

# Use MCMC instead of SVI
scribe-infer --config-path ./conf data=singer inference=mcmc

# Override multiple settings
scribe-infer --config-path ./conf data=singer \
    variable_capture=true \
    parameterization=mean_odds \
    unconstrained=true \
    prob_prior=neg \
    guide_rank=8 \
    inference.n_steps=100000
```

---

## Configuration reference

### Global config (`config.yaml`)

The global config controls model behavior, priors, guide settings, and output
layout. Key sections:

| Section | Key fields | Description |
|---------|-----------|-------------|
| **Model flags** | `zero_inflation`, `variable_capture` | Toggle model components. Maps to `nbdm` / `zinb` / `nbvcp` / `zinbvcp` |
| **Overdispersion** | `overdispersion`, `overdispersion_prior` | `"none"` or `"bnb"` with horseshoe/NEG prior |
| **Parameterization** | `parameterization`, `unconstrained` | `canonical`, `linked` (mean_prob), or `mean_odds` |
| **Gene-level priors** | `expression_prior`, `prob_prior`, `zero_inflation_prior` | `"none"`, `"gaussian"`, `"horseshoe"`, `"neg"` |
| **Multi-dataset** | `dataset_key`, `n_datasets`, `expression_dataset_prior`, ... | Joint multi-dataset fitting |
| **Guide** | `guide_rank`, `joint_params`, `dense_params` | Low-rank / joint low-rank guide. `joint_params`/`dense_params` accept shorthands (`"all"`, `"biological"`, `"mean"`, `"prob"`, `"gate"`) or explicit lists |
| **Flow guide** | `guide_flow`, `guide_flow_num_layers`, ... | Normalizing flow guide (mutually exclusive with `guide_rank`) |
| **Mixture** | `n_components`, `mixture_params` | Mixture model components. `mixture_params` defaults to `"all"` and accepts shorthands or explicit lists |
| **Priors** | `priors.organism`, `priors.eta_capture`, ... | Biology-informed and base distribution priors |
| **Anchoring** | `expression_anchor`, `expression_anchor_sigma` | Mean anchoring prior |
| **Amortization** | `amortization.capture.*` | Amortized capture inference |
| **Annotations** | `annotation_key`, `annotation_confidence` | Annotation-informed mixture priors |

For the meaning of each parameter, see the
[Parameter Reference](parameters.md) and the
[`scribe.fit()` Interface](fit.md).

### Dataset config (`data/*.yaml`)

Each dataset gets its own YAML file under `conf/data/`. Required fields:

| Field | Description |
|-------|-------------|
| `name` | Short identifier used in output paths and job names |
| `path` | Path to count matrix (`.h5ad` or `.csv`) |

Optional fields:

| Field | Description |
|-------|-------------|
| `layer` | AnnData layer name when counts are not in `adata.X` |
| `dataset_key` | Column in `adata.obs` identifying dataset membership (used for dataset-level hierarchical priors); overrides global `dataset_key` when set |
| `split_by` | Column in `adata.obs` for automatic split orchestration |
| `filter_obs` | Pre-filter observations before fitting (dict of column → allowed values) |
| `preprocessing` | Scanpy-like pipeline (`filter_cells`, `filter_genes`, `normalize_total`, `log1p`, `highly_variable_genes`) |

!!! note "SCRIBE fits on counts"
    Even when preprocessing includes HVG selection or normalization, SCRIBE
    always uses the raw count matrix for model fitting. Preprocessing is
    applied for gene selection only.

### Inference configs (`inference/*.yaml`)

Three presets are provided:

| Config | Method | Key defaults |
|--------|--------|-------------|
| `svi.yaml` | SVI | `n_steps=50000`, `batch_size=null`, `early_stopping.enabled=false` |
| `mcmc.yaml` | MCMC | `n_samples=2000`, `n_warmup=1000`, `n_chains=1`, `enable_x64=true` |
| `vae.yaml` | VAE | Inherits SVI settings + `vae_latent_dim=10`, `vae_flow_type=coupling_spline` |

Select at runtime:

```bash
scribe-infer --config-path ./conf data=singer inference=mcmc
```

---

## Dispatch behavior

`scribe-infer` inspects each selected `data=<key>` config under
`<config-path>/data/`:

- If at least one data config defines `split_by`, **split mode** is used.
  The CLI launches one independent inference run per unique value (or value
  combination when `split_by` is a list).
- Otherwise, **direct inference** mode is used.

All remaining CLI tokens are forwarded as Hydra overrides unchanged.

---

## SLURM integration

### Interactive launch

Use `--slurm` to route execution through Hydra's `submitit_slurm` launcher
with interactive prompts for cluster resources:

```bash
scribe-infer --slurm --config-path ./conf data=singer
```

The CLI prompts for partition (required), account (optional), CPUs, memory,
and timeout. The same command auto-dispatches to direct vs. split mode.

### Reusable SLURM profiles

Keep reusable cluster settings in `conf/slurm/*.yaml`:

```bash
scribe-infer --slurm-profile default --config-path ./conf data=singer
```

Resolution rules:

- Named profile `default` resolves to `./conf/slurm/default.yaml`
- You may also pass an explicit path to `--slurm-profile`
- `--slurm-set key=value` entries override profile values
- Missing core fields fall back to interactive prompts (or defaults where safe)
- `--slurm-profile` and `--slurm-set` automatically enable SLURM mode

### Per-run overrides

```bash
scribe-infer --slurm-profile default \
    --slurm-set partition=gpu \
    --slurm-set timeout=0-08:00 \
    --slurm-set mem_gb=128 \
    --config-path ./conf data=singer
```

### Default SLURM profile fields

| Field | Default | Description |
|-------|---------|-------------|
| `partition` | `null` (required) | Cluster partition name |
| `account` | `null` | Optional account/project string |
| `cpus_per_task` | `4` | CPU cores per task |
| `mem_gb` | `64` | Memory in GB |
| `timeout_min` | `240` | Wall-time limit in minutes |
| `array_parallelism` | `1` | Max concurrent array jobs (split mode) |
| `job_name` | `scribe-infer` | SLURM job name |
| `submitit_folder` | `slurm_logs/submitit/%j` | Log directory |
| `gres` | `null` | Generic resources (e.g., `gpu:1`) |
| `launcher_overrides` | `{}` | Escape hatch for cluster-specific submitit keys |

---

## Config root structure

The CLI expects this minimum structure:

```text
conf/
├── config.yaml
├── data/
│   └── <dataset_key>.yaml
└── inference/
    ├── svi.yaml
    ├── mcmc.yaml
    └── vae.yaml
```

Override config root and top-level config name:

```bash
scribe-infer --config-path /path/to/conf --config-name config data=my_dataset
```

---

## CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--config-path` | `./conf` | Hydra config root directory |
| `--config-name` | `config` | Top-level Hydra config filename (without `.yaml`) |
| `--initialize [PATH]` | --- | Scaffold starter configs. Cannot be combined with `--slurm` |
| `--slurm` | `false` | Launch via submitit with interactive resource prompts |
| `--slurm-profile PROFILE` | `null` | Load a reusable SLURM profile (auto-enables SLURM mode) |
| `--slurm-set KEY=VALUE` | --- | Per-run SLURM override (repeatable, auto-enables SLURM mode) |

---

## Workflow recipes

### Fit NBVCP with biology-informed capture on a cluster

```bash
scribe-infer --slurm-profile default \
    --slurm-set partition=gpu \
    --slurm-set gres=gpu:1 \
    --config-path ./conf \
    data=my_dataset \
    variable_capture=true \
    amortization.capture.enabled=true \
    priors.organism=human \
    inference.n_steps=100000
```

### Covariate-split run (one fit per condition)

Create `conf/data/experiment.yaml`:

```yaml
# @package data
name: "experiment"
path: "data/experiment.h5ad"
split_by: "condition"
```

Then:

```bash
scribe-infer --config-path ./conf data=experiment variable_capture=true
```

The CLI detects `split_by` and launches one independent fit per unique value
of the `condition` column.

### MCMC warm-started from SVI

```bash
# Step 1: SVI
scribe-infer --config-path ./conf data=singer inference=svi

# Step 2: MCMC initialized from SVI results
scribe-infer --config-path ./conf data=singer inference=mcmc \
    svi_init=/path/to/svi/results.pkl
```

---

For the full Python API, see the [`scribe.fit()` Interface](fit.md).
For model and parameter details, see
[Model Selection](model-selection.md) and the
[Parameter Reference](parameters.md).

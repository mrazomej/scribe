# `scribe-visualize` CLI Guide

`scribe-visualize` is the post-inference visualization CLI for SCRIBE. It
reads completed inference outputs (a result `*.pkl` file plus
`.hydra/config.yaml` in the same run directory) and generates a suite of
diagnostic plots --- from quick training summaries to detailed posterior
predictive checks.

Like `scribe-infer`, it supports **recursive directory traversal**, **wildcard
patterns**, and **SLURM submission** for batch visualization of many runs at
once.

---

## Installation

The CLI shares the same Hydra dependency group as `scribe-infer`:

```bash
pip install 'scribe[hydra]'
```

---

## Quick start

```bash
# Generate default plots (loss curve) for a single run
scribe-visualize outputs/my_run

# Use an explicit result file path (custom filename supported)
scribe-visualize outputs/my_run/custom_results.pkl

# Generate all available plots
scribe-visualize outputs/my_run --all

# Recursively visualize every run under a directory
scribe-visualize outputs/ --recursive --all

# Recursively match custom result filenames
scribe-visualize outputs/ --recursive "*_results.pkl" --all
```

---

## Available plots

`scribe-visualize` provides 13 plot types. Two are enabled by default (loss,
ECDF); the rest are opt-in via flags or `--all`.

### Default plots (on unless `--no-*` is passed)

| Plot             | Flag                   | Description                                                      |
| ---------------- | ---------------------- | ---------------------------------------------------------------- |
| **Loss history** | `--no-loss` to disable | ELBO / loss curve over SVI steps; MCMC diagnostics for MCMC runs |
| **ECDF**         | `--no-ecdf` to disable | Empirical CDF of observed counts for a sample of genes           |

### Opt-in plots (off unless explicitly enabled)

| Plot                                 | Flag                    | Description                                                                           | Requirements                        |
| ------------------------------------ | ----------------------- | ------------------------------------------------------------------------------------- | ----------------------------------- |
| **Posterior Predictive Check (PPC)** | `--ppc`                 | Grid of per-gene histograms comparing observed counts to posterior predictive samples | ---                                 |
| **Biological PPC**                   | `--bio-ppc`             | NB(r, p) credible bands overlaid with denoised data histograms                        | ---                                 |
| **UMAP**                             | `--umap`                | Joint UMAP embedding of observed and synthetic (PPC) data                             | ---                                 |
| **Correlation heatmap**              | `--heatmap`             | Posterior gene-gene correlation matrix                                                | ---                                 |
| **Mixture PPC**                      | `--mixture-ppc`         | Per-component posterior predictive check                                              | Mixture model (`n_components >= 2`) |
| **Mixture composition**              | `--mixture-composition` | MAP component assignment barplot                                                      | Mixture model                       |
| **Annotation PPC**                   | `--annotation-ppc`      | Per-annotation-label posterior predictive                                             | Mixture model + `annotation_key`    |
| **Capture anchor**                   | `--capture-anchor`      | Eta capture-anchor diagnostic (prior vs. posterior)                                   | Biology-informed capture prior      |
| **p_capture scaling**                | `--p-capture-scaling`   | Capture probability vs. library size                                                  | VCP model                           |
| **Mean calibration**                 | `--mean-calibration`    | Log-log scatter of observed vs. predicted per-gene means                              | ---                                 |
| **Mean pairwise**                    | `--mean-pairwise`       | Pairwise dataset-level mean comparison                                                | Multi-dataset model                 |

Use `--all` to enable every plot at once.

---

## Plot gallery

!!! info "Example plots coming soon"
    Each plot type will have an example figure here. Placeholders are included
    below for reference.

### Loss history

<!-- TODO: Add example loss plot -->

*Placeholder --- example loss curve will be added here.*

### ECDF

<!-- TODO: Add example ECDF plot -->

*Placeholder --- example ECDF plot will be added here.*

### Posterior Predictive Check (PPC)

<!-- TODO: Add example PPC plot -->

*Placeholder --- example PPC grid will be added here.*

### Biological PPC

<!-- TODO: Add example bio-PPC plot -->

*Placeholder --- example bio-PPC plot will be added here.*

### UMAP

<!-- TODO: Add example UMAP plot -->

*Placeholder --- example UMAP overlay will be added here.*

### Correlation heatmap

<!-- TODO: Add example heatmap -->

*Placeholder --- example correlation heatmap will be added here.*

### Mixture PPC

<!-- TODO: Add example mixture PPC plot -->

*Placeholder --- example mixture PPC will be added here.*

### Mixture composition

<!-- TODO: Add example mixture composition plot -->

*Placeholder --- example component assignment barplot will be added here.*

### Annotation PPC

<!-- TODO: Add example annotation PPC plot -->

*Placeholder --- example per-annotation PPC will be added here.*

### Capture anchor

<!-- TODO: Add example capture anchor plot -->

*Placeholder --- example capture-anchor diagnostic will be added here.*

### p_capture scaling

<!-- TODO: Add example p_capture scaling plot -->

*Placeholder --- example capture-probability-vs-library-size will be added here.*

### Mean calibration

<!-- TODO: Add example mean calibration plot -->

*Placeholder --- example observed-vs-predicted scatter will be added here.*

### Mu pairwise

<!-- TODO: Add example mu pairwise plot -->

*Placeholder --- example pairwise dataset mu comparison will be added here.*

---

## Customization options

Many plots accept fine-tuning parameters via CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `png` | Output format: `png`, `pdf`, `svg`, `eps` |
| `--ecdf-genes` | `25` | Number of genes shown in the ECDF panel |
| `--ppc-rows` | `5` | Rows in the PPC grid |
| `--ppc-cols` | `5` | Columns in the PPC grid |
| `--ppc-samples` | `512` | Number of posterior predictive samples for PPC |
| `--umap-ppc-samples` | `50` | PPC samples for the UMAP overlay |
| `--overwrite` | off | Re-generate plots even if output files already exist |

Additional fine-grained options (heatmap gene count, UMAP hyperparameters,
capture-anchor scatter settings, etc.) can be configured in
`conf/viz/default.yaml`, which is scaffolded by `scribe-infer --initialize`.

---

## Directory processing

### Single run

```bash
scribe-visualize outputs/my_run --ppc --umap
```

### Recursive search

Finds every matching result file recursively. With no pattern value,
`--recursive` defaults to `scribe_results.pkl`:

```bash
scribe-visualize outputs/ --recursive --all
```

### Recursive search with custom filename pattern

```bash
scribe-visualize outputs/ --recursive "*_results.pkl" --all
```

### Wildcard patterns

Shell-style globs for selective processing (directories or explicit files):

```bash
# All ZINB runs
scribe-visualize "outputs/*/zinb*/*" --heatmap

# Unquoted expansion (shell expands first)
scribe-visualize outputs/bleo_study0*/zinbvcp/* --recursive --umap

# Explicit file glob
scribe-visualize "outputs/**/*_results.pkl" --ppc
```

---

## SLURM integration

Submit visualization as a batch job with the same profile system as
`scribe-infer`:

```bash
# Interactive prompts for cluster resources
scribe-visualize --slurm outputs/ --recursive --all

# Reusable profile
scribe-visualize --slurm-profile default outputs/ --recursive --all

# Per-run overrides
scribe-visualize --slurm-profile default \
    --slurm-set partition=gpu \
    --slurm-set mem_gb=32 \
    outputs/ --recursive --all
```

The SLURM flags (`--slurm`, `--slurm-profile`, `--slurm-set`) follow the
same conventions as [`scribe-infer`](cli_infer.md#slurm-integration).

---

## CLI flags reference

| Flag                    | Default    | Description                                                                                                 |
| ----------------------- | ---------- | ----------------------------------------------------------------------------------------------------------- |
| `run_target`            | (required) | One or more run targets (directories, result `.pkl` files, or glob patterns)                                |
| `--all`                 | off        | Enable all plot types                                                                                       |
| `--no-loss`             | off        | Disable the loss curve                                                                                      |
| `--no-ecdf`             | off        | Disable the ECDF plot                                                                                       |
| `--ppc`                 | off        | Enable PPC grid                                                                                             |
| `--bio-ppc`             | off        | Enable biological PPC                                                                                       |
| `--umap`                | off        | Enable UMAP overlay                                                                                         |
| `--heatmap`             | off        | Enable correlation heatmap                                                                                  |
| `--mixture-ppc`         | off        | Enable mixture PPC                                                                                          |
| `--mixture-composition` | off        | Enable mixture composition barplot                                                                          |
| `--annotation-ppc`      | off        | Enable per-annotation PPC                                                                                   |
| `--capture-anchor`      | off        | Enable capture-anchor diagnostic                                                                            |
| `--p-capture-scaling`   | off        | Enable capture probability vs. library size                                                                 |
| `--mean-calibration`    | off        | Enable mean calibration scatter                                                                             |
| `--mean-pairwise`       | off        | Enable dataset-level mean comparison                                                                        |
| `--recursive [PATTERN]` | off        | Recursively search directories for result files; defaults to `scribe_results.pkl` when used without PATTERN |
| `--overwrite`           | off        | Regenerate existing plots                                                                                   |
| `--format`              | `png`      | Output format (`png`, `pdf`, `svg`, `eps`)                                                                  |
| `--slurm`               | off        | Launch as SLURM batch job                                                                                   |
| `--slurm-profile`       | ---        | Reusable SLURM profile name or path                                                                         |
| `--slurm-set`           | ---        | Per-run SLURM overrides (repeatable)                                                                        |

---

For inference CLI usage, see [`scribe-infer`](cli_infer.md). For the Python
visualization API, see the [API Reference](../reference/).

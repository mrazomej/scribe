# `scribe-infer` CLI Guide

`scribe-infer` is the unified CLI for SCRIBE inference.

It automatically detects whether your selected dataset config should run as:

- standard single-run inference (`split_by` not set), or
- covariate-split orchestration (`split_by` set in the data config).

## Installation

Install Hydra-enabled CLI dependencies with:

```bash
pip install 'scribe[hydra]'
```

## Usage

```bash
scribe-infer --config-path ./conf data=<dataset_key> [hydra_overrides...]
```

## Initialize Starter Configs

Use `--initialize` to scaffold a starter `conf/` tree with documented YAMLs:

```bash
# Interactive path selection (or defaults to ./conf in non-interactive mode)
scribe-infer --initialize

# Explicit target
scribe-infer --initialize ./conf
scribe-infer --initialize /path/to/conf
```

The generated tree includes:

- `config.yaml`
- `data/example.yaml`
- `inference/{svi,mcmc,vae}.yaml`
- `amortization/capture.yaml`
- `dirname_aliases/default.yaml`
- `paths/{paths.yaml,paths.local.yaml.example}`
- `README.md`

If managed files already exist, `scribe-infer` prompts before overwriting each
file.

## Interactive SLURM Launch

Use `--slurm` to route execution through Hydra's `submitit_slurm` launcher with
interactive prompts for cluster resources:

```bash
scribe-infer --slurm --config-path ./conf data=singer
```

Notes:

- Partition is required and has no built-in default.
- Account is optional.
- The same command still auto-dispatches direct vs split mode.

### Reusable SLURM Profiles (`conf/slurm`)

You can keep reusable cluster settings in profile YAML files:

```bash
scribe-infer --slurm-profile default --config-path ./conf data=singer
```

Resolution rules:

- Named profile `default` resolves to `./conf/slurm/default.yaml`.
- You may also pass an explicit path to `--slurm-profile`.
- Repeated `--slurm-set key=value` entries override profile values.
- Missing core fields fall back to interactive prompts (or defaults where safe).
- `--slurm-profile` and `--slurm-set` automatically enable SLURM launch mode.

Example per-run override:

```bash
scribe-infer --slurm-profile default --slurm-set partition=gpu --slurm-set timeout=0-08:00 --config-path ./conf data=singer
```

### Common Examples

```bash
# Standard run (no split_by in data config)
scribe-infer --config-path ./conf data=singer model=zinb

# Auto-dispatch to split orchestration (split_by present)
scribe-infer --config-path ./conf data=bleo_study01 variable_capture=true

# Split run on Slurm (auto split dispatch + submitit via --slurm mode)
scribe-infer --slurm --config-path ./conf data=bleo_study01
```

## Config Root Expectations

By default, `scribe-infer` uses `./conf`.

The command expects this minimum structure:

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

You can override config root and top-level config name:

```bash
scribe-infer --config-path /path/to/conf --config-name config data=my_dataset
```

## Dispatch Behavior

`scribe-infer` inspects each selected `data=<dataset_key>` config under
`<config-path>/data/`.

- If at least one selected data config defines `split_by`, split mode is used.
- Otherwise, direct inference mode is used.

All remaining CLI tokens are forwarded as Hydra overrides unchanged.


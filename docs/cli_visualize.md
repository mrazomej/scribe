# `scribe-visualize` CLI Guide

`scribe-visualize` is the packaged CLI for post-inference visualization.

It processes one or many run directories that contain:

- `scribe_results.pkl`
- `.hydra/config.yaml`

## Installation

```bash
uv sync --extra hydra
```

## Basic Usage

```bash
scribe-visualize outputs/my_run --all
```

## Recursive Processing

```bash
scribe-visualize outputs/ --recursive --ppc --umap
```

## Wildcard Processing

```bash
scribe-visualize "outputs/*/zinb*/*" --heatmap
```

## SLURM Submission

You can submit recursive visualization workloads as an SLURM batch job:

```bash
scribe-visualize --slurm outputs/ --recursive --all
```

Profile/set options follow the same model as `scribe-infer`:

```bash
scribe-visualize --slurm-profile default outputs/ --recursive --all
scribe-visualize --slurm-set partition=gpu outputs/ --recursive --all
```

Notes:

- `--slurm-profile` and `--slurm-set` automatically enable SLURM mode.
- `partition` is required (prompted if missing from profile/set).

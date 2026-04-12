# SLURM Profiles for `scribe-infer` and `scribe-visualize`

This directory stores reusable SLURM profile files used by:

```bash
scribe-infer --slurm --slurm-profile <profile_name> --config-path ./conf data=<dataset>
scribe-visualize --slurm --slurm-profile <profile_name> outputs/ --recursive
```

Profile and `--slurm-set` values are merged **before** interactive prompts; any
field already set (including `job_name` and `gres`) skips the corresponding
prompt.

## Resolution rules

- `--slurm-profile default` resolves to `conf/slurm/default.yaml`.
- You can also pass an explicit file path:
  `--slurm-profile /path/to/profile.yaml`.

## Merge precedence

Both CLIs resolve values in this order:

1. Values from the profile YAML.
2. Per-run overrides from repeated `--slurm-set key=value`.
3. Interactive prompts for any missing core values (and, for these CLIs, job
   name defaults: `scribe-infer` vs `scribe-visualize`).

Core values include:

- `partition` (required, no default),
- `array_parallelism`,
- `cpus_per_task`,
- `mem_gb`,
- `timeout_min`.

`scribe-visualize --slurm` additionally prompts for a GPU count when `gres` is
not already set; that maps to SLURM `gpu:N`. Clusters that need a different
`gres` syntax should set `gres` in the profile or via `--slurm-set gres=...`.

## Advanced launcher passthrough

For cluster-specific submitit options, use:

```bash
--slurm-set launcher.<hydra_launcher_key>=<value>
```

Examples:

```bash
--slurm-set launcher.max_num_timeout=3
--slurm-set launcher.comment=myqueuehint
```

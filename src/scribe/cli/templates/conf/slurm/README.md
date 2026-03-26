# SLURM Profiles for `scribe-infer`

This directory stores reusable SLURM profile files used by:

```bash
scribe-infer --slurm --slurm-profile <profile_name> --config-path ./conf data=<dataset>
```

## Resolution rules

- `--slurm-profile default` resolves to `conf/slurm/default.yaml`.
- You can also pass an explicit file path:
  `--slurm-profile /path/to/profile.yaml`.

## Merge precedence

`scribe-infer` resolves values in this order:

1. Values from the profile YAML.
2. Per-run overrides from repeated `--slurm-set key=value`.
3. Interactive prompts for any missing core values.

Core values include:

- `partition` (required, no default),
- `array_parallelism`,
- `cpus_per_task`,
- `mem_gb`,
- `timeout_min`.

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

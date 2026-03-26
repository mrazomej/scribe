"""Help text constants for ``scribe-infer``."""

DETAILED_DESCRIPTION = """
Run SCRIBE inference from Hydra configs with automatic split dispatch.

The command inspects the selected ``data=<name>`` config in ``<config-path>/data``.
If that config defines ``split_by``, ``scribe-infer`` launches split orchestration.
Otherwise, it runs a standard single-run inference.

Use ``--initialize [path]`` to scaffold starter YAML configs for a new project.
Use ``--slurm`` to launch through submitit with interactive SLURM prompts.
Use ``--slurm-profile`` and ``--slurm-set`` for reusable/advanced SLURM settings.
"""

EPILOG = """
Install requirement:
  pip install 'scribe[hydra]'

Default config root:
  --config-path defaults to ./conf

Expected configuration layout:
  conf/
    config.yaml  (i.e., conf/config.yaml)
    data/
      <dataset>.yaml
    inference/
      svi.yaml
      mcmc.yaml
      vae.yaml

Common examples:
  scribe-infer --config-path ./conf data=singer model=zinb
  scribe-infer --config-path ./conf data=bleo_study01 variable_capture=true
  scribe-infer --config-path ./conf data=bleo_study01 split.launcher=submitit_slurm

Initialize starter configs:
  scribe-infer --initialize
  scribe-infer --initialize ./conf
  scribe-infer --initialize /path/to/my_project_conf

Interactive SLURM launch:
  scribe-infer --slurm --config-path ./conf data=singer
  scribe-infer --slurm --slurm-profile default --config-path ./conf data=singer
  scribe-infer --slurm --slurm-set partition=gpu --slurm-set timeout=0-08:00 --config-path ./conf data=singer
  (partition is required and has no built-in default)

For a complete guide, see docs/cli_infer.md.
"""


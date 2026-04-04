"""SLURM submission helpers for ``scribe-visualize``."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

from .slurm_common import SlurmPromptConfig


def _minutes_to_slurm_time(timeout_min: int) -> str:
    """Convert integer timeout minutes to SLURM ``D-HH:MM`` format."""
    days, rem = divmod(timeout_min, 24 * 60)
    hours, minutes = divmod(rem, 60)
    return f"{days}-{hours:02d}:{minutes:02d}"


def _build_batch_script(
    slurm_cfg: SlurmPromptConfig, *, project_dir: Path, forwarded_args: list[str]
) -> str:
    """Build a SLURM batch script for recursive visualization jobs."""
    quoted_args = " ".join(shlex.quote(token) for token in forwarded_args)
    timeout = _minutes_to_slurm_time(slurm_cfg.timeout_min)
    job_name = slurm_cfg.job_name or "scribe-visualize"
    log_dir = "slurm_logs"
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm_cfg.partition}",
        f"#SBATCH --cpus-per-task={slurm_cfg.cpus_per_task}",
        f"#SBATCH --mem={slurm_cfg.mem_gb}G",
        f"#SBATCH --time={timeout}",
        f"#SBATCH --output={log_dir}/{job_name}_%j.out",
        f"#SBATCH --error={log_dir}/{job_name}_%j.err",
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(project_dir))}",
        "source .venv/bin/activate",
        f"python -m scribe.cli.visualize {quoted_args}",
        "",
    ]
    if slurm_cfg.account:
        script_lines.insert(3, f"#SBATCH --account={slurm_cfg.account}")
    if slurm_cfg.qos:
        script_lines.insert(4, f"#SBATCH --qos={slurm_cfg.qos}")
    if slurm_cfg.constraint:
        script_lines.insert(5, f"#SBATCH --constraint={slurm_cfg.constraint}")
    if slurm_cfg.exclude:
        script_lines.insert(6, f"#SBATCH --exclude={slurm_cfg.exclude}")
    if slurm_cfg.nodelist:
        script_lines.insert(7, f"#SBATCH --nodelist={slurm_cfg.nodelist}")
    if slurm_cfg.reservation:
        script_lines.insert(8, f"#SBATCH --reservation={slurm_cfg.reservation}")
    if slurm_cfg.gres:
        script_lines.insert(9, f"#SBATCH --gres={slurm_cfg.gres}")
    if slurm_cfg.mail_user:
        script_lines.insert(10, f"#SBATCH --mail-user={slurm_cfg.mail_user}")
    if slurm_cfg.mail_type:
        script_lines.insert(11, f"#SBATCH --mail-type={slurm_cfg.mail_type}")
    return "\n".join(script_lines)


def submit_visualize_slurm_job(
    *,
    slurm_cfg: SlurmPromptConfig,
    project_dir: Path,
    forwarded_args: list[str],
) -> int:
    """Submit ``scribe-visualize`` execution as a single SLURM batch job."""
    (project_dir / "slurm_logs").mkdir(parents=True, exist_ok=True)
    script_text = _build_batch_script(
        slurm_cfg, project_dir=project_dir, forwarded_args=forwarded_args
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".sh",
        prefix="scribe_visualize_",
        delete=False,
        dir=project_dir / "slurm_logs",
    ) as handle:
        handle.write(script_text)
        script_path = Path(handle.name)
    script_path.chmod(0o755)
    result = subprocess.run(["sbatch", str(script_path)], cwd=project_dir)
    return int(result.returncode)


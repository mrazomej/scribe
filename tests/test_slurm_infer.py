"""Tests for SLURM inference launcher helpers."""

from pathlib import Path

from slurm_infer import (
    SlurmConfig,
    apply_slurm_default_overrides,
    build_batch_script,
)


def test_apply_slurm_default_overrides_injects_progress_logging_default():
    """Inject default logging override when user does not set the key."""
    overrides = [
        "data=bleo_splits/bleo_study03_bleomycin",
        "model=zinbvcp",
    ]
    effective = apply_slurm_default_overrides(overrides)
    assert "inference.log_progress_lines=true" in effective
    assert effective[-2:] == overrides


def test_apply_slurm_default_overrides_respects_explicit_user_value():
    """Do not inject the default when user sets the key explicitly."""
    overrides = [
        "data=bleo_splits/bleo_study03_bleomycin",
        "inference.log_progress_lines=false",
    ]
    effective = apply_slurm_default_overrides(overrides)
    assert effective == overrides


def test_apply_slurm_default_overrides_normalizes_hydra_prefixes():
    """Treat +/~ Hydra key prefixes as explicit user overrides."""
    overrides = [
        "+inference.log_progress_lines=false",
        "data=bleo_splits/bleo_study06_control",
    ]
    effective = apply_slurm_default_overrides(overrides)
    assert effective == overrides


def test_build_batch_script_multirun_sets_loky_cpu_count_to_slurm_allocation():
    """Use full SLURM CPU allocation for loky in multirun mode.

    The launcher controls process-level parallelism via ``hydra.launcher.n_jobs``.
    ``LOKY_MAX_CPU_COUNT`` should therefore reflect the full task allocation,
    not the number of parallel jobs.
    """
    slurm = SlurmConfig(gpus=4, cpus_per_task=32)
    batch = build_batch_script(
        slurm=slurm,
        hydra_overrides=["data=bleo_study01_bleomycin"],
        project_dir=Path("/tmp/project"),
        is_multirun=True,
    )
    assert "export LOKY_MAX_CPU_COUNT=32" in batch
    assert "export LOKY_MAX_CPU_COUNT=4" not in batch


def test_build_batch_script_multirun_partitions_threads_per_worker():
    """Split BLAS/OMP threads across parallel jobs in multirun mode."""
    slurm = SlurmConfig(gpus=4, cpus_per_task=32)
    batch = build_batch_script(
        slurm=slurm,
        hydra_overrides=["data=bleo_study01_bleomycin"],
        project_dir=Path("/tmp/project"),
        is_multirun=True,
    )
    assert "export OMP_NUM_THREADS=8" in batch
    assert "export MKL_NUM_THREADS=8" in batch
    assert "export OPENBLAS_NUM_THREADS=8" in batch

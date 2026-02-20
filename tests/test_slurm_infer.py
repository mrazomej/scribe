"""Tests for SLURM inference launcher helpers."""

from pathlib import Path

from slurm_infer import (
    SlurmConfig,
    apply_slurm_default_overrides,
    build_batch_script,
    resolve_inference_script,
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
        inference_script="infer.py",
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
        inference_script="infer.py",
    )
    assert "export OMP_NUM_THREADS=8" in batch
    assert "export MKL_NUM_THREADS=8" in batch
    assert "export OPENBLAS_NUM_THREADS=8" in batch


def test_resolve_inference_script_uses_split_entrypoint_when_split_by_present(
    tmp_path: Path,
):
    """Select infer_split.py when data config contains split_by."""
    conf_data_dir = tmp_path / "conf" / "data" / "lung_bleo"
    conf_data_dir.mkdir(parents=True, exist_ok=True)
    config_path = conf_data_dir / "lung_bleo_splits.yaml"
    config_path.write_text(
        "\n".join(
            [
                "# @package data",
                "name: lung_bleo",
                "path: /tmp/mock.h5ad",
                "split_by: exp_condition",
            ]
        )
    )

    script = resolve_inference_script(
        overrides=["data=lung_bleo/lung_bleo_splits"],
        project_dir=tmp_path,
    )
    assert script == "infer_split.py"


def test_resolve_inference_script_falls_back_when_split_by_is_null_override(
    tmp_path: Path,
):
    """Use infer.py when user explicitly disables split_by."""
    conf_data_dir = tmp_path / "conf" / "data"
    conf_data_dir.mkdir(parents=True, exist_ok=True)
    config_path = conf_data_dir / "mock_data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "# @package data",
                "name: mock_data",
                "path: /tmp/mock.h5ad",
                "split_by: condition",
            ]
        )
    )

    script = resolve_inference_script(
        overrides=["data=mock_data", "data.split_by=null"],
        project_dir=tmp_path,
    )
    assert script == "infer.py"

"""Tests for SLURM inference launcher helpers."""

from pathlib import Path

from slurm_infer import (
    SlurmConfig,
    _slurm_time_to_minutes,
    apply_slurm_default_overrides,
    build_batch_script,
    build_submitit_multirun_command,
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


def test_build_batch_script_single_run_sets_full_thread_count():
    """Single-run batch script should use full CPU allocation for threads."""
    slurm = SlurmConfig(gpus=4, cpus_per_task=32)
    batch = build_batch_script(
        slurm=slurm,
        hydra_overrides=["data=bleo_study01_bleomycin"],
        project_dir=Path("/tmp/project"),
        is_multirun=False,
        inference_script="infer.py",
    )
    assert "export OMP_NUM_THREADS=32" in batch
    assert "export MKL_NUM_THREADS=32" in batch
    assert "export OPENBLAS_NUM_THREADS=32" in batch


def test_build_batch_script_does_not_enable_joblib_launcher():
    """Single-run batch script should not inject any joblib launcher overrides."""
    slurm = SlurmConfig(gpus=4, cpus_per_task=32)
    batch = build_batch_script(
        slurm=slurm,
        hydra_overrides=["data=bleo_study01_bleomycin"],
        project_dir=Path("/tmp/project"),
        is_multirun=False,
        inference_script="infer.py",
    )
    assert "hydra/launcher=joblib" not in batch
    assert "hydra.launcher.n_jobs=" not in batch


def test_slurm_time_to_minutes_supports_days_and_hours():
    """Convert SLURM D-HH:MM and HH:MM strings to integer minutes."""
    assert _slurm_time_to_minutes("0-04:00") == 240
    assert _slurm_time_to_minutes("2-01:30") == 2970
    assert _slurm_time_to_minutes("06:15") == 375


def test_build_submitit_multirun_command_enforces_one_gpu_per_job():
    """Submitit multirun command should enforce strict per-job GPU isolation."""
    slurm = SlurmConfig(
        partition="base",
        account="hybrid-modeling",
        gpus=8,
        cpus_per_task=16,
        mem=128,
        time="0-04:00",
        job_name="scribe_infer",
    )
    cmd = build_submitit_multirun_command(
        slurm=slurm,
        hydra_overrides=["data=bleo_splits/bleo_study01_control"],
        project_dir=Path("/tmp/project"),
        inference_script="infer.py",
    )
    assert "hydra/launcher=submitit_slurm" in cmd
    assert "hydra.launcher.gpus_per_node=1" in cmd
    assert "hydra.launcher.tasks_per_node=1" in cmd
    assert "hydra.launcher.array_parallelism=8" in cmd


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

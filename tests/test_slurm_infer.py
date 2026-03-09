"""Tests for SLURM inference launcher helpers."""

from pathlib import Path

from slurm_infer import (
    SlurmConfig,
    _slurm_time_to_minutes,
    apply_slurm_default_overrides,
    build_batch_script,
    build_split_submitit_orchestrator_command,
    build_submitit_multirun_command,
    get_data_config_names,
    resolve_inference_script,
    with_single_data_override,
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


def test_build_split_submitit_orchestrator_command_passes_split_overrides():
    """infer_split orchestration should receive submitit split launcher settings."""
    slurm = SlurmConfig(
        partition="base",
        account="hybrid-modeling",
        gpus=6,
        cpus_per_task=12,
        mem=256,
        time="0-06:00",
        job_name="scribe_infer",
    )
    cmd = build_split_submitit_orchestrator_command(
        slurm=slurm,
        hydra_overrides=["data=lung_bleo/lung_bleo_splits"],
        project_dir=Path("/tmp/project"),
    )
    assert cmd[0:2] == ["python", "infer_split.py"]
    assert "split.launcher=submitit_slurm" in cmd
    assert "data.n_jobs=6" in cmd
    assert "split.array_parallelism=6" in cmd
    assert "split.cpus_per_task=12" in cmd
    assert "split.mem_gb=256" in cmd
    assert "split.timeout_min=360" in cmd


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


def test_resolve_inference_script_uses_split_entrypoint_when_split_by_is_list(
    tmp_path: Path,
):
    """Select infer_split.py when split_by is a YAML list of column names."""
    conf_data_dir = tmp_path / "conf" / "data" / "batch_correction"
    conf_data_dir.mkdir(parents=True, exist_ok=True)
    config_path = conf_data_dir / "a549.yaml"
    # YAML list notation for split_by
    config_path.write_text(
        "\n".join(
            [
                "# @package data",
                "name: a549",
                "path: /tmp/mock.h5ad",
                "split_by:",
                "  - treatment",
                "  - kit",
            ]
        )
    )

    script = resolve_inference_script(
        overrides=["data=batch_correction/a549"],
        project_dir=tmp_path,
    )
    assert script == "infer_split.py"


def test_get_data_config_names_parses_comma_separated_data_override():
    """Extract a list of data config keys from a comma-separated data override."""
    names = get_data_config_names(
        ["model=nbdm", "data=panfibrosis/CKD/a,panfibrosis/IPF/b"]
    )
    assert names == ["panfibrosis/CKD/a", "panfibrosis/IPF/b"]


def test_with_single_data_override_replaces_existing_data_override():
    """Return overrides with exactly one data entry for a target dataset."""
    updated = with_single_data_override(
        ["data=d1,d2", "model=nbdm", "guide_rank=8"],
        data_name="d2",
    )
    assert updated == ["model=nbdm", "guide_rank=8", "data=d2"]


def test_resolve_inference_script_uses_split_when_any_multidata_config_has_split(
    tmp_path: Path,
):
    """Multi-data inputs should select infer_split.py if any config has split_by."""
    conf_data_dir = tmp_path / "conf" / "data" / "panfibrosis"
    conf_data_dir.mkdir(parents=True, exist_ok=True)
    (conf_data_dir / "a.yaml").write_text(
        "\n".join(
            [
                "# @package data",
                "name: a",
                "path: /tmp/a.h5ad",
                "split_by: condition",
            ]
        )
    )
    (conf_data_dir / "b.yaml").write_text(
        "\n".join(
            [
                "# @package data",
                "name: b",
                "path: /tmp/b.h5ad",
                "split_by: null",
            ]
        )
    )

    script = resolve_inference_script(
        overrides=["data=panfibrosis/a,panfibrosis/b"],
        project_dir=tmp_path,
    )
    assert script == "infer_split.py"

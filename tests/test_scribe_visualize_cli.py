"""Tests for ``scribe-visualize`` CLI wrapper behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.cli.slurm_common import SlurmPromptConfig
from scribe.cli.slurm_visualize import _build_batch_script
from scribe.cli.visualize import main as visualize_cli_main


def test_visualize_cli_delegates_to_pipeline_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local mode should forward all args to packaged visualization pipeline."""
    captured: dict[str, list[str]] = {}

    def _fake_pipeline_main(argv: list[str] | None = None) -> None:
        captured["argv"] = list(argv or [])

    monkeypatch.setattr(
        "scribe.cli.visualize.pipeline.main", _fake_pipeline_main
    )

    visualize_cli_main(["outputs/foo", "--recursive", "--all"])
    assert captured["argv"] == ["outputs/foo", "--recursive", "--all"]


def test_visualize_cli_forwards_recursive_pattern_and_explicit_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local mode should preserve optional recursive pattern tokens."""
    captured: dict[str, list[str]] = {}

    def _fake_pipeline_main(argv: list[str] | None = None) -> None:
        captured["argv"] = list(argv or [])

    monkeypatch.setattr(
        "scribe.cli.visualize.pipeline.main", _fake_pipeline_main
    )

    visualize_cli_main(
        [
            "outputs/foo/custom_results.pkl",
            "--recursive",
            "*_results.pkl",
            "--all",
        ]
    )
    assert captured["argv"] == [
        "outputs/foo/custom_results.pkl",
        "--recursive",
        "*_results.pkl",
        "--all",
    ]


def test_visualize_cli_profile_implies_slurm_submission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Providing ``--slurm-profile`` should trigger SLURM submission mode."""
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "scribe.cli.visualize.load_slurm_profile",
        lambda *_args, **_kwargs: (
            {
                "partition": "gpu",
                "cpus_per_task": 4,
                "mem_gb": 64,
                "timeout_min": 240,
            },
            tmp_path / "default.yaml",
        ),
    )
    monkeypatch.setattr(
        "scribe.cli.visualize.parse_slurm_set_entries",
        lambda _entries: ({}, {}),
    )
    monkeypatch.setattr(
        "scribe.cli.visualize.prompt_slurm_config",
        lambda seed_values, **_kwargs: SlurmPromptConfig(
            partition=str(seed_values["partition"]),
            account=None,
            array_parallelism=1,
            cpus_per_task=int(seed_values["cpus_per_task"]),
            mem_gb=int(seed_values["mem_gb"]),
            timeout_min=int(seed_values["timeout_min"]),
        ),
    )

    def _fake_submit(**kwargs) -> int:
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(
        "scribe.cli.visualize.submit_visualize_slurm_job",
        _fake_submit,
    )

    with pytest.raises(SystemExit) as exc_info:
        visualize_cli_main(
            [
                "--slurm-profile",
                "default",
                "outputs/foo",
                "--recursive",
            ]
        )

    assert exc_info.value.code == 0
    assert calls["forwarded_args"] == ["outputs/foo", "--recursive"]
    assert isinstance(calls["slurm_cfg"], SlurmPromptConfig)


def test_visualize_cli_slurm_keeps_recursive_pattern_and_file_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SLURM submission should forward recursive pattern + explicit file path."""
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "scribe.cli.visualize.load_slurm_profile",
        lambda *_args, **_kwargs: (
            {
                "partition": "gpu",
                "cpus_per_task": 2,
                "mem_gb": 16,
                "timeout_min": 60,
            },
            tmp_path / "default.yaml",
        ),
    )
    monkeypatch.setattr(
        "scribe.cli.visualize.parse_slurm_set_entries",
        lambda _entries: ({}, {}),
    )
    monkeypatch.setattr(
        "scribe.cli.visualize.prompt_slurm_config",
        lambda seed_values, **_kwargs: SlurmPromptConfig(
            partition=str(seed_values["partition"]),
            account=None,
            array_parallelism=1,
            cpus_per_task=int(seed_values["cpus_per_task"]),
            mem_gb=int(seed_values["mem_gb"]),
            timeout_min=int(seed_values["timeout_min"]),
        ),
    )

    def _fake_submit(**kwargs) -> int:
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(
        "scribe.cli.visualize.submit_visualize_slurm_job",
        _fake_submit,
    )

    with pytest.raises(SystemExit) as exc_info:
        visualize_cli_main(
            [
                "--slurm-profile",
                "default",
                "outputs/foo/custom_results.pkl",
                "--recursive",
                "*_results.pkl",
            ]
        )

    assert exc_info.value.code == 0
    assert calls["forwarded_args"] == [
        "outputs/foo/custom_results.pkl",
        "--recursive",
        "*_results.pkl",
    ]


def test_visualize_slurm_batch_script_includes_gres_when_configured(
    tmp_path: Path,
) -> None:
    """Raw ``sbatch`` script must emit ``--gres`` when ``SlurmPromptConfig`` sets it."""
    cfg = SlurmPromptConfig(
        partition="base",
        account=None,
        array_parallelism=1,
        cpus_per_task=4,
        mem_gb=64,
        timeout_min=60,
        gres="gpu:1",
        job_name="my_viz",
    )
    script = _build_batch_script(cfg, project_dir=tmp_path, forwarded_args=["out", "--all"])
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --job-name=my_viz" in script

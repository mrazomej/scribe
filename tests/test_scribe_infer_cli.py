"""Tests for the unified ``scribe-infer`` command behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.cli.dispatch import (
    extract_data_keys,
    should_use_split_mode,
)
from scribe.cli.infer import (
    SlurmPromptConfig,
    _build_parser,
    _parse_slurm_set_entries,
    main as infer_cli_main,
)


def test_extract_data_keys_expands_comma_delimited_values() -> None:
    """Expand comma-delimited `data=` values into individual config keys."""
    keys = extract_data_keys(["model=zinb", "data=a/b,c/d", "seed=42"])
    assert keys == ["a/b", "c/d"]


def test_should_use_split_mode_detects_split_by(tmp_path: Path) -> None:
    """Enable split mode when selected data config defines `split_by`."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "split_case.yaml").write_text(
        "# @package data\nname: split_case\npath: /tmp/mock.h5ad\nsplit_by: treatment\n"
    )
    assert should_use_split_mode(tmp_path, ["data=split_case"]) is True


def test_should_use_split_mode_returns_false_without_split_by(
    tmp_path: Path,
) -> None:
    """Keep direct mode when selected data config has no `split_by`."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "direct_case.yaml").write_text(
        "# @package data\nname: direct_case\npath: /tmp/mock.h5ad\n"
    )
    assert should_use_split_mode(tmp_path, ["data=direct_case"]) is False


def test_cli_dispatches_to_direct_infer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dispatch to direct infer module when split mode is not required."""
    commands: list[list[str]] = []

    def _fake_call(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(
        "scribe.cli.infer._ensure_hydra_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "scribe.cli.infer.should_use_split_mode", lambda *_: False
    )
    monkeypatch.setattr("scribe.cli.infer.subprocess.call", _fake_call)

    with pytest.raises(SystemExit) as exc_info:
        infer_cli_main(
            [
                "--config-path",
                str(tmp_path),
                "--config-name",
                "config",
                "data=direct_case",
                "model=zinb",
            ]
        )

    assert exc_info.value.code == 0
    assert len(commands) == 1
    assert commands[0][2] == "scribe.cli.infer_runner"
    assert "data=direct_case" in commands[0]


def test_cli_dispatches_to_split_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dispatch to split orchestrator module when split mode is required."""
    commands: list[list[str]] = []

    def _fake_call(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(
        "scribe.cli.infer._ensure_hydra_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "scribe.cli.infer.should_use_split_mode", lambda *_: True
    )
    monkeypatch.setattr("scribe.cli.infer.subprocess.call", _fake_call)

    with pytest.raises(SystemExit) as exc_info:
        infer_cli_main(
            [
                "--config-path",
                str(tmp_path),
                "data=split_case",
                "model=nbvcp",
            ]
        )

    assert exc_info.value.code == 0
    assert len(commands) == 1
    assert commands[0][2] == "scribe.cli.split_orchestrator"
    assert "data=split_case" in commands[0]


def test_help_text_mentions_split_conf_and_doc_pointer() -> None:
    """Help output should explain split behavior and conf layout guidance."""
    help_text = _build_parser().format_help()
    assert "split_by" in help_text
    assert "./conf" in help_text
    assert "conf/config.yaml" in help_text
    assert "docs/cli_infer.md" in help_text
    assert "pip install 'scribe[hydra]'" in help_text
    assert "--initialize" in help_text


def test_initialize_mode_skips_hydra_and_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialize mode should not run infer dispatch or Hydra checks."""
    calls = {"initialize": 0, "hydra_check": 0, "subprocess": 0}

    def _fake_initialize(arg: str | None) -> None:
        calls["initialize"] += 1
        assert arg == "./custom_conf"

    def _fake_hydra_check() -> None:
        calls["hydra_check"] += 1

    def _fake_subprocess(_cmd: list[str]) -> int:
        calls["subprocess"] += 1
        return 0

    monkeypatch.setattr("scribe.cli.infer.initialize_conf", _fake_initialize)
    monkeypatch.setattr(
        "scribe.cli.infer._ensure_hydra_extra_installed", _fake_hydra_check
    )
    monkeypatch.setattr("scribe.cli.infer.subprocess.call", _fake_subprocess)

    infer_cli_main(["--initialize", "./custom_conf"])
    assert calls["initialize"] == 1
    assert calls["hydra_check"] == 0
    assert calls["subprocess"] == 0


def test_cli_slurm_mode_dispatches_to_direct_submitit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Enable submitit launch path for non-split runs with ``--slurm``."""
    commands: list[list[str]] = []

    def _fake_call(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(
        "scribe.cli.infer._ensure_hydra_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "scribe.cli.infer.should_use_split_mode", lambda *_: False
    )
    monkeypatch.setattr(
        "scribe.cli.infer.prompt_slurm_config",
        lambda *_args, **_kwargs: SlurmPromptConfig(
            partition="gpuA100",
            account=None,
            array_parallelism=3,
            cpus_per_task=8,
            mem_gb=96,
            timeout_min=300,
        ),
    )
    monkeypatch.setattr("scribe.cli.infer.subprocess.call", _fake_call)

    with pytest.raises(SystemExit) as exc_info:
        infer_cli_main(
            [
                "--slurm",
                "--config-path",
                str(tmp_path),
                "data=direct_case",
                "model=zinb",
            ]
        )

    assert exc_info.value.code == 0
    assert len(commands) == 1
    assert commands[0][2] == "scribe.cli.infer_runner"
    assert "--multirun" in commands[0]
    assert "hydra/launcher=submitit_slurm" in commands[0]
    assert "hydra.launcher.partition=gpuA100" in commands[0]
    assert "hydra.launcher.array_parallelism=3" in commands[0]


def test_cli_slurm_mode_dispatches_to_split_submitit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward split submitit overrides when split mode is detected."""
    commands: list[list[str]] = []

    def _fake_call(cmd: list[str]) -> int:
        commands.append(cmd)
        return 0

    monkeypatch.setattr(
        "scribe.cli.infer._ensure_hydra_extra_installed", lambda: None
    )
    monkeypatch.setattr("scribe.cli.infer.should_use_split_mode", lambda *_: True)
    monkeypatch.setattr(
        "scribe.cli.infer.prompt_slurm_config",
        lambda *_args, **_kwargs: SlurmPromptConfig(
            partition="longq",
            account="lab123",
            array_parallelism=2,
            cpus_per_task=6,
            mem_gb=80,
            timeout_min=180,
        ),
    )
    monkeypatch.setattr("scribe.cli.infer.subprocess.call", _fake_call)

    with pytest.raises(SystemExit) as exc_info:
        infer_cli_main(
            [
                "--slurm",
                "--config-path",
                str(tmp_path),
                "data=split_case",
                "model=nbvcp",
            ]
        )

    assert exc_info.value.code == 0
    assert len(commands) == 1
    assert commands[0][2] == "scribe.cli.split_orchestrator"
    assert "split.launcher=submitit_slurm" in commands[0]
    assert "split.partition=longq" in commands[0]
    assert "split.account=lab123" in commands[0]


def test_parse_slurm_set_entries_supports_timeout_and_launcher_passthrough() -> None:
    """Parse known keys plus launcher passthrough from repeated entries."""
    core, launcher = _parse_slurm_set_entries(
        [
            "partition=gpu",
            "cpus_per_task=12",
            "timeout=0-06:00",
            "launcher.max_num_timeout=3",
        ]
    )
    assert core["partition"] == "gpu"
    assert core["cpus_per_task"] == 12
    assert core["timeout_min"] == 360
    assert launcher == {"max_num_timeout": "3"}


def test_cli_rejects_slurm_set_without_slurm_flag() -> None:
    """Guard against ambiguous launcher config when ``--slurm`` is absent."""
    with pytest.raises(SystemExit) as exc_info:
        infer_cli_main(["--slurm-set", "partition=gpu", "data=singer"])
    assert "--slurm-profile and --slurm-set require --slurm" in str(exc_info.value)


def test_cli_slurm_profile_and_set_merge_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use profile values as base and let ``--slurm-set`` override them."""
    commands: list[list[str]] = []
    captured_seed: dict[str, object] = {}

    (tmp_path / "slurm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "slurm" / "team.yaml").write_text(
        "\n".join(
            [
                "partition: base",
                "cpus_per_task: 4",
                "mem_gb: 32",
                "timeout_min: 120",
                "launcher_overrides:",
                "  max_num_timeout: 2",
            ]
        )
    )

    def _fake_prompt(seed_values, *, allow_interactive):
        captured_seed.update(seed_values)
        return SlurmPromptConfig(
            partition=str(seed_values["partition"]),
            account=seed_values.get("account"),
            array_parallelism=int(seed_values.get("array_parallelism", 1)),
            cpus_per_task=int(seed_values["cpus_per_task"]),
            mem_gb=int(seed_values["mem_gb"]),
            timeout_min=int(seed_values["timeout_min"]),
            launcher_overrides=seed_values.get("launcher_overrides"),
        )

    monkeypatch.setattr(
        "scribe.cli.infer._ensure_hydra_extra_installed", lambda: None
    )
    monkeypatch.setattr(
        "scribe.cli.infer.should_use_split_mode", lambda *_: False
    )
    monkeypatch.setattr("scribe.cli.infer.prompt_slurm_config", _fake_prompt)
    monkeypatch.setattr(
        "scribe.cli.infer.subprocess.call", lambda cmd: commands.append(cmd) or 0
    )

    with pytest.raises(SystemExit):
        infer_cli_main(
            [
                "--slurm",
                "--config-path",
                str(tmp_path),
                "--slurm-profile",
                "team",
                "--slurm-set",
                "cpus_per_task=16",
                "--slurm-set",
                "partition=boost",
                "--slurm-set",
                "launcher.comment=nightly",
                "data=direct_case",
            ]
        )

    assert captured_seed["partition"] == "boost"
    assert captured_seed["cpus_per_task"] == 16
    assert captured_seed["mem_gb"] == 32
    assert captured_seed["launcher_overrides"] == {
        "max_num_timeout": "2",
        "comment": "nightly",
    }
    assert "hydra.launcher.partition=boost" in commands[0]
    assert "hydra.launcher.cpus_per_task=16" in commands[0]
    assert "hydra.launcher.max_num_timeout=2" in commands[0]
    assert "hydra.launcher.comment=nightly" in commands[0]

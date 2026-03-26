"""Tests for the unified ``scribe-infer`` command behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.cli.dispatch import (
    extract_data_keys,
    should_use_split_mode,
)
from scribe.cli.infer import _build_parser, main as infer_cli_main


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
    assert commands[0][2] == "infer"
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
    assert commands[0][2] == "infer_split"
    assert "data=split_case" in commands[0]


def test_help_text_mentions_split_conf_and_doc_pointer() -> None:
    """Help output should explain split behavior and conf layout guidance."""
    help_text = _build_parser().format_help()
    assert "split_by" in help_text
    assert "./conf" in help_text
    assert "conf/config.yaml" in help_text
    assert "docs/cli_infer.md" in help_text
    assert "pip install 'scribe[hydra]'" in help_text

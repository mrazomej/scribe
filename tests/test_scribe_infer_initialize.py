"""Tests for `scribe-infer --initialize` helper logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe.cli import initialize as init_mod


def test_resolve_initialize_target_uses_explicit_path() -> None:
    """Explicit initialize path should always be honored."""
    resolved = init_mod.resolve_initialize_target("/tmp/custom_conf")
    assert resolved == Path("/tmp/custom_conf")


def test_resolve_initialize_target_non_interactive_defaults_conf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-path initialize should default to ./conf in non-interactive mode."""
    monkeypatch.setattr("scribe.cli.initialize._is_interactive", lambda: False)
    resolved = init_mod.resolve_initialize_target("")
    assert resolved == Path("./conf")


def test_resolve_initialize_target_interactive_custom_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive mode should accept user-entered custom path."""
    monkeypatch.setattr("scribe.cli.initialize._is_interactive", lambda: True)
    answers = iter(["n", "/tmp/from_prompt"])
    resolved = init_mod.resolve_initialize_target(
        "",
        input_func=lambda _prompt: next(answers),
    )
    assert resolved == Path("/tmp/from_prompt")


def test_initialize_conf_creates_all_managed_files(tmp_path: Path) -> None:
    """Initialize should scaffold all managed template files."""
    out_lines: list[str] = []
    summary = init_mod.initialize_conf(
        str(tmp_path / "conf"),
        input_func=lambda _prompt: "y",
        output_func=lambda line: out_lines.append(line),
    )
    assert summary.created_count == len(init_mod._TEMPLATE_REL_PATHS)
    assert summary.overwritten_count == 0
    assert summary.skipped_count == 0

    # Sanity-check critical files requested by the plan/user.
    assert (summary.target_root / "config.yaml").exists()
    assert (summary.target_root / "data" / "example.yaml").exists()
    assert (summary.target_root / "inference" / "svi.yaml").exists()
    assert (summary.target_root / "inference" / "mcmc.yaml").exists()
    assert (summary.target_root / "inference" / "vae.yaml").exists()
    assert (summary.target_root / "amortization" / "capture.yaml").exists()
    assert (summary.target_root / "dirname_aliases" / "default.yaml").exists()
    assert (summary.target_root / "paths" / "paths.yaml").exists()


def test_initialize_conf_prompts_before_overwrite(tmp_path: Path) -> None:
    """Existing files should be skipped when overwrite prompt is declined."""
    target_root = tmp_path / "conf"
    (target_root / "paths").mkdir(parents=True, exist_ok=True)
    existing = target_root / "paths" / "paths.yaml"
    existing.write_text("outputs_dir: keep_me\n")

    # Decline overwrite prompts so existing content remains untouched.
    summary = init_mod.initialize_conf(
        str(target_root), input_func=lambda _p: "n"
    )

    assert summary.skipped_count >= 1
    assert existing.read_text() == "outputs_dir: keep_me\n"

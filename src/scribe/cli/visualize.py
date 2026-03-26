"""Unified ``scribe-visualize`` command entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from scribe.viz import pipeline

from .slurm_common import (
    load_slurm_profile,
    parse_slurm_set_entries,
    prompt_slurm_config,
)
from .slurm_visualize import submit_visualize_slurm_job


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for SLURM wrapper options for visualization CLI."""
    parser = argparse.ArgumentParser(
        prog="scribe-visualize",
        add_help=False,
    )
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--slurm-profile", default=None, metavar="PROFILE")
    parser.add_argument(
        "--slurm-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    return parser


def _strip_slurm_tokens(argv: list[str]) -> list[str]:
    """Return argv without SLURM wrapper flags and their values."""
    stripped: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--slurm":
            i += 1
            continue
        if token == "--slurm-profile":
            i += 2
            continue
        if token == "--slurm-set":
            i += 2
            continue
        stripped.append(token)
        i += 1
    return stripped


def main(argv: list[str] | None = None) -> None:
    """Run visualization CLI locally or as a SLURM batch job."""
    raw_argv = list(argv or [])
    if argv is None:
        import sys

        raw_argv = list(sys.argv[1:])

    parser = _build_parser()
    known_args, forwarded = parser.parse_known_args(raw_argv)
    slurm_requested = bool(
        known_args.slurm
        or known_args.slurm_profile is not None
        or len(known_args.slurm_set) > 0
    )

    if not slurm_requested:
        pipeline.main(forwarded)
        return

    # Best-effort config-root resolution; visualization defaults do not require
    # config files, but we align SLURM profile lookup with other CLIs.
    config_root = Path("./conf").resolve()
    if "--config-path" in forwarded:
        idx = forwarded.index("--config-path")
        if idx + 1 < len(forwarded):
            config_root = Path(forwarded[idx + 1]).resolve()

    profile_values, _ = load_slurm_profile(
        known_args.slurm_profile, config_root=config_root
    )
    set_values, launcher_values = parse_slurm_set_entries(known_args.slurm_set)
    merged_values = {**profile_values, **set_values}
    merged_values["launcher_overrides"] = {
        **profile_values.get("launcher_overrides", {}),
        **launcher_values,
    }
    slurm_cfg = prompt_slurm_config(merged_values, allow_interactive=True)

    submission_args = _strip_slurm_tokens(raw_argv)
    project_dir = Path(__file__).resolve().parents[3]
    raise SystemExit(
        submit_visualize_slurm_job(
            slurm_cfg=slurm_cfg,
            project_dir=project_dir,
            forwarded_args=submission_args,
        )
    )


if __name__ == "__main__":
    main()


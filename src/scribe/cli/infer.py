"""Unified ``scribe-infer`` command entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .dispatch import should_use_split_mode
from .infer_help import DETAILED_DESCRIPTION, EPILOG
from .initialize import initialize_conf
from .slurm import (
    build_slurm_command as _build_slurm_command,
)
from .slurm_common import (
    SlurmPromptConfig,
    load_slurm_profile as _load_slurm_profile,
    parse_slurm_set_entries as _parse_slurm_set_entries,
    prompt_slurm_config,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for ``scribe-infer``.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for SCRIBE CLI options plus pass-through Hydra
        overrides.
    """
    parser = argparse.ArgumentParser(
        prog="scribe-infer",
        description=DETAILED_DESCRIPTION.strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG.strip(),
    )
    parser.add_argument(
        "--config-path",
        default="./conf",
        help=(
            "Hydra config root. Expected to contain config.yaml and data/*.yaml "
            "(default: ./conf)."
        ),
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Top-level Hydra config name (default: config).",
    )
    parser.add_argument(
        "--initialize",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Initialize a starter conf directory. If PATH is omitted, the CLI "
            "prompts in interactive mode or defaults to ./conf in non-interactive mode."
        ),
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "Launch via Hydra submitit_slurm with interactive resource prompts. "
            "Partition is required and has no default."
        ),
    )
    parser.add_argument(
        "--slurm-profile",
        default=None,
        metavar="PROFILE",
        help=(
            "Optional SLURM profile name (resolved under <config-path>/slurm as "
            "<PROFILE>.yaml) or explicit YAML path. Supplying this flag "
            "automatically enables SLURM launch mode."
        ),
    )
    parser.add_argument(
        "--slurm-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Optional per-run SLURM override. Repeat as needed. Supports common "
            "keys (partition, account, cpus_per_task, mem_gb, timeout/timeout_min, "
            "qos, constraint, exclude, nodelist, reservation, gres, mail_user, "
            "mail_type, job_name, submitit_folder) and launcher.<key> passthrough. "
            "Supplying this flag automatically enables SLURM launch mode."
        ),
    )
    return parser


def _ensure_hydra_extra_installed() -> None:
    """Check optional Hydra dependencies and fail with actionable guidance.

    Returns
    -------
    None
        Raises ``SystemExit`` with an installation hint when dependencies are
        missing.
    """
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing optional Hydra dependencies required by scribe-infer.\n"
            "Install with: pip install 'scribe[hydra]'\n"
            f"Original import error: {exc}"
        ) from exc


def _build_subprocess_command(
    module_name: str,
    config_path: str,
    config_name: str,
    forwarded_overrides: list[str],
) -> list[str]:
    """Build the Python module command used for downstream execution.

    Parameters
    ----------
    module_name : str
        Python module executed with ``python -m``.
    config_path : str
        Config directory path passed to Hydra-aware modules.
    config_name : str
        Top-level Hydra config name.
    forwarded_overrides : list[str]
        User-provided Hydra overrides.

    Returns
    -------
    list[str]
        Command suitable for ``subprocess.call``.
    """
    return [
        sys.executable,
        "-m",
        module_name,
        "--config-path",
        config_path,
        "--config-name",
        config_name,
        *forwarded_overrides,
    ]


def main(argv: list[str] | None = None) -> None:
    """Run unified SCRIBE inference CLI dispatch.

    Parameters
    ----------
    argv : list[str] or None, optional
        Optional CLI token override. When ``None``, uses process ``sys.argv``.
    """
    parser = _build_parser()
    known_args, forwarded = parser.parse_known_args(argv)

    # Initialize mode is available even without hydra extras because it only
    # writes starter YAML templates for users to edit.
    if known_args.initialize is not None:
        if known_args.slurm or known_args.slurm_profile or known_args.slurm_set:
            raise SystemExit(
                "--initialize cannot be combined with --slurm/--slurm-profile/--slurm-set."
            )
        initialize_conf(known_args.initialize)
        return

    # Any explicit SLURM profile/override implies SLURM launch mode.
    slurm_requested = bool(
        known_args.slurm
        or known_args.slurm_profile is not None
        or len(known_args.slurm_set) > 0
    )

    _ensure_hydra_extra_installed()

    config_root = Path(known_args.config_path).resolve()

    # Decide execution mode by inspecting the selected data config(s).
    split_mode = should_use_split_mode(config_root, forwarded)
    target_module = (
        "scribe.cli.split_orchestrator"
        if split_mode
        else "scribe.cli.infer_runner"
    )
    if slurm_requested:
        profile_values, _ = _load_slurm_profile(
            known_args.slurm_profile, config_root=config_root
        )
        set_values, launcher_values = _parse_slurm_set_entries(known_args.slurm_set)
        merged_values = {**profile_values, **set_values}
        merged_values["launcher_overrides"] = {
            **profile_values.get("launcher_overrides", {}),
            **launcher_values,
        }
        slurm_cfg = prompt_slurm_config(merged_values, allow_interactive=True)
        command = _build_slurm_command(
            split_mode=split_mode,
            config_path=str(config_root),
            config_name=known_args.config_name,
            forwarded_overrides=forwarded,
            slurm_cfg=slurm_cfg,
        )
    else:
        command = _build_subprocess_command(
            module_name=target_module,
            config_path=str(config_root),
            config_name=known_args.config_name,
            forwarded_overrides=forwarded,
        )
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()


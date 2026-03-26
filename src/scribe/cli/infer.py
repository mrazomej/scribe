"""Unified ``scribe-infer`` command entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .dispatch import should_use_split_mode
from .infer_help import DETAILED_DESCRIPTION, EPILOG


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
    _ensure_hydra_extra_installed()

    parser = _build_parser()
    known_args, forwarded = parser.parse_known_args(argv)
    config_root = Path(known_args.config_path).resolve()

    # Decide execution mode by inspecting the selected data config(s).
    split_mode = should_use_split_mode(config_root, forwarded)
    target_module = "infer_split" if split_mode else "infer"
    command = _build_subprocess_command(
        module_name=target_module,
        config_path=str(config_root),
        config_name=known_args.config_name,
        forwarded_overrides=forwarded,
    )
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()


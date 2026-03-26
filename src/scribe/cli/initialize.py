"""Initialize starter Hydra configuration files for ``scribe-infer``."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import sys
from typing import Callable


# Managed starter templates copied by ``scribe-infer --initialize``.
_TEMPLATE_REL_PATHS = [
    "conf/README.md",
    "conf/config.yaml",
    "conf/data/example.yaml",
    "conf/inference/svi.yaml",
    "conf/inference/mcmc.yaml",
    "conf/inference/vae.yaml",
    "conf/amortization/capture.yaml",
    "conf/dirname_aliases/default.yaml",
    "conf/paths/paths.yaml",
    "conf/paths/paths.local.yaml.example",
]


@dataclass(frozen=True)
class InitializeSummary:
    """Summary of initialize scaffold operations.

    Attributes
    ----------
    target_root : Path
        Absolute path where templates were written.
    created_count : int
        Number of files newly created.
    overwritten_count : int
        Number of existing files replaced after user confirmation.
    skipped_count : int
        Number of existing files left untouched.
    """

    target_root: Path
    created_count: int
    overwritten_count: int
    skipped_count: int


def _template_root() -> Path:
    """Return package path to the scaffold template root.

    Returns
    -------
    Path
        Filesystem path for ``scribe/cli/templates`` resources.
    """
    return Path(str(files("scribe.cli.templates")))


def _is_interactive() -> bool:
    """Return whether stdin is attached to a TTY.

    Returns
    -------
    bool
        ``True`` when the current process is interactive.
    """
    return bool(sys.stdin.isatty())


def resolve_initialize_target(
    initialize_arg: str | None,
    *,
    input_func: Callable[[str], str] = input,
) -> Path:
    """Resolve initialize target path from CLI input and interaction mode.

    Parameters
    ----------
    initialize_arg : str or None
        Raw value from ``--initialize``.
        - ``None`` means no initialize mode.
        - ``""`` means flag provided without explicit path.
        - non-empty value is treated as the target path.
    input_func : callable, optional
        Prompt function used for interactive path selection.

    Returns
    -------
    Path
        Target path where starter config files should be scaffolded.
    """
    default_target = Path("./conf")

    if initialize_arg is None:
        return default_target

    cleaned = initialize_arg.strip()
    if cleaned:
        return Path(cleaned)

    # With ``--initialize`` and no explicit path:
    # - interactive: ask user whether to use default or custom path.
    # - non-interactive: use default automatically.
    if not _is_interactive():
        return default_target

    use_default = input_func(
        "No initialize path provided. Use default './conf'? [Y/n]: "
    ).strip().lower()
    if use_default in ("", "y", "yes"):
        return default_target

    custom_path = input_func("Enter custom conf path: ").strip()
    if custom_path == "":
        return default_target
    return Path(custom_path)


def _should_overwrite_file(
    file_path: Path,
    *,
    input_func: Callable[[str], str] = input,
) -> bool:
    """Prompt user for overwrite confirmation on an existing file.

    Parameters
    ----------
    file_path : Path
        Existing file path that might be overwritten.
    input_func : callable, optional
        Prompt function used for user confirmation.

    Returns
    -------
    bool
        ``True`` if the user confirmed overwrite, otherwise ``False``.
    """
    response = input_func(
        f"File exists: {file_path}. Overwrite? [y/N]: "
    ).strip().lower()
    return response in ("y", "yes")


def initialize_conf(
    initialize_arg: str | None,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> InitializeSummary:
    """Create starter ``conf`` files for users running SCRIBE CLI.

    Parameters
    ----------
    initialize_arg : str or None
        Value from CLI ``--initialize`` option.
    input_func : callable, optional
        Prompt function for path and overwrite interactions.
    output_func : callable, optional
        Output printer used for status messages.

    Returns
    -------
    InitializeSummary
        Aggregate stats for created/overwritten/skipped scaffold files.
    """
    target_root = resolve_initialize_target(
        initialize_arg=initialize_arg,
        input_func=input_func,
    ).resolve()
    template_root = _template_root()

    created_count = 0
    overwritten_count = 0
    skipped_count = 0

    # Each managed template is copied relative to the conf root. Paths are
    # normalized so users can initialize into custom directories cleanly.
    for template_rel in _TEMPLATE_REL_PATHS:
        template_path = template_root / template_rel
        target_rel = Path(template_rel).relative_to("conf")
        target_path = target_root / target_rel

        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = template_path.read_text()

        if target_path.exists():
            if _should_overwrite_file(target_path, input_func=input_func):
                target_path.write_text(content)
                overwritten_count += 1
                output_func(f"Overwritten: {target_path}")
            else:
                skipped_count += 1
                output_func(f"Skipped: {target_path}")
            continue

        target_path.write_text(content)
        created_count += 1
        output_func(f"Created: {target_path}")

    summary = InitializeSummary(
        target_root=target_root,
        created_count=created_count,
        overwritten_count=overwritten_count,
        skipped_count=skipped_count,
    )
    output_func(
        "Initialization complete: "
        f"created={summary.created_count}, "
        f"overwritten={summary.overwritten_count}, "
        f"skipped={summary.skipped_count}. "
        f"Target: {summary.target_root}"
    )
    return summary


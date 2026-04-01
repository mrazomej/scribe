"""
Progress backend adapters for SVI training loops.

This module centralizes progress rendering policy so inference code can choose a
UI backend (Rich, tqdm, or no-op) based on runtime environment and explicit
user preference.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal, Optional, Protocol
import sys

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

ProgressBackendName = Literal["auto", "rich", "tqdm", "none"]
ResolvedProgressBackend = Literal["rich", "tqdm", "none"]


class ProgressReporter(Protocol):
    """
    Protocol for progress reporter implementations.

    Implementations expose a minimal, backend-agnostic API consumed by the SVI
    loop.  Keeping this contract small avoids leaking backend-specific concepts
    (for example Rich task IDs) into inference code.
    """

    def start(
        self, *, description: str, total: int, completed: int, loss_info: str
    ) -> None:
        """Initialize and render a progress task."""

    def update(
        self, *, advance: int = 1, loss_info: Optional[str] = None
    ) -> None:
        """Advance progress and optionally refresh displayed loss metadata."""

    def print_message(self, message: str) -> None:
        """Emit an informational message through the active backend."""

    def close(self) -> None:
        """Tear down backend resources (context managers, handles, etc.)."""


def _is_marimo_notebook() -> bool:
    """
    Return ``True`` when running inside a marimo notebook session.

    Returns
    -------
    bool
        ``True`` when marimo APIs indicate notebook mode; otherwise ``False``.
    """
    try:
        import marimo as mo

        # Prefer marimo's explicit notebook detector when available.
        running_in_notebook = getattr(mo, "running_in_notebook", None)
        if callable(running_in_notebook):
            return bool(running_in_notebook())
        # Fallback for older marimo versions.
        running_as_script = getattr(mo, "running_as_script", None)
        if callable(running_as_script):
            return running_as_script() is False
    except Exception:
        return False
    return False


def _is_ipython_notebook() -> bool:
    """
    Return ``True`` when running in an IPython notebook-like shell.

    Returns
    -------
    bool
        ``True`` for ZMQ-backed notebook shells (Jupyter/Lab/Colab).
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _is_tty_stdout() -> bool:
    """
    Return ``True`` when stdout is attached to an interactive terminal.

    Returns
    -------
    bool
        ``True`` when ``sys.stdout`` supports and reports ``isatty()``.
    """
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(callable(isatty) and isatty())


def _tqdm_available() -> bool:
    """
    Check whether ``tqdm.auto`` can be imported.

    Returns
    -------
    bool
        ``True`` when ``tqdm.auto`` import succeeds.
    """
    try:
        import tqdm.auto  # noqa: F401

        return True
    except Exception:
        return False


def resolve_progress_backend(
    progress_backend: ProgressBackendName = "auto",
) -> ResolvedProgressBackend:
    """
    Resolve the runtime progress backend from a policy name.

    Parameters
    ----------
    progress_backend : {"auto", "rich", "tqdm", "none"}, default="auto"
        Requested backend policy. ``"auto"`` selects a backend from runtime
        environment signals.

    Returns
    -------
    {"rich", "tqdm", "none"}
        Concrete backend to instantiate.
    """
    # Honor explicit user requests first.
    if progress_backend == "none":
        return "none"
    if progress_backend == "rich":
        return "rich"
    if progress_backend == "tqdm":
        return "tqdm" if _tqdm_available() else "none"

    # Auto policy:
    # - marimo / notebook frontends: prefer tqdm fallback for live updates.
    # - terminal sessions: keep current rich behavior.
    # - non-interactive streams: suppress interactive bars.
    if _is_marimo_notebook() or _is_ipython_notebook():
        return "tqdm" if _tqdm_available() else "none"
    if _is_tty_stdout():
        return "rich"
    return "none"


@dataclass
class RichProgressReporter(AbstractContextManager):
    """
    Rich-based progress renderer for terminal sessions.

    Parameters
    ----------
    progress : rich.progress.Progress
        Configured Rich progress object used for rendering.
    """

    progress: Progress
    _task_id: Optional[int] = None

    def __enter__(self) -> "RichProgressReporter":
        """Start the Rich progress context."""
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        """Close the Rich progress context."""
        self.progress.__exit__(exc_type, exc, tb)
        return None

    def start(
        self, *, description: str, total: int, completed: int, loss_info: str
    ) -> None:
        """Create a new Rich progress task for the SVI run."""
        self._task_id = self.progress.add_task(
            description,
            total=total,
            completed=completed,
            loss_info=loss_info,
        )

    def update(
        self, *, advance: int = 1, loss_info: Optional[str] = None
    ) -> None:
        """Advance the Rich task and refresh loss metadata when provided."""
        if self._task_id is None:
            return
        if loss_info is None:
            self.progress.update(self._task_id, advance=advance)
        else:
            self.progress.update(
                self._task_id, advance=advance, loss_info=loss_info
            )

    def print_message(self, message: str) -> None:
        """Print a message through the Progress console."""
        self.progress.console.print(message)

    def close(self) -> None:
        """Rich context cleanup is handled by ``__exit__``."""
        return


@dataclass
class TqdmProgressReporter(AbstractContextManager):
    """
    tqdm-based progress renderer for notebook-like sessions.

    Attributes
    ----------
    _bar : object or None
        Runtime tqdm progress bar object, initialized in ``start``.
    """

    _bar: Optional[object] = None

    def start(
        self, *, description: str, total: int, completed: int, loss_info: str
    ) -> None:
        """Create a tqdm progress bar."""
        from tqdm.auto import tqdm

        # Keep tqdm format compact while preserving rich-like metadata in
        # postfix text.
        self._bar = tqdm(
            total=total,
            initial=completed,
            desc=description,
            dynamic_ncols=True,
            leave=True,
        )
        self._bar.set_postfix_str(loss_info, refresh=False)

    def update(
        self, *, advance: int = 1, loss_info: Optional[str] = None
    ) -> None:
        """Advance tqdm and optionally update postfix metadata."""
        if self._bar is None:
            return
        self._bar.update(advance)
        if loss_info is not None:
            self._bar.set_postfix_str(loss_info, refresh=False)

    def print_message(self, message: str) -> None:
        """Emit messages in a tqdm-safe way."""
        if self._bar is None:
            print(message)
            return
        self._bar.write(message)

    def close(self) -> None:
        """Close tqdm handle if it was created."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        """Ensure the tqdm bar is closed on context exit."""
        self.close()
        return None


class NullProgressReporter(AbstractContextManager):
    """
    No-op progress renderer used when interactive bars are disabled.
    """

    def start(
        self, *, description: str, total: int, completed: int, loss_info: str
    ) -> None:
        """No-op start for disabled progress mode."""
        return

    def update(
        self, *, advance: int = 1, loss_info: Optional[str] = None
    ) -> None:
        """No-op update for disabled progress mode."""
        return

    def print_message(self, message: str) -> None:
        """Fallback plain-text message printing."""
        print(message)

    def close(self) -> None:
        """No-op close for disabled progress mode."""
        return

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        """No-op context exit."""
        return None


def build_progress_reporter(
    *,
    progress: bool,
    progress_backend: ProgressBackendName,
) -> ProgressReporter:
    """
    Build a concrete progress reporter from runtime policy.

    Parameters
    ----------
    progress : bool
        Master on/off flag for interactive progress bars.
    progress_backend : {"auto", "rich", "tqdm", "none"}
        Backend preference policy.

    Returns
    -------
    ProgressReporter
        Concrete reporter implementation used by the training loop.
    """
    if not progress:
        return NullProgressReporter()

    resolved_backend = resolve_progress_backend(progress_backend)

    if resolved_backend == "rich":
        rich_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[loss_info]}"),
            console=Console(),
            disable=False,
        )
        return RichProgressReporter(progress=rich_progress)

    if resolved_backend == "tqdm":
        return TqdmProgressReporter()

    return NullProgressReporter()

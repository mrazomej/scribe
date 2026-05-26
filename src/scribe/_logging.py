"""Centralized logging configuration for scribe.

Provides colored, readable log output via :class:`rich.logging.RichHandler`.
Users see output like::

    [11:30:05] INFO     [scribe.laplace.priors] Building TSLN-rate priors...
    [11:30:06] WARNING  [scribe.api.stages.gene_coverage] Applied gene_coverage...

The handler is attached to the ``"scribe"`` root logger at INFO level by default.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

# Module-level flag so setup_logging() is idempotent across repeated imports.
_LOGGING_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the scribe library logger with RichHandler.

    Parameters
    ----------
    level : int
        Logging level for the ``"scribe"`` logger hierarchy.
        Use ``logging.WARNING`` to suppress informational messages.
    """
    global _LOGGING_CONFIGURED

    logger = logging.getLogger("scribe")
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
            log_time_format="[%X]",
        )
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        # Keep scribe messages on our RichHandler only. Without this, records
        # also bubble to the root logger (e.g. Jupyter / marimo default
        # handlers) and appear twice with different formatting.
        logger.propagate = False
        _LOGGING_CONFIGURED = True
    logger.setLevel(level)

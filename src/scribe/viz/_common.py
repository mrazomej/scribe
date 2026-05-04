"""Shared utilities for the packaged ``scribe.viz`` module."""

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


# Keep model-family checks in one place so each plotting module uses the same
# robust enum/string matching rules across current and legacy result objects.
def _is_pln_model(results) -> bool:
    """Return whether a results object uses the PLN parameterization.

    Parameters
    ----------
    results : object
        Results-like object produced by Scribe inference (for example
        ``ScribeSVIResults`` / ``ScribeVAEResults``). The object is expected
        to expose ``model_config.parameterization`` as either an enum member
        or a plain string.

    Returns
    -------
    bool
        ``True`` when ``parameterization`` corresponds to
        ``poisson_lognormal`` (enum name ``POISSON_LOGNORMAL``), otherwise
        ``False``.

    Notes
    -----
    The check intentionally supports both enum and string forms so it remains
    stable for:

    - freshly-created configs where ``parameterization`` is an enum member;
    - deserialized / legacy objects where the value may be persisted as a
      lowercase string.
    """
    param = getattr(
        getattr(results, "model_config", None), "parameterization", None
    )
    param_value = getattr(param, "value", param)
    param_name = getattr(param, "name", None)
    return (
        isinstance(param_value, str)
        and param_value == "poisson_lognormal"
    ) or (
        isinstance(param_name, str) and param_name == "POISSON_LOGNORMAL"
    )

__all__ = [
    "console",
    "_is_pln_model",
    "Progress",
    "SpinnerColumn",
    "BarColumn",
    "TextColumn",
    "TimeElapsedColumn",
]

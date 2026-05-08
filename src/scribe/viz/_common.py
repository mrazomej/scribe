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
    """Return whether a results object uses the POISSON_LOGNORMAL parameterization.

    This includes both the Poisson-LogNormal (``model="pln"``) and the
    NB-LogNormal (``model="nbln"``) families: both share the same
    log-rate decoder and the same ``μ, W, d`` global structure on
    log-rate latents, with NBLN adding gene dispersion ``r_g`` on
    top.  Almost every viz site that gates on this check wants the
    PLN-family behaviour (e.g. "skip NB-family count diagnostics",
    "route to Gaussian-on-log-rates correlation accessors"); both
    PLN and NBLN belong to that family.  When NBLN-specific behaviour
    is needed, callers should additionally check
    ``model_config.base_model``.

    Parameters
    ----------
    results : object
        Results-like object produced by Scribe inference (for example
        ``ScribeSVIResults`` / ``ScribeVAEResults`` / ``ScribeLaplaceResults``).
        The object is expected to expose ``model_config.parameterization``
        as either an enum member or a plain string.

    Returns
    -------
    bool
        ``True`` when ``parameterization`` corresponds to
        ``poisson_lognormal`` (enum name ``POISSON_LOGNORMAL``).  This
        is the case for both ``base_model="pln"`` and
        ``base_model="nbln"``.

    Notes
    -----
    The check intentionally supports both enum and string forms so it remains
    stable for:

    - freshly-created configs where ``parameterization`` is an enum member;
    - deserialized / legacy objects where the value may be persisted as a
      lowercase string.

    The legacy name ``_is_pln_model`` is kept for backward compatibility;
    the alias :data:`_is_pln_family_model` documents the broader semantics
    for new call sites.
    """
    param = getattr(
        getattr(results, "model_config", None), "parameterization", None
    )
    param_value = getattr(param, "value", param)
    param_name = getattr(param, "name", None)
    # Accept both the canonical name (``count_lognormal`` /
    # ``COUNT_LOGNORMAL``) and the legacy name (``poisson_lognormal`` /
    # ``POISSON_LOGNORMAL``) so that pre-rename pickles persisted as
    # raw strings also resolve correctly.  The runtime ``Parameterization``
    # alias ensures ``Parameterization.POISSON_LOGNORMAL is
    # Parameterization.COUNT_LOGNORMAL`` for in-memory objects, but
    # name/value-string forms of either label are also valid here.
    return (
        isinstance(param_value, str)
        and param_value in ("count_lognormal", "poisson_lognormal")
    ) or (
        isinstance(param_name, str)
        and param_name in ("COUNT_LOGNORMAL", "POISSON_LOGNORMAL")
    )


# Forward-looking alias.  ``_is_pln_model`` historically named the
# check after PLN, but NBLN also lives in this family.  New code
# should prefer ``_is_pln_family_model`` for clarity.
_is_pln_family_model = _is_pln_model


def _is_nbln_model(results) -> bool:
    """Return whether a results object is specifically the NB-LogNormal model.

    Differs from :func:`_is_pln_model` (which returns True for both
    PLN and NBLN since they share the POISSON_LOGNORMAL parameterization):
    this checks ``model_config.base_model == "nbln"`` strictly.  Use
    when downstream code needs to distinguish PLN's Poisson observation
    channel from NBLN's NB observation channel (e.g. count-noise PPCs,
    accessing NBLN's gene dispersion ``r``).
    """
    bm = getattr(getattr(results, "model_config", None), "base_model", None)
    bm_value = getattr(bm, "value", bm)
    return isinstance(bm_value, str) and bm_value == "nbln"


__all__ = [
    "console",
    "_is_pln_model",
    "_is_pln_family_model",
    "_is_nbln_model",
    "Progress",
    "SpinnerColumn",
    "BarColumn",
    "TextColumn",
    "TimeElapsedColumn",
]

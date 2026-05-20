"""Shared utilities for the packaged ``scribe.viz`` module."""

import warnings

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Suppress Rich's repeated "install ipywidgets for Jupyter support"
# warning that fires on every console.print / progress.update call
# in IPython / marimo environments where ipywidgets is not installed.
warnings.filterwarnings(
    "ignore",
    message=r'install "ipywidgets"',
    category=UserWarning,
)

console = Console()


# Keep model-family checks in one place so each plotting module uses the same
# robust enum/string matching rules across current and legacy result objects.
def _is_pln_model(results) -> bool:
    """Return whether a results object is in the PLN-family of log-rate-latent models.

    The PLN family is the set of models whose latent decoder is
    ``μ + W z + sqrt(d) ε`` in log-rate (PLN/NBLN) or activation-log-odds-
    shifted (TSLN) coordinates.  Members:

    - **PLN** (``base_model="pln"``, ``parameterization="count_lognormal"``).
    - **NBLN** (``base_model="nbln"``, same parameterization).
    - **TSLN-Rate** (``base_model="twostate_ln_rate"``, with a
      ``two_state_*`` parameterization inherited from the upstream SVI
      source).  Although TSLN-Rate's parameterization string is not
      ``count_lognormal``, it shares the ``μ, W, d`` log-rate decoder
      and the same gauge-along-all-ones structure, so the viz call
      sites that gate on this check (mean-calibration's PLN path,
      correlation accessors, gauge diagnostics) want it.

    Almost every viz site that gates on this check wants the PLN-family
    behaviour (e.g. "skip NB-family count diagnostics", "route to
    Gaussian-on-log-rates correlation accessors").  When PLN-specific
    or NBLN-specific behaviour is needed, callers should additionally
    check ``model_config.base_model``.

    Parameters
    ----------
    results : object
        Results-like object produced by Scribe inference (for example
        ``ScribeSVIResults`` / ``ScribeVAEResults`` / ``ScribeLaplaceResults``).

    Returns
    -------
    bool
        ``True`` for PLN, NBLN, and TSLN-Rate.

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
    model_config = getattr(results, "model_config", None)
    param = getattr(model_config, "parameterization", None)
    param_value = getattr(param, "value", param)
    param_name = getattr(param, "name", None)
    # Accept both the canonical name (``count_lognormal`` /
    # ``COUNT_LOGNORMAL``) and the legacy name (``poisson_lognormal`` /
    # ``POISSON_LOGNORMAL``) so that pre-rename pickles persisted as
    # raw strings also resolve correctly.  The runtime ``Parameterization``
    # alias ensures ``Parameterization.POISSON_LOGNORMAL is
    # Parameterization.COUNT_LOGNORMAL`` for in-memory objects, but
    # name/value-string forms of either label are also valid here.
    if (
        isinstance(param_value, str)
        and param_value in ("count_lognormal", "poisson_lognormal")
    ) or (
        isinstance(param_name, str)
        and param_name in ("COUNT_LOGNORMAL", "POISSON_LOGNORMAL")
    ):
        return True
    # TSLN-Rate and TSLN-Logit use ``two_state_*`` parameterizations
    # but share the ``μ + Wz`` decoder structure with PLN/NBLN.
    # (TSLN-Logit's ``μ`` is zeros — the gene baseline lives in
    # ``eta_anchor`` — but the latent prior on ``z`` and the W/d
    # low-rank structure are identical.)  Gate on the base_model so
    # the viz call sites flow through the PLN path for both.
    bm = getattr(model_config, "base_model", None)
    bm_value = getattr(bm, "value", bm)
    return isinstance(bm_value, str) and bm_value in (
        "twostate_ln_rate", "twostate_ln_logit",
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

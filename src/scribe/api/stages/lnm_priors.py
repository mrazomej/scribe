"""
Stage 2c-LNM: Auto-calibrate LNM-family priors from data.

Delegates to :func:`scribe.core.lnm_data_init.resolve_lnm_priors`
which computes moment-of-moments or biology-informed defaults for
``r_T`` / ``mu_T`` / ``phi_T`` depending on whether the capture
anchor is active.

FitContext reads : model, count_data, priors, kwargs[parameterization]
FitContext writes: priors (may add auto-calibrated entries)
"""

import logging

import jax.numpy as jnp

from ..context import FitContext

_log = logging.getLogger(__name__)


def resolve_lnm_auto_priors(ctx: FitContext) -> None:
    """
    Auto-set LNM-family scalar priors from data moments.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.priors`` may be updated with
        auto-calibrated prior entries.
    """
    from ...core.lnm_data_init import resolve_lnm_priors, CAPTURE_ANCHOR_KEYS

    parameterization = ctx.kwargs.get("parameterization", "canonical")

    _resolved_priors = resolve_lnm_priors(
        ctx.model, parameterization, ctx.count_data, ctx.priors
    )
    if not _resolved_priors:
        return

    # Never mutate a caller's dict.
    priors = dict(ctx.priors) if ctx.priors is not None else {}
    priors.update(_resolved_priors)

    # Log a summary so the user can confirm the auto-defaults.
    _capture_active = any(k in priors for k in CAPTURE_ANCHOR_KEYS)
    _parts = []
    for _k, (_mu_log, _sigma_log) in _resolved_priors.items():
        _parts.append(
            f"{_k}=LogNormal(mu={_mu_log:.3f}, "
            f"sigma={_sigma_log:.3f}, "
            f"median={float(jnp.exp(_mu_log)):.1f})"
        )
    _log.info(
        "LNM[%s]: auto-set priors (%s anchor): %s. "
        "Override any of these via priors=...",
        parameterization,
        "capture" if _capture_active else "no capture",
        ", ".join(_parts),
    )

    ctx.priors = priors

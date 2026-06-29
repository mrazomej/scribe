"""
Stage 0: Resolve model string from boolean feature flags.

When ``variable_capture`` or ``zero_inflation`` is explicitly set, the
model string is derived from the flags.  An explicit ``model=`` that
conflicts with the flags raises ``ValueError``.

FitContext reads : model, priors, kwargs[variable_capture, zero_inflation]
FitContext writes: model (resolved)
"""

import logging

from ..context import FitContext

_log = logging.getLogger(__name__)


def resolve_model_flags(ctx: FitContext) -> None:
    """
    Derive the canonical model name from the boolean feature flags.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.model`` is updated in place with
        the resolved model string.

    Raises
    ------
    ValueError
        If the explicit ``model=`` conflicts with the feature flags, or
        if zero-inflation is requested for a model family that does not
        support it (LNM, PLN).
    """
    kw = ctx.kwargs
    variable_capture = kw.get("variable_capture")
    zero_inflation = kw.get("zero_inflation")
    priors = ctx.priors
    model = ctx.model

    _is_lnm_model = model.lower() in ("lnm", "lnmvcp")
    # NBLN sits in the PLN family: it shares the VAE/Laplace inference
    # path and the POISSON_LOGNORMAL parameterization, with capture as
    # an internal flag (no separate "nblnvcp" model string).
    _is_pln_model = model.lower() in ("pln", "nbln")
    _is_twostate_model = model.lower() in ("twostate", "twostatevcp")
    _is_twostate_ln_model = model.lower() in (
        "twostate_ln_rate", "twostate_ln_logit"
    )
    _default_model = "nbvcp"

    # TSLN family: capture is supplied via cascade or biology-anchored
    # prior (capture_anchor); not via the variable_capture flag.
    # Reject the flags up-front with a clear message.
    if _is_twostate_ln_model:
        if variable_capture is not None or zero_inflation is not None:
            # Per-variant guidance: TSLN-Rate accepts capture_anchor
            # priors; TSLN-Logit does NOT (Rev 4 — joint Newton with
            # the cross-Hessian is deferred to phase 3).
            if model.lower() == "twostate_ln_logit":
                _capture_hint = (
                    "Capture for 'twostate_ln_logit' is supported only "
                    "as a fixed offset from the cascade source's per-cell "
                    "eta_capture / p_capture (PR-2 Rev 4).  Biology-"
                    "anchored capture_anchor is deferred to phase 3."
                )
            else:
                _capture_hint = (
                    "Capture is supplied through the cascade_source's "
                    "eta_capture (when available) or through an explicit "
                    "capture_anchor / biology-informed prior."
                )
            raise ValueError(
                f"model='{model}' does not accept variable_capture or "
                "zero_inflation flags. "
                + _capture_hint
                + "  Drop variable_capture and zero_inflation."
            )

        # PR-2 capture restriction (Rev 4): TSLN-Logit cannot accept
        # ANY biology-informed capture prior because activating it
        # would require the joint (z, eta_cap) Newton path whose
        # cross-Hessian is deferred to phase 3.  Detect the prior
        # here at config-resolution time so the user gets a clear
        # message at the API boundary rather than a downstream
        # NotImplementedError from the obs-model constructor.
        #
        # Reject ALL of the following keys (matching the canonical
        # set in ``builder._CAPTURE_PRIOR_KEYS`` plus the user-facing
        # aliases from ``parameter_mapping.PRIOR_KEY_ALIASES``):
        #
        #   * ``capture_efficiency``  user-facing alias for eta_capture
        #   * ``eta_capture``         direct internal key
        #   * ``organism``            biology-shortcut → resolves to
        #                              eta_capture via the organism table
        #   * ``mu_eta``              hierarchical capture scaling
        #   * ``capture_scaling``     alias for mu_eta
        #
        # The capture-scaling family now lives in the priors dict under
        # ``capture_scaling`` / ``mu_eta`` (no ``capture_scaling_prior`` kwarg),
        # so the priors-key scan below covers every capture configuration.
        if model.lower() == "twostate_ln_logit":
            _capture_keys = {
                "capture_efficiency",
                "eta_capture",
                "organism",
                "mu_eta",
                "capture_scaling",
            }
            _priors_dict = (
                priors if isinstance(priors, dict) else None
            )
            _capture_keys_present = (
                sorted(set(_priors_dict) & _capture_keys)
                if _priors_dict is not None
                else []
            )
            if _capture_keys_present:
                _bits = [f"priors keys {_capture_keys_present!r}"]
                raise NotImplementedError(
                    f"model='twostate_ln_logit' does not accept "
                    "biology-informed capture priors (received "
                    + " and ".join(_bits)
                    + ") in PR-2: the joint (z, eta_cap) Newton "
                    "path with the proper cross-Hessian is deferred "
                    "to phase 3.  Supported capture configurations:\n"
                    "  * No capture: drop the capture priors.\n"
                    "  * Fixed-offset capture from cascade: drop the "
                    "capture priors and pass "
                    "informative_priors_from=svi_results where the SVI "
                    "source already has per-cell capture; the cascade "
                    "adapter will pin eta as a fixed offset "
                    "(x_only_offset Newton)."
                )
        return

    if variable_capture is None and zero_inflation is None:
        return

    _zi = zero_inflation if zero_inflation is not None else False
    _vc = variable_capture if variable_capture is not None else True

    # -- PLN family: capture is an internal flag, no "plnvcp" string ----------
    if _is_pln_model:
        if _zi:
            raise ValueError(
                "Zero-inflation is not supported for the PLN family "
                "(model='pln'). Drop zero_inflation=True "
                "or pick a DM-family model."
            )
        if _vc:
            # Warn if no capture prior was supplied to actually activate it.
            from ...core.lnm_data_init import CAPTURE_ANCHOR_KEYS
            from ...models.config.parameter_mapping import PRIOR_KEY_ALIASES

            _capture_alias_set = {
                alias
                for alias, target in PRIOR_KEY_ALIASES.items()
                if target in CAPTURE_ANCHOR_KEYS
            }
            _all_capture_keys = set(CAPTURE_ANCHOR_KEYS) | _capture_alias_set
            _has_capture_prior = isinstance(priors, dict) and any(
                k in priors for k in _all_capture_keys
            )
            if not _has_capture_prior:
                _log.warning(
                    "variable_capture=True with model='pln' has no "
                    "effect unless you also supply a capture prior "
                    "(e.g. priors={'capture_efficiency': (log_M0, "
                    "sigma_M)} or priors={'organism': 'human'}). "
                    "The PLN model will be fitted without capture "
                    "correction."
                )

    # -- TwoState (Poisson-Beta) family: zero-inflation not supported in phase 1
    elif _is_twostate_model:
        if _zi:
            raise ValueError(
                "Zero-inflation is not supported for the TwoState family in "
                "phase 1. Drop zero_inflation=True or pick a DM-family model."
            )
        _resolved = "twostatevcp" if _vc else "twostate"
        # Accept the hint when ``model='twostate'`` was passed alongside
        # ``variable_capture=True``; reject the contradictory inverse
        # (``model='twostatevcp', variable_capture=False``).
        if model.lower() != "twostate" and model.lower() != _resolved:
            raise ValueError(
                f"model='{model}' conflicts with the feature flags "
                f"(zero_inflation={zero_inflation}, "
                f"variable_capture={variable_capture}) which resolve to "
                f"'{_resolved}'. Pass model='twostate' to let "
                f"variable_capture select the variant, or set "
                f"variable_capture to match the model name."
            )
        ctx.model = _resolved

    # -- LNM family: zero-inflation not supported -----------------------------
    elif _is_lnm_model:
        if _zi:
            raise ValueError(
                "Zero-inflation is not supported for the LNM family "
                "(model='lnm' / 'lnmvcp'). Drop zero_inflation=True "
                "or pick a DM-family model."
            )
        _resolved = "lnmvcp" if _vc else "lnm"
        if model.lower() != "lnm" and model.lower() != _resolved:
            raise ValueError(
                f"model='{model}' conflicts with the feature flags "
                f"(zero_inflation={zero_inflation}, "
                f"variable_capture={variable_capture}) which resolve to "
                f"'{_resolved}'. Pass model='lnm' to let "
                f"variable_capture select the variant, or set "
                f"variable_capture to match the model name."
            )
        ctx.model = _resolved

    # -- DM family: standard NB / ZINB / VCP resolution -----------------------
    else:
        _resolved = (
            "zinbvcp"
            if _zi and _vc
            else "zinb" if _zi else "nbvcp" if _vc else "nbdm"
        )
        if model.lower() != _default_model and model.lower() != _resolved:
            raise ValueError(
                f"model='{model}' conflicts with the feature flags "
                f"(zero_inflation={zero_inflation}, "
                f"variable_capture={variable_capture}) which resolve to "
                f"'{_resolved}'. Use one or the other, not both."
            )
        ctx.model = _resolved

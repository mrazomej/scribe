"""
Stage 1: Validate user inputs before any data processing.

Checks model name, parameterization, inference method, and ``svi_init``
constraints.  Normalizes deprecated model aliases.

FitContext reads : model, kwargs[parameterization, inference_method,
                   svi_init, model_config, inference_config]
FitContext writes: model (after alias normalization)
"""

import logging

from ..constants import (
    VALID_MODELS,
    VALID_PARAMETERIZATIONS,
    VALID_INFERENCE_METHODS,
    _DEPRECATED_MODEL_ALIASES,
)
from ..context import FitContext

_log = logging.getLogger(__name__)


def validate_inputs(ctx: FitContext) -> None:
    """
    Validate model, parameterization, inference method, and svi_init.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.model`` may be updated if a
        deprecated alias is detected.

    Raises
    ------
    ValueError
        If model, parameterization, or inference_method is unrecognized,
        or if ``svi_init`` is used with a non-MCMC method.
    """
    kw = ctx.kwargs
    model = ctx.model
    model_config = kw.get("model_config")
    inference_config = kw.get("inference_config")

    # -- Model and parameterization validation (skip if user-supplied config) --
    if model_config is None:
        model_lower = model.lower()

        # Normalize deprecated aliases before validation.
        if model_lower in _DEPRECATED_MODEL_ALIASES:
            canonical = _DEPRECATED_MODEL_ALIASES[model_lower]
            _log.warning(
                f"Model name '{model_lower}' is deprecated; "
                f"use '{canonical}' instead."
            )
            model = canonical
            model_lower = canonical

        if model_lower not in VALID_MODELS:
            raise ValueError(
                f"Unknown model: '{model}'. "
                f"Valid models are: {', '.join(sorted(VALID_MODELS))}"
            )

        parameterization = kw.get("parameterization", "canonical")
        param_lower = parameterization.lower()
        if param_lower not in VALID_PARAMETERIZATIONS:
            raise ValueError(
                f"Unknown parameterization: '{parameterization}'. "
                f"Valid parameterizations are: "
                f"{', '.join(sorted(VALID_PARAMETERIZATIONS))}"
            )

    # -- Inference method validation ------------------------------------------
    if inference_config is None:
        inference_method = kw.get("inference_method", "svi")
        method_lower = inference_method.lower()
        if method_lower not in VALID_INFERENCE_METHODS:
            raise ValueError(
                f"Unknown inference_method: '{inference_method}'. "
                f"Valid methods are: "
                f"{', '.join(sorted(VALID_INFERENCE_METHODS))}"
            )

    # -- svi_init constraint: only allowed with MCMC --------------------------
    svi_init = kw.get("svi_init")
    if svi_init is not None:
        inference_method = kw.get("inference_method", "svi")
        _effective_method = (
            inference_method.lower()
            if inference_config is None
            else inference_config.method.value
        )
        if _effective_method != "mcmc":
            raise ValueError(
                f"svi_init is only supported with inference_method='mcmc', "
                f"got '{_effective_method}'."
            )

    ctx.model = model

"""
Cross-parameterization conversion for SVI-to-MCMC chain initialization.

This module provides a utility to compute MCMC initialization values from
SVI MAP estimates, bridging parameterization and VCP capture-parameter
differences so that any SVI result can initialize any MCMC run on the same
base model.

NumPyro's ``init_to_value`` receives constrained-space values and maps
them to unconstrained space internally, so no manual bijector handling is
needed.  Site names are also identical regardless of the ``unconstrained``
flag in the new model system (``SigmoidNormalSpec`` and ``BetaSpec`` both
use site name ``"p"`` in constrained space ``(0, 1)``).
"""

import warnings
from typing import Any, Dict

import jax.numpy as jnp

from ..models.config.base import ModelConfig
from ..models.config.enums import Parameterization

# Small epsilon for clamping derived init values away from distribution
# support boundaries.  NumPyro rejects boundary values during
# initialization because the log-probability is -inf there.
_EPS = 1e-6

# Parameterization enum values that correspond to "mean_odds" family
_MEAN_ODDS_PARAMETERIZATIONS = {
    Parameterization.MEAN_ODDS,
    Parameterization.ODDS_RATIO,
}

# Parameterization enum values that correspond to "mean_prob" family
_MEAN_PROB_PARAMETERIZATIONS = {
    Parameterization.MEAN_PROB,
    Parameterization.LINKED,
}

# Known parameter names and their distribution support types.
# "unit" → Beta(0, 1);  "positive" → (0, ∞)
_SUPPORT: Dict[str, str] = {
    "p": "unit",
    "p_capture": "unit",
    "p_hat": "unit",
    "phi": "positive",
    "phi_capture": "positive",
    "mu": "positive",
    "r": "positive",
}

# ------------------------------------------------------------------------------


def clamp_init_values(
    init: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Clamp init values away from distribution support boundaries.

    SVI MAP estimates (stored in float32) can land exactly on support
    boundaries — e.g. ``phi_capture = 0.0`` or ``p = 1.0`` — where the
    log-probability is ``-inf``.  This makes ``init_to_value`` reject
    the initialization.

    Parameters
    ----------
    init : Dict[str, jnp.ndarray]
        Init values keyed by parameter name.

    Returns
    -------
    Dict[str, jnp.ndarray]
        A shallow copy with boundary values nudged into the interior.
    """
    out = dict(init)
    for name, arr in out.items():
        support = _SUPPORT.get(name)
        if support == "unit":
            out[name] = jnp.clip(arr, _EPS, 1.0 - _EPS)
        elif support == "positive":
            out[name] = jnp.clip(arr, _EPS, None)
    return out


# ------------------------------------------------------------------------------


def compute_init_values(
    svi_map: Dict[str, jnp.ndarray],
    target_config: ModelConfig,
) -> Dict[str, jnp.ndarray]:
    """Compute MCMC init values from SVI MAP estimates.

    Ensures the returned dict contains constrained-space values for all
    sampled sites of the target model's parameterization.  Missing
    parameters are derived from the canonical pair ``(p, r)`` which is
    always present when ``get_map(canonical=True)`` is used.

    Parameters
    ----------
    svi_map : Dict[str, jnp.ndarray]
        MAP estimates from SVI, typically obtained via
        ``svi_results.get_map(use_mean=True, canonical=True)``.
        Must contain at least ``"p"`` and ``"r"``.
    target_config : ModelConfig
        Model configuration for the target MCMC run.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Init values keyed by site name, all in constrained space.
        Includes the original SVI MAP entries plus any derived
        parameters needed by the target parameterization.

    Raises
    ------
    ValueError
        If canonical parameters ``p`` and ``r`` are missing from
        *svi_map* and cannot be derived.

    Notes
    -----
    ``init_to_value`` only initializes ``numpyro.sample`` sites.  Extra
    keys in the returned dict (e.g. deterministic sites ``r`` when the
    target is ``mean_prob``) are harmlessly ignored by NumPyro.

    Hierarchical hyperparameters (``logit_p_loc``, ``log_phi_scale``,
    etc.) live in different spaces across parameterizations and cannot
    be reliably converted.  They are omitted and will fall back to
    ``init_to_uniform`` inside NumPyro.
    """
    init = dict(svi_map)
    target_param = target_config.parameterization

    # ------------------------------------------------------------------
    # Ensure canonical (p, r) are present
    # ------------------------------------------------------------------
    if "p" not in init and "phi" in init:
        init["p"] = jnp.clip(1.0 / (1.0 + init["phi"]), _EPS, 1.0 - _EPS)
    if "r" not in init:
        if "mu" in init and "p" in init:
            p = init["p"]
            init["r"] = jnp.clip(init["mu"] * (1.0 - p) / p, _EPS, None)
        elif "mu" in init and "phi" in init:
            init["r"] = jnp.clip(init["mu"] * init["phi"], _EPS, None)

    if "p" not in init or "r" not in init:
        raise ValueError(
            "SVI MAP must contain canonical parameters 'p' and 'r' "
            "(or enough information to derive them).  "
            "Use svi_results.get_map(canonical=True)."
        )

    # Clamp canonical values away from support boundaries.  SVI MAP
    # estimates can land exactly on boundaries (e.g. p = 1.0) which
    # makes derived quantities like mu = r*p/(1-p) blow up.
    init["p"] = jnp.clip(init["p"], _EPS, 1.0 - _EPS)
    init["r"] = jnp.clip(init["r"], _EPS, None)

    p = init["p"]
    r = init["r"]

    # ------------------------------------------------------------------
    # Derive missing core parameters for the target parameterization.
    # All derived values are clamped away from distribution support
    # boundaries so that NumPyro's init_to_value can compute finite
    # log-probabilities.
    # ------------------------------------------------------------------
    if (
        target_param in _MEAN_ODDS_PARAMETERIZATIONS
        or target_param in _MEAN_PROB_PARAMETERIZATIONS
    ):
        if "mu" not in init:
            init["mu"] = jnp.clip(r * p / (1.0 - p), _EPS, None)

    if target_param in _MEAN_ODDS_PARAMETERIZATIONS:
        if "phi" not in init:
            init["phi"] = jnp.clip((1.0 - p) / p, _EPS, None)

    # ------------------------------------------------------------------
    # Handle VCP capture-parameter name differences
    # ------------------------------------------------------------------
    _convert_capture_params(init, target_config)

    return init


# ------------------------------------------------------------------------------


def _convert_capture_params(
    init: Dict[str, Any],
    target_config: ModelConfig,
) -> None:
    """Convert VCP capture parameters between p_capture and phi_capture.

    Mutates *init* in-place.

    Parameters
    ----------
    init : Dict[str, Any]
        Mutable dict of init values.
    target_config : ModelConfig
        Target MCMC model configuration.
    """
    if not target_config.uses_variable_capture:
        return

    target_param = target_config.parameterization
    target_uses_phi_capture = target_param in _MEAN_ODDS_PARAMETERIZATIONS

    if target_uses_phi_capture:
        # Target expects phi_capture (BetaPrime support: (0, inf))
        if "phi_capture" not in init and "p_capture" in init:
            p_cap = init["p_capture"]
            init["phi_capture"] = jnp.clip((1.0 - p_cap) / p_cap, _EPS, None)
    else:
        # Target expects p_capture (Beta support: (0, 1))
        if "p_capture" not in init and "phi_capture" in init:
            phi_cap = init["phi_capture"]
            init["p_capture"] = jnp.clip(
                1.0 / (1.0 + phi_cap), _EPS, 1.0 - _EPS
            )

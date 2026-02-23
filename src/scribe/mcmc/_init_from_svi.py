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


# Parameterization enum values that correspond to "mean_odds" family
_MEAN_ODDS_PARAMETERIZATIONS = {
    Parameterization.MEAN_ODDS,
    Parameterization.ODDS_RATIO,
    Parameterization.HIERARCHICAL_MEAN_ODDS,
}

# Parameterization enum values that correspond to "mean_prob" family
_MEAN_PROB_PARAMETERIZATIONS = {
    Parameterization.MEAN_PROB,
    Parameterization.LINKED,
    Parameterization.HIERARCHICAL_MEAN_PROB,
}

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
        init["p"] = 1.0 / (1.0 + init["phi"])
    if "r" not in init:
        if "mu" in init and "p" in init:
            p = init["p"]
            init["r"] = init["mu"] * (1.0 - p) / p
        elif "mu" in init and "phi" in init:
            init["r"] = init["mu"] * init["phi"]

    if "p" not in init or "r" not in init:
        raise ValueError(
            "SVI MAP must contain canonical parameters 'p' and 'r' "
            "(or enough information to derive them).  "
            "Use svi_results.get_map(canonical=True)."
        )

    p = init["p"]
    r = init["r"]

    # ------------------------------------------------------------------
    # Derive missing core parameters for the target parameterization
    # ------------------------------------------------------------------
    if (
        target_param in _MEAN_ODDS_PARAMETERIZATIONS
        or target_param in _MEAN_PROB_PARAMETERIZATIONS
    ):
        # Both families need mu
        if "mu" not in init:
            init["mu"] = r * p / (1.0 - p)

    if target_param in _MEAN_ODDS_PARAMETERIZATIONS:
        if "phi" not in init:
            init["phi"] = (1.0 - p) / p

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
        # Target expects phi_capture
        if "phi_capture" not in init and "p_capture" in init:
            p_cap = init["p_capture"]
            init["phi_capture"] = (1.0 - p_cap) / p_cap
    else:
        # Target expects p_capture
        if "p_capture" not in init and "phi_capture" in init:
            phi_cap = init["phi_capture"]
            init["p_capture"] = 1.0 / (1.0 + phi_cap)

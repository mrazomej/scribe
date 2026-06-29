"""Single source of truth for Laplace-side derived-quantity formulas.

Background
----------
Several ``ScribeLaplaceResults`` fields are **derived** from positive
parents at result-construction time:

* TSLN-Rate: ``(alpha, beta, r_hat)`` derive from ``(gene_mean, burst_size,
  k_off)`` via the mean-preserving :func:`_twostate_reparam` mapping.
* TSLN-Logit: ``(alpha, beta)`` derive from ``(kappa, eta_anchor)`` via
  ``alpha = kappa * sigmoid(eta_anchor)``,
  ``beta = kappa * (1 - sigmoid(eta_anchor))``.
* LNM: ``p`` derives from ``(mu_T, r_T)`` via the NB success-probability
  convention ``p = r_T / (r_T + mu_T)``.

When :meth:`ScribeLaplaceResults.with_jacobian_map` corrects the
*positive parents* via the Jacobian-corrected MAP, the derived
quantities must be recomputed from the corrected values — otherwise the
result dict becomes internally inconsistent
(``alpha ≠ formula(gene_mean, burst_size, k_off)`` etc.).

This module centralises those derivation formulas as pure functions so
both ``pack_result`` (existing path) and ``with_jacobian_map`` (new
path) call a single implementation. The TSLN helper imports
``_twostate_reparam`` directly from
:mod:`scribe.models.components.likelihoods.two_state` so that any
future change to the floor / clip constants
(``_BURST_MIN``, ``_ALPHA_MIN``, ``_ALPHA_MAX``, ``_K_OFF_MIN``,
``_K_OFF_MAX``) propagates automatically.

Verified against:
* :func:`_twostate_reparam` at
  ``src/scribe/models/components/likelihoods/two_state.py:74-124``.
* LNM ``p`` convention at ``tests/laplace/test_lnm_laplace.py:417-421``
  (``p = r_T / (r_T + mu_T)``).
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from scribe.models.components.likelihoods.two_state import _twostate_reparam


def twostate_rate_derived_from_parents(
    gene_mean: jnp.ndarray,
    burst_size: jnp.ndarray,
    k_off: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """TSLN-Rate: derive ``(alpha, beta, r_hat, eff_burst_size)`` from
    ``(gene_mean, burst_size, k_off)``.

    Re-exports :func:`_twostate_reparam` with the mapping
    ``mu = gene_mean`` and returns a dict with named keys for easy
    consumption by ``with_jacobian_map`` and other callers.

    Parameters
    ----------
    gene_mean, burst_size, k_off : jnp.ndarray
        Positive parents, all shape ``(G,)``. Same coordinate as the
        ``self.gene_mean`` / ``self.burst_size`` / ``self.k_off`` fields
        on ``ScribeLaplaceResults``.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{"alpha", "beta", "r_hat", "eff_burst_size"}``.

        * ``alpha = clip(gene_mean / burst_size, _ALPHA_MIN, _ALPHA_MAX)``
        * ``beta = clip(k_off, _K_OFF_MIN, _K_OFF_MAX)``
        * ``r_hat = gene_mean * (alpha + beta) / alpha`` (= ``rate`` output
          of :func:`_twostate_reparam`, which after the mean-preserving
          floor equals the NB shape proxy ``r_hat`` for TSLN-Rate.)
        * ``eff_burst_size = gene_mean / alpha`` (equals input
          ``burst_size`` when neither clamp activates).
    """
    alpha, beta, rate, eff_burst_size = _twostate_reparam(
        mu=gene_mean,
        burst_size=burst_size,
        k_off=k_off,
    )
    # NOTE on naming: ``_twostate_reparam`` returns ``rate`` which is the
    # NB shape proxy ``r_hat`` for TSLN-Rate. We rename it here to match
    # the ``r_hat`` field on ScribeLaplaceResults.
    return {
        "alpha": alpha,
        "beta": beta,
        "r_hat": rate,
        "eff_burst_size": eff_burst_size,
    }


def twostate_logit_derived_from_parents(
    kappa: jnp.ndarray,
    eta_anchor: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """TSLN-Logit: derive ``(alpha, beta, gene_mean)`` at ``z = 0``.

    Formulas (matching
    ``src/scribe/laplace/_obs_twostate_ln_logit.py:989-990``):

    * ``phi_anchor = sigmoid(eta_anchor)``
    * ``alpha = kappa * phi_anchor``
    * ``beta = kappa * (1 - phi_anchor)``
    * ``gene_mean = rate * alpha / (alpha + beta)`` *if rate is also*
      *known* (here we return only the (alpha, beta) pair since
      ``rate`` is sampled directly, not derived).

    Parameters
    ----------
    kappa : jnp.ndarray
        Positive shape sum ``alpha + beta``, shape ``(G,)``.
    eta_anchor : jnp.ndarray
        Per-gene activation log-odds, shape ``(G,)``.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{"alpha", "beta", "phi_anchor"}``.
    """
    phi_anchor = jax.nn.sigmoid(eta_anchor)
    alpha = kappa * phi_anchor
    beta = kappa * (1.0 - phi_anchor)
    return {
        "alpha": alpha,
        "beta": beta,
        "phi_anchor": phi_anchor,
    }


def lnm_p_from_parents(
    mu_T: jnp.ndarray,
    r_T: jnp.ndarray,
) -> jnp.ndarray:
    """LNM: derive NB success probability from positive parents.

    Convention: ``p = r_T / (r_T + mu_T)``. This is the NB success-prob
    convention where the shape parameter is in the numerator. Verified
    against ``tests/laplace/test_lnm_laplace.py:417-421``.

    Parameters
    ----------
    mu_T, r_T : jnp.ndarray
        Positive parents (mean, shape), same coordinate as the
        ``self.mu_T`` / ``self.r_T`` fields on ``ScribeLaplaceResults``.

    Returns
    -------
    jnp.ndarray
        Per-gene NB success probability ``p`` in ``(0, 1)``.
    """
    return r_T / (r_T + mu_T)


__all__ = [
    "twostate_rate_derived_from_parents",
    "twostate_logit_derived_from_parents",
    "lnm_p_from_parents",
]

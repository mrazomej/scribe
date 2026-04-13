"""Beta Negative Binomial likelihood classes for excess overdispersion.

This module provides BNB counterparts to every NB-based likelihood.
Each BNB class inherits from the corresponding NB class and overrides
only the distribution construction hook (``_make_count_dist`` and, for
VCP variants, ``_make_count_dist_logits``), keeping all plate logic,
batching, VAE paths, annotation priors, and dataset indexing unchanged.

The BNB arises by placing a Beta prior on the NB success probability:

    q_i ~ Beta(alpha, kappa)
    X_i | q_i ~ NB(r, probs=q_i)

which marginally gives X ~ BetaNegativeBinomial(alpha, kappa, r).
A mean-preserving parameterisation ensures that the BNB mean equals
the NB mean (r * p / (1 - p) under SCRIBE's canonical mapping), so
compositional normalisation and differential expression
remain valid.

Classes
-------
BetaNegativeBinomialLikelihood
    BNB counterpart of NegativeBinomialLikelihood.
ZeroInflatedBNBLikelihood
    BNB counterpart of ZeroInflatedNBLikelihood.
BNBWithVCPLikelihood
    BNB counterpart of NBWithVCPLikelihood.
ZIBNBWithVCPLikelihood
    BNB counterpart of ZINBWithVCPLikelihood.

Functions
---------
build_bnb_dist
    Core reparameterisation: omega_g -> (kappa, alpha) -> BNB.
build_count_dist
    Convenience dispatcher returning NB or BNB for use in functional
    APIs (``log_likelihood.py``, ``sampling.py``).
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

# BetaNegativeBinomial was added to numpyro >= 0.20.  Fall back to the
# local implementation in scribe.stats.distributions for older versions.
try:
    from numpyro.distributions.conjugate import BetaNegativeBinomial
except ImportError:
    from scribe.stats.distributions import BetaNegativeBinomial

from .negative_binomial import NegativeBinomialLikelihood
from .zero_inflated import ZeroInflatedNBLikelihood
from .vcp import NBWithVCPLikelihood, ZINBWithVCPLikelihood

# Minimum epsilon for clamping to prevent log(0) / division-by-zero.
_P_EPS = 1e-6

# kappa must be > 2 for the BNB variance to exist.
_KAPPA_MIN = 2.0 + _P_EPS


# =========================================================================
# Utility functions
# =========================================================================


def build_bnb_dist(
    r: jnp.ndarray,
    p: jnp.ndarray,
    omega: jnp.ndarray,
) -> BetaNegativeBinomial:
    """Build a BetaNegativeBinomial with mean-preserving reparameterisation.

    Converts the learnable excess-dispersion fraction ``omega_g`` into
    the BNB concentration ``kappa_g`` and the first Beta shape ``alpha``
    such that E[X] = r * p / (1 - p), matching the NB mean under
    SCRIBE's canonical ``(r, p) -> mu`` mapping.

    Parameters
    ----------
    r : jnp.ndarray
        NB dispersion (``total_count``, >0).
    p : jnp.ndarray
        ``NegativeBinomialProbs`` probability parameter, clamped to
        ``(eps, 1-eps)``.
    omega : jnp.ndarray
        Per-gene excess-dispersion fraction (>0).

    Returns
    -------
    BetaNegativeBinomial
        Distribution with ``concentration1=alpha``,
        ``concentration0=kappa``, ``n=r``.
    """
    omega = jnp.clip(omega, _P_EPS, None)
    kappa = 2.0 + (r + 1.0) / omega
    kappa = jnp.clip(kappa, _KAPPA_MIN, None)

    # Mean-preserving: NB mean = r*p/(1-p) = BNB mean = r*alpha/(kappa-1)
    alpha = p * (kappa - 1.0) / (1.0 - p)

    return BetaNegativeBinomial(concentration1=alpha, concentration0=kappa, n=r)


def build_count_dist(
    r: jnp.ndarray,
    p: jnp.ndarray,
    bnb_concentration: Optional[jnp.ndarray] = None,
) -> dist.Distribution:
    """Return an NB or BNB distribution depending on ``bnb_concentration``.

    This is a convenience dispatcher used by the functional APIs in
    ``log_likelihood.py`` and ``sampling.py`` that do not go through
    the OO likelihood classes.

    Parameters
    ----------
    r : jnp.ndarray
        NB dispersion (>0).
    p : jnp.ndarray
        ``NegativeBinomialProbs`` probability parameter, clamped to
        ``(eps, 1-eps)``.
    bnb_concentration : jnp.ndarray, optional
        Per-gene excess-dispersion fraction.  When ``None`` an
        ordinary ``NegativeBinomialProbs`` is returned.

    Returns
    -------
    dist.Distribution
        Either ``NegativeBinomialProbs`` or ``BetaNegativeBinomial``.
    """
    if bnb_concentration is None:
        return dist.NegativeBinomialProbs(r, p)
    return build_bnb_dist(r, p, bnb_concentration)


# =========================================================================
# BNB Likelihood (base NBDM)
# =========================================================================


class BetaNegativeBinomialLikelihood(NegativeBinomialLikelihood):
    """Beta Negative Binomial likelihood for UMI count data.

    Identical to ``NegativeBinomialLikelihood`` except that the base
    count distribution is BNB instead of NB, allowing heavier-than-NB
    tails controlled by the per-gene ``bnb_concentration`` parameter.

    The ``bnb_concentration`` (omega_g) is extracted from
    ``param_values`` at the start of ``_build_dist`` / ``sample`` and
    used inside ``_make_count_dist`` to construct the BNB.
    """

    def _build_dist(
        self, param_values: Dict[str, jnp.ndarray], **kwargs
    ) -> dist.Distribution:
        # Stash omega_g so the overridden _make_count_dist can use it.
        self._bnb_concentration = param_values["bnb_concentration"]
        return super()._build_dist(param_values, **kwargs)

    def _build_annotated_mixture_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        annotation_logits_batch: jnp.ndarray,
        **kwargs,
    ) -> dist.Distribution:
        self._bnb_concentration = param_values["bnb_concentration"]
        return super()._build_annotated_mixture_dist(
            param_values, annotation_logits_batch, **kwargs
        )

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        return build_bnb_dist(r, p, self._bnb_concentration)


# =========================================================================
# Zero-Inflated BNB Likelihood
# =========================================================================


class ZeroInflatedBNBLikelihood(ZeroInflatedNBLikelihood):
    """Zero-Inflated Beta Negative Binomial likelihood.

    Identical to ``ZeroInflatedNBLikelihood`` except the base count
    distribution is BNB.
    """

    def _build_dist(
        self, param_values: Dict[str, jnp.ndarray], **kwargs
    ) -> dist.Distribution:
        self._bnb_concentration = param_values["bnb_concentration"]
        return super()._build_dist(param_values, **kwargs)

    def _build_annotated_mixture_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        annotation_logits_batch: jnp.ndarray,
        **kwargs,
    ) -> dist.Distribution:
        self._bnb_concentration = param_values["bnb_concentration"]
        return super()._build_annotated_mixture_dist(
            param_values, annotation_logits_batch, **kwargs
        )

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        return build_bnb_dist(r, p, self._bnb_concentration)


# =========================================================================
# BNB with Variable Capture Probability
# =========================================================================


class BNBWithVCPLikelihood(NBWithVCPLikelihood):
    """NB-VCP likelihood with BNB base distribution.

    Identical to ``NBWithVCPLikelihood`` except that the base count
    distribution is BNB.  In the mean-odds (logits) path the BNB
    requires explicit probabilities, so ``_make_count_dist_logits``
    converts logits to probs before building the BNB.
    """

    def sample(self, param_values, *args, **kwargs):
        self._bnb_concentration = param_values.get("bnb_concentration")
        return super().sample(param_values, *args, **kwargs)

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        return build_bnb_dist(r, p, self._bnb_concentration)

    def _make_count_dist_logits(
        self, r: jnp.ndarray, logits: jnp.ndarray
    ) -> dist.Distribution:
        # BNB needs explicit probs; convert from logits.
        p_hat = jax.nn.sigmoid(logits)
        p_hat = jnp.clip(p_hat, _P_EPS, 1.0 - _P_EPS)
        return build_bnb_dist(r, p_hat, self._bnb_concentration)


# =========================================================================
# Zero-Inflated BNB with Variable Capture Probability
# =========================================================================


class ZIBNBWithVCPLikelihood(ZINBWithVCPLikelihood):
    """Zero-inflated NB-VCP likelihood with BNB base distribution.

    Identical to ``ZINBWithVCPLikelihood`` except that the base count
    distribution is BNB.
    """

    def sample(self, param_values, *args, **kwargs):
        self._bnb_concentration = param_values.get("bnb_concentration")
        return super().sample(param_values, *args, **kwargs)

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        return build_bnb_dist(r, p, self._bnb_concentration)

    def _make_count_dist_logits(
        self, r: jnp.ndarray, logits: jnp.ndarray
    ) -> dist.Distribution:
        p_hat = jax.nn.sigmoid(logits)
        p_hat = jnp.clip(p_hat, _P_EPS, 1.0 - _P_EPS)
        return build_bnb_dist(r, p_hat, self._bnb_concentration)

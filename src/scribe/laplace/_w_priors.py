"""Modular shrinkage priors on the loadings matrix W.

This module introduces a pluggable strategy interface for shrinkage
priors on the per-factor columns of the loadings matrix ``W`` used by
the PLN and NBLN observation models in the Laplace inference path.

Motivation
----------
At generous ``vae_latent_dim`` (e.g. 32), the gauge-invariant singular
value spectrum of ``W_⟂`` shows a long flat shelf — the model uses all
available latent dimensions to fit noise, producing spurious cross-gene
correlations visible in ``plot_compositional_corner_ppc``. A shrinkage
prior on the per-factor column scales lets users keep ``vae_latent_dim``
generous and have the prior pick the effective rank adaptively.

The four strategies in v1 are all **column-wise** (per-factor scales).
Future row-wise or element-wise families plug in with one new class and
one registry entry.

Architecture
------------
- ``WPriorStrategy``: protocol declaring three methods
  (:meth:`init_aux_params`, :meth:`log_prior`, :meth:`diagnostics`) plus
  the registry-key ``type_name`` and the ``aux_param_names`` tuple.
- Four concrete strategies: :class:`NoneWPrior` (default no-op),
  :class:`GaussianColumnwiseWPrior` (simple ridge),
  :class:`HorseshoeColumnwiseWPrior`, :class:`NEGColumnwiseWPrior`.
- Registry + factory: :func:`build_w_prior_strategy` converts a user-
  facing dict config (``{"type": "horseshoe_columnwise", ...}``) into a
  strategy instance. Unknown types raise ``ValueError``.

Key design decisions (see ``okay-look-at-all-foamy-sphinx.md`` Phase 3)
----------------------------------------------------------------------
- **Softplus-floor reparameterization**: positive scale parameters live
  in unconstrained ``raw_*`` space; the constrained value is
  ``lambda_min + softplus(raw)``. This blocks the
  scale-collapse-to-zero MAP singularity that an unconstrained
  ``exp(log_*)`` parameterization has when ``W → 0`` simultaneously.
- **W_⟂ projection lives at the obs-model boundary**: the obs model
  hands a gauge-cleaned ``W_for_prior`` to the strategy, plus an
  ``n_constraints`` flag telling the strategy how many linear
  constraints reduce the effective Gaussian dimension. The strategy
  itself stays model-agnostic.
- **Centered-column Gaussian density**: the per-column Normal log-prob
  is written manually as
  ``-0.5 * quad - d_eff * log(σ) - 0.5 * d_eff * log(2π)`` with
  ``d_eff = G - n_constraints``. This avoids the off-by-one normalizer
  that ``dist.Normal.log_prob`` would produce on a ``(G-1)``-dim
  subspace.
- **Headline diagnostic is the column norm of W_⟂**, not the aux scale.
  Aux scales can be weakly identified under heavy tails; column norms
  directly drive ``W_⟂ W_⟂^⊤`` and hence the compositional covariance
  that downstream PPCs visualize.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Protocol, Tuple, Type

import jax
import jax.numpy as jnp
import numpyro.distributions as dist


# =====================================================================
# Helpers
# =====================================================================


def _inv_softplus(y: float) -> float:
    """Inverse of ``softplus(x) = log(1 + exp(x))``.

    Used to choose ``raw`` aux-param initializations such that the
    constrained scale is at a desired starting value (default 1.0).
    """
    # softplus^{-1}(y) = log(exp(y) - 1).  For y >> 0 use the stable form
    # log(exp(y) - 1) = y + log(1 - exp(-y)).
    if y > 20.0:
        return float(y + math.log1p(-math.exp(-y)))
    return float(math.log(math.expm1(y)))


def _validate_positive_finite(value: float, name: str) -> None:
    """Raise ``ValueError`` if ``value`` is non-positive or non-finite."""
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0; got {value!r}.")


def _validate_floor_below_init(
    floor: float, init_scale: float, floor_name: str
) -> None:
    """Raise ``ValueError`` if the constrained-scale floor is not strictly
    below the init scale used by ``init_aux_params``.

    The default init logic computes ``raw = inv_softplus(init_scale - floor)``,
    which is only well-defined when ``floor < init_scale``.
    """
    if float(floor) >= float(init_scale):
        raise ValueError(
            f"{floor_name} ({floor!r}) must be strictly less than the "
            f"default init scale ({init_scale!r}); the init "
            "transform inv_softplus(init_scale - floor) is otherwise "
            "undefined. Choose a smaller floor or set init_scale "
            "explicitly via an init_scale kwarg (v1.1)."
        )


# =====================================================================
# Strategy protocol
# =====================================================================


class WPriorStrategy(Protocol):
    """Pluggable shrinkage prior on the loadings matrix W.

    Concrete subclasses declare:

    - ``type_name``: registry key, matches the ``w_prior["type"]``
      config string (e.g. ``"horseshoe_columnwise"``).
    - ``aux_param_names``: names of optimizer-state parameters this
      prior adds (e.g. ``("w_raw_lambda_k", "w_raw_tau")``). These ride
      through optax's M-step alongside ``mu``/``W``/``d_loc``. Names
      are prefixed with ``w_`` to namespace them.
    - Three methods: :meth:`init_aux_params`, :meth:`log_prior`,
      :meth:`diagnostics`.

    Concrete classes validate their hyperparameters in ``__init__``
    (raises ``ValueError`` for non-positive or non-finite values
    before any JAX tracing).

    The strategy is **W-input-agnostic**: the obs model decides what to
    pass. For NBLN/PLN, the obs model projects ``W → W_⟂`` before
    calling ``log_prior`` / ``diagnostics`` and passes
    ``n_constraints=1`` so the strategy uses ``d_eff = G - 1`` in the
    centered-column Gaussian normalizer. For LNM-family (future), the
    obs model would pass raw W since LNM's W is already in ALR
    coordinates.

    Notes on terminology
    --------------------
    The aux scales are optimized in real-valued unconstrained
    (``raw``) space and transformed to the positive constrained domain
    via a softplus floor: ``λ_k = lambda_min + softplus(raw_λ_k)``.
    This is **log-scale constrained MAP**, not the MCMC-literature
    non-centered parameterization (NCP) — the latter would optimize a
    separate ``Z`` with ``W = λ · Z``. We optimize ``W`` directly.
    """

    type_name: str
    aux_param_names: Tuple[str, ...]

    def init_aux_params(
        self, G: int, k_latent: int, rng_key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        """Return initial values for the aux optimizer parameters.

        Aux params are in unconstrained (``raw``) space; strategies
        that constrain a positive scale apply the softplus floor
        internally in ``log_prior`` / ``diagnostics``.
        """
        ...

    def log_prior(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> jnp.ndarray:
        """Scalar log-prior on ``(W_for_prior, aux_params)``.

        Includes the Jacobian terms for any constrained transforms
        applied to the aux params (softplus-floor for positive scales).

        ``n_constraints`` specifies how many linear constraints each
        column of ``W_for_prior`` satisfies, reducing the effective
        Gaussian dimensionality from ``G`` to ``d_eff = G -
        n_constraints``. NBLN/PLN obs models pass ``n_constraints=1``
        (each centered column sums to zero); LNM-family or
        debugging-on-raw-W calls pass ``0``.
        """
        ...

    def diagnostics(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> Dict[str, Any]:
        """Post-fit interpretable diagnostics.

        Headline keys for shrinkage strategies:

        - ``"column_frobenius_compositional"`` — ``||W_⟂[:, k]||``
        - ``"column_norm_effective_rank"`` — primary rank diagnostic
        - ``"effective_rank"`` — alias for the above (terse API)
        - ``"sigma_k"`` — per-column scales (constrained MAP)
        - ``"scale_effective_rank"`` — secondary rank from sigma_k

        Attached to ``result.w_prior_diagnostics`` post-fit. The obs
        model additionally attaches a ``"column_frobenius_raw"``
        side-channel from the raw ``W``.
        """
        ...


# =====================================================================
# Concrete strategies
# =====================================================================


class NoneWPrior:
    """No-op strategy — no shrinkage.

    Used when ``w_prior=None`` (the default). Numerically identical to
    a fit with no prior on W. The obs model still constructs this
    instance so the integration code path is uniform across configs.
    """

    type_name: str = "none"
    aux_param_names: Tuple[str, ...] = ()

    def init_aux_params(
        self, G: int, k_latent: int, rng_key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        return {}

    def log_prior(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> jnp.ndarray:
        return jnp.zeros((), dtype=W_for_prior.dtype)

    def diagnostics(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> Dict[str, Any]:
        # The obs model adds ``column_frobenius_raw`` from the raw W
        # separately; we report the compositional norm here for
        # parity with the other strategies (useful diagnostic even when
        # no shrinkage is applied).
        col_norm = jnp.linalg.norm(W_for_prior, axis=0)
        return {
            "column_frobenius_compositional": col_norm,
        }


class GaussianColumnwiseWPrior:
    """Simple Gaussian (ridge) prior with a single shared scale.

    ``W[:, k] ~ Normal(0, scale)`` independently for each column.
    No auxiliary parameters — the scale is a fixed hyperparameter.

    Use case: simple sanity-check baseline. Tells you whether the
    symptom is "any shrinkage helps" or "specifically sparsity helps"
    (in which case horseshoe / NEG are the right answer).
    """

    type_name: str = "gaussian"
    aux_param_names: Tuple[str, ...] = ()

    def __init__(self, scale: float = 1.0):
        _validate_positive_finite(scale, "GaussianColumnwiseWPrior.scale")
        self.scale = float(scale)

    def init_aux_params(
        self, G: int, k_latent: int, rng_key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        return {}

    def log_prior(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> jnp.ndarray:
        G = W_for_prior.shape[0]
        k = W_for_prior.shape[1]
        d_eff = G - int(n_constraints)
        scale = self.scale
        quad = jnp.sum(W_for_prior ** 2) / (scale ** 2)
        log_norm = -d_eff * k * jnp.log(scale) - 0.5 * d_eff * k * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * quad + log_norm

    def diagnostics(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> Dict[str, Any]:
        col_norm = jnp.linalg.norm(W_for_prior, axis=0)
        if col_norm.size > 0:
            thr = 0.05 * float(jnp.max(col_norm))
            rank = int(jnp.sum(col_norm > thr))
        else:
            rank = 0
        return {
            "column_frobenius_compositional": col_norm,
            "column_norm_effective_rank": rank,
            "effective_rank": rank,
            "scale": jnp.asarray(self.scale),
        }


class HorseshoeColumnwiseWPrior:
    """Column-wise horseshoe shrinkage.

    Hierarchy (``λ_k`` is the *standard deviation*, not the variance):

    ``W[:, k] | λ_k  ~  Normal(0, λ_k)``
    ``λ_k     | τ   ~  HalfCauchy(τ)``
    ``τ              ~  HalfCauchy(tau_scale)``

    The aux scales are parameterized in unconstrained space with a
    softplus floor:

    ``λ_k = lambda_min + softplus(raw_λ_k)``
    ``τ   = tau_min    + softplus(raw_τ)``

    The floor blocks the MAP scale-collapse singularity (without it,
    the joint Normal-on-W + log-Jacobian log-prior is unbounded above
    as ``λ_k → 0``).

    Aimed at the column-wise sparsity pattern produced by an
    over-parameterized latent: keeps strong factors at full scale
    while shrinking spurious ones to near-zero.
    """

    type_name: str = "horseshoe_columnwise"
    aux_param_names: Tuple[str, ...] = ("w_raw_lambda_k", "w_raw_tau")

    def __init__(
        self,
        tau_scale: float = 1.0,
        lambda_min: float = 1e-3,
        tau_min: float = 1e-3,
        init_scale: float = 1.0,
    ):
        _validate_positive_finite(
            tau_scale, "HorseshoeColumnwiseWPrior.tau_scale"
        )
        _validate_positive_finite(
            lambda_min, "HorseshoeColumnwiseWPrior.lambda_min"
        )
        _validate_positive_finite(
            tau_min, "HorseshoeColumnwiseWPrior.tau_min"
        )
        _validate_positive_finite(
            init_scale, "HorseshoeColumnwiseWPrior.init_scale"
        )
        _validate_floor_below_init(
            lambda_min, init_scale, "HorseshoeColumnwiseWPrior.lambda_min"
        )
        _validate_floor_below_init(
            tau_min, init_scale, "HorseshoeColumnwiseWPrior.tau_min"
        )
        self.tau_scale = float(tau_scale)
        self.lambda_min = float(lambda_min)
        self.tau_min = float(tau_min)
        self.init_scale = float(init_scale)

    def init_aux_params(
        self, G: int, k_latent: int, rng_key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        # Initialize so the constrained scales are ``init_scale`` (1.0
        # by default). For softplus floor f, lambda = f + softplus(raw),
        # so raw = inv_softplus(init_scale - f).
        raw_lambda_init = _inv_softplus(self.init_scale - self.lambda_min)
        raw_tau_init = _inv_softplus(self.init_scale - self.tau_min)
        return {
            "w_raw_lambda_k": jnp.full(
                (int(k_latent),), raw_lambda_init, dtype=jnp.float32
            ),
            "w_raw_tau": jnp.asarray(raw_tau_init, dtype=jnp.float32),
        }

    def _constrained(self, aux: Dict[str, jnp.ndarray]):
        """Return constrained (lambda_k, tau)."""
        raw_lambda_k = aux["w_raw_lambda_k"]
        raw_tau = aux["w_raw_tau"]
        lambda_k = self.lambda_min + jax.nn.softplus(raw_lambda_k)
        tau = self.tau_min + jax.nn.softplus(raw_tau)
        return raw_lambda_k, raw_tau, lambda_k, tau

    def log_prior(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> jnp.ndarray:
        raw_lambda_k, raw_tau, lambda_k, tau = self._constrained(aux_params)
        G = W_for_prior.shape[0]
        d_eff = G - int(n_constraints)

        # Subspace-corrected centered-column Gaussian log-density.
        # Scale is std (= lambda_k); variance term uses lambda_k^2.
        quad = jnp.sum((W_for_prior / lambda_k[None, :]) ** 2, axis=0)
        log_norm = -d_eff * jnp.log(lambda_k) - 0.5 * d_eff * jnp.log(
            2 * jnp.pi
        )
        lp_w = jnp.sum(-0.5 * quad + log_norm)

        # Local-scale prior on lambda_k | tau, plus softplus-floor Jacobian.
        log_jac_lambda = jax.nn.log_sigmoid(raw_lambda_k)
        lp_lambda = (
            jnp.sum(dist.HalfCauchy(tau).log_prob(lambda_k))
            + jnp.sum(log_jac_lambda)
        )

        # Global-scale prior on tau, plus softplus-floor Jacobian.
        log_jac_tau = jax.nn.log_sigmoid(raw_tau)
        lp_tau = (
            dist.HalfCauchy(self.tau_scale).log_prob(tau)
            + log_jac_tau
        )

        return lp_w + lp_lambda + lp_tau

    def diagnostics(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> Dict[str, Any]:
        _raw_lambda_k, _raw_tau, lambda_k, tau = self._constrained(
            aux_params
        )
        col_norm = jnp.linalg.norm(W_for_prior, axis=0)
        if col_norm.size > 0:
            thr_col = 0.05 * float(jnp.max(col_norm))
            rank_col = int(jnp.sum(col_norm > thr_col))
        else:
            rank_col = 0
        if lambda_k.size > 0:
            thr_scale = 0.05 * float(jnp.max(lambda_k))
            rank_scale = int(jnp.sum(lambda_k > thr_scale))
        else:
            rank_scale = 0
        return {
            "sigma_k": lambda_k,
            "column_frobenius_compositional": col_norm,
            "column_norm_effective_rank": rank_col,
            "effective_rank": rank_col,
            "scale_effective_rank": rank_scale,
            "tau": tau,
        }


class NEGColumnwiseWPrior:
    """Column-wise Normal-Exponential-Gamma shrinkage.

    Hierarchy (``ψ_k`` is the *variance*, not the standard deviation):

    ``W[:, k] | ψ_k  ~  Normal(0, sqrt(ψ_k))``
    ``ψ_k     | γ   ~  Exponential(rate=γ)``
    ``γ              ~  Gamma(alpha, beta)``

    More aggressive near-zero shrinkage than horseshoe at default
    hyperparameters; useful when horseshoe is insufficient to kill
    unused factors. Aux scales use the same softplus-floor
    reparameterization as horseshoe.
    """

    type_name: str = "neg_columnwise"
    aux_param_names: Tuple[str, ...] = ("w_raw_psi_k", "w_raw_gamma")

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        psi_min: float = 1e-6,
        gamma_min: float = 1e-6,
        init_scale: float = 1.0,
    ):
        _validate_positive_finite(alpha, "NEGColumnwiseWPrior.alpha")
        _validate_positive_finite(beta, "NEGColumnwiseWPrior.beta")
        _validate_positive_finite(psi_min, "NEGColumnwiseWPrior.psi_min")
        _validate_positive_finite(
            gamma_min, "NEGColumnwiseWPrior.gamma_min"
        )
        _validate_positive_finite(
            init_scale, "NEGColumnwiseWPrior.init_scale"
        )
        _validate_floor_below_init(
            psi_min, init_scale, "NEGColumnwiseWPrior.psi_min"
        )
        _validate_floor_below_init(
            gamma_min, init_scale, "NEGColumnwiseWPrior.gamma_min"
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.psi_min = float(psi_min)
        self.gamma_min = float(gamma_min)
        self.init_scale = float(init_scale)

    def init_aux_params(
        self, G: int, k_latent: int, rng_key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        raw_psi_init = _inv_softplus(self.init_scale - self.psi_min)
        raw_gamma_init = _inv_softplus(self.init_scale - self.gamma_min)
        return {
            "w_raw_psi_k": jnp.full(
                (int(k_latent),), raw_psi_init, dtype=jnp.float32
            ),
            "w_raw_gamma": jnp.asarray(raw_gamma_init, dtype=jnp.float32),
        }

    def _constrained(self, aux: Dict[str, jnp.ndarray]):
        raw_psi_k = aux["w_raw_psi_k"]
        raw_gamma = aux["w_raw_gamma"]
        psi_k = self.psi_min + jax.nn.softplus(raw_psi_k)
        gamma = self.gamma_min + jax.nn.softplus(raw_gamma)
        return raw_psi_k, raw_gamma, psi_k, gamma

    def log_prior(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> jnp.ndarray:
        raw_psi_k, raw_gamma, psi_k, gamma = self._constrained(aux_params)
        G = W_for_prior.shape[0]
        d_eff = G - int(n_constraints)

        # Subspace-corrected centered-column Gaussian log-density.
        # Variance = psi_k; std = sqrt(psi_k).
        quad = jnp.sum(W_for_prior ** 2, axis=0) / psi_k
        log_norm = -0.5 * d_eff * jnp.log(psi_k) - 0.5 * d_eff * jnp.log(
            2 * jnp.pi
        )
        lp_w = jnp.sum(-0.5 * quad + log_norm)

        # Variance | gamma ~ Exponential(rate=gamma); softplus-floor Jacobian.
        log_jac_psi = jax.nn.log_sigmoid(raw_psi_k)
        lp_psi = (
            jnp.sum(dist.Exponential(gamma).log_prob(psi_k))
            + jnp.sum(log_jac_psi)
        )

        # gamma ~ Gamma(alpha, beta); softplus-floor Jacobian.
        log_jac_gamma = jax.nn.log_sigmoid(raw_gamma)
        lp_gamma = (
            dist.Gamma(self.alpha, self.beta).log_prob(gamma)
            + log_jac_gamma
        )

        return lp_w + lp_psi + lp_gamma

    def diagnostics(
        self,
        W_for_prior: jnp.ndarray,
        aux_params: Dict[str, jnp.ndarray],
        *,
        n_constraints: int = 0,
    ) -> Dict[str, Any]:
        _raw_psi_k, _raw_gamma, psi_k, gamma = self._constrained(
            aux_params
        )
        sigma_k = jnp.sqrt(psi_k)  # std equivalent for comparability
        col_norm = jnp.linalg.norm(W_for_prior, axis=0)
        if col_norm.size > 0:
            thr_col = 0.05 * float(jnp.max(col_norm))
            rank_col = int(jnp.sum(col_norm > thr_col))
        else:
            rank_col = 0
        if sigma_k.size > 0:
            thr_scale = 0.05 * float(jnp.max(sigma_k))
            rank_scale = int(jnp.sum(sigma_k > thr_scale))
        else:
            rank_scale = 0
        return {
            "sigma_k": sigma_k,
            "psi_k": psi_k,
            "column_frobenius_compositional": col_norm,
            "column_norm_effective_rank": rank_col,
            "effective_rank": rank_col,
            "scale_effective_rank": rank_scale,
            "gamma": gamma,
        }


# =====================================================================
# Registry + factory
# =====================================================================


_W_PRIOR_REGISTRY: Dict[str, Type[WPriorStrategy]] = {
    "none": NoneWPrior,
    "gaussian": GaussianColumnwiseWPrior,
    "horseshoe_columnwise": HorseshoeColumnwiseWPrior,
    "neg_columnwise": NEGColumnwiseWPrior,
}


def build_w_prior_strategy(
    config: Any,
) -> WPriorStrategy:
    """Build a W-prior strategy from a user-facing dict config.

    ``None`` → :class:`NoneWPrior` (no shrinkage).
    Dict with ``"type"`` key → look up in the registry, instantiate
    with the remaining kwargs.

    Parameters
    ----------
    config : ``None`` or dict
        The user-facing W-prior configuration. ``None`` or
        ``{"type": "none"}`` both produce :class:`NoneWPrior`.
        ``{"type": "horseshoe_columnwise", "tau_scale": 1.0}`` produces
        a configured :class:`HorseshoeColumnwiseWPrior`.

    Returns
    -------
    WPriorStrategy
        A configured strategy instance.

    Raises
    ------
    ValueError
        If ``config["type"]`` is missing or unknown, or if any
        strategy-specific hyperparameter validation fails.
    """
    if config is None:
        return NoneWPrior()
    if not isinstance(config, dict):
        raise ValueError(
            f"w_prior must be None or a dict; got {type(config).__name__}."
        )
    cfg = dict(config)
    type_name = cfg.pop("type", None)
    if type_name is None:
        raise ValueError(
            "w_prior dict must include a 'type' key. Got keys: "
            f"{list(cfg.keys())}"
        )
    if type_name not in _W_PRIOR_REGISTRY:
        raise ValueError(
            f"Unknown w_prior type {type_name!r}. Registered: "
            f"{sorted(_W_PRIOR_REGISTRY.keys())}."
        )
    return _W_PRIOR_REGISTRY[type_name](**cfg)


__all__ = [
    "WPriorStrategy",
    "NoneWPrior",
    "GaussianColumnwiseWPrior",
    "HorseshoeColumnwiseWPrior",
    "NEGColumnwiseWPrior",
    "build_w_prior_strategy",
]

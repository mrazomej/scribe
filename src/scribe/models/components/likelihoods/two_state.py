"""Two-state promoter (Poisson-Beta compound) likelihood for count data.

Implements the Poisson-Beta compound likelihood

    p_gc ~ Beta(α_g, β_g)
    u_gc | p_gc ~ Poisson(r̂_g · p_gc · ν_c)

with the sampled (µ_g, b_g, k_off_g) parameterization:

    α_g = k_on  = µ_g / b_g
    β_g = k_off = k_off_g
    r̂_g        = µ_g + b_g · k_off_g    (mean-preserving)

The natural mean-preserving identity is

    ⟨count⟩ = r̂ · α / (α + β) = µ.

Supports both single-component and **mixture** models (K ≥ 2).
In the mixture path, per-gene parameters (mu, burst_size, k_off, …)
carry a component axis, and the observation distribution becomes a
``MixtureGeneral`` over K ``PoissonBetaCompound`` component
distributions, weighted by ``mixing_weights``.

Multi-dataset hierarchical models (``dataset_indices`` provided and
``model_config.n_datasets`` set) are supported: per-dataset gene
parameters carry a leading dataset axis ``(D, G)`` (or ``(K, D, G)``
for mixtures), and inside the cell plate each cell is mapped to its
dataset's parameters via :func:`index_dataset_params`.  The
gene/dataset-level deterministic sites (k_on, k_off, r̂, …) are emitted
once on the full ``(D, G)`` arrays *outside* the cell plate so they
keep gene/dataset rank rather than per-subsample rank.

Features NOT supported:

- VAE inference (``vae_cell_fn``)
- annotation priors (``annotation_prior_logits``)
- BNB overdispersion (``OverdispersionType.BNB``)
- the ``phi_capture`` capture-parameter variant on VCP

Each of those raises ``NotImplementedError`` at sample time. The
configuration layer (build-time validation) catches most of them
earlier; the runtime guards are defence-in-depth.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import (
    Likelihood,
    build_mixture_general,
    index_dataset_params,
    sample_capture_param,
)

from ....core.axis_layout import (
    AxisLayout,
    build_param_layouts,
    broadcast_param_to_layout,
    subset_layouts,
    DATASETS,
)

if TYPE_CHECKING:
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig


def _drop_dataset_axis(
    param_layouts: Optional[Dict[str, "AxisLayout"]],
) -> Optional[Dict[str, "AxisLayout"]]:
    """Return a copy of *param_layouts* with the ``"datasets"`` axis removed.

    After :func:`index_dataset_params` collapses the dataset dimension into
    the leading cell (batch) axis, layouts that carried a ``"datasets"`` axis
    must be updated so that downstream broadcasting treats the new leading
    dimension as a batch dim rather than a semantic axis.  This mirrors the
    identically-named helper defined in the sibling likelihood modules
    (``negative_binomial.py``, ``zero_inflated.py``, ``vcp.py``).

    Parameters
    ----------
    param_layouts : dict or None
        Original layouts built from ``param_specs``.

    Returns
    -------
    dict or None
        Updated layouts with ``"datasets"`` removed where present.
    """
    if param_layouts is None:
        return None
    return subset_layouts(param_layouts, DATASETS)


# ==============================================================================
# Numerical floors for the (µ, b, k_off) → (α, β, rate) reparameterization
# ==============================================================================
#
# Floors keep α and β away from the Jacobi-fragile regime (values
# below ~0.05 make the Beta highly U-shaped and Golub-Welsch
# recurrence coefficients lose precision). _ALPHA_MIN is enforced
# DIRECTLY on the derived α so that low-expression genes (where µ
# itself is small) cannot push α below the floor through the
# µ/burst_size division.

_BURST_MIN = 1e-4
_ALPHA_MIN = 0.05
_ALPHA_MAX = 1.0e3
_K_OFF_MIN = 0.05
_K_OFF_MAX = 1.0e3


def _twostate_reparam(
    mu: jnp.ndarray,
    burst_size: jnp.ndarray,
    k_off: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(µ, b, k_off) → (α, β, rate, effective_burst_size).

    Two-stage map:

    1. Natural map (mean-preserving by construction):

           α_nat = µ / b
           β_nat = k_off
           rate_nat = µ + b · k_off
           E[count] = rate_nat · α_nat / (α_nat + β_nat) = µ.

    2. Quadrature-safety floors AND ceilings, with mean-preserving
       rescaling. Below the lower floor (``_ALPHA_MIN`` /
       ``_K_OFF_MIN``) the Beta is so U-shaped that Golub-Welsch
       recurrence coefficients lose precision; above the upper cap
       (``_ALPHA_MAX`` / ``_K_OFF_MAX``) the Jacobi recurrence
       coefficient ``b² − a²`` suffers float32 catastrophic
       cancellation for very large b. Both clamps preserve the mean
       via

           rate = µ · (α + β) / α,

       which gives ``rate · α / (α + β) = µ`` identically.

       The upper cap matters most for highly-expressed genes that
       give ``α_nat = µ/burst_size`` in the thousands at init (e.g.
       a "pooled-other" gene-coverage column with µ ≈ 10⁴ and a
       default burst_size ≈ 0.7 gives α_nat ≈ 14000, well into the
       cancellation regime in float32).

    Returns
    -------
    alpha, beta, rate : jnp.ndarray
        Floored-and-capped, mean-preserving (α, β, rate) at gene rank.
    eff_burst_size : jnp.ndarray
        ``µ / α`` — the burst size implied by the (floored, capped) α.
        Equals the input ``burst_size`` when neither clamp activated;
        otherwise reflects the effective shape under the safety clamp.
    """
    burst_size = jnp.maximum(burst_size, _BURST_MIN)
    alpha_nat = mu / burst_size
    alpha = jnp.clip(alpha_nat, min=_ALPHA_MIN, max=_ALPHA_MAX)
    beta = jnp.clip(k_off, min=_K_OFF_MIN, max=_K_OFF_MAX)
    rate = mu * (alpha + beta) / alpha
    eff_burst_size = mu / alpha
    return alpha, beta, rate, eff_burst_size


def _twostate_ratio_reparam(
    mu: jnp.ndarray,
    burst_size: jnp.ndarray,
    switching_ratio: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(µ, b, s) → (α, β, rate, effective_burst_size).

    Relative-switching parameterization (analogous to NBDM's
    ``mean_prob`` / ``mean_odds`` aligning with the data's mean axis).
    Sampled parameters are ``mu`` (gene mean), ``burst_size``, and
    ``switching_ratio = k_off / k_on``.

    Natural forward map (mean-preserving by construction):

        k_on  = µ / b
        k_off = s · k_on
        α_nat = k_on
        β_nat = k_off
        rate_nat = µ · (1 + s)
        E[count] = rate_nat · α_nat / (α_nat + β_nat)
                 = µ(1+s) · k_on / (k_on + s·k_on)
                 = µ                          identically.

    Quadrature-safety clamps are applied identically to
    :func:`_twostate_reparam`: when ``α`` or ``β`` fall outside
    ``[_ALPHA_MIN, _ALPHA_MAX]`` or ``[_K_OFF_MIN, _K_OFF_MAX]`` the
    mean-preserving rescaling ``rate = µ · (α + β) / α`` recovers the
    target mean exactly. The natural ``rate = µ(1+s)`` matches this
    formula when no clamp activates (then ``α = k_on``, ``β = k_off``,
    and ``α + β = k_on(1+s)``, giving ``rate = µ(1+s)``).
    """
    burst_size = jnp.maximum(burst_size, _BURST_MIN)
    k_on = mu / burst_size
    k_off = switching_ratio * k_on
    alpha = jnp.clip(k_on, min=_ALPHA_MIN, max=_ALPHA_MAX)
    beta = jnp.clip(k_off, min=_K_OFF_MIN, max=_K_OFF_MAX)
    # Mean-preserving rescaling: when no clamp triggers this equals
    # µ · (1 + s) by construction; when a clamp triggers it preserves
    # the mean exactly (rate · α / (α + β) = µ).
    rate = mu * (alpha + beta) / alpha
    eff_burst_size = mu / alpha
    return alpha, beta, rate, eff_burst_size


def _twostate_moments_reparam(
    mu: jnp.ndarray,
    excess_fano: jnp.ndarray,
    concentration: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(µ, excess_fano, κ) → (α, β, rate, effective_burst_size).

    Mean-Fano parameterization (analog of NBDM's ``mean_odds``).
    Sampled parameters are ``mu`` (gene mean), ``excess_fano`` (the
    excess Fano factor Var/Mean − 1), and ``concentration`` (Beta
    concentration κ = α + β).

    Natural forward map::

        denom = µ + excess_fano · (κ + 1)
        α     = κ · µ / denom            ( = k_on )
        β     = κ · excess_fano · (κ + 1) / denom    ( = k_off )
        rate  = denom

    Moment guarantees (by construction)::

        E[count]                  = µ
        Var[count] / E[count] − 1 = excess_fano

    Quadrature-safety clamps mirror :func:`_twostate_reparam`: when
    ``α`` or ``β`` fall outside ``[_ALPHA_MIN, _ALPHA_MAX]`` or
    ``[_K_OFF_MIN, _K_OFF_MAX]`` the mean-preserving rescaling
    ``rate = µ · (α + β) / α`` recovers the target mean exactly.
    The natural ``rate = denom`` already matches that formula when
    no clamp activates, so the clamp does not change the unclamped
    answer.
    """
    # Materialise the natural alpha, beta, rate.
    denom = mu + excess_fano * (concentration + 1.0)
    # Avoid division by zero in the degenerate excess_fano=0,
    # concentration=0 corner; the clamps below take care of the
    # downstream alpha/beta.
    denom_safe = jnp.maximum(denom, _BURST_MIN * _BURST_MIN)
    alpha_nat = concentration * mu / denom_safe
    beta_nat = concentration * excess_fano * (concentration + 1.0) / denom_safe

    alpha = jnp.clip(alpha_nat, min=_ALPHA_MIN, max=_ALPHA_MAX)
    beta = jnp.clip(beta_nat, min=_K_OFF_MIN, max=_K_OFF_MAX)
    rate = mu * (alpha + beta) / alpha
    eff_burst_size = mu / alpha
    return alpha, beta, rate, eff_burst_size


def _twostate_moment_delta_reparam(
    mu: jnp.ndarray,
    excess_fano: jnp.ndarray,
    inv_concentration: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(mu, excess_fano, delta) → (alpha, beta, rate, eff_burst_size).

    Moment-delta parameterization (mean and Fano preserved by
    construction; shape coordinate bounded in (0, 1)).  Sampled
    parameters are ``mu`` (gene mean), ``excess_fano`` (Fano - 1),
    and ``delta = 1/(kappa + 1)``.

    Natural forward map (mean- and Fano-preserving)::

        denom = mu * delta + excess_fano
        alpha = mu * (1 - delta) / denom               ( = k_on )
        beta  = excess_fano * (1 - delta) / (delta * denom)
                                                       ( = k_off )
        rate  = denom / delta                          ( = mu + e/delta )

    Moment guarantees::

        E[count]                     = mu
        Var[count] / E[count] - 1    = excess_fano

    Quadrature-safety clamps mirror :func:`_twostate_moments_reparam`:
    when alpha or beta land outside ``[_ALPHA_MIN, _ALPHA_MAX]`` or
    ``[_K_OFF_MIN, _K_OFF_MAX]``, the mean-preserving rescaling
    ``rate = mu * (alpha + beta) / alpha`` recovers the target
    mean exactly.
    """
    # Keep delta strictly inside (0, 1) to avoid 1/delta or
    # (1-delta) blowups at the boundary.  The sigmoid output is
    # already in (0, 1) but can produce values arbitrarily close
    # to 0 or 1 in float32; clamp with a small epsilon.
    _DELTA_EPS = 1e-6
    delta = jnp.clip(inv_concentration, _DELTA_EPS, 1.0 - _DELTA_EPS)
    one_minus_delta = 1.0 - delta
    denom = mu * delta + excess_fano
    denom_safe = jnp.maximum(denom, _BURST_MIN * _BURST_MIN)
    alpha_nat = mu * one_minus_delta / denom_safe
    beta_nat = excess_fano * one_minus_delta / (delta * denom_safe)

    alpha = jnp.clip(alpha_nat, min=_ALPHA_MIN, max=_ALPHA_MAX)
    beta = jnp.clip(beta_nat, min=_K_OFF_MIN, max=_K_OFF_MAX)
    rate = mu * (alpha + beta) / alpha
    eff_burst_size = mu / alpha
    return alpha, beta, rate, eff_burst_size


def _twostate_dispatch_reparam(
    param_values: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pick the right reparam based on which third parameter was sampled.

    Returns ``(alpha, beta, rate, eff_burst_size, raw_k_off)``.

    ``raw_k_off`` is the un-clamped ``β`` value — used downstream only
    to compute the ``beta_floor_active`` indicator. For each
    parameterization:

    - ``natural``  : raw_k_off is the sampled value.
    - ``ratio``    : raw_k_off = switching_ratio · k_on  (derived).
    - ``mean_fano``: raw_k_off = κ · excess_fano · (κ + 1) / denom
      (derived).
    """
    mu = param_values["mu"]
    if "inv_concentration" in param_values:
        # Moment-delta parameterization: (mu, excess_fano, delta).
        excess_fano = param_values["excess_fano"]
        delta = param_values["inv_concentration"]
        _DELTA_EPS = 1e-6
        delta_safe = jnp.clip(delta, _DELTA_EPS, 1.0 - _DELTA_EPS)
        denom = mu * delta_safe + excess_fano
        denom_safe = jnp.maximum(denom, _BURST_MIN * _BURST_MIN)
        raw_k_off = (
            excess_fano * (1.0 - delta_safe) / (delta_safe * denom_safe)
        )
        alpha, beta, rate, eff_burst_size = _twostate_moment_delta_reparam(
            mu, excess_fano, delta
        )
    elif "excess_fano" in param_values:
        excess_fano = param_values["excess_fano"]
        concentration = param_values["concentration"]
        denom = mu + excess_fano * (concentration + 1.0)
        denom_safe = jnp.maximum(denom, _BURST_MIN * _BURST_MIN)
        raw_k_off = (
            concentration * excess_fano * (concentration + 1.0) / denom_safe
        )
        alpha, beta, rate, eff_burst_size = _twostate_moments_reparam(
            mu, excess_fano, concentration
        )
    elif "switching_ratio" in param_values:
        burst_size = param_values["burst_size"]
        s = param_values["switching_ratio"]
        burst_safe = jnp.maximum(burst_size, _BURST_MIN)
        raw_k_off = s * (mu / burst_safe)
        alpha, beta, rate, eff_burst_size = _twostate_ratio_reparam(
            mu, burst_size, s
        )
    else:
        burst_size = param_values["burst_size"]
        raw_k_off = param_values["k_off"]
        alpha, beta, rate, eff_burst_size = _twostate_reparam(
            mu, burst_size, raw_k_off
        )
    return alpha, beta, rate, eff_burst_size, raw_k_off


# ==============================================================================
# Phase-1 guard helper
# ==============================================================================


def _reject_phase1_unsupported(
    param_values: Optional[Dict[str, jnp.ndarray]],
    model_config: Optional["ModelConfig"],
    kwargs: Dict[str, object],
) -> None:
    """Runtime guard: reject unsupported features with clear messages.

    Defence-in-depth alongside the build-time validators on
    :class:`~scribe.models.config.ModelConfig`. The build-time path
    catches these early; this runtime path catches direct invocations
    that bypass the builder.

    Mixture models (``mixing_weights`` in ``param_values``) are
    fully supported and NOT rejected here.
    """
    if kwargs.get("vae_cell_fn") is not None:
        raise NotImplementedError(
            "TwoState + VAE inference is not supported in phase 1."
        )
    if kwargs.get("annotation_prior_logits") is not None:
        raise NotImplementedError(
            "TwoState + annotation priors are not supported in phase 1."
        )
    # Multi-dataset IS supported.  Callers pass ``dataset_indices=None`` here
    # once they have confirmed ``use_dataset_indexing`` (i.e. both
    # ``dataset_indices`` and ``model_config.n_datasets`` are set).  A
    # non-None value reaching this guard therefore means dataset assignments
    # were supplied without ``n_datasets`` being configured — a setup error.
    if kwargs.get("dataset_indices") is not None:
        raise ValueError(
            "TwoState received dataset_indices but model_config.n_datasets "
            "is not set; configure n_datasets >= 2 for multi-dataset fits."
        )
    if model_config is not None:
        # Compare against the enum member (not None / not a string).
        from ...config.enums import OverdispersionType

        od = getattr(model_config, "overdispersion", OverdispersionType.NONE)
        if od != OverdispersionType.NONE:
            raise NotImplementedError(
                "TwoState + BNB overdispersion is not supported in phase 1."
            )
    # NB: biology-informed capture prior is guarded inside
    # TwoStateVCPLikelihood.sample because it depends on
    # ``self.biology_informed_spec``, which this free helper cannot see.


# ==============================================================================
# TwoStateLikelihood — no capture
# ==============================================================================


class TwoStateLikelihood(Likelihood):
    """Two-state promoter (non-bursty) likelihood. No per-cell capture.

    Consumes the per-gene sampled parameters ``mu``, ``burst_size``,
    ``k_off`` from ``param_values`` and emits observations from the
    Poisson-Beta compound at gene rank.

    Phase-1 limitations are listed in the module docstring.
    """

    # ------------------------------------------------------------------
    # Reparam + distribution construction
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_deterministics(
        param_values: Dict[str, jnp.ndarray],
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        rate: jnp.ndarray,
        eff_burst_size: jnp.ndarray,
        raw_k_off: jnp.ndarray,
    ) -> None:
        """Expose derived quantities for posterior summaries.

        Called OUTSIDE the cell plate so deterministics emit at gene
        rank (or gene × component rank for mixtures).

        Dispatch logic:
        - ``natural``:      burst_size and k_off are sampled.
        - ``ratio``:        burst_size sampled; k_off derived.
        - ``mean_fano``:    neither sampled; both derived.
        - ``moment_delta``: neither sampled; both derived, and
          ``concentration`` is also derived from delta.
        """
        mu = param_values["mu"]
        is_moment_delta = "inv_concentration" in param_values
        is_mean_fano = "excess_fano" in param_values and not is_moment_delta
        is_ratio = "switching_ratio" in param_values

        if is_ratio or is_mean_fano or is_moment_delta:
            numpyro.deterministic("k_off", beta)
        if is_mean_fano or is_moment_delta:
            numpyro.deterministic("burst_size", eff_burst_size)
        if is_moment_delta:
            _delta = param_values["inv_concentration"]
            _delta_safe = jnp.clip(_delta, 1e-6, 1.0 - 1e-6)
            numpyro.deterministic(
                "concentration", (1.0 - _delta_safe) / _delta_safe
            )
        numpyro.deterministic("k_on", alpha)
        numpyro.deterministic("alpha", alpha)
        numpyro.deterministic("beta", beta)
        numpyro.deterministic("r_hat", rate)
        # Activation log-odds θ_g = log(k_on / k_off) = log(α / β).
        # The downstream TSLN-Logit cascade adapter consumes this
        # preferentially over re-deriving from (mu, burst_size, k_off),
        # because the SVI reparameterization may have activated
        # mean-preserving floors in _twostate_reparam that the raw
        # params don't reflect.  See plan §4.C.1 (Rev 4).
        numpyro.deterministic("eta_act", jnp.log(alpha) - jnp.log(beta))
        numpyro.deterministic("effective_burst_size", eff_burst_size)
        if is_mean_fano or is_moment_delta:
            numpyro.deterministic(
                "alpha_floor_active",
                alpha <= _ALPHA_MIN,
            )
        else:
            burst_size = param_values["burst_size"]
            numpyro.deterministic(
                "alpha_floor_active",
                (mu / jnp.maximum(burst_size, _BURST_MIN)) < _ALPHA_MIN,
            )
        numpyro.deterministic(
            "beta_floor_active",
            raw_k_off < _K_OFF_MIN,
        )

    def _build_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
        n_quad_nodes: Optional[int] = None,
    ) -> dist.Distribution:
        """Run the (µ, b, *third*) → (α, β, rate) map and build the
        observation distribution.

        Supports both single-component and mixture paths:

        - **Non-mixture** (no ``mixing_weights`` in ``param_values``):
          returns a ``PoissonBetaCompound(...).to_event(1)`` at gene rank.
        - **Mixture** (``mixing_weights`` present):
          returns a ``MixtureGeneral`` over K ``PoissonBetaCompound``
          components, each sliced along the component axis.

        Parameters
        ----------
        param_values : dict
            Sampled parameter arrays.  Must contain ``"mu"`` and one of
            the three extra-param triplets.  May contain
            ``"mixing_weights"`` for the mixture path.
        param_layouts : dict, optional
            Semantic :class:`AxisLayout` per parameter key.  Required
            for mixture models to correctly broadcast mu, alpha, beta,
            rate across ``(genes, components)``.

        Returns
        -------
        dist.Distribution
            The observation distribution (already ``to_event(1)``).
        """
        # Reparameterize, emit the gene-rank deterministic sites, then build
        # the distribution from the resulting (α, β, rate).
        alpha, beta, rate, eff_burst_size, raw_k_off = (
            _twostate_dispatch_reparam(param_values)
        )
        self._emit_deterministics(
            param_values, alpha, beta, rate, eff_burst_size, raw_k_off
        )
        return self._build_bare_dist_from_reparam(
            alpha, beta, rate, param_values, n_quad_nodes=n_quad_nodes
        )

    def _build_bare_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
        n_quad_nodes: Optional[int] = None,
    ) -> dist.Distribution:
        """Build the observation distribution WITHOUT emitting deterministics.

        Used on the multi-dataset path, where the per-cell parameters have
        already been gathered by :func:`index_dataset_params` (shape
        ``(batch, G)`` or ``(batch, K, G)``).  The gene/dataset-level
        deterministic sites are emitted separately, once, on the full
        ``(D, G)`` arrays outside the cell plate — re-emitting them here on
        the per-subsample arrays would corrupt their rank.

        Parameters
        ----------
        param_values : dict
            Per-cell parameter arrays (already dataset-indexed).
        param_layouts : dict, optional
            Layouts with the ``"datasets"`` axis already dropped.

        Returns
        -------
        dist.Distribution
            The observation distribution (already ``to_event(1)``).
        """
        del param_layouts  # reserved for interface symmetry with _build_dist
        alpha, beta, rate, _, _ = _twostate_dispatch_reparam(param_values)
        return self._build_bare_dist_from_reparam(
            alpha, beta, rate, param_values, n_quad_nodes=n_quad_nodes
        )

    @staticmethod
    def _build_bare_dist_from_reparam(
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        rate: jnp.ndarray,
        param_values: Dict[str, jnp.ndarray],
        n_quad_nodes: Optional[int] = None,
    ) -> dist.Distribution:
        """Assemble the Poisson-Beta (or mixture) distribution from (α, β, rate).

        Shared by both :meth:`_build_dist` and :meth:`_build_bare_dist` so the
        single-dataset and multi-dataset paths construct identical
        distributions from their respective parameter arrays.

        ``n_quad_nodes`` controls the Gauss-Legendre node count of the
        :class:`PoissonBetaCompound`; ``None`` falls back to its default
        (60), keeping legacy behavior bit-identical.
        """
        from ....stats.distributions import PoissonBetaCompound

        # None → use the PoissonBetaCompound default (60).
        _k = n_quad_nodes if n_quad_nodes is not None else 60

        is_mixture = "mixing_weights" in param_values
        if is_mixture:
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Alpha, beta, rate inherit their shape from the dispatched
            # reparam which already broadcasts correctly when the sampled
            # params carry a component axis (..., K, G).
            return build_mixture_general(
                mixing_dist,
                lambda comp_idx: PoissonBetaCompound(
                    alpha=alpha[..., comp_idx, :],
                    beta=beta[..., comp_idx, :],
                    rate=rate[..., comp_idx, :],
                    n_quad_nodes=_k,
                ).to_event(1),
            )

        # Standard (non-mixture) path
        return PoissonBetaCompound(
            alpha=alpha, beta=beta, rate=rate, n_quad_nodes=_k
        ).to_event(1)

    # ------------------------------------------------------------------
    # Likelihood interface
    # ------------------------------------------------------------------

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        total_count_max: Optional[int] = None,
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
    ) -> None:
        """Sample from the two-state likelihood (no capture).

        Handles three plate modes:

        - prior predictive: ``counts is None``, ``batch_size is None``
        - full sampling:   ``counts`` provided, ``batch_size is None``
        - batched SVI:     ``batch_size`` set (with or without ``counts``)

        For mixture models (``mixing_weights`` in ``param_values``),
        the distribution built by ``_build_dist`` is already a
        ``MixtureGeneral`` — the cell plate works unchanged.

        On the multi-dataset path (``model_config.n_datasets`` set and
        ``dataset_indices`` provided), per-dataset gene parameters carry a
        leading dataset axis.  The deterministic sites are emitted once on
        the full ``(D, G)`` arrays outside the plate; inside the plate each
        cell's parameters are gathered with :func:`index_dataset_params`.
        """
        # Multi-dataset indexing is active only when BOTH the per-cell
        # dataset assignments and the dataset count are available.
        n_datasets = getattr(model_config, "n_datasets", None)
        use_dataset_indexing = (
            n_datasets is not None and dataset_indices is not None
        )

        _reject_phase1_unsupported(
            param_values,
            model_config,
            {
                "vae_cell_fn": vae_cell_fn,
                "annotation_prior_logits": annotation_prior_logits,
                # Pass None on the supported multi-dataset path; a non-None
                # value here means dataset_indices arrived without n_datasets.
                "dataset_indices": (
                    None if use_dataset_indexing else dataset_indices
                ),
            },
        )
        del total_count_max
        del cell_specs  # no per-cell parameters in the no-capture variant

        # Build layouts from model_config.param_specs if not provided
        # (legacy callers that don't pass param_layouts).
        if param_layouts is None:
            specs = getattr(model_config, "param_specs", None) or []
            if specs:
                param_layouts = build_param_layouts(specs, param_values)

        n_cells = dims["n_cells"]

        # Gauss-Legendre node count for the PoissonBetaCompound; None lets
        # the build helpers fall back to the distribution default (60).
        n_quad_nodes = getattr(model_config, "n_quad_nodes", None)

        # ------------------------------------------------------------------
        # Multi-dataset path: per-dataset gene parameters are indexed inside
        # the cell plate so each cell uses its own dataset's parameters.
        # ------------------------------------------------------------------
        if use_dataset_indexing:
            # Reparameterize on the full (D, G) arrays and emit the
            # gene/dataset-rank deterministic sites OUTSIDE the cell plate.
            alpha, beta, rate, eff_burst_size, raw_k_off = (
                _twostate_dispatch_reparam(param_values)
            )
            self._emit_deterministics(
                param_values, alpha, beta, rate, eff_burst_size, raw_k_off
            )
            # After gathering, the dataset axis collapses into the leading
            # cell (batch) axis — drop the "datasets" semantic axis.
            ds_layouts = _drop_dataset_axis(param_layouts)
            specs = getattr(model_config, "param_specs", None)

            if batch_size is not None:
                with numpyro.plate(
                    "cells", n_cells, subsample_size=batch_size
                ) as idx:
                    cell_pv = index_dataset_params(
                        param_values, dataset_indices[idx], n_datasets,
                        param_specs=specs,
                    )
                    obs = counts[idx] if counts is not None else None
                    numpyro.sample(
                        "counts",
                        self._build_bare_dist(
                            cell_pv, ds_layouts, n_quad_nodes=n_quad_nodes
                        ),
                        obs=obs,
                    )
            elif counts is None:
                with numpyro.plate("cells", n_cells):
                    cell_pv = index_dataset_params(
                        param_values, dataset_indices, n_datasets,
                        param_specs=specs,
                    )
                    numpyro.sample(
                        "counts",
                        self._build_bare_dist(
                            cell_pv, ds_layouts, n_quad_nodes=n_quad_nodes
                        ),
                    )
            else:
                with numpyro.plate("cells", n_cells):
                    cell_pv = index_dataset_params(
                        param_values, dataset_indices, n_datasets,
                        param_specs=specs,
                    )
                    numpyro.sample(
                        "counts",
                        self._build_bare_dist(
                            cell_pv, ds_layouts, n_quad_nodes=n_quad_nodes
                        ),
                        obs=counts,
                    )
            return

        # ------------------------------------------------------------------
        # Single-dataset path: build the distribution once outside the plate.
        # ------------------------------------------------------------------
        base_dist = self._build_dist(
            param_values, param_layouts, n_quad_nodes=n_quad_nodes
        )

        if batch_size is not None:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                obs = counts[idx] if counts is not None else None
                numpyro.sample("counts", base_dist, obs=obs)
        elif counts is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", base_dist)
        else:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", base_dist, obs=counts)

    # ------------------------------------------------------------------

    def log_prob(
        self,
        counts: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        param_layouts: Mapping[str, "AxisLayout"],
        *,
        return_by: str = "cell",
        cells_axis: int = 0,
        r_floor: float = 1e-6,
        p_floor: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
    ) -> jnp.ndarray:
        """Log-likelihood of ``counts`` under the two-state model.

        Thin wrapper around :func:`twostate_log_prob`.  Supports both
        single-component and mixture paths (``split_components``,
        ``weights``, ``weight_type`` forwarded to the mixture reduce).

        Parameters
        ----------
        counts : jnp.ndarray
            Observed count matrix, shape ``(n_cells, n_genes)``.
        params : dict
            Posterior parameter dictionary.
        param_layouts : Mapping[str, AxisLayout]
            Semantic layouts for every parameter in ``params``.
        return_by : {"cell", "gene"}
            Output reduction axis.
        cells_axis : int
            Orientation of ``counts``.
        r_floor, p_floor : float
            Accepted for interface compatibility; unused by TwoState
            (its own alpha/beta floors are in ``_twostate_reparam``).
        dtype : jnp.dtype
            Working dtype.
        split_components : bool
            Mixture-only: if ``True``, return per-component log probs.
        weights : jnp.ndarray or None
            Mixture-only: optional per-cell or per-gene weights.
        weight_type : str or None
            Mixture-only: ``"multiplicative"`` or ``"additive"``.

        Returns
        -------
        jnp.ndarray
            Log-likelihood values.
        """
        from ._log_prob import twostate_log_prob

        del r_floor, p_floor
        return twostate_log_prob(
            counts,
            params,
            param_layouts=param_layouts,
            return_by=return_by,
            cells_axis=cells_axis,
            dtype=dtype,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
        )


# ==============================================================================
# TwoStateVCPLikelihood — with per-cell capture
# ==============================================================================


class TwoStateVCPLikelihood(Likelihood):
    """Two-state promoter likelihood with per-cell capture efficiency.

    Per-cell ``p_capture`` is sampled inside the cell plate; the
    Poisson rate scales as

        rate_per_cell = (µ + b · k_off) · p_capture

    i.e. capture efficiency simply multiplies the gene-level rate.
    The closure property of the Poisson-Beta compound under binomial
    thinning (see the qmd derivation) makes this mathematically
    exact rather than an approximation.

    Phase-1 limitations are listed in the module docstring; the
    ``__init__`` signature mirrors :class:`NBWithVCPLikelihood`
    so the factory's capture-parameter wiring works unchanged.
    """

    def __init__(
        self,
        capture_param_name: Optional[str] = None,
        is_unconstrained: bool = False,
        transform: Optional[dist.transforms.Transform] = None,
        constrained_name: Optional[str] = None,
        biology_informed_spec: Optional[object] = None,
    ):
        """Mirror :class:`NBWithVCPLikelihood` so the factory's
        capture-instantiation block reuses its wiring unchanged.

        Phase-1 only accepts ``capture_param_name="p_capture"`` (or
        the auto-detect default). ``phi_capture`` is rejected at
        sample time because the rate composition would no longer
        be a simple multiplication.
        """
        self.capture_param_name = capture_param_name
        self.is_unconstrained = is_unconstrained
        self.transform = transform
        self.constrained_name = constrained_name or capture_param_name
        self.biology_informed_spec = biology_informed_spec

    # ------------------------------------------------------------------

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        total_count_max: Optional[int] = None,
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
    ) -> None:
        """Sample from the two-state-VCP likelihood.

        Computes gene-level reparam *outside* the cell plate (so the
        gene-rank deterministics are not broadcast against the plate's
        cell axis), then enters the cell plate to sample
        ``p_capture`` and emit ``counts``.

        For mixture models, the per-gene parameters (alpha, beta,
        rate_gene) carry a trailing component axis.  Inside the cell
        plate, ``p_capture`` is expanded to ``(batch, 1, 1)`` so that
        it broadcasts against ``(genes, components)`` before the rate
        composition.  A ``MixtureGeneral`` is then built over K
        ``PoissonBetaCompound`` component distributions.
        """
        from ....stats.distributions import PoissonBetaCompound

        # Multi-dataset indexing is active only when both the per-cell
        # dataset assignments and the dataset count are available.
        n_datasets = getattr(model_config, "n_datasets", None)
        use_dataset_indexing = (
            n_datasets is not None and dataset_indices is not None
        )

        _reject_phase1_unsupported(
            param_values,
            model_config,
            {
                "vae_cell_fn": vae_cell_fn,
                "annotation_prior_logits": annotation_prior_logits,
                # Pass None on the supported multi-dataset path (see the
                # no-capture sample() for the rationale).
                "dataset_indices": (
                    None if use_dataset_indexing else dataset_indices
                ),
            },
        )
        if self.capture_param_name == "phi_capture":
            raise NotImplementedError(
                "TwoState + phi_capture (mean-odds NB compatibility) is "
                "not supported; use capture_param_name='p_capture'."
            )
        del total_count_max

        # Build layouts from model_config.param_specs if not provided.
        if param_layouts is None:
            specs = getattr(model_config, "param_specs", None) or []
            if specs:
                param_layouts = build_param_layouts(specs, param_values)

        n_cells = dims["n_cells"]
        is_mixture = "mixing_weights" in param_values

        # Gauss-Legendre node count for the PoissonBetaCompound; None →
        # the distribution default (60), keeping legacy behavior intact.
        n_quad_nodes = getattr(model_config, "n_quad_nodes", None)
        _k = n_quad_nodes if n_quad_nodes is not None else 60

        # ----- gene-level reparam: OUTSIDE the cell plate -----
        alpha, beta, rate_gene, eff_burst_size, raw_k_off = (
            _twostate_dispatch_reparam(param_values)
        )
        TwoStateLikelihood._emit_deterministics(
            param_values, alpha, beta, rate_gene, eff_burst_size, raw_k_off
        )

        # Capture prior parameters: read from param_specs the same
        # way NBVCP does. Default to Beta(1, 1) if absent.
        capture_prior_params: Tuple[float, float] = (1.0, 1.0)
        for pspec in getattr(model_config, "param_specs", None) or []:
            if pspec.name == "p_capture" and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # ----- biology-informed capture pre-plate setup -----
        bio_spec = getattr(self, "biology_informed_spec", None)
        bio_log_lib_sizes = None
        bio_log_M0 = None
        if bio_spec is not None:
            if bio_spec.use_phi_capture:
                raise NotImplementedError(
                    "TwoState + biology-informed phi_capture is not "
                    "supported; use the p_capture form "
                    "(``priors={'capture_efficiency': (log_M0, sigma_M)}``)."
                )
            if bio_spec.mu_eta_prior is not None:
                raise NotImplementedError(
                    "TwoState + data-driven biology-informed capture "
                    "(hierarchical mu_eta) is not supported. "
                    "Pass fixed (log_M0, sigma_M) instead."
                )
            if counts is not None:
                bio_log_lib_sizes = jnp.log(
                    jnp.maximum(counts.sum(axis=-1), 1.0).astype(jnp.float32)
                )
            else:
                bio_log_lib_sizes = jnp.full(n_cells, bio_spec.log_M0 - 1.0)
            bio_log_M0 = bio_spec.log_M0

        # ----- inside the cell plate: sample p_capture, emit counts -----
        from .base import _sample_capture_biology_informed

        def _sample_p_capture(idx):
            """Sample p_capture per cell."""
            if bio_spec is not None:
                log_lib_batch = (
                    bio_log_lib_sizes[idx]
                    if idx is not None
                    else bio_log_lib_sizes
                )
                return _sample_capture_biology_informed(
                    log_lib_batch,
                    bio_log_M0,
                    bio_spec.sigma_M,
                    use_phi_capture=False,
                )
            return sample_capture_param(
                use_phi_capture=False,
                prior_params=capture_prior_params,
                is_unconstrained=self.is_unconstrained,
                transform=self.transform,
                constrained_name=self.constrained_name,
            )

        # Parameter specs are needed to gather dataset-specific params.
        specs = getattr(model_config, "param_specs", None)

        def _build_obs_dist(p_capture_val, alpha_v, beta_v, rate_v, indexed):
            """Compose per-cell rate = gene_rate · p_capture and build the dist.

            When ``indexed`` is True the gene rate already carries the leading
            cell (batch) axis — it was gathered per cell by
            :func:`index_dataset_params` — so it is multiplied by p_capture
            WITHOUT an extra leading expansion.  Adding ``[None, ...]`` in that
            case would balloon the broadcast to ``(batch, batch, …)`` (silent
            corruption / OOM), which is the bug this branch guards against.
            """
            if is_mixture:
                # p_capture is a per-cell scalar; expand to broadcast against
                # the trailing (components, genes) axes.
                capture_expanded = p_capture_val[:, None, None]
                if indexed:
                    # rate_v is already (batch, K, G).
                    rate_cell = rate_v * capture_expanded
                else:
                    # rate_v is gene-rank (K, G); add the cell (batch) axis.
                    rate_cell = rate_v[None, :, :] * capture_expanded
                mixing_weights = param_values["mixing_weights"]
                mixing_dist = dist.Categorical(probs=mixing_weights)
                return build_mixture_general(
                    mixing_dist,
                    lambda comp_idx: PoissonBetaCompound(
                        alpha=alpha_v[..., comp_idx, :],
                        beta=beta_v[..., comp_idx, :],
                        rate=rate_cell[..., comp_idx, :],
                        n_quad_nodes=_k,
                    ).to_event(1),
                )
            # Non-mixture path.
            if indexed:
                # rate_v is already (batch, G).
                rate_cell = rate_v * p_capture_val[:, None]
            else:
                # rate_v is gene-rank (G,); add the cell (batch) axis.
                rate_cell = rate_v[None, :] * p_capture_val[:, None]
            return PoissonBetaCompound(
                alpha=alpha_v, beta=beta_v, rate=rate_cell, n_quad_nodes=_k
            ).to_event(1)

        def _cell_reparam(plate_idx):
            """Return the (α, β, rate) this cell-batch should use.

            On the single-dataset path the gene-rank reparam computed outside
            the plate is reused verbatim.  On the multi-dataset path each
            cell's dataset-specific parameters are gathered with
            :func:`index_dataset_params` and re-reparameterized, yielding
            per-cell ``(batch, …)`` arrays.
            """
            if not use_dataset_indexing:
                return alpha, beta, rate_gene
            # ``plate_idx`` is the subsample index array under batching, or
            # None for full-data plates (where all cells are present in order).
            ds_idx = (
                dataset_indices[plate_idx]
                if plate_idx is not None
                else dataset_indices
            )
            cell_pv = index_dataset_params(
                param_values, ds_idx, n_datasets, param_specs=specs
            )
            a_c, b_c, r_c, _, _ = _twostate_dispatch_reparam(cell_pv)
            return a_c, b_c, r_c

        if batch_size is not None:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                p_capture = _sample_p_capture(idx)
                a_c, b_c, r_c = _cell_reparam(idx)
                obs_dist = _build_obs_dist(
                    p_capture, a_c, b_c, r_c, use_dataset_indexing
                )
                obs = counts[idx] if counts is not None else None
                numpyro.sample("counts", obs_dist, obs=obs)
        elif counts is None:
            with numpyro.plate("cells", n_cells):
                p_capture = _sample_p_capture(None)
                a_c, b_c, r_c = _cell_reparam(None)
                obs_dist = _build_obs_dist(
                    p_capture, a_c, b_c, r_c, use_dataset_indexing
                )
                numpyro.sample("counts", obs_dist)
        else:
            with numpyro.plate("cells", n_cells):
                p_capture = _sample_p_capture(None)
                a_c, b_c, r_c = _cell_reparam(None)
                obs_dist = _build_obs_dist(
                    p_capture, a_c, b_c, r_c, use_dataset_indexing
                )
                numpyro.sample("counts", obs_dist, obs=counts)

    # ------------------------------------------------------------------

    def log_prob(
        self,
        counts: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        param_layouts: Mapping[str, "AxisLayout"],
        *,
        return_by: str = "cell",
        cells_axis: int = 0,
        r_floor: float = 1e-6,
        p_floor: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
    ) -> jnp.ndarray:
        """Log-likelihood under the two-state-VCP model.

        Same wrapper as :meth:`TwoStateLikelihood.log_prob`; the
        VCP branch is selected inside ``twostate_log_prob`` via the
        presence of ``"p_capture"`` in ``params``.

        Parameters
        ----------
        counts : jnp.ndarray
            Observed count matrix, shape ``(n_cells, n_genes)``.
        params : dict
            Posterior parameter dictionary.
        param_layouts : Mapping[str, AxisLayout]
            Semantic layouts for every parameter in ``params``.
        return_by : {"cell", "gene"}
            Output reduction axis.
        cells_axis : int
            Orientation of ``counts``.
        r_floor, p_floor : float
            Accepted for interface compatibility; unused by TwoState.
        dtype : jnp.dtype
            Working dtype.
        split_components : bool
            Mixture-only: if ``True``, return per-component log probs.
        weights : jnp.ndarray or None
            Mixture-only: optional weighting.
        weight_type : str or None
            Mixture-only: ``"multiplicative"`` or ``"additive"``.

        Returns
        -------
        jnp.ndarray
            Log-likelihood values.
        """
        from ._log_prob import twostate_log_prob

        del r_floor, p_floor
        return twostate_log_prob(
            counts,
            params,
            param_layouts=param_layouts,
            return_by=return_by,
            cells_axis=cells_axis,
            dtype=dtype,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
        )


__all__ = [
    "TwoStateLikelihood",
    "TwoStateVCPLikelihood",
]

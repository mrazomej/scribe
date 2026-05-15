"""Two-state promoter (non-bursty) likelihood for count data.

Implements the Poisson-Beta compound likelihood

    p_gc ~ Beta(α_g, β_g)
    u_gc | p_gc ~ Poisson(r̂_g · p_gc · ν_c)

with the sampled (µ_g, b_g, k_off_g) parameterisation:

    α_g = k_on  = µ_g / b_g
    β_g = k_off = k_off_g
    r̂_g        = µ_g + b_g · k_off_g    (mean-preserving)

The natural mean-preserving identity is

    ⟨count⟩ = r̂ · α / (α + β) = µ.

Phase 1 supports SVI / MCMC fitting with the plain and VCP variants.
Phase 1 explicitly does NOT support:

- mixtures (``mixing_weights`` in ``param_values``)
- VAE inference (``vae_cell_fn``)
- annotation priors (``annotation_prior_logits``)
- multi-dataset indexing (``dataset_indices``)
- BNB overdispersion (``OverdispersionType.BNB``)
- biology-informed capture priors (``biology_informed_spec`` on VCP)
- the ``phi_capture`` capture-parameter variant on VCP

Each of those raises ``NotImplementedError`` at sample time. The
configuration layer (phase-1 build-time validation) catches most of
them earlier; the runtime guards are defence-in-depth.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import Likelihood, sample_capture_param

if TYPE_CHECKING:
    from ....core.axis_layout import AxisLayout
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig


# ==============================================================================
# Numerical floors for the (µ, b, k_off) → (α, β, rate) reparameterisation
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
_K_OFF_MIN = 0.05


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

    2. Quadrature-safety floors. When α_nat drops below ``_ALPHA_MIN``
       (low-expression gene) or k_off drops below ``_K_OFF_MIN`` (
       bursty gene), the clamped values would shift the mean unless
       we also adjust rate. The mean-preserving correction is

           rate = µ · (α + β) / α

       which gives ``rate · α / (α + β) = µ`` identically. The clamp
       changes the *shape* of the marginal in the bursty regime (less
       U-shaped Beta) but does not bias the mean.

    Returns
    -------
    alpha, beta, rate : jnp.ndarray
        Floored, mean-preserving (α, β, rate) at gene rank.
    eff_burst_size : jnp.ndarray
        ``µ / α`` — the burst size implied by the floored α. Equals
        the input ``burst_size`` when no floor activated; otherwise
        reflects the effective shape under the safety clamp.
    """
    burst_size = jnp.maximum(burst_size, _BURST_MIN)
    alpha_nat = mu / burst_size
    alpha = jnp.maximum(alpha_nat, _ALPHA_MIN)
    beta = jnp.maximum(k_off, _K_OFF_MIN)
    rate = mu * (alpha + beta) / alpha
    eff_burst_size = mu / alpha
    return alpha, beta, rate, eff_burst_size


# ==============================================================================
# Phase-1 guard helper
# ==============================================================================


def _reject_phase1_unsupported(
    param_values: Optional[Dict[str, jnp.ndarray]],
    model_config: Optional["ModelConfig"],
    kwargs: Dict[str, object],
) -> None:
    """Phase-1 guard: reject unsupported features with clear messages.

    Defence-in-depth alongside the build-time validators on
    :class:`~scribe.models.config.ModelConfig`. The build-time path
    catches these early; this runtime path catches direct invocations
    that bypass the builder.
    """
    if param_values is not None and "mixing_weights" in param_values:
        raise NotImplementedError(
            "TwoState mixture models are not supported in phase 1; "
            "see paper/_two_state_promoter.qmd for the planned design."
        )
    if kwargs.get("vae_cell_fn") is not None:
        raise NotImplementedError(
            "TwoState + VAE inference is not supported in phase 1."
        )
    if kwargs.get("annotation_prior_logits") is not None:
        raise NotImplementedError(
            "TwoState + annotation priors are not supported in phase 1."
        )
    if kwargs.get("dataset_indices") is not None:
        raise NotImplementedError(
            "TwoState + multi-dataset is not supported in phase 1."
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

    def _build_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
    ) -> dist.Distribution:
        """Run the (µ, b, k_off) → (α, β, rate) map and emit deterministics.

        Returns the to-event-1 Poisson-Beta compound at gene rank;
        caller is responsible for entering the cell plate around the
        ``numpyro.sample("counts", ...)`` call.
        """
        # Local import to avoid a circular dependency at module load
        # (scribe.stats.distributions -> ... -> scribe.models).
        from ....stats.distributions import PoissonBetaCompound

        mu = param_values["mu"]
        burst_size = param_values["burst_size"]
        k_off = param_values["k_off"]

        alpha, beta, rate, eff_burst_size = _twostate_reparam(
            mu, burst_size, k_off
        )

        # Expose derived quantities for posterior summaries (computed
        # OUTSIDE the cell plate, so they emit at gene rank).
        numpyro.deterministic("k_on", alpha)
        numpyro.deterministic("alpha", alpha)
        numpyro.deterministic("beta", beta)
        numpyro.deterministic("r_hat", rate)
        numpyro.deterministic("effective_burst_size", eff_burst_size)
        numpyro.deterministic(
            "alpha_floor_active",
            (mu / jnp.maximum(burst_size, _BURST_MIN)) < _ALPHA_MIN,
        )
        numpyro.deterministic(
            "beta_floor_active",
            k_off < _K_OFF_MIN,
        )

        return PoissonBetaCompound(alpha=alpha, beta=beta, rate=rate).to_event(
            1
        )

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
        """
        _reject_phase1_unsupported(
            param_values,
            model_config,
            {
                "vae_cell_fn": vae_cell_fn,
                "annotation_prior_logits": annotation_prior_logits,
                "dataset_indices": dataset_indices,
            },
        )
        del total_count_max  # unused; phase-1 path does not need it
        del cell_specs  # no per-cell parameters in the no-capture variant
        del param_layouts  # phase 1 path is non-mixture; layouts unused

        n_cells = dims["n_cells"]
        base_dist = self._build_dist(param_values)

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

        Thin wrapper around :func:`twostate_log_prob`. The ``r_floor``
        and ``p_floor`` arguments are accepted for interface
        compatibility with the NB family but are unused; the
        two-state model has its own floors on ``α`` and ``β`` inside
        :func:`_twostate_reparam`.
        """
        from ._log_prob import twostate_log_prob

        del r_floor, p_floor, param_layouts
        if split_components or weights is not None or weight_type is not None:
            raise NotImplementedError(
                "TwoState mixture log_prob features are not supported in "
                "phase 1."
            )
        return twostate_log_prob(
            counts,
            params,
            return_by=return_by,
            cells_axis=cells_axis,
            dtype=dtype,
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
        """
        from ....stats.distributions import PoissonBetaCompound

        _reject_phase1_unsupported(
            param_values,
            model_config,
            {
                "vae_cell_fn": vae_cell_fn,
                "annotation_prior_logits": annotation_prior_logits,
                "dataset_indices": dataset_indices,
            },
        )
        # Biology-informed capture guard (lives here because it
        # depends on a constructor-injected spec, not on the model
        # config alone).
        if getattr(self, "biology_informed_spec", None) is not None:
            raise NotImplementedError(
                "TwoState + biology-informed capture prior is not "
                "supported in phase 1; pass a flat Beta prior on "
                "p_capture instead."
            )
        if self.capture_param_name == "phi_capture":
            raise NotImplementedError(
                "TwoState + phi_capture (mean-odds NB compatibility) is "
                "not supported in phase 1; use capture_param_name='p_capture'."
            )
        del total_count_max, param_layouts

        n_cells = dims["n_cells"]

        # ----- gene-level reparam: OUTSIDE the cell plate -----
        mu = param_values["mu"]
        burst_size = param_values["burst_size"]
        k_off = param_values["k_off"]
        alpha, beta, rate_gene, eff_burst_size = _twostate_reparam(
            mu, burst_size, k_off
        )
        numpyro.deterministic("k_on", alpha)
        numpyro.deterministic("alpha", alpha)
        numpyro.deterministic("beta", beta)
        numpyro.deterministic("r_hat", rate_gene)
        numpyro.deterministic("effective_burst_size", eff_burst_size)
        numpyro.deterministic(
            "alpha_floor_active",
            (mu / jnp.maximum(burst_size, _BURST_MIN)) < _ALPHA_MIN,
        )
        numpyro.deterministic(
            "beta_floor_active",
            k_off < _K_OFF_MIN,
        )

        # Capture prior parameters: read from param_specs the same
        # way NBVCP does. Default to Beta(1, 1) if absent.
        capture_prior_params: Tuple[float, float] = (1.0, 1.0)
        for pspec in getattr(model_config, "param_specs", None) or []:
            if pspec.name == "p_capture" and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # ----- inside the cell plate: sample p_capture, emit counts -----
        if batch_size is not None:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                p_capture = sample_capture_param(
                    use_phi_capture=False,
                    prior_params=capture_prior_params,
                    is_unconstrained=self.is_unconstrained,
                    transform=self.transform,
                    constrained_name=self.constrained_name,
                )
                rate = rate_gene[None, :] * p_capture[:, None]
                obs = counts[idx] if counts is not None else None
                numpyro.sample(
                    "counts",
                    PoissonBetaCompound(
                        alpha=alpha, beta=beta, rate=rate
                    ).to_event(1),
                    obs=obs,
                )
        elif counts is None:
            with numpyro.plate("cells", n_cells):
                p_capture = sample_capture_param(
                    use_phi_capture=False,
                    prior_params=capture_prior_params,
                    is_unconstrained=self.is_unconstrained,
                    transform=self.transform,
                    constrained_name=self.constrained_name,
                )
                rate = rate_gene[None, :] * p_capture[:, None]
                numpyro.sample(
                    "counts",
                    PoissonBetaCompound(
                        alpha=alpha, beta=beta, rate=rate
                    ).to_event(1),
                )
        else:
            with numpyro.plate("cells", n_cells):
                p_capture = sample_capture_param(
                    use_phi_capture=False,
                    prior_params=capture_prior_params,
                    is_unconstrained=self.is_unconstrained,
                    transform=self.transform,
                    constrained_name=self.constrained_name,
                )
                rate = rate_gene[None, :] * p_capture[:, None]
                numpyro.sample(
                    "counts",
                    PoissonBetaCompound(
                        alpha=alpha, beta=beta, rate=rate
                    ).to_event(1),
                    obs=counts,
                )

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
        """
        from ._log_prob import twostate_log_prob

        del r_floor, p_floor, param_layouts
        if split_components or weights is not None or weight_type is not None:
            raise NotImplementedError(
                "TwoState mixture log_prob features are not supported in "
                "phase 1."
            )
        return twostate_log_prob(
            counts,
            params,
            return_by=return_by,
            cells_axis=cells_axis,
            dtype=dtype,
        )


__all__ = [
    "TwoStateLikelihood",
    "TwoStateVCPLikelihood",
]

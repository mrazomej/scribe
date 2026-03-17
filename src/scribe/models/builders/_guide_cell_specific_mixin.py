"""Cell-specific guide dispatch implementations.

This module contains non-amortized and biology-informed cell-level guide
registrations used inside the cells plate.
"""

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints

from .parameter_specs import (
    BetaPrimeSpec,
    BetaSpec,
    BiologyInformedCaptureSpec,
    NormalWithTransformSpec,
)
from scribe.stats.distributions import BetaPrime
from ..components.guide_families import MeanFieldGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


@dispatch(BetaSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for cell-specific Beta parameter (e.g., p_capture).

    When batch_idx is provided, we index into the full parameter arrays
    to get only the parameters for the current mini-batch.

    Parameters
    ----------
    spec : BetaSpec
        Parameter specification (must have is_cell_specific=True).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices of cells in current mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value for the current batch.
    """
    n_cells = dims["n_cells"]
    params = spec.guide if spec.guide is not None else spec.default_params

    # Variational parameters for ALL cells (allocated once, indexed into)
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(n_cells, params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(n_cells, params[1]),
        constraint=constraints.positive,
    )

    if batch_idx is None:
        # Full sampling: use all parameters
        return numpyro.sample(spec.name, dist.Beta(alpha, beta))
    else:
        # Batch sampling: index into parameters for this mini-batch
        return numpyro.sample(
            spec.name, dist.Beta(alpha[batch_idx], beta[batch_idx])
        )


# ------------------------------------------------------------------------------
# BetaPrime Distribution MeanField Guide (Cell-Specific)
# ------------------------------------------------------------------------------


@dispatch(BetaPrimeSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaPrimeSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """
    MeanField guide for cell-specific BetaPrime parameter (e.g., phi_capture).

    When batch_idx is provided, we index into the full parameter arrays to get
    only the parameters for the current mini-batch.

    Parameters
    ----------
    spec : BetaPrimeSpec
        Parameter specification (must have is_cell_specific=True).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices of cells in current mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value for the current batch.
    """
    n_cells = dims["n_cells"]
    params = spec.guide if spec.guide is not None else spec.default_params

    # Variational parameters for ALL cells (allocated once, indexed into)
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(n_cells, params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(n_cells, params[1]),
        constraint=constraints.positive,
    )

    if batch_idx is None:
        # Full sampling: use all parameters
        return numpyro.sample(spec.name, BetaPrime(alpha, beta))
    else:
        # Batch sampling: index into parameters for this mini-batch
        return numpyro.sample(
            spec.name, BetaPrime(alpha[batch_idx], beta[batch_idx])
        )


@dispatch(NormalWithTransformSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: NormalWithTransformSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for cell-specific unconstrained parameters.

    Works for SigmoidNormalSpec (Beta -> [0,1]), PositiveNormalSpec (BetaPrime
    -> [0,+inf)), and other NormalWithTransformSpec subclasses.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (SigmoidNormalSpec, PositiveNormalSpec, etc.).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space.
    """
    n_cells = dims["n_cells"]
    params = spec.guide if spec.guide is not None else spec.default_params

    loc = numpyro.param(f"{spec.name}_loc", jnp.full(n_cells, params[0]))
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(n_cells, params[1]),
        constraint=constraints.positive,
    )

    if batch_idx is None:
        base_dist = dist.Normal(loc, scale)
    else:
        base_dist = dist.Normal(loc[batch_idx], scale[batch_idx])

    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)
    return numpyro.sample(spec.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# ==============================================================================
# Pre-plate: hierarchical mu_eta variational parameters
# ==============================================================================


def guide_mu_eta_hierarchy(
    spec: BiologyInformedCaptureSpec,
    n_datasets: int,
) -> None:
    """Create variational parameters for the hierarchical mu_eta prior.

    Matches the sample sites produced by the model-side helpers in
    ``likelihoods.base._sample_hierarchical_mu_eta_*``.

    Called by ``GuideBuilder.build()`` *before* the cell plate.
    When ``n_datasets < 2`` the hierarchy collapses to a single scalar
    ``mu_eta ~ N(loc, scale)`` (same as the old data-driven path).

    Parameters
    ----------
    spec : BiologyInformedCaptureSpec
        Capture specification (carries ``mu_eta_prior``, ``log_M0``, etc.).
    n_datasets : int
        Number of datasets.  When ``< 2``, falls back to scalar guide.
    """
    prior = spec.mu_eta_prior

    if n_datasets < 2:
        # Single-dataset fallback: scalar mu_eta
        _loc = numpyro.param("mu_eta_loc", jnp.array(spec.log_M0))
        _scale = numpyro.param(
            "mu_eta_scale",
            jnp.array(0.1),
            constraint=constraints.positive,
        )
        numpyro.sample("mu_eta", dist.Normal(_loc, _scale))
        return

    # -- Population mean (shared across all prior types) --------------------
    pop_loc = numpyro.param("mu_eta_pop_loc", jnp.array(spec.log_M0))
    pop_scale = numpyro.param(
        "mu_eta_pop_scale",
        jnp.array(0.1),
        constraint=constraints.positive,
    )
    numpyro.sample("mu_eta_pop", dist.Normal(pop_loc, pop_scale))

    # -- Prior-type-specific auxiliary params --------------------------------
    if prior == "gaussian":
        # tau_eta: inter-dataset spread (Softplus-transformed Normal)
        tau_loc = numpyro.param("tau_eta_loc", jnp.array(-2.0))
        tau_scale = numpyro.param(
            "tau_eta_scale",
            jnp.array(0.1),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "tau_eta",
            dist.TransformedDistribution(
                dist.Normal(tau_loc, tau_scale),
                dist.transforms.SoftplusTransform(),
            ),
        )

    elif prior == "horseshoe":
        # Global shrinkage tau
        _hs_tau_loc = numpyro.param("tau_mu_eta_loc", jnp.array(0.1))
        _hs_tau_sc = numpyro.param(
            "tau_mu_eta_scale",
            jnp.array(0.1),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "tau_mu_eta",
            dist.TransformedDistribution(
                dist.Normal(_hs_tau_loc, _hs_tau_sc),
                dist.transforms.SoftplusTransform(),
            ),
        )
        # Per-dataset local shrinkage lambda
        _lam_loc = numpyro.param(
            "lambda_mu_eta_loc", jnp.full(n_datasets, 0.1)
        )
        _lam_sc = numpyro.param(
            "lambda_mu_eta_scale",
            jnp.full(n_datasets, 0.1),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "lambda_mu_eta",
            dist.TransformedDistribution(
                dist.Normal(_lam_loc, _lam_sc),
                dist.transforms.SoftplusTransform(),
            ).to_event(1),
        )
        # Slab width c^2
        _csq_loc = numpyro.param("c_sq_mu_eta_loc", jnp.array(2.0))
        _csq_sc = numpyro.param(
            "c_sq_mu_eta_scale",
            jnp.array(0.1),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "c_sq_mu_eta",
            dist.TransformedDistribution(
                dist.Normal(_csq_loc, _csq_sc),
                dist.transforms.SoftplusTransform(),
            ),
        )

    elif prior == "neg":
        # Per-dataset zeta (outer Gamma rate)
        _z_loc = numpyro.param(
            "zeta_mu_eta_loc", jnp.full(n_datasets, 1.0)
        )
        _z_sc = numpyro.param(
            "zeta_mu_eta_scale",
            jnp.full(n_datasets, 0.1),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "zeta_mu_eta",
            dist.TransformedDistribution(
                dist.Normal(_z_loc, _z_sc),
                dist.transforms.SoftplusTransform(),
            ).to_event(1),
        )
        # Per-dataset psi (inner Gamma variance)
        _p_loc = numpyro.param(
            "psi_mu_eta_loc", jnp.full(n_datasets, 1.0)
        )
        _p_sc = numpyro.param(
            "psi_mu_eta_scale",
            jnp.full(n_datasets, 0.1),
            constraint=constraints.positive,
        )
        numpyro.sample(
            "psi_mu_eta",
            dist.TransformedDistribution(
                dist.Normal(_p_loc, _p_sc),
                dist.transforms.SoftplusTransform(),
            ).to_event(1),
        )

    # -- NCP deviations (shared across all hierarchical prior types) --------
    raw_loc = numpyro.param("mu_eta_raw_loc", jnp.zeros(n_datasets))
    raw_scale = numpyro.param(
        "mu_eta_raw_scale",
        jnp.full(n_datasets, 0.1),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mu_eta_raw",
        dist.Normal(raw_loc, raw_scale).to_event(1),
    )


# ------------------------------------------------------------------------------
# Biology-Informed Capture MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(BiologyInformedCaptureSpec, MeanFieldGuide, dict, object)
def setup_cell_specific_guide(
    spec: BiologyInformedCaptureSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for biology-informed capture parameter.

    Provides per-cell variational parameters for eta_capture (the
    latent log(M_c / L_c), constrained >= 0).

    Two guide parameterizations are supported, controlled by
    ``model_config.eta_capture_guide``:

    * ``"softplus_normal"`` (default) — samples an unconstrained
      Normal and maps through softplus, yielding a logit-normal on
      nu_c with smooth gradients everywhere. Variational params:
      ``eta_capture_raw_loc``, ``eta_capture_raw_scale``.
    * ``"truncated_normal"`` (legacy) — uses TruncatedNormal(low=0)
      to enforce eta >= 0 directly. Variational params:
      ``eta_capture_loc``, ``eta_capture_scale``.

    The model then applies the exact transformation to phi_capture or
    p_capture inside the likelihood (unchanged by guide choice).

    For data-driven mode, the hierarchical mu_eta variational parameters
    are registered by ``guide_mu_eta_hierarchy`` (called before the plate).

    Parameters
    ----------
    spec : BiologyInformedCaptureSpec
        The biology-informed capture specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimensions including n_cells.
    model_config : ModelConfig
        Model configuration.
    batch_idx : Optional[jnp.ndarray]
        Indices for mini-batch. None for full sampling.

    Returns
    -------
    jnp.ndarray
        Sampled eta_capture values and deterministic capture parameter.
    """
    n_cells = dims["n_cells"]
    eta_guide = getattr(model_config, "eta_capture_guide", "truncated_normal")

    if eta_guide == "softplus_normal":
        # Softplus-normal guide: sample unconstrained eta_capture_raw,
        # then map through softplus to get eta_capture in (0, inf).
        # This induces a logit-normal on nu_c = sigmoid(-raw), with
        # smooth bounded gradients and no truncation boundary.
        # softplus_inv(x) = log(exp(x) - 1) ≈ x for x >> 1
        raw_init = jnp.log(jnp.expm1(jnp.full(n_cells, spec.log_M0)))
        raw_loc = numpyro.param("eta_capture_raw_loc", raw_init)
        raw_scale = numpyro.param(
            "eta_capture_raw_scale",
            jnp.full(n_cells, spec.sigma_M),
            constraint=constraints.positive,
        )

        if batch_idx is None:
            base_normal = dist.Normal(raw_loc, raw_scale)
        else:
            base_normal = dist.Normal(
                raw_loc[batch_idx], raw_scale[batch_idx]
            )

        # eta = softplus(raw) lives in (0, inf), matching the
        # TruncatedNormal prior's support.
        eta_dist = dist.TransformedDistribution(
            base_normal, dist.transforms.SoftplusTransform()
        )
        eta = numpyro.sample("eta_capture", eta_dist)

    else:
        # Legacy TruncatedNormal guide: direct truncation at 0.
        eta_loc = numpyro.param(
            "eta_capture_loc", jnp.full(n_cells, spec.log_M0)
        )
        eta_scale = numpyro.param(
            "eta_capture_scale",
            jnp.full(n_cells, spec.sigma_M),
            constraint=constraints.positive,
        )

        if batch_idx is None:
            base_dist = dist.TruncatedNormal(
                eta_loc, eta_scale, low=0.0
            )
        else:
            base_dist = dist.TruncatedNormal(
                eta_loc[batch_idx], eta_scale[batch_idx], low=0.0
            )
        eta = numpyro.sample("eta_capture", base_dist)

    # Deterministic transform to capture parameter (identical for
    # both guide types — the model physics are unchanged).
    if spec.use_phi_capture:
        capture_value = jnp.exp(eta) - 1.0
        numpyro.deterministic("phi_capture", capture_value)
    else:
        capture_value = jnp.exp(-eta)
        numpyro.deterministic("p_capture", capture_value)

    return capture_value

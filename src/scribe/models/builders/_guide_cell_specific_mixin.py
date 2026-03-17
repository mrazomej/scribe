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
    latent log(M_c / L_c), constrained >= 0). Uses TruncatedNormal(low=0)
    to enforce the physical constraint that a cell cannot emit more
    molecules than it contains. The model then applies the exact
    transformation to phi_capture or p_capture inside the likelihood.

    For data-driven mode, also provides variational parameters for the
    shared mu_eta parameter.

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

    # Per-cell eta_capture variational parameters
    # (mu_eta for data-driven mode is sampled before the plate in the
    # GuideBuilder.build() method)
    # Initialize loc near the prior mean: log_M0 (will be offset by
    # -log_lib in the likelihood, but the guide learns the full eta)
    eta_loc = numpyro.param("eta_capture_loc", jnp.full(n_cells, spec.log_M0))
    eta_scale = numpyro.param(
        "eta_capture_scale",
        jnp.full(n_cells, spec.sigma_M),
        constraint=constraints.positive,
    )

    # TruncatedNormal(low=0) matches the model prior and enforces
    # eta_c >= 0 <=> p_capture <= 1.
    if batch_idx is None:
        base_dist = dist.TruncatedNormal(eta_loc, eta_scale, low=0.0)
    else:
        base_dist = dist.TruncatedNormal(
            eta_loc[batch_idx], eta_scale[batch_idx], low=0.0
        )

    eta = numpyro.sample("eta_capture", base_dist)

    # Deterministic transform to capture parameter
    if spec.use_phi_capture:
        capture_value = jnp.exp(eta) - 1.0
        numpyro.deterministic("phi_capture", capture_value)
    else:
        capture_value = jnp.exp(-eta)
        numpyro.deterministic("p_capture", capture_value)

    return capture_value

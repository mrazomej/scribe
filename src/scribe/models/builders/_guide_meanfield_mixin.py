"""Standard mean-field guide dispatch implementations.

This module holds multipledispatch registrations for common mean-field
variational families (Beta, BetaPrime, LogNormal, transformed Normal,
and Dirichlet).
"""

from typing import TYPE_CHECKING, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints

from .parameter_specs import (
    BetaPrimeSpec,
    BetaSpec,
    DirichletSpec,
    LogNormalSpec,
    NormalWithTransformSpec,
    resolve_shape,
)
from scribe.stats.distributions import BetaPrime
from ..components.guide_families import MeanFieldGuide

if TYPE_CHECKING:
    from ..config import ModelConfig

@dispatch(BetaSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: BetaSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Beta parameter.

    Creates learnable variational parameters (alpha, beta) for a Beta
    distribution approximation.

    Parameters
    ----------
    spec : BetaSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from variational distribution.

    Notes
    -----
    Beta distribution parameters (alpha, beta) must be positive.
    The sampled value will satisfy spec.constraint (unit_interval).
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    # Variational params for Beta must be positive
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(shape, params[0]) if shape else jnp.array(params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(shape, params[1]) if shape else jnp.array(params[1]),
        constraint=constraints.positive,
    )

    if shape == ():
        return numpyro.sample(spec.name, dist.Beta(alpha, beta))
    return numpyro.sample(
        spec.name, dist.Beta(alpha, beta).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# BetaPrime Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(BetaPrimeSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: BetaPrimeSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for BetaPrime parameter.

    Creates learnable variational parameters (alpha, beta) for a BetaPrime
    distribution approximation.

    Parameters
    ----------
    spec : BetaPrimeSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from variational distribution.
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    # Variational params for BetaPrime must be positive
    alpha = numpyro.param(
        f"{spec.name}_alpha",
        jnp.full(shape, params[0]) if shape else jnp.array(params[0]),
        constraint=constraints.positive,
    )
    beta = numpyro.param(
        f"{spec.name}_beta",
        jnp.full(shape, params[1]) if shape else jnp.array(params[1]),
        constraint=constraints.positive,
    )

    # Sample from BetaPrime distribution for scalar parameters
    if shape == ():
        return numpyro.sample(spec.name, BetaPrime(alpha, beta))
    # Sample from BetaPrime distribution for non-scalar parameters
    return numpyro.sample(
        spec.name, BetaPrime(alpha, beta).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# LogNormal Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(LogNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: LogNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for LogNormal parameter.

    Creates learnable variational parameters (loc, scale) for a LogNormal
    distribution approximation.

    Parameters
    ----------
    spec : LogNormalSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from variational distribution.

    Notes
    -----
    scale must be positive. The sampled value will satisfy
    spec.constraint (positive).
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    loc = numpyro.param(f"{spec.name}_loc", jnp.full(shape, params[0]))
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(shape, params[1]),
        constraint=constraints.positive,
    )

    return numpyro.sample(
        spec.name, dist.LogNormal(loc, scale).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# Normal with Transform Distribution MeanField Guide
# ------------------------------------------------------------------------------


@dispatch(NormalWithTransformSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: NormalWithTransformSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for transformed Normal parameters.

    Uses TransformedDistribution(Normal, spec.transform) for:
    - Automatic Jacobian handling in ELBO
    - Samples directly in constrained space
    - Constraint derived from spec.transform.codomain

    Works for SigmoidNormalSpec, PositiveNormalSpec, SoftplusNormalSpec, etc.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (or subclass).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space.
    """
    params = spec.guide if spec.guide is not None else spec.default_params
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    # Variational parameters for the base Normal
    loc = numpyro.param(
        f"{spec.name}_loc",
        jnp.full(shape, params[0]) if shape else jnp.array(params[0]),
    )
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(shape, params[1]) if shape else jnp.array(params[1]),
        constraint=constraints.positive,
    )

    # Create base Normal and wrap with transform
    if shape == ():
        base_dist = dist.Normal(loc, scale)
    else:
        base_dist = dist.Normal(loc, scale).to_event(len(shape))

    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)

    # Sample in constrained space (Jacobian handled automatically)
    return numpyro.sample(spec.constrained_name, transformed_dist)

@dispatch(DirichletSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: DirichletSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Dirichlet parameter.

    Creates learnable variational parameters (concentrations) for a Dirichlet
    distribution approximation. Used for mixture weights and compositional
    parameters.

    Parameters
    ----------
    spec : DirichletSpec
        Parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from Dirichlet distribution (on simplex).

    Notes
    -----
    - Concentration parameters must all be positive.
    - The sampled value will be on the simplex (sums to 1).
    - For mixture weights, the concentration vector length equals n_components.
    """
    # Get guide params or use default
    params = spec.guide if spec.guide is not None else spec.default_params

    # Convert tuple to array and ensure all values are positive
    concentrations = jnp.array(params)

    # Determine parameter name: use "mixing_concentrations" for mixing_weights
    # (matching legacy code), otherwise use "{name}_concentrations"
    if spec.name == "mixing_weights":
        param_name = "mixing_concentrations"
    else:
        param_name = f"{spec.name}_concentrations"

    # Create variational parameter with positive constraint
    concentrations_param = numpyro.param(
        param_name,
        concentrations,
        constraint=constraints.positive,
    )

    # Sample from Dirichlet distribution
    return numpyro.sample(spec.name, dist.Dirichlet(concentrations_param))

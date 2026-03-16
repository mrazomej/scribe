"""NEG (Normal-Exponential-Gamma) guide dispatch implementations.

This module includes registrations for NEG hyperparameter guides (GammaSpec)
and NCP raw-latent guides for NEG hierarchical specs (gene-level and
dataset-level). The NEG prior uses a Gamma-Gamma hierarchy instead of
Half-Cauchy scales, but the NCP structure (z ~ Normal(0,1) → transform) is
identical to the horseshoe, so the raw-latent guides follow the same pattern.
"""

from typing import TYPE_CHECKING, Dict

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints

from .parameter_specs import (
    GammaSpec,
    NEGDatasetExpNormalSpec,
    NEGDatasetSigmoidNormalSpec,
    NEGHierarchicalExpNormalSpec,
    NEGHierarchicalSigmoidNormalSpec,
    resolve_shape,
)
from ..components.guide_families import LowRankGuide, MeanFieldGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


# ------------------------------------------------------------------------------
# GammaSpec MeanField Guide (Gamma variational family)
# ------------------------------------------------------------------------------
# The Gamma variational family is the conjugate match for Gamma priors and,
# crucially, can place its mode at zero when concentration < 1.  This lets
# the optimizer learn strong shrinkage (psi → 0) for the NEG hierarchy,
# which LogNormal cannot achieve because exp(·) > 0 always.
# Initialization: concentration=0.5 (mode at 0), rate=1.0 (mean 0.5).
# The rate_name field is only used in the model for prior sampling; the
# guide samples the parameter directly.
# ------------------------------------------------------------------------------


@dispatch(GammaSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: GammaSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Gamma parameters using Gamma variational family.

    Gamma is the natural conjugate variational family for Gamma posteriors.
    With concentration < 1 the density diverges at zero, letting the
    optimizer express strong shrinkage (psi → 0).

    Parameters
    ----------
    spec : GammaSpec
        Gamma parameter specification (psi or zeta in NEG hierarchy).
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    concentration = numpyro.param(
        f"{spec.name}_concentration",
        jnp.full(shape, 0.5) if shape else jnp.array(0.5),
        constraint=constraints.positive,
    )
    rate = numpyro.param(
        f"{spec.name}_rate",
        jnp.full(shape, 1.0) if shape else jnp.array(1.0),
        constraint=constraints.positive,
    )

    if shape == ():
        d = dist.Gamma(concentration, rate)
    else:
        d = dist.Gamma(concentration, rate).to_event(len(shape))

    return numpyro.sample(spec.name, d)



# ------------------------------------------------------------------------------
# NEGHierarchicalSigmoidNormalSpec MeanField Guide (Normal for raw NCP site)
# ------------------------------------------------------------------------------
# The NEG hierarchical specs use NCP: z ~ Normal(0,1) in the model. The guide
# targets the raw_name site with Normal(loc, scale), same as horseshoe.
# ------------------------------------------------------------------------------


@dispatch(NEGHierarchicalSigmoidNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: NEGHierarchicalSigmoidNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for NEG NCP z variable (gene-level p/gate).

    Targets the raw ``z`` variable with a Normal(loc, scale) guide.
    The constrained parameter is handled by numpyro.deterministic in the model.

    Parameters
    ----------
    spec : NEGHierarchicalSigmoidNormalSpec
        NEG gene-level parameter specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value (unconstrained).
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.full(shape, 0.0) if shape else jnp.array(0.0),
    )
    scale = numpyro.param(
        f"{spec.raw_name}_scale",
        jnp.full(shape, 1.0) if shape else jnp.array(1.0),
        constraint=constraints.positive,
    )

    if shape == ():
        z_dist = dist.Normal(loc, scale)
    else:
        z_dist = dist.Normal(loc, scale).to_event(len(shape))

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGHierarchicalSigmoidNormalSpec LowRank Guide
# ------------------------------------------------------------------------------


@dispatch(NEGHierarchicalSigmoidNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: NEGHierarchicalSigmoidNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for NEG NCP z variable (gene-level p/gate).

    Parameters
    ----------
    spec : NEGHierarchicalSigmoidNormalSpec
        NEG gene-level parameter specification.
    guide : LowRankGuide
        Low-rank guide marker with rank.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value (unconstrained).
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank

    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"{spec.raw_name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"{spec.raw_name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    z_dist = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    if n_batch_dims > 0:
        z_dist = z_dist.to_event(n_batch_dims)

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGHierarchicalExpNormalSpec MeanField Guide (Normal for raw NCP site)
# ------------------------------------------------------------------------------


@dispatch(NEGHierarchicalExpNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: NEGHierarchicalExpNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for NEG NCP z variable (gene-level phi).

    Parameters
    ----------
    spec : NEGHierarchicalExpNormalSpec
        NEG gene-level phi specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value (unconstrained).
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.full(shape, 0.0) if shape else jnp.array(0.0),
    )
    scale = numpyro.param(
        f"{spec.raw_name}_scale",
        jnp.full(shape, 1.0) if shape else jnp.array(1.0),
        constraint=constraints.positive,
    )

    if shape == ():
        z_dist = dist.Normal(loc, scale)
    else:
        z_dist = dist.Normal(loc, scale).to_event(len(shape))

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGHierarchicalExpNormalSpec LowRank Guide
# ------------------------------------------------------------------------------


@dispatch(NEGHierarchicalExpNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: NEGHierarchicalExpNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for NEG NCP z variable (gene-level phi).

    Parameters
    ----------
    spec : NEGHierarchicalExpNormalSpec
        NEG gene-level phi specification.
    guide : LowRankGuide
        Low-rank guide marker with rank.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value (unconstrained).
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank

    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"{spec.raw_name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"{spec.raw_name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    z_dist = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    if n_batch_dims > 0:
        z_dist = z_dist.to_event(n_batch_dims)

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGDatasetExpNormalSpec MeanField Guide (Normal for raw NCP site)
# ------------------------------------------------------------------------------


@dispatch(NEGDatasetExpNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: NEGDatasetExpNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for NEG NCP z variable (dataset-level mu/r).

    Parameters
    ----------
    spec : NEGDatasetExpNormalSpec
        NEG dataset-level mu/r specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value with shape ``(n_datasets, n_genes)``.
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.full(shape, 0.0) if shape else jnp.array(0.0),
    )
    scale = numpyro.param(
        f"{spec.raw_name}_scale",
        jnp.full(shape, 1.0) if shape else jnp.array(1.0),
        constraint=constraints.positive,
    )

    if shape == ():
        z_dist = dist.Normal(loc, scale)
    else:
        z_dist = dist.Normal(loc, scale).to_event(len(shape))

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGDatasetExpNormalSpec LowRank Guide
# ------------------------------------------------------------------------------


@dispatch(NEGDatasetExpNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: NEGDatasetExpNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for NEG NCP z variable (dataset-level mu/r).

    Parameters
    ----------
    spec : NEGDatasetExpNormalSpec
        NEG dataset-level mu/r specification.
    guide : LowRankGuide
        Low-rank guide marker with rank.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank

    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"{spec.raw_name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"{spec.raw_name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    z_dist = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    if n_batch_dims > 0:
        z_dist = z_dist.to_event(n_batch_dims)

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGDatasetSigmoidNormalSpec MeanField Guide (Normal for raw NCP site)
# ------------------------------------------------------------------------------


@dispatch(NEGDatasetSigmoidNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: NEGDatasetSigmoidNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for NEG NCP z variable (dataset-level p/gate).

    Parameters
    ----------
    spec : NEGDatasetSigmoidNormalSpec
        NEG dataset-level p or gate specification.
    guide : MeanFieldGuide
        Mean-field guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value.
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.full(shape, 0.0) if shape else jnp.array(0.0),
    )
    scale = numpyro.param(
        f"{spec.raw_name}_scale",
        jnp.full(shape, 1.0) if shape else jnp.array(1.0),
        constraint=constraints.positive,
    )

    if shape == ():
        z_dist = dist.Normal(loc, scale)
    else:
        z_dist = dist.Normal(loc, scale).to_event(len(shape))

    return numpyro.sample(spec.raw_name, z_dist)


# ------------------------------------------------------------------------------
# NEGDatasetSigmoidNormalSpec LowRank Guide
# ------------------------------------------------------------------------------


@dispatch(NEGDatasetSigmoidNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: NEGDatasetSigmoidNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for NEG NCP z variable (dataset-level p/gate).

    Parameters
    ----------
    spec : NEGDatasetSigmoidNormalSpec
        NEG dataset-level p or gate specification.
    guide : LowRankGuide
        Low-rank guide marker with rank.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled z value.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank

    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    loc = numpyro.param(
        f"{spec.raw_name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"{spec.raw_name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"{spec.raw_name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    z_dist = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)

    if n_batch_dims > 0:
        z_dist = z_dist.to_event(n_batch_dims)

    return numpyro.sample(spec.raw_name, z_dist)

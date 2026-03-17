"""Horseshoe guide dispatch implementations.

This module includes registrations for horseshoe hyperparameter guides and
NCP raw-latent guides (gene-level and dataset-level).
"""

from typing import TYPE_CHECKING, Dict

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints

from .parameter_specs import (
    HalfCauchySpec,
    HorseshoeDatasetPositiveNormalSpec,
    HorseshoeDatasetSigmoidNormalSpec,
    HorseshoeHierarchicalPositiveNormalSpec,
    HorseshoeHierarchicalSigmoidNormalSpec,
    InverseGammaSpec,
    resolve_shape,
)
from ..components.guide_families import LowRankGuide, MeanFieldGuide

if TYPE_CHECKING:
    from ..config import ModelConfig

@dispatch(HalfCauchySpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: HalfCauchySpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Half-Cauchy parameters using LogNormal.

    LogNormal is the natural variational family for Half-Cauchy posteriors:
    positive support, right-skewed, and reparameterizable.

    Parameters
    ----------
    spec : HalfCauchySpec
        Half-Cauchy parameter specification.
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

    loc = numpyro.param(
        f"{spec.name}_loc",
        jnp.full(shape, 0.0) if shape else jnp.array(0.0),
    )
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(shape, 0.5) if shape else jnp.array(0.5),
        constraint=constraints.positive,
    )

    if shape == ():
        d = dist.LogNormal(loc, scale)
    else:
        d = dist.LogNormal(loc, scale).to_event(len(shape))

    return numpyro.sample(spec.name, d)


# ------------------------------------------------------------------------------
# HalfCauchy LowRank Guide (LogNormal variational family)
# ------------------------------------------------------------------------------


@dispatch(HalfCauchySpec, LowRankGuide, dict, object)
def setup_guide(
    spec: HalfCauchySpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for Half-Cauchy parameters using LogNormal.

    Parameters
    ----------
    spec : HalfCauchySpec
        Half-Cauchy parameter specification (gene-specific lambda).
    guide : LowRankGuide
        Low-rank guide marker with rank.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
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
        f"log_{spec.name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"log_{spec.name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"log_{spec.name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    transformed_dist = dist.TransformedDistribution(
        base, dist.transforms.ExpTransform()
    )

    if n_batch_dims > 0:
        transformed_dist = transformed_dist.to_event(n_batch_dims)

    return numpyro.sample(spec.name, transformed_dist)


# ------------------------------------------------------------------------------
# InverseGamma MeanField Guide (LogNormal variational family)
# ------------------------------------------------------------------------------


@dispatch(InverseGammaSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: InverseGammaSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for Inverse-Gamma parameters using LogNormal.

    Parameters
    ----------
    spec : InverseGammaSpec
        Inverse-Gamma parameter specification (slab c^2).
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

    loc = numpyro.param(
        f"{spec.name}_loc",
        jnp.full(shape, 1.0) if shape else jnp.array(1.0),
    )
    scale = numpyro.param(
        f"{spec.name}_scale",
        jnp.full(shape, 0.5) if shape else jnp.array(0.5),
        constraint=constraints.positive,
    )

    if shape == ():
        d = dist.LogNormal(loc, scale)
    else:
        d = dist.LogNormal(loc, scale).to_event(len(shape))

    return numpyro.sample(spec.name, d)


# ------------------------------------------------------------------------------
# InverseGamma LowRank Guide (LogNormal variational family)
# ------------------------------------------------------------------------------


@dispatch(InverseGammaSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: InverseGammaSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for Inverse-Gamma parameters using LogNormal.

    Parameters
    ----------
    spec : InverseGammaSpec
        Inverse-Gamma parameter specification.
    guide : LowRankGuide
        Low-rank guide marker with rank.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank
    G = resolved_shape[-1] if resolved_shape else 1

    loc = numpyro.param(
        f"log_{spec.name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"log_{spec.name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"log_{spec.name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    transformed_dist = dist.TransformedDistribution(
        base, dist.transforms.ExpTransform()
    )
    return numpyro.sample(spec.name, transformed_dist)

@dispatch(HorseshoeHierarchicalSigmoidNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: HorseshoeHierarchicalSigmoidNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for horseshoe NCP z variable (gene-level).

    Targets the raw ``z`` variable with a Normal(loc, scale) guide.
    The constrained parameter is handled by numpyro.deterministic in the model.

    Parameters
    ----------
    spec : HorseshoeHierarchicalSigmoidNormalSpec
        Horseshoe gene-level parameter specification.
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
# Gene-level Horseshoe Sigmoid (for p, gate) — LowRank
# ------------------------------------------------------------------------------


@dispatch(HorseshoeHierarchicalSigmoidNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: HorseshoeHierarchicalSigmoidNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for horseshoe NCP z variable (gene-level).

    Parameters
    ----------
    spec : HorseshoeHierarchicalSigmoidNormalSpec
        Horseshoe gene-level parameter specification.
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
# Gene-level Horseshoe Exp (for phi under mean_odds) — MeanField
# ------------------------------------------------------------------------------


@dispatch(HorseshoeHierarchicalPositiveNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: HorseshoeHierarchicalPositiveNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for horseshoe NCP z variable (gene-level phi).

    Parameters
    ----------
    spec : HorseshoeHierarchicalPositiveNormalSpec
        Horseshoe gene-level phi specification.
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
# Gene-level Horseshoe Exp (for phi under mean_odds) — LowRank
# ------------------------------------------------------------------------------


@dispatch(HorseshoeHierarchicalPositiveNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: HorseshoeHierarchicalPositiveNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for horseshoe NCP z variable (gene-level phi).

    Parameters
    ----------
    spec : HorseshoeHierarchicalPositiveNormalSpec
        Horseshoe gene-level phi specification.
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
# Dataset-level Horseshoe Exp (for mu) — MeanField
# ------------------------------------------------------------------------------


@dispatch(HorseshoeDatasetPositiveNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: HorseshoeDatasetPositiveNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for horseshoe NCP z variable (dataset-level mu).

    Parameters
    ----------
    spec : HorseshoeDatasetPositiveNormalSpec
        Horseshoe dataset-level mu specification.
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
# Dataset-level Horseshoe Exp (for mu) — LowRank
# ------------------------------------------------------------------------------


@dispatch(HorseshoeDatasetPositiveNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: HorseshoeDatasetPositiveNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for horseshoe NCP z variable (dataset-level mu).

    Parameters
    ----------
    spec : HorseshoeDatasetPositiveNormalSpec
        Horseshoe dataset-level mu specification.
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
# Dataset-level Horseshoe Sigmoid (for p, gate) — MeanField
# ------------------------------------------------------------------------------


@dispatch(HorseshoeDatasetSigmoidNormalSpec, MeanFieldGuide, dict, object)
def setup_guide(
    spec: HorseshoeDatasetSigmoidNormalSpec,
    guide: MeanFieldGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """MeanField guide for horseshoe NCP z variable (dataset-level p/gate).

    Parameters
    ----------
    spec : HorseshoeDatasetSigmoidNormalSpec
        Horseshoe dataset-level p or gate specification.
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
# Dataset-level Horseshoe Sigmoid (for p, gate) — LowRank
# ------------------------------------------------------------------------------


@dispatch(HorseshoeDatasetSigmoidNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: HorseshoeDatasetSigmoidNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for horseshoe NCP z variable (dataset-level p/gate).

    Parameters
    ----------
    spec : HorseshoeDatasetSigmoidNormalSpec
        Horseshoe dataset-level p or gate specification.
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

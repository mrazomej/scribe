"""Standard low-rank guide dispatch implementations.

This module registers low-rank guide handlers for non-horseshoe parameter
specifications.
"""

from typing import TYPE_CHECKING, Dict

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch

from .parameter_specs import (
    LogNormalSpec,
    NormalWithTransformSpec,
    resolve_shape,
)
from ..components.guide_families import LowRankGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


@dispatch(LogNormalSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: LogNormalSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for LogNormal - captures gene correlations.

    Uses a low-rank multivariate normal in log-space with covariance:
        Σ = W @ W.T + diag(D)

    This captures correlations between genes with O(n × rank) memory
    instead of O(n²) for full covariance.

    Parameters
    ----------
    spec : LogNormalSpec
        Parameter specification.
    guide : LowRankGuide
        Low-rank guide marker with rank attribute.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value from low-rank variational distribution.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank

    # LowRankMVN uses the last dimension (n_genes) as the event
    # dimension. Any leading dimensions (components, datasets, or both)
    # become batch dimensions of the MVN and are promoted to event dims
    # via to_event() so the guide's event_dim matches the model's.
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


@dispatch(NormalWithTransformSpec, LowRankGuide, dict, object)
def setup_guide(
    spec: NormalWithTransformSpec,
    guide: LowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """LowRank guide for unconstrained parameters (Normal + Transform).

    Uses a low-rank multivariate normal wrapped with TransformedDistribution to
    handle the transform to constrained space. The covariance structure is:
        Σ = W @ W.T + diag(D)

    This matches the pattern used for mean-field guides, ensuring consistent
    Jacobian handling via TransformedDistribution.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (e.g., PositiveNormalSpec for positive
        parameters).
    guide : LowRankGuide
        Low-rank guide marker with rank attribute.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with guide hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space (transform applied via
        TransformedDistribution).

    Notes
    -----
    This handler enables low-rank guides for unconstrained models (e.g., models
    using PositiveNormalSpec, SigmoidNormalSpec). The low-rank MVN is defined in
    unconstrained space, then wrapped with TransformedDistribution to apply the
    transform (exp, sigmoid, etc.) and handle Jacobian automatically.

    Examples
    --------
    Works with PositiveNormalSpec for positive parameters:
        >>> spec = PositiveNormalSpec("r", ("n_genes",), (0.0, 1.0))
        >>> guide = LowRankGuide(rank=10)
        >>> setup_guide(spec, guide, {"n_genes": 100}, model_config)
        # Samples "r" from TransformedDistribution(LowRankMultivariateNormal, ExpTransform)
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    k = guide.rank

    # LowRankMVN uses the last dimension (n_genes) as the event
    # dimension. Any leading dimensions (components, datasets, or both)
    # become batch dimensions of the MVN and are promoted to event dims
    # via to_event() so the guide's event_dim matches the model's.
    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    loc = numpyro.param(
        f"{spec.name}_loc",
        jnp.zeros(resolved_shape) if resolved_shape else jnp.zeros(G),
    )
    W = numpyro.param(
        f"{spec.name}_W",
        (
            0.01 * jnp.ones((*resolved_shape, k))
            if resolved_shape
            else 0.01 * jnp.ones((G, k))
        ),
    )
    raw_diag = numpyro.param(
        f"{spec.name}_raw_diag",
        (
            -3.0 * jnp.ones(resolved_shape)
            if resolved_shape
            else -3.0 * jnp.ones(G)
        ),
    )

    D = jax.nn.softplus(raw_diag) + 1e-4
    base = dist.LowRankMultivariateNormal(loc=loc, cov_factor=W, cov_diag=D)
    transformed_dist = dist.TransformedDistribution(base, spec.transform)

    if n_batch_dims > 0:
        transformed_dist = transformed_dist.to_event(n_batch_dims)

    return numpyro.sample(spec.constrained_name, transformed_dist)

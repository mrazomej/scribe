"""Normalizing-flow guide dispatch implementations.

This module registers multipledispatch overloads for
``NormalizingFlowGuide`` — the per-parameter normalizing-flow
variational family.  Each overload wraps a ``FlowChain`` in a
``FlowDistribution`` and then applies the appropriate parameter
transform (ExpTransform for LogNormal, spec.transform for
NormalWithTransformSpec subclasses).
"""

from typing import TYPE_CHECKING, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.contrib.module import flax_module

from scribe.flows import FlowChain, FlowDistribution
from .parameter_specs import (
    LogNormalSpec,
    NormalWithTransformSpec,
    resolve_shape,
)
from ..components.guide_families import NormalizingFlowGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_flow_chain(
    guide: NormalizingFlowGuide,
    features: int,
    context_dim: int = 0,
) -> FlowChain:
    """Construct a ``FlowChain`` from a ``NormalizingFlowGuide`` marker.

    Parameters
    ----------
    guide : NormalizingFlowGuide
        Guide marker carrying flow hyperparameters.
    features : int
        Dimensionality of the flow (typically ``n_genes``).
    context_dim : int
        Continuous context dimensionality (used in joint guides).

    Returns
    -------
    FlowChain
        Configured Flax Linen module ready for ``flax_module`` registration.
    """
    return FlowChain(
        features=features,
        num_layers=guide.num_layers,
        flow_type=guide.flow_type,
        hidden_dims=list(guide.hidden_dims),
        activation=guide.activation,
        n_bins=guide.n_bins,
        context_dim=context_dim,
    )


def _build_flow_distribution(
    flow_fn,
    base: dist.Distribution,
    spec,
    is_log_normal: bool,
    n_batch_dims: int,
) -> dist.Distribution:
    """Create a ``TransformedDistribution`` wrapping a ``FlowDistribution``.

    Parameters
    ----------
    flow_fn : callable
        Registered ``flax_module`` callable ``(x, reverse=bool) -> (y, log_det)``.
    base : dist.Distribution
        Base distribution (typically diagonal Normal).
    spec : ParamSpec
        Parameter specification carrying the constraint transform.
    is_log_normal : bool
        If True, apply ``ExpTransform`` (for ``LogNormalSpec``).
    n_batch_dims : int
        Number of leading batch dimensions to promote to event dims.

    Returns
    -------
    dist.Distribution
        Distribution ready for ``numpyro.sample``.
    """
    flow_dist = FlowDistribution(flow_fn, base)
    if n_batch_dims > 0:
        flow_dist = flow_dist.to_event(n_batch_dims)

    if is_log_normal:
        transform = dist.transforms.ExpTransform()
    else:
        transform = spec.transform

    return dist.TransformedDistribution(flow_dist, transform)


# ---------------------------------------------------------------------------
# LogNormalSpec + NormalizingFlowGuide
# ---------------------------------------------------------------------------


@dispatch(LogNormalSpec, NormalizingFlowGuide, dict, object)
def setup_guide(
    spec: LogNormalSpec,
    guide: NormalizingFlowGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """NormalizingFlow guide for LogNormal parameters.

    Registers a ``FlowChain`` via ``flax_module`` and wraps it in a
    ``FlowDistribution`` + ``ExpTransform``.  The flow operates in
    unconstrained space with a standard Normal(0, 1) base — the flow
    itself is expressive enough to learn any target density, so a
    learnable base would be redundant.

    Parameters
    ----------
    spec : LogNormalSpec
        Parameter specification.
    guide : NormalizingFlowGuide
        Flow guide marker with flow hyperparameters.
    dims : Dict[str, int]
        Dimension sizes (must contain keys referenced by ``spec.shape_dims``).
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained (positive) space.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    # The flow treats the last axis as the event dimension (n_genes);
    # leading axes are batch dimensions promoted to event dims.
    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    # Build and register the flow as a Flax module
    flow_chain = _build_flow_chain(guide, features=G)
    flow_fn = flax_module(
        f"flow_{spec.name}",
        flow_chain,
        input_shape=(G,),
    )

    # Standard Normal(0, I) base — the flow is expressive enough
    base = dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1)

    transformed = _build_flow_distribution(
        flow_fn,
        base,
        spec,
        is_log_normal=True,
        n_batch_dims=n_batch_dims,
    )

    return numpyro.sample(spec.name, transformed)


# ---------------------------------------------------------------------------
# NormalWithTransformSpec + NormalizingFlowGuide
# ---------------------------------------------------------------------------


@dispatch(NormalWithTransformSpec, NormalizingFlowGuide, dict, object)
def setup_guide(
    spec: NormalWithTransformSpec,
    guide: NormalizingFlowGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    **kwargs,
) -> jnp.ndarray:
    """NormalizingFlow guide for transformed-Normal parameters.

    Works for ``PositiveNormalSpec``, ``SigmoidNormalSpec``,
    ``SoftplusNormalSpec``, etc. — any subclass of
    ``NormalWithTransformSpec``.  The flow operates in unconstrained
    space with a standard Normal(0, 1) base; ``spec.transform`` maps
    to constrained space and the Jacobian is handled automatically by
    ``TransformedDistribution``.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification (or subclass).
    guide : NormalizingFlowGuide
        Flow guide marker with flow hyperparameters.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space.
    """
    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    n_batch_dims = len(resolved_shape) - 1
    G = resolved_shape[-1] if resolved_shape else 1

    # Build and register the flow
    flow_chain = _build_flow_chain(guide, features=G)
    flow_fn = flax_module(
        f"flow_{spec.name}",
        flow_chain,
        input_shape=(G,),
    )

    # Standard Normal(0, I) base — the flow is expressive enough
    base = dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1)

    transformed = _build_flow_distribution(
        flow_fn,
        base,
        spec,
        is_log_normal=False,
        n_batch_dims=n_batch_dims,
    )

    return numpyro.sample(spec.constrained_name, transformed)

"""Normalizing-flow guide dispatch implementations.

This module registers multipledispatch overloads for
``NormalizingFlowGuide`` — the per-parameter normalizing-flow
variational family.  Each overload wraps a ``FlowChain`` in a
``FlowDistribution`` and then applies the appropriate constraint
transform.  For ``LogNormalSpec`` the transform is resolved from
``model_config.positive_transform`` (``SoftplusTransform`` by default);
for ``NormalWithTransformSpec`` subclasses it uses ``spec.transform``.

When a parameter has leading batch axes (``is_mixture=True`` and / or
``is_dataset=True``), the flow is wrapped in a
``ComponentFlowDistribution`` that creates per-index flows according
to the ``mixture_strategy`` setting on the guide marker:

* ``"independent"`` — separate ``FlowChain`` per index
  (most expressive, K× parameters).
* ``"shared"`` — single ``FlowChain`` conditioned on a one-hot
  index vector (parameter-efficient, shared backbone).
"""

from typing import TYPE_CHECKING, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.contrib.module import flax_module

from scribe.flows import ComponentFlowDistribution, FlowChain, FlowDistribution
from .parameter_specs import (
    LogNormalSpec,
    NormalWithTransformSpec,
    resolve_shape,
)
from ..components.guide_families import NormalizingFlowGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _resolve_positive_transform(model_config) -> dist.transforms.Transform:
    """Return the positive-constraint transform from ``model_config``.

    Reads ``model_config.positive_transform`` (``"softplus"`` or ``"exp"``)
    and returns the matching NumPyro transform.  Defaults to SoftplusTransform
    when the attribute is missing.
    """
    _pt = getattr(model_config, "positive_transform", "softplus")
    if _pt == "softplus":
        return dist.transforms.SoftplusTransform()
    return dist.transforms.ExpTransform()


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
        zero_init_output=guide.zero_init_output,
        use_layer_norm=guide.use_layer_norm,
        use_residual=guide.use_residual,
    )


def _build_flow_distribution(
    flow_fn,
    base: dist.Distribution,
    transform: dist.transforms.Transform,
    n_batch_dims: int,
) -> dist.Distribution:
    """Create a ``TransformedDistribution`` wrapping a ``FlowDistribution``.

    Parameters
    ----------
    flow_fn : callable
        Registered ``flax_module`` callable ``(x, reverse=bool) -> (y, log_det)``.
    base : dist.Distribution
        Base distribution (typically diagonal Normal).
    transform : dist.transforms.Transform
        Constraint transform mapping unconstrained flow output to the
        parameter's support (e.g. ``SoftplusTransform``, ``SigmoidTransform``).
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

    return dist.TransformedDistribution(flow_dist, transform)


# ---------------------------------------------------------------------------
# Batch-axis (mixture / dataset) helpers
# ---------------------------------------------------------------------------


def _batch_axis_info(
    spec,
    dims: Dict[str, int],
) -> Tuple[Tuple[int, ...], List[str]]:
    """Determine leading batch sizes and axis labels for a parameter spec.

    Returns ``(batch_shape, axis_names)`` where ``batch_shape`` lists
    the sizes of leading batch dimensions in the order they appear
    in the resolved shape and ``axis_names`` gives each one a label.

    Parameters
    ----------
    spec : ParamSpec
        Parameter specification.
    dims : dict
        Dimension mapping.

    Returns
    -------
    batch_shape : tuple of int
        Sizes of leading batch axes (e.g. ``(K,)`` or ``(K, D)``).
    axis_names : list of str
        Parallel list of axis labels.
    """
    sizes: List[int] = []
    names: List[str] = []
    # Mixture is outermost (prepended last by resolve_shape)
    if spec.is_mixture and "n_components" in dims:
        sizes.append(dims["n_components"])
        names.append("component")
    # Dataset is next-innermost
    if spec.is_dataset and "n_datasets" in dims:
        sizes.append(dims["n_datasets"])
        names.append("dataset")
    return tuple(sizes), names


def _build_batch_flow_dist(
    guide: NormalizingFlowGuide,
    spec,
    G: int,
    batch_shape: Tuple[int, ...],
    axis_names: List[str],
    name_prefix: str,
) -> dist.Distribution:
    """Recursively build (possibly nested) ``ComponentFlowDistribution``.

    Handles arbitrary depth of leading batch axes.  At each level the
    strategy (independent vs shared) is chosen from the guide marker.

    Parameters
    ----------
    guide : NormalizingFlowGuide
        Guide marker.
    spec : ParamSpec
        Parameter specification (used only for axis info).
    G : int
        Gene / trailing event dimension size.
    batch_shape : tuple of int
        Remaining batch sizes to expand (e.g. ``(K, D)``).
    axis_names : list of str
        Parallel labels for each element of *batch_shape*.
    name_prefix : str
        Prefix for ``flax_module`` registration.

    Returns
    -------
    dist.Distribution
        Leaf ``FlowDistribution`` (no batch) or nested
        ``ComponentFlowDistribution``.
    """
    if not batch_shape:
        # Leaf: single flow, no batch axes
        chain = _build_flow_chain(guide, features=G)
        fn = flax_module(name_prefix, chain, input_shape=(G,))
        base = dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1)
        return FlowDistribution(fn, base)

    N = batch_shape[0]
    remaining_shape = batch_shape[1:]
    remaining_names = axis_names[1:]
    current_axis = axis_names[0]

    if guide.mixture_strategy == "independent":
        component_dists = []
        for i in range(N):
            inner = _build_batch_flow_dist(
                guide,
                spec,
                G,
                remaining_shape,
                remaining_names,
                name_prefix=f"{name_prefix}_idx{i}",
            )
            component_dists.append(inner)
        return ComponentFlowDistribution(component_dists, axis_name=current_axis)

    # --- shared strategy ---
    # One FlowChain conditioned on a one-hot vector for every batch
    # axis.  Total context_dim = sum of all batch_shape entries.
    total_context = sum(batch_shape)
    return _build_shared_flow_dist(
        guide,
        G,
        batch_shape,
        axis_names,
        name_prefix,
        total_context,
    )


def _build_shared_flow_dist(
    guide: NormalizingFlowGuide,
    G: int,
    batch_shape: Tuple[int, ...],
    axis_names: List[str],
    name_prefix: str,
    total_context: int,
) -> dist.Distribution:
    """Build a nested ``ComponentFlowDistribution`` that shares one FlowChain.

    A single Flax module is registered; each leaf distribution is a
    closure that binds the concatenated one-hot vectors for all batch
    axes so ``get_component(k)`` returns a ready-to-use distribution.

    Parameters
    ----------
    guide : NormalizingFlowGuide
        Guide marker.
    G : int
        Trailing event dimension.
    batch_shape : tuple of int
        Sizes of batch axes to expand.
    axis_names : list of str
        Labels for each batch axis.
    name_prefix : str
        Flax module registration name.
    total_context : int
        Sum of all batch axis sizes (one-hot encoding).

    Returns
    -------
    dist.Distribution
        Nested ``ComponentFlowDistribution`` sharing one FlowChain.
    """
    chain = _build_flow_chain(guide, features=G, context_dim=total_context)
    fn = flax_module(
        name_prefix,
        chain,
        input_shape=(G,),
        context=jnp.zeros(total_context),
    )
    base_factory = lambda: dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1)

    def _recurse(level: int, partial_ctx: List[jnp.ndarray]):
        """Build distributions recursively, binding one-hot context per level."""
        if level == len(batch_shape):
            # Leaf: create conditioned FlowDistribution
            ctx = jnp.concatenate(partial_ctx)

            def conditioned(x, reverse=False, _ctx=ctx):
                return fn(x, reverse=reverse, context=_ctx)

            return FlowDistribution(conditioned, base_factory())

        N = batch_shape[level]
        component_dists = []
        for i in range(N):
            oh = jax.nn.one_hot(i, N)
            component_dists.append(
                _recurse(level + 1, partial_ctx + [oh])
            )
        return ComponentFlowDistribution(
            component_dists, axis_name=axis_names[level]
        )

    return _recurse(0, [])


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
    ``FlowDistribution`` + the positive-constraint transform resolved from
    ``model_config.positive_transform`` (``SoftplusTransform`` by default).
    The flow operates in unconstrained space with a standard Normal(0, 1) base.

    For mixture / multi-dataset parameters the flow is expanded into
    a ``ComponentFlowDistribution`` according to ``guide.mixture_strategy``.

    Parameters
    ----------
    spec : LogNormalSpec
        Parameter specification.
    guide : NormalizingFlowGuide
        Flow guide marker with flow hyperparameters.
    dims : Dict[str, int]
        Dimension sizes (must contain keys referenced by ``spec.shape_dims``).
    model_config : ModelConfig
        Model configuration (``positive_transform`` controls the constraint).

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained (positive) space.
    """
    pos_transform = _resolve_positive_transform(model_config)

    resolved_shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )
    G = resolved_shape[-1] if resolved_shape else 1
    batch_shape, axis_names = _batch_axis_info(spec, dims)

    # --- Mixture / dataset: per-index flows via ComponentFlowDistribution ---
    if batch_shape:
        flow_dist = _build_batch_flow_dist(
            guide, spec, G, batch_shape, axis_names,
            name_prefix=f"flow_{spec.name}",
        )
        transformed = dist.TransformedDistribution(flow_dist, pos_transform)
        return numpyro.sample(spec.name, transformed)

    # --- Standard (no batch axes) ---
    n_batch_dims = len(resolved_shape) - 1
    flow_chain = _build_flow_chain(guide, features=G)
    flow_fn = flax_module(f"flow_{spec.name}", flow_chain, input_shape=(G,))
    base = dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1)
    transformed = _build_flow_distribution(
        flow_fn, base, pos_transform, n_batch_dims=n_batch_dims,
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

    For mixture / multi-dataset parameters the flow is expanded into
    a ``ComponentFlowDistribution`` according to ``guide.mixture_strategy``.

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
    G = resolved_shape[-1] if resolved_shape else 1
    batch_shape, axis_names = _batch_axis_info(spec, dims)

    # --- Mixture / dataset: per-index flows via ComponentFlowDistribution ---
    if batch_shape:
        flow_dist = _build_batch_flow_dist(
            guide, spec, G, batch_shape, axis_names,
            name_prefix=f"flow_{spec.name}",
        )
        transformed = dist.TransformedDistribution(flow_dist, spec.transform)
        return numpyro.sample(spec.constrained_name, transformed)

    # --- Standard (no batch axes) ---
    n_batch_dims = len(resolved_shape) - 1
    flow_chain = _build_flow_chain(guide, features=G)
    flow_fn = flax_module(f"flow_{spec.name}", flow_chain, input_shape=(G,))
    base = dist.Normal(jnp.zeros(G), jnp.ones(G)).to_event(1)
    transformed = _build_flow_distribution(
        flow_fn, base, spec.transform, n_batch_dims=n_batch_dims,
    )
    return numpyro.sample(spec.constrained_name, transformed)

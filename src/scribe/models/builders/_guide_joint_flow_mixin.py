"""Joint normalizing-flow guide implementation.

Implements the chain-rule decomposition
``q(θ₁, θ₂, ...) = q(θ₁) × q(θ₂|θ₁) × q(θ₃|θ₁,θ₂) × ...``
using normalizing flows.  Each conditional ``q(θ_i | θ_{<i})`` is a
full ``FlowChain`` that receives the concatenated unconstrained samples
of all previous parameters as a continuous *context* vector.

When ``dense_params`` is set on the marker, only the dense subset goes
through the flow chain.  Non-dense parameters receive diagonal-Normal
treatment with learned regression on the dense-flow residuals and a
per-gene autoregressive chain among themselves — identical to the
pattern in ``_guide_structured_joint_mixin``.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module

from scribe.flows import FlowChain, FlowDistribution
from .parameter_specs import (
    NormalWithTransformSpec,
    resolve_shape,
)
from ._guide_joint_mixin import (
    _is_batch_prefix,
    _is_joint_ncp_spec,
    _select_reference_batch_shape,
)
from ..components.guide_families import JointNormalizingFlowGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


# ======================================================================
# Flow helpers
# ======================================================================


def _build_flow_chain_for_joint(
    guide: JointNormalizingFlowGuide,
    features: int,
    context_dim: int = 0,
) -> FlowChain:
    """Construct a ``FlowChain`` for one block in the joint flow chain.

    Parameters
    ----------
    guide : JointNormalizingFlowGuide
        Marker carrying flow hyperparameters.
    features : int
        Dimensionality of this parameter block.
    context_dim : int
        Total context dimensionality from all previously sampled blocks.

    Returns
    -------
    FlowChain
        Configured Flax module.
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


def _register_flow(
    name: str,
    flow_chain: FlowChain,
    features: int,
    context_dim: int = 0,
):
    """Register a FlowChain via ``flax_module``.

    Handles the subtlety that ``FlowChain`` with ``context_dim > 0``
    needs a dummy context vector at init time so the conditioner Dense
    layers get the correct input shape.

    Parameters
    ----------
    name : str
        Module name for ``numpyro.param`` registration.
    flow_chain : FlowChain
        The chain to register.
    features : int
        Dimensionality of the flow.
    context_dim : int
        Context dimensionality (0 = no context).

    Returns
    -------
    callable
        ``(x, reverse=bool, context=None) -> (y, log_det)``
    """
    # flax_module passes **kwargs to both init and apply.
    # When context_dim > 0, provide a dummy context at init time.
    init_kwargs = {}
    if context_dim > 0:
        init_kwargs["context"] = jnp.zeros(context_dim)

    return flax_module(
        name,
        flow_chain,
        input_shape=(features,),
        **init_kwargs,
    )


def _sample_flow_spec(
    flow_fn,
    features: int,
    spec: NormalWithTransformSpec,
    is_scalar_in_joint: bool,
    n_batch_dims: int,
    context: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a parameter from a flow distribution; return both spaces.

    Uses a standard Normal(0, I) base — the flow is expressive enough
    to learn any target density.  Handles scalar-in-joint expansion /
    collapse and NCP (horseshoe/NEG) specs.

    Parameters
    ----------
    flow_fn : callable
        Registered ``flax_module`` callable.
    features : int
        Dimensionality of the flow (i.e. the trailing event dim).
    spec : NormalWithTransformSpec
        Parameter specification.
    is_scalar_in_joint : bool
        If True, the parameter is scalar expanded with trailing dim of 1.
    n_batch_dims : int
        Leading batch dims to promote to event dims.
    context : jnp.ndarray, optional
        Continuous context from previously sampled parameters.

    Returns
    -------
    constrained : jnp.ndarray
        Sample in constrained space.
    unconstrained : jnp.ndarray
        Sample in unconstrained space (expanded for scalars).
    """
    # Build a closure that bakes in the context vector
    if context is not None:

        def _conditioned_flow(x, reverse=False):
            return flow_fn(x, reverse=reverse, context=context)

        active_flow = _conditioned_flow
    else:
        active_flow = flow_fn

    # Standard Normal(0, I) base
    base = dist.Normal(jnp.zeros(features), jnp.ones(features)).to_event(1)
    flow_dist = FlowDistribution(active_flow, base)

    if n_batch_dims > 0:
        flow_dist = flow_dist.to_event(n_batch_dims)

    if is_scalar_in_joint:
        # Scalar expanded to (1,).  Sample the raw 1-d vector, squeeze to
        # scalar, apply transform, register constrained deterministic.
        if _is_joint_ncp_spec(spec):
            raw = numpyro.sample(spec.raw_name, flow_dist)
            unconstrained_scalar = raw[..., 0]
            constrained = spec.transform(unconstrained_scalar)
            numpyro.deterministic(spec.constrained_name, constrained)
        else:
            raw = numpyro.sample(
                f"_jflow_{spec.name}_raw",
                flow_dist,
            )
            unconstrained_scalar = raw[..., 0]
            constrained = spec.transform(unconstrained_scalar)
            numpyro.deterministic(spec.constrained_name, constrained)
        return constrained, raw

    # Gene-specific parameter — full-dimensional flow
    if _is_joint_ncp_spec(spec):
        unconstrained = numpyro.sample(spec.raw_name, flow_dist)
        constrained = spec.transform(unconstrained)
        numpyro.deterministic(spec.constrained_name, constrained)
    else:
        transformed = dist.TransformedDistribution(flow_dist, spec.transform)
        constrained = numpyro.sample(spec.constrained_name, transformed)
        unconstrained = spec.transform.inv(constrained)

    return constrained, unconstrained


# ======================================================================
# Scalar-in-joint fallback
# ======================================================================


def _sample_scalar_in_joint(
    spec: NormalWithTransformSpec,
    guide: JointNormalizingFlowGuide,
    context: jnp.ndarray,
    context_dim: int,
    n_batch_dims: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a scalar parameter in a joint group without a flow.

    Coupling flows cannot operate on a single feature.  Instead, we use
    a Normal distribution with learnable ``loc`` / ``scale``.  When
    ``context`` is available (i.e., this is not the first block), a
    learned linear regression from the context adjusts the loc, giving
    the scalar a dependency on previously sampled parameters.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification.
    guide : JointNormalizingFlowGuide
        Guide marker (used for naming).
    context : jnp.ndarray or None
        Concatenated unconstrained samples of previous blocks.
    context_dim : int
        Dimensionality of the context vector.
    n_batch_dims : int
        Number of batch dimensions.

    Returns
    -------
    constrained : jnp.ndarray
        Sample in constrained space (scalar).
    unconstrained : jnp.ndarray
        Sample in unconstrained space (shape ``(1,)``).
    """
    prefix = f"joint_flow_{guide.group}_{spec.name}"
    loc = numpyro.param(f"{prefix}_scalar_loc", jnp.zeros(()))
    raw_scale = numpyro.param(f"{prefix}_scalar_raw_scale", jnp.zeros(()))
    scale = jax.nn.softplus(raw_scale) + 1e-4

    # If context is available, shift loc via linear regression
    if context is not None and context_dim > 0:
        alpha = numpyro.param(
            f"{prefix}_scalar_ctx_alpha",
            jnp.zeros(context_dim),
        )
        loc = loc + jnp.dot(context, alpha)

    base_d = dist.Normal(loc, scale)
    if n_batch_dims > 0:
        base_d = base_d.to_event(n_batch_dims)

    if _is_joint_ncp_spec(spec):
        unconstrained = numpyro.sample(spec.raw_name, base_d)
        constrained = spec.transform(unconstrained)
        numpyro.deterministic(spec.constrained_name, constrained)
    else:
        transformed = dist.TransformedDistribution(base_d, spec.transform)
        constrained = numpyro.sample(spec.constrained_name, transformed)
        unconstrained = spec.transform.inv(constrained)

    # Return unconstrained expanded to (1,) for context concatenation
    return constrained, unconstrained[..., None]


# ======================================================================
# Nondense helpers
# ======================================================================


def _can_regress_on(
    nondense_is_scalar: bool,
    dense_is_scalar: bool,
) -> bool:
    """Whether a nondense spec can regress on a dense spec.

    Gene-local regression is skipped when the nondense param is scalar
    but the dense param is gene-specific (would introduce a gene axis
    the scalar does not have).
    """
    if nondense_is_scalar and not dense_is_scalar:
        return False
    return True


def _reduce_dense_residual(
    residual: jnp.ndarray,
    nondense_batch: Tuple[int, ...],
    dense_batch: Tuple[int, ...],
) -> jnp.ndarray:
    """Mean-reduce extra batch dims so residual matches nondense batch."""
    extra = len(dense_batch) - len(nondense_batch)
    if extra <= 0:
        return residual
    for _ in range(extra):
        residual = jnp.mean(residual, axis=len(nondense_batch), keepdims=False)
    return residual


# ======================================================================
# Public entry point
# ======================================================================


def setup_joint_flow_guide(
    specs: List["NormalWithTransformSpec"],
    guide: JointNormalizingFlowGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> List[jnp.ndarray]:
    """Setup a joint normalizing-flow guide over multiple parameters.

    When ``guide.dense_params`` is None, **all** specs go through the
    flow chain with context-conditioned conditionals.  When set, only
    the named dense subset uses flows; non-dense specs get diagonal
    Normals with learned regression on the dense-flow residuals and a
    per-gene autoregressive chain.

    Parameters
    ----------
    specs : List[NormalWithTransformSpec]
        Ordered parameter specifications in the joint group.
    guide : JointNormalizingFlowGuide
        Guide marker with flow and group configuration.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    List[jnp.ndarray]
        Constrained samples for every spec, preserving input order.
    """
    # ------------------------------------------------------------------
    # Partition specs into flow (dense) vs diagonal (nondense)
    # ------------------------------------------------------------------
    if guide.dense_params is not None:
        dense_set = set(guide.dense_params)
        flow_specs = [s for s in specs if s.name in dense_set]
        nondense_specs = [s for s in specs if s.name not in dense_set]
    else:
        flow_specs = list(specs)
        nondense_specs = []

    # ------------------------------------------------------------------
    # Shape resolution for all specs
    # ------------------------------------------------------------------
    all_expanded: Dict[str, Tuple[int, ...]] = {}
    all_scalar_flags: Dict[str, bool] = {}

    for spec in specs:
        rs = resolve_shape(
            spec.shape_dims,
            dims,
            is_mixture=spec.is_mixture,
            is_dataset=spec.is_dataset,
        )
        if not spec.is_gene_specific:
            expanded = (*rs, 1) if rs else (1,)
            all_scalar_flags[spec.name] = True
        else:
            expanded = rs
            all_scalar_flags[spec.name] = False
        all_expanded[spec.name] = expanded

    # Validate batch compatibility for flow specs
    flow_batch_shapes = [all_expanded[s.name][:-1] for s in flow_specs]
    if flow_batch_shapes:
        ref_batch = _select_reference_batch_shape(flow_batch_shapes)
        for i, bs in enumerate(flow_batch_shapes):
            if not _is_batch_prefix(bs, ref_batch):
                raise ValueError(
                    f"Joint flow group '{guide.group}': spec "
                    f"'{flow_specs[i].name}' batch shape {bs} is "
                    f"incompatible with reference {ref_batch}."
                )
        # Batch-rank ordering guard
        for i in range(1, len(flow_batch_shapes)):
            if not _is_batch_prefix(
                flow_batch_shapes[i - 1], flow_batch_shapes[i]
            ):
                raise ValueError(
                    f"Joint flow group '{guide.group}': specs must be "
                    f"ordered from shorter to longer batch shape. "
                    f"'{flow_specs[i - 1].name}' {flow_batch_shapes[i - 1]} "
                    f"-> '{flow_specs[i].name}' {flow_batch_shapes[i]}."
                )

    # ==================================================================
    # Phase 1: Flow chain
    # ==================================================================

    flow_constrained: Dict[str, jnp.ndarray] = {}
    flow_unconstrained: Dict[str, jnp.ndarray] = {}
    # Flat list of unconstrained samples for building context vectors
    flow_unconstrained_list: List[jnp.ndarray] = []

    for idx, spec in enumerate(flow_specs):
        exp_shape = all_expanded[spec.name]
        is_scalar = all_scalar_flags[spec.name]
        G_i = exp_shape[-1]
        n_batch_dims_i = len(exp_shape) - 1

        # Context = cumulative dim of all previously sampled flow blocks
        context_dim = sum(
            all_expanded[flow_specs[j].name][-1] for j in range(idx)
        )

        # Build context from previously sampled unconstrained values
        context = None
        if flow_unconstrained_list:
            context = jnp.concatenate(flow_unconstrained_list, axis=-1)

        # Coupling flows cannot operate on features=1 (binary mask
        # would give one empty partition).  For scalars expanded to
        # dim 1, use a context-conditioned Normal instead.
        if G_i <= 1:
            constrained, unconstrained = _sample_scalar_in_joint(
                spec,
                guide,
                context,
                context_dim,
                n_batch_dims_i,
            )
            flow_constrained[spec.name] = constrained
            if (
                is_scalar
                and unconstrained.ndim > 0
                and unconstrained.shape[-1] != 1
            ):
                unconstrained = unconstrained[..., None]
            flow_unconstrained[spec.name] = unconstrained
            flow_unconstrained_list.append(unconstrained)
            continue

        # Build and register the flow module
        flow_chain = _build_flow_chain_for_joint(
            guide,
            features=G_i,
            context_dim=context_dim,
        )
        module_name = f"joint_flow_{guide.group}_{spec.name}"
        flow_fn = _register_flow(
            module_name,
            flow_chain,
            features=G_i,
            context_dim=context_dim,
        )

        constrained, unconstrained = _sample_flow_spec(
            flow_fn,
            G_i,
            spec,
            is_scalar,
            n_batch_dims_i,
            context=context,
        )

        flow_constrained[spec.name] = constrained

        # Ensure unconstrained has the expanded trailing dim for scalars
        if (
            is_scalar
            and unconstrained.ndim > 0
            and unconstrained.shape[-1] != 1
        ):
            unconstrained = unconstrained[..., None]

        flow_unconstrained[spec.name] = unconstrained
        flow_unconstrained_list.append(unconstrained)

    # ==================================================================
    # Phase 2: Nondense block (diagonal + regression)
    # ==================================================================

    nondense_constrained: Dict[str, jnp.ndarray] = {}
    nondense_unconstrained: Dict[str, jnp.ndarray] = {}
    nondense_cond_locs: Dict[str, jnp.ndarray] = {}
    nondense_order: List[str] = []

    for spec in nondense_specs:
        exp_shape = all_expanded[spec.name]
        is_scalar = all_scalar_flags[spec.name]
        nd_batch = exp_shape[:-1]

        # Learnable loc and diagonal scale
        loc_s = numpyro.param(
            f"joint_flow_{guide.group}_{spec.name}_loc",
            jnp.zeros(exp_shape),
        )
        raw_diag_s = numpyro.param(
            f"joint_flow_{guide.group}_{spec.name}_raw_diag",
            -3.0 * jnp.ones(exp_shape),
        )
        sigma_s = jnp.sqrt(jax.nn.softplus(raw_diag_s) + 1e-4)

        # ----- Dense regression: alpha coefficients ----- #
        dense_shift = jnp.zeros_like(loc_s)
        for d_spec in flow_specs:
            d_is_scalar = all_scalar_flags[d_spec.name]
            if not _can_regress_on(is_scalar, d_is_scalar):
                continue

            alpha_d = numpyro.param(
                f"joint_flow_{guide.group}_{spec.name}_alpha_{d_spec.name}",
                jnp.zeros(exp_shape),
            )

            # With a standard Normal(0, I) base, the unconstrained
            # sample IS the residual (deviation from the base mean of 0)
            d_residual = flow_unconstrained[d_spec.name]

            d_batch = all_expanded[d_spec.name][:-1]
            d_residual = _reduce_dense_residual(d_residual, nd_batch, d_batch)
            dense_shift = dense_shift + alpha_d * d_residual

        # ----- Nondense autoregressive chain: beta coefficients ----- #
        nondense_shift = jnp.zeros_like(loc_s)
        for earlier_name in nondense_order:
            earlier_scalar = all_scalar_flags[earlier_name]
            # Only chain gene-to-gene or scalar-to-scalar
            if is_scalar and not earlier_scalar:
                continue
            if not is_scalar and earlier_scalar:
                continue

            beta_j = numpyro.param(
                f"joint_flow_{guide.group}_{spec.name}_beta_{earlier_name}",
                jnp.zeros(exp_shape),
            )
            earlier_residual = (
                nondense_unconstrained[earlier_name]
                - nondense_cond_locs[earlier_name]
            )
            e_batch = all_expanded[earlier_name][:-1]
            earlier_residual = _reduce_dense_residual(
                earlier_residual, nd_batch, e_batch
            )
            nondense_shift = nondense_shift + beta_j * earlier_residual

        # ----- Conditional distribution and sampling ----- #
        loc_cond = loc_s + dense_shift + nondense_shift
        nondense_cond_locs[spec.name] = loc_cond
        n_batch_dims_s = len(exp_shape) - 1

        if _is_joint_ncp_spec(spec):
            if is_scalar:
                base_d = dist.Normal(loc_cond[..., 0], sigma_s[..., 0])
                if n_batch_dims_s > 0:
                    base_d = base_d.to_event(n_batch_dims_s)
            else:
                base_d = dist.Independent(
                    dist.Normal(loc_cond, sigma_s),
                    reinterpreted_batch_ndims=1,
                )
                if n_batch_dims_s > 0:
                    base_d = base_d.to_event(n_batch_dims_s)
            unconstrained = numpyro.sample(spec.raw_name, base_d)
            constrained = spec.transform(unconstrained)
            numpyro.deterministic(spec.constrained_name, constrained)
        else:
            if is_scalar:
                base_d = dist.Normal(loc_cond[..., 0], sigma_s[..., 0])
                if n_batch_dims_s > 0:
                    base_d = base_d.to_event(n_batch_dims_s)
            else:
                base_d = dist.Independent(
                    dist.Normal(loc_cond, sigma_s),
                    reinterpreted_batch_ndims=1,
                )
                if n_batch_dims_s > 0:
                    base_d = base_d.to_event(n_batch_dims_s)
            transformed = dist.TransformedDistribution(base_d, spec.transform)
            constrained = numpyro.sample(spec.constrained_name, transformed)
            unconstrained = spec.transform.inv(constrained)

        nondense_constrained[spec.name] = constrained
        if is_scalar:
            unconstrained = unconstrained[..., None]
        nondense_unconstrained[spec.name] = unconstrained
        nondense_order.append(spec.name)

    # ==================================================================
    # Assemble results in the original spec order
    # ==================================================================
    results: List[jnp.ndarray] = []
    for spec in specs:
        if spec.name in flow_constrained:
            results.append(flow_constrained[spec.name])
        else:
            results.append(nondense_constrained[spec.name])
    return results

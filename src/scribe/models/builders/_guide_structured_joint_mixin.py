"""Structured joint low-rank guide.

Dense parameters get full cross-gene low-rank coupling (Woodbury chain).
Non-dense parameters get gene-local conditioning on dense params plus a
per-gene autoregressive chain among themselves (equivalent to a per-gene
lower-triangular Cholesky block).

The implementation reuses helpers from ``_guide_joint_mixin`` for the
dense block and adds per-gene regression / chain-rule logic for the
non-dense block.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .parameter_specs import (
    NormalWithTransformSpec,
    resolve_shape,
)
from ._guide_joint_mixin import (
    _build_base_distribution_for_joint_spec,
    _build_distribution_for_spec,
    _is_batch_prefix,
    _is_joint_ncp_spec,
    _select_reference_batch_shape,
    _woodbury_conditional_params,
)
from ..components.guide_families import JointLowRankGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


# ======================================================================
# Nondense helpers
# ======================================================================


def _can_regress_on(
    nondense_expanded: Tuple[int, ...],
    dense_expanded: Tuple[int, ...],
    nondense_is_scalar: bool,
    dense_is_scalar: bool,
) -> bool:
    """Determine whether a nondense spec can regress on a dense spec.

    Gene-local regression requires both to have a gene dimension, or both
    to be scalar.  When the nondense spec is scalar but the dense spec is
    gene-specific the regression would introduce a gene axis the scalar
    does not have, so it is skipped.

    Batch-shape mismatches (e.g. nondense ``(K, G)`` vs dense
    ``(K, D, G)``) are handled by reducing the extra dense batch dims
    before regression.

    Parameters
    ----------
    nondense_expanded : tuple
        Expanded shape of the nondense spec.
    dense_expanded : tuple
        Expanded shape of the dense spec.
    nondense_is_scalar : bool
        Whether the nondense spec is scalar-in-joint.
    dense_is_scalar : bool
        Whether the dense spec is scalar-in-joint.

    Returns
    -------
    bool
        True if per-gene regression is valid.
    """
    if nondense_is_scalar and not dense_is_scalar:
        return False
    return True


def _reduce_dense_residual(
    residual: jnp.ndarray,
    nondense_batch: Tuple[int, ...],
    dense_batch: Tuple[int, ...],
) -> jnp.ndarray:
    """Mean-reduce extra batch dims in a dense residual.

    When the dense param has more batch dimensions than the nondense param
    (e.g. dense ``(K, D, G)`` vs nondense ``(K, G)``), average over the
    extra axes so the residual matches the nondense batch shape.

    Parameters
    ----------
    residual : jnp.ndarray
        Dense unconstrained residual ``(dense_unconstrained - dense_loc)``.
    nondense_batch : tuple
        Batch shape of the nondense spec (all dims except trailing gene).
    dense_batch : tuple
        Batch shape of the dense spec.

    Returns
    -------
    jnp.ndarray
        Residual with batch shape compatible with nondense_batch.
    """
    extra = len(dense_batch) - len(nondense_batch)
    if extra <= 0:
        return residual
    # Extra batch axes sit right after the shared prefix.  Each is one
    # axis that should be averaged out; they start at position
    # len(nondense_batch) in the residual (before the trailing gene dim).
    for _ in range(extra):
        residual = jnp.mean(residual, axis=len(nondense_batch), keepdims=False)
    return residual


# ======================================================================
# Public entry point
# ======================================================================


def setup_structured_joint_guide(
    specs: List["NormalWithTransformSpec"],
    guide: JointLowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> List[jnp.ndarray]:
    """Setup a structured joint guide with dense and nondense blocks.

    Dense parameters (listed in ``guide.dense_params``) are modeled by a
    full low-rank MVN with cross-gene coupling (Woodbury chain, identical
    to ``setup_joint_guide``).

    Non-dense parameters are sampled per-gene, conditioned on the dense
    unconstrained samples at the same gene via learned regression
    coefficients (``alpha``).  Non-dense parameters additionally form a
    per-gene autoregressive chain (``beta`` coefficients) so that each
    non-dense param at gene *i* can correlate with earlier non-dense
    params at the same gene.  This is equivalent to a per-gene
    lower-triangular Cholesky block among the non-dense params.

    Parameters
    ----------
    specs : List[NormalWithTransformSpec]
        All specs in the joint group (both dense and non-dense), in the
        order declared by the user.
    guide : JointLowRankGuide
        Guide marker carrying ``rank``, ``group``, and ``dense_params``.
    dims : Dict[str, int]
        Dimension sizes (n_genes, n_components, n_datasets, ...).
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    List[jnp.ndarray]
        Constrained samples for every spec, in the same order as *specs*.
    """
    k = guide.rank
    dense_set = set(guide.dense_params)

    # -----------------------------------------------------------------
    # Partition specs into dense and nondense, preserving declaration
    # order within each group.
    # -----------------------------------------------------------------
    dense_specs: List[NormalWithTransformSpec] = []
    nondense_specs: List[NormalWithTransformSpec] = []
    for s in specs:
        if s.name in dense_set:
            dense_specs.append(s)
        else:
            nondense_specs.append(s)

    # -----------------------------------------------------------------
    # Shape resolution for ALL specs (needed for the result array and
    # for nondense regression shapes).
    # -----------------------------------------------------------------
    all_resolved: Dict[str, Tuple[int, ...]] = {}
    all_expanded: Dict[str, Tuple[int, ...]] = {}
    all_scalar_flags: Dict[str, bool] = {}

    for spec in specs:
        rs = resolve_shape(
            spec.shape_dims,
            dims,
            is_mixture=spec.is_mixture,
            is_dataset=spec.is_dataset,
        )
        all_resolved[spec.name] = rs
        if not spec.is_gene_specific:
            expanded = (*rs, 1) if rs else (1,)
            all_scalar_flags[spec.name] = True
        else:
            expanded = rs
            all_scalar_flags[spec.name] = False
        all_expanded[spec.name] = expanded

    # =================================================================
    # Phase 1: Dense block — Woodbury chain
    # =================================================================

    # Validate batch compatibility for dense specs.
    dense_batch_shapes = [
        all_expanded[s.name][:-1] for s in dense_specs
    ]
    if dense_batch_shapes:
        dense_ref_batch = _select_reference_batch_shape(dense_batch_shapes)
        for i, bs in enumerate(dense_batch_shapes):
            if not _is_batch_prefix(bs, dense_ref_batch):
                raise ValueError(
                    f"Structured joint group '{guide.group}': dense spec "
                    f"'{dense_specs[i].name}' has batch shape {bs}, "
                    f"incompatible with reference {dense_ref_batch}."
                )

    # Order guard: batch ranks must not decrease within the dense chain.
    for i in range(1, len(dense_batch_shapes)):
        if not _is_batch_prefix(
            dense_batch_shapes[i - 1], dense_batch_shapes[i]
        ):
            raise ValueError(
                f"Structured joint group '{guide.group}': dense specs "
                f"must be ordered from shorter to longer batch shape. "
                f"'{dense_specs[i - 1].name}' {dense_batch_shapes[i - 1]} "
                f"-> '{dense_specs[i].name}' {dense_batch_shapes[i]}."
            )

    # Register variational params for dense specs and run Woodbury chain.
    dense_locs: Dict[str, jnp.ndarray] = {}
    dense_Ws: Dict[str, jnp.ndarray] = {}
    dense_Ds: Dict[str, jnp.ndarray] = {}
    dense_unconstrained: Dict[str, jnp.ndarray] = {}
    dense_constrained: Dict[str, jnp.ndarray] = {}

    _chain_locs: List[jnp.ndarray] = []
    _chain_Ws: List[jnp.ndarray] = []
    _chain_Ds: List[jnp.ndarray] = []
    _chain_unconstrained: List[jnp.ndarray] = []

    for idx, spec in enumerate(dense_specs):
        exp_shape = all_expanded[spec.name]
        is_scalar = all_scalar_flags[spec.name]

        loc_i = numpyro.param(
            f"joint_{guide.group}_{spec.name}_loc",
            jnp.zeros(exp_shape),
        )
        W_i = numpyro.param(
            f"joint_{guide.group}_{spec.name}_W",
            0.01 * jnp.ones((*exp_shape, k)),
        )
        raw_diag_i = numpyro.param(
            f"joint_{guide.group}_{spec.name}_raw_diag",
            -3.0 * jnp.ones(exp_shape),
        )
        D_i = jax.nn.softplus(raw_diag_i) + 1e-4

        dense_locs[spec.name] = loc_i
        dense_Ws[spec.name] = W_i
        dense_Ds[spec.name] = D_i
        _chain_locs.append(loc_i)
        _chain_Ws.append(W_i)
        _chain_Ds.append(D_i)

        n_batch_dims_i = len(exp_shape) - 1

        # Woodbury chain: condition on all previously sampled dense params
        cond_loc = loc_i
        cond_W = W_i
        cond_D = D_i
        for j in range(idx):
            cond_loc, cond_W, cond_D = _woodbury_conditional_params(
                W1=_chain_Ws[j],
                D1=_chain_Ds[j],
                W2=cond_W,
                D2=cond_D,
                loc1=_chain_locs[j],
                loc2=cond_loc,
                theta1_sample=_chain_unconstrained[j],
            )

        # Sample the dense parameter
        if _is_joint_ncp_spec(spec):
            base = _build_base_distribution_for_joint_spec(
                cond_loc, cond_W, cond_D, is_scalar, n_batch_dims_i
            )
            unconstrained = numpyro.sample(spec.raw_name, base)
            constrained = spec.transform(unconstrained)
            numpyro.deterministic(spec.constrained_name, constrained)
        else:
            transformed = _build_distribution_for_spec(
                cond_loc, cond_W, cond_D, spec, is_scalar, n_batch_dims_i
            )
            constrained = numpyro.sample(spec.constrained_name, transformed)
            unconstrained = spec.transform.inv(constrained)

        dense_constrained[spec.name] = constrained

        # Re-expand scalar samples so they have the trailing 1 dim for
        # consistent regression later.
        if is_scalar:
            unconstrained = unconstrained[..., None]

        dense_unconstrained[spec.name] = unconstrained
        _chain_unconstrained.append(unconstrained)

        # Update conditional W/loc for subsequent dense params in chain
        if idx > 0:
            _chain_Ws[idx] = cond_W
            _chain_locs[idx] = cond_loc

    # =================================================================
    # Phase 2: Nondense block — per-gene autoregressive
    # =================================================================

    nondense_unconstrained: Dict[str, jnp.ndarray] = {}
    nondense_constrained: Dict[str, jnp.ndarray] = {}
    nondense_cond_locs: Dict[str, jnp.ndarray] = {}
    # Track the order nondense specs are processed (for beta chain)
    nondense_order: List[str] = []

    for spec in nondense_specs:
        exp_shape = all_expanded[spec.name]
        is_scalar = all_scalar_flags[spec.name]
        nd_batch = exp_shape[:-1]

        # Register base variational params (loc + diagonal variance)
        loc_s = numpyro.param(
            f"joint_{guide.group}_{spec.name}_loc",
            jnp.zeros(exp_shape),
        )
        raw_diag_s = numpyro.param(
            f"joint_{guide.group}_{spec.name}_raw_diag",
            -3.0 * jnp.ones(exp_shape),
        )
        sigma_s = jnp.sqrt(jax.nn.softplus(raw_diag_s) + 1e-4)

        # ----- Dense regression: alpha coefficients ----- #
        dense_shift = jnp.zeros_like(loc_s)
        for d_spec in dense_specs:
            d_is_scalar = all_scalar_flags[d_spec.name]
            if not _can_regress_on(exp_shape, all_expanded[d_spec.name],
                                   is_scalar, d_is_scalar):
                continue

            # Determine the alpha shape: same as the nondense expanded
            # shape (since it is element-wise per gene).
            alpha_shape = exp_shape
            alpha_d = numpyro.param(
                f"joint_{guide.group}_{spec.name}_alpha_{d_spec.name}",
                jnp.zeros(alpha_shape),
            )

            # Dense residual in the expanded (unconstrained) space
            d_residual = (
                dense_unconstrained[d_spec.name]
                - dense_locs[d_spec.name]
            )

            # Reduce extra batch dims if the dense param is
            # dataset-specific but the nondense param is not.
            d_batch = all_expanded[d_spec.name][:-1]
            d_residual = _reduce_dense_residual(d_residual, nd_batch, d_batch)

            dense_shift = dense_shift + alpha_d * d_residual

        # ----- Nondense chain: beta coefficients ----- #
        nondense_shift = jnp.zeros_like(loc_s)
        for earlier_name in nondense_order:
            earlier_scalar = all_scalar_flags[earlier_name]
            earlier_expanded = all_expanded[earlier_name]

            # Only chain gene-to-gene or scalar-to-scalar
            if is_scalar and not earlier_scalar:
                continue
            if not is_scalar and earlier_scalar:
                continue

            beta_shape = exp_shape
            beta_j = numpyro.param(
                f"joint_{guide.group}_{spec.name}_beta_{earlier_name}",
                jnp.zeros(beta_shape),
            )

            earlier_residual = (
                nondense_unconstrained[earlier_name]
                - nondense_cond_locs[earlier_name]
            )

            # Reduce if batch shapes differ
            e_batch = earlier_expanded[:-1]
            earlier_residual = _reduce_dense_residual(
                earlier_residual, nd_batch, e_batch
            )

            nondense_shift = nondense_shift + beta_j * earlier_residual

        # ----- Conditional distribution and sampling ----- #
        loc_cond = loc_s + dense_shift + nondense_shift
        nondense_cond_locs[spec.name] = loc_cond

        n_batch_dims_s = len(exp_shape) - 1

        if _is_joint_ncp_spec(spec):
            # NCP raw sampling (horseshoe / NEG)
            if is_scalar:
                scalar_loc = loc_cond[..., 0]
                base = dist.Normal(scalar_loc, sigma_s[..., 0])
                if n_batch_dims_s > 0:
                    base = base.to_event(n_batch_dims_s)
            else:
                base = dist.Independent(
                    dist.Normal(loc_cond, sigma_s),
                    reinterpreted_batch_ndims=1,
                )
                if n_batch_dims_s > 0:
                    base = base.to_event(n_batch_dims_s)
            unconstrained = numpyro.sample(spec.raw_name, base)
            constrained = spec.transform(unconstrained)
            numpyro.deterministic(spec.constrained_name, constrained)
        else:
            if is_scalar:
                scalar_loc = loc_cond[..., 0]
                scalar_sigma = sigma_s[..., 0]
                base = dist.Normal(scalar_loc, scalar_sigma)
                if n_batch_dims_s > 0:
                    base = base.to_event(n_batch_dims_s)
            else:
                base = dist.Independent(
                    dist.Normal(loc_cond, sigma_s),
                    reinterpreted_batch_ndims=1,
                )
                if n_batch_dims_s > 0:
                    base = base.to_event(n_batch_dims_s)
            transformed = dist.TransformedDistribution(base, spec.transform)
            constrained = numpyro.sample(spec.constrained_name, transformed)
            unconstrained = spec.transform.inv(constrained)

        nondense_constrained[spec.name] = constrained

        # Re-expand scalar unconstrained for consistent beta chain
        if is_scalar:
            unconstrained = unconstrained[..., None]
        nondense_unconstrained[spec.name] = unconstrained
        nondense_order.append(spec.name)

    # =================================================================
    # Assemble results in the original spec order
    # =================================================================
    results: List[jnp.ndarray] = []
    for spec in specs:
        if spec.name in dense_constrained:
            results.append(dense_constrained[spec.name])
        else:
            results.append(nondense_constrained[spec.name])

    return results

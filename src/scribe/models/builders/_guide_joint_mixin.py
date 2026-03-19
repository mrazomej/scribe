"""Joint low-rank guide helpers.

This module contains Woodbury-based conditional utilities and the
`setup_joint_guide` implementation used by `GuideBuilder`.
"""

from typing import TYPE_CHECKING, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .parameter_specs import (
    HorseshoeDatasetPositiveNormalSpec,
    HorseshoeDatasetSigmoidNormalSpec,
    HorseshoeHierarchicalPositiveNormalSpec,
    HorseshoeHierarchicalSigmoidNormalSpec,
    NEGDatasetPositiveNormalSpec,
    NEGDatasetSigmoidNormalSpec,
    NEGHierarchicalPositiveNormalSpec,
    NEGHierarchicalSigmoidNormalSpec,
    NormalWithTransformSpec,
    resolve_shape,
)
from ..components.guide_families import JointLowRankGuide

if TYPE_CHECKING:
    from ..config import ModelConfig


def _is_batch_prefix(shorter: Tuple[int, ...], longer: Tuple[int, ...]) -> bool:
    """Return whether ``shorter`` matches the leading axes of ``longer``.

    Parameters
    ----------
    shorter : tuple of int
        Candidate prefix batch shape (for example ``()`` or ``(K,)``).
    longer : tuple of int
        Candidate reference batch shape (for example ``(K, D)``).

    Returns
    -------
    bool
        True when ``shorter`` is a valid prefix of ``longer``.
    """
    if len(shorter) > len(longer):
        return False
    return shorter == longer[: len(shorter)]


def _select_reference_batch_shape(
    batch_shapes: List[Tuple[int, ...]],
) -> Tuple[int, ...]:
    """Select the longest compatible batch shape in a joint group.

    Parameters
    ----------
    batch_shapes : list of tuple of int
        Batch shapes from each expanded joint specification
        (that is, all dimensions except the trailing gene axis).

    Returns
    -------
    tuple of int
        The reference batch shape with maximal rank.
    """
    return max(batch_shapes, key=len)


def _broadcast_to_batch_prefix(
    arr: jnp.ndarray,
    target_batch: Tuple[int, ...],
    keep_last_dims: int,
    name: str,
) -> jnp.ndarray:
    """Broadcast a tensor to ``target_batch`` using prefix-compatible semantics.

    Parameters
    ----------
    arr : jnp.ndarray
        Tensor to broadcast.
    target_batch : tuple of int
        Desired batch shape.
    keep_last_dims : int
        Number of trailing event/matrix dimensions to preserve.
    name : str
        Tensor name for validation errors.

    Returns
    -------
    jnp.ndarray
        Tensor with batch shape equal to ``target_batch``.
    """
    if keep_last_dims <= 0:
        raise ValueError("keep_last_dims must be positive")

    arr_batch = arr.shape[:-keep_last_dims]
    arr_tail = arr.shape[-keep_last_dims:]
    if len(arr_batch) > len(target_batch):
        raise ValueError(
            f"Tensor '{name}' has batch shape {arr_batch}, which has higher rank "
            f"than target batch shape {target_batch}."
        )
    if not _is_batch_prefix(arr_batch, target_batch):
        raise ValueError(
            f"Tensor '{name}' has incompatible batch shape {arr_batch}; expected "
            f"a prefix of target batch shape {target_batch}."
        )

    pad = (1,) * (len(target_batch) - len(arr_batch))
    reshaped = jnp.reshape(arr, arr_batch + pad + arr_tail)
    return jnp.broadcast_to(reshaped, target_batch + arr_tail)


def _woodbury_conditional_params(
    W1: jnp.ndarray,
    D1: jnp.ndarray,
    W2: jnp.ndarray,
    D2: jnp.ndarray,
    loc1: jnp.ndarray,
    loc2: jnp.ndarray,
    theta1_sample: jnp.ndarray,
) -> tuple:
    """Compute the conditional LowRankMVN parameters via Woodbury identity.

    Given a joint LowRankMVN over [theta_1, theta_2] with factor W = [W1; W2]
    and diagonal D = diag(D1, D2), computes the conditional distribution
    theta_2 | theta_1 using the Schur complement with Woodbury inversion.

    The conditional is itself a LowRankMVN:
        cov_factor = W2 @ L_M   (where M = (I + W1^T D1^{-1} W1)^{-1})
        cov_diag   = D2          (unchanged)
        loc        = loc2 + W2 @ M @ W1^T @ D1^{-1} @ (theta1 - loc1)

    **Heterogeneous dimensions**: G1 and G2 may differ.  The
    intermediate matrices H, M, L_M are all k x k regardless of G1 or
    G2, so the Woodbury formulas apply without modification.

    **Heterogeneous batch ranks** are also supported when the conditioning
    batch is prefix-compatible with the target batch (for example `(K,)`
    conditioning `(K, D)`). The helper aligns the conditioning tensors to
    the target batch rank via singleton insertion + broadcasting.

    Parameters
    ----------
    W1 : jnp.ndarray, shape (..., G1, k)
        Low-rank factor for parameter group 1.
    D1 : jnp.ndarray, shape (..., G1)
        Diagonal covariance for parameter group 1.
    W2 : jnp.ndarray, shape (..., G2, k)
        Low-rank factor for parameter group 2.
    D2 : jnp.ndarray, shape (..., G2)
        Diagonal covariance for parameter group 2.
    loc1 : jnp.ndarray, shape (..., G1)
        Location for parameter group 1.
    loc2 : jnp.ndarray, shape (..., G2)
        Location for parameter group 2.
    theta1_sample : jnp.ndarray, shape (..., G1)
        Sampled value of theta_1 (in unconstrained space).

    Returns
    -------
    cond_loc : jnp.ndarray, shape (..., G2)
        Conditional mean for theta_2 | theta_1.
    cond_W : jnp.ndarray, shape (..., G2, k)
        Conditional covariance factor for theta_2 | theta_1.
    cond_D : jnp.ndarray, shape (..., G2)
        Conditional diagonal covariance (same as D2).
    """
    # Align theta_1 tensors to theta_2 batch rank so mixed-rank joint groups
    # (for example `(K,) -> (K, D)`) can use the same Woodbury algebra.
    target_batch = W2.shape[:-2]
    W1 = _broadcast_to_batch_prefix(
        W1, target_batch, keep_last_dims=2, name="W1"
    )
    D1 = _broadcast_to_batch_prefix(
        D1, target_batch, keep_last_dims=1, name="D1"
    )
    loc1 = _broadcast_to_batch_prefix(
        loc1, target_batch, keep_last_dims=1, name="loc1"
    )
    theta1_sample = _broadcast_to_batch_prefix(
        theta1_sample, target_batch, keep_last_dims=1, name="theta1_sample"
    )
    loc2 = _broadcast_to_batch_prefix(
        loc2, target_batch, keep_last_dims=1, name="loc2"
    )
    D2 = _broadcast_to_batch_prefix(
        D2, target_batch, keep_last_dims=1, name="D2"
    )

    # H = W1^T @ diag(1/D1) @ W1, shape (..., k, k)
    D1_inv = 1.0 / D1
    W1_scaled = W1 * D1_inv[..., :, None]  # (..., G, k) * (..., G, 1)
    H = jnp.einsum("...gk,...gl->...kl", W1_scaled, W1)  # (..., k, k)

    # M = (I_k + H)^{-1} via Cholesky, shape (..., k, k)
    k = W1.shape[-1]
    IpH = jnp.eye(k) + H
    L_IpH = jnp.linalg.cholesky(IpH)
    # Solve (I+H) M = I for M via triangular solves.
    # Broadcast eye(k) to match batch dims so cho_solve doesn't confuse
    # the batch axis (e.g. n_components) with the matrix axis (k).
    eye_k = jnp.broadcast_to(jnp.eye(k), L_IpH.shape)
    M = jax.scipy.linalg.cho_solve((L_IpH, True), eye_k)

    # L_M = cholesky(M) for the conditional covariance factor
    L_M = jnp.linalg.cholesky(M)

    # Conditional covariance factor: W2 @ L_M, shape (..., G, k)
    cond_W = jnp.einsum("...gk,...kl->...gl", W2, L_M)

    # Conditional mean: loc2 + W2 @ M @ W1^T @ D1^{-1} @ (theta1 - loc1)
    delta = theta1_sample - loc1  # (..., G)
    v1 = delta * D1_inv  # (..., G)
    v2 = jnp.einsum("...gk,...g->...k", W1, v1)  # (..., k)
    v3 = jnp.einsum("...kl,...l->...k", M, v2)  # (..., k)
    v4 = jnp.einsum("...gk,...k->...g", W2, v3)  # (..., G)
    cond_loc = loc2 + v4

    cond_D = D2

    return cond_loc, cond_W, cond_D


def _build_distribution_for_spec(
    loc: jnp.ndarray,
    cov_factor: jnp.ndarray,
    cov_diag: jnp.ndarray,
    spec: "NormalWithTransformSpec",
    is_scalar_in_joint: bool,
    n_batch_dims: int,
) -> dist.Distribution:
    """Build the sampling distribution for a single spec in a joint group.

    For gene-specific specs (is_scalar_in_joint=False), constructs a
    ``LowRankMultivariateNormal`` over the gene dimension, optionally
    wrapped with ``to_event`` for batch dimensions.

    For scalar specs (is_scalar_in_joint=True), the variational params
    have been expanded with a trailing dimension of 1. Here we collapse
    that 1-dim LowRankMVN into a scalar ``Normal`` so the event shape
    matches the model's prior. The variance is
    ``sum(W[..., 0, :]**2) + D[..., 0]``.

    Parameters
    ----------
    loc : jnp.ndarray
        Location parameter (expanded shape for scalars).
    cov_factor : jnp.ndarray
        Low-rank factor W (expanded shape for scalars).
    cov_diag : jnp.ndarray
        Diagonal covariance D (expanded shape for scalars).
    spec : NormalWithTransformSpec
        Parameter specification.
    is_scalar_in_joint : bool
        Whether this spec is a scalar parameter expanded for joint modeling.
    n_batch_dims : int
        Number of batch dimensions in the expanded shape.

    Returns
    -------
    dist.Distribution
        Transformed distribution ready for ``numpyro.sample``.
    """
    if is_scalar_in_joint:
        # Collapse from 1-dim LowRankMVN to scalar Normal.
        # var = sum(W[..., 0, :]^2, axis=-1) + D[..., 0]
        scalar_var = (
            jnp.sum(cov_factor[..., 0, :] ** 2, axis=-1) + cov_diag[..., 0]
        )
        scalar_loc = loc[..., 0]
        base = dist.Normal(scalar_loc, jnp.sqrt(scalar_var))
        # After collapsing the trailing 1, the Normal has the
        # original resolved shape.  Wrap all those dims as event
        # dims so the event_dim matches the model (which applies
        # .to_event(len(shape)) for non-scalar shapes).
        # n_batch_dims = len(expanded_shape) - 1 = len(resolved_shape)
        if n_batch_dims > 0:
            base = base.to_event(n_batch_dims)
    else:
        base = dist.LowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )
        if n_batch_dims > 0:
            base = base.to_event(n_batch_dims)

    return dist.TransformedDistribution(base, spec.transform)


def _build_base_distribution_for_joint_spec(
    loc: jnp.ndarray,
    cov_factor: jnp.ndarray,
    cov_diag: jnp.ndarray,
    is_scalar_in_joint: bool,
    n_batch_dims: int,
) -> dist.Distribution:
    """Build the unconstrained base distribution for a joint-guide spec.

    This helper mirrors the shape handling of ``_build_distribution_for_spec``
    but intentionally does **not** apply any transform. It is used by the
    horseshoe/NEG NCP-aware joint path to sample ``*_raw`` latent sites
    directly in unconstrained space.

    Parameters
    ----------
    loc : jnp.ndarray
        Location parameter (expanded shape for scalar-in-joint specs).
    cov_factor : jnp.ndarray
        Low-rank covariance factor with trailing rank dimension.
    cov_diag : jnp.ndarray
        Diagonal covariance term.
    is_scalar_in_joint : bool
        Whether this spec is represented as a scalar expanded with trailing 1.
    n_batch_dims : int
        Number of batch dimensions in the expanded shape.

    Returns
    -------
    dist.Distribution
        A ``Normal`` (scalar-in-joint) or ``LowRankMultivariateNormal``
        (gene-specific) in unconstrained space with event dims aligned to model.
    """
    # Scalar parameters are represented as expanded (..., 1) vectors in the
    # Woodbury chain. Collapse to a scalar Normal for sampling to preserve the
    # model's event shape semantics.
    if is_scalar_in_joint:
        scalar_var = (
            jnp.sum(cov_factor[..., 0, :] ** 2, axis=-1) + cov_diag[..., 0]
        )
        scalar_loc = loc[..., 0]
        base = dist.Normal(scalar_loc, jnp.sqrt(scalar_var))
        if n_batch_dims > 0:
            base = base.to_event(n_batch_dims)
        return base

    # Gene-specific parameters keep the LowRankMVN event over the last axis.
    base = dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=cov_factor,
        cov_diag=cov_diag,
    )
    if n_batch_dims > 0:
        base = base.to_event(n_batch_dims)
    return base


def _is_joint_ncp_spec(spec: "NormalWithTransformSpec") -> bool:
    """Return whether a joint-guide spec uses NCP raw latents (horseshoe or NEG).

    Parameters
    ----------
    spec : NormalWithTransformSpec
        Parameter specification in a joint low-rank group.

    Returns
    -------
    bool
        True when the specification is one of the horseshoe or NEG parameter
        classes that define a ``raw_name`` latent site and deterministic
        constrained transform in the model.
    """
    return isinstance(
        spec,
        (
            HorseshoeHierarchicalSigmoidNormalSpec,
            HorseshoeHierarchicalPositiveNormalSpec,
            HorseshoeDatasetSigmoidNormalSpec,
            HorseshoeDatasetPositiveNormalSpec,
            NEGHierarchicalSigmoidNormalSpec,
            NEGHierarchicalPositiveNormalSpec,
            NEGDatasetSigmoidNormalSpec,
            NEGDatasetPositiveNormalSpec,
        ),
    )


def setup_joint_guide(
    specs: List["NormalWithTransformSpec"],
    guide: JointLowRankGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> List[jnp.ndarray]:
    """Setup a joint low-rank guide over multiple parameters.

    Supports **heterogeneous dimensions**: specs may mix gene-specific
    vectors (shape ``(G,)`` or ``(C, G)``) with scalar parameters (shape
    ``()`` or ``(C,)``).  Scalar specs are internally expanded by
    appending a trailing dimension of 1, so the Woodbury chain operates
    uniformly on ``(..., G_i, k)`` tensors.  At sampling time, scalar
    specs are collapsed back to a ``Normal`` distribution matching the
    model's event shape.

    Mixed batch-rank groups are allowed when each shorter batch shape is a
    prefix of the longest batch shape in the group (for example ``()`` with
    ``(K, D)``). For stable conditioning semantics, specs must be ordered
    from shorter/shared batches to longer batches.

    The chain rule decomposition samples each parameter in unconstrained
    space from the marginal/conditional LowRankMVN, then applies the
    transform via ``TransformedDistribution``.  Unconstrained samples are
    used for subsequent conditioning steps.

    Parameters
    ----------
    specs : List[NormalWithTransformSpec]
        Ordered list of parameter specifications to model jointly.
        May include both gene-specific and scalar specs.
    guide : JointLowRankGuide
        Joint low-rank guide marker.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    List[jnp.ndarray]
        Sampled parameter values in constrained space.
    """
    k = guide.rank

    # ------------------------------------------------------------------
    # Per-spec shape resolution and scalar expansion
    # ------------------------------------------------------------------
    resolved_shapes: List[tuple] = []
    expanded_shapes: List[tuple] = []
    is_scalar_flags: List[bool] = []

    for spec in specs:
        rs = resolve_shape(
            spec.shape_dims,
            dims,
            is_mixture=spec.is_mixture,
            is_dataset=spec.is_dataset,
        )
        resolved_shapes.append(rs)

        if not spec.is_gene_specific:
            # Scalar parameter: append a trailing dim of 1 so that the
            # Woodbury chain can treat it as a 1-dim gene vector.
            expanded = (*rs, 1) if rs else (1,)
            is_scalar_flags.append(True)
        else:
            expanded = rs
            is_scalar_flags.append(False)
        expanded_shapes.append(expanded)

    # Validate batch compatibility across specs. Unlike the earlier strict
    # equality requirement, mixed-rank batches are now supported when each
    # shorter batch is a prefix of the longest batch. This enables
    # shared-across-dataset parameters (for example `(K,)`) to be jointly
    # modeled with dataset-specific parameters `(K, D)` in the same group.
    batch_shapes = [es[:-1] for es in expanded_shapes]
    ref_batch = _select_reference_batch_shape(batch_shapes)

    for i, bs in enumerate(batch_shapes):
        if not _is_batch_prefix(bs, ref_batch):
            raise ValueError(
                f"Joint group '{guide.group}': spec '{specs[i].name}' has "
                f"batch shape {bs}, which is incompatible with reference "
                f"batch shape {ref_batch}. Joint groups require each spec "
                f"batch shape to be a prefix of the longest batch shape "
                f"(only the trailing gene dimension may differ)."
            )

    # Guard chain order: the Woodbury conditioning path assumes batch ranks
    # do not decrease as we iterate through specs. In practice this means
    # shared/shorter-batch parameters must appear before longer-batch specs.
    for i in range(1, len(batch_shapes)):
        prev_bs = batch_shapes[i - 1]
        curr_bs = batch_shapes[i]
        if not _is_batch_prefix(prev_bs, curr_bs):
            raise ValueError(
                f"Joint group '{guide.group}': invalid order for specs "
                f"'{specs[i - 1].name}' -> '{specs[i].name}' with batch "
                f"shapes {prev_bs} -> {curr_bs}. For mixed batch ranks, "
                f"order specs from shared/shorter batches to longer batches "
                f"(for example () -> (K,) -> (K, D))."
            )

    # ------------------------------------------------------------------
    # Register per-parameter variational params using expanded shapes
    # ------------------------------------------------------------------
    locs = []
    Ws = []
    Ds = []
    for idx, spec in enumerate(specs):
        exp_shape = expanded_shapes[idx]

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
        locs.append(loc_i)
        Ws.append(W_i)
        Ds.append(D_i)

    # ------------------------------------------------------------------
    # Chain-rule sampling loop
    # ------------------------------------------------------------------
    unconstrained_samples = []
    constrained_samples = []

    # Current conditional factor and loc for each parameter, updated as
    # we condition on earlier parameters in the chain.
    current_Ws = list(Ws)
    current_locs = list(locs)

    for i, spec in enumerate(specs):
        n_batch_dims_i = len(expanded_shapes[i]) - 1
        is_scalar = is_scalar_flags[i]

        if i == 0:
            # First parameter: sample from its marginal distribution
            # Horseshoe/NEG specs are parameterized with an explicit NCP raw
            # latent site in the model (e.g., gate_raw). The joint block must
            # sample that raw site directly to keep model/guide sample-site
            # alignment.
            if _is_joint_ncp_spec(spec):
                base = _build_base_distribution_for_joint_spec(
                    current_locs[0],
                    current_Ws[0],
                    Ds[0],
                    is_scalar,
                    n_batch_dims_i,
                )
                unconstrained = numpyro.sample(spec.raw_name, base)
                constrained = spec.transform(unconstrained)
                numpyro.deterministic(spec.constrained_name, constrained)
            else:
                transformed = _build_distribution_for_spec(
                    current_locs[0],
                    current_Ws[0],
                    Ds[0],
                    spec,
                    is_scalar,
                    n_batch_dims_i,
                )
                constrained = numpyro.sample(spec.constrained_name, transformed)
                unconstrained = spec.transform.inv(constrained)
            constrained_samples.append(constrained)

            # The Woodbury chain always works in expanded unconstrained space.
            # Re-expand scalar samples so later conditioning has shape (..., 1).
            if is_scalar:
                unconstrained = unconstrained[..., None]
            unconstrained_samples.append(unconstrained)

        else:
            # Condition on all previously sampled parameters via
            # iterated Woodbury updates.  For the common case n=2 this
            # is a single conditioning step.
            cond_loc = current_locs[i]
            cond_W = current_Ws[i]
            cond_D = Ds[i]

            for j in range(i):
                cond_loc, cond_W, cond_D = _woodbury_conditional_params(
                    W1=current_Ws[j],
                    D1=Ds[j],
                    W2=cond_W,
                    D2=cond_D,
                    loc1=current_locs[j],
                    loc2=cond_loc,
                    theta1_sample=unconstrained_samples[j],
                )

            # As above: horseshoe/NEG specs in joint groups must sample raw NCP
            # latents so guide/model sample sites match exactly.
            if _is_joint_ncp_spec(spec):
                base = _build_base_distribution_for_joint_spec(
                    cond_loc,
                    cond_W,
                    cond_D,
                    is_scalar,
                    n_batch_dims_i,
                )
                unconstrained = numpyro.sample(spec.raw_name, base)
                constrained = spec.transform(unconstrained)
                numpyro.deterministic(spec.constrained_name, constrained)
            else:
                transformed = _build_distribution_for_spec(
                    cond_loc,
                    cond_W,
                    cond_D,
                    spec,
                    is_scalar,
                    n_batch_dims_i,
                )
                constrained = numpyro.sample(spec.constrained_name, transformed)
                unconstrained = spec.transform.inv(constrained)
            constrained_samples.append(constrained)

            if is_scalar:
                unconstrained = unconstrained[..., None]
            unconstrained_samples.append(unconstrained)

            # Update this parameter's conditional W and loc for use by
            # subsequent parameters in the chain (needed for n > 2)
            current_Ws[i] = cond_W
            current_locs[i] = cond_loc

    return constrained_samples

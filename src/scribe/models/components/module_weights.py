"""
Per-leaf regulatory-module weights for hierarchical gene-gene correlation
(NB-LogNormal "Rung 1.5").

This module implements the *shared primitive* behind the hierarchical
correlation model described in ``paper/_nb_lognormal.qmd``
(``sec-nbln-hierarchical-correlation``). Both inference paths consume it:

- **Phase A (SVI / VAE sanity check)** scales the *K-dimensional latent* prior
  per leaf -- ``z_c ~ Normal(0, diag(s_{sigma(c)}^2))`` -- so the shared linear
  decoder ``W`` *induces* the leaf-specific log-rate covariance
  ``W diag(s_d^2) W^T + diag(d)``.
- **Phase B (Laplace production)** folds the same ``s_d`` into *effective
  loadings* ``W_eff,d = W . diag(s_d)`` so that
  ``Sigma_d = W_eff,d W_eff,d^T + diag(d)`` keeps the low-rank-plus-diagonal
  form the Woodbury / Newton kernel already consumes.

Terminology
-----------
A **module** is a column of the shared loadings ``W`` -- a gene co-expression
program. The **module weight** ``s_{d,k} > 0`` is how strongly module ``k``
fluctuates (co-activates its genes) among the cells of leaf ``d``. ``W`` (the
modules / which genes co-vary) is shared across leaves; only the per-leaf
*weight* of each module floats.

The model (additive, crossed/nested over grouping factors)
----------------------------------------------------------
For ``K`` latent modules and a grouping into ``L`` present leaves declared by
``hierarchy`` / ``interactions``, the log module-weight of each leaf decomposes
additively over the grouping factors ``f`` (the same factors the mean ``mu``
uses), exactly mirroring ``paper/_hierarchical_datasets.qmd``
(``sec-hdata-multifactor``):

.. math::

    \\log s_{\\ell,k} = \\sum_f \\alpha^{(f)}_{\\kappa_f(\\ell),\\,k},
    \\qquad
    \\alpha^{(f)}_{\\cdot,k} = \\mathrm{scale}_f \\cdot
        \\mathrm{center}\\big(z^{(f)}_{\\cdot,k}\\big),
    \\qquad z^{(f)}_{\\cdot,k} \\sim \\mathcal N(0, 1),

where ``kappa_f(l)`` maps leaf ``l`` to factor ``f``'s level, ``z^{(f)}`` has
shape ``(L_f, K)``, and ``scale_f = softplus(tau_raw_f)`` (learned) for
``effect_type="random"`` or a fixed ``fixed_scale`` for ``effect_type="fixed"``.
There is **no population intercept**: the absolute magnitude of module ``k``
lives entirely in ``||W[:, k]||``. Families are **gaussian only** in this
version (the module axis ``K`` is low-dimensional, so a learned per-factor
global scale ``tau_f`` is the right tool; the sparse per-element horseshoe/NEG
priors used on the high-dimensional per-gene axis add funnel fragility with few
factor levels and are rejected by :func:`build_module_weight_operators`).

Identifiability (three non-redundant constraints, per module ``k``)
-------------------------------------------------------------------
1. **Global leaf-axis anchor** (the only constraint strictly required to remove
   the per-column scale gauge ``W[:, k] -> a_k W[:, k]``, ``s -> s / a_k``):
   after summing the factor effects, subtract the realized leaf mean so
   ``sum_l log s_{l,k} = 0``. This is the additive generalization of the flat
   ``module_weights_from_raw`` centering.
2. **Leaf-count-weighted per-factor sum-to-zero** ``sum_l n^(f)_l alpha_{l,k} =
   0`` (with ``n^(f)_l`` = number of present leaves at level ``l``) -- separates
   the main effects under unbalanced / incomplete crossing.
3. **Interaction zero-margin** over present cells (grand mean + both parent
   margins) -- separates interaction from its parents.

Constraints (1)-(3) compose cleanly: the leaf-count-weighted centering makes
every gathered factor effect have zero leaf-sum, so the anchor acts as the
identity on the factor effects and only pins the residual overall scale.

The *rotation* gauge ``W -> W R`` is broken only generically (when the realized
per-leaf weight profiles are sufficiently distinct); column **sign** and
**permutation** gauges always remain and are handled at interpretation time,
exactly as for the single-dataset ``W``.

Implementation note (centering vs. contrasts)
--------------------------------------------
Constraints are enforced by (leaf-count-weighted) *centering* projections
``P_f`` applied to the raw NCP draws, leaving one redundant, prior-killed
degree of freedom per factor -- harmless for SVI and Laplace alike, and simpler
than a Helmert / sum-to-zero contrast basis. The projections and the
identifiability **rank guard** are built once, at spec time, by
:func:`build_module_weight_operators`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist

# Site-name suffixes for the module-weight block. Centralized here so the
# model-side sampler and the guide-side block agree on every site name (a
# mismatch would silently break the mean-field ELBO pairing). Per-factor sites
# additionally carry a ``__{factor}`` suffix (``:`` in interaction names ->
# ``__`` so the key is dict/NumPyro safe).
_TAU_RAW_SUFFIX = "_tau_raw"  # scalar Normal raw for a factor's scale tau_f
_RAW_SUFFIX = "_raw"  # (L_f, K) Normal(0, 1) NCP raw effects
_LOG_SUFFIX = "_log"  # (n_leaves, K) deterministic log s (post assembly)
_EFFECT_SUFFIX = "_effect"  # (L_f, K) deterministic centered per-factor effect
# The bare ``site_prefix`` (no suffix) names the constrained ``s`` deterministic.

# Default hyperprior on a factor's between-level scale tau_f. Mirrors the
# ``tau_mu`` treatment in ``paper/_hierarchical_datasets.qmd`` (softplus of a
# Normal placing prior mass on small values): softplus(-2) ~ 0.127, i.e. a
# prior median around 13% multiplicative level-to-level module-weight variation.
_DEFAULT_TAU_LOC = -2.0
_DEFAULT_TAU_SCALE = 0.5

# Default fixed scale for ``effect_type="fixed"`` factors whose ``GroupLevel``
# leaves ``fixed_scale`` as ``None``. A permissive log-scale value (e^1.0 ~ 2.7x
# per unit centered effect) so a deliberate low-cardinality contrast (e.g. a
# two-level treatment) is free to express a real module-weight shift rather than
# being over-shrunk. Tunable per factor via ``GroupLevel(fixed_scale=...)``.
_DEFAULT_MODULE_WEIGHT_FIXED_SCALE = 1.0

# The grouping TARGET_NAME / canonical prior key for the module-weight
# hierarchy. Registered in ``parameter_mapping.PARAM_REGISTRY`` and
# ``grouping.TARGET_NAMES``; ``Factor.family(_MODULE_WEIGHT_TARGET)`` selects
# per-factor participation.
_MODULE_WEIGHT_TARGET = "module_weight"

# Prior families accepted on the module-weight target in this version.
_ALLOWED_MODULE_WEIGHT_FAMILIES = frozenset({"none", "gaussian"})


def module_weights_from_raw(
    raw: jnp.ndarray, tau_raw: jnp.ndarray
) -> jnp.ndarray:
    """Map NCP raw effects + unconstrained ``tau`` to constrained weights ``s``.

    This is the flat, single-factor kernel: it applies, in order,
    ``tau = softplus(tau_raw)``; the (unweighted) sum-to-zero centering of
    ``raw`` across leaves (per module); then ``s = exp(tau . centered)``. It is
    reused as the **single-factor fast path** of
    :func:`module_weights_leaf_from_factors` (where every leaf is its own level,
    so leaf-count weighting is trivial and no global anchor is needed), and by
    the results bridge for the single-factor case, guaranteeing bit-identical
    behaviour with the pre-refold flat model.

    Parameters
    ----------
    raw : jnp.ndarray, shape ``(n_leaves, latent_dim)``
        Standard-Normal NCP raw effects (pre-centering).
    tau_raw : jnp.ndarray, scalar
        Unconstrained between-leaf scale; ``softplus`` maps it to positive
        ``tau``.

    Returns
    -------
    s : jnp.ndarray, shape ``(n_leaves, latent_dim)``
        Constrained relative per-leaf module weights, strictly positive, with
        ``mean_l log s_{l,k} == 0`` per module.
    """
    tau_s = jnn.softplus(tau_raw)
    centered = raw - jnp.mean(raw, axis=0, keepdims=True)
    return jnp.exp(tau_s * centered)


# ---------------------------------------------------------------------------
# Operator builder: per-factor centering/zero-margin projections + rank guard
# ---------------------------------------------------------------------------


def _weighted_projection(C: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``W``-orthogonal projection off ``span(C)``: ``I - C (C^T W C)^+ C^T W``.

    With ``W = diag(w)`` (leaf-count weights) this removes the ``w``-weighted
    component of the input lying in the column space of ``C``. For a base
    factor ``C`` is the all-ones vector (removes the weighted mean ->
    ``sum_l w_l a_l = 0``); for an interaction ``C = [1 | parent indicators]``
    (removes the weighted grand mean and both parent margins). Idempotent; its
    rank is ``L - rank(C)``.

    Parameters
    ----------
    C : numpy.ndarray, shape ``(L, m)``
        Constraint directions (columns) to project out.
    w : numpy.ndarray, shape ``(L,)``
        Positive per-level leaf counts (the weighting metric ``W``).

    Returns
    -------
    numpy.ndarray, shape ``(L, L)``
        The oblique (``W``-orthogonal) projection matrix.
    """
    L = C.shape[0]
    Wd = np.diag(w.astype(np.float64))
    CtW = C.T @ Wd  # (m, L)
    gram = CtW @ C  # (m, m)
    proj_coeff = C @ np.linalg.pinv(gram) @ CtW  # (L, L)
    return np.eye(L) - proj_coeff


def _interaction_operand_levels(
    factor,
    base_factors: Dict[str, object],
) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Reconstruct, per interaction level, its operand level indices.

    Interaction levels are the *present* operand combinations. We recover the
    ``(operand-a level, operand-b level, ...)`` tuple for each interaction level
    by scanning the leaves: every leaf sharing an interaction level shares the
    same operand levels (that is how interaction levels are defined), so the
    first leaf per interaction level suffices. Operand factor names are taken
    from the ``":"``-joined interaction factor name.

    Parameters
    ----------
    factor : Factor
        The interaction factor (``kind == "interaction"``).
    base_factors : dict
        Map ``name -> Factor`` for the base factors (to read their
        ``leaf_to_level``).

    Returns
    -------
    operand_names : tuple of str
        The operand base-factor names, in order.
    ops_by_level : numpy.ndarray, shape ``(L_g, n_operands)``
        ``ops_by_level[i]`` is the operand level indices for interaction level
        ``i``.
    """
    operand_names = tuple(factor.name.split(":"))
    inter_l2l = np.asarray(factor.leaf_to_level)
    L_g = factor.n_levels
    n_ops = len(operand_names)
    ops_by_level = -np.ones((L_g, n_ops), dtype=np.int64)
    base_l2l = {
        nm: np.asarray(base_factors[nm].leaf_to_level) for nm in operand_names
    }
    for leaf in range(inter_l2l.shape[0]):
        i = int(inter_l2l[leaf])
        if ops_by_level[i, 0] < 0:  # first leaf seen for this interaction level
            for j, nm in enumerate(operand_names):
                ops_by_level[i, j] = int(base_l2l[nm][leaf])
    return operand_names, ops_by_level


def build_module_weight_operators(grouping_spec) -> Optional[Dict]:
    """Build per-factor projections + gather maps and run the rank guard.

    Assembles, once at spec-build time (pure NumPy on the static
    :class:`GroupingSpec`), everything the traced assembly needs: for each
    factor participating in the module-weight hierarchy
    (``factor.family("module_weight") != "none"``), its leaf-count-weighted
    centering / zero-margin projection ``P_f``, its ``leaf_to_level`` gather,
    its ``effect_type`` and resolved ``fixed_scale``. Also performs:

    - the **gaussian-only reject** (raises on any non-gaussian family), and
    - the **identifiability rank guard**: the stacked, gathered orthonormal
      bases of ``range(P_f)`` over the present leaves must have full column rank
      (equal to ``sum_f rank(P_f)``, which for a complete/saturated design is
      ``n_leaves - 1``). A column-rank-deficient design (an over-parameterized
      interaction under incomplete crossing) raises, rather than shipping a flat
      posterior direction.

    Parameters
    ----------
    grouping_spec : GroupingSpec or None
        The resolved grouping. ``None`` (no grouping) returns ``None``.

    Returns
    -------
    dict or None
        ``None`` if no factor participates. Otherwise a dict with keys:
        ``"n_leaves"`` (int), ``"fast_path"`` (bool -- single random base factor
        with identity gather), and ``"factors"`` (list of per-factor dicts with
        keys ``name``, ``site``, ``kind``, ``is_random``, ``fixed_scale``,
        ``n_levels``, ``P`` (jnp ``(L_f, L_f)``), ``leaf_to_level`` (jnp int
        ``(n_leaves,)``)).

    Raises
    ------
    ValueError
        On a non-gaussian family, or a rank-deficient (non-identifiable)
        design.
    """
    if grouping_spec is None:
        return None

    base_factors = {
        f.name: f for f in grouping_spec.factors if f.kind == "base"
    }
    n_leaves = int(grouping_spec.n_leaves)

    participating: List = []
    for f in grouping_spec.factors:
        fam = f.family(_MODULE_WEIGHT_TARGET)
        if fam == "none":
            continue
        if fam not in _ALLOWED_MODULE_WEIGHT_FAMILIES:
            raise ValueError(
                f"module_weight hierarchy supports only the 'gaussian' family "
                f"in this version, but factor {f.name!r} requests {fam!r}. "
                f"(The module axis K is low-dimensional; sparse horseshoe/neg "
                f"priors are reserved for the per-gene axis.)"
            )
        participating.append(f)

    if not participating:
        return None

    factor_ops: List[Dict] = []
    # Accumulate the stacked gathered bases for the rank guard.
    gathered_bases: List[np.ndarray] = []
    # Static (NumPy) compacted gather per factor, for the fast-path decision.
    static_l2ls: List[np.ndarray] = []
    total_free = 0

    for f in participating:
        l2l_orig = np.asarray(f.leaf_to_level, dtype=np.int64)
        # Compact to the PRESENT levels only. A ``Factor`` may declare more
        # levels than occur in the data -- e.g. a pandas categorical retains
        # unused categories after an AnnData subset. Such a phantom level maps
        # to no leaf, so its NCP row would never be gathered yet would still
        # leak a free draw into the leaf-count-weighted centering (miscentering
        # the per-factor effect and adding an unidentified parameter). Working
        # on present levels (re-indexed 0..L-1) removes the phantom entirely.
        present, l2l = np.unique(l2l_orig, return_inverse=True)
        l2l = np.asarray(l2l, dtype=np.int64)
        L = int(present.shape[0])
        counts = np.bincount(l2l, minlength=L).astype(np.float64)  # all >= 1

        if f.kind == "interaction":
            _, ops_by_level = _interaction_operand_levels(f, base_factors)
            # Restrict operand levels to the present interaction levels (a
            # no-op for interactions, whose levels are present combos).
            ops_by_level = ops_by_level[present]
            cols = [np.ones((L, 1))]
            operand_names = tuple(f.name.split(":"))
            for j, nm in enumerate(operand_names):
                # Original operand width; unused base categories become all-zero
                # indicator columns, dropped by the pinv in _weighted_projection.
                n_lvl = int(base_factors[nm].n_levels)
                ind = np.zeros((L, n_lvl))
                ind[np.arange(L), ops_by_level[:, j]] = 1.0
                cols.append(ind)
            C = np.concatenate(cols, axis=1)  # [ones | parent indicators]
        else:  # base (or nested -- nesting is encoded in leaf_to_level)
            C = np.ones((L, 1))

        P = _weighted_projection(C, counts)  # (L, L)

        # Orthonormal basis of range(P) for the rank guard. P is an oblique
        # (W-orthogonal) projector, so its nonzero singular values need not be
        # ~1; use a RELATIVE threshold to split them from the numerical zeros.
        U, sv, _ = np.linalg.svd(P)
        rel_tol = (
            float(sv[0]) * max(P.shape) * np.finfo(sv.dtype).eps
            if sv.size
            else 0.0
        )
        rank_f = int(np.sum(sv > rel_tol))
        Bf = U[:, :rank_f]  # (L, rank_f)
        # Gather each basis column onto the leaf axis: (n_leaves, rank_f).
        gathered_bases.append(Bf[l2l, :])
        static_l2ls.append(l2l)
        total_free += rank_f

        is_random = f.effect_type != "fixed"
        fixed_scale = (
            None
            if is_random
            else (
                float(f.fixed_scale)
                if f.fixed_scale is not None
                else _DEFAULT_MODULE_WEIGHT_FIXED_SCALE
            )
        )
        factor_ops.append(
            {
                "name": f.name,
                "site": f.name.replace(":", "__"),
                "kind": f.kind,
                "is_random": is_random,
                "fixed_scale": fixed_scale,
                "n_levels": L,
                # No explicit dtype: honor jax_enable_x64 so the centering is
                # full-precision under an x64 solve (float32 otherwise). The
                # assembly casts to the param dtype at use.
                "P": jnp.asarray(P),
                "leaf_to_level": jnp.asarray(l2l, dtype=jnp.int32),
            }
        )

    # --- Rank guard: stacked gathered bases must have full column rank -------
    # matrix_rank's default tolerance is RELATIVE (sv.max * max(dim) * eps), so
    # it scales correctly with design size (an absolute tol would misjudge
    # large designs). Skip entirely when there are no free parameters (e.g. a
    # single leaf, where every P is rank 0): matrix_rank on the resulting
    # (n_leaves, 0) array would take max() over an empty spectrum.
    if total_free == 0:
        eff_rank = 0
    else:
        A = np.concatenate(gathered_bases, axis=1)  # (n_leaves, total_free)
        eff_rank = int(np.linalg.matrix_rank(A))
    if eff_rank < total_free:
        names = ", ".join(fo["name"] for fo in factor_ops)
        raise ValueError(
            "module_weight hierarchy is not identifiable for this design: the "
            f"factor effects [{names}] contribute {total_free} free parameters "
            f"but map to only {eff_rank} independent per-leaf directions "
            f"(n_leaves={n_leaves}). This happens with an interaction under "
            "incomplete crossing (missing operand combinations). Drop the "
            "interaction from priors['module_weight'] (fold it into the main "
            "effects), or complete the design."
        )

    # Single-factor fast path: one random base factor whose (compacted) gather
    # is identity. Decide from the STATIC compacted NumPy ``l2l`` collected
    # above, never a stored ``jnp`` value: this builder runs inside the SVI
    # model/guide trace, where a host op on a trace-time array would raise
    # ``TracerArrayConversionError`` under ``jit``.
    fast_path = False
    if len(participating) == 1:
        fo0 = factor_ops[0]
        l2l0 = static_l2ls[0]
        if (
            fo0["kind"] == "base"
            and fo0["is_random"]
            and fo0["n_levels"] == n_leaves
            and np.array_equal(l2l0, np.arange(n_leaves))
        ):
            fast_path = True

    return {
        "n_leaves": n_leaves,
        "fast_path": fast_path,
        "factors": factor_ops,
    }


def module_weights_leaf_from_factors(
    ops: Dict,
    raw_by_factor: Dict[str, jnp.ndarray],
    tau_raw_by_factor: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Assemble the realized per-leaf module weights from per-factor effects.

    The **single source of truth** for the additive raw -> ``s`` transform,
    shared by the SVI sampler (:func:`sample_module_weights_hierarchical`) and
    the Laplace obs model (which optimizes the per-factor ``raw``/``tau_raw`` as
    plain params and rebuilds ``s`` here). Keeping the gauge in one place means
    the two inference backends cannot drift.

    Computes ``log s = sum_f gather(scale_f . (P_f @ z_f))`` then the global
    leaf anchor ``log s -= mean_leaf(log s)`` and returns ``exp(log s)``.

    Parameters
    ----------
    ops : dict
        The operator bundle from :func:`build_module_weight_operators`.
    raw_by_factor : dict
        Map factor ``name`` -> NCP raw ``z^(f)`` of shape ``(L_f, K)``.
    tau_raw_by_factor : dict
        Map factor ``name`` -> unconstrained scalar ``tau_raw_f`` for random
        factors. Fixed factors are ignored here (their scale is baked into
        ``ops``).

    Returns
    -------
    s_leaf : jnp.ndarray, shape ``(n_leaves, K)``
        Constrained per-leaf module weights, strictly positive, with
        ``sum_l log s_{l,k} == 0`` per module.
    """
    factors = ops["factors"]

    # Fast path: exactly one random base factor with identity gather reduces to
    # the flat kernel, guaranteeing bit-identity with the pre-refold model.
    if ops["fast_path"]:
        f = factors[0]
        return module_weights_from_raw(
            raw_by_factor[f["name"]], tau_raw_by_factor[f["name"]]
        )

    n_leaves = ops["n_leaves"]
    # Latent dim K from any factor's raw draw.
    K = raw_by_factor[factors[0]["name"]].shape[1]
    log_s = jnp.zeros((n_leaves, K), dtype=jnp.float32)
    for f in factors:
        z = raw_by_factor[f["name"]]  # (L_f, K)
        if f["is_random"]:
            scale = jnn.softplus(tau_raw_by_factor[f["name"]])
        else:
            scale = f["fixed_scale"]
        # Cast P to the param dtype so a float64 (x64) fit keeps full precision
        # and a float32 fit avoids a dtype-promotion mismatch.
        alpha = scale * (f["P"].astype(z.dtype) @ z)  # (L_f,K) centered effect
        log_s = log_s + jnp.take(alpha, f["leaf_to_level"], axis=0)
    # Global leaf-axis anchor: removes the residual scale gauge. Because every
    # factor effect is leaf-count-weighted zero-mean, its gathered contribution
    # already has zero leaf-sum, so this only pins the overall level.
    log_s = log_s - jnp.mean(log_s, axis=0, keepdims=True)
    return jnp.exp(log_s)


def module_weight_effects_from_raw(
    ops: Dict,
    raw_by_factor: Dict[str, jnp.ndarray],
    tau_raw_by_factor: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """Return the centered per-factor effects ``alpha^(f)`` (for interpretation).

    Same projection/scale as :func:`module_weights_leaf_from_factors` but
    returns the per-factor ``(L_f, K)`` effects instead of the assembled
    per-leaf weights -- e.g. "how does the treatment shift each module?".

    Parameters
    ----------
    ops : dict
        The operator bundle from :func:`build_module_weight_operators`.
    raw_by_factor, tau_raw_by_factor : dict
        As in :func:`module_weights_leaf_from_factors`.

    Returns
    -------
    dict
        Map factor ``name`` -> centered effect ``alpha^(f)`` of shape
        ``(L_f, K)``.
    """
    out: Dict[str, jnp.ndarray] = {}
    for f in ops["factors"]:
        z = raw_by_factor[f["name"]]
        scale = (
            jnn.softplus(tau_raw_by_factor[f["name"]])
            if f["is_random"]
            else f["fixed_scale"]
        )
        out[f["name"]] = scale * (f["P"].astype(z.dtype) @ z)
    return out


# ---------------------------------------------------------------------------
# SVI model + guide (Phase A)
# ---------------------------------------------------------------------------


def sample_module_weights_hierarchical(
    grouping_spec,
    latent_dim: int,
    *,
    ops: Optional[Dict] = None,
    tau_loc: float = _DEFAULT_TAU_LOC,
    tau_scale: float = _DEFAULT_TAU_SCALE,
    site_prefix: str = "module_weight",
) -> jnp.ndarray:
    """Sample per-leaf module weights ``s`` under the additive hierarchy.

    Model-side sampler for the crossed/nested additive module-weight hierarchy.
    For each participating factor it draws an NCP raw block (and a scalar
    ``tau_raw`` for random factors), then assembles the realized per-leaf
    weights via the shared :func:`module_weights_leaf_from_factors`. The
    matching guide is :func:`guide_module_weights_hierarchical`, which must use
    the same ``site_prefix`` and grouping for the mean-field ELBO to pair.

    Sampled sites (per participating factor ``f``; ``{f}`` = ``name`` with
    ``:`` -> ``__``)
    ------------------------------------------------------------------------
    ``{prefix}_tau_raw__{f}`` : scalar (random factors only)
        Unconstrained Normal; ``scale_f = softplus(.)``.
    ``{prefix}_raw__{f}`` : shape ``(L_f, latent_dim)``
        Standard-Normal NCP raw effects (``to_event(2)``).

    Deterministic sites
    -------------------
    ``{prefix}_effect__{f}`` : ``(L_f, K)`` centered per-factor effect.
    ``{prefix}_log`` : ``(n_leaves, K)`` assembled ``log s``.
    ``{prefix}`` : ``(n_leaves, K)`` constrained ``s``.

    Parameters
    ----------
    grouping_spec : GroupingSpec
        The resolved grouping (its factors carry ``family("module_weight")``).
    latent_dim : int
        Number of latent modules ``K``.
    ops : dict, optional
        Prebuilt operator bundle. If ``None``, built from ``grouping_spec``.
    tau_loc, tau_scale : float, optional
        Location/scale of the Normal on each random factor's ``tau_raw``.
    site_prefix : str, optional
        Prefix for every site name. Default ``"module_weight"``.

    Returns
    -------
    s : jnp.ndarray, shape ``(n_leaves, latent_dim)``
        Per-leaf relative module weights.
    """
    if ops is None:
        ops = build_module_weight_operators(grouping_spec)

    raw_by: Dict[str, jnp.ndarray] = {}
    tau_by: Dict[str, jnp.ndarray] = {}
    for f in ops["factors"]:
        site = f["site"]
        L = f["n_levels"]
        raw_by[f["name"]] = numpyro.sample(
            f"{site_prefix}{_RAW_SUFFIX}__{site}",
            dist.Normal(0.0, 1.0).expand([L, latent_dim]).to_event(2),
        )
        if f["is_random"]:
            tau_by[f["name"]] = numpyro.sample(
                f"{site_prefix}{_TAU_RAW_SUFFIX}__{site}",
                dist.Normal(tau_loc, tau_scale),
            )

    effects = module_weight_effects_from_raw(ops, raw_by, tau_by)
    for f in ops["factors"]:
        numpyro.deterministic(
            f"{site_prefix}{_EFFECT_SUFFIX}__{f['site']}", effects[f["name"]]
        )
    s = module_weights_leaf_from_factors(ops, raw_by, tau_by)
    numpyro.deterministic(f"{site_prefix}{_LOG_SUFFIX}", jnp.log(s))
    numpyro.deterministic(site_prefix, s)
    return s


def guide_module_weights_hierarchical(
    grouping_spec,
    latent_dim: int,
    *,
    ops: Optional[Dict] = None,
    tau_loc: float = _DEFAULT_TAU_LOC,
    tau_scale: float = _DEFAULT_TAU_SCALE,
    site_prefix: str = "module_weight",
) -> None:
    """Mean-field variational block for :func:`sample_module_weights_hierarchical`.

    Registers learnable location/scale params + matching ``numpyro.sample``
    sites for every *latent* site the model sampler emits (per factor:
    ``{prefix}_raw__{f}`` and, for random factors, ``{prefix}_tau_raw__{f}``).
    The deterministic sites are functions of these and need no guide entry.

    Parameters
    ----------
    grouping_spec : GroupingSpec
        Must match the model sampler.
    latent_dim : int
        Number of latent modules ``K``.
    ops : dict, optional
        Prebuilt operator bundle. If ``None``, built from ``grouping_spec``.
    tau_loc, tau_scale : float, optional
        Initial location/scale for each random factor's ``tau_raw`` variational
        Normal.
    site_prefix : str, optional
        Prefix for every guide site name. Must equal the sampler's.

    Returns
    -------
    None
        Side-effecting: registers guide sites in the active NumPyro trace.
    """
    if ops is None:
        ops = build_module_weight_operators(grouping_spec)

    for f in ops["factors"]:
        site = f["site"]
        L = f["n_levels"]
        raw_loc_q = numpyro.param(
            f"{site_prefix}{_RAW_SUFFIX}__{site}_loc",
            jnp.zeros((L, latent_dim)),
        )
        raw_scale_raw_q = numpyro.param(
            f"{site_prefix}{_RAW_SUFFIX}__{site}_scale_raw",
            jnp.full((L, latent_dim), _inv_softplus(1.0)),
        )
        numpyro.sample(
            f"{site_prefix}{_RAW_SUFFIX}__{site}",
            dist.Normal(raw_loc_q, jnn.softplus(raw_scale_raw_q)).to_event(2),
        )
        if f["is_random"]:
            tau_loc_q = numpyro.param(
                f"{site_prefix}{_TAU_RAW_SUFFIX}__{site}_loc", tau_loc
            )
            tau_scale_raw_q = numpyro.param(
                f"{site_prefix}{_TAU_RAW_SUFFIX}__{site}_scale_raw",
                _inv_softplus(tau_scale),
            )
            numpyro.sample(
                f"{site_prefix}{_TAU_RAW_SUFFIX}__{site}",
                dist.Normal(tau_loc_q, jnn.softplus(tau_scale_raw_q)),
            )


def module_weights_active(model_config, dataset_indices) -> bool:
    """Whether the per-leaf module-weight hierarchy is active for this fit.

    Both the model builder and the guide builder **must** gate the
    module-weight block on this identical condition; if they disagree the
    mean-field ELBO sees a latent site present in one trace but not the other
    and raises. Centralizing the predicate here makes drift impossible.

    Active iff a grouping is present, at least one factor declares a
    (non-``"none"``) ``module_weight`` prior family, there are ``>= 2`` leaves,
    and per-cell ``dataset_indices`` are available.

    Parameters
    ----------
    model_config : ModelConfig
        Read for ``grouping_spec`` and ``n_datasets``.
    dataset_indices : jnp.ndarray or None
        Per-cell leaf index array. Must be non-``None`` (a grouped fit).

    Returns
    -------
    bool
        ``True`` iff the module-weight hierarchy should be wired in.
    """
    grouping_spec = getattr(model_config, "grouping_spec", None)
    if grouping_spec is None:
        return False
    if not any(
        f.family(_MODULE_WEIGHT_TARGET) != "none" for f in grouping_spec.factors
    ):
        return False
    n_datasets = getattr(model_config, "n_datasets", None) or 0
    if n_datasets < 2:
        # With fewer than two leaves there is no between-leaf structure to
        # share; the sum-to-zero gauge would force s == 1 identically anyway.
        return False
    if dataset_indices is None:
        return False
    return True


def effective_loadings(W: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """Fold per-leaf module weights into effective low-rank loadings.

    Computes ``W_eff,d = W . diag(s_d)`` -- i.e. scales column ``k`` of the
    shared loadings ``W`` by leaf ``d``'s module weight ``s_{d,k}``. This is the
    algebraic identity that keeps the hierarchical covariance in
    low-rank-plus-diagonal form:

    .. math::

        \\Sigma_d = W \\, \\mathrm{diag}(s_d^2) \\, W^T + \\mathrm{diag}(d)
                  = W_{\\mathrm{eff},d} W_{\\mathrm{eff},d}^T + \\mathrm{diag}(d),

    so the existing Woodbury / Newton machinery is reused unchanged with
    ``W_eff,d`` in place of ``W``.

    Parameters
    ----------
    W : jnp.ndarray, shape ``(G, K)``
        Shared low-rank loadings (``G`` genes, ``K`` modules).
    s : jnp.ndarray, shape ``(K,)`` or ``(D, K)``
        Module weights. A 1-D array is treated as a single leaf and yields a
        ``(G, K)`` result; a 2-D ``(D, K)`` array yields the stacked
        ``(D, G, K)`` effective loadings, one ``(G, K)`` slice per leaf.

    Returns
    -------
    W_eff : jnp.ndarray
        Effective loadings. Shape ``(G, K)`` if ``s`` is 1-D, else
        ``(D, G, K)``.

    Raises
    ------
    ValueError
        If ``s`` is neither 1-D nor 2-D, or its trailing dimension does not
        match ``W``'s module axis ``K``.
    """
    if W.ndim != 2:
        raise ValueError(
            f"W must be 2-D (G, K); got shape {tuple(W.shape)}."
        )
    K = W.shape[1]
    if s.ndim == 1:
        # Single-leaf case: (G, K) * (1, K) -> (G, K).
        if s.shape[0] != K:
            raise ValueError(
                f"s has length {s.shape[0]} but W has K={K} modules."
            )
        return W * s[None, :]
    if s.ndim == 2:
        # Multi-leaf case: (1, G, K) * (D, 1, K) -> (D, G, K).
        if s.shape[1] != K:
            raise ValueError(
                f"s has trailing dim {s.shape[1]} but W has K={K} modules."
            )
        return W[None, :, :] * s[:, None, :]
    raise ValueError(
        f"s must be 1-D (K,) or 2-D (D, K); got ndim={s.ndim}."
    )


def _inv_softplus(y: float) -> float:
    """Inverse of ``softplus`` for initializing unconstrained scale params.

    ``softplus(x) = log(1 + exp(x))``; its inverse is ``x = log(exp(y) - 1)``.
    Used so a guide scale param initialized at ``_inv_softplus(sigma)`` yields
    ``softplus(.) == sigma`` at step 0.

    Implemented with the pure-Python :mod:`math` module (not ``jnp``) on
    purpose: this runs at ``numpyro.param`` init time *inside the JIT-traced
    guide*, where ``jnp`` ops on a Python constant produce a tracer and
    ``float(...)`` would raise ``ConcretizationTypeError``. The input is always
    a concrete Python float (a default scale), so plain ``math`` is exact and
    trace-safe.

    Parameters
    ----------
    y : float
        Target positive value (a standard deviation). Must be ``> 0``.

    Returns
    -------
    float
        The unconstrained pre-softplus value ``x`` with ``softplus(x) == y``.
    """
    # ``expm1`` is the numerically stable ``exp(y) - 1`` for small ``y``.
    return math.log(math.expm1(y))

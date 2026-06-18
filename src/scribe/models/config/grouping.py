"""Multi-factor grouping for hierarchical models.

This module generalises the single ``dataset_key`` grouping axis to an
arbitrary number of grouping **factors** (e.g. ``treatment`` and ``sample``)
forming crossed and/or nested hierarchies, with optional interactions. The
mathematical model is described in ``paper/_hierarchical_datasets.qmd`` (section
"Multi-factor (crossed and nested) hierarchies").

Three user-facing spellings normalise into one canonical :class:`GroupingSpec`:

- ``dataset_key="sample"`` — a single grouping factor (legacy behaviour).
- ``dataset_key=["treatment", "sample"]`` — several **crossed** factors.
- ``hierarchy=[GroupLevel("treatment"), GroupLevel("sample",
  nested_in="treatment")]`` — the structured power-user form for nesting and
  per-factor effect types; crossing is implicit, nesting is explicit.

Key concepts
------------
- A **leaf** is a unique *present* combination of factor levels (the rows that
  actually occur in the data). The leaf axis coincides with the existing
  ``n_datasets`` axis, so the per-cell likelihood is unchanged: leaf index =
  the per-cell ``dataset_indices``.
- Each factor carries its level set and a ``leaf_to_level`` map (leaf index ->
  factor-level index). These are stored as plain Python tuples so the spec
  pickles cleanly on :class:`~scribe.models.config.base.ModelConfig`.
- Prior **families** stay in the existing ``*_dataset_prior`` keyword arguments
  (now ``str | dict[factor -> family]``); this module only resolves and records
  them per factor.

In Milestone 1 the multi-factor leaves flow through the existing single-axis
hierarchy; the additive per-factor decomposition (and ``effect_type`` /
``fixed_scale``) is consumed in Milestone 2.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# ------------------------------------------------------------------------------
# Prior targets
# ------------------------------------------------------------------------------

# Canonical target keys for the five ``*_dataset_prior`` keyword arguments.
# These are the short names used inside :class:`Factor.priors`.
TARGET_NAMES: Tuple[str, ...] = (
    "expression",
    "prob",
    "zero_inflation",
    "overdispersion",
    "regime",
)

# Valid prior families (mirrors ``HierarchicalPriorType`` values).
_VALID_FAMILIES = frozenset({"none", "gaussian", "horseshoe", "neg"})


# ------------------------------------------------------------------------------
# User-facing declaration object
# ------------------------------------------------------------------------------


class GroupLevel(BaseModel):
    """One grouping factor in a multi-factor hierarchy (user-facing).

    Declares the *structure* of a factor (its name, whether it is nested, and
    its effect type). Prior **families** are not declared here — they stay in
    the ``*_dataset_prior`` keyword arguments of :func:`scribe.fit`.

    Parameters
    ----------
    name : str
        Name of the ``adata.obs`` column that defines this factor.
    nested_in : str, optional
        Name of the parent factor this one is nested in. ``None`` (default)
        means the factor is *crossed* with the others (the common case).
    effect_type : {"random", "fixed"}, default="random"
        ``"random"`` — a zero-mean effect with a *learned* shrinkage scale
        (appropriate for grouping factors with enough levels, e.g. donors).
        ``"fixed"`` — a zero-mean effect with a *fixed* scale and no adaptive
        shrinkage (appropriate for a deliberate low-cardinality contrast of
        interest, e.g. a two-level treatment). Consumed in Milestone 2.
    fixed_scale : float, optional
        Fixed prior scale for ``effect_type="fixed"`` factors. Must be > 0.
        When ``None`` a per-target default is used downstream. Consumed in
        Milestone 2.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    nested_in: Optional[str] = None
    effect_type: str = Field("random")
    fixed_scale: Optional[float] = None

    def __init__(self, name: Optional[str] = None, **data):
        # Allow the ergonomic positional form ``GroupLevel("treatment")`` in
        # addition to the keyword form ``GroupLevel(name="treatment")``.
        if name is not None:
            data["name"] = name
        super().__init__(**data)

    @field_validator("effect_type")
    @classmethod
    def _check_effect_type(cls, v: str) -> str:
        if v not in ("random", "fixed"):
            raise ValueError(
                f"effect_type must be 'random' or 'fixed', got {v!r}."
            )
        return v

    @field_validator("fixed_scale")
    @classmethod
    def _check_fixed_scale(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (v > 0):
            raise ValueError(f"fixed_scale must be > 0, got {v!r}.")
        return v


# ------------------------------------------------------------------------------
# Canonical (normalized) objects
# ------------------------------------------------------------------------------


class Factor(BaseModel):
    """A resolved grouping factor with its level set and leaf->level map.

    Attributes
    ----------
    name : str
        Factor name. For interaction factors this is the ``":"``-joined name
        of the operands (e.g. ``"treatment:sample"``).
    kind : {"base", "interaction"}
        ``"base"`` for a directly-observed factor; ``"interaction"`` for a
        derived factor whose levels are present operand combinations.
    nested_in : str, optional
        Parent factor name when nested; ``None`` otherwise.
    effect_type : {"random", "fixed"}
        See :class:`GroupLevel`.
    fixed_scale : float, optional
        See :class:`GroupLevel`.
    levels : tuple of str
        Ordered human-readable level labels (``L_f`` of them).
    leaf_to_level : tuple of int
        Length = number of leaves; ``leaf_to_level[leaf]`` is the index into
        ``levels`` for that leaf.
    priors : dict
        Map ``target -> family`` for the targets where this factor carries a
        (non-"none") dataset-level prior. Keys are a subset of
        :data:`TARGET_NAMES`; values are in {"gaussian", "horseshoe", "neg"}.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: str = "base"
    nested_in: Optional[str] = None
    effect_type: str = "random"
    fixed_scale: Optional[float] = None
    levels: Tuple[str, ...]
    leaf_to_level: Tuple[int, ...]
    priors: Dict[str, str] = Field(default_factory=dict)

    @property
    def n_levels(self) -> int:
        """Number of levels (``L_f``)."""
        return len(self.levels)


class GroupingSpec(BaseModel):
    """Canonical normalized grouping descriptor — the single internal form.

    Stores only leaf-level structure (factors with their levels and
    ``leaf_to_level`` maps, plus leaf labels). The per-cell leaf assignment
    lives at runtime as ``dataset_indices`` and is not stored here, so the spec
    pickles small and is data-independent.

    Attributes
    ----------
    factors : tuple of Factor
        Ordered factors (base factors in declaration order, followed by any
        interaction factors).
    leaf_labels : tuple of str
        Length = ``n_leaves``; human-readable label per leaf (e.g.
        ``"panobinostat | D3"``).
    n_leaves : int
        Number of present leaf combinations. Equals
        ``ModelConfig.n_datasets``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    factors: Tuple[Factor, ...]
    leaf_labels: Tuple[str, ...]
    n_leaves: int

    @property
    def factor_names(self) -> Tuple[str, ...]:
        """Names of all factors (base, then interaction)."""
        return tuple(f.name for f in self.factors)

    @property
    def base_factors(self) -> Tuple[Factor, ...]:
        """The base (non-interaction) factors, in order."""
        return tuple(f for f in self.factors if f.kind == "base")

    def leaf_coords(self) -> List[Dict[str, str]]:
        """Return the present-combination table: leaf index -> {factor: level}.

        Uses only the base factors (interaction levels are derived). The result
        is the human-readable coordinate of each leaf, used for labelling
        comparisons (donor IDs, condition names) instead of ``dataset_<i>``.
        """
        bases = self.base_factors
        coords: List[Dict[str, str]] = []
        for leaf in range(self.n_leaves):
            coords.append(
                {f.name: f.levels[f.leaf_to_level[leaf]] for f in bases}
            )
        return coords


# ------------------------------------------------------------------------------
# Prior-dict resolution
# ------------------------------------------------------------------------------


def resolve_dataset_prior_dict(
    value: Union[str, Dict[str, str]],
    factor_names: Tuple[str, ...],
) -> Dict[str, str]:
    """Resolve a ``*_dataset_prior`` value into a per-factor family map.

    Parameters
    ----------
    value : str or dict
        A bare family string (broadcast to every factor) or a
        ``{factor_name -> family}`` dict.
    factor_names : tuple of str
        All declared factor names (base + interaction). Dict keys are validated
        against this set.

    Returns
    -------
    dict
        ``{factor_name -> family}`` for every factor in ``factor_names``;
        factors absent from a dict default to ``"none"``.

    Raises
    ------
    ValueError
        If ``value`` is not a str/dict, a family is not one of
        {"none", "gaussian", "horseshoe", "neg"}, or a dict key is not a
        declared factor name.
    """
    if isinstance(value, str):
        family = value.lower()
        if family not in _VALID_FAMILIES:
            raise ValueError(
                f"Unknown dataset-prior family {value!r}. "
                f"Valid families: {sorted(_VALID_FAMILIES)}."
            )
        return {name: family for name in factor_names}

    if isinstance(value, dict):
        resolved: Dict[str, str] = {name: "none" for name in factor_names}
        for key, fam in value.items():
            if key not in resolved:
                raise ValueError(
                    f"dataset-prior dict key {key!r} is not a declared factor. "
                    f"Declared factors: {list(factor_names)}."
                )
            fam_l = str(fam).lower()
            if fam_l not in _VALID_FAMILIES:
                raise ValueError(
                    f"Unknown dataset-prior family {fam!r} for factor {key!r}. "
                    f"Valid families: {sorted(_VALID_FAMILIES)}."
                )
            resolved[key] = fam_l
        return resolved

    raise ValueError(
        f"dataset prior must be a str or dict[factor -> family], "
        f"got {type(value).__name__}."
    )


def _reduce_leaf_axis_family(spec: "GroupingSpec", target: str) -> str:
    """Reduce per-factor families for ``target`` to one leaf-axis family.

    In Milestone 1 the multi-factor leaves flow through the existing single
    ``*_dataset_prior`` enum, which needs a single family. We return ``"none"``
    when no factor carries the target, otherwise the first non-"none" family in
    factor order. (Milestone 2 consumes the full per-factor families directly.)
    """
    for factor in spec.factors:
        fam = factor.priors.get(target, "none")
        if fam != "none":
            return fam
    return "none"


# ------------------------------------------------------------------------------
# Declaration normalization (structure only, no data)
# ------------------------------------------------------------------------------


class _FactorDecl:
    """Lightweight factor declaration (name + structure, no levels)."""

    __slots__ = ("name", "kind", "nested_in", "effect_type", "fixed_scale", "operands")

    def __init__(
        self,
        name: str,
        kind: str,
        nested_in: Optional[str],
        effect_type: str,
        fixed_scale: Optional[float],
        operands: Optional[Tuple[str, ...]] = None,
    ):
        self.name = name
        self.kind = kind
        self.nested_in = nested_in
        self.effect_type = effect_type
        self.fixed_scale = fixed_scale
        self.operands = operands  # only for interaction factors


def _declare_factors(
    dataset_key: Optional[Union[str, List[str]]],
    hierarchy: Optional[List[GroupLevel]],
    interactions: Optional[List[Tuple[str, ...]]],
) -> List[_FactorDecl]:
    """Normalize the three spellings into an ordered list of factor declarations.

    Validates mutual exclusivity, interaction operands, and ``nested_in``
    references. Does not touch the data (no levels/maps yet).
    """
    if dataset_key is not None and hierarchy is not None:
        raise ValueError(
            "Pass either `dataset_key` or `hierarchy`, not both. "
            "`dataset_key` (str or list) declares crossed factors; "
            "`hierarchy` (list of GroupLevel) is the structured form for "
            "nesting and per-factor effect types."
        )

    decls: List[_FactorDecl] = []
    if hierarchy is not None:
        for gl in hierarchy:
            decls.append(
                _FactorDecl(
                    name=gl.name,
                    kind="base",
                    nested_in=gl.nested_in,
                    effect_type=gl.effect_type,
                    fixed_scale=gl.fixed_scale,
                )
            )
    elif dataset_key is not None:
        names = [dataset_key] if isinstance(dataset_key, str) else list(dataset_key)
        for name in names:
            decls.append(
                _FactorDecl(
                    name=name,
                    kind="base",
                    nested_in=None,
                    effect_type="random",
                    fixed_scale=None,
                )
            )
    else:
        if interactions:
            raise ValueError(
                "interactions require grouping factors; pass `dataset_key` "
                "(str or list) or `hierarchy` alongside `interactions`."
            )
        return []

    base_names = [d.name for d in decls]
    if len(set(base_names)) != len(base_names):
        raise ValueError(f"Duplicate factor names in grouping: {base_names}.")

    # Validate nested_in references an earlier-declared factor (no cycles/forward refs).
    seen: set = set()
    for d in decls:
        if d.nested_in is not None:
            if d.nested_in == d.name:
                raise ValueError(f"Factor {d.name!r} cannot be nested in itself.")
            if d.nested_in not in seen:
                raise ValueError(
                    f"Factor {d.name!r} is nested_in {d.nested_in!r}, which is "
                    f"not a factor declared before it. Declare the parent first."
                )
        seen.add(d.name)

    # Interaction factors (derived).
    if interactions:
        if len(base_names) < 2:
            raise ValueError(
                "interactions require >= 2 grouping factors; got "
                f"{base_names}."
            )
        for ops in interactions:
            ops_t = tuple(ops)
            if len(ops_t) < 2:
                raise ValueError(
                    f"each interaction needs >= 2 operands, got {ops_t}."
                )
            for op in ops_t:
                if op not in base_names:
                    raise ValueError(
                        f"interaction operand {op!r} is not a declared factor. "
                        f"Declared factors: {base_names}."
                    )
            inter_name = ":".join(ops_t)
            if inter_name in [d.name for d in decls]:
                raise ValueError(f"Duplicate interaction factor {inter_name!r}.")
            decls.append(
                _FactorDecl(
                    name=inter_name,
                    kind="interaction",
                    nested_in=None,
                    effect_type="random",
                    fixed_scale=None,
                    operands=ops_t,
                )
            )

    return decls


# ------------------------------------------------------------------------------
# Full normalization (uses the data)
# ------------------------------------------------------------------------------


def normalize_grouping(
    *,
    dataset_key: Optional[Union[str, List[str]]],
    hierarchy: Optional[List[GroupLevel]],
    interactions: Optional[List[Tuple[str, ...]]],
    obs,
    dataset_priors: Dict[str, Union[str, Dict[str, str]]],
) -> Optional[Tuple[GroupingSpec, np.ndarray]]:
    """Build the canonical :class:`GroupingSpec` and per-cell leaf index.

    Leaves are the **present** combinations of the base factors, computed with
    ``numpy.unique(..., axis=0, return_inverse=True)`` so that absent
    combinations are never allocated. Level order within a factor follows the
    pandas categorical category order; leaf order is the deterministic order of
    ``numpy.unique`` over the per-factor code rows.

    Parameters
    ----------
    dataset_key, hierarchy, interactions
        The three user-facing spellings (see module docstring).
    obs : pandas.DataFrame
        ``adata.obs``; factor names must be columns.
    dataset_priors : dict
        ``{target -> (str | dict)}`` for the five targets in
        :data:`TARGET_NAMES`. Each value is resolved per factor.

    Returns
    -------
    (GroupingSpec, numpy.ndarray) or None
        The spec plus the per-cell leaf index (int32, shape ``(n_cells,)``).
        ``None`` when no grouping was requested.

    Raises
    ------
    ValueError
        On any declaration/validation error or a missing ``obs`` column.
    """
    decls = _declare_factors(dataset_key, hierarchy, interactions)
    if not decls:
        return None

    base_decls = [d for d in decls if d.kind == "base"]
    inter_decls = [d for d in decls if d.kind == "interaction"]

    # Validate columns exist.
    for d in base_decls:
        if d.name not in obs.columns:
            raise ValueError(
                f"Grouping factor {d.name!r} not found in adata.obs. "
                f"Available columns: {list(obs.columns)}."
            )

    # Per-base-factor categorical codes + level labels (pandas category order).
    base_levels: List[Tuple[str, ...]] = []
    code_cols: List[np.ndarray] = []
    for d in base_decls:
        cat = obs[d.name].astype("category")
        codes = np.asarray(cat.cat.codes.values, dtype=np.int64)
        # Missing/NaN values get categorical code -1; left in place they would
        # silently index the *last* level when building leaf labels, producing
        # wrong/duplicated leaves. Reject them explicitly.
        n_missing = int((codes < 0).sum())
        if n_missing:
            raise ValueError(
                f"Grouping factor {d.name!r} has {n_missing} cell(s) with "
                f"missing/NaN values. Every cell must have a defined value for "
                f"each grouping factor; filter or impute these cells first."
            )
        base_levels.append(tuple(str(c) for c in cat.cat.categories))
        code_cols.append(codes)

    # Present leaf combinations via numpy.unique over the code matrix.
    code_matrix = np.stack(code_cols, axis=1)  # (n_cells, n_base)
    unique_rows, leaf_index = np.unique(
        code_matrix, axis=0, return_inverse=True
    )
    leaf_index = np.asarray(leaf_index, dtype=np.int32).reshape(-1)
    n_leaves = int(unique_rows.shape[0])

    # Resolve per-factor prior families (need all factor names first).
    factor_names = tuple(d.name for d in decls)
    resolved_priors: Dict[str, Dict[str, str]] = {}
    for target in TARGET_NAMES:
        value = dataset_priors.get(target, "none")
        resolved_priors[target] = resolve_dataset_prior_dict(value, factor_names)

    def _priors_for(name: str) -> Dict[str, str]:
        out = {}
        for target in TARGET_NAMES:
            fam = resolved_priors[target].get(name, "none")
            if fam != "none":
                out[target] = fam
        return out

    factors: List[Factor] = []

    # Base factors.
    for j, d in enumerate(base_decls):
        factors.append(
            Factor(
                name=d.name,
                kind="base",
                nested_in=d.nested_in,
                effect_type=d.effect_type,
                fixed_scale=d.fixed_scale,
                levels=base_levels[j],
                leaf_to_level=tuple(int(x) for x in unique_rows[:, j]),
                priors=_priors_for(d.name),
            )
        )

    # Interaction factors: levels are present operand combinations.
    base_index = {d.name: j for j, d in enumerate(base_decls)}
    for d in inter_decls:
        op_cols = [base_index[op] for op in d.operands]
        pair_rows = unique_rows[:, op_cols]  # (n_leaves, k)
        inter_unique, inter_l2l = np.unique(
            pair_rows, axis=0, return_inverse=True
        )
        inter_l2l = np.asarray(inter_l2l, dtype=np.int32).reshape(-1)
        inter_levels = tuple(
            " x ".join(
                base_levels[op_cols[m]][int(inter_unique[r, m])]
                for m in range(len(op_cols))
            )
            for r in range(inter_unique.shape[0])
        )
        factors.append(
            Factor(
                name=d.name,
                kind="interaction",
                nested_in=None,
                effect_type=d.effect_type,
                fixed_scale=d.fixed_scale,
                levels=inter_levels,
                leaf_to_level=tuple(int(x) for x in inter_l2l),
                priors=_priors_for(d.name),
            )
        )

    # Leaf labels from the present base-factor combinations.
    leaf_labels = tuple(
        " | ".join(
            base_levels[j][int(unique_rows[leaf, j])]
            for j in range(len(base_decls))
        )
        for leaf in range(n_leaves)
    )

    spec = GroupingSpec(
        factors=tuple(factors),
        leaf_labels=leaf_labels,
        n_leaves=n_leaves,
    )
    return spec, leaf_index

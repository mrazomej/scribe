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

from .parameter_mapping import (
    DESCRIPTIVE_NAMES,
    HIERARCHY_TARGET_BY_SITE,
    _DESCRIPTIVE_TO_INTERNAL,
)

# ------------------------------------------------------------------------------
# Prior targets
# ------------------------------------------------------------------------------

# Canonical target keys used inside :class:`Factor.priors`. ``expression`` and
# the four technical targets back the legacy ``*_dataset_prior`` kwargs;
# ``dispersion`` (NB size ``r``) has no flat kwarg and is reached only via the
# unified ``priors`` dict, feeding the factory's condition-specific-r path.
TARGET_NAMES: Tuple[str, ...] = (
    "expression",
    "dispersion",
    "prob",
    "zero_inflation",
    "overdispersion",
    "regime",
)

# Valid prior families (mirrors ``HierarchicalPriorType`` values).
_VALID_FAMILIES = frozenset({"none", "gaussian", "horseshoe", "neg"})


class PriorFamilySpec(BaseModel):
    """Normalized hierarchical-prior family + its hyperparameters.

    The single internal value model for a (target, level) prior. A bare family
    name (``"gaussian"``) carries no hyperparameters; the dict form
    (``{"type": "horseshoe", "tau0": 1.0, ...}``) carries per-spec
    hyperparameters that supersede any global defaults. Stored inside
    :attr:`Factor.priors` (``{target -> PriorFamilySpec}``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str = "none"
    # Regularized-horseshoe hyperparameters.
    tau0: Optional[float] = None
    slab_df: Optional[int] = None
    slab_scale: Optional[float] = None
    # NEG hyperparameters.
    u: Optional[float] = None
    a: Optional[float] = None
    tau: Optional[float] = None
    # Probability-hierarchy structure (``"scalar"``/``"gene_specific"``/
    # ``"two_level"``); only meaningful on the probability target.
    mode: Optional[str] = None

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        v = str(v).lower()
        if v not in _VALID_FAMILIES:
            raise ValueError(
                f"Unknown prior family {v!r}. "
                f"Valid families: {sorted(_VALID_FAMILIES)}."
            )
        return v

    @classmethod
    def from_value(
        cls, value: "Union[str, Dict, 'PriorFamilySpec']"
    ) -> "PriorFamilySpec":
        """Coerce a string / dict-spec / spec into a :class:`PriorFamilySpec`."""
        if isinstance(value, PriorFamilySpec):
            return value
        if isinstance(value, str):
            return cls(type=value)
        if isinstance(value, dict):
            if "type" not in value:
                raise ValueError(
                    "dict-form prior family must contain a 'type' key, "
                    f"got keys {sorted(value)}."
                )
            return cls(**value)
        raise ValueError(
            "prior family must be a str, a dict with a 'type' key, or a "
            f"PriorFamilySpec; got {type(value).__name__}."
        )

    @property
    def is_none(self) -> bool:
        """True when this is the no-hierarchy sentinel family."""
        return self.type == "none"


# Shared no-hierarchy sentinel.
NONE_FAMILY = PriorFamilySpec(type="none")


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
    priors: Dict[str, PriorFamilySpec] = Field(default_factory=dict)

    @field_validator("priors", mode="before")
    @classmethod
    def _coerce_priors(cls, v):
        # Accept legacy ``{target -> str}`` (old pickles / direct construction)
        # and dict-form family specs; normalize every value to a
        # PriorFamilySpec so the field invariant holds locally.
        if isinstance(v, dict):
            return {
                key: PriorFamilySpec.from_value(val) for key, val in v.items()
            }
        return v

    @property
    def n_levels(self) -> int:
        """Number of levels (``L_f``)."""
        return len(self.levels)

    def family(self, target: str) -> str:
        """Family *type* for ``target`` (``"none"`` when not hierarchicalized).

        Parameters
        ----------
        target : str
            One of :data:`TARGET_NAMES` (e.g. ``"expression"``,
            ``"dispersion"``) — the model target whose dataset-level prior
            family is requested.

        Returns
        -------
        str
            The family type (``"gaussian"``, ``"horseshoe"``, ``"neg"``), or
            ``"none"`` when ``target`` carries no dataset-level prior on this
            factor.

        Notes
        -----
        Tolerates the legacy ``{target: str}`` form found in pickles written by
        older `scribe` versions: pydantic's ``_coerce_priors`` validator (which
        would upgrade those strings to :class:`PriorFamilySpec`) does not re-run
        when a model is unpickled, so a value may still be a bare family-name
        string here. In that case the string *is* the family type and is
        returned as-is, keeping cached hierarchical fits loadable.
        """
        spec = self.priors.get(target)
        if spec is None:
            return "none"
        return spec if isinstance(spec, str) else spec.type


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
    value: Union[str, Dict],
    factor_names: Tuple[str, ...],
) -> Dict[str, PriorFamilySpec]:
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
        spec = PriorFamilySpec.from_value(value)
        return {name: spec for name in factor_names}

    if isinstance(value, dict):
        resolved: Dict[str, PriorFamilySpec] = {
            name: NONE_FAMILY for name in factor_names
        }
        for key, fam in value.items():
            if key not in resolved:
                raise ValueError(
                    f"dataset-prior dict key {key!r} is not a declared factor. "
                    f"Declared factors: {list(factor_names)}."
                )
            resolved[key] = PriorFamilySpec.from_value(fam)
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
        fam = factor.family(target)
        if fam != "none":
            return fam
    return "none"


def _resolve_prior_site(name: str) -> str:
    """Resolve a user ``priors`` key (canonical name or internal site) to a site."""
    if name in _DESCRIPTIVE_TO_INTERNAL:
        return _DESCRIPTIVE_TO_INTERNAL[name]
    if name in DESCRIPTIVE_NAMES:  # already an internal site name
        return name
    raise ValueError(
        f"Unknown prior target {name!r}. Use a canonical parameter name "
        f"(e.g. 'mean_expression', 'dispersion', 'probability', 'odds_ratio', "
        f"'zero_inflation', 'overdispersion') or an internal site name."
    )


def normalize_unified_priors(
    priors: Optional[Dict],
    level_names: Tuple[str, ...],
) -> Tuple[
    Dict[str, object],
    Dict[str, PriorFamilySpec],
    Dict[str, Dict[str, PriorFamilySpec]],
]:
    """Parse the unified ``priors`` dict per the routing contract.

    Each ``priors[name] = value`` entry is bucketed by value *shape* (the
    discriminator) after resolving ``name`` to an internal site:

    - ``tuple`` -> **base** prior hyperparameters (``ModelConfig.priors``).
    - ``{"type": ...}`` on ``loadings`` -> **base** W-strategy spec.
    - family ``str``, or ``{"type": ...}`` with **no** level keys, on a core
      param -> **gene-level** family selector (the old ``*_prior`` field).
    - dict **without** ``"type"`` (a level-mapping; keys are ``GroupLevel``
      names plus the reserved ``"base"``) -> **dataset/factor** hierarchy;
      a ``"base"`` key routes to the gene-level selector.

    The reserved keys ``"base"`` and ``"type"`` may not be ``GroupLevel``
    names (enforced at hierarchy declaration).

    Returns
    -------
    (base, gene_level, hierarchical)
        ``base`` : ``{original key -> raw value}`` (tuples / W-strategy dicts
        / unrecognized pass-through values; resolved later by with_priors).
        ``gene_level`` : ``{site -> PriorFamilySpec}``.
        ``hierarchical`` : ``{internal target -> {level -> PriorFamilySpec}}``.
    """
    base: Dict[str, object] = {}
    gene_level: Dict[str, PriorFamilySpec] = {}
    hierarchical: Dict[str, Dict[str, PriorFamilySpec]] = {}
    if not priors:
        return base, gene_level, hierarchical

    for name, value in priors.items():
        # Base hyperparameter overrides (tuples) and any other unrecognized
        # value pass through to ``base`` with their ORIGINAL key; downstream
        # ``with_priors`` / ``normalize_prior_keys`` resolves them. This
        # preserves raw hyperprior-override keys (e.g. ``logit_p_loc``).
        if isinstance(value, tuple):
            base[name] = value
            continue

        # A single family-spec (dict carrying the reserved 'type' key, no
        # per-level structure).
        if isinstance(value, dict) and "type" in value:
            site = _resolve_prior_site(name)
            if site == "W":
                base[name] = value  # loadings W-strategy spec (raw)
            else:
                gene_level[site] = PriorFamilySpec.from_value(value)
            continue

        # Bare family name -> gene-level selector.
        if isinstance(value, str):
            site = _resolve_prior_site(name)
            gene_level[site] = PriorFamilySpec.from_value(value)
            continue

        # Level-mapping -> dataset/factor hierarchy (+ optional 'base').
        if isinstance(value, dict):
            site = _resolve_prior_site(name)
            target = HIERARCHY_TARGET_BY_SITE.get(site)
            for level, fam in value.items():
                if level == "base":
                    gene_level[site] = PriorFamilySpec.from_value(fam)
                    continue
                if level not in level_names:
                    raise ValueError(
                        f"priors[{name!r}] references level {level!r}, which is "
                        f"not a declared grouping level. Declared levels: "
                        f"{sorted(level_names)} (plus the reserved 'base')."
                    )
                if target is None:
                    raise ValueError(
                        f"priors[{name!r}] requests a dataset/factor hierarchy, "
                        f"but {name!r} (site {site!r}) has no hierarchy target."
                    )
                hierarchical.setdefault(target, {})[level] = (
                    PriorFamilySpec.from_value(fam)
                )
            continue

        # Anything else (arrays, custom base specs) -> base pass-through;
        # validated downstream by with_priors.
        base[name] = value

    return base, gene_level, hierarchical


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
    resolved_priors: Dict[str, Dict[str, PriorFamilySpec]] = {}
    for target in TARGET_NAMES:
        value = dataset_priors.get(target, "none")
        resolved_priors[target] = resolve_dataset_prior_dict(value, factor_names)

    def _priors_for(name: str) -> Dict[str, PriorFamilySpec]:
        out: Dict[str, PriorFamilySpec] = {}
        for target in TARGET_NAMES:
            spec = resolved_priors[target].get(name, NONE_FAMILY)
            if not spec.is_none:
                out[target] = spec
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

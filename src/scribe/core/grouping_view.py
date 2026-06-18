"""Indexable views over the leaf grouping structure of a multi-factor fit.

A multi-factor hierarchical fit (``dataset_key=[...]`` / ``hierarchy=[...]``)
flattens the present combinations of its grouping factors into a single leaf
axis. ``GroupView`` lets downstream code recover a *slice* of that grid — the
set of leaves that share fixed level(s) of one or more factors — and
materialise each leaf as a single-dataset results view.

Fixing one factor (``results.get_group(sample="D3")``) returns however many
leaves share that level: two for a control/disease contrast, three or more for
a multi-arm design. The view is indexed by the *free* factors (those not pinned
by the filter), so ``g["control"]`` / ``g["panobinostat"]`` return the
per-condition leaf views for that sample.

The helpers here are results-agnostic: they take any object exposing
``model_config.grouping_spec`` and ``get_dataset(index)``, so a single
implementation serves both the SVI and MCMC results objects.
"""

from typing import Dict, Iterator, List, Tuple


def _grouping_spec(results):
    return getattr(getattr(results, "model_config", None), "grouping_spec", None)


def _require_spec(results):
    spec = _grouping_spec(results)
    if spec is None:
        raise ValueError(
            "get_group()/iter_groups()/group_levels() require a multi-factor "
            "fit (results.model_config.grouping_spec must be set; build it with "
            "scribe.fit(..., dataset_key=[...]) or hierarchy=[...])."
        )
    return spec


def _base_factor_names(spec) -> List[str]:
    return [f.name for f in spec.base_factors]


class GroupView:
    """A slice of the leaf grid sharing fixed factor level(s).

    Returned by :meth:`get_group`. Holds the leaf indices whose coordinates
    match the pinned levels, exposes their labels/coordinates, and lazily
    materialises each leaf as a single-dataset results view via the parent's
    ``get_dataset``. Index it by the *free* factor level(s) — the factors not
    pinned by the filter (a plain level when one factor is free, a tuple of
    levels in declared order when several are).

    Attributes
    ----------
    fixed : dict
        The ``{factor: level}`` filter that produced this group.
    leaves : list of int
        Leaf indices in the group, ascending.
    coords : list of dict
        Per-leaf ``{factor: level}`` coordinate, aligned with ``leaves``.
    free_factors : list of str
        Base factors not pinned by ``fixed`` — the indexing key axes.
    """

    def __init__(self, results, fixed, leaves, coords, free_factors):
        self._results = results
        self.fixed: Dict[str, str] = dict(fixed)
        self.leaves: List[int] = list(leaves)
        self.coords: List[Dict[str, str]] = list(coords)
        self.free_factors: List[str] = list(free_factors)

    @property
    def labels(self) -> List[str]:
        """Human-readable leaf label per leaf (e.g. ``"panobinostat | D3"``)."""
        spec = _grouping_spec(self._results)
        return [spec.leaf_labels[i] for i in self.leaves]

    def _key_for(self, coord):
        """Index key for a leaf: the free-factor level(s)."""
        if len(self.free_factors) == 1:
            return coord[self.free_factors[0]]
        return tuple(coord[f] for f in self.free_factors)

    def keys(self) -> list:
        """Index keys of the group, aligned with ``leaves``."""
        return [self._key_for(c) for c in self.coords]

    def leaf_for(self, key) -> int:
        """Leaf index for a free-factor key (without materialising the view)."""
        for leaf, coord in zip(self.leaves, self.coords):
            if self._key_for(coord) == key:
                return leaf
        raise KeyError(
            f"no leaf with {self.free_factors}={key!r} in this group; "
            f"available keys: {self.keys()}."
        )

    def __getitem__(self, key):
        """Materialise the leaf for ``key`` as a single-dataset results view."""
        return self._results.get_dataset(self.leaf_for(key))

    def view(self, key=None):
        """Single-dataset view for ``key`` (or the sole leaf when unambiguous)."""
        if key is None:
            if len(self.leaves) != 1:
                raise ValueError(
                    f"view() needs a key: the group has {len(self.leaves)} "
                    f"leaves (keys: {self.keys()})."
                )
            return self._results.get_dataset(self.leaves[0])
        return self[key]

    @property
    def dataset(self):
        """The single leaf's view; errors unless the group holds exactly one."""
        return self.view()

    def views(self) -> dict:
        """``{key: single-dataset view}`` for every leaf in the group."""
        return {
            self._key_for(c): self._results.get_dataset(leaf)
            for leaf, c in zip(self.leaves, self.coords)
        }

    def items(self):
        """Iterate ``(key, single-dataset view)`` pairs."""
        for leaf, coord in zip(self.leaves, self.coords):
            yield self._key_for(coord), self._results.get_dataset(leaf)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        return any(self._key_for(c) == key for c in self.coords)

    def __len__(self):
        return len(self.leaves)

    def __repr__(self):
        return (
            f"GroupView(fixed={self.fixed}, free={self.free_factors}, "
            f"leaves={self.leaves}, keys={self.keys()})"
        )


def group_levels(results, factor: str) -> List[str]:
    """Present levels of a base ``factor``, in its declared order."""
    spec = _require_spec(results)
    names = _base_factor_names(spec)
    if factor not in names:
        raise ValueError(f"unknown factor {factor!r}; base factors: {names}.")
    coords = spec.leaf_coords()
    present = {c[factor] for c in coords}
    fac = next(f for f in spec.base_factors if f.name == factor)
    return [lv for lv in fac.levels if lv in present]


def get_group(results, **factor_levels) -> GroupView:
    """Return the leaves whose coordinates match the fixed factor level(s).

    Parameters
    ----------
    results : object
        A multi-factor results object (``model_config.grouping_spec`` set).
    **factor_levels
        ``factor=level`` filters over base factors. Pin a subset (commonly one,
        e.g. ``sample="D3"``); the unpinned factors become the group's index.

    Returns
    -------
    GroupView
        The matching leaves, indexable by the free factor level(s).
    """
    spec = _require_spec(results)
    names = _base_factor_names(spec)
    if not factor_levels:
        raise ValueError(
            "get_group() needs at least one factor=level filter; "
            f"base factors: {names}."
        )
    bad = [k for k in factor_levels if k not in names]
    if bad:
        raise ValueError(f"unknown factor(s) {bad}; base factors: {names}.")

    coords = spec.leaf_coords()
    leaves, kept = [], []
    for leaf, coord in enumerate(coords):
        if all(coord.get(k) == v for k, v in factor_levels.items()):
            leaves.append(leaf)
            kept.append(coord)
    if not leaves:
        raise ValueError(
            f"no present leaves match {factor_levels}; only realised "
            "factor combinations exist (check the level spellings)."
        )
    free = [n for n in names if n not in factor_levels]
    return GroupView(results, factor_levels, leaves, kept, free)


def iter_groups(results, by: str) -> Iterator[Tuple[str, GroupView]]:
    """Yield ``(level, GroupView)`` for every present level of factor ``by``.

    Enumerates the same grouping ``compare_groups`` pairs over — e.g.
    ``for sample, g in results.iter_groups("sample"): g["control"], ...``.
    """
    for level in group_levels(results, by):
        yield level, get_group(results, **{by: level})

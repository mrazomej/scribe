"""
Semantic axis metadata for parameter tensors.

This module provides the :class:`AxisLayout` descriptor that tracks what each
axis of a parameter tensor represents (genes, components, datasets, cells,
etc.).  Layouts are built from
:class:`~scribe.models.builders.parameter_specs.ParamSpec` at inference time and
travel with the results objects so that downstream code never needs to *guess*
axis semantics from ``ndim`` / ``shape`` heuristics.

For backward compatibility with pickles that predate this module,
:func:`infer_layout` reconstructs a layout from tensor shape and the scalar
metadata fields on :class:`~scribe.models.config.ModelConfig` that always
survive serialisation (``n_components``, ``n_datasets``, ``mixture_params``,
etc.).

Classes
-------
AxisLayout
    Frozen dataclass describing the semantic axes of a parameter tensor.

Functions
---------
layout_from_param_spec
    Build an ``AxisLayout`` from a ``ParamSpec`` (exact, no heuristics).
infer_layout
    Reconstruct an ``AxisLayout`` from tensor shape + config metadata
    (backward-compatibility path).
build_param_layouts
    Build a full ``{key: AxisLayout}`` dict from ``param_specs`` and a
    parameter dictionary.
reconstruct_param_layouts
    Reconstruct layouts for all keys in a parameter dictionary using only
    config metadata and tensor shapes (no ``param_specs`` required).
align_to_layout
    Insert singleton dimensions so that a tensor broadcasts to a target layout.
merge_layouts
    Compute the union of several layouts (all axes from all inputs, in
    canonical order).
broadcast_param_to_layout
    Like ``align_to_layout`` but auto-detects a leading batch/cell
    dimension that is not described by the layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import jax.numpy as jnp

if TYPE_CHECKING:
    from scribe.models.builders.parameter_specs import ParamSpec

# Canonical axis names, in the order that ``resolve_shape`` prepends them.
COMPONENTS = "components"
DATASETS = "datasets"
GENES = "genes"
CELLS = "cells"
SAMPLES = "samples"

# Maps symbolic shape-dim names from ParamSpec to canonical axis names.
_DIM_TO_AXIS = {
    "n_genes": GENES,
    "n_cells": CELLS,
}

# Canonical ordering: components > datasets > genes/cells.
# This matches the convention in ``resolve_shape``.
_AXIS_ORDER = {COMPONENTS: 0, DATASETS: 1, GENES: 2, CELLS: 2}

# Frozen parameter name tables for backward-compat layout reconstruction.
#
# These sets are consulted ONLY by ``infer_layout()`` — the fallback
# path for old pickles that lack ``param_specs``.  They must NOT be
# extended for new parameters.  New parameters should be recognised
# exclusively through their ``ParamSpec`` via ``layout_from_param_spec``.
_KNOWN_GENE_PARAMS = frozenset({"r", "mu", "phi", "gate"})
_KNOWN_CELL_PARAMS = frozenset({"p_capture", "phi_capture", "eta_capture"})
_KNOWN_COMPONENT_ONLY_PARAMS = frozenset({"mixing_weights", "mixing_logits"})


# =========================================================================
# AxisLayout dataclass
# =========================================================================


@dataclass(frozen=True)
class AxisLayout:
    """Semantic description of the axes of a parameter tensor.

    Parameters
    ----------
    axes : tuple of str
        Ordered axis names **excluding** a leading sample dimension.
        Valid names: ``"components"``, ``"datasets"``, ``"genes"``,
        ``"cells"``.  An empty tuple means the parameter is a scalar.
    has_sample_dim : bool, default False
        If ``True`` the tensor carries an additional leading axis for
        posterior draws (MCMC samples or variational samples).  All
        axis-index properties account for this offset automatically.

    Examples
    --------
    >>> layout = AxisLayout(("components", "genes"))
    >>> layout.gene_axis
    1
    >>> layout.with_sample_dim().gene_axis
    2
    """

    axes: Tuple[str, ...]
    has_sample_dim: bool = False

    # ------------------------------------------------------------------
    # Axis index helpers (return None when the axis is absent)
    # ------------------------------------------------------------------

    @property
    def _offset(self) -> int:
        """Leading-axis offset caused by the sample dimension."""
        return 1 if self.has_sample_dim else 0

    @property
    def gene_axis(self) -> Optional[int]:
        """Integer index of the gene axis, or ``None``."""
        if GENES in self.axes:
            return self.axes.index(GENES) + self._offset
        return None

    @property
    def component_axis(self) -> Optional[int]:
        """Integer index of the mixture-component axis, or ``None``."""
        if COMPONENTS in self.axes:
            return self.axes.index(COMPONENTS) + self._offset
        return None

    @property
    def dataset_axis(self) -> Optional[int]:
        """Integer index of the dataset axis, or ``None``."""
        if DATASETS in self.axes:
            return self.axes.index(DATASETS) + self._offset
        return None

    @property
    def cell_axis(self) -> Optional[int]:
        """Integer index of the cell axis, or ``None``."""
        if CELLS in self.axes:
            return self.axes.index(CELLS) + self._offset
        return None

    @property
    def rank(self) -> int:
        """Expected ``ndim`` of the tensor (including sample dim)."""
        return len(self.axes) + self._offset

    # ------------------------------------------------------------------
    # Constructors for common transformations
    # ------------------------------------------------------------------

    def with_sample_dim(self) -> "AxisLayout":
        """Return a copy with the sample-dimension flag set."""
        if self.has_sample_dim:
            return self
        return AxisLayout(axes=self.axes, has_sample_dim=True)

    def without_sample_dim(self) -> "AxisLayout":
        """Return a copy with the sample-dimension flag cleared."""
        if not self.has_sample_dim:
            return self
        return AxisLayout(axes=self.axes, has_sample_dim=False)

    def with_axis(self, axis_name: str) -> "AxisLayout":
        """Return a copy with *axis_name* inserted in canonical order.

        The canonical ordering is components > datasets > genes/cells
        (matching ``_AXIS_ORDER``).  If *axis_name* is already present,
        returns ``self`` unchanged.

        Parameters
        ----------
        axis_name : str
            The axis to add (e.g. ``"datasets"``).

        Returns
        -------
        AxisLayout
            Layout with the new axis inserted.
        """
        if axis_name in self.axes:
            return self
        new_axes = sorted(
            list(self.axes) + [axis_name],
            key=lambda a: _AXIS_ORDER.get(a, 99),
        )
        return AxisLayout(
            axes=tuple(new_axes), has_sample_dim=self.has_sample_dim
        )

    def subset_axis(self, axis_name: str) -> "AxisLayout":
        """Return a copy with *axis_name* removed (after indexing it out).

        Parameters
        ----------
        axis_name : str
            The axis to drop (e.g. ``"datasets"`` after ``get_dataset``).

        Returns
        -------
        AxisLayout
            Layout with the named axis removed.

        Raises
        ------
        ValueError
            If *axis_name* is not present in :attr:`axes`.
        """
        if axis_name not in self.axes:
            raise ValueError(f"Axis {axis_name!r} not in layout {self.axes}")
        new_axes = tuple(a for a in self.axes if a != axis_name)
        return AxisLayout(axes=new_axes, has_sample_dim=self.has_sample_dim)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def broadcast_to(
        self, target: "AxisLayout"
    ) -> Tuple[Union[slice, None], ...]:
        """Compute an indexing tuple that inserts singletons for broadcasting.

        Returns a tuple of ``slice(None)`` (keep) and ``None``
        (``jnp.newaxis``) entries that, when applied to a tensor with
        ``self`` layout, makes it broadcastable against a tensor with
        *target* layout.

        Both layouts must have the same ``has_sample_dim`` setting.

        Parameters
        ----------
        target : AxisLayout
            The layout to broadcast *to*.

        Returns
        -------
        tuple of (slice | None)
            Indexing tuple suitable for ``tensor[idx]``.

        Examples
        --------
        >>> src = AxisLayout(("components",))
        >>> tgt = AxisLayout(("components", "genes"))
        >>> src.broadcast_to(tgt)
        (slice(None, None, None), None)
        """
        # Initialize an empty list to build the indexing tuple for broadcasting.
        idx: list[Union[slice, None]] = []

        # If the source tensor has a sample dimension, preserve it with
        # slice(None).
        if self.has_sample_dim:
            idx.append(slice(None))

        # src_pos tracks the current position in the source axes.
        src_pos = 0

        # Loop through each axis of the target layout.
        for ax in target.axes:
            # If the current target axis matches the next unused source axis,
            # keep the dimension.
            if src_pos < len(self.axes) and self.axes[src_pos] == ax:
                idx.append(slice(None))
                src_pos += 1
            # Otherwise, insert a singleton dimension (None) for broadcasting.
            else:
                idx.append(None)

        # Return the completed indexing tuple, which can be used for
        # broadcasting.
        return tuple(idx)

    # ------------------------------------------------------------------
    # Readable repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sample_str = ", has_sample_dim=True" if self.has_sample_dim else ""
        return f"AxisLayout(axes={self.axes}{sample_str})"


# =========================================================================
# Bulk layout transformations
# =========================================================================


def subset_layouts(
    layouts: Dict[str, AxisLayout],
    axis_name: str,
) -> Dict[str, AxisLayout]:
    """Return a copy of *layouts* with *axis_name* removed where present.

    For each entry whose :attr:`~AxisLayout.axes` contain *axis_name*,
    :meth:`AxisLayout.subset_axis` is called to drop the axis.  Entries
    that do not carry the axis are passed through unchanged.

    This is the canonical way to update a ``param_layouts`` dict after a
    slicing operation that collapses one semantic axis (e.g.
    ``get_dataset`` removes ``"datasets"``, single-component extraction
    removes ``"components"``).

    Parameters
    ----------
    layouts : dict of str to AxisLayout
        Source layout dict (not mutated).
    axis_name : str
        The semantic axis to drop (e.g. ``"datasets"``, ``"components"``).

    Returns
    -------
    dict of str to AxisLayout
        New dict with the axis removed from every layout that had it.

    Examples
    --------
    >>> from scribe.core.axis_layout import AxisLayout, subset_layouts
    >>> layouts = {
    ...     "r": AxisLayout(("components", "datasets", "genes")),
    ...     "p": AxisLayout(("components",)),
    ... }
    >>> subset_layouts(layouts, "datasets")
    {'r': AxisLayout(axes=('components', 'genes')), 'p': AxisLayout(axes=('components',))}
    """
    out: Dict[str, AxisLayout] = {}
    for key, layout in layouts.items():
        if axis_name in layout.axes:
            out[key] = layout.subset_axis(axis_name)
        else:
            out[key] = layout
    return out


# =========================================================================
# Factory: build from ParamSpec (exact path)
# =========================================================================


def layout_from_param_spec(
    spec: "ParamSpec",
    has_sample_dim: bool = False,
) -> AxisLayout:
    """Build an :class:`AxisLayout` from a :class:`ParamSpec`.

    The axis ordering follows the convention of
    :func:`~scribe.models.builders.parameter_specs.resolve_shape`:
    components (outermost) > datasets > shape_dims (genes / cells).

    Parameters
    ----------
    spec : ParamSpec
        The parameter specification.
    has_sample_dim : bool, default False
        Whether the tensor has a leading sample dimension.

    Returns
    -------
    AxisLayout
    """
    axes: list[str] = []
    if spec.is_mixture:
        axes.append(COMPONENTS)
    if getattr(spec, "is_dataset", False):
        axes.append(DATASETS)
    for dim in spec.shape_dims:
        mapped = _DIM_TO_AXIS.get(dim)
        if mapped is not None:
            axes.append(mapped)
    return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)


# =========================================================================
# Backward-compat: reconstruct from tensor shape + config metadata
# =========================================================================


def _strip_param_key(key: str) -> str:
    """Extract the base parameter name from a variational-parameter key.

    Variational parameters use suffixes like ``_loc``, ``_scale`` and
    prefixes like ``log_``.  This function strips those to recover the
    base name used in ``ParamSpec.name`` so we can look it up in the
    known-parameter tables.

    Joint-guide alpha regression coefficients use the pattern
    ``joint_{group}_{nondense}_alpha_{dense}``; the ``_alpha_*`` suffix
    is stripped so the key resolves to the nondense spec name.

    Examples
    --------
    >>> _strip_param_key("r_loc")
    'r'
    >>> _strip_param_key("log_mu_scale")
    'mu'
    >>> _strip_param_key("mixing_weights_loc")
    'mixing_weights'
    >>> _strip_param_key("joint_joint_p_alpha_r")
    'p'
    >>> _strip_param_key("mixing_concentrations")
    'mixing_weights'
    """
    # Known variational aliases that reparameterize a spec under a
    # different name.  Map them to the canonical spec name first.
    _VARIATIONAL_ALIASES = {
        "mixing_concentrations": "mixing_weights",
        "mixing_logits_unconstrained": "mixing_weights",
    }

    # Skip Flax nested-dict keys (e.g. "flow_p$params")
    if "$" in key:
        return key

    name = _VARIATIONAL_ALIASES.get(key, key)

    # Strip joint-guide prefix: "joint_{group}_" → keep the remainder
    if name.startswith("joint_"):
        rest = name[len("joint_") :]
        _, sep, remainder = rest.partition("_")
        if sep and remainder:
            name = remainder

    # Strip nondense alpha regression suffix before other suffixes.
    # Pattern: ``{nondense_name}_alpha_{dense_name}`` → ``{nondense_name}``
    _alpha_idx = name.find("_alpha_")
    if _alpha_idx > 0:
        name = name[:_alpha_idx]

    # Strip common suffixes (longer first to avoid partial matches).  ``_W`` and
    # ``_raw_diag`` appear in structured-guide / low-rank factor names so the
    # base name matches ``ParamSpec.name``.
    for suffix in (
        "_base_loc",
        "_base_scale",
        "_loc",
        "_scale",
        "_W",
        "_raw_diag",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Strip common prefixes
    for prefix in ("log_", "logit_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    return name


def _is_mixture_param(
    base_name: str,
    mixture_params: Optional[List[str]],
) -> bool:
    """Decide whether *base_name* should carry a component axis.

    ``mixture_params=None`` means *all* non-cell-specific params are
    mixture-specific (the scribe default).
    """
    if (
        base_name in _KNOWN_CELL_PARAMS
        or base_name in _KNOWN_COMPONENT_ONLY_PARAMS
    ):
        return False
    if mixture_params is None:
        return True
    return base_name in mixture_params


def _is_dataset_param(
    base_name: str,
    dataset_params: Optional[List[str]],
) -> bool:
    """Decide whether *base_name* should carry a dataset axis."""
    if dataset_params is None:
        return False
    return base_name in dataset_params


def expand_membership_from_derived(
    member_names: Collection[str],
    derived_params: list,
) -> set:
    """Expand a set of parameter names to include derived params whose
    dependencies overlap with the current members.

    A derived parameter inherits structural axes (component, dataset)
    from its dependencies — this mirrors the ``merge_layouts`` semantics
    used at model-execution time in ``ModelBuilder.build()``.  If *any*
    dep of a ``DerivedParam`` is already a member, the derived param is
    added to the set.

    Uses a fixed-point loop so that transitive dependencies are handled
    correctly (e.g. if ``q`` depends on ``p`` which depends on ``phi``,
    and ``phi`` is a member, both ``p`` and ``q`` will be added).

    Parameters
    ----------
    member_names : Collection[str]
        Initial set of parameter names that carry the axis in question
        (e.g. ``mixture_params`` or ``dataset_params``).
    derived_params : list of DerivedParam
        Derived parameter descriptors carrying ``name`` and ``deps``
        fields.  Only those two attributes are read.

    Returns
    -------
    set of str
        Expanded set including any derived param whose dependency chain
        reaches into *member_names*.
    """
    expanded = set(member_names)
    changed = True
    while changed:
        changed = False
        for d in derived_params:
            if d.name not in expanded and any(
                dep in expanded for dep in d.deps
            ):
                expanded.add(d.name)
                changed = True
    return expanded


def infer_layout(
    key: str,
    value: jnp.ndarray,
    *,
    n_genes: Optional[int] = None,
    n_cells: Optional[int] = None,
    n_components: Optional[int] = None,
    n_datasets: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    dataset_params: Optional[List[str]] = None,
    gene_axis_hint: Optional[int] = None,
    has_sample_dim: bool = False,
) -> AxisLayout:
    """Reconstruct an :class:`AxisLayout` from tensor shape and config metadata.

    This is the backward-compatibility path used when ``param_specs`` are
    not available (e.g. old pickles).  It consolidates the heuristics that
    were previously scattered across ``_infer_dataset_axis``,
    ``_infer_component_axis``, and ``broadcast_param_for_mixture`` into a
    single canonical location.

    Parameters
    ----------
    key : str
        Variational-parameter key (e.g. ``"r_loc"``, ``"p_capture_loc"``).
    value : jnp.ndarray
        The tensor whose layout we are inferring.
    n_genes : int or None, optional
        Number of genes in the model.  ``None`` skips gene-axis matching,
        which is useful when the caller only needs component/dataset axes (the gene axis is not matched by size in that
        case).
    n_cells : int, optional
        Number of cells (needed to identify cell-specific params).
    n_components : int, optional
        Number of mixture components.
    n_datasets : int, optional
        Number of datasets.
    mixture_params : list of str, optional
        Which parameter names are mixture-specific.  ``None`` means *all*.
    dataset_params : list of str, optional
        Which parameter names are dataset-specific.
    gene_axis_hint : int, optional
        Pre-computed gene axis index (e.g. from ``_gene_axis_by_key``).
    has_sample_dim : bool, default False
        Whether the tensor has a leading sample/draw axis.

    Returns
    -------
    AxisLayout
    """
    # Skip Flax nested-dict / flow keys -- treat as opaque
    if "$" in key:
        return AxisLayout(axes=(), has_sample_dim=has_sample_dim)

    base = _strip_param_key(key)
    shape = value.shape
    ndim = value.ndim

    # Account for the sample dimension when reading shape
    effective_shape = shape[1:] if has_sample_dim else shape
    effective_ndim = len(effective_shape)

    # --- Special-case: known component-only params (mixing_weights) ---
    if base in _KNOWN_COMPONENT_ONLY_PARAMS:
        if effective_ndim == 0:
            return AxisLayout(axes=(), has_sample_dim=has_sample_dim)
        axes: list[str] = []
        if n_components is not None and effective_shape[0] == n_components:
            axes.append(COMPONENTS)
        elif n_datasets is not None and effective_ndim >= 2:
            # (D, K) layout for per-dataset mixing weights
            if effective_shape[0] == n_datasets:
                axes.append(DATASETS)
            if n_components is not None and effective_shape[-1] == n_components:
                axes.append(COMPONENTS)
        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    # --- Special-case: known cell-specific params ---
    if base in _KNOWN_CELL_PARAMS:
        if (
            n_cells is not None
            and effective_ndim >= 1
            and effective_shape[0] == n_cells
        ):
            return AxisLayout(axes=(CELLS,), has_sample_dim=has_sample_dim)
        return AxisLayout(axes=(), has_sample_dim=has_sample_dim)

    # --- General parameter: build axes list from shape analysis ---
    axes = []

    # If gene_axis_hint is provided, use it directly to determine structure
    if gene_axis_hint is not None:
        effective_gene_axis = gene_axis_hint - (1 if has_sample_dim else 0)
        # Fill in preceding axes
        for i in range(effective_gene_axis):
            if (
                n_components is not None
                and effective_shape[i] == n_components
                and COMPONENTS not in axes
                and _is_mixture_param(base, mixture_params)
            ):
                axes.append(COMPONENTS)
            elif (
                n_datasets is not None
                and effective_shape[i] == n_datasets
                and DATASETS not in axes
                and _is_dataset_param(base, dataset_params)
            ):
                axes.append(DATASETS)
        axes.append(GENES)
        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    # No gene_axis_hint -- infer from shape and config metadata.
    # Walk effective_shape left-to-right assigning semantic names.
    remaining_dims = list(effective_shape)
    pos = 0

    # Component axis (outermost)
    if (
        n_components is not None
        and _is_mixture_param(base, mixture_params)
        and pos < len(remaining_dims)
        and remaining_dims[pos] == n_components
    ):
        axes.append(COMPONENTS)
        pos += 1

    # Dataset axis (between components and genes)
    if (
        n_datasets is not None
        and _is_dataset_param(base, dataset_params)
        and pos < len(remaining_dims)
        and remaining_dims[pos] == n_datasets
    ):
        axes.append(DATASETS)
        pos += 1

    # Gene axis (last meaningful axis)
    if pos < len(remaining_dims) and remaining_dims[pos] == n_genes:
        axes.append(GENES)
        pos += 1
    elif (
        pos < len(remaining_dims)
        and n_cells is not None
        and remaining_dims[pos] == n_cells
    ):
        axes.append(CELLS)
        pos += 1

    return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)


# =========================================================================
# Bulk layout builders
# =========================================================================


def build_param_layouts(
    param_specs: List["ParamSpec"],
    params: Dict[str, Any],
    has_sample_dim: bool = False,
) -> Dict[str, AxisLayout]:
    """Build ``{key: AxisLayout}`` from ``param_specs`` and parameter dict.

    Each key in *params* is matched to the most appropriate spec by name
    using ``_strip_param_key``.  Keys that don't match any spec are given
    an empty layout.

    Parameters
    ----------
    param_specs : list of ParamSpec
        Specifications from ``model_config.param_specs``.
    params : dict
        Parameter dictionary (``results.params`` or ``results.samples``).
    has_sample_dim : bool, default False
        Whether tensors carry a leading sample dimension.

    Returns
    -------
    dict of str to AxisLayout
    """
    if not param_specs:
        return {}

    # Build a name -> spec lookup (first spec wins for a given name).
    # Also register companion specs (auxiliary sample sites generated
    # by hierarchical priors like biology-informed capture).
    spec_by_name: Dict[str, "ParamSpec"] = {}
    for spec in param_specs:
        if spec.name not in spec_by_name:
            spec_by_name[spec.name] = spec
        for companion in getattr(spec, "companion_specs", []):
            if companion.name not in spec_by_name:
                spec_by_name[companion.name] = companion

    layouts: Dict[str, AxisLayout] = {}
    for key in params:
        if "$" in key:
            continue
        base = _strip_param_key(key)
        spec = spec_by_name.get(base)
        # Fallback: some spec names include a prefix (e.g.
        # ``log_r_dataset_loc``).  ``_strip_param_key`` aggressively
        # removes ``log_`` / ``logit_`` and may overshoot.  Re-try
        # with the prefix restored when the first lookup misses.
        if spec is None:
            for pfx in ("log_", "logit_"):
                spec = spec_by_name.get(pfx + base)
                if spec is not None:
                    break
        if spec is not None:
            layouts[key] = layout_from_param_spec(
                spec, has_sample_dim=has_sample_dim
            )
        else:
            layouts[key] = AxisLayout(axes=(), has_sample_dim=has_sample_dim)
    return layouts


# Posterior ``samples`` dicts often include derived keys (e.g. ``"p"``) that are
# not in ``param_specs``.  This builder uses ``layout_from_param_spec`` when a
# spec matches and :func:`infer_layout` otherwise, unlike
# :func:`build_param_layouts` which leaves unknown keys empty.
def build_sample_layouts(
    param_specs: List["ParamSpec"],
    samples: Dict[str, Any],
    *,
    n_genes: Optional[int] = None,
    n_cells: Optional[int] = None,
    n_components: Optional[int] = None,
    n_datasets: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    dataset_params: Optional[List[str]] = None,
    has_sample_dim: bool = False,
) -> Dict[str, AxisLayout]:
    """Build layouts for a samples dict, using specs where possible.

    Unlike :func:`build_param_layouts` which gives empty layouts for
    unrecognised keys, this function falls back to :func:`infer_layout`
    for keys that do not match any ``ParamSpec``.  This is necessary for
    posterior-sample dictionaries that contain derived quantities
    (e.g. ``"p"``, ``"gate"``) not described by ``param_specs``.

    Parameters
    ----------
    param_specs : list of ParamSpec
        Specifications from ``model_config.param_specs``.
    samples : dict
        Sample dictionary (e.g. ``results.posterior_samples``).
    n_genes : int or None, optional
        Number of genes (passed through to :func:`infer_layout` for keys
        without a matching spec).  ``None`` skips gene-axis matching.
    n_cells : int, optional
        Number of cells.
    n_components : int, optional
        Number of mixture components.
    n_datasets : int, optional
        Number of datasets.
    mixture_params : list of str, optional
        Which parameters are mixture-specific (``None`` = all).
    dataset_params : list of str, optional
        Which parameters are dataset-specific.
    has_sample_dim : bool, default False
        Whether tensors carry a leading sample dimension.

    Returns
    -------
    dict of str to AxisLayout
    """
    # Build name -> spec lookup, including companion specs.
    spec_by_name: Dict[str, "ParamSpec"] = {}
    for spec in param_specs or []:
        if spec.name not in spec_by_name:
            spec_by_name[spec.name] = spec
        for companion in getattr(spec, "companion_specs", []):
            if companion.name not in spec_by_name:
                spec_by_name[companion.name] = companion

    layouts: Dict[str, AxisLayout] = {}
    for key, value in samples.items():
        if "$" in key or not hasattr(value, "shape"):
            continue
        base = _strip_param_key(key)
        spec = spec_by_name.get(base)
        # Prefix-restoration fallback (same logic as build_param_layouts).
        if spec is None:
            for pfx in ("log_", "logit_"):
                spec = spec_by_name.get(pfx + base)
                if spec is not None:
                    break

        # Use the spec when the spec-based layout rank matches the
        # tensor's actual ndim.  After subsetting (e.g. get_component),
        # a tensor may have fewer dimensions than the spec declares
        # (is_mixture=True but component axis was indexed out).  In
        # that case, fall through to heuristic inference.
        if spec is not None:
            candidate = layout_from_param_spec(
                spec, has_sample_dim=has_sample_dim
            )
            if candidate.rank == value.ndim:
                layouts[key] = candidate
                continue

        # Derived / canonical keys not described by specs, or spec whose
        # rank doesn't match the tensor (post-subsetting) — use
        # shape-based heuristics so axes are still detected correctly.
        layouts[key] = infer_layout(
            key,
            value,
            n_genes=n_genes,
            n_cells=n_cells,
            n_components=n_components,
            n_datasets=n_datasets,
            mixture_params=mixture_params,
            dataset_params=dataset_params,
            has_sample_dim=has_sample_dim,
        )
    return layouts


# Thin adapter: callers that already computed ``AxisLayout`` objects should use
# this instead of re-deriving gene axes from ``ParamSpec`` lists alone.
def gene_axes_from_layouts(
    layouts: Dict[str, AxisLayout],
) -> Dict[str, int]:
    """Extract a ``{key: gene_axis}`` mapping from a layouts dict.

    Parameters
    ----------
    layouts : dict of str to AxisLayout
        Layouts produced by :func:`build_param_layouts`,
        :func:`build_sample_layouts`, or the ``layouts`` property on
        results objects.

    Returns
    -------
    dict of str to int
        Mapping from parameter key to its gene-axis index.  Keys without
        a gene axis are omitted.
    """
    return {
        key: layout.gene_axis
        for key, layout in layouts.items()
        if layout.gene_axis is not None
    }


def reconstruct_param_layouts(
    params: Dict[str, Any],
    *,
    n_genes: int,
    n_cells: Optional[int] = None,
    n_components: Optional[int] = None,
    n_datasets: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    dataset_params: Optional[List[str]] = None,
    gene_axis_by_key: Optional[Dict[str, int]] = None,
    has_sample_dim: bool = False,
) -> Dict[str, AxisLayout]:
    """Reconstruct layouts for every key in *params* without ``param_specs``.

    This is the top-level backward-compatibility entry point used by the
    ``layouts`` property on results objects when ``param_layouts`` is
    ``None`` (old pickles).

    Parameters
    ----------
    params : dict
        Parameter dictionary.
    n_genes : int
        Number of genes.
    n_cells : int, optional
        Number of cells.
    n_components : int, optional
        Number of mixture components.
    n_datasets : int, optional
        Number of datasets.
    mixture_params : list of str, optional
        Which parameters are mixture-specific (``None`` = all).
    dataset_params : list of str, optional
        Which parameters are dataset-specific.
    gene_axis_by_key : dict, optional
        Pre-computed gene-axis hints (SVI ``_gene_axis_by_key``).
    has_sample_dim : bool, default False
        Whether tensors carry a leading sample dimension.

    Returns
    -------
    dict of str to AxisLayout
    """
    # If a gene_axis_by_key mapping is provided, use it; otherwise, use an empty
    # dict. This mapping gives pre-computed gene-axis positions for specific
    # parameter keys, sometimes provided by SVI inference to hint what axis (if
    # any) is "genes".
    gax = gene_axis_by_key or {}

    # Initialize the layouts dictionary, which will map each param key to its
    # inferred AxisLayout.
    layouts: Dict[str, AxisLayout] = {}

    # Iterate over all parameter keys and values in the params dictionary.
    for key, value in params.items():
        # Skip keys from Flax nested dicts (they have "$" in the key), which are
        # not leaf parameter tensor entries.
        if "$" in key:
            continue
        # Skip parameter values that do not have a "shape" attribute (e.g., not
        # JAX arrays), as there is nothing to infer.
        if not hasattr(value, "shape"):
            continue
        # For each valid parameter tensor, infer its AxisLayout using the
        # available metadata (shape, key, config info, gene axis hints, etc).
        layouts[key] = infer_layout(
            key,
            value,
            n_genes=n_genes,
            n_cells=n_cells,
            n_components=n_components,
            n_datasets=n_datasets,
            mixture_params=mixture_params,
            dataset_params=dataset_params,
            gene_axis_hint=gax.get(key),
            has_sample_dim=has_sample_dim,
        )
    # Return the mapping from parameter keys to their inferred AxisLayouts.
    return layouts


# =========================================================================
# Broadcasting helper
# =========================================================================


def align_to_layout(
    tensor: jnp.ndarray,
    source: AxisLayout,
    target: AxisLayout,
) -> jnp.ndarray:
    """Insert singleton dims so *tensor* broadcasts to *target* layout.

    Parameters
    ----------
    tensor : jnp.ndarray
        Array with *source* layout.
    source : AxisLayout
        Layout describing *tensor*.
    target : AxisLayout
        Layout to broadcast towards.

    Returns
    -------
    jnp.ndarray
        Reshaped tensor (no data copy, just view with singletons).
    """
    idx = source.broadcast_to(target)
    return tensor[idx]


# =========================================================================
# Layout merging
# =========================================================================


def merge_layouts(*layouts: AxisLayout) -> AxisLayout:
    """Compute the union of several layouts in canonical axis order.

    Returns a new :class:`AxisLayout` whose ``axes`` contain every axis
    that appears in *any* of the input layouts, sorted according to
    ``_AXIS_ORDER`` (components > datasets > genes/cells).

    This is useful when computing a derived parameter whose layout is the
    superset of its inputs.  For example, multiplying ``p`` with layout
    ``("components",)`` by ``mu`` with layout ``("components", "genes")``
    produces a result whose layout is ``("components", "genes")``.

    Parameters
    ----------
    *layouts : AxisLayout
        One or more layouts to merge.

    Returns
    -------
    AxisLayout
        Layout containing all axes from all inputs.

    Examples
    --------
    >>> merge_layouts(
    ...     AxisLayout(("components",)),
    ...     AxisLayout(("components", "genes")),
    ... )
    AxisLayout(axes=('components', 'genes'))

    >>> merge_layouts(
    ...     AxisLayout(("genes",)),
    ...     AxisLayout(("components", "datasets", "genes")),
    ... )
    AxisLayout(axes=('components', 'datasets', 'genes'))
    """
    # Collect all unique axis names from every input layout.
    all_axes: set[str] = set()
    for layout in layouts:
        all_axes.update(layout.axes)

    # Sort the axes using the canonical ordering defined in _AXIS_ORDER.
    # Axes not in _AXIS_ORDER (unlikely, but defensive) sort last.
    sorted_axes = tuple(sorted(all_axes, key=lambda a: _AXIS_ORDER.get(a, 99)))
    return AxisLayout(axes=sorted_axes)


# =========================================================================
# Batch-aware broadcasting helper
# =========================================================================


def broadcast_param_to_layout(
    param: jnp.ndarray,
    param_layout: AxisLayout,
    target_layout: AxisLayout,
) -> jnp.ndarray:
    """Broadcast *param* so it is compatible with *target_layout*.

    Works like :func:`align_to_layout` but also handles a leading
    **batch / cell** dimension that is not described by either layout.
    During likelihood evaluation, tensors may acquire a leading cells
    axis after dataset indexing.  This helper detects that extra
    dimension automatically and preserves it while inserting singletons
    for the semantic axes that are missing in *param_layout*.

    Parameters
    ----------
    param : jnp.ndarray
        Array to broadcast.
    param_layout : AxisLayout
        Semantic layout of *param* (excluding the batch dimension).
    target_layout : AxisLayout
        Layout to broadcast towards (excluding the batch dimension).

    Returns
    -------
    jnp.ndarray
        *param* with singleton dimensions inserted so that it broadcasts
        correctly against a tensor with *target_layout*.

    Examples
    --------
    Non-batched: gene-specific ``p`` aligned to ``(K, G)``

    >>> p = jnp.ones(100)
    >>> broadcast_param_to_layout(
    ...     p,
    ...     AxisLayout(("genes",)),
    ...     AxisLayout(("components", "genes")),
    ... ).shape
    (1, 100)

    Batched: after dataset indexing, ``p`` is ``(batch, G)`` and ``r``
    is ``(batch, K, G)``.  The batch dimension is preserved.

    >>> p_batch = jnp.ones((32, 100))
    >>> broadcast_param_to_layout(
    ...     p_batch,
    ...     AxisLayout(("genes",)),
    ...     AxisLayout(("components", "genes")),
    ... ).shape
    (32, 1, 100)
    """
    # Check whether the tensor has more dimensions than the layout
    # describes.  The extra leading dimension(s) come from the cells
    # plate / dataset indexing and are not part of the semantic layout.
    has_batch = param.ndim > param_layout.rank

    if has_batch:
        # Temporarily mark both layouts as having a sample dim so that
        # broadcast_to prepends a slice(None) for the batch axis.
        src = param_layout.with_sample_dim()
        tgt = target_layout.with_sample_dim()
    else:
        src = param_layout
        tgt = target_layout

    return align_to_layout(param, src, tgt)


# =========================================================================
# Unified axis-membership derivation
# =========================================================================


def _derive_dataset_params_from_flags(model_config) -> List[str]:
    """Derive dataset-specific parameter names from hierarchical-prior flags.

    Inspects the four ``*_dataset_prior`` fields on *model_config* and
    maps each active prior to the canonical parameter name used by the
    current ``parameterization`` strategy.

    Parameters
    ----------
    model_config
        Model configuration object carrying hierarchical-prior flags
        and a ``parameterization`` string.

    Returns
    -------
    list of str
        Canonical names (may be empty if no dataset priors are active).
    """
    # Lazy import: HierarchicalPriorType lives in models.config, which
    # is higher in the dependency stack than core/.
    from ..models.config.enums import HierarchicalPriorType

    _NONE = HierarchicalPriorType.NONE

    def _is_active(field_name: str) -> bool:
        """Check whether a hierarchical-prior field is active.

        Treats both missing attributes and Python ``None`` as inactive
        (equivalent to ``HierarchicalPriorType.NONE``).
        """
        val = getattr(model_config, field_name, None)
        return val is not None and val != _NONE

    ds: List[str] = []
    param = getattr(model_config, "parameterization", "linked")

    if _is_active("expression_dataset_prior"):
        ds.append("r" if param in ("canonical", "standard") else "mu")
    if _is_active("prob_dataset_prior"):
        ds.append("phi" if param in ("mean_odds", "odds_ratio") else "p")
    if _is_active("zero_inflation_dataset_prior"):
        ds.append("gate")
    if _is_active("overdispersion_dataset_prior"):
        ds.append("bnb_concentration")

    # Biology-informed VCP capture: when data_driven is True and
    # n_datasets >= 2, the model creates a hierarchical mu_eta with
    # per-dataset values via _sample_hierarchical_mu_eta.  The (D,)-shaped
    # variational keys need a dataset axis so get_dataset() slices them.
    # This covers: mu_eta_raw (NCP noise), lambda_mu_eta (horseshoe
    # local scales), zeta_mu_eta / psi_mu_eta (NEG rates/variances).
    if (
        getattr(model_config, "uses_biology_informed_capture", False)
        and getattr(model_config, "n_datasets", None) is not None
        and getattr(model_config, "n_datasets", 0) >= 2
    ):
        _mu_eta_ds_names = [
            "mu_eta",
            "mu_eta_raw",
            "lambda_mu_eta",
            "zeta_mu_eta",
            "psi_mu_eta",
        ]
        for _name in _mu_eta_ds_names:
            if _name not in ds:
                ds.append(_name)

    return ds


def _scan_concat_dataset_keys(
    samples: Dict[str, Any],
    n_datasets: int,
    has_sample_dim: bool,
) -> List[str]:
    """Detect parameter keys whose leading axis matches *n_datasets*.

    When single-dataset results are concatenated via
    ``ScribeSVIResults.concat`` / ``ScribeMCMCResults.concat``, every
    non-cell parameter acquires a leading dataset dimension but no
    hierarchical-prior flags are set.  This scanner identifies those
    keys by shape inspection.

    Parameters
    ----------
    samples : dict
        Parameter or sample arrays.
    n_datasets : int
        Expected size of the dataset dimension.
    has_sample_dim : bool
        Whether tensors carry a leading posterior-draw axis before the
        dataset axis.

    Returns
    -------
    list of str
        Keys whose axis at *offset* equals *n_datasets*.
    """
    _offset = 1 if has_sample_dim else 0
    ds: List[str] = []
    for key, arr in samples.items():
        if not hasattr(arr, "shape"):
            continue
        # Strip variational suffixes so cell-param detection uses the
        # base name (e.g. "p_capture_loc" -> "p_capture").
        base = key.split("_loc")[0].split("_scale")[0]
        if base in _KNOWN_CELL_PARAMS:
            continue
        if arr.ndim > _offset and arr.shape[_offset] == n_datasets:
            ds.append(key)
    return ds


def derive_axis_membership(
    model_config,
    *,
    samples: Optional[Dict[str, Any]] = None,
    has_sample_dim: bool = False,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Derive ``mixture_params`` and ``dataset_params`` from *model_config*.

    Centralises the logic for figuring out which parameter names carry a
    component axis and which carry a dataset axis.  Multiple information
    sources are checked in priority order so that callers never need to
    re-implement the derivation cascade.

    **Priority for ``mixture_params``**:

    1. ``model_config.mixture_params`` (explicit user setting).
    2. ``ParamSpec.is_mixture`` flags from ``model_config.param_specs``.
    3. ``None`` — by convention this means *all non-cell-specific
       parameters* are mixture-specific (the scribe default).

    **Priority for ``dataset_params``**:

    1. ``model_config.dataset_params`` (explicit user setting).
    2. ``ParamSpec.is_dataset`` flags from ``model_config.param_specs``.
    3. ``HierarchicalPriorType`` flags on *model_config*
       (``expression_dataset_prior``, ``prob_dataset_prior``, etc.).
    4. Concatenated multi-dataset shape scan: when ``n_datasets >= 2``
       and no priors are active, any non-cell key whose leading
       dimension equals ``n_datasets`` is treated as dataset-specific.

    After resolution, both lists are expanded through the
    parameterization's ``DerivedParam`` dependency graph so that
    derived canonical keys (e.g. ``"r"``, ``"p"``) inherit axis
    membership from their sources (e.g. ``"mu"``, ``"phi"``).

    Parameters
    ----------
    model_config
        Model configuration object.  Must expose at least
        ``parameterization`` (str).  Optional fields consulted:
        ``mixture_params``, ``dataset_params``, ``param_specs``,
        ``n_datasets``, ``expression_dataset_prior``,
        ``prob_dataset_prior``, ``zero_inflation_dataset_prior``,
        ``overdispersion_dataset_prior``.
    samples : dict, optional
        Parameter / sample arrays.  Only needed for the concatenated
        multi-dataset shape scan (priority 4 for ``dataset_params``).
        Can be ``None`` when the caller knows no concat edge case
        applies (e.g. the ``layouts`` backward-compat property).
    has_sample_dim : bool, default False
        Whether tensors in *samples* carry a leading posterior-draw
        axis.  Shifts the dataset-axis scan by one position.

    Returns
    -------
    tuple of (list[str] | None, list[str] | None)
        ``(mixture_params, dataset_params)``.  Either element may be
        ``None`` when no specific list could be derived (which means
        "all non-cell params" for mixture, or "no dataset axis" for
        dataset).

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> cfg = SimpleNamespace(
    ...     parameterization="linked",
    ...     mixture_params=["p", "mu"],
    ...     dataset_params=None,
    ...     param_specs=[],
    ...     n_datasets=None,
    ... )
    >>> mp, dp = derive_axis_membership(cfg)
    >>> "r" in mp  # derived from "mu" dep in linked/mean_prob
    True
    >>> dp is None
    True
    """
    specs = getattr(model_config, "param_specs", None) or []

    # ---- mixture_params ----
    mp: Optional[List[str]] = getattr(model_config, "mixture_params", None)
    if mp is None and specs:
        inferred = [s.name for s in specs if getattr(s, "is_mixture", False)]
        mp = inferred if inferred else None

    # ---- dataset_params ----
    dp: Optional[List[str]] = getattr(model_config, "dataset_params", None)

    if dp is None and specs:
        # Priority 2: ParamSpec.is_dataset flags
        inferred = [s.name for s in specs if getattr(s, "is_dataset", False)]
        dp = inferred if inferred else None

    if dp is None:
        # Priority 3: HierarchicalPriorType flags
        ds_from_flags = _derive_dataset_params_from_flags(model_config)
        dp = ds_from_flags if ds_from_flags else None

    if (
        dp is None
        and samples is not None
        and getattr(model_config, "n_datasets", None) is not None
        and getattr(model_config, "n_datasets", None) >= 2
    ):
        # Priority 4: concat multi-dataset shape scan
        ds_from_scan = _scan_concat_dataset_keys(
            samples,
            int(getattr(model_config, "n_datasets")),
            has_sample_dim,
        )
        dp = ds_from_scan if ds_from_scan else None

    # ---- Expand both lists through the DerivedParam graph ----
    # Derived canonical keys (r, p) inherit axis membership from their
    # source params (mu, phi) so that infer_layout assigns the correct
    # component / dataset axes to them.
    if mp is not None or dp is not None:
        try:
            from ..models.parameterizations import PARAMETERIZATIONS
        except ImportError:
            # Defensive: if the import fails (e.g. in a minimal test
            # environment), skip expansion rather than crashing.
            pass
        else:
            _param = getattr(model_config, "parameterization", "linked")
            _strategy = PARAMETERIZATIONS.get(_param)
            _derived = (
                _strategy.build_derived_params()
                if _strategy is not None
                else []
            )
            if _derived:
                if mp is not None:
                    mp = sorted(expand_membership_from_derived(mp, _derived))
                if dp is not None:
                    dp = sorted(expand_membership_from_derived(dp, _derived))

    return mp, dp

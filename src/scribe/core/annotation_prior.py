"""
Annotation prior utilities for mixture models.

This module provides functions for building cell-specific prior logits from
categorical annotations in ``adata.obs``. The logits are used to nudge the
global mixture weights toward annotated component assignments on a per-cell
basis, implementing a Bayesian soft-label mechanism.

Mathematical Basis
------------------
In a standard mixture model every cell shares a global mixing weight vector:

π ~ Dirichlet(α),  xᵢ ~ MixtureSameFamily(Cat(π), Fₖ)

With annotation priors the mixing is made cell-specific:

πᵢ = softmax(log π + κ · 𝟙[annotationᵢ])

where κ ("confidence") controls how strongly the annotation influences the
prior. κ = 0 recovers the standard model and κ → ∞ hard-assigns the cell.

Functions
---------
build_annotation_prior_logits
    Build a ``(n_cells, n_components)`` logit-offset matrix from annotations.
validate_annotation_prior_logits
    Validate shape and finiteness of a logit-offset matrix.
build_component_mapping
    Identify shared vs dataset-specific components across multiple datasets.

Classes
-------
ComponentMapping
    Dataclass recording which components are shared across datasets.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)

#: Separator used to join values from multiple ``.obs`` columns into a
#: single composite label (e.g. ``"T_cell__ctrl"``).
COMPOSITE_LABEL_SEP = "__"


# ==============================================================================
# Internal helpers
# ==============================================================================


def _resolve_composite_annotations(
    adata: "AnnData",
    obs_keys: List[str],
) -> pd.Series:
    """
    Build a composite annotation Series from multiple ``.obs`` columns.

    For each cell the values of the requested columns are joined with
    :data:`COMPOSITE_LABEL_SEP`.  If **any** column has a missing value
    for a given cell, the composite label for that cell is ``NaN``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_keys : list of str
        Column names in ``adata.obs`` to combine.

    Returns
    -------
    pd.Series
        A Series of length ``n_cells`` with composite string labels or
        ``NaN`` for unlabeled cells.

    Raises
    ------
    ValueError
        If any of the requested columns is missing from ``adata.obs``.
    """
    for key in obs_keys:
        if key not in adata.obs.columns:
            raise ValueError(
                f"Annotation key '{key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # Convert each column to object dtype so pd.isna works uniformly
    cols = []
    for key in obs_keys:
        col = adata.obs[key]
        if hasattr(col, "cat"):
            col = col.astype(object)
        cols.append(col)

    # A cell is labeled only if *all* columns are non-null
    all_labeled = pd.Series(True, index=adata.obs.index)
    for col in cols:
        all_labeled = all_labeled & ~pd.isna(col)

    # Build composite labels
    composite = pd.Series(np.nan, index=adata.obs.index, dtype=object)
    labeled_idx = all_labeled[all_labeled].index
    if len(labeled_idx) > 0:
        parts = [col.loc[labeled_idx].astype(str) for col in cols]
        composite.loc[labeled_idx] = parts[0]
        for part in parts[1:]:
            composite.loc[labeled_idx] = (
                composite.loc[labeled_idx] + COMPOSITE_LABEL_SEP + part
            )

    return composite


# ==============================================================================
# Public API
# ==============================================================================


def build_annotation_prior_logits(
    adata: "AnnData",
    obs_key: Union[str, List[str]],
    n_components: int,
    confidence: float = 3.0,
    component_order: Optional[List[str]] = None,
    min_cells: int = 0,
) -> Tuple[jnp.ndarray, Dict[str, int]]:
    """
    Build cell-specific prior logits from categorical annotation columns.

    Reads one or more columns from ``adata.obs``, maps each unique label
    (or composite label) to a mixture component index, and returns a
    ``(n_cells, n_components)`` matrix of additive logit offsets.  Cells
    whose annotation is missing or ``NaN`` (in any of the requested
    columns) receive an all-zero row (agnostic prior, equivalent to the
    global mixing weights).

    When ``obs_key`` is a **list** of column names, a composite label is
    formed for each cell by joining the per-column values with ``"__"``
    (double underscore).  For example, if a cell has
    ``cell_type="T_cell"`` and ``treatment="ctrl"``, the composite label
    is ``"T_cell__ctrl"``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_key : str or list of str
        Column name(s) in ``adata.obs``.  If a single string, that column
        is used directly.  If a list of strings, composite labels are
        formed from the Cartesian product of unique values across the
        specified columns.
    n_components : int
        Number of mixture components *K*.  The number of unique non-missing
        (composite) labels must be ``<= n_components``.
    confidence : float, optional
        Scaling factor :math:`\\kappa` applied to the one-hot encoding.

        * ``confidence = 0`` — annotations are ignored (all-zero logits).
        * ``confidence = 3`` (default) — annotated component gets
          :math:`\\exp(3) \\approx 20\\times` boost in prior weight.
        * ``confidence → ∞`` — hard assignment.
    component_order : list of str, optional
        Explicit mapping from label strings to component indices.  The
        *i*-th element of the list is assigned to component *i*.  When
        using multiple ``obs_key`` columns, these should be composite
        labels using ``"__"`` as separator (e.g.
        ``["T_cell__ctrl", "T_cell__stim", ...]``).  If ``None``, unique
        labels are sorted alphabetically.
    min_cells : int, optional
        Minimum number of cells required for an annotation label to be
        considered.  Labels with fewer than ``min_cells`` cells are
        treated as unlabeled (their logit rows are set to zero, i.e. no
        bias toward any component).  Those labels are excluded from
        ``label_map`` and do not occupy a component index.  A warning is
        logged listing the filtered-out labels and their cell counts.
        When ``component_order`` is provided, any remaining labels not
        covered by that order are also treated as unlabeled when
        ``min_cells > 0`` (instead of raising), which supports workflows
        where component orders are built from per-dataset label survival.
        Default is ``0`` (no filtering).

    Returns
    -------
    prior_logits : jnp.ndarray, shape ``(n_cells, n_components)``
        Additive logit offsets.  Zero rows for unlabeled cells.
    label_map : dict of str -> int
        Mapping from (composite) annotation label to component index.

    Raises
    ------
    ValueError
        If any ``obs_key`` is not in ``adata.obs``, if ``confidence < 0``,
        or if the number of unique labels exceeds ``n_components``.

    Examples
    --------
    Single-column annotation:

    >>> import anndata, pandas as pd, numpy as np
    >>> adata = anndata.AnnData(
    ...     X=np.random.poisson(5, (6, 3)),
    ...     obs=pd.DataFrame({"cell_type": ["A", "A", "B", np.nan, "B", "A"]}),
    ... )
    >>> logits, lmap = build_annotation_prior_logits(
    ...     adata, "cell_type", n_components=3, confidence=3.0
    ... )
    >>> logits.shape
    (6, 3)
    >>> lmap
    {'A': 0, 'B': 1}

    Multi-column annotation (composite labels):

    >>> adata = anndata.AnnData(
    ...     X=np.random.poisson(5, (4, 3)),
    ...     obs=pd.DataFrame({
    ...         "cell_type": ["T", "T", "B", "B"],
    ...         "treatment": ["ctrl", "stim", "ctrl", "stim"],
    ...     }),
    ... )
    >>> logits, lmap = build_annotation_prior_logits(
    ...     adata, ["cell_type", "treatment"], n_components=4, confidence=3.0
    ... )
    >>> sorted(lmap.keys())
    ['B__ctrl', 'B__stim', 'T__ctrl', 'T__stim']
    """
    # ------------------------------------------------------------------
    # Normalise obs_key to a list
    # ------------------------------------------------------------------
    if isinstance(obs_key, str):
        obs_keys = [obs_key]
    else:
        obs_keys = list(obs_key)

    if len(obs_keys) == 0:
        raise ValueError("obs_key must be a non-empty string or list of strings")

    # ------------------------------------------------------------------
    # Confidence validation
    # ------------------------------------------------------------------
    if confidence < 0:
        raise ValueError(f"confidence must be >= 0, got {confidence}")

    # ------------------------------------------------------------------
    # Build the (possibly composite) annotation Series
    # ------------------------------------------------------------------
    if len(obs_keys) == 1:
        # Single column: fast path — no composite construction needed
        key = obs_keys[0]
        if key not in adata.obs.columns:
            raise ValueError(
                f"Annotation key '{key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        annotations = adata.obs[key]
        if hasattr(annotations, "cat"):
            annotations = annotations.astype(object)
    else:
        # Multiple columns: build composite labels
        annotations = _resolve_composite_annotations(adata, obs_keys)

    # ------------------------------------------------------------------
    # Identify labeled / unlabeled cells
    # ------------------------------------------------------------------
    is_labeled = ~pd.isna(annotations)
    labels = annotations[is_labeled]

    # ------------------------------------------------------------------
    # Filter out rare labels (fewer than min_cells)
    # ------------------------------------------------------------------
    if min_cells > 0:
        label_counts = labels.value_counts()
        rare_labels = set(label_counts[label_counts < min_cells].index)
        if rare_labels:
            logger.warning(
                "Annotation labels with fewer than %d cells will be "
                "treated as unlabeled (zero bias): %s",
                min_cells,
                {
                    str(l): int(label_counts[l])
                    for l in sorted(rare_labels, key=str)
                },
            )
            is_labeled = is_labeled & ~annotations.isin(rare_labels)
            labels = annotations[is_labeled]

    # ------------------------------------------------------------------
    # Build label -> component index mapping
    # ------------------------------------------------------------------
    if component_order is not None:
        label_map: Dict[str, int] = {
            label: idx for idx, label in enumerate(component_order)
        }
        # Check that all observed labels are covered
        unique_labels = set(str(l) for l in labels.unique())
        missing = unique_labels - set(label_map.keys())
        if missing:
            # When min_cells filtering is active, allow uncovered labels to be
            # treated as unlabeled so per-dataset component_order generation
            # can intentionally drop labels that do not survive in any dataset.
            if min_cells > 0:
                logger.warning(
                    "Annotation labels not covered by component_order after "
                    "min_cells filtering will be treated as unlabeled (zero "
                    "bias): %s",
                    sorted(missing),
                )
                missing_mask = annotations.apply(
                    lambda value: (
                        (not pd.isna(value)) and (str(value) in missing)
                    )
                )
                is_labeled = is_labeled & ~missing_mask
                labels = annotations[is_labeled]
            else:
                obs_key_display = (
                    obs_keys[0] if len(obs_keys) == 1 else str(obs_keys)
                )
                raise ValueError(
                    f"The following annotation labels are not in "
                    f"component_order: {missing}. component_order must "
                    f"cover all labels present in "
                    f"adata.obs[{obs_key_display!r}]."
                )
    else:
        unique_labels_sorted = sorted(str(l) for l in labels.unique())
        label_map = {
            label: idx for idx, label in enumerate(unique_labels_sorted)
        }

    n_unique = len(label_map)
    if n_unique > n_components:
        raise ValueError(
            f"Number of unique annotation labels ({n_unique}) exceeds "
            f"n_components ({n_components}). Either increase n_components "
            f"or provide a component_order that maps labels to fewer "
            f"components."
        )

    # ------------------------------------------------------------------
    # Build the logit matrix
    # ------------------------------------------------------------------
    n_cells = adata.n_obs
    prior_logits = np.zeros((n_cells, n_components), dtype=np.float32)

    is_labeled_np = is_labeled.to_numpy()
    for cell_idx in range(n_cells):
        if is_labeled_np[cell_idx]:
            label = str(annotations.iloc[cell_idx])
            component_idx = label_map[label]
            prior_logits[cell_idx, component_idx] = confidence

    return jnp.array(prior_logits), label_map


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------


def validate_annotation_prior_logits(
    logits: jnp.ndarray,
    n_cells: int,
    n_components: int,
) -> None:
    """
    Validate shape and finiteness of an annotation prior logit matrix.

    Parameters
    ----------
    logits : jnp.ndarray
        Logit-offset matrix to validate.
    n_cells : int
        Expected number of cells (first dimension).
    n_components : int
        Expected number of mixture components (second dimension).

    Raises
    ------
    ValueError
        If the shape does not match ``(n_cells, n_components)`` or if the
        matrix contains non-finite values.
    """
    if logits.shape != (n_cells, n_components):
        raise ValueError(
            f"annotation_prior_logits has shape {logits.shape}, "
            f"expected ({n_cells}, {n_components})"
        )
    if not jnp.all(jnp.isfinite(logits)):
        raise ValueError(
            "annotation_prior_logits contains non-finite values (NaN or Inf)"
        )


# ==============================================================================
# Component mapping for multi-dataset mixtures
# ==============================================================================


@dataclass
class ComponentMapping:
    """Records which mixture components are shared across datasets.

    When fitting a mixture model jointly across multiple datasets, some
    components (cell types) may be present in all datasets while others
    are exclusive to a subset.  This dataclass stores the result of
    comparing annotation labels across datasets so that downstream code
    can apply per-component dataset-level hierarchy only to shared
    components.

    Attributes
    ----------
    component_order : list of str
        Ordered list of all labels (union across datasets).  The *i*-th
        element maps to component index *i*.
    label_map : dict of str to int
        Mapping from annotation label to component index.
    per_dataset_labels : dict of str to set of str
        For each dataset name, the set of annotation labels present
        (after ``min_cells`` filtering).
    shared_indices : tuple of int
        Component indices that appear in 2 or more datasets.
    exclusive_indices : tuple of int
        Component indices that appear in exactly 1 dataset.
    shared_mask : tuple of bool
        Boolean mask of length ``n_components``; ``True`` for shared
        components.

    Notes
    -----
    **Future (Approach B):** Non-shared components currently still get a
    ``(D, G)`` tensor with clamped scale.  A future refactoring could give
    them shape ``(G,)`` directly, eliminating wasted parameters at the cost
    of mixed-shape tensors in the likelihood.  See the plan for details.
    """

    component_order: List[str]
    label_map: Dict[str, int]
    per_dataset_labels: Dict[str, Set[str]]
    shared_indices: Tuple[int, ...]
    exclusive_indices: Tuple[int, ...]
    shared_mask: Tuple[bool, ...]

    @property
    def n_components(self) -> int:
        """Total number of components (union of all labels)."""
        return len(self.component_order)

    @property
    def n_shared(self) -> int:
        """Number of components shared across 2+ datasets."""
        return len(self.shared_indices)

    @property
    def n_datasets(self) -> int:
        """Number of datasets."""
        return len(self.per_dataset_labels)


def build_component_mapping(
    adata: "AnnData",
    annotation_key: Union[str, List[str]],
    dataset_key: str,
    min_cells: int = 0,
    shared_components: Optional[List[str]] = None,
) -> ComponentMapping:
    """Identify shared vs dataset-specific mixture components.

    Groups cells by ``dataset_key``, collects unique annotation labels
    per dataset (applying ``min_cells`` filtering within each dataset),
    and builds a :class:`ComponentMapping` recording which labels appear
    in multiple datasets ("shared") and which are exclusive to one.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing all datasets concatenated.
    annotation_key : str or list of str
        Column(s) in ``adata.obs`` used for annotation labels.  When a
        list, composite labels are formed with ``"__"`` separator (same
        as :func:`build_annotation_prior_logits`).
    dataset_key : str
        Column in ``adata.obs`` identifying datasets.
    min_cells : int, optional
        Minimum number of cells for a label to be retained *within each
        dataset*.  Labels below the threshold in a given dataset are
        excluded from that dataset's label set but may survive in other
        datasets.  Default ``0``.
    shared_components : list of str, optional
        Manual override for which labels are considered shared.  When
        provided, only these labels are marked as shared; all others are
        treated as dataset-specific regardless of how many datasets they
        appear in.

    Returns
    -------
    ComponentMapping
        The mapping between component indices and datasets.

    Raises
    ------
    ValueError
        If ``dataset_key`` or ``annotation_key`` is not in ``adata.obs``,
        or if ``shared_components`` contains labels not in the union.
    """
    # Normalise annotation_key to a list
    if isinstance(annotation_key, str):
        obs_keys = [annotation_key]
    else:
        obs_keys = list(annotation_key)

    # Validate columns exist
    for key in obs_keys:
        if key not in adata.obs.columns:
            raise ValueError(
                f"Annotation key '{key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
    if dataset_key not in adata.obs.columns:
        raise ValueError(
            f"dataset_key '{dataset_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Build annotation series (single or composite)
    if len(obs_keys) == 1:
        annotations = adata.obs[obs_keys[0]]
        if hasattr(annotations, "cat"):
            annotations = annotations.astype(object)
    else:
        annotations = _resolve_composite_annotations(adata, obs_keys)

    ds_col = adata.obs[dataset_key]
    datasets = sorted(str(d) for d in ds_col.unique())

    # Collect labels per dataset with min_cells filtering
    per_dataset_labels: Dict[str, Set[str]] = {}
    for ds_name in datasets:
        ds_mask = ds_col.astype(str) == ds_name
        ds_annotations = annotations[ds_mask]
        ds_labeled = ds_annotations[~pd.isna(ds_annotations)]

        if min_cells > 0 and len(ds_labeled) > 0:
            counts = ds_labeled.value_counts()
            surviving = set(counts[counts >= min_cells].index)
        elif len(ds_labeled) > 0:
            surviving = set(ds_labeled.unique())
        else:
            surviving = set()

        per_dataset_labels[ds_name] = {str(l) for l in surviving}

    # Build global union and sort alphabetically for deterministic ordering
    all_labels = set()
    for ds_labels in per_dataset_labels.values():
        all_labels |= ds_labels
    component_order = sorted(all_labels)
    label_map = {label: idx for idx, label in enumerate(component_order)}

    # Determine shared vs exclusive indices
    if shared_components is not None:
        # Manual override: validate all specified labels exist in the union
        unknown = set(shared_components) - all_labels
        if unknown:
            raise ValueError(
                f"shared_components contains labels not found in any "
                f"dataset: {unknown}. Available labels: {sorted(all_labels)}"
            )
        shared_label_set = set(shared_components)
    else:
        # Automatic: labels in 2+ datasets are shared
        shared_label_set = set()
        for label in all_labels:
            n_present = sum(
                label in ds_labels
                for ds_labels in per_dataset_labels.values()
            )
            if n_present >= 2:
                shared_label_set.add(label)

    shared_indices = tuple(
        label_map[l] for l in component_order if l in shared_label_set
    )
    exclusive_indices = tuple(
        label_map[l] for l in component_order if l not in shared_label_set
    )
    shared_mask = tuple(l in shared_label_set for l in component_order)

    logger.info(
        "Component mapping: %d total, %d shared, %d exclusive. "
        "Shared: %s",
        len(component_order),
        len(shared_indices),
        len(exclusive_indices),
        [component_order[i] for i in shared_indices],
    )

    return ComponentMapping(
        component_order=component_order,
        label_map=label_map,
        per_dataset_labels=per_dataset_labels,
        shared_indices=shared_indices,
        exclusive_indices=exclusive_indices,
        shared_mask=shared_mask,
    )

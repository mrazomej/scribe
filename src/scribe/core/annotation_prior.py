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
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

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
    # Build label -> component index mapping
    # ------------------------------------------------------------------
    if component_order is not None:
        label_map: Dict[str, int] = {
            label: idx for idx, label in enumerate(component_order)
        }
        # Check that all observed labels are covered
        unique_labels = set(labels.unique())
        missing = unique_labels - set(label_map.keys())
        if missing:
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

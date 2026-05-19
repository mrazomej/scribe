"""
Internal helper functions shared across :mod:`scribe.api` stages.

These are pure utility functions with no side-effects on the inference
pipeline state.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from anndata import AnnData

_log = logging.getLogger(__name__)


def _count_unique_labels(
    adata: "AnnData",
    annotation_key: Union[str, List[str]],
    min_cells: int = 0,
) -> int:
    """
    Count the number of unique non-null annotation labels.

    When ``annotation_key`` is a list of column names, composite labels
    are formed (identical to the logic in
    :func:`build_annotation_prior_logits`) and the unique composites are
    counted.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    annotation_key : str or list of str
        Column name(s) in ``adata.obs``.
    min_cells : int, optional
        Minimum number of cells for a label to be counted.  Labels with
        fewer than ``min_cells`` cells are excluded from the count.
        Default is ``0`` (no filtering).

    Returns
    -------
    int
        Number of unique non-null labels (or composite labels) that meet
        the ``min_cells`` threshold.
    """
    import pandas as pd

    if isinstance(annotation_key, str):
        obs_keys = [annotation_key]
    else:
        obs_keys = list(annotation_key)

    if len(obs_keys) == 1:
        col = adata.obs[obs_keys[0]]
        if hasattr(col, "cat"):
            col = col.astype(object)
        non_null = col.dropna()
    else:
        from ..core.annotation_prior import _resolve_composite_annotations

        composite = _resolve_composite_annotations(adata, obs_keys)
        non_null = composite.dropna()

    if min_cells > 0:
        counts = non_null.value_counts()
        return int(len(counts[counts >= min_cells]))
    return int(len(non_null.unique()))


def _coerce_batch_size_for_dataset(
    batch_size: Optional[int], n_cells: int
) -> Optional[int]:
    """
    Normalize mini-batch size against dataset size.

    This helper preserves user-specified mini-batch sizes when they are valid
    and automatically switches to full-batch mode when the requested
    ``batch_size`` exceeds the number of cells in the current dataset.
    Full-batch mode is represented by ``None`` in SCRIBE's SVI/VAE configs.

    Parameters
    ----------
    batch_size : int or None
        Requested mini-batch size. ``None`` means full-batch mode.
    n_cells : int
        Number of cells available for inference after data processing.

    Returns
    -------
    int or None
        Effective batch size. Returns:

        - ``None`` when ``batch_size`` is ``None``.
        - The original ``batch_size`` when it is less than or equal to
          ``n_cells``.
        - ``None`` when ``batch_size`` is greater than ``n_cells``.

    Warns
    -----
    Emits an INFO log when ``batch_size`` exceeds ``n_cells`` and is coerced to
    ``None``.
    """
    if batch_size is None:
        return None
    if batch_size > n_cells:
        _log.info(
            f"batch_size={batch_size} exceeds n_cells={n_cells}; "
            "using full-batch mode (batch_size=None)."
        )
        return None
    return batch_size


def _normalize_prior_type_name(prior: Any) -> str:
    """Normalize a hierarchical prior selector to a lowercase string value.

    Parameters
    ----------
    prior : Any
        Prior selector coming from ``fit`` kwargs. This may be either a
        ``HierarchicalPriorType`` enum instance or a plain string.

    Returns
    -------
    str
        Lowercase prior name (e.g. ``"none"``, ``"gaussian"``).
    """
    if isinstance(prior, str):
        return prior.lower()
    return str(getattr(prior, "value", prior)).lower()

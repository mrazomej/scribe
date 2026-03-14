"""Label-based component matching utilities for multi-dataset DE.

When mixture models are fit with ``annotation_key``, each result object
stores a ``_label_map`` that maps annotation labels (e.g. cell type
names) to integer component indices.  These utilities enable automatic
component alignment across independently-fit or jointly-fit results.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    pass


def match_components_by_label(
    results_A,
    results_B,
    label: str,
) -> Tuple[int, int]:
    """Look up matching component indices for a label in two results.

    Both result objects must have a ``_label_map`` attribute (populated
    when ``annotation_key`` is provided to ``fit()``).

    Parameters
    ----------
    results_A : ScribeSVIResults
        First results object (e.g. condition A).
    results_B : ScribeSVIResults
        Second results object (e.g. condition B).
    label : str
        Annotation label to look up (e.g. ``"Fibroblast"``).

    Returns
    -------
    idx_A : int
        Component index in ``results_A`` for the given label.
    idx_B : int
        Component index in ``results_B`` for the given label.

    Raises
    ------
    ValueError
        If either result lacks a ``_label_map`` or the label is not
        found in one of them.

    Examples
    --------
    >>> idx_A, idx_B = match_components_by_label(
    ...     results_bleo, results_ctrl, "Fibroblast"
    ... )
    >>> de = compare(
    ...     results_bleo, results_ctrl,
    ...     method="empirical",
    ...     component_A=idx_A, component_B=idx_B,
    ... )
    """
    map_A = getattr(results_A, "_label_map", None)
    map_B = getattr(results_B, "_label_map", None)

    if map_A is None:
        raise ValueError(
            "results_A does not have a _label_map. "
            "Was annotation_key provided when calling fit()?"
        )
    if map_B is None:
        raise ValueError(
            "results_B does not have a _label_map. "
            "Was annotation_key provided when calling fit()?"
        )

    if label not in map_A:
        raise ValueError(
            f"Label {label!r} not found in results_A._label_map. "
            f"Available: {sorted(map_A.keys())}"
        )
    if label not in map_B:
        raise ValueError(
            f"Label {label!r} not found in results_B._label_map. "
            f"Available: {sorted(map_B.keys())}"
        )

    return map_A[label], map_B[label]


def get_shared_labels(
    results_A,
    results_B,
) -> List[str]:
    """Return labels present in both results' label maps.

    Parameters
    ----------
    results_A : ScribeSVIResults
        First results object.
    results_B : ScribeSVIResults
        Second results object.

    Returns
    -------
    list of str
        Sorted list of labels present in both label maps.

    Raises
    ------
    ValueError
        If either result lacks a ``_label_map``.
    """
    map_A = getattr(results_A, "_label_map", None)
    map_B = getattr(results_B, "_label_map", None)

    if map_A is None:
        raise ValueError(
            "results_A does not have a _label_map. "
            "Was annotation_key provided when calling fit()?"
        )
    if map_B is None:
        raise ValueError(
            "results_B does not have a _label_map. "
            "Was annotation_key provided when calling fit()?"
        )

    return sorted(set(map_A.keys()) & set(map_B.keys()))

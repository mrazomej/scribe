"""Utilities for exporting posterior samples as feature matrices.

This module centralizes conversion from dictionary-structured posterior
samples (``{param_name: tensor}``) to a flat matrix suitable for
downstream analysis and visualization.  The conversion is intentionally
plotting-library agnostic and relies on semantic axis layouts so tensor
dimensions are interpreted consistently across SVI and MCMC results.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .axis_layout import AxisLayout, derive_axis_membership, infer_layout
from ..models.config.parameter_mapping import DESCRIPTIVE_NAMES


def _to_name_set(values: Optional[Iterable[str]]) -> Optional[set[str]]:
    """Normalize an optional iterable of names to a set.

    Parameters
    ----------
    values : iterable of str or None
        Names supplied by the caller.

    Returns
    -------
    set of str or None
        ``None`` if *values* is ``None``; otherwise a set of names.
    """
    if values is None:
        return None
    return {str(v) for v in values}


def _derived_parameter_keys(
    model_config: Any,
    descriptive_names: bool,
) -> set[str]:
    """Infer likely deterministic/derived parameter keys for exclusion.

    The intent is to provide a conservative default for analysis-focused
    exports.  We only exclude well-known derived parameters for each
    parameterization so we do not accidentally hide sampled latent
    variables.

    Parameters
    ----------
    model_config : Any
        Model configuration object carrying at least
        ``parameterization`` and ``uses_variable_capture`` fields.
    descriptive_names : bool
        Whether output keys are descriptive aliases instead of internal
        names.

    Returns
    -------
    set of str
        Parameter keys that are typically deterministic in the current
        parameterization.
    """
    parameterization = getattr(
        getattr(model_config, "parameterization", None),
        "value",
        getattr(model_config, "parameterization", None),
    )
    parameterization = str(parameterization)

    # Base deterministic keys by parameterization family.
    derived = set()
    if parameterization in {"standard", "canonical"}:
        derived.update({"mu"})
    elif parameterization in {"linked", "mean_prob"}:
        derived.update({"r"})
    elif parameterization in {"odds_ratio", "mean_odds"}:
        derived.update({"p", "r"})

    # Variable-capture odds parameterizations usually derive p_capture from
    # phi_capture through a deterministic transform.
    if (
        getattr(model_config, "uses_variable_capture", False)
        and parameterization in {"odds_ratio", "mean_odds"}
    ):
        derived.add("p_capture")

    if not descriptive_names:
        return derived

    # Mirror deterministic keys when descriptive aliases are requested.
    mirrored = {
        DESCRIPTIVE_NAMES.get(name, name)
        for name in derived
    }
    return mirrored


def _selector_from_coords(
    *,
    axis_name: str,
    axis_size: int,
    axis_selector: Any,
    var_index: Optional[Sequence[Any]],
) -> np.ndarray:
    """Normalize a coordinate selector into explicit integer indices.

    Parameters
    ----------
    axis_name : str
        Semantic axis name (``"genes"``, ``"components"``, etc.).
    axis_size : int
        Axis length in the current posterior tensor.
    axis_selector : Any
        User-provided selector for this axis.
    var_index : sequence or None
        Gene names used when ``axis_name == "genes"`` and selectors are
        string-based.

    Returns
    -------
    numpy.ndarray
        Integer indices to keep along the axis.

    Raises
    ------
    ValueError
        If selector values are invalid for the axis.
    """
    if axis_selector is None:
        return np.arange(axis_size, dtype=int)

    # Slice selectors are normalized into a concrete integer range.
    if isinstance(axis_selector, slice):
        return np.arange(axis_size, dtype=int)[axis_selector]

    arr = np.asarray(axis_selector)
    if arr.ndim == 0:
        arr = np.asarray([arr.item()])

    # Boolean masks are accepted as long as length matches axis size.
    if arr.dtype == np.bool_:
        if arr.shape[0] != axis_size:
            raise ValueError(
                f"Boolean mask for axis '{axis_name}' has length "
                f"{arr.shape[0]}, expected {axis_size}."
            )
        return np.flatnonzero(arr)

    # Integer selectors are used directly after bounds checking.
    if np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(int).reshape(-1)
        if np.any((idx < 0) | (idx >= axis_size)):
            raise ValueError(
                f"Selector for axis '{axis_name}' contains out-of-bounds "
                f"indices for size {axis_size}."
            )
        return idx

    # Gene selectors can be provided by name when var metadata exists.
    if axis_name == "genes":
        if var_index is None:
            raise ValueError(
                "Gene-name selection requires gene metadata ('var')."
            )
        name_to_idx = {str(name): i for i, name in enumerate(var_index)}
        idx = []
        for raw_name in arr.reshape(-1):
            key = str(raw_name)
            if key not in name_to_idx:
                raise ValueError(
                    f"Unknown gene name '{key}' in coords['genes']."
                )
            idx.append(name_to_idx[key])
        return np.asarray(idx, dtype=int)

    raise ValueError(
        f"Unsupported selector type for axis '{axis_name}': "
        f"{arr.dtype}."
    )


def _label_from_index(
    *,
    key: str,
    axis_names: Sequence[str],
    axis_indices: Sequence[int],
    var_index: Optional[Sequence[Any]],
) -> str:
    """Build a stable human-readable feature label.

    Parameters
    ----------
    key : str
        Posterior parameter key.
    axis_names : sequence of str
        Semantic non-sample axes in order.
    axis_indices : sequence of int
        Index tuple for a single flattened feature.
    var_index : sequence or None
        Optional gene names used to render gene labels.

    Returns
    -------
    str
        Label string corresponding to one matrix column.
    """
    if len(axis_names) == 0:
        return key

    parts = []
    for axis_name, axis_idx in zip(axis_names, axis_indices):
        if axis_name == "genes" and var_index is not None:
            axis_value = str(var_index[axis_idx])
        else:
            axis_value = str(axis_idx)
        axis_tag = axis_name[:-1] if axis_name.endswith("s") else axis_name
        parts.append(f"{axis_tag}={axis_value}")
    return f"{key}[{', '.join(parts)}]"


def posterior_samples_to_matrix(
    *,
    posterior_samples: Mapping[str, Any],
    base_layouts: Mapping[str, AxisLayout],
    model_config: Any,
    n_genes: int,
    n_cells: int,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    exclude_deterministic: bool = True,
    coords: Optional[Mapping[str, Any]] = None,
    var_index: Optional[Sequence[Any]] = None,
    descriptive_names: bool = False,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """Flatten posterior sample tensors to a 2D feature matrix.

    Parameters
    ----------
    posterior_samples : mapping
        Posterior dictionary with leading sample axis on tensor values.
    base_layouts : mapping of str to AxisLayout
        Semantic axis metadata keyed by parameter name. Layouts may be
        provided with or without sample dimension; this function enforces
        sample-aware layouts internally.
    model_config : Any
        Model configuration carrying axis-membership metadata used as a
        fallback when some posterior keys are missing from ``base_layouts``.
    n_genes : int
        Number of genes in the model.
    n_cells : int
        Number of cells in the model.
    include : iterable of str or None, optional
        Optional whitelist of parameter keys to include.
    exclude : iterable of str or None, optional
        Optional blacklist of parameter keys to exclude.
    exclude_deterministic : bool, default=True
        Exclude common deterministic/derived keys (e.g., ``mu`` in
        canonical parameterization).
    coords : mapping or None, optional
        Optional axis selectors such as ``{"genes": [0, 1]}`` or
        ``{"genes": ["GeneA", "GeneB"]}``.
    var_index : sequence or None, optional
        Gene names used for label rendering and string-based gene
        selection in ``coords``.
    descriptive_names : bool, default=False
        Whether posterior keys are descriptive aliases. Used to mirror
        deterministic-key filtering.

    Returns
    -------
    matrix : numpy.ndarray
        Posterior matrix with shape ``(n_draws, n_features)``.
    columns : list of str
        Feature label for each matrix column.
    metadata : list of dict
        Per-feature metadata aligned with ``columns``. Each record
        contains the parameter key and axis index/value mappings.

    Raises
    ------
    ValueError
        If no eligible posterior arrays are found or if posterior sample
        counts differ across included keys.
    """
    include_set = _to_name_set(include)
    exclude_set = _to_name_set(exclude) or set()
    if exclude_deterministic:
        exclude_set.update(
            _derived_parameter_keys(
                model_config=model_config,
                descriptive_names=descriptive_names,
            )
        )

    # Axis-membership hints improve fallback layout reconstruction.
    mixture_params, dataset_params = derive_axis_membership(model_config)
    n_datasets = getattr(model_config, "n_datasets", None)

    # Build sample-aware layouts for known keys so sample axis offsets are
    # interpreted correctly during flattening.
    sample_layouts = {
        key: layout.with_sample_dim()
        for key, layout in base_layouts.items()
    }

    coords = dict(coords or {})
    matrices: List[np.ndarray] = []
    columns: List[str] = []
    metadata: List[Dict[str, Any]] = []
    n_draws: Optional[int] = None

    for key, value in posterior_samples.items():
        if include_set is not None and key not in include_set:
            continue
        if key in exclude_set:
            continue
        if not hasattr(value, "shape"):
            continue

        arr = np.asarray(value)
        if arr.ndim == 0:
            continue

        # Resolve layout from stored metadata first, then infer when keys
        # are absent (common for deterministic sites).
        layout = sample_layouts.get(key)
        if layout is None:
            layout = infer_layout(
                key,
                arr,
                n_genes=n_genes,
                n_cells=n_cells,
                n_components=getattr(model_config, "n_components", None),
                n_datasets=n_datasets,
                mixture_params=mixture_params,
                dataset_params=dataset_params,
                has_sample_dim=True,
            )

        # Ensure sample axis exists; if absent, broadcast a singleton draw
        # dimension for consistent return shape.
        if not layout.has_sample_dim:
            layout = layout.with_sample_dim()
        if arr.ndim == len(layout.axes):
            arr = np.expand_dims(arr, axis=0)

        if n_draws is None:
            n_draws = int(arr.shape[0])
        elif int(arr.shape[0]) != n_draws:
            raise ValueError(
                f"Posterior key '{key}' has {arr.shape[0]} draws but "
                f"expected {n_draws}."
            )

        # Apply optional coordinate filtering along semantic non-sample axes.
        non_sample_shape = arr.shape[1:]
        selectors = []
        selected_indices_by_axis: Dict[str, np.ndarray] = {}
        for axis_name, axis_size in zip(layout.axes, non_sample_shape):
            selector = _selector_from_coords(
                axis_name=axis_name,
                axis_size=int(axis_size),
                axis_selector=coords.get(axis_name),
                var_index=var_index,
            )
            selectors.append(selector)
            selected_indices_by_axis[axis_name] = selector

        # np.ix_ keeps the cartesian product of selected indices for each axis.
        if selectors:
            indexing = np.ix_(*selectors)
            arr_selected = arr[(slice(None),) + indexing]
        else:
            arr_selected = arr

        feature_block = arr_selected.reshape(arr_selected.shape[0], -1)
        matrices.append(feature_block)

        # Build labels + machine-readable metadata in the exact flatten order.
        if layout.axes:
            selected_axis_lists = [
                selected_indices_by_axis[axis_name]
                for axis_name in layout.axes
            ]
            for index_tuple in np.ndindex(*[len(v) for v in selected_axis_lists]):
                axis_indices = [
                    int(selected_axis_lists[i][index_tuple[i]])
                    for i in range(len(index_tuple))
                ]
                label = _label_from_index(
                    key=key,
                    axis_names=layout.axes,
                    axis_indices=axis_indices,
                    var_index=var_index,
                )
                columns.append(label)
                metadata.append(
                    {
                        "parameter": key,
                        "axis_indices": {
                            axis_name: axis_idx
                            for axis_name, axis_idx in zip(
                                layout.axes, axis_indices
                            )
                        },
                        "axis_values": {
                            axis_name: (
                                str(var_index[axis_idx])
                                if (
                                    axis_name == "genes"
                                    and var_index is not None
                                )
                                else axis_idx
                            )
                            for axis_name, axis_idx in zip(
                                layout.axes, axis_indices
                            )
                        },
                    }
                )
        else:
            columns.append(key)
            metadata.append(
                {
                    "parameter": key,
                    "axis_indices": {},
                    "axis_values": {},
                }
            )

    if not matrices:
        raise ValueError(
            "No posterior sample arrays matched the requested filters."
        )

    matrix = np.concatenate(matrices, axis=1)
    return matrix, columns, metadata


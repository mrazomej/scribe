"""
Results classes for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any, List
from dataclasses import dataclass

import jax.numpy as jnp
import pandas as pd

from ..models.config import ModelConfig
from ..core.serialization import make_model_config_pickle_safe
from ._dataset import _build_cell_specific_keys
from ._gene_subsetting import build_gene_axis_by_key

try:
    from anndata import AnnData
except ImportError:
    AnnData = None

# Import mixins
from ._core import CoreResultsMixin
from ._parameter_extraction import ParameterExtractionMixin
from ._gene_subsetting import GeneSubsettingMixin
from ._component import ComponentMixin
from ._dataset import DatasetMixin
from ._model_helpers import ModelHelpersMixin
from ._sampling import SamplingMixin
from ._likelihood import LikelihoodMixin
from ._mixture_analysis import MixtureAnalysisMixin
from ._normalization import NormalizationMixin

# ------------------------------------------------------------------------------
# Base class for inference results
# ------------------------------------------------------------------------------


@dataclass
class ScribeSVIResults(
    CoreResultsMixin,
    ParameterExtractionMixin,
    GeneSubsettingMixin,
    ComponentMixin,
    DatasetMixin,
    ModelHelpersMixin,
    SamplingMixin,
    LikelihoodMixin,
    MixtureAnalysisMixin,
    NormalizationMixin,
):
    """
    Base class for SCRIBE variational inference results.

    This class stores the results from SCRIBE's variational inference procedure,
    including model parameters, loss history, dataset dimensions, and model
    configuration. It can optionally store metadata from an AnnData object and
    posterior/predictive samples.

    Attributes
    ----------
    params : Dict
        Dictionary of inferred model parameters from SCRIBE
    loss_history : jnp.ndarray
        Array containing the ELBO loss values during training
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_type : str
        Type of model used for inference
    model_config : ModelConfig
        Configuration object specifying model architecture and priors
    prior_params : Dict[str, Any]
        Dictionary of prior parameter values used during inference
    obs : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    var : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    n_obs : Optional[int]
        Number of observations (cells), if provided
    n_vars : Optional[int]
        Number of variables (genes), if provided
    posterior_samples : Optional[Dict]
        Samples of parameters from the posterior distribution, if generated
    predictive_samples : Optional[Dict]
        Predictive samples generated from the model, if generated
    n_components : Optional[int]
        Number of mixture components, if using a mixture model
    """

    # Core inference results
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    prior_params: Dict[str, Any]

    # Standard metadata from AnnData object
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # Optional results
    posterior_samples: Optional[Dict] = None
    predictive_samples: Optional[Dict] = None
    n_components: Optional[int] = None
    denoised_counts: Optional[jnp.ndarray] = None

    # Per-dataset cell counts for multi-dataset models.  Set during
    # inference when ``dataset_indices`` is available.  Used by
    # ``get_dataset(d)`` to set the correct ``n_cells`` on the returned
    # single-dataset view so that downstream PPC generation works with
    # the right number of cells.
    _n_cells_per_dataset: Optional[jnp.ndarray] = None

    # Per-cell dataset assignment array, shape ``(n_cells,)`` with values
    # in ``{0, ..., n_datasets-1}``.  Stored so that ``get_dataset(d)``
    # can subset **cell-specific** variational parameters (e.g.
    # ``phi_capture_loc``) whose shape is ``(n_cells,)`` rather than
    # ``(n_datasets, ...)``.
    _dataset_indices: Optional[jnp.ndarray] = None

    # Internal: tracks original gene count before subsetting (for amortizer
    # validation) When using amortized capture probability, counts must have
    # shape (n_cells, _original_n_genes) because the amortizer computes
    # sufficient statistics (e.g., total UMI count) by summing across ALL genes.
    _original_n_genes: Optional[int] = None

    # Internal: gene axis per param key for metadata-based subsetting (when
    # param_specs set). When present, _subset_params and
    # _subset_posterior_samples use this instead of heuristics.
    _gene_axis_by_key: Optional[Dict[str, int]] = None

    @classmethod
    def concat(
        cls,
        results_list: List["ScribeSVIResults"],
        *,
        align_genes: str = "strict",
        join: str = "cells",
        check_model: bool = True,
    ) -> "ScribeSVIResults":
        """Concatenate multiple SVI results objects along the cell axis.

        This utility is intended for combining result views that share the same
        model specification and gene space but differ in their cell dimension.
        Typical use cases include recombining per-dataset views produced by
        ``get_dataset()`` or combining partitioned inference outputs that share
        identical non-cell-specific parameters.

        Parameters
        ----------
        results_list : list of ScribeSVIResults
            Results objects to concatenate. At least one element is required.
        align_genes : {"strict"}, default="strict"
            Gene-alignment strategy. ``"strict"`` requires matching gene sets;
            if all objects include ``var``, differing gene order is resolved by
            reordering to match the first object.
        join : {"cells"}, default="cells"
            Concatenation axis. Only cell-axis concatenation is supported.
        check_model : bool, default=True
            If ``True``, require exact agreement for model-level fields
            (``model_type``, ``model_config``, and ``prior_params``).

        Returns
        -------
        ScribeSVIResults
            A new results object representing the concatenated cells.

        Raises
        ------
        ValueError
            If inputs are empty, incompatible, or use unsupported options.
        TypeError
            If inputs are not all ``ScribeSVIResults`` instances.
        """
        if not results_list:
            raise ValueError("results_list must contain at least one element.")
        if join != "cells":
            raise ValueError("Only join='cells' is currently supported.")
        if align_genes != "strict":
            raise ValueError("Only align_genes='strict' is currently supported.")

        for idx, res in enumerate(results_list):
            if not isinstance(res, cls):
                raise TypeError(
                    f"All entries must be {cls.__name__}; got {type(res)} at index {idx}."
                )

        reference = results_list[0]
        if check_model:
            _validate_svi_model_compatibility(reference, results_list)

        aligned_results = _align_svi_genes_to_reference(
            reference=reference, results_list=results_list
        )
        first = aligned_results[0]

        # Build the key set for cell-specific variational parameters once from
        # metadata, then apply the same policy to all objects.
        cell_param_keys = _build_cell_specific_keys(
            first.model_config.param_specs or [],
            first.params,
        )

        params = _concat_svi_param_dicts(
            dicts=[res.params for res in aligned_results],
            cell_specific_keys=cell_param_keys,
        )

        posterior_samples = _concat_svi_optional_sample_dicts(
            dicts=[
                res.posterior_samples for res in aligned_results
            ],
            cell_specific_keys=cell_param_keys,
            field_name="posterior_samples",
        )

        predictive_samples = _concat_svi_predictive_samples(
            [res.predictive_samples for res in aligned_results]
        )

        obs = _concat_optional_obs([res.obs for res in aligned_results])
        var = first.var.copy() if first.var is not None else None
        uns = _merge_optional_uns([res.uns for res in aligned_results])

        n_cells_total = int(sum(res.n_cells for res in aligned_results))
        n_obs_total = (
            int(sum(res.n_obs for res in aligned_results))
            if all(res.n_obs is not None for res in aligned_results)
            else (obs.shape[0] if obs is not None else None)
        )

        loss_history = jnp.concatenate(
            [jnp.asarray(res.loss_history) for res in aligned_results], axis=0
        )

        combined = cls(
            params=params,
            loss_history=loss_history,
            n_cells=n_cells_total,
            n_genes=first.n_genes,
            model_type=first.model_type,
            model_config=first.model_config,
            prior_params=first.prior_params,
            obs=obs,
            var=var,
            uns=uns,
            n_obs=n_obs_total,
            n_vars=first.n_genes,
            posterior_samples=posterior_samples,
            predictive_samples=predictive_samples,
            n_components=first.n_components,
            denoised_counts=None,
            _n_cells_per_dataset=_merge_cells_per_dataset(
                [getattr(res, "_n_cells_per_dataset", None) for res in aligned_results]
            ),
            _dataset_indices=_concat_dataset_indices(
                [getattr(res, "_dataset_indices", None) for res in aligned_results]
            ),
            _original_n_genes=_merge_original_n_genes(aligned_results),
            _gene_axis_by_key=getattr(first, "_gene_axis_by_key", None),
        )
        return combined

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle-safe state for ``ScribeSVIResults``.

        Returns
        -------
        Dict[str, Any]
            Instance state with a serialization-safe ``model_config``.
        """
        state = dict(self.__dict__)
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore instance state after unpickling."""
        self.__dict__.update(state)


def _validate_svi_model_compatibility(
    reference: ScribeSVIResults, results_list: List[ScribeSVIResults]
) -> None:
    """Validate model-level compatibility for SVI concatenation."""
    for idx, res in enumerate(results_list[1:], start=1):
        if res.model_type != reference.model_type:
            raise ValueError(
                f"model_type mismatch at index {idx}: "
                f"{res.model_type} != {reference.model_type}"
            )
        if res.model_config != reference.model_config:
            raise ValueError(f"model_config mismatch at index {idx}.")
        if res.prior_params != reference.prior_params:
            raise ValueError(f"prior_params mismatch at index {idx}.")
        if set(res.params.keys()) != set(reference.params.keys()):
            raise ValueError(f"params keys mismatch at index {idx}.")


def _align_svi_genes_to_reference(
    reference: ScribeSVIResults, results_list: List[ScribeSVIResults]
) -> List[ScribeSVIResults]:
    """Align SVI results to the reference gene order when ``var`` is available."""
    all_have_var = all(res.var is not None for res in results_list)
    if not all_have_var:
        if any(res.n_genes != reference.n_genes for res in results_list):
            raise ValueError(
                "All results must have identical n_genes when var metadata is missing."
            )
        return results_list

    ref_names = pd.Index(reference.var.index)
    aligned: List[ScribeSVIResults] = []
    for idx, res in enumerate(results_list):
        names = pd.Index(res.var.index)
        if len(names) != len(ref_names):
            raise ValueError(f"Gene count mismatch at index {idx}.")
        if set(names) != set(ref_names):
            raise ValueError(f"Gene set mismatch at index {idx}.")
        if names.equals(ref_names):
            aligned.append(res)
            continue

        indexer = names.get_indexer(ref_names)
        if (indexer < 0).any():
            raise ValueError(f"Could not align genes at index {idx}.")
        aligned.append(_reorder_svi_result_genes(res, indexer))

    return aligned


def _reorder_svi_result_genes(
    result: ScribeSVIResults, gene_indexer: jnp.ndarray
) -> ScribeSVIResults:
    """Return a copy of SVI results with all gene-specific fields reordered."""
    gene_indexer = jnp.asarray(gene_indexer)
    param_gene_axis = _svi_param_gene_axes(result)
    posterior_gene_axis = {
        key: axis + 1 for key, axis in param_gene_axis.items()
    }

    params = _reorder_dict_by_gene_axis(result.params, param_gene_axis, gene_indexer)
    posterior_samples = (
        _reorder_dict_by_gene_axis(
            result.posterior_samples, posterior_gene_axis, gene_indexer
        )
        if result.posterior_samples is not None
        else None
    )
    predictive_samples = (
        jnp.take(result.predictive_samples, gene_indexer, axis=-1)
        if result.predictive_samples is not None
        else None
    )
    var = result.var.iloc[list(map(int, gene_indexer.tolist()))].copy()

    return type(result)(
        params=params,
        loss_history=result.loss_history,
        n_cells=result.n_cells,
        n_genes=result.n_genes,
        model_type=result.model_type,
        model_config=result.model_config,
        prior_params=result.prior_params,
        obs=result.obs,
        var=var,
        uns=result.uns,
        n_obs=result.n_obs,
        n_vars=result.n_vars,
        posterior_samples=posterior_samples,
        predictive_samples=predictive_samples,
        n_components=result.n_components,
        denoised_counts=None,
        _n_cells_per_dataset=getattr(result, "_n_cells_per_dataset", None),
        _dataset_indices=getattr(result, "_dataset_indices", None),
        _original_n_genes=getattr(result, "_original_n_genes", None),
        _gene_axis_by_key=param_gene_axis or getattr(result, "_gene_axis_by_key", None),
    )


def _svi_param_gene_axes(result: ScribeSVIResults) -> Dict[str, int]:
    """Infer gene-axis mapping for SVI params using stored metadata when possible."""
    existing = getattr(result, "_gene_axis_by_key", None)
    if existing is not None:
        return existing
    return (
        build_gene_axis_by_key(
            result.model_config.param_specs or [],
            result.params,
            result.n_genes,
        )
        or {}
    )


def _reorder_dict_by_gene_axis(data: Dict, axis_by_key: Dict[str, int], indexer) -> Dict:
    """Apply deterministic reordering on keys that carry a gene axis."""
    reordered: Dict[str, Any] = {}
    for key, value in data.items():
        if not hasattr(value, "ndim") or key not in axis_by_key:
            reordered[key] = value
            continue
        reordered[key] = jnp.take(value, indexer, axis=axis_by_key[key])
    return reordered


def _concat_svi_param_dicts(
    dicts: List[Dict[str, Any]], cell_specific_keys: set
) -> Dict[str, Any]:
    """Concatenate SVI variational params with a cell-aware policy."""
    keys = dicts[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        values = [d[key] for d in dicts]
        if key in cell_specific_keys:
            out[key] = jnp.concatenate(values, axis=0)
            continue
        if not _all_equal(values):
            raise ValueError(
                f"Non-cell-specific parameter '{key}' differs across results."
            )
        out[key] = values[0]
    return out


def _concat_svi_optional_sample_dicts(
    dicts: List[Optional[Dict[str, Any]]],
    cell_specific_keys: set,
    field_name: str,
) -> Optional[Dict[str, Any]]:
    """Concatenate optional SVI sample dictionaries with validation."""
    if all(d is None for d in dicts):
        return None
    if any(d is None for d in dicts):
        raise ValueError(
            f"{field_name} must be present on all inputs or none of them."
        )

    keys = dicts[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        values = [d[key] for d in dicts]
        if key in cell_specific_keys:
            for idx, val in enumerate(values):
                if not hasattr(val, "ndim") or val.ndim < 2:
                    raise ValueError(
                        f"Cell-specific '{field_name}[{key}]' must have ndim >= 2; "
                        f"found {getattr(val, 'ndim', None)} at index {idx}."
                    )
            sample_sizes = [val.shape[0] for val in values]
            if len(set(sample_sizes)) != 1:
                raise ValueError(
                    f"Sample-axis mismatch for '{field_name}[{key}]': {sample_sizes}"
                )
            out[key] = jnp.concatenate(values, axis=1)
            continue
        if not _all_equal(values):
            raise ValueError(
                f"Non-cell-specific '{field_name}[{key}]' differs across results."
            )
        out[key] = values[0]
    return out


def _concat_svi_predictive_samples(
    values: List[Optional[jnp.ndarray]],
) -> Optional[jnp.ndarray]:
    """Concatenate predictive samples along the cell axis."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(
            "predictive_samples must be present on all inputs or none of them."
        )
    sample_sizes = [v.shape[0] for v in values]
    gene_sizes = [v.shape[-1] for v in values]
    if len(set(sample_sizes)) != 1:
        raise ValueError(f"predictive_samples sample-axis mismatch: {sample_sizes}")
    if len(set(gene_sizes)) != 1:
        raise ValueError(f"predictive_samples gene-axis mismatch: {gene_sizes}")
    return jnp.concatenate(values, axis=1)


def _concat_optional_obs(values: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Concatenate optional ``obs`` frames row-wise."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("obs must be present on all inputs or none of them.")
    return pd.concat(values, axis=0)


def _merge_optional_uns(values: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Merge ``uns`` by requiring all non-null values to be equal."""
    if all(v is None for v in values):
        return None
    base = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if not _all_equal([base, value]):
            raise ValueError(f"uns mismatch at index {idx}.")
    return base


def _merge_cells_per_dataset(values: List[Optional[jnp.ndarray]]) -> Optional[jnp.ndarray]:
    """Merge per-dataset cell counts by element-wise sum when available."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(
            "_n_cells_per_dataset must be present on all inputs or none of them."
        )
    lengths = [len(v) for v in values]
    if len(set(lengths)) != 1:
        raise ValueError(f"_n_cells_per_dataset length mismatch: {lengths}")
    stacked = jnp.stack(values, axis=0)
    return jnp.sum(stacked, axis=0)


def _concat_dataset_indices(values: List[Optional[jnp.ndarray]]) -> Optional[jnp.ndarray]:
    """Concatenate dataset-index vectors when present."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("_dataset_indices must be present on all inputs or none of them.")
    return jnp.concatenate(values, axis=0)


def _merge_original_n_genes(results_list: List[ScribeSVIResults]) -> Optional[int]:
    """Merge tracked original gene counts with consistency validation."""
    values = [getattr(r, "_original_n_genes", None) for r in results_list]
    non_null = [v for v in values if v is not None]
    if not non_null:
        return None
    if len(set(non_null)) != 1:
        raise ValueError(
            f"_original_n_genes mismatch across inputs: {non_null}"
        )
    return int(non_null[0])


def _all_equal(values: List[Any]) -> bool:
    """Return ``True`` when all values compare equal with array support."""
    first = values[0]
    for other in values[1:]:
        if not _value_equal(first, other):
            return False
    return True


def _value_equal(left: Any, right: Any) -> bool:
    """Compare values with explicit handling for arrays and pandas objects."""
    if type(left) is not type(right):
        return False
    if isinstance(left, pd.DataFrame):
        return left.equals(right)
    if isinstance(left, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_value_equal(left[k], right[k]) for k in left)
    if hasattr(left, "shape") and hasattr(right, "shape"):
        return bool(jnp.array_equal(jnp.asarray(left), jnp.asarray(right)))
    return left == right

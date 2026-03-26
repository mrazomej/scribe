"""
Results class for SCRIBE MCMC inference.

``ScribeMCMCResults`` is a ``@dataclass`` that wraps a NumPyro MCMC object
and composes analysis functionality from mixins, mirroring the SVI results
architecture.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

import jax.numpy as jnp
import pandas as pd

from ..models.config import ModelConfig
from ..core.serialization import make_model_config_pickle_safe
from ._dataset import _build_cell_specific_keys
from ..svi._gene_subsetting import build_gene_axis_by_key

# Mixin imports
from ._parameter_extraction import ParameterExtractionMixin
from ._gene_subsetting import GeneSubsettingMixin
from ._component import ComponentMixin
from ._dataset import DatasetMixin
from ._model_helpers import ModelHelpersMixin
from ._sampling import SamplingMixin
from ._likelihood import LikelihoodMixin
from ._normalization import NormalizationMixin
from ._mixture_analysis import MixtureAnalysisMixin


# ==============================================================================
# MCMC Results
# ==============================================================================


@dataclass
class ScribeMCMCResults(
    ParameterExtractionMixin,
    GeneSubsettingMixin,
    ComponentMixin,
    DatasetMixin,
    ModelHelpersMixin,
    SamplingMixin,
    LikelihoodMixin,
    NormalizationMixin,
    MixtureAnalysisMixin,
):
    """SCRIBE MCMC results.

    Stores posterior samples and provides analysis methods via mixins.
    The underlying ``numpyro.infer.MCMC`` object is wrapped (composition)
    rather than inherited, so gene/component subsetting always returns
    another ``ScribeMCMCResults`` instance.

    Attributes
    ----------
    samples : Dict
        Raw posterior samples keyed by parameter name.
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes in the dataset.
    model_type : str
        Model identifier (e.g. ``"nbdm"``, ``"zinb_mix"``).
    model_config : ModelConfig
        Configuration used for inference.
    prior_params : Dict[str, Any]
        Prior parameter values used during inference.
    obs : Optional[pd.DataFrame]
        Cell-level metadata from ``adata.obs``.
    var : Optional[pd.DataFrame]
        Gene-level metadata from ``adata.var``.
    uns : Optional[Dict]
        Unstructured metadata from ``adata.uns``.
    n_obs : Optional[int]
        Number of observations (cells).
    n_vars : Optional[int]
        Number of variables (genes).
    predictive_samples : Optional[jnp.ndarray]
        Predictive samples from :meth:`get_ppc_samples`.
    n_components : Optional[int]
        Number of mixture components (``None`` for non-mixture models).
    denoised_counts : Optional[jnp.ndarray]
        Denoised counts from :meth:`denoise_counts`.
    _mcmc : Optional[Any]
        Wrapped ``numpyro.infer.MCMC`` object for diagnostics.
        ``None`` on subsets produced by gene/component indexing.
    """

    # -- core fields ---------------------------------------------------------
    samples: Dict
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    prior_params: Dict[str, Any]

    # -- AnnData metadata ----------------------------------------------------
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # -- optional computed results -------------------------------------------
    predictive_samples: Optional[jnp.ndarray] = None
    n_components: Optional[int] = None
    denoised_counts: Optional[jnp.ndarray] = None

    # Per-dataset cell counts for multi-dataset models.  Set during
    # inference when ``dataset_indices`` is available.  Used by
    # ``get_dataset(d)`` to set the correct ``n_cells`` on the returned
    # single-dataset view.
    _n_cells_per_dataset: Optional[jnp.ndarray] = None

    # Per-cell dataset assignment array, shape ``(n_cells,)`` with values
    # in ``{0, ..., n_datasets-1}``.  Stored so that ``get_dataset(d)``
    # can subset cell-specific parameters.
    _dataset_indices: Optional[jnp.ndarray] = None

    # Keys that were stacked along a new dataset axis during concat
    # promotion (single-dataset → multi-dataset).  Used by get_dataset()
    # to know which samples need dataset-axis slicing even though they
    # lack ``is_dataset`` in their ParamSpec.
    _promoted_dataset_keys: Optional[set] = None

    # -- wrapped MCMC object (None on subsets) -------------------------------
    _mcmc: Optional[Any] = field(default=None, repr=False)

    @classmethod
    def concat(
        cls,
        results_list: List["ScribeMCMCResults"],
        *,
        align_genes: str = "assume_aligned",
        join: str = "cells",
        check_model: bool = True,
        validation: str = "var_only",
    ) -> "ScribeMCMCResults":
        """Concatenate multiple MCMC results objects along the cell axis.

        The method supports combining objects that represent the same model and
        gene space while differing in cell count. Cell-specific posterior
        samples are concatenated along the cell axis, while non-cell-specific
        samples must be identical across inputs.

        Parameters
        ----------
        results_list : list of ScribeMCMCResults
            Results objects to concatenate. At least two elements are required.
        align_genes : {"strict", "assume_aligned"}, default="strict"
            Gene-alignment strategy. ``"strict"`` requires matching gene sets;
            if all objects include ``var``, differing gene order is resolved by
            reordering to match the first object. ``"assume_aligned"`` skips
            gene-set/order validation and assumes all inputs are already aligned.
        join : {"cells"}, default="cells"
            Concatenation axis. Only cell-axis concatenation is supported.
        check_model : bool, default=True
            If ``True``, require exact agreement for model-level fields
            (``model_type``, ``model_config``, and ``prior_params``).
        validation : {"strict", "var_only"}, default="strict"
            Validation policy for non-cell-specific fields.
            ``"strict"`` enforces deep equality checks for shared sample sites
            and metadata. ``"var_only"`` performs fast key-level checks and
            relies on the user-trusted model fit plus gene validation from
            ``var`` (or ``n_genes`` when ``var`` is missing), taking
            non-cell-specific values from the first object.

        Returns
        -------
        ScribeMCMCResults
            Concatenated MCMC results with ``_mcmc=None``.

        Raises
        ------
        ValueError
            If inputs are empty, incompatible, or use unsupported options.
        TypeError
            If inputs are not all ``ScribeMCMCResults`` instances.
        """
        # Guard against accidentally passing a single results object instead of
        # a list/tuple. Because results support ``__getitem__``, iterating over
        # a single object can trigger expensive implicit indexing.
        if isinstance(results_list, cls):
            raise TypeError(
                "results_list must be a sequence of results, e.g. "
                "ScribeMCMCResults.concat([res_a, res_b])."
            )
        if not results_list or len(results_list) < 2:
            raise ValueError(
                "results_list must contain at least two elements. "
                "Note: concat is a classmethod — call "
                "ScribeMCMCResults.concat([res_a, res_b]), not "
                "res_a.concat([res_b])."
            )
        if join != "cells":
            raise ValueError("Only join='cells' is currently supported.")
        if align_genes not in {"strict", "assume_aligned"}:
            raise ValueError(
                "align_genes must be one of {'strict', 'assume_aligned'}."
            )
        if validation not in {"strict", "var_only"}:
            raise ValueError(
                "validation must be one of {'strict', 'var_only'}."
            )

        for idx, res in enumerate(results_list):
            if not isinstance(res, cls):
                raise TypeError(
                    f"All entries must be {cls.__name__}; got {type(res)} at index {idx}."
                )

        reference = results_list[0]
        if check_model:
            _validate_mcmc_model_compatibility(reference, results_list)

        aligned_results = (
            _align_mcmc_genes_to_reference(
                reference=reference, results_list=results_list
            )
            if align_genes == "strict"
            else results_list
        )
        first = aligned_results[0]

        _validate_equal_mcmc_sample_sizes(aligned_results)

        # Classify cell-specific sample sites from ParamSpec metadata so only
        # cell-axis variables are concatenated along axis 1.
        cell_sample_keys = _build_cell_specific_keys(
            first.model_config.param_specs or [],
            first.samples,
        )
        samples = _concat_mcmc_samples(
            sample_dicts=[res.samples for res in aligned_results],
            cell_specific_keys=cell_sample_keys,
            strict=(validation == "strict"),
        )

        obs = _concat_optional_obs([res.obs for res in aligned_results])
        var = first.var.copy() if first.var is not None else None
        uns = _merge_optional_uns(
            [res.uns for res in aligned_results],
            strict=(validation == "strict"),
        )

        n_cells_total = int(sum(res.n_cells for res in aligned_results))
        n_obs_total = (
            int(sum(res.n_obs for res in aligned_results))
            if all(res.n_obs is not None for res in aligned_results)
            else (obs.shape[0] if obs is not None else None)
        )

        # --- dataset metadata: merge existing or promote single-dataset ---
        n_cells_per_dataset = _merge_cells_per_dataset(
            [
                getattr(res, "_n_cells_per_dataset", None)
                for res in aligned_results
            ]
        )
        dataset_indices = _concat_dataset_indices(
            [
                getattr(res, "_dataset_indices", None)
                for res in aligned_results
            ]
        )

        # When all inputs are single-dataset and we are combining more than
        # one, promote to a multi-dataset result so that ``get_dataset(i)``
        # can retrieve the i-th original result's cells.
        n_inputs = len(aligned_results)
        combined_config = first.model_config
        promoted_dataset_keys = None
        if n_cells_per_dataset is None and n_inputs > 1:
            n_cells_per_dataset = jnp.array(
                [int(res.n_cells) for res in aligned_results],
                dtype=jnp.int32,
            )
            dataset_indices = jnp.concatenate(
                [
                    jnp.full(int(res.n_cells), i, dtype=jnp.int32)
                    for i, res in enumerate(aligned_results)
                ]
            )
            combined_config = first.model_config.model_copy(
                update={"n_datasets": n_inputs}
            )

            # Stack non-cell-specific samples along a new dataset axis (1,
            # after the sample axis at 0) so get_dataset(i) can recover
            # per-dataset values.
            promoted_dataset_keys = set()
            for key in samples:
                if key not in cell_sample_keys:
                    stacked = jnp.stack(
                        [res.samples[key] for res in aligned_results],
                        axis=1,
                    )
                    samples[key] = stacked
                    promoted_dataset_keys.add(key)

        return cls(
            samples=samples,
            n_cells=n_cells_total,
            n_genes=first.n_genes,
            model_type=first.model_type,
            model_config=combined_config,
            prior_params=first.prior_params,
            obs=obs,
            var=var,
            uns=uns,
            n_obs=n_obs_total,
            n_vars=first.n_genes,
            predictive_samples=None,
            n_components=first.n_components,
            denoised_counts=None,
            _n_cells_per_dataset=n_cells_per_dataset,
            _dataset_indices=dataset_indices,
            _promoted_dataset_keys=promoted_dataset_keys,
            _mcmc=None,
        )

    # -------------------------------------------------------------------------
    # Post-init validation
    # -------------------------------------------------------------------------

    def __post_init__(self):
        """Validate model configuration and set derived attributes."""
        if (
            self.n_components is None
            and self.model_config.n_components is not None
        ):
            self.n_components = self.model_config.n_components

        self._validate_model_config()

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

    @classmethod
    def from_mcmc(
        cls,
        mcmc,
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ) -> "ScribeMCMCResults":
        """Create results from an existing ``numpyro.infer.MCMC`` instance.

        Extracts samples once and stores the MCMC object for diagnostics.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Completed MCMC run.
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        model_type : str
            Model identifier.
        model_config : ModelConfig
            Model configuration.
        prior_params : Dict[str, Any]
            Prior parameter values.
        **kwargs
            Forwarded to the dataclass constructor (e.g. ``obs``, ``var``).

        Returns
        -------
        ScribeMCMCResults
        """
        return cls(
            samples=mcmc.get_samples(group_by_chain=False),
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            _mcmc=mcmc,
            **kwargs,
        )

    @classmethod
    def from_anndata(
        cls,
        mcmc,
        adata,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ) -> "ScribeMCMCResults":
        """Create results from an MCMC instance and AnnData object.

        Parameters
        ----------
        mcmc : numpyro.infer.MCMC
            Completed MCMC run.
        adata : AnnData
            AnnData object with cell/gene metadata.
        model_type : str
            Model identifier.
        model_config : ModelConfig
            Model configuration.
        prior_params : Dict[str, Any]
            Prior parameter values.
        **kwargs
            Forwarded to the dataclass constructor.

        Returns
        -------
        ScribeMCMCResults
        """
        return cls.from_mcmc(
            mcmc=mcmc,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Posterior sample access
    # -------------------------------------------------------------------------

    def get_posterior_samples(self, descriptive_names: bool = False) -> Dict:
        """Return posterior samples.

        MCMC samples already contain canonical parameters (``p``, ``r``,
        ``mixing_weights``, etc.) because derived parameters are
        registered as ``numpyro.deterministic`` sites and unconstrained
        specs sample via ``TransformedDistribution`` in constrained
        space.

        Parameters
        ----------
        descriptive_names : bool, default=False
            If True, rename dict keys from internal short names to
            user-friendly descriptive names.

        Returns
        -------
        Dict
            Parameter name -> sample array.
        """
        from ..models.config.parameter_mapping import rename_dict_keys

        return rename_dict_keys(self.samples, descriptive_names)

    def get_samples(self, group_by_chain: bool = False) -> Dict:
        """Return samples with optional chain grouping.

        Parameters
        ----------
        group_by_chain : bool, default=False
            Preserve the chain dimension (requires the original MCMC
            object).

        Returns
        -------
        Dict
            Parameter samples.
        """
        if group_by_chain:
            if self._mcmc is None:
                raise RuntimeError(
                    "group_by_chain requires the original MCMC object "
                    "(not available on subsets)."
                )
            return self._mcmc.get_samples(group_by_chain=True)
        return self.samples

    # -------------------------------------------------------------------------
    # MCMC diagnostic delegation
    # -------------------------------------------------------------------------

    def print_summary(self, **kwargs):
        """Print MCMC summary statistics (delegates to the wrapped MCMC).

        Raises
        ------
        RuntimeError
            If the MCMC object is not available (e.g. on subsets).
        """
        if self._mcmc is None:
            raise RuntimeError(
                "print_summary requires the original MCMC object "
                "(not available on subsets)."
            )
        self._mcmc.print_summary(**kwargs)

    def get_extra_fields(self, **kwargs) -> Dict:
        """Return MCMC extra fields (e.g. potential_energy, diverging).

        Returns an empty dict when the MCMC object is not available
        (subsets).
        """
        if self._mcmc is None:
            return {}
        return self._mcmc.get_extra_fields(**kwargs)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_model_config(self):
        """Validate that model_config is consistent with model_type."""
        expected_base = (
            self.model_type[:-4]
            if self.model_type.endswith("_mix")
            else self.model_type
        )
        if self.model_config.base_model != expected_base:
            raise ValueError(
                f"Model type '{self.model_type}' does not match config "
                f"base model '{self.model_config.base_model}'"
            )

        if self.n_components is not None:
            if not self.model_type.endswith("_mix"):
                raise ValueError(
                    f"Model type '{self.model_type}' is not a mixture model "
                    f"but n_components={self.n_components} was specified"
                )
            if self.model_config.n_components != self.n_components:
                raise ValueError(
                    f"n_components mismatch: {self.n_components} vs "
                    f"{self.model_config.n_components} in model_config"
                )

    # -------------------------------------------------------------------------
    # Convenience property
    # -------------------------------------------------------------------------

    @property
    def posterior_samples(self) -> Dict:
        """Posterior samples (read-only property)."""
        return self.get_posterior_samples()

    # -------------------------------------------------------------------------
    # Pickle support
    # -------------------------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle-safe state for ``ScribeMCMCResults``.

        Notes
        -----
        The wrapped ``_mcmc`` object retains local closure functions from model
        building and is intentionally dropped to ensure portability.
        """
        state = dict(self.__dict__)
        state["_mcmc"] = None
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore instance state after unpickling."""
        self.__dict__.update(state)


def _validate_mcmc_model_compatibility(
    reference: ScribeMCMCResults, results_list: List[ScribeMCMCResults]
) -> None:
    """Validate model-level compatibility for MCMC concatenation."""
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
        if set(res.samples.keys()) != set(reference.samples.keys()):
            raise ValueError(f"samples keys mismatch at index {idx}.")


def _align_mcmc_genes_to_reference(
    reference: ScribeMCMCResults, results_list: List[ScribeMCMCResults]
) -> List[ScribeMCMCResults]:
    """Align MCMC results to the reference gene order when ``var`` is available."""
    all_have_var = all(res.var is not None for res in results_list)
    if not all_have_var:
        if any(res.n_genes != reference.n_genes for res in results_list):
            raise ValueError(
                "All results must have identical n_genes when var metadata is missing."
            )
        return results_list

    ref_names = pd.Index(reference.var.index)
    aligned: List[ScribeMCMCResults] = []
    for idx, res in enumerate(results_list):
        names = pd.Index(res.var.index)
        # Fast no-op path when names are already identical.
        if names.equals(ref_names):
            aligned.append(res)
            continue
        if len(names) != len(ref_names):
            raise ValueError(f"Gene count mismatch at index {idx}.")
        if set(names) != set(ref_names):
            raise ValueError(f"Gene set mismatch at index {idx}.")

        indexer = names.get_indexer(ref_names)
        if (indexer < 0).any():
            raise ValueError(f"Could not align genes at index {idx}.")
        aligned.append(_reorder_mcmc_result_genes(res, indexer))

    return aligned


def _reorder_mcmc_result_genes(
    result: ScribeMCMCResults, gene_indexer: jnp.ndarray
) -> ScribeMCMCResults:
    """Return a copy of MCMC results with all gene-specific sample sites reordered."""
    gene_indexer = jnp.asarray(gene_indexer)
    sample_gene_axis = _mcmc_sample_gene_axes(result)
    samples = _reorder_dict_by_gene_axis(
        result.samples, sample_gene_axis, gene_indexer
    )
    var = result.var.iloc[list(map(int, gene_indexer.tolist()))].copy()

    return type(result)(
        samples=samples,
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
        predictive_samples=None,
        n_components=result.n_components,
        denoised_counts=None,
        _n_cells_per_dataset=getattr(result, "_n_cells_per_dataset", None),
        _dataset_indices=getattr(result, "_dataset_indices", None),
        _mcmc=None,
    )


def _mcmc_sample_gene_axes(result: ScribeMCMCResults) -> Dict[str, int]:
    """Infer gene-axis mapping for MCMC samples from ParamSpec metadata."""
    specs = result.model_config.param_specs or []
    if specs:
        spec_axes: Dict[str, int] = {}
        for spec in specs:
            is_gene_specific = getattr(spec, "is_gene_specific", False) or (
                "n_genes" in getattr(spec, "shape_dims", ())
            )
            if not is_gene_specific or "n_genes" not in getattr(
                spec, "shape_dims", ()
            ):
                continue
            axis_without_sample = list(spec.shape_dims).index("n_genes")
            axis_with_sample = axis_without_sample + 1
            for key in result.samples:
                if key == spec.name and hasattr(result.samples[key], "ndim"):
                    if result.samples[key].ndim > axis_with_sample:
                        spec_axes[key] = axis_with_sample

        if spec_axes:
            return spec_axes

    fallback = (
        build_gene_axis_by_key(specs, result.samples, result.n_genes) or {}
    )
    return fallback


def _validate_equal_mcmc_sample_sizes(
    results_list: List[ScribeMCMCResults],
) -> None:
    """Ensure all MCMC results carry the same posterior draw count."""
    sample_counts: List[int] = []
    for res in results_list:
        count = None
        for value in res.samples.values():
            if hasattr(value, "ndim") and value.ndim > 0:
                count = int(value.shape[0])
                break
        if count is None:
            raise ValueError(
                "Could not infer MCMC sample count from samples dictionary."
            )
        sample_counts.append(count)
    if len(set(sample_counts)) != 1:
        raise ValueError(
            "All concatenated MCMC results must have the same number of samples; "
            f"got {sample_counts}."
        )


def _concat_mcmc_samples(
    sample_dicts: List[Dict[str, Any]], cell_specific_keys: set, strict: bool
) -> Dict[str, Any]:
    """Concatenate MCMC sample dictionaries with cell-aware semantics."""
    keys = sample_dicts[0].keys()
    out: Dict[str, Any] = {}
    for key in keys:
        values = [d[key] for d in sample_dicts]
        if key in cell_specific_keys:
            for idx, val in enumerate(values):
                if not hasattr(val, "ndim") or val.ndim < 2:
                    raise ValueError(
                        f"Cell-specific sample '{key}' must have ndim >= 2; "
                        f"found {getattr(val, 'ndim', None)} at index {idx}."
                    )
            out[key] = jnp.concatenate(values, axis=1)
            continue
        if strict and not _all_equal(values):
            raise ValueError(
                f"Non-cell-specific sample '{key}' differs across results."
            )
        out[key] = values[0]
    return out


def _reorder_dict_by_gene_axis(
    data: Dict, axis_by_key: Dict[str, int], indexer
) -> Dict:
    """Apply deterministic reordering on keys that carry a gene axis."""
    reordered: Dict[str, Any] = {}
    for key, value in data.items():
        if not hasattr(value, "ndim") or key not in axis_by_key:
            reordered[key] = value
            continue
        reordered[key] = jnp.take(value, indexer, axis=axis_by_key[key])
    return reordered


def _concat_optional_obs(
    values: List[Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """Concatenate optional ``obs`` frames row-wise."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("obs must be present on all inputs or none of them.")
    return pd.concat(values, axis=0)


def _merge_optional_uns(
    values: List[Optional[Dict[str, Any]]], strict: bool
) -> Optional[Dict[str, Any]]:
    """Merge ``uns`` with configurable strictness."""
    if all(v is None for v in values):
        return None
    base = values[0]
    # In fast mode, trust the caller and keep the first uns payload.
    if not strict:
        return base
    for idx, value in enumerate(values[1:], start=1):
        if not _all_equal([base, value]):
            raise ValueError(f"uns mismatch at index {idx}.")
    return base


def _merge_cells_per_dataset(
    values: List[Optional[jnp.ndarray]],
) -> Optional[jnp.ndarray]:
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


def _concat_dataset_indices(
    values: List[Optional[jnp.ndarray]],
) -> Optional[jnp.ndarray]:
    """Concatenate dataset-index vectors when present."""
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(
            "_dataset_indices must be present on all inputs or none of them."
        )
    return jnp.concatenate(values, axis=0)


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

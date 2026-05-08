"""
Mutable state bag threaded through :func:`scribe.api.fit` stages.

``FitContext`` captures every piece of state that flows between the
stages of the ``fit`` orchestration pipeline.  Each stage function
receives the context, reads the fields it needs, and writes back any
fields it produces.  Field names intentionally mirror the local
variable names in the original monolithic ``fit`` body so that diffs
against the pre-partition code are minimal.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from anndata import AnnData

    from ..models.config import DataConfig, InferenceConfig, ModelConfig


@dataclass
class FitContext:
    """
    Shared state for the ``fit`` pipeline.

    Attributes are grouped by lifecycle: *inputs* are set once by the
    orchestrator before any stage runs; *evolving* fields are written
    and/or updated by successive stages.

    Parameters
    ----------
    counts : Any
        Raw user input (array or AnnData) before processing.
    adata : AnnData or None
        Resolved AnnData reference (``None`` when ``counts`` is a raw
        array).
    count_data : Any
        Processed count matrix (cells x genes JAX array).
    n_cells : int
        Number of cells after data processing.
    n_genes : int
        Number of genes after data processing (may shrink after gene
        coverage filtering).
    data_config : DataConfig
        Data access configuration (``cells_axis``, ``layer``).
    model : str
        Resolved model string (may be mutated by ``model_flags`` stage).
    priors : dict or None
        Prior overrides dict (may be mutated by ``lnm_priors`` stage).
    n_components : int or None
        Number of mixture components (may be inferred by
        ``annotation_priors`` stage).
    n_datasets : int or None
        Number of datasets (inferred from ``dataset_key``).
    dataset_indices : Any
        Per-cell dataset index array, or ``None``.
    model_config : ModelConfig or None
        Built or user-supplied model configuration.
    inference_config : InferenceConfig or None
        Built or user-supplied inference configuration.
    annotation_prior_logits : Any
        Annotation prior logit matrix, or ``None``.
    effective_mixture_params : Any
        Resolved mixture_params (may be cleared on annotation
        downgrade).
    alr_reference_idx : int or None
        Resolved ALR reference gene index (LNM models only).
    _total_count_max : int
        Upper bound on per-cell total counts for predictive sampling.
    _gene_coverage_mask : Any
        Boolean mask over original gene space, or ``None``.
    _gene_coverage_rank : Any
        Per-gene coverage rank array, or ``None``.
    _excluded_gene_names : list of str or None
        Gene names pooled into the trailing "other" column.
    _filtered_gene_names : list of str or None
        Gene names retained after coverage filtering.
    _original_n_genes : int
        Gene count before coverage filtering.
    _adata_for_inference : AnnData or None
        AnnData reference passed to inference engines (``None`` when
        gene coverage created a pooled "other" pseudo-gene).
    _label_map : dict or None
        Annotation label-to-component mapping.
    _component_mapping : Any
        ``ComponentMapping`` object, or ``None``.
    effective_x64 : bool
        Whether to enable float64 precision for inference.
    results : Any
        Inference results object (set after ``_run_inference``).
    """

    # -- Inputs (set once by the orchestrator) --------------------------------
    counts: Any = None
    adata: Optional["AnnData"] = None
    count_data: Any = None
    n_cells: int = 0
    n_genes: int = 0
    data_config: Optional["DataConfig"] = None

    # -- Evolving state (written by stages) -----------------------------------
    model: str = "nbvcp"
    priors: Optional[Dict[str, Any]] = None
    n_components: Optional[int] = None
    n_datasets: Optional[int] = None
    dataset_indices: Any = None
    model_config: Optional["ModelConfig"] = None
    inference_config: Optional["InferenceConfig"] = None
    annotation_prior_logits: Any = None
    effective_mixture_params: Any = None
    alr_reference_idx: Optional[int] = None

    # -- Derived / cached metadata --------------------------------------------
    _total_count_max: int = 0
    _gene_coverage_mask: Any = None
    _gene_coverage_rank: Any = None
    _excluded_gene_names: Optional[List[str]] = None
    _filtered_gene_names: Optional[List[str]] = None
    _original_n_genes: int = 0
    _adata_for_inference: Optional["AnnData"] = None
    _label_map: Optional[Dict] = None
    _component_mapping: Any = None

    # -- Post-inference -------------------------------------------------------
    effective_x64: bool = False
    results: Any = None

    # -- Pass-through kwargs (original fit() arguments) -----------------------
    # These are stored here so stages can access them without requiring
    # the orchestrator to pass every kwarg individually.
    kwargs: Dict[str, Any] = field(default_factory=dict)

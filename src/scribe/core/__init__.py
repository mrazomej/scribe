"""
Core shared components for SCRIBE inference.

This module contains shared functionality used by both SVI and MCMC inference methods.
"""

from .input_processor import InputProcessor
from .normalization import normalize_counts_from_posterior
from .normalization_logistic import fit_logistic_normal_from_posterior
from .annotation_prior import (
    build_annotation_prior_logits,
    validate_annotation_prior_logits,
)
from .serialization import make_model_config_pickle_safe
from ._array_dispatch import (
    _array_module,
    _stats_norm,
    _special_module,
    _gpu_memory_budget,
    _vmap_chunk_size,
)
# AxisLayout helpers: ``build_sample_layouts`` / ``gene_axes_from_layouts`` support
# posterior-sample dicts and gene-axis maps alongside ``build_param_layouts``.
# ``merge_layouts`` and ``broadcast_param_to_layout`` are Phase B additions for
# layout-aware parameter broadcasting.
from .axis_layout import (
    AxisLayout,
    layout_from_param_spec,
    infer_layout,
    build_param_layouts,
    build_sample_layouts,
    gene_axes_from_layouts,
    reconstruct_param_layouts,
    align_to_layout,
    merge_layouts,
    broadcast_param_to_layout,
    derive_axis_membership,
)

# Differential expression module has moved to scribe.de (top-level)

# from .cell_type_assignment import (
#     compute_cell_type_probabilities,
#     compute_cell_type_probabilities_map,
#     temperature_scaling,
#     hellinger_distance_weights,
#     differential_expression_weights,
#     top_genes_mask,
# )

__all__ = [
    "InputProcessor",
    "normalize_counts_from_posterior",
    "fit_logistic_normal_from_posterior",
    "build_annotation_prior_logits",
    "validate_annotation_prior_logits",
    "make_model_config_pickle_safe",
    "_array_module",
    "_stats_norm",
    "_special_module",
    "_gpu_memory_budget",
    "_vmap_chunk_size",
    "AxisLayout",
    "layout_from_param_spec",
    "infer_layout",
    "build_param_layouts",
    "build_sample_layouts",
    "gene_axes_from_layouts",
    "reconstruct_param_layouts",
    "align_to_layout",
    "merge_layouts",
    "broadcast_param_to_layout",
    "derive_axis_membership",
    # "de" has moved to scribe.de (top-level)
    "compute_cell_type_probabilities",
    "compute_cell_type_probabilities_map",
    "temperature_scaling",
    "hellinger_distance_weights",
    "differential_expression_weights",
    "top_genes_mask",
]

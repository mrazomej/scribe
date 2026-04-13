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
from .axis_layout import (
    AxisLayout,
    layout_from_param_spec,
    infer_layout,
    build_param_layouts,
    reconstruct_param_layouts,
    align_to_layout,
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
    "AxisLayout",
    "layout_from_param_spec",
    "infer_layout",
    "build_param_layouts",
    "reconstruct_param_layouts",
    "align_to_layout",
    # "de" has moved to scribe.de (top-level)
    "compute_cell_type_probabilities",
    "compute_cell_type_probabilities_map",
    "temperature_scaling",
    "hellinger_distance_weights",
    "differential_expression_weights",
    "top_genes_mask",
]

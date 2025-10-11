"""
Core shared components for SCRIBE inference.

This module contains shared functionality used by both SVI and MCMC inference methods.
"""

from .input_processor import InputProcessor
from .normalization import normalize_counts_from_posterior
from .normalization_logistic import fit_logistic_normal_from_posterior

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
    "compute_cell_type_probabilities",
    "compute_cell_type_probabilities_map",
    "temperature_scaling",
    "hellinger_distance_weights",
    "differential_expression_weights",
    "top_genes_mask",
]

"""
viz_utils package – visualization and configuration utilities for SCRIBE.

Re-exports all public symbols for backward compatibility with
``from viz_utils import ...``.
"""

from .config import _get_config_values
from .dispatch import _get_predictive_samples_for_plot, _get_training_diagnostic_payload
from .cache import _build_umap_cache_path
from .loss import plot_loss
from .ecdf import plot_ecdf
from .ppc import plot_ppc
from .bio_ppc import plot_bio_ppc
from .umap import plot_umap
from .heatmap import plot_correlation_heatmap
from .mixture_ppc import plot_mixture_ppc, plot_mixture_composition
from .annotation_ppc import plot_annotation_ppc

__all__ = [
    "plot_loss",
    "plot_ecdf",
    "plot_ppc",
    "plot_bio_ppc",
    "plot_umap",
    "plot_correlation_heatmap",
    "plot_mixture_ppc",
    "plot_mixture_composition",
    "plot_annotation_ppc",
    "_get_config_values",
    "_get_predictive_samples_for_plot",
    "_get_training_diagnostic_payload",
    "_build_umap_cache_path",
]

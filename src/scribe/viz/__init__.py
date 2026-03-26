"""SCRIBE visualization package with lazy exports.

The package is imported during ``scribe`` initialization, so heavy visualization
submodules are loaded lazily to avoid circular imports.
"""

from __future__ import annotations

from .style import matplotlib_style, colors, Colors

_LAZY_EXPORTS = {
    "plot_posterior": ("scribe.viz.style", "plot_posterior"),
    "plot_histogram_credible_regions": (
        "scribe.viz.style",
        "plot_histogram_credible_regions",
    ),
    "plot_histogram_credible_regions_stairs": (
        "scribe.viz.style",
        "plot_histogram_credible_regions_stairs",
    ),
    "plot_ecdf_credible_regions_stairs": (
        "scribe.viz.style",
        "plot_ecdf_credible_regions_stairs",
    ),
    "_get_config_values": ("scribe.viz.config", "_get_config_values"),
    "_get_predictive_samples_for_plot": (
        "scribe.viz.dispatch",
        "_get_predictive_samples_for_plot",
    ),
    "_get_training_diagnostic_payload": (
        "scribe.viz.dispatch",
        "_get_training_diagnostic_payload",
    ),
    "_build_umap_cache_path": ("scribe.viz.cache", "_build_umap_cache_path"),
    "plot_loss": ("scribe.viz.loss", "plot_loss"),
    "plot_ecdf": ("scribe.viz.ecdf", "plot_ecdf"),
    "plot_ppc": ("scribe.viz.ppc", "plot_ppc"),
    "plot_bio_ppc": ("scribe.viz.bio_ppc", "plot_bio_ppc"),
    "plot_umap": ("scribe.viz.umap", "plot_umap"),
    "plot_correlation_heatmap": (
        "scribe.viz.heatmap",
        "plot_correlation_heatmap",
    ),
    "plot_mixture_ppc": ("scribe.viz.mixture_ppc", "plot_mixture_ppc"),
    "plot_mixture_composition": (
        "scribe.viz.mixture_ppc",
        "plot_mixture_composition",
    ),
    "plot_annotation_ppc": ("scribe.viz.annotation_ppc", "plot_annotation_ppc"),
    "plot_capture_anchor": ("scribe.viz.capture_anchor", "plot_capture_anchor"),
    "plot_p_capture_scaling": (
        "scribe.viz.capture_anchor",
        "plot_p_capture_scaling",
    ),
    "plot_mean_calibration": (
        "scribe.viz.mean_calibration",
        "plot_mean_calibration",
    ),
    "plot_mu_pairwise": ("scribe.viz.mu_pairwise", "plot_mu_pairwise"),
}


def __getattr__(name: str):
    """Resolve heavy visualization exports lazily on first access."""
    if name in _LAZY_EXPORTS:
        import importlib

        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'scribe.viz' has no attribute '{name}'")


__all__ = ["matplotlib_style", "colors", "Colors", *_LAZY_EXPORTS.keys()]

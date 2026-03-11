"""Shared PPC rendering helpers for performance-sensitive histogram plots.

This module centralizes logic that keeps PPC plots responsive for high-count
genes by:

- capping histogram bin ranges before credible-region computation
- auto-switching from stairs rendering to decimated line rendering when bin
  counts are very large
- optionally interpolating downsampled curves to preserve smooth shapes
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def get_ppc_render_options(viz_cfg):
    """Extract PPC rendering-performance options from visualization config.

    Parameters
    ----------
    viz_cfg : OmegaConf or mapping-like
        Visualization configuration object expected to contain ``ppc_opts``.
        Missing keys are filled with stable defaults.

    Returns
    -------
    dict
        Rendering and histogram-cap options with normalized numeric values:

        - ``hist_max_bin_quantile`` : float
            Quantile used to determine the upper x-axis / histogram bin cap.
        - ``hist_max_bin_floor`` : int
            Minimum bin cap to avoid over-truncating low-expression genes.
        - ``render_auto_line_bin_threshold`` : int
            Bin-count threshold above which line/fill rendering is used instead
            of stairs rendering.
        - ``render_line_target_points`` : int
            Target number of x-points used in line mode after decimation.
        - ``render_line_interpolate`` : bool
            Whether to interpolate to target x-points (``True``) or select
            nearest source points (``False``).
    """
    ppc_opts = viz_cfg.get("ppc_opts", {})
    return {
        "hist_max_bin_quantile": float(ppc_opts.get("hist_max_bin_quantile", 0.99)),
        "hist_max_bin_floor": int(ppc_opts.get("hist_max_bin_floor", 10)),
        "render_auto_line_bin_threshold": int(
            ppc_opts.get("render_auto_line_bin_threshold", 1000)
        ),
        "render_line_target_points": int(
            ppc_opts.get("render_line_target_points", 200)
        ),
        "render_line_interpolate": bool(
            ppc_opts.get("render_line_interpolate", True)
        ),
    }


def compute_adaptive_max_bin(observed_counts, render_opts):
    """Compute a robust upper histogram bin from observed counts.

    Parameters
    ----------
    observed_counts : array-like
        One-dimensional vector of observed counts for a single gene panel.
    render_opts : dict
        Dictionary returned by :func:`get_ppc_render_options`.

    Returns
    -------
    int
        Adaptive upper-bin cap based on the configured quantile and floor.
        The result is always at least ``hist_max_bin_floor``.
    """
    quantile = float(render_opts["hist_max_bin_quantile"])
    floor = int(render_opts["hist_max_bin_floor"])
    q_value = int(np.ceil(np.quantile(np.asarray(observed_counts), quantile)))
    return max(q_value, floor)


def should_use_line_mode(n_bins, render_opts):
    """Return whether PPC bands should use line/fill instead of stairs.

    Parameters
    ----------
    n_bins : int
        Effective number of bins being plotted for a panel.
    render_opts : dict
        Dictionary returned by :func:`get_ppc_render_options`.

    Returns
    -------
    bool
        ``True`` when ``n_bins`` exceeds the configured threshold.
    """
    return int(n_bins) > int(render_opts["render_auto_line_bin_threshold"])


def _resample_y(x_source, y_source, x_target, interpolate):
    """Resample y-values from source x-grid onto target x-grid."""
    if x_source.size == 0 or y_source.size == 0:
        return np.zeros_like(x_target, dtype=float)

    y_source = np.asarray(y_source, dtype=float)
    if bool(interpolate):
        return np.interp(x_target, x_source, y_source)

    nearest_idx = np.searchsorted(x_source, x_target, side="left")
    nearest_idx = np.clip(nearest_idx, 0, x_source.size - 1)
    return y_source[nearest_idx]


def _build_target_x(x_source, target_points, interpolate):
    """Construct a compact target x-grid for large-bin plotting."""
    n_bins = x_source.size
    if n_bins <= target_points:
        return x_source

    if bool(interpolate):
        return np.linspace(x_source[0], x_source[-1], target_points)

    idx = np.linspace(0, n_bins - 1, target_points, dtype=int)
    return x_source[idx]


def plot_histogram_credible_regions_adaptive(
    ax,
    hist_results,
    *,
    cmap,
    alpha,
    max_bin,
    render_opts,
):
    """Plot histogram credible regions with automatic stairs/line fallback.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis receiving the credible-region bands.
    hist_results : dict
        Output from ``compute_histogram_credible_regions``.
    cmap : str
        Matplotlib colormap name used for region fills.
    alpha : float
        Fill alpha used for region bands.
    max_bin : int
        Maximum bin index to include in plotting.
    render_opts : dict
        Dictionary returned by :func:`get_ppc_render_options`.

    Returns
    -------
    dict
        Rendering metadata with:

        - ``mode`` : ``"stairs"`` or ``"line"``
        - ``x_target`` : target x-grid used in line mode, else ``None``
        - ``line_interpolate`` : interpolation flag used for line mode
    """
    import scribe

    n_bins_available = int(len(hist_results["bin_edges"]) - 1)
    n_plot_bins = min(int(max_bin), n_bins_available)

    if not should_use_line_mode(n_plot_bins, render_opts):
        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, hist_results, cmap=cmap, alpha=alpha, max_bin=max_bin
        )
        return {"mode": "stairs", "x_target": None, "line_interpolate": None}

    x_source = np.asarray(hist_results["bin_edges"][:-1], dtype=float)[:n_plot_bins]
    target_points = max(2, int(render_opts["render_line_target_points"]))
    line_interpolate = bool(render_opts["render_line_interpolate"])
    x_target = _build_target_x(x_source, target_points, line_interpolate)

    cr_values = sorted(hist_results["regions"].keys(), reverse=True)
    n_regions = len(cr_values)
    cmap_obj = plt.get_cmap(cmap)
    if n_regions == 1:
        colors = [cmap_obj(0.5)]
    else:
        colors = [cmap_obj(i / (n_regions - 1)) for i in range(n_regions)][::-1]

    for cr, color in zip(cr_values, colors):
        region = hist_results["regions"][cr]
        lower = np.asarray(region["lower"], dtype=float)[:n_plot_bins]
        upper = np.asarray(region["upper"], dtype=float)[:n_plot_bins]
        lower_ds = _resample_y(x_source, lower, x_target, line_interpolate)
        upper_ds = _resample_y(x_source, upper, x_target, line_interpolate)
        ax.fill_between(x_target, lower_ds, upper_ds, color=color, alpha=alpha)

    median = np.asarray(hist_results["regions"][cr_values[0]]["median"], dtype=float)[
        :n_plot_bins
    ]
    median_ds = _resample_y(x_source, median, x_target, line_interpolate)
    ax.plot(x_target, median_ds, color="black", alpha=0.2, linewidth=1.5)

    return {
        "mode": "line",
        "x_target": x_target,
        "line_interpolate": line_interpolate,
    }


def plot_observed_histogram_adaptive(
    ax,
    hist_results,
    *,
    max_bin,
    render_meta,
    color="black",
    label="data",
    linewidth=1.0,
):
    """Plot observed histogram in a style consistent with PPC band rendering.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis receiving the observed-data overlay.
    hist_results : tuple
        Tuple returned by ``numpy.histogram``.
    max_bin : int
        Maximum number of bins to display.
    render_meta : dict
        Metadata returned by :func:`plot_histogram_credible_regions_adaptive`.
    color : str, optional
        Overlay color.
    label : str, optional
        Legend label for the overlay.
    linewidth : float, optional
        Line width for line-mode plotting.
    """
    n_hist_bins = len(hist_results[0])
    n_plot_bins = min(int(max_bin), n_hist_bins)

    if render_meta["mode"] == "stairs":
        ax.step(
            hist_results[1][:n_plot_bins],
            hist_results[0][:n_plot_bins],
            where="post",
            label=label,
            color=color,
            linewidth=linewidth,
        )
        return

    x_source = np.asarray(hist_results[1][:-1], dtype=float)[:n_plot_bins]
    y_source = np.asarray(hist_results[0], dtype=float)[:n_plot_bins]
    x_target = np.asarray(render_meta["x_target"], dtype=float)
    line_interpolate = bool(render_meta["line_interpolate"])
    y_target = _resample_y(x_source, y_source, x_target, line_interpolate)
    ax.plot(x_target, y_target, color=color, linewidth=linewidth, label=label)

"""Dataset-level pairwise mean (mu) comparison diagnostic.

This module provides a corner-style plot for hierarchical multi-dataset
models. The plot uses MAP-estimated gene means (``mu``) and compares
datasets pairwise:

- diagonal panels: marginal distributions for each dataset's gene means
- lower-triangle panels: pairwise comparisons in log-log space
- identity line on each pairwise panel for visual calibration

The diagnostic is intentionally skipped for single-dataset runs.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from ._common import console
from ._interactive import (
    _create_or_validate_grid_axes,
    plot_function,
)
from .dispatch import _get_layouts_for_plot, _get_map_estimates_for_plot


def _get_dataset_count(results, mu_values, layouts=None):
    """Infer the number of datasets from result metadata and ``mu`` shape.

    Parameters
    ----------
    results : object
        Fitted results object that may expose ``model_config.n_datasets``.
    mu_values : ndarray
        MAP estimate for ``mu``.
    layouts : dict of str to AxisLayout or None
        When provided, ``layouts["mu"].dataset_axis`` is checked before
        falling back to shape-based inference.

    Returns
    -------
    int
        Inferred number of datasets. Returns ``1`` when no multi-dataset
        signal is available.
    """
    # Prefer explicit model metadata.
    model_cfg = getattr(results, "model_config", None)
    n_datasets = getattr(model_cfg, "n_datasets", None)
    if n_datasets is not None:
        return int(n_datasets)

    # Use layout metadata to determine dataset count.
    mu_arr = np.asarray(mu_values)
    ds_ax = layouts["mu"].dataset_axis
    if ds_ax is not None:
        return int(mu_arr.shape[ds_ax])
    return 1


def _collapse_mixture_axis(mu_values, mixing_weights, layouts=None):
    """Collapse mixture-specific ``mu`` into dataset-level ``mu``.

    Parameters
    ----------
    mu_values : ndarray
        MAP estimate for mean parameter ``mu`` with shape one of:
        ``(D, G)``, ``(K, D, G)``, ``(K, G)``, or ``(G,)``.
    mixing_weights : ndarray or None
        Mixture weights if available. Supported shapes are ``(K,)`` and
        ``(K, D)``.
    layouts : dict of str to AxisLayout or None
        When provided, ``layouts["mu"].component_axis`` determines
        whether there is a mixture dimension to collapse.

    Returns
    -------
    ndarray
        Dataset-level ``mu`` matrix of shape ``(D, G)`` when multi-dataset,
        or ``(1, G)`` for single-dataset inputs.
    """
    mu_arr = np.asarray(mu_values, dtype=float)

    # Read the component axis from layout metadata.
    _comp_ax = layouts["mu"].component_axis

    if _comp_ax is not None:
        # Mixture model: weighted-average (or simple average) over component axis.
        if mixing_weights is None:
            return np.mean(mu_arr, axis=_comp_ax)
        w = np.asarray(mixing_weights, dtype=float)
        # Expand weight dims so they broadcast with mu along the
        # component axis.  Other axes get singleton dimensions.
        shape = [1] * mu_arr.ndim
        shape[_comp_ax] = w.shape[0]
        if w.ndim == 1:
            w = w.reshape(shape)
        elif w.ndim == 2:
            w = w.reshape(shape[:_comp_ax] + [w.shape[0]] + [w.shape[1]] + [1])
        return np.sum(w * mu_arr, axis=_comp_ax)

    # No component axis: ensure output is at least 2-D for downstream plotting.
    if mu_arr.ndim == 1:
        return mu_arr[None, :]
    return mu_arr


def _resolve_dataset_names(dataset_names, n_datasets):
    """Resolve dataset labels with stable defaults.

    Parameters
    ----------
    dataset_names : sequence of str or None
        Dataset names passed from the visualization pipeline.
    n_datasets : int
        Number of datasets to label.

    Returns
    -------
    list of str
        Dataset labels with length equal to ``n_datasets``.
    """
    if dataset_names is None:
        return [f"dataset_{idx}" for idx in range(n_datasets)]

    names = [str(name) for name in dataset_names]
    if len(names) < n_datasets:
        names.extend(
            [f"dataset_{idx}" for idx in range(len(names), n_datasets)]
        )
    return names[:n_datasets]


# Suffix ``mean_pairwise`` avoids clashing with ``mean_calibration`` outputs
# and matches ``--mean-pairwise`` (we do not use ``mu`` in the filename).
@plot_function(
    suffix="mean_pairwise",
    save_label="mean pairwise plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_mu_pairwise(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    dataset_names=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Render pairwise dataset ``mu`` comparisons as a corner plot.

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Fitted model results object.
    counts : array-like
        Observed count matrix. This is forwarded to MAP extraction to support
        models that require counts-aware MAP reconstruction.
    figs_dir : str
        Output figure directory.
    cfg : OmegaConf
        Run configuration loaded from ``.hydra/config.yaml``.
    viz_cfg : OmegaConf
        Visualization configuration.
    dataset_names : sequence of str, optional
        Optional names for dataset indices.

    Returns
    -------
    PlotResult or None
        Wrapped result, or ``None`` when the plot is skipped.
    """
    console.print("[dim]Plotting pairwise mu dataset comparison...[/dim]")
    if ax is not None:
        raise ValueError(
            "Mu pairwise is a multi-panel corner plot; provide `fig` or `axes`."
        )
    map_estimates = _get_map_estimates_for_plot(
        results, counts=counts, targets=["mu"]
    )
    mu_values = map_estimates.get("mu")
    if mu_values is None:
        console.print(
            "[yellow]Skipping mu pairwise plot: mu unavailable in MAP "
            "estimates.[/yellow]"
        )
        return None

    # Fetch layout metadata for axis-driven lookups.
    _layouts = _get_layouts_for_plot(results)

    inferred_n_datasets = _get_dataset_count(
        results, mu_values, layouts=_layouts,
    )
    if inferred_n_datasets <= 1:
        console.print(
            "[yellow]Skipping mu pairwise plot: run is not multi-dataset."
            "[/yellow]"
        )
        return None

    # Collapse optional mixture axis so we always plot one mu vector per dataset.
    mixing_weights = None
    try:
        mix_map = _get_map_estimates_for_plot(
            results, counts=counts, targets=["mixing_weights"]
        )
        mixing_weights = mix_map.get("mixing_weights")
    except ValueError:
        mixing_weights = None
    mu_dataset = _collapse_mixture_axis(
        mu_values=mu_values,
        mixing_weights=mixing_weights,
        layouts=_layouts,
    )
    if mu_dataset.ndim != 2:
        mu_dataset = np.asarray(mu_dataset, dtype=float).reshape(
            inferred_n_datasets, -1
        )

    n_datasets = min(inferred_n_datasets, mu_dataset.shape[0])
    mu_dataset = mu_dataset[:n_datasets]
    labels = _resolve_dataset_names(dataset_names, n_datasets)

    _mu_opts = viz_cfg.get("mu_pairwise_opts", {}) if viz_cfg is not None else {}
    pseudocount = float(_mu_opts.get("pseudocount", 1.0))
    n_bins = int(_mu_opts.get("hist_bins", 40))
    point_alpha = float(_mu_opts.get("point_alpha", 0.25))
    point_size = float(_mu_opts.get("point_size", 5.0))

    mu_log = np.log10(np.clip(mu_dataset, a_min=0.0, a_max=None) + pseudocount)
    # Pre-compute per-dataset axis limits so all panels in a column/row align.
    # This keeps the corner grid visually tight and avoids panel drift caused by
    # per-panel autoscaling.
    axis_limits = []
    for dataset_idx in range(n_datasets):
        values = mu_log[dataset_idx]
        lo = float(np.min(values))
        hi = float(np.max(values))
        margin = (hi - lo) * 0.05
        if margin <= 0:
            margin = 0.5
        axis_limits.append((lo - margin, hi + margin))

    fig, axes_grid, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_datasets,
        n_cols=n_datasets,
        fig=fig,
        axes=axes,
        figsize=figsize or (2.8 * n_datasets, 2.8 * n_datasets),
    )

    # Populate a corner-style layout:
    # - diagonal: marginal histograms
    # - lower triangle: pairwise scatter + identity line
    # - upper triangle: hidden
    for row_idx in range(n_datasets):
        for col_idx in range(n_datasets):
            axis = axes_grid[row_idx, col_idx]
            x_values = mu_log[col_idx]
            y_values = mu_log[row_idx]

            if row_idx == col_idx:
                axis.hist(
                    x_values,
                    bins=n_bins,
                    color="steelblue",
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.4,
                )
                axis.set_xlim(*axis_limits[col_idx])
                axis.set_title(
                    rf"{labels[row_idx]}  ($\log_{{10}}(\mu + {pseudocount:g})$)",
                    fontsize=9,
                )
                axis.set_yticks([])
            elif row_idx > col_idx:
                axis.scatter(
                    x_values,
                    y_values,
                    s=point_size,
                    alpha=point_alpha,
                    color="royalblue",
                    edgecolors="none",
                    rasterized=True,
                )
                x_lo, x_hi = axis_limits[col_idx]
                y_lo, y_hi = axis_limits[row_idx]
                axis.set_xlim(x_lo, x_hi)
                axis.set_ylim(y_lo, y_hi)
                # Identity line is drawn in the plotted coordinate system so
                # each pairwise panel is directly comparable.
                line_lo = min(x_lo, y_lo)
                line_hi = max(x_hi, y_hi)
                axis.plot(
                    [line_lo, line_hi],
                    [line_lo, line_hi],
                    "--",
                    color="0.3",
                    lw=1.0,
                )
            else:
                axis.axis("off")
                continue

            # Only the bottom row exposes x-axis ticks/labels.
            if row_idx == n_datasets - 1:
                axis.set_xlabel(
                    rf"{labels[col_idx]}  ($\log_{{10}}(\mu + {pseudocount:g})$)",
                    fontsize=8,
                )
            else:
                axis.set_xticklabels([])
                axis.tick_params(axis="x", which="both", bottom=False)

            # Only the first column of lower-triangle panels exposes y labels.
            if col_idx == 0 and row_idx > col_idx:
                axis.set_ylabel(
                    rf"{labels[row_idx]}  ($\log_{{10}}(\mu + {pseudocount:g})$)",
                    fontsize=8,
                )
            else:
                axis.set_yticklabels([])
                axis.tick_params(axis="y", which="both", left=False)

            # Diagonal panels should not inherit x ticks unless they are in the
            # bottom row; this avoids tick clutter above pairwise scatter.
            if row_idx == col_idx and row_idx != n_datasets - 1:
                axis.set_xticklabels([])
                axis.tick_params(axis="x", which="both", bottom=False)

    fig.suptitle(
        r"Dataset Pairwise Mean Comparison ($\log_{10}(\mu + c)$)",
        fontsize=11,
        y=1.01,
    )
    # Keep panel spacing very tight so the corner layout reads as a single
    # matrix rather than isolated subplots.
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.08,
        top=0.92,
        wspace=0.03,
        hspace=0.03,
    )

    return fig, axes_flat, n_datasets * n_datasets

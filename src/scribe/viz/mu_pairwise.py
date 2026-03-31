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
from .dispatch import _get_map_estimates_for_plot


def _get_dataset_count(results, mu_values):
    """Infer the number of datasets from result metadata and ``mu`` shape.

    Parameters
    ----------
    results : object
        Fitted results object that may expose ``model_config.n_datasets``.
    mu_values : ndarray
        MAP estimate for ``mu``.

    Returns
    -------
    int
        Inferred number of datasets. Returns ``1`` when no multi-dataset
        signal is available.
    """
    # Prefer explicit model metadata because it is the most reliable source.
    model_cfg = getattr(results, "model_config", None)
    n_datasets = getattr(model_cfg, "n_datasets", None)
    if n_datasets is not None:
        return int(n_datasets)

    # Fall back to shape-based inference for ad hoc test stubs.
    mu_arr = np.asarray(mu_values)
    if mu_arr.ndim >= 2:
        return int(mu_arr.shape[0])
    return 1


def _collapse_mixture_axis(mu_values, mixing_weights):
    """Collapse mixture-specific ``mu`` into dataset-level ``mu``.

    Parameters
    ----------
    mu_values : ndarray
        MAP estimate for mean parameter ``mu`` with shape one of:
        ``(D, G)``, ``(K, D, G)``, ``(K, G)``, or ``(G,)``.
    mixing_weights : ndarray or None
        Mixture weights if available. Supported shapes are ``(K,)`` and
        ``(K, D)``.

    Returns
    -------
    ndarray
        Dataset-level ``mu`` matrix of shape ``(D, G)`` when multi-dataset,
        or ``(1, G)`` for single-dataset inputs.
    """
    mu_arr = np.asarray(mu_values, dtype=float)
    if mu_arr.ndim == 1:
        return mu_arr[None, :]
    if mu_arr.ndim == 2:
        return mu_arr

    # For (K, D, G) or similar high-rank structures, use provided mixture
    # weights when available and otherwise average across component axis.
    if mu_arr.ndim >= 3:
        if mixing_weights is None:
            return np.mean(mu_arr, axis=0)

        w = np.asarray(mixing_weights, dtype=float)
        if w.ndim == 1 and w.shape[0] == mu_arr.shape[0]:
            w = w[:, None, None]
            return np.sum(w * mu_arr, axis=0)
        if w.ndim == 2 and w.shape == mu_arr.shape[:2]:
            w = w[:, :, None]
            return np.sum(w * mu_arr, axis=0)
        return np.mean(mu_arr, axis=0)

    # Degenerate fallback keeps output 2D to simplify downstream plotting.
    return mu_arr.reshape(1, -1)


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


@plot_function(
    suffix="mu_pairwise",
    save_label="mu pairwise plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_mu_pairwise(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    dataset_names=None,
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

    inferred_n_datasets = _get_dataset_count(results, mu_values)
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
        figsize=(2.8 * n_datasets, 2.8 * n_datasets),
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

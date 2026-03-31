"""Posterior predictive check plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import scribe

from ._common import (
    console,
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from ._interactive import (
    _create_or_validate_grid_axes,
    _finalize_figure,
    _resolve_render_flags,
)
from .config import _get_config_values
from .dispatch import _get_predictive_samples_for_plot
from .gene_selection import _select_genes
from .ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    plot_histogram_credible_regions_adaptive,
    plot_observed_histogram_adaptive,
)


def plot_ppc(
    results,
    counts,
    figs_dir=None,
    cfg=None,
    viz_cfg=None,
    *,
    fig=None,
    axes=None,
    ax=None,
    save=None,
    show=None,
    close=None,
):
    """Plot posterior predictive checks for selected genes.

    Parameters
    ----------
    results : object
        Fitted result object exposing predictive sampling.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    figs_dir : str, optional
        Output directory used when ``save`` resolves to ``True``.
    cfg : OmegaConf, optional
        Run configuration used for filename generation.
    viz_cfg : OmegaConf
        Visualization config containing ``ppc_opts``.
    fig : matplotlib.figure.Figure, optional
        Figure used to create/host the PPC grid.
    axes : array-like of matplotlib.axes.Axes, optional
        Explicit axis collection with exactly ``n_rows * n_cols`` axes.
    ax : matplotlib.axes.Axes, optional
        Unsupported for this multi-panel plot. Use ``fig`` or ``axes``.
    save, show, close : bool, optional
        Rendering controls for dual-mode usage.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.
    """
    _fig_owned = fig is None and axes is None
    console.print("[dim]Plotting PPC...[/dim]")
    if ax is not None:
        raise ValueError(
            "PPC requires multiple axes; provide `fig` or `axes` instead of `ax`."
        )
    save, show, close = _resolve_render_flags(
        figs_dir=figs_dir,
        save=save,
        show=show,
        close=close,
    )
    render_opts = get_ppc_render_options(viz_cfg)

    n_rows = viz_cfg.ppc_opts.n_rows
    n_cols = viz_cfg.ppc_opts.n_cols
    console.print(
        f"[dim]Using n_rows={n_rows}, n_cols={n_cols} for PPC plot (log-spaced binning)[/dim]"
    )
    selected_idx, mean_counts = _select_genes(counts, n_rows, n_cols)

    selected_means = mean_counts[selected_idx]
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]
    n_genes_selected = len(selected_idx_sorted)
    console.print(
        f"[dim]Selected {n_genes_selected} genes across {n_rows} expression bins[/dim]"
    )

    results_subset = results[selected_idx]

    n_samples = viz_cfg.ppc_opts.n_samples
    console.print(
        f"[dim]Generating {n_samples} posterior predictive samples...[/dim]"
    )
    _ = _get_predictive_samples_for_plot(
        results_subset,
        rng_key=random.PRNGKey(42),
        n_samples=n_samples,
        counts=counts,
        batch_size=None,
        store_samples=True,
    )

    # results[selected_idx] now preserves the caller-specified gene order, so
    # subset_positions must map each gene's original index to its position in
    # selected_idx (not in the old sorted-original order).
    subset_positions = {
        int(gene_idx): pos for pos, gene_idx in enumerate(selected_idx)
    }

    fig, _, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_rows,
        n_cols=n_cols,
        fig=fig,
        axes=axes,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Plotting PPC panels...", total=n_genes_selected
        )

        for i, panel_ax in enumerate(axes_flat):
            if i >= n_genes_selected:
                panel_ax.axis("off")
                continue

            gene_idx = selected_idx_sorted[i]
            subset_pos = subset_positions[gene_idx]
            true_counts = counts[:, gene_idx]
            max_bin = compute_adaptive_max_bin(true_counts, render_opts)

            credible_regions = scribe.stats.compute_histogram_credible_regions(
                results_subset.predictive_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
                max_bin=max_bin,
            )

            hist_results = np.histogram(
                true_counts, bins=credible_regions["bin_edges"], density=True
            )
            # Plot style is selected adaptively: stairs for moderate bin counts
            # and line/fill for very large bin counts.
            render_meta = plot_histogram_credible_regions_adaptive(
                panel_ax,
                credible_regions,
                cmap="Blues",
                alpha=0.5,
                max_bin=max_bin,
                render_opts=render_opts,
            )
            plot_observed_histogram_adaptive(
                panel_ax,
                hist_results,
                max_bin=max_bin,
                render_meta=render_meta,
                label="data",
                color="black",
            )

            panel_ax.set_xlabel("counts")
            panel_ax.set_ylabel("frequency")
            actual_mean_expr = np.mean(counts[:, gene_idx])
            mean_expr_formatted = f"{actual_mean_expr:.2f}"
            panel_ax.set_title(
                f"$\\langle U \\rangle = {mean_expr_formatted}$",
                fontsize=8,
            )

            progress.update(task, advance=1)

    fig.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    if save:
        output_format = viz_cfg.get("format", "png")
        config_vals = _get_config_values(cfg, results=results)
        fname = (
            f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
            f"{config_vals['model_type'].replace('_', '-')}_"
            f"{config_vals['n_components']:02d}components_"
            f"{config_vals['run_size_token']}_ppc.{output_format}"
        )
    else:
        fname = None

    del results_subset
    return _finalize_figure(
        fig=fig,
        axes=axes_flat,
        n_panels=n_rows * n_cols,
        save=save,
        show=show,
        close=close,
        figs_dir=figs_dir,
        filename=fname,
        save_kwargs={"bbox_inches": "tight"},
        save_label="PPC plot",
        _fig_owned=_fig_owned,
    )

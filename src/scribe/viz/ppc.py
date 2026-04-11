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
    _resolve_ppc_grid,
    plot_function,
)
from .dispatch import _get_predictive_samples_for_plot
from .gene_selection import _coerce_counts, _get_gene_names, _select_genes
from .ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    plot_histogram_credible_regions_adaptive,
    plot_observed_histogram_adaptive,
)


def _prepare_ppc_data(results, counts, viz_cfg, *, n_rows, n_cols, n_samples):
    """Prepare gene selection and predictive samples for PPC plotting.

    Selects genes across expression bins, generates posterior
    predictive samples, and builds the gene-position mapping needed
    for histogram rendering.

    Parameters
    ----------
    results : object
        Fitted model results exposing predictive sampling.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    viz_cfg : OmegaConf or None
        Visualization configuration (used only for render options).
    n_rows : int
        Number of grid rows (already resolved).
    n_cols : int
        Number of grid columns (already resolved).
    n_samples : int
        Number of posterior predictive samples (already resolved).

    Returns
    -------
    dict
        Dictionary with keys ``n_rows``, ``n_cols``,
        ``selected_idx_sorted``, ``subset_positions``,
        ``n_genes_selected``, ``render_opts``, ``results_subset``.
    """
    render_opts = get_ppc_render_options(viz_cfg)

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

    console.print(
        f"[dim]Generating {n_samples} posterior predictive samples...[/dim]"
    )
    _ = _get_predictive_samples_for_plot(
        results_subset,
        rng_key=random.PRNGKey(42),
        n_samples=n_samples,
        counts=counts,
        store_samples=True,
    )

    # results[selected_idx] preserves the caller-specified gene order, so
    # subset_positions maps each gene's original index to its position in
    # selected_idx (not the old sorted-original order).
    subset_positions = {
        int(gene_idx): pos for pos, gene_idx in enumerate(selected_idx)
    }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "selected_idx_sorted": selected_idx_sorted,
        "subset_positions": subset_positions,
        "n_genes_selected": n_genes_selected,
        "render_opts": render_opts,
        "results_subset": results_subset,
    }


@plot_function(
    suffix="ppc",
    save_label="PPC plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_ppc(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    n_rows=None,
    n_cols=None,
    n_genes=None,
    n_samples=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
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
    viz_cfg : OmegaConf or None
        Visualization config containing ``ppc_opts``.  Optional in
        interactive sessions — built-in defaults are used when ``None``.
    n_rows : int, optional
        Number of grid rows.  Overrides ``viz_cfg.ppc_opts.n_rows``.
    n_cols : int, optional
        Number of grid columns.  Overrides ``viz_cfg.ppc_opts.n_cols``.
    n_genes : int, optional
        Total number of genes to display.  When given without ``n_cols``,
        derives ``n_cols = ceil(n_genes / n_rows)``.
    n_samples : int, optional
        Number of posterior predictive samples.  Overrides
        ``viz_cfg.ppc_opts.n_samples``.
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
    console.print("[dim]Plotting PPC...[/dim]")
    counts = _coerce_counts(counts)
    if ax is not None:
        raise ValueError(
            "PPC requires multiple axes; provide `fig` or `axes` instead of `ax`."
        )
    # Resolve grid dimensions: explicit kwargs > viz_cfg > defaults
    grid = _resolve_ppc_grid(
        n_rows=n_rows, n_cols=n_cols, n_genes=n_genes,
        n_samples=n_samples, viz_cfg=viz_cfg,
    )
    prep = _prepare_ppc_data(
        results, counts, viz_cfg,
        n_rows=grid["n_rows"], n_cols=grid["n_cols"],
        n_samples=grid["n_samples"],
    )
    n_rows = prep["n_rows"]
    n_cols = prep["n_cols"]
    selected_idx_sorted = prep["selected_idx_sorted"]
    subset_positions = prep["subset_positions"]
    n_genes_selected = prep["n_genes_selected"]
    render_opts = prep["render_opts"]
    results_subset = prep["results_subset"]
    gene_names = _get_gene_names(results)

    fig, _, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_rows,
        n_cols=n_cols,
        fig=fig,
        axes=axes,
        figsize=figsize or (2.5 * n_cols, 2.5 * n_rows),
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
            title = f"$\\langle U \\rangle = {mean_expr_formatted}$"
            if gene_names is not None:
                title = f"{gene_names[gene_idx]}\n{title}"
            panel_ax.set_title(title, fontsize=8)

            progress.update(task, advance=1)

    fig.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    del results_subset
    return fig, axes_flat, n_rows * n_cols

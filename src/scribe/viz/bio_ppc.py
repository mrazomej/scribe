"""Biological posterior predictive check plotting.

Mirrors :func:`plot_ppc` but uses:
- **Bands**: biological PPC samples from NB(r, p) only (no capture prob / gate)
- **Data line**: MAP-denoised observed counts instead of raw counts
"""

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
from .dispatch import (
    _get_biological_ppc_samples_for_plot,
    _get_denoised_counts_for_plot,
)
from .gene_selection import _coerce_counts, _get_gene_names, _select_genes
from .ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    plot_histogram_credible_regions_adaptive,
    plot_observed_histogram_adaptive,
)


@plot_function(
    suffix="bio_ppc",
    save_label="bio-PPC plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_bio_ppc(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    n_rows=None,
    n_cols=None,
    n_genes=None,
    n_samples=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot biological posterior predictive checks with denoised data.

    Uses the same gene-selection logic as :func:`plot_ppc` (log-spaced
    expression bins) but replaces the standard predictive samples with
    biological NB(r, p) samples and shows the denoised observed counts
    as the data histogram.

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Fitted model results.
    counts : array-like
        Observed UMI count matrix ``(n_cells, n_genes)``.
    figs_dir : str, optional
        Directory to save the output figure.
    cfg : OmegaConf, optional
        Hydra run configuration.
    viz_cfg : OmegaConf or None
        Visualization configuration.  Optional in interactive sessions —
        built-in defaults are used when ``None``.
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

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.
    """
    if ax is not None:
        raise ValueError(
            "Biological PPC requires multiple axes; provide `fig` or `axes`."
        )
    console.print("[dim]Plotting biological PPC (denoised)...[/dim]")
    counts = _coerce_counts(counts)
    render_opts = get_ppc_render_options(viz_cfg)

    # Resolve grid dimensions: explicit kwargs > viz_cfg > defaults
    grid = _resolve_ppc_grid(
        n_rows=n_rows, n_cols=n_cols, n_genes=n_genes,
        n_samples=n_samples, viz_cfg=viz_cfg,
    )
    n_rows = grid["n_rows"]
    n_cols = grid["n_cols"]
    n_samples = grid["n_samples"]

    # ------------------------------------------------------------------
    # Gene selection (identical to standard PPC)
    # ------------------------------------------------------------------
    console.print(
        f"[dim]Using n_rows={n_rows}, n_cols={n_cols} "
        "for bio-PPC plot (log-spaced binning)[/dim]"
    )
    selected_idx, mean_counts = _select_genes(counts, n_rows, n_cols)

    selected_means = mean_counts[selected_idx]
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]
    n_genes_selected = len(selected_idx_sorted)
    console.print(
        f"[dim]Selected {n_genes_selected} genes across "
        f"{n_rows} expression bins[/dim]"
    )

    gene_names = _get_gene_names(results)

    # ------------------------------------------------------------------
    # Biological PPC samples (gene-subset)
    # ------------------------------------------------------------------
    # Keep subset ordering aligned with plotting traversal and denoising logic.
    results_subset = results[selected_idx_sorted]

    rng_key = random.PRNGKey(42)

    console.print(
        f"[dim]Generating {n_samples} biological PPC samples...[/dim]"
    )
    bio_predictive_samples = _get_biological_ppc_samples_for_plot(
        results_subset,
        rng_key=rng_key,
        n_samples=n_samples,
        counts=counts,
        batch_size=None,
        store_samples=False,
    )

    # ------------------------------------------------------------------
    # Denoised observed counts (only selected genes)
    # ------------------------------------------------------------------
    # Use ("sample", "sample") so the denoised histogram has realistic
    # stochastic variability comparable to the biological PPC bands.
    # ("mean", "sample") would create artificial density spikes because
    # every cell with the same observed count maps to a single value.
    console.print("[dim]Denoising observed counts (MAP, sample)...[/dim]")
    key_denoise = random.PRNGKey(99)
    # Denoising only selected genes significantly reduces peak device memory.
    _bio_opts = viz_cfg.get("bio_ppc_opts", {}) if viz_cfg is not None else {}
    denoise_cell_batch_size = int(
        _bio_opts.get("denoise_cell_batch_size", 256) if hasattr(_bio_opts, "get") else 256
    )
    # Keep denoising order aligned with ``results_subset`` and plotting.
    counts_subset = counts[:, selected_idx_sorted]
    denoised_subset = _get_denoised_counts_for_plot(
        results_subset,
        counts=counts_subset,
        rng_key=key_denoise,
        method=("sample", "sample"),
        cell_batch_size=denoise_cell_batch_size,
    )

    # Build a position map: original gene index → position inside the
    # gene-subset that was passed to the biological PPC sampler.
    subset_positions = {
        int(gene_idx): pos for pos, gene_idx in enumerate(selected_idx_sorted)
    }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
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
            "[cyan]Plotting bio-PPC panels...", total=n_genes_selected
        )

        for i, panel_ax in enumerate(axes_flat):
            if i >= n_genes_selected:
                panel_ax.axis("off")
                continue

            gene_idx = selected_idx_sorted[i]
            subset_pos = subset_positions[gene_idx]
            denoised_gene = denoised_subset[:, subset_pos]
            max_bin = compute_adaptive_max_bin(denoised_gene, render_opts)

            # Credible bands from biological PPC
            credible_regions = scribe.stats.compute_histogram_credible_regions(
                bio_predictive_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
                max_bin=max_bin,
            )

            # Denoised data histogram
            hist_results = np.histogram(
                denoised_gene,
                bins=credible_regions["bin_edges"],
                density=True,
            )
            # Bio PPC credible-region bands use the same adaptive style as
            # standard PPC so very large count ranges remain fast to render.
            render_meta = plot_histogram_credible_regions_adaptive(
                panel_ax,
                credible_regions,
                cmap="Greens",
                alpha=0.5,
                max_bin=max_bin,
                render_opts=render_opts,
            )
            plot_observed_histogram_adaptive(
                panel_ax,
                hist_results,
                max_bin=max_bin,
                render_meta=render_meta,
                label="denoised",
                color="black",
            )

            panel_ax.set_xlabel("counts")
            panel_ax.set_ylabel("frequency")
            denoised_mean = np.mean(denoised_gene)
            title = rf"$\langle \hat{{X}} \rangle = {denoised_mean:.2f}$"
            if gene_names is not None:
                title = f"{gene_names[gene_idx]}\n{title}"
            panel_ax.set_title(title, fontsize=8)

            progress.update(task, advance=1)

    fig.tight_layout()
    fig.suptitle("Biological PPC (denoised)", y=1.02)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    del results_subset, counts_subset, denoised_subset, bio_predictive_samples
    return fig, axes_flat, n_rows * n_cols

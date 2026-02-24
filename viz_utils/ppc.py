"""Posterior predictive check plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import scribe

from ._common import console, Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from .config import _get_config_values
from .dispatch import _get_predictive_samples_for_plot
from .gene_selection import _select_genes


def plot_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """Plot and save the posterior predictive checks."""
    console.print("[dim]Plotting PPC...[/dim]")

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

    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = axes.flatten()

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

        for i, ax in enumerate(axes):
            if i >= n_genes_selected:
                ax.axis("off")
                continue

            gene_idx = selected_idx_sorted[i]
            subset_pos = subset_positions[gene_idx]
            true_counts = counts[:, gene_idx]

            credible_regions = scribe.stats.compute_histogram_credible_regions(
                results_subset.predictive_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
            )

            hist_results = np.histogram(
                true_counts, bins=credible_regions["bin_edges"], density=True
            )

            cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
            max_bin = np.max(
                [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10]
            )

            scribe.viz.plot_histogram_credible_regions_stairs(
                ax, credible_regions, cmap="Blues", alpha=0.5, max_bin=max_bin
            )

            max_bin_hist = (
                max_bin
                if len(hist_results[0]) > max_bin
                else len(hist_results[0])
            )
            ax.step(
                hist_results[1][:max_bin_hist],
                hist_results[0][:max_bin_hist],
                where="post",
                label="data",
                color="black",
            )

            ax.set_xlabel("counts")
            ax.set_ylabel("frequency")
            actual_mean_expr = np.mean(counts[:, gene_idx])
            mean_expr_formatted = f"{actual_mean_expr:.2f}"
            ax.set_title(
                f"$\\langle U \\rangle = {mean_expr_formatted}$",
                fontsize=8,
            )

            progress.update(task, advance=1)

    plt.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_ppc.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved PPC plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

    del results_subset

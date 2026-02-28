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
from .config import _get_config_values
from .dispatch import (
    _get_biological_ppc_samples_for_plot,
    _get_denoised_counts_for_plot,
)
from .gene_selection import _select_genes


def plot_bio_ppc(results, counts, figs_dir, cfg, viz_cfg):
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
    figs_dir : str
        Directory to save the output figure.
    cfg : OmegaConf
        Hydra run configuration.
    viz_cfg : OmegaConf
        Visualization configuration (must contain ``ppc_opts``).
    """
    console.print("[dim]Plotting biological PPC (denoised)...[/dim]")

    # ------------------------------------------------------------------
    # Gene selection (identical to standard PPC)
    # ------------------------------------------------------------------
    n_rows = viz_cfg.ppc_opts.n_rows
    n_cols = viz_cfg.ppc_opts.n_cols
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

    # ------------------------------------------------------------------
    # Biological PPC samples (gene-subset, stored on results)
    # ------------------------------------------------------------------
    results_subset = results[selected_idx]

    n_samples = viz_cfg.ppc_opts.n_samples
    rng_key = random.PRNGKey(42)

    console.print(
        f"[dim]Generating {n_samples} biological PPC samples...[/dim]"
    )
    _ = _get_biological_ppc_samples_for_plot(
        results_subset,
        rng_key=rng_key,
        n_samples=n_samples,
        counts=counts,
        batch_size=None,
        store_samples=True,
    )

    # ------------------------------------------------------------------
    # Denoised observed counts (full matrix, then gene-index)
    # ------------------------------------------------------------------
    # Use ("sample", "sample") so the denoised histogram has realistic
    # stochastic variability comparable to the biological PPC bands.
    # ("mean", "sample") would create artificial density spikes because
    # every cell with the same observed count maps to a single value.
    console.print("[dim]Denoising observed counts (MAP, sample)...[/dim]")
    key_denoise = random.PRNGKey(99)
    denoised_full = _get_denoised_counts_for_plot(
        results,
        counts=counts,
        rng_key=key_denoise,
        method=("sample", "sample"),
    )

    # Build a position map: original gene index → position inside the
    # gene-subset that was passed to the biological PPC sampler.
    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
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
            "[cyan]Plotting bio-PPC panels...", total=n_genes_selected
        )

        for i, ax in enumerate(axes):
            if i >= n_genes_selected:
                ax.axis("off")
                continue

            gene_idx = selected_idx_sorted[i]
            subset_pos = subset_positions[gene_idx]

            # Credible bands from biological PPC
            credible_regions = scribe.stats.compute_histogram_credible_regions(
                results_subset.predictive_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
            )

            # Denoised data histogram
            denoised_gene = denoised_full[:, gene_idx]
            hist_results = np.histogram(
                denoised_gene,
                bins=credible_regions["bin_edges"],
                density=True,
            )

            # Determine a reasonable x-axis upper bound
            cumsum_indices = np.where(
                np.cumsum(hist_results[0]) <= 0.99
            )[0]
            max_bin = np.max(
                [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10]
            )

            # Bio PPC credible-region bands
            scribe.viz.plot_histogram_credible_regions_stairs(
                ax,
                credible_regions,
                cmap="Greens",
                alpha=0.5,
                max_bin=max_bin,
            )

            # Denoised data step histogram
            max_bin_hist = (
                max_bin
                if len(hist_results[0]) > max_bin
                else len(hist_results[0])
            )
            ax.step(
                hist_results[1][:max_bin_hist],
                hist_results[0][:max_bin_hist],
                where="post",
                label="denoised",
                color="black",
            )

            ax.set_xlabel("counts")
            ax.set_ylabel("frequency")
            denoised_mean = np.mean(denoised_gene)
            ax.set_title(
                rf"$\langle \hat{{X}} \rangle = {denoised_mean:.2f}$",
                fontsize=8,
            )

            progress.update(task, advance=1)

    plt.tight_layout()
    fig.suptitle("Biological PPC (denoised)", y=1.02)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_bio_ppc.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved bio-PPC plot to[/dim] "
        f"[cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

    del results_subset

"""ECDF plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._common import console
from .config import _get_config_values
from .gene_selection import _select_genes_simple


def plot_ecdf(counts, figs_dir, cfg, viz_cfg):
    """Plot and save the ECDF of selected genes."""
    console.print("[dim]Plotting ECDF...[/dim]")

    n_genes = viz_cfg.ecdf_opts.n_genes
    selected_idx, _ = _select_genes_simple(counts, n_genes)
    selected_idx = np.sort(selected_idx)

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    for i, idx in enumerate(selected_idx):
        sns.ecdfplot(
            data=counts[:, idx],
            ax=ax,
            color=sns.color_palette("Blues", n_colors=n_genes)[i],
            lw=1.5,
            label=None,
        )
    ax.set_xlabel("UMI count")
    ax.set_xscale("log")
    ax.set_ylabel("ECDF")
    plt.tight_layout()

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_example_ecdf.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved ECDF plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

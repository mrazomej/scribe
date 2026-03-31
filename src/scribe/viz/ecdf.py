"""ECDF plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._common import console
from ._interactive import (
    _create_or_validate_single_axis,
    _finalize_figure,
    _resolve_render_flags,
)
from .config import _get_config_values
from .gene_selection import _select_genes_simple


def plot_ecdf(
    counts,
    figs_dir=None,
    cfg=None,
    viz_cfg=None,
    *,
    fig=None,
    ax=None,
    axes=None,
    save=None,
    show=None,
    close=None,
):
    """Plot the ECDF of selected genes.

    Parameters
    ----------
    counts : array-like
        Observed UMI count matrix ``(n_cells, n_genes)``.
    figs_dir : str, optional
        Output directory used when ``save`` resolves to ``True``.
    cfg : OmegaConf, optional
        Run configuration used to build output filenames.
    viz_cfg : OmegaConf
        Visualization config containing ``ecdf_opts``.
    fig, ax, axes : matplotlib objects, optional
        Interactive plotting handles. For this single-panel plot, provide
        either ``ax`` or one-item ``axes``.
    save, show, close : bool, optional
        Rendering controls for dual-mode usage.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.
    """
    _fig_owned = fig is None and ax is None and axes is None
    console.print("[dim]Plotting ECDF...[/dim]")
    save, show, close = _resolve_render_flags(
        figs_dir=figs_dir,
        save=save,
        show=show,
        close=close,
    )

    n_genes = viz_cfg.ecdf_opts.n_genes
    selected_idx, _ = _select_genes_simple(counts, n_genes)
    selected_idx = np.sort(selected_idx)

    fig, ax = _create_or_validate_single_axis(
        fig=fig,
        ax=ax,
        axes=axes,
        figsize=(3.5, 3.0),
    )
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
    fig.tight_layout()

    if save:
        output_format = viz_cfg.get("format", "png")
        config_vals = _get_config_values(cfg)
        fname = (
            f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
            f"{config_vals['model_type'].replace('_', '-')}_"
            f"{config_vals['n_components']:02d}components_"
            f"{config_vals['run_size_token']}_example_ecdf.{output_format}"
        )
    else:
        fname = None
    return _finalize_figure(
        fig=fig,
        axes=[ax],
        n_panels=1,
        save=save,
        show=show,
        close=close,
        figs_dir=figs_dir,
        filename=fname,
        save_kwargs={"bbox_inches": "tight"},
        save_label="ECDF plot",
        _fig_owned=_fig_owned,
    )

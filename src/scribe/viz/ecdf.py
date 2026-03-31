"""ECDF plotting."""

import numpy as np
import seaborn as sns

from ._common import console
from ._interactive import (
    _create_or_validate_single_axis,
    plot_function,
)
from .gene_selection import _coerce_counts, _select_genes_simple


_ECDF_DEFAULT_N_GENES = 25


@plot_function(
    suffix="example_ecdf",
    save_label="ECDF plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_ecdf(
    counts,
    *,
    ctx,
    viz_cfg=None,
    n_genes=None,
    fig=None,
    axes=None,
    ax=None,
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
    viz_cfg : OmegaConf or None
        Visualization config containing ``ecdf_opts``.  Optional in
        interactive sessions — a default of 25 genes is used when ``None``.
    n_genes : int, optional
        Number of genes to display.  Overrides
        ``viz_cfg.ecdf_opts.n_genes``.
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
    console.print("[dim]Plotting ECDF...[/dim]")
    counts = _coerce_counts(counts)

    # Resolve n_genes: explicit kwarg > viz_cfg > default
    if n_genes is None:
        if viz_cfg is not None:
            ecdf_opts = (
                viz_cfg.get("ecdf_opts", {}) if hasattr(viz_cfg, "get") else {}
            )
            n_genes = int(
                ecdf_opts.get("n_genes", _ECDF_DEFAULT_N_GENES)
                if hasattr(ecdf_opts, "get")
                else _ECDF_DEFAULT_N_GENES
            )
        else:
            n_genes = _ECDF_DEFAULT_N_GENES
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

    return fig, [ax], 1

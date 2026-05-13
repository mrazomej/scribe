"""W-shrinkage spectrum diagnostic plot.

Renders the per-factor compositional-loading spectrum from a Laplace
result's ``w_prior_diagnostics``.  Companion plot to
:func:`scribe.viz.plot_compositional_corner_ppc`: a clean elbow here
correlates with collapsed cross-block diagonals in the compositional
corner panels.

Reads ``column_frobenius_compositional`` (primary, ``||W_⟂[:, k]||``
sorted descending) and, when ``show_sigma_k=True``, overlays the aux
MAP ``sigma_k`` as a secondary dashed line.  A horizontal dashed line
marks the 5%-of-max threshold used by ``column_norm_effective_rank``.
"""

from __future__ import annotations

import numpy as np

from ._common import console
from ._interactive import _create_or_validate_single_axis, plot_function


@plot_function(
    suffix="w_shrinkage",
    save_label="W shrinkage spectrum",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_w_shrinkage_spectrum(
    results,
    *,
    ctx,
    viz_cfg=None,
    figsize=None,
    fig=None,
    ax=None,
    axes=None,
    show_sigma_k: bool = True,
    show_threshold: bool = True,
    threshold_fraction: float = 0.05,
):
    """Plot the per-factor compositional-loading spectrum.

    Parameters
    ----------
    results : object
        Fitted SCRIBE Laplace result with ``w_prior_diagnostics``
        populated.  Raises ``ValueError`` when missing.
    viz_cfg : OmegaConf or None
        Optional visualization config (currently unused).
    figsize : tuple, optional
        Figure size override.  Defaults to ``(5, 3.5)``.
    fig, ax : matplotlib objects, optional
        Pre-allocated figure / axis.  Pass ``ax=...`` to embed in an
        existing grid; pass ``fig=...`` to add to an existing figure.
    show_sigma_k : bool, default True
        Overlay the aux-scale spectrum (``sigma_k``) as a dashed
        secondary line.  Gracefully degrades to "primary only" when the
        diagnostic dict has no ``sigma_k`` (e.g. ``gaussian`` strategy).
    show_threshold : bool, default True
        Draw a horizontal dashed line at ``threshold_fraction × max``
        of the compositional spectrum to mark the
        ``column_norm_effective_rank`` cutoff.
    threshold_fraction : float, default 0.05
        Fraction of the dominant compositional column used as the
        active-factor threshold.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axis, and metadata.

    Raises
    ------
    ValueError
        If ``results`` has no ``w_prior_diagnostics`` field or it is
        ``None`` (e.g. an LNM-family result in v1 that doesn't yet
        run the W-prior integration).

    Notes
    -----
    The primary spectrum is drawn from
    ``w_prior_diagnostics["column_frobenius_compositional"]`` — the
    gauge-invariant data-supported per-factor norm.  It drives the
    headline ``column_norm_effective_rank`` (and its alias
    ``effective_rank``) reported in the diagnostics dict.  The
    secondary ``sigma_k`` overlay is the strategy's aux MAP scale,
    which can be weakly identified under heavy-tailed priors —
    interpret the column-norm spectrum as the headline.
    """
    console.print("[dim]Plotting W-shrinkage spectrum...[/dim]")
    diag = getattr(results, "w_prior_diagnostics", None)
    if diag is None:
        raise ValueError(
            "results has no w_prior_diagnostics — this plot requires a "
            "Laplace fit configured with a W-shrinkage strategy "
            "(via `w_prior=...` on scribe.fit).  LNM-family results "
            "are not supported in v1."
        )

    col_norm_key = "column_frobenius_compositional"
    if col_norm_key not in diag:
        raise ValueError(
            f"w_prior_diagnostics is missing {col_norm_key!r}.  Has "
            f"keys: {sorted(diag.keys())}."
        )
    col_norm = np.asarray(diag[col_norm_key])
    sorted_col = np.sort(col_norm)[::-1]
    k_axis = np.arange(sorted_col.size)

    sigma_k = diag.get("sigma_k")
    if sigma_k is not None and show_sigma_k:
        sorted_sigma = np.sort(np.asarray(sigma_k))[::-1]
    else:
        sorted_sigma = None

    fig, ax = _create_or_validate_single_axis(
        fig=fig, ax=ax, axes=axes, figsize=figsize or (5.0, 3.5),
    )

    # Primary: compositional column norm.
    ax.plot(
        k_axis, sorted_col,
        marker="o", linewidth=1.6, color="steelblue",
        label=r"$\|W_\perp[:, k]\|$ (data-supported)",
    )

    # Secondary: aux MAP scale.
    if sorted_sigma is not None:
        ax2 = ax.twinx()
        ax2.plot(
            k_axis, sorted_sigma,
            marker="s", linestyle="--", linewidth=1.2,
            color="crimson", alpha=0.75,
            label=r"$\sigma_k$ (aux MAP)",
        )
        ax2.set_ylabel(r"$\sigma_k$", color="crimson")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor="crimson")
        # Add the secondary handle to the primary legend.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    else:
        ax.legend(fontsize=8, loc="best")

    # Threshold line on the primary axis.
    if show_threshold and sorted_col.size > 0:
        thr = threshold_fraction * float(sorted_col.max())
        ax.axhline(
            thr, color="gray", linestyle=":", linewidth=1.0,
            label=f"{threshold_fraction:.0%} of max",
        )

    ax.set_xlabel("factor index $k$ (sorted)")
    ax.set_ylabel(r"$\|W_\perp[:, k]\|$")
    ax.set_yscale("log")

    strategy = diag.get("strategy_type", "unknown")
    rank = diag.get("effective_rank", diag.get("column_norm_effective_rank"))
    title = f"W shrinkage spectrum ({strategy})"
    if rank is not None:
        title = f"{title} — effective rank {int(rank)}"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    return fig, [ax], 1, {
        "column_frobenius_compositional": sorted_col,
        "sigma_k": sorted_sigma,
        "effective_rank": rank,
        "strategy_type": strategy,
    }

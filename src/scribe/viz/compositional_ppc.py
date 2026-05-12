"""Compositional posterior predictive check plotting.

Tests the **gauge-invariant compositional structure** of the fitted model
(see Theorem 1 in ``paper/_diffexp_nbln_robustness.qmd``) by comparing
model-drawn softmax-of-latent population compositions against two
empirical comparators:

1. **Per-cell empirical compositions** — the cloud of observed
   normalised counts ``u_c / N_c`` across cells.  These carry the
   per-cell Multinomial sampling noise on top of the true compositional
   distribution.  For low-abundance genes (where ``p_g ≪ 1/N``), the
   model's smooth distribution will be *narrower* than the empirical
   one — that is the correct prediction, not a misfit.
2. **Dataset-level pseudobulk composition** — the single G-vector
   ``sum_c u_c / sum_c N_c``.  Multinomial noise per cell averages
   away, leaving an unbiased estimator of the population mean
   composition.  The model's density should cover this point in its
   high-density region.

Unlike ``plot_ppc(ppc_level="library_anchored")``, this diagnostic
draws *pre-observation-noise* simplex samples directly from
``results.get_compositional_samples()`` and is therefore the cleanest
test available of the model's compositional structure independent of
the count-observation layer.  For NBLN Laplace fits this is
particularly informative when the gauge contamination ratio is
non-negligible — compositions are exactly gauge-invariant.
"""

import numpy as np
import matplotlib.pyplot as plt
from jax import random

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
from .gene_selection import (
    _coerce_and_align_counts_to_results,
    _coerce_counts,
    _get_gene_names,
    _select_genes,
)


def _compute_empirical_compositions(counts):
    """Per-cell and pseudobulk empirical compositions.

    Parameters
    ----------
    counts : ndarray of shape ``(n_cells, G)``
        Observed counts.  Cells with zero library size are dropped
        from the per-cell comparator (their composition is undefined)
        but their counts still contribute to the pseudobulk total.

    Returns
    -------
    per_cell : ndarray of shape ``(n_cells_nonzero, G)``
        Per-cell empirical compositions ``u_c / N_c``.
    pseudobulk : ndarray of shape ``(G,)``
        Dataset-level pseudobulk composition
        ``sum_c u_c / sum_c N_c`` — a single point on the simplex.
    """
    counts_arr = np.asarray(counts, dtype=float)
    library_sizes = counts_arr.sum(axis=-1)
    nonzero_mask = library_sizes > 0
    per_cell = (
        counts_arr[nonzero_mask] / library_sizes[nonzero_mask, None]
    )
    total_counts = counts_arr.sum(axis=0)
    grand_total = float(counts_arr.sum())
    if grand_total <= 0:
        pseudobulk = np.zeros(counts_arr.shape[1], dtype=float)
    else:
        pseudobulk = total_counts / grand_total
    return per_cell, pseudobulk


def _resolve_compositional_bin_range(model_g, empirical_g, n_bins=40):
    """Pick a shared bin grid for the model and empirical histograms.

    Uses the union of the two value ranges with a small padding so
    rare extremes from one source are not clipped against the other's
    range.  Returns the bin edges and the (lo, hi) tuple in case
    callers want to set axis limits to match.
    """
    lo = float(min(np.min(model_g), np.min(empirical_g)))
    hi = float(max(np.max(model_g), np.max(empirical_g)))
    span = hi - lo
    if span <= 0:
        # All values equal — render a degenerate symmetric range.
        center = lo
        eps = max(abs(center) * 1e-3, 1e-6)
        lo, hi = center - eps, center + eps
    else:
        pad = 0.02 * span
        lo = max(0.0, lo - pad)
        hi = min(1.0, hi + pad)
    edges = np.linspace(lo, hi, n_bins + 1)
    return edges, (lo, hi)


def _render_compositional_diagonal_panel(
    ax,
    model_samples_g,
    empirical_per_cell_g,
    pseudobulk_g,
    *,
    gene_label=None,
    model_color="steelblue",
    model_alpha=0.45,
    empirical_color="black",
    pseudobulk_color="crimson",
    n_bins=40,
):
    """Render a single 1-D compositional PPC panel.

    Layers, bottom to top:
    1. Filled model histogram (model's population-level compositional
       distribution for gene g, normalised to density).
    2. Step histogram overlay of per-cell empirical compositions
       (line only — emphasises distribution shape rather than mass).
    3. Vertical line at the dataset-level pseudobulk value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    model_samples_g : ndarray of shape ``(n_samples,)``
        Compositional draws for gene g from
        ``results.get_compositional_samples()``.
    empirical_per_cell_g : ndarray of shape ``(n_cells_nonzero,)``
        Per-cell empirical compositions for gene g.
    pseudobulk_g : float
        Pseudobulk composition for gene g.
    gene_label : str or None
        Optional title text for the panel.
    """
    edges, (lo, hi) = _resolve_compositional_bin_range(
        model_samples_g, empirical_per_cell_g, n_bins=n_bins,
    )

    ax.hist(
        model_samples_g,
        bins=edges,
        density=True,
        color=model_color,
        alpha=model_alpha,
        label="model",
    )
    ax.hist(
        empirical_per_cell_g,
        bins=edges,
        density=True,
        histtype="step",
        color=empirical_color,
        linewidth=1.2,
        label="per-cell empirical",
    )
    ax.axvline(
        float(pseudobulk_g),
        color=pseudobulk_color,
        linestyle="--",
        linewidth=1.4,
        label="pseudobulk",
    )

    ax.set_xlim(lo, hi)
    ax.set_xlabel("composition $\\rho_g$")
    ax.set_ylabel("density")

    title = f"$\\bar\\rho_g^{{\\mathrm{{pb}}}} = {float(pseudobulk_g):.2e}$"
    if gene_label is not None:
        title = f"{gene_label}\n{title}"
    ax.set_title(title, fontsize=8)


def _select_compositional_genes(counts, n_genes, min_mean_umi=5.0):
    """Pick ``n_genes`` to display, filtered by abundance.

    Compositional comparisons are noise-floor-dominated for very low
    abundance genes (where per-cell Multinomial noise exceeds ``p_g``).
    Filter candidates by mean UMI then select using the existing
    log-spaced binning helper so the chosen genes span the dynamic
    range of the dataset.
    """
    counts_arr = _coerce_counts(counts)
    mean_umi = np.asarray(counts_arr, dtype=float).mean(axis=0)
    candidate_pool = np.where(mean_umi >= float(min_mean_umi))[0]
    if candidate_pool.size == 0:
        raise ValueError(
            "No genes passed the compositional-PPC abundance filter "
            f"(min_mean_umi={min_mean_umi}). Lower the threshold or "
            "supply explicit gene indices."
        )
    n_to_pick = int(min(n_genes, candidate_pool.size))
    # Among the pool, pick log-spaced quantiles of mean UMI so the panel
    # set spans low → high abundance.
    pool_means = mean_umi[candidate_pool]
    sort_order = np.argsort(pool_means)
    sorted_pool = candidate_pool[sort_order]
    sorted_means = pool_means[sort_order]
    if n_to_pick == 1:
        return np.array([sorted_pool[-1]], dtype=int)
    min_safe = max(float(sorted_means[0]), 0.1)
    max_safe = float(sorted_means[-1])
    log_quantiles = np.logspace(
        np.log10(min_safe), np.log10(max_safe), num=n_to_pick,
    )
    selected_positions = np.searchsorted(sorted_means, log_quantiles)
    selected_positions = np.clip(
        selected_positions, 0, sorted_pool.size - 1,
    )
    selected_positions = np.unique(selected_positions)
    # If unique-ing collapsed the set, top up with the largest unused
    # positions until we hit n_to_pick (or run out).
    if selected_positions.size < n_to_pick:
        remaining = np.setdiff1d(
            np.arange(sorted_pool.size), selected_positions,
        )
        fill = remaining[-(n_to_pick - selected_positions.size):]
        selected_positions = np.sort(
            np.concatenate([selected_positions, fill])
        )
    return sorted_pool[selected_positions]


@plot_function(
    suffix="compositional_ppc",
    save_label="compositional PPC plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_compositional_ppc(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    n_rows=None,
    n_cols=None,
    n_genes=None,
    n_samples=None,
    min_mean_umi=5.0,
    rng_key=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
    gene_indices=None,
):
    """Plot compositional posterior predictive checks for selected genes.

    Each panel shows three overlays:

    - **Shaded model histogram**: density of the fitted model's
      compositional samples (``softmax(μ + W_⟂ z + √d ε)`` per draw)
      for the panel's gene.
    - **Step empirical histogram (black)**: per-cell empirical
      compositions ``u_c,g / N_c`` across observed cells.
    - **Dashed vertical line (crimson)**: dataset-level pseudobulk
      composition ``sum_c u_c,g / sum_c N_c``.

    The model histogram represents the *pre-observation-noise*
    population compositional distribution; the per-cell empirical
    includes Multinomial sampling noise (``Var ≈ p_g(1-p_g)/N_c``); the
    pseudobulk averages noise away and gives the dataset-level mean
    composition as a single reference point.

    Parameters
    ----------
    results : object
        Fitted SCRIBE results exposing ``get_compositional_samples``.
        Supported families: PLN, NBLN, LNM, LNMVCP (all Laplace
        results); SVI results when they expose the same method.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    viz_cfg : OmegaConf or None
        Optional visualization config (uses ``ppc_opts`` for grid
        defaults when present).
    n_rows : int, optional
        Number of grid rows. Overrides ``viz_cfg.ppc_opts.n_rows``.
    n_cols : int, optional
        Number of grid columns. Overrides ``viz_cfg.ppc_opts.n_cols``.
    n_genes : int, optional
        Number of genes to display. Falls back to the grid product
        if unset.
    n_samples : int, optional
        Number of compositional draws per panel (default ``2048``).
    min_mean_umi : float, default 5.0
        Lower threshold on per-gene mean UMI for auto-selection.
        Below this floor, per-cell Multinomial noise dominates the
        empirical histogram and the comparison becomes hard to
        interpret.  Ignored when ``gene_indices`` is supplied.
    rng_key : jax.Array, optional
        PRNG key for compositional sampling.  Defaults to
        ``jax.random.PRNGKey(42)``.
    figsize : tuple, optional
        Figure size override.
    fig, axes : matplotlib objects, optional
        Pre-allocated figure / axes grid.
    ax : matplotlib.axes.Axes, optional
        Unsupported — multi-panel plot requires a grid.
    gene_indices : array-like of int, optional
        Explicit column indices to display.  Bypasses auto-selection.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.

    Raises
    ------
    ValueError
        If ``ax`` is supplied (incompatible with multi-panel layout),
        if no genes pass ``min_mean_umi``, or if ``results`` lacks
        ``get_compositional_samples``.

    Notes
    -----
    The model's compositional samples use the gauge-invariant
    projection ``W_⟂`` for PLN/NBLN; LNM/LNMVCP already work in ALR
    coordinates where the gauge structure is absent by construction.
    See ``paper/_diffexp_nbln_robustness.qmd`` for the full theory and
    ``src/scribe/laplace/README.md`` for routing details.
    """
    console.print("[dim]Plotting compositional PPC...[/dim]")
    if ax is not None:
        raise ValueError(
            "Compositional PPC requires multiple axes; provide `fig` or "
            "`axes` instead of `ax`."
        )
    if not hasattr(results, "get_compositional_samples"):
        raise ValueError(
            "results object does not expose `get_compositional_samples` — "
            "compositional PPC requires PLN/NBLN/LNM-family results."
        )

    raw_counts = _coerce_counts(counts)
    counts = _coerce_and_align_counts_to_results(
        raw_counts, results, context="plot_compositional_ppc",
    )

    grid = _resolve_ppc_grid(
        n_rows=n_rows,
        n_cols=n_cols,
        n_genes=n_genes,
        n_samples=n_samples,
        viz_cfg=viz_cfg,
    )
    n_rows = grid["n_rows"]
    n_cols = grid["n_cols"]
    _n_samples = max(int(grid["n_samples"]), 1024)
    n_panel = n_rows * n_cols

    # ------------------------------------------------------------------
    # Gene selection
    # ------------------------------------------------------------------
    if gene_indices is not None:
        selected_idx = np.asarray(gene_indices, dtype=int)
    else:
        selected_idx = _select_compositional_genes(
            counts, n_panel, min_mean_umi=min_mean_umi,
        )
    n_panel = int(selected_idx.size)

    # ------------------------------------------------------------------
    # Sampling + empirical compositions
    # ------------------------------------------------------------------
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    console.print(
        f"[dim]Drawing {_n_samples} compositional samples...[/dim]"
    )
    model_samples = np.asarray(
        results.get_compositional_samples(
            n_samples=_n_samples,
            rng_key=rng_key,
            store_samples=False,
        )
    )
    per_cell, pseudobulk = _compute_empirical_compositions(counts)

    gene_names = _get_gene_names(results)

    # ------------------------------------------------------------------
    # Grid + rendering
    # ------------------------------------------------------------------
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
            "[cyan]Plotting compositional PPC panels...", total=n_panel,
        )
        for i, panel_ax in enumerate(axes_flat):
            if i >= n_panel:
                panel_ax.axis("off")
                continue
            gene_idx = int(selected_idx[i])
            label = (
                str(gene_names[gene_idx])
                if gene_names is not None and gene_idx < len(gene_names)
                else None
            )
            _render_compositional_diagonal_panel(
                panel_ax,
                model_samples[:, gene_idx],
                per_cell[:, gene_idx],
                float(pseudobulk[gene_idx]),
                gene_label=label,
            )
            if i == 0:
                # Legend on the first panel only — avoids visual clutter.
                panel_ax.legend(fontsize=6, loc="upper right")
            progress.update(task, advance=1)

    fig.suptitle("Compositional PPC", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig, axes_flat, n_panel, {
        "selected_gene_indices": selected_idx,
        "pseudobulk": pseudobulk,
    }

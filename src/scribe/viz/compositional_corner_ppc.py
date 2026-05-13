"""Compositional corner posterior predictive check plotting.

Corner-style ``N x N`` grid analog of ``plot_compositional_ppc``.
Diagonals show per-gene 1-D compositional PPC panels; lower-triangle
panels show pairwise 2-D model density (contours) overlaid with
per-cell empirical scatter and a dataset-level pseudobulk marker.

This is the cleanest diagnostic for inspecting **gauge-invariant
cross-gene compositional structure** — Theorem 1 in
``paper/_diffexp_nbln_robustness.qmd`` guarantees that the model's
compositional output is exactly invariant under the rigid-translation
gauge, so any structure visible here reflects the biologically
interpretable signal regardless of how the freeze pinned the gauge.
"""

import numpy as np
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
from .compositional_ppc import (
    _compute_empirical_compositions,
    _render_compositional_diagonal_panel,
    _resolve_compositional_bin_range,
    _select_compositional_correlation_diverse_genes,
    _select_compositional_genes,
)
from .gene_selection import (
    _coerce_and_align_counts_to_results,
    _coerce_counts,
    _get_gene_names,
)


# Density-estimator bound mirrored from corner_ppc.py.  KDE is only
# safe for moderate sample counts; pooled compositional samples are
# subsampled below this cap.
_KDE_MAX_POINTS = 50_000


def _render_compositional_offdiag_panel(
    ax,
    model_samples_x,
    model_samples_y,
    empirical_x,
    empirical_y,
    pseudobulk_x,
    pseudobulk_y,
    *,
    contour_mass_levels=(0.5, 0.68, 0.95, 0.99),
    contour_cmap="Blues",
    contour_alpha=0.6,
    draw_contour_edges=True,
    contour_edgecolor="gray",
    contour_edgewidth=0.7,
    scatter_alpha=0.4,
    scatter_size=6,
    scatter_color="black",
    pseudobulk_color="crimson",
    pseudobulk_marker="X",
    pseudobulk_size=120,
    density_method="hist2d",
    hist2d_bins=80,
    x_range=None,
    y_range=None,
    rng=None,
    empirical_clip_percentiles=(0.5, 99.5),
):
    """Render one lower-triangle compositional panel.

    Three overlays:

    1. **Model 2-D density** (filled HPD contours): from the pooled
       compositional samples ``softmax(latent)[:, (gx, gy)]``.
    2. **Per-cell empirical scatter** (black points): each cell's
       ``(u_c,gx / N_c, u_c,gy / N_c)``.
    3. **Pseudobulk marker** (crimson ``X``): single dataset-level
       point ``(sum_c u_c,gx / sum_c N_c, sum_c u_c,gy / sum_c N_c)``.
    """
    from scipy.stats import gaussian_kde

    pooled_x = np.asarray(model_samples_x, dtype=float).ravel()
    pooled_y = np.asarray(model_samples_y, dtype=float).ravel()
    obs_x = np.asarray(empirical_x, dtype=float)
    obs_y = np.asarray(empirical_y, dtype=float)

    if density_method not in {"hist2d", "kde"}:
        raise ValueError("density_method must be one of {'hist2d', 'kde'}.")

    # Build plotting domain: empirical extents are clipped to robust
    # percentiles so a handful of outlier cells (e.g. unusually high
    # mitochondrial fraction) don't stretch the axis and hide the
    # bulk distribution.  Model extents enter as min/max because the
    # model samples are draws from the fitted population — they don't
    # carry biological outliers in the same way.
    p_lo, p_hi = empirical_clip_percentiles
    obs_x_lo = float(np.percentile(obs_x, p_lo))
    obs_x_hi = float(np.percentile(obs_x, p_hi))
    obs_y_lo = float(np.percentile(obs_y, p_lo))
    obs_y_hi = float(np.percentile(obs_y, p_hi))

    if x_range is None:
        lo_x = float(min(pooled_x.min(), obs_x_lo, float(pseudobulk_x)))
        hi_x = float(max(pooled_x.max(), obs_x_hi, float(pseudobulk_x)))
        pad_x = max((hi_x - lo_x) * 0.05, 1e-6)
        x_lo, x_hi = max(0.0, lo_x - pad_x), min(1.0, hi_x + pad_x)
    else:
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
    if y_range is None:
        lo_y = float(min(pooled_y.min(), obs_y_lo, float(pseudobulk_y)))
        hi_y = float(max(pooled_y.max(), obs_y_hi, float(pseudobulk_y)))
        pad_y = max((hi_y - lo_y) * 0.05, 1e-6)
        y_lo, y_hi = max(0.0, lo_y - pad_y), min(1.0, hi_y + pad_y)
    else:
        y_lo, y_hi = float(y_range[0]), float(y_range[1])

    # Model density field.
    if density_method == "kde":
        if pooled_x.size > _KDE_MAX_POINTS:
            if rng is None:
                rng = np.random.default_rng(0)
            idx = rng.choice(pooled_x.size, size=_KDE_MAX_POINTS, replace=False)
            pooled_x = pooled_x[idx]
            pooled_y = pooled_y[idx]
        n_grid = 80
        xi = np.linspace(x_lo, x_hi, n_grid)
        yi = np.linspace(y_lo, y_hi, n_grid)
        xx, yy = np.meshgrid(xi, yi)
        try:
            kde = gaussian_kde(np.vstack([pooled_x, pooled_y]))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        except np.linalg.LinAlgError:
            zz = None
    else:
        n_bins = max(int(hist2d_bins), 8)
        x_edges = np.linspace(x_lo, x_hi, n_bins + 1)
        y_edges = np.linspace(y_lo, y_hi, n_bins + 1)
        hist, _xe, _ye = np.histogram2d(
            pooled_x, pooled_y, bins=[x_edges, y_edges], density=True,
        )
        x_centers = 0.5 * (_xe[:-1] + _xe[1:])
        y_centers = 0.5 * (_ye[:-1] + _ye[1:])
        xx, yy = np.meshgrid(x_centers, y_centers)
        zz = hist.T

    # Layer 1: per-cell empirical scatter — drawn first so contour fills
    # sit over them.
    ax.scatter(
        obs_x, obs_y,
        s=scatter_size, alpha=scatter_alpha, color=scatter_color,
        edgecolors="none", rasterized=True, zorder=1,
    )

    # Layer 2: model HPD contour fill.
    if zz is not None:
        z_min = float(np.nanmin(zz))
        z_max = float(np.nanmax(zz))
        if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
            mass_levels = np.asarray(contour_mass_levels, dtype=float)
            mass_levels = np.unique(np.sort(mass_levels))
            flat = zz[np.isfinite(zz)].ravel()
            flat = flat[flat > 0.0]
            if flat.size > 0:
                dens_desc = np.sort(flat)[::-1]
                cdf = np.cumsum(dens_desc)
                cdf /= cdf[-1]
                thresholds = []
                for mass in mass_levels:
                    idx = int(np.searchsorted(cdf, mass, side="left"))
                    idx = min(max(idx, 0), dens_desc.size - 1)
                    thresholds.append(float(dens_desc[idx]))
                levels = np.r_[
                    np.sort(np.unique(np.asarray(thresholds))),
                    z_max,
                ]
                if levels.size >= 2 and np.all(np.diff(levels) > 0):
                    ax.contourf(
                        xx, yy, zz,
                        levels=levels,
                        cmap=contour_cmap,
                        alpha=contour_alpha,
                        zorder=2,
                    )
                    if draw_contour_edges and levels.size > 1:
                        ax.contour(
                            xx, yy, zz,
                            levels=levels[:-1],
                            colors=contour_edgecolor,
                            linewidths=float(contour_edgewidth),
                            zorder=3,
                        )

    # Layer 3: pseudobulk marker on top.
    ax.scatter(
        [float(pseudobulk_x)], [float(pseudobulk_y)],
        s=pseudobulk_size,
        marker=pseudobulk_marker,
        color=pseudobulk_color,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
        label="pseudobulk",
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


@plot_function(
    suffix="compositional_corner_ppc",
    save_label="compositional corner PPC plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_compositional_corner_ppc(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    gene_indices=None,
    n_genes=5,
    min_mean_umi=5.0,
    n_samples=None,
    rng_key=None,
    contour_mass_levels=(0.5, 0.68, 0.95, 0.99),
    contour_cmap="Blues",
    contour_alpha=0.6,
    draw_contour_edges=True,
    contour_edgecolor="gray",
    contour_edgewidth=0.7,
    scatter_alpha=0.4,
    scatter_size=6,
    scatter_color="black",
    pseudobulk_color="crimson",
    pseudobulk_marker="X",
    pseudobulk_size=120,
    density_method="hist2d",
    hist2d_bins=80,
    match_offdiag_limits_to_marginals=True,
    gene_selection="correlation_diverse",
    empirical_clip_percentiles=(0.5, 99.5),
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Render a corner-plot-style compositional PPC grid.

    ``N x N`` triangular layout for the auto- or user-selected gene set:

    - **Diagonal**: per-gene compositional PPC panel (shaded model
      histogram + step per-cell empirical + dashed pseudobulk line).
    - **Lower triangle**: pairwise 2-D compositional PPC panels — model
      density contours + per-cell empirical scatter + pseudobulk marker.
    - **Upper triangle**: hidden.

    Compositional samples are drawn from
    ``results.get_compositional_samples(n_samples=...)`` and represent
    the model's pre-observation-noise population compositional
    distribution.  Per-cell empirical compositions
    (``u_c / N_c`` per cell) include Multinomial sampling noise on top
    of the true compositions; the pseudobulk
    (``sum_c u_c / sum_c N_c``) averages noise away and gives a
    single dataset-level reference point.

    Parameters
    ----------
    results : object
        Fitted SCRIBE results exposing ``get_compositional_samples``.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    viz_cfg : OmegaConf or None
        Optional visualization config.
    gene_indices : array-like of int, optional
        Explicit gene column indices.  Bypasses auto-selection.
    n_genes : int
        Number of genes when auto-selecting (default 5).
    min_mean_umi : float, default 5.0
        Lower threshold on per-gene mean UMI for the auto-selection
        candidate pool.  Below this floor, the empirical per-cell
        composition is noise-dominated and the comparison is hard to
        interpret.
    n_samples : int or None
        Number of compositional draws.  Defaults to ``viz_cfg`` or 2048.
    rng_key : jax.Array, optional
        PRNG key.  Defaults to ``jax.random.PRNGKey(42)``.
    contour_mass_levels : tuple of float
        HPD-style cumulative mass levels for the model contour fill.
    contour_cmap, contour_alpha, draw_contour_edges, contour_edgecolor,
    contour_edgewidth : style controls for the model density layer.
    scatter_alpha, scatter_size, scatter_color : style controls for
        per-cell empirical scatter.
    pseudobulk_color, pseudobulk_marker, pseudobulk_size : style
        controls for the dataset-level pseudobulk marker (large
        ``X`` by default to stand out against the scatter cloud).
    density_method : {"hist2d", "kde"}
        Density estimator for the model field.  ``hist2d`` is faster.
    hist2d_bins : int
        Bins per axis for the histogram-based density.
    match_offdiag_limits_to_marginals : bool, default True
        Enforce off-diagonal axis limits to match the corresponding
        diagonal panels.  Keeps the corner geometry visually consistent.
    gene_selection : {"correlation_diverse", "abundance"}
        Auto-selection strategy.  ``"correlation_diverse"`` (default)
        seeds with the most + and most − correlated gene pairs in the
        model's compositional correlation matrix
        (``W_perp W_perp^T + diag(d)`` for Laplace fits) and greedily
        fills with diversity — produces a panel set that exhibits
        meaningful cross-gene structure (positive + negative + diverse).
        ``"abundance"`` falls back to log-spaced quantiles of mean UMI.
    empirical_clip_percentiles : tuple of two floats
        Robust ``(low, high)`` percentile range applied to per-cell
        empirical compositions when choosing per-panel axis limits.
        Default ``(0.5, 99.5)`` keeps a handful of outlier cells (e.g.
        very high mitochondrial fraction) from stretching the axes
        and hiding the bulk distribution.  The data points themselves
        are not removed; only the rendered range is adjusted.
    figsize : tuple or None
        Figure size; defaults to ``(3.0 * N, 3.0 * N)``.
    fig, axes : matplotlib objects, optional
        Pre-allocated figure/axes grid.
    ax : matplotlib.axes.Axes, optional
        Unsupported — multi-panel plot requires a grid.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata
        (``selected_gene_indices``, ``pseudobulk``).

    Raises
    ------
    ValueError
        If ``ax`` is supplied, if ``results`` lacks
        ``get_compositional_samples``, or if no genes pass
        ``min_mean_umi``.
    """
    console.print("[dim]Plotting compositional corner PPC...[/dim]")
    if ax is not None:
        raise ValueError(
            "Compositional corner PPC requires multiple axes; provide "
            "`fig` or `axes` instead of `ax`."
        )
    if not hasattr(results, "get_compositional_samples"):
        raise ValueError(
            "results object does not expose `get_compositional_samples`."
        )

    raw_counts = _coerce_counts(counts)
    counts = _coerce_and_align_counts_to_results(
        raw_counts, results, context="plot_compositional_corner_ppc",
    )

    # Gene selection.  Default ``correlation_diverse`` uses the model's
    # *compositional* correlation matrix (W_perp W_perp^T + diag(d) for
    # Laplace fits; empirical fallback otherwise) to seed with the
    # most + and most − correlated gene pairs, then greedily fills.
    # Use ``"abundance"`` for the legacy log-spaced-by-mean-UMI behaviour.
    if gene_indices is not None:
        selected_idx = np.asarray(gene_indices, dtype=int)
    elif gene_selection == "correlation_diverse":
        selected_idx = _select_compositional_correlation_diverse_genes(
            results, counts, n_genes, min_mean_umi=min_mean_umi,
        )
    elif gene_selection == "abundance":
        selected_idx = _select_compositional_genes(
            counts, n_genes, min_mean_umi=min_mean_umi,
        )
    else:
        raise ValueError(
            "gene_selection must be 'correlation_diverse' or 'abundance'; "
            f"got {gene_selection!r}."
        )
    n_panel = int(selected_idx.size)
    gene_names = _get_gene_names(results)
    console.print(
        f"[dim]Compositional corner PPC: {n_panel} genes selected[/dim]"
    )

    # Sampling.
    grid = _resolve_ppc_grid(
        n_rows=None, n_cols=None, n_genes=None,
        n_samples=n_samples, viz_cfg=viz_cfg,
    )
    _n_samples = max(int(grid["n_samples"]), 1024)
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

    # Grid.
    fig, axes_grid, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_panel,
        n_cols=n_panel,
        fig=fig,
        axes=axes,
        figsize=figsize or (3.0 * n_panel, 3.0 * n_panel),
    )

    # Rendering helpers.
    _rng = np.random.default_rng(42)
    _diag_rendered = np.zeros(n_panel, dtype=bool)

    def _ensure_diagonal(diag_idx):
        if _diag_rendered[diag_idx]:
            return
        gene_idx = int(selected_idx[diag_idx])
        label = (
            str(gene_names[gene_idx])
            if gene_names is not None and gene_idx < len(gene_names)
            else None
        )
        _render_compositional_diagonal_panel(
            axes_grid[diag_idx, diag_idx],
            model_samples[:, gene_idx],
            per_cell[:, gene_idx],
            float(pseudobulk[gene_idx]),
            gene_label=label,
            empirical_clip_percentiles=empirical_clip_percentiles,
        )
        axes_grid[diag_idx, diag_idx].set_yticks([])
        _diag_rendered[diag_idx] = True
        progress.update(task, advance=1)

    def _diagonal_xlim(diag_idx):
        return axes_grid[diag_idx, diag_idx].get_xlim()

    n_renderable = n_panel + n_panel * (n_panel - 1) // 2

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Plotting compositional corner PPC...",
            total=n_renderable,
        )

        for row_idx in range(n_panel):
            for col_idx in range(n_panel):
                axis = axes_grid[row_idx, col_idx]
                if row_idx == col_idx:
                    _ensure_diagonal(row_idx)
                elif row_idx > col_idx:
                    _ensure_diagonal(col_idx)
                    _ensure_diagonal(row_idx)
                    gene_x = int(selected_idx[col_idx])
                    gene_y = int(selected_idx[row_idx])
                    x_range = None
                    y_range = None
                    if match_offdiag_limits_to_marginals:
                        x_range = _diagonal_xlim(col_idx)
                        y_range = _diagonal_xlim(row_idx)
                    _render_compositional_offdiag_panel(
                        axis,
                        model_samples[:, gene_x],
                        model_samples[:, gene_y],
                        per_cell[:, gene_x],
                        per_cell[:, gene_y],
                        float(pseudobulk[gene_x]),
                        float(pseudobulk[gene_y]),
                        contour_mass_levels=contour_mass_levels,
                        contour_cmap=contour_cmap,
                        contour_alpha=contour_alpha,
                        draw_contour_edges=draw_contour_edges,
                        contour_edgecolor=contour_edgecolor,
                        contour_edgewidth=contour_edgewidth,
                        scatter_alpha=scatter_alpha,
                        scatter_size=scatter_size,
                        scatter_color=scatter_color,
                        pseudobulk_color=pseudobulk_color,
                        pseudobulk_marker=pseudobulk_marker,
                        pseudobulk_size=pseudobulk_size,
                        density_method=density_method,
                        hist2d_bins=hist2d_bins,
                        x_range=x_range,
                        y_range=y_range,
                        rng=_rng,
                        empirical_clip_percentiles=(
                            empirical_clip_percentiles
                        ),
                    )
                    progress.update(task, advance=1)
                else:
                    axis.axis("off")
                    continue

                # Corner-style tick / label trimming.
                is_bottom = (row_idx == n_panel - 1)
                is_left = (col_idx == 0)
                if not is_bottom:
                    axis.set_xlabel("")
                    axis.set_xticklabels([])
                if not is_left:
                    axis.set_ylabel("")
                    axis.set_yticklabels([])
                if row_idx == col_idx:
                    # Diagonals get a y-axis label but no tick labels.
                    axis.set_ylabel("")

    fig.suptitle("Compositional corner PPC", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    return fig, axes_flat, n_panel * n_panel, {
        "selected_gene_indices": selected_idx,
        "pseudobulk": pseudobulk,
    }

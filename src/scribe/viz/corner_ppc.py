"""Corner-plot-style posterior predictive check visualization.

Renders an ``n_genes x n_genes`` triangular grid for a small set of
selected genes:

- **Diagonal**: marginal PPC histograms — shaded credible-region bands
  from posterior predictive samples with the observed count histogram
  overlaid (identical to :func:`plot_ppc`).
- **Lower triangle**: bivariate PPC panels — 2-D density contours
  computed from pooled posterior predictive samples, with the observed
  gene–gene scatter plotted on top.
- **Upper triangle**: hidden (``axis.off``).
"""

import numpy as np
import matplotlib.pyplot as plt
from jax import random
from scipy.stats import gaussian_kde
import scribe

from ._common import (
    _is_pln_model,
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
from .gene_selection import (
    _coerce_and_align_counts_to_results,
    _coerce_counts,
    _get_gene_names,
)
from .ppc import _requires_full_gene_sampling, _resolve_ppc_sampling_counts
from .ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    plot_histogram_credible_regions_adaptive,
    plot_observed_histogram_adaptive,
)

# Maximum number of points fed to scipy.stats.gaussian_kde to keep
# the 2-D off-diagonal contours fast even on large datasets.
_KDE_MAX_POINTS = 50_000


# ---------------------------------------------------------------------------
# Correlation-aware gene auto-selection
# ---------------------------------------------------------------------------


def _empirical_correlation_residual(counts, method, n_components):
    """Remove dominant PCA directions from an empirical covariance matrix.

    When no model-implied ``W`` matrix is available, this performs
    eigendecomposition of the empirical covariance of ``log1p(counts)``
    and removes the top principal components before converting back to
    a correlation matrix.

    Parameters
    ----------
    counts : ndarray
        Observed count matrix ``(n_cells, n_genes)``.
    method : {"pc"}
        Only ``"pc"`` is supported for the empirical path (there is no
        model-implied library-size direction without a ``W`` matrix).
    n_components : int
        Number of top principal components to project out.

    Returns
    -------
    ndarray
        Residual correlation matrix of shape ``(n_genes, n_genes)``.
    """
    log_counts = np.log1p(counts.astype(float))
    cov = np.cov(log_counts, rowvar=False)

    # Eigendecomposition, largest eigenvalues last
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Project out the top-n_components directions
    n_remove = min(n_components, len(eigvals))
    for i in range(1, n_remove + 1):
        v = eigvecs[:, -i]
        cov = cov - np.outer(v, v) * eigvals[-i]

    # Convert residual covariance to correlation
    std = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    corr = cov / (std[:, None] * std[None, :])
    # Clamp to [-1, 1] to handle numerical noise
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def _get_correlation_matrix_for_selection(
    results, counts, *, subtract_direction=None, n_pcs_to_remove=1,
):
    """Obtain a gene-gene correlation matrix for auto-selection.

    The source is chosen based on the model type:

    - **Laplace** (PLN / LNM / NBLN): analytic ``W W^T + diag(d)``
      converted to a Pearson correlation matrix.
    - **VAE PLN**: ``get_pln_correlation()`` (closed-form from the
      learned decoder weights).
    - **Everything else**: empirical Pearson correlation on
      ``log1p(counts)``.

    When ``subtract_direction`` is set, nuisance directions (library
    size or top PCs) are projected out before computing the correlation.
    This mirrors the ``subtract_direction`` option in
    :func:`plot_correlation_heatmap`.

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : ndarray
        Observed count matrix ``(n_cells, n_genes)``.
    subtract_direction : {None, "library_size", "pc"}, optional
        Nuisance-direction removal strategy:

        - ``None`` (default): use the full correlation matrix.
        - ``"library_size"``: project out the latent direction closest
          to the all-ones gene vector (Laplace / VAE PLN only; falls
          back to ``"pc"`` for empirical correlation).
        - ``"pc"``: project out the top ``n_pcs_to_remove`` principal
          components.
    n_pcs_to_remove : int
        Number of PCs to remove when ``subtract_direction="pc"``
        (default 1).

    Returns
    -------
    corr : ndarray
        Correlation matrix of shape ``(G_eff, G_eff)`` where
        ``G_eff`` may be smaller than ``counts.shape[1]`` for ALR-space
        models (LNM).
    n_genes_counts : int
        Number of gene columns in ``counts`` so the caller can detect
        dimension mismatches with ``corr``.
    """
    n_genes_counts = int(counts.shape[1])

    # Laplace fits expose analytic correlation (and residual) from W, d
    if isinstance(results, scribe.ScribeLaplaceResults):
        if subtract_direction is not None:
            corr = np.asarray(results.get_correlation_residual(
                method=subtract_direction,
                n_components=int(n_pcs_to_remove),
                include_diagonal_d=False,
            ))
        else:
            corr = np.asarray(results.get_correlation())
        return corr, n_genes_counts

    # VAE PLN-family fits expose an analytic correlation in log-rate
    # space, with optional residual projection.
    if (isinstance(results, scribe.ScribeVAEResults)
            and _is_pln_model(results)):
        if subtract_direction is not None:
            for accessor in (
                "get_nbln_correlation_residual",
                "get_pln_correlation_residual",
            ):
                fn = getattr(results, accessor, None)
                if callable(fn):
                    try:
                        corr = np.asarray(fn(
                            method=subtract_direction,
                            n_components=int(n_pcs_to_remove),
                            include_diagonal_d=False,
                        ))
                        return corr, n_genes_counts
                    except Exception:
                        continue
        else:
            for accessor in ("get_nbln_correlation", "get_pln_correlation"):
                fn = getattr(results, accessor, None)
                if callable(fn):
                    try:
                        corr = np.asarray(fn())
                        return corr, n_genes_counts
                    except Exception:
                        continue

    # Fallback: empirical Pearson on log1p(counts)
    if subtract_direction is not None:
        # "library_size" requires a W matrix; for the empirical path
        # we fall back to PCA removal.
        _method = "pc" if subtract_direction == "library_size" else subtract_direction
        corr = _empirical_correlation_residual(
            counts, method=_method, n_components=int(n_pcs_to_remove),
        )
    else:
        log_counts = np.log1p(counts.astype(float))
        corr = np.corrcoef(log_counts, rowvar=False)
    return corr, n_genes_counts


def _select_genes_by_correlation_diversity(corr_matrix, n_genes, counts):
    """Select genes that span the correlation spectrum for corner PPC.

    The algorithm seeds with the most positively and most negatively
    correlated gene pairs, then greedily fills the remaining slots by
    choosing genes that maximise pairwise diversity (the gene whose
    minimum absolute correlation with all already-selected genes is
    largest).

    Parameters
    ----------
    corr_matrix : ndarray
        Square correlation matrix ``(G_eff, G_eff)``.
    n_genes : int
        Target number of genes to select.
    counts : ndarray
        Observed count matrix ``(n_cells, n_genes_counts)``.  Used only
        to sort the final selection by median expression.

    Returns
    -------
    ndarray of int
        Selected gene column indices into ``counts``, sorted by median
        expression (ascending).

    Notes
    -----
    When ``corr_matrix`` has fewer rows than ``counts`` columns (e.g.
    Laplace LNM in ALR space with G-1 dimensions), the last count
    column (ALR reference gene) is excluded from candidate selection
    and the returned indices map directly into the count matrix.
    """
    g_eff = corr_matrix.shape[0]
    n_genes_counts = int(counts.shape[1])

    # Handle ALR dimension mismatch: if the correlation matrix is
    # smaller than the count matrix, the trailing gene is the ALR
    # reference and is excluded from candidates.
    if g_eff < n_genes_counts:
        candidate_pool = np.arange(g_eff)
    else:
        candidate_pool = np.arange(n_genes_counts)

    # Cap at the number of available candidates
    n_select = min(n_genes, len(candidate_pool))
    if n_select <= 0:
        return np.array([], dtype=int)
    if n_select >= len(candidate_pool):
        selected = candidate_pool.copy()
        medians = np.median(counts[:, selected].astype(float), axis=0)
        return selected[np.argsort(medians)]

    # Work in correlation-matrix index space (0..g_eff-1)
    # Mask the diagonal so it doesn't dominate argmax/argmin
    _corr = corr_matrix.copy()
    np.fill_diagonal(_corr, np.nan)

    # Restrict to the valid candidate pool
    _corr_pool = _corr[np.ix_(candidate_pool, candidate_pool)]

    # Step 1: seed with the most positively correlated pair
    _upper = np.triu_indices(len(candidate_pool), k=1)
    _upper_vals = _corr_pool[_upper]

    # Handle NaN-only matrices gracefully (degenerate case)
    if np.all(np.isnan(_upper_vals)):
        selected = candidate_pool[:n_select]
        medians = np.median(counts[:, selected].astype(float), axis=0)
        return selected[np.argsort(medians)]

    # Most positive pair (indices into candidate_pool)
    _pos_idx = np.nanargmax(_upper_vals)
    _a_local, _b_local = _upper[0][_pos_idx], _upper[1][_pos_idx]
    selected_set = {int(candidate_pool[_a_local]),
                    int(candidate_pool[_b_local])}

    # Step 2: seed with the most negatively correlated pair
    _neg_idx = np.nanargmin(_upper_vals)
    _c_local, _d_local = _upper[0][_neg_idx], _upper[1][_neg_idx]
    selected_set.add(int(candidate_pool[_c_local]))
    selected_set.add(int(candidate_pool[_d_local]))

    selected_list = sorted(selected_set)

    # Step 3: greedy fill until we have n_select genes
    remaining = sorted(set(candidate_pool.tolist()) - selected_set)
    while len(selected_list) < n_select and remaining:
        best_gene = None
        best_score = -np.inf
        for gene in remaining:
            # Diversity score: minimum |corr| with every already-selected gene
            abs_corrs = [abs(float(corr_matrix[gene, s]))
                         for s in selected_list]
            score = min(abs_corrs)
            if score > best_score:
                best_score = score
                best_gene = gene
        selected_list.append(best_gene)
        remaining.remove(best_gene)

    selected_arr = np.asarray(selected_list, dtype=int)

    # Sort by median expression for consistent panel ordering
    medians = np.median(counts[:, selected_arr].astype(float), axis=0)
    return selected_arr[np.argsort(medians)]


def _resolve_gene_indices(
    results,
    counts,
    gene_indices,
    gene_names_list,
    n_genes,
    *,
    subtract_direction=None,
    n_pcs_to_remove=1,
):
    """Resolve which gene columns to display in the corner grid.

    Three modes are supported, checked in priority order:

    1. ``gene_indices`` — explicit integer column indices.
    2. ``gene_names_list`` — gene name strings matched against
       ``results.var.index``.
    3. Correlation-diversity auto-selection via
       :func:`_select_genes_by_correlation_diversity` — seeds with the
       most positively and negatively correlated pairs, then greedily
       fills remaining slots to maximise pairwise diversity.

    Parameters
    ----------
    results : object
        Fitted model results (must have ``var`` for name-based lookup).
    counts : ndarray
        Observed count matrix ``(n_cells, n_genes)`` — used for
        auto-selection and for sorting by median expression.
    gene_indices : array-like of int or None
        Explicit column indices (highest priority).
    gene_names_list : list of str or None
        Gene names to resolve via ``results.var.index``.
    n_genes : int
        Number of genes when auto-selecting.
    subtract_direction : {None, "library_size", "pc"}, optional
        Nuisance-direction removal applied to the correlation matrix
        before auto-selection.  Ignored when ``gene_indices`` or
        ``gene_names_list`` is provided.
    n_pcs_to_remove : int
        Number of PCs to remove when ``subtract_direction="pc"``.

    Returns
    -------
    ndarray of int
        1-D array of gene column indices.

    Raises
    ------
    ValueError
        If ``gene_names_list`` contains names not found in ``results.var``.
    """
    if gene_indices is not None:
        return np.asarray(gene_indices, dtype=int)

    if gene_names_list is not None:
        all_names = _get_gene_names(results)
        if all_names is None:
            raise ValueError(
                "gene_names_list was provided but results object does not "
                "expose gene names (results.var is missing)."
            )
        idx = []
        missing = []
        for name in gene_names_list:
            matches = np.where(all_names == name)[0]
            if len(matches) == 0:
                missing.append(name)
            else:
                idx.append(int(matches[0]))
        if missing:
            raise ValueError(
                f"The following gene names were not found in results.var: "
                f"{missing}"
            )
        return np.asarray(idx, dtype=int)

    # Auto-select: correlation-diversity strategy
    corr_matrix, _n_genes_counts = _get_correlation_matrix_for_selection(
        results, counts,
        subtract_direction=subtract_direction,
        n_pcs_to_remove=n_pcs_to_remove,
    )
    return _select_genes_by_correlation_diversity(corr_matrix, n_genes, counts)


def _render_diagonal_panel(
    ax,
    ppc_samples_gene,
    true_counts,
    render_opts,
    *,
    gene_label=None,
    cmap="Blues",
    alpha=0.5,
):
    """Render one diagonal PPC panel (marginal histogram + credible bands).

    This reuses the same rendering pipeline as :func:`plot_ppc`: shaded
    credible-region bands from posterior predictive samples overlaid with
    the observed count histogram.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis for this panel.
    ppc_samples_gene : ndarray
        Posterior predictive samples for a single gene with shape
        ``(n_draws, n_cells)``.
    true_counts : ndarray
        Observed counts for this gene, shape ``(n_cells,)``.
    render_opts : dict
        Rendering options from :func:`get_ppc_render_options`.
    gene_label : str or None
        Gene name shown as the panel title.
    cmap : str
        Colormap name for credible-region fills.
    alpha : float
        Fill transparency for the credible bands.
    """
    max_bin = compute_adaptive_max_bin(true_counts, render_opts)

    # Credible-region bands from posterior predictive draws
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        ppc_samples_gene,
        credible_regions=[95, 68, 50],
        max_bin=max_bin,
    )

    # Observed data histogram on the same bin grid
    hist_results = np.histogram(
        true_counts, bins=credible_regions["bin_edges"], density=True
    )

    render_meta = plot_histogram_credible_regions_adaptive(
        ax,
        credible_regions,
        cmap=cmap,
        alpha=alpha,
        max_bin=max_bin,
        render_opts=render_opts,
    )
    plot_observed_histogram_adaptive(
        ax,
        hist_results,
        max_bin=max_bin,
        render_meta=render_meta,
        label="data",
        color="black",
    )

    # Title: gene name + mean expression
    actual_mean = np.mean(true_counts)
    title = f"$\\langle U \\rangle = {actual_mean:.2f}$"
    if gene_label is not None:
        title = f"{gene_label}\n{title}"
    ax.set_title(title, fontsize=8)


def _render_offdiag_panel(
    ax,
    ppc_samples_x,
    ppc_samples_y,
    obs_x,
    obs_y,
    *,
    n_contour_levels=6,
    contour_cmap="Blues",
    contour_alpha=0.6,
    scatter_alpha=0.25,
    scatter_size=4,
    scatter_color="black",
    log_scale=False,
    rng=None,
):
    """Render one lower-triangle panel (2-D PPC contour + observed scatter).

    Pools posterior predictive samples across draws, estimates a 2-D KDE,
    and renders filled contour levels.  Observed data is scattered on top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis for this panel.
    ppc_samples_x : ndarray
        PPC samples for the x-axis gene, shape ``(n_draws, n_cells)``.
    ppc_samples_y : ndarray
        PPC samples for the y-axis gene, shape ``(n_draws, n_cells)``.
    obs_x : ndarray
        Observed counts for the x-axis gene, shape ``(n_cells,)``.
    obs_y : ndarray
        Observed counts for the y-axis gene, shape ``(n_cells,)``.
    n_contour_levels : int
        Number of filled contour levels.
    contour_cmap : str
        Matplotlib colormap for the contour fill.
    contour_alpha : float
        Transparency of the contour fill.
    scatter_alpha : float
        Transparency of the observed-data scatter points.
    scatter_size : float
        Marker size for scatter points.
    scatter_color : str
        Colour for scatter points.
    log_scale : bool
        If ``True``, apply ``log1p`` to both PPC samples and observed
        data before KDE estimation and scatter plotting.
    rng : numpy.random.Generator or None
        Random generator used for subsampling large PPC pools.
    """
    # Pool PPC samples across draws → (n_draws * n_cells,)
    pooled_x = ppc_samples_x.ravel().astype(float)
    pooled_y = ppc_samples_y.ravel().astype(float)

    if log_scale:
        pooled_x = np.log1p(pooled_x)
        pooled_y = np.log1p(pooled_y)
        obs_x_plot = np.log1p(obs_x.astype(float))
        obs_y_plot = np.log1p(obs_y.astype(float))
    else:
        obs_x_plot = obs_x.astype(float)
        obs_y_plot = obs_y.astype(float)

    # Subsample if the pooled array is too large for KDE
    n_total = len(pooled_x)
    if n_total > _KDE_MAX_POINTS:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(n_total, size=_KDE_MAX_POINTS, replace=False)
        pooled_x = pooled_x[idx]
        pooled_y = pooled_y[idx]

    # Add tiny jitter to avoid singular covariance on integer-valued data
    _jitter_scale = 0.1
    if rng is None:
        rng = np.random.default_rng(0)
    pooled_x = pooled_x + rng.normal(scale=_jitter_scale, size=len(pooled_x))
    pooled_y = pooled_y + rng.normal(scale=_jitter_scale, size=len(pooled_y))

    # 2-D KDE on the pooled PPC samples
    try:
        kde = gaussian_kde(np.vstack([pooled_x, pooled_y]))
    except np.linalg.LinAlgError:
        # Degenerate case: KDE cannot be computed, fall back to scatter only
        ax.scatter(
            obs_x_plot, obs_y_plot,
            s=scatter_size, alpha=scatter_alpha, color=scatter_color,
            edgecolors="none", rasterized=True,
        )
        return

    # Evaluate KDE on a regular grid spanning the observed data range
    _n_grid = 80
    x_lo, x_hi = float(obs_x_plot.min()), float(obs_x_plot.max())
    y_lo, y_hi = float(obs_y_plot.min()), float(obs_y_plot.max())

    # Small margin so contours don't clip at the boundary
    x_margin = max((x_hi - x_lo) * 0.05, 0.5)
    y_margin = max((y_hi - y_lo) * 0.05, 0.5)
    xi = np.linspace(x_lo - x_margin, x_hi + x_margin, _n_grid)
    yi = np.linspace(y_lo - y_margin, y_hi + y_margin, _n_grid)
    _xx, _yy = np.meshgrid(xi, yi)
    positions = np.vstack([_xx.ravel(), _yy.ravel()])
    _zz = kde(positions).reshape(_xx.shape)

    # Filled contour of the PPC joint density
    ax.contourf(
        _xx, _yy, _zz,
        levels=n_contour_levels,
        cmap=contour_cmap,
        alpha=contour_alpha,
    )

    # Observed data scatter on top
    ax.scatter(
        obs_x_plot, obs_y_plot,
        s=scatter_size, alpha=scatter_alpha, color=scatter_color,
        edgecolors="none", rasterized=True,
    )


@plot_function(
    suffix="corner_ppc",
    save_label="corner PPC plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_corner_ppc(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    gene_indices=None,
    gene_names_list=None,
    n_genes=5,
    subtract_direction=None,
    n_pcs_to_remove=1,
    n_samples=None,
    ppc_level="library_anchored",
    n_contour_levels=6,
    contour_cmap="Blues",
    contour_alpha=0.6,
    scatter_alpha=0.25,
    scatter_size=4,
    scatter_color="black",
    log_scale=False,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Render a corner-plot-style posterior predictive check grid.

    For a set of selected genes the plot creates an ``N x N`` triangular
    grid where:

    - **Diagonal**: marginal PPC histograms (shaded predictive credible
      bands + observed count histogram), identical to :func:`plot_ppc`.
    - **Lower triangle**: bivariate PPC panels showing 2-D density
      contours from pooled posterior predictive samples with the observed
      gene-gene scatter overlaid.
    - **Upper triangle**: hidden.

    Parameters
    ----------
    results : object
        Fitted model results exposing predictive sampling.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    viz_cfg : OmegaConf or None
        Visualization configuration.  Optional in interactive sessions —
        built-in defaults are used when ``None``.
    gene_indices : array-like of int or None
        Explicit column indices of genes to display.  Takes highest
        priority over ``gene_names_list`` and ``n_genes``.
    gene_names_list : list of str or None
        Gene names to display.  Resolved against ``results.var.index``.
    n_genes : int
        Number of genes when auto-selecting (default 5).  Ignored when
        ``gene_indices`` or ``gene_names_list`` is given.
    subtract_direction : {None, "library_size", "pc"}, optional
        Remove nuisance directions from the correlation matrix before
        automatic gene selection.  Useful when library-size variation
        dominates and hides secondary block structure:

        - ``None`` (default): use the full correlation matrix.
        - ``"library_size"``: project out the latent direction closest
          to the all-ones gene vector.  Available for Laplace and VAE
          PLN fits; falls back to ``"pc"`` for the empirical path.
        - ``"pc"``: project out the top ``n_pcs_to_remove`` principal
          components from the covariance.

        Ignored when ``gene_indices`` or ``gene_names_list`` is given.
    n_pcs_to_remove : int
        Number of principal components to project out when
        ``subtract_direction="pc"`` (default 1).
    n_samples : int or None
        Number of posterior predictive draws.  Overrides
        ``viz_cfg.ppc_opts.n_samples``.
    ppc_level : str
        PPC level passed to the predictive sampler (default
        ``"library_anchored"``).
    n_contour_levels : int
        Number of filled contour levels for off-diagonal panels.
    contour_cmap : str
        Colormap for off-diagonal contour fills.
    contour_alpha : float
        Transparency for contour fills.
    scatter_alpha : float
        Transparency for observed-data scatter points.
    scatter_size : float
        Marker size for scatter points.
    scatter_color : str
        Colour for observed-data scatter points.
    log_scale : bool
        If ``True``, apply ``log1p`` to counts before 2-D KDE estimation
        and scatter plotting in off-diagonal panels.
    figsize : tuple of float or None
        Figure size.  Defaults to ``(3.0 * N, 3.0 * N)`` where ``N`` is
        the number of selected genes.
    fig : matplotlib.figure.Figure or None
        Optional figure to draw into.
    axes : array-like of matplotlib.axes.Axes or None
        Optional ``N x N`` axes grid.
    ax : matplotlib.axes.Axes or None
        Unsupported for this multi-panel plot.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.

    Raises
    ------
    ValueError
        If ``ax`` is provided (multi-panel plot requires ``fig`` or
        ``axes``), or if gene name resolution fails.
    """
    console.print("[dim]Plotting corner PPC...[/dim]")
    if ax is not None:
        raise ValueError(
            "Corner PPC requires multiple axes; provide `fig` or `axes` "
            "instead of `ax`."
        )

    # ------------------------------------------------------------------
    # Count alignment
    # ------------------------------------------------------------------
    raw_counts = _coerce_counts(counts)
    counts = _coerce_and_align_counts_to_results(
        raw_counts, results, context="plot_corner_ppc"
    )
    counts_for_sampling = _resolve_ppc_sampling_counts(
        results, raw_counts, counts
    )

    # ------------------------------------------------------------------
    # Gene selection
    # ------------------------------------------------------------------
    selected_idx = _resolve_gene_indices(
        results, counts, gene_indices, gene_names_list, n_genes,
        subtract_direction=subtract_direction,
        n_pcs_to_remove=n_pcs_to_remove,
    )
    n_panel = len(selected_idx)
    gene_names = _get_gene_names(results)
    console.print(
        f"[dim]Corner PPC: {n_panel} genes selected[/dim]"
    )

    # ------------------------------------------------------------------
    # PPC sampling
    # ------------------------------------------------------------------
    grid = _resolve_ppc_grid(
        n_rows=None, n_cols=None, n_genes=None,
        n_samples=n_samples, viz_cfg=viz_cfg,
    )
    _n_samples = grid["n_samples"]

    # Decide whether sampling must run on the full gene space (VAE /
    # amortized capture) or can use the selected subset only.
    sample_full_space = _requires_full_gene_sampling(results)
    if sample_full_space:
        sampling_results = results
        sampling_counts = counts_for_sampling
    else:
        sampling_results = results[selected_idx]
        sampling_counts = counts[:, selected_idx]

    console.print(
        f"[dim]Generating {_n_samples} posterior predictive samples...[/dim]"
    )
    _ = _get_predictive_samples_for_plot(
        sampling_results,
        rng_key=random.PRNGKey(42),
        n_samples=_n_samples,
        counts=sampling_counts,
        store_samples=True,
        ppc_level=ppc_level,
    )

    # Reuse the object that received predictive samples. For the subset
    # sampling path, sampling_results is already gene-aligned and now carries
    # predictive_samples. Re-creating results[selected_idx] here would drop the
    # freshly populated cache when the parent results object had no cache.
    results_subset = (
        results[selected_idx] if sample_full_space else sampling_results
    )

    # Build a mapping from original gene index → position inside the
    # gene-subset (needed because results[selected_idx] preserves
    # the caller-specified gene order).
    subset_positions = {
        int(gene_idx): pos
        for pos, gene_idx in enumerate(selected_idx)
    }

    render_opts = get_ppc_render_options(viz_cfg)

    # ------------------------------------------------------------------
    # Grid creation
    # ------------------------------------------------------------------
    fig, axes_grid, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_panel,
        n_cols=n_panel,
        fig=fig,
        axes=axes,
        figsize=figsize or (3.0 * n_panel, 3.0 * n_panel),
    )

    # ------------------------------------------------------------------
    # Rendering loop
    # ------------------------------------------------------------------
    # Total renderable panels: n_panel diagonal + n_panel*(n_panel-1)/2
    n_renderable = n_panel + n_panel * (n_panel - 1) // 2
    _rng = np.random.default_rng(42)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Plotting corner PPC panels...", total=n_renderable
        )

        for row_idx in range(n_panel):
            for col_idx in range(n_panel):
                axis = axes_grid[row_idx, col_idx]

                if row_idx == col_idx:
                    # --- Diagonal: marginal PPC histogram ---
                    gene_idx = int(selected_idx[row_idx])
                    subset_pos = subset_positions[gene_idx]
                    true_counts = counts[:, gene_idx]
                    ppc_gene = results_subset.predictive_samples[
                        :, :, subset_pos
                    ]
                    _label = (
                        str(gene_names[gene_idx])
                        if gene_names is not None
                        else None
                    )
                    _render_diagonal_panel(
                        axis,
                        ppc_gene,
                        true_counts,
                        render_opts,
                        gene_label=_label,
                    )
                    # Remove y-axis ticks on diagonal to mirror corner style
                    axis.set_yticks([])
                    progress.update(task, advance=1)

                elif row_idx > col_idx:
                    # --- Lower triangle: bivariate PPC contour ---
                    gene_x = int(selected_idx[col_idx])
                    gene_y = int(selected_idx[row_idx])
                    pos_x = subset_positions[gene_x]
                    pos_y = subset_positions[gene_y]
                    ppc_x = results_subset.predictive_samples[:, :, pos_x]
                    ppc_y = results_subset.predictive_samples[:, :, pos_y]
                    obs_x = counts[:, gene_x]
                    obs_y = counts[:, gene_y]

                    _render_offdiag_panel(
                        axis,
                        ppc_x,
                        ppc_y,
                        obs_x,
                        obs_y,
                        n_contour_levels=n_contour_levels,
                        contour_cmap=contour_cmap,
                        contour_alpha=contour_alpha,
                        scatter_alpha=scatter_alpha,
                        scatter_size=scatter_size,
                        scatter_color=scatter_color,
                        log_scale=log_scale,
                        rng=_rng,
                    )
                    progress.update(task, advance=1)

                else:
                    # --- Upper triangle: hidden ---
                    axis.axis("off")
                    continue

                # ----------------------------------------------------------
                # Tick / label management (corner-style layout)
                # ----------------------------------------------------------

                # Bottom row: show x-axis labels with gene name
                if row_idx == n_panel - 1:
                    _xlabel = ""
                    if gene_names is not None:
                        _xlabel = str(gene_names[int(selected_idx[col_idx])])
                    if log_scale and row_idx != col_idx:
                        _xlabel += "\n(log1p counts)"
                    elif row_idx != col_idx:
                        _xlabel += "\n(counts)"
                    axis.set_xlabel(_xlabel, fontsize=8)
                else:
                    axis.set_xticklabels([])
                    axis.tick_params(axis="x", which="both", bottom=False)

                # Left column of lower-triangle panels: show y-axis labels
                if col_idx == 0 and row_idx > col_idx:
                    _ylabel = ""
                    if gene_names is not None:
                        _ylabel = str(gene_names[int(selected_idx[row_idx])])
                    if log_scale:
                        _ylabel += "\n(log1p counts)"
                    else:
                        _ylabel += "\n(counts)"
                    axis.set_ylabel(_ylabel, fontsize=8)
                elif row_idx != col_idx:
                    axis.set_yticklabels([])
                    axis.tick_params(axis="y", which="both", left=False)

                # Diagonal panels above the last row should not show x ticks
                if row_idx == col_idx and row_idx != n_panel - 1:
                    axis.set_xticklabels([])
                    axis.tick_params(axis="x", which="both", bottom=False)

    fig.suptitle("Corner PPC", fontsize=11, y=1.01)
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.08,
        top=0.92,
        wspace=0.03,
        hspace=0.03,
    )

    del results_subset
    return fig, axes_flat, n_panel * n_panel

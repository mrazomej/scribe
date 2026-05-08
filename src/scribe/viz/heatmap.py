"""Correlation heatmap plotting."""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
import jax.numpy as jnp

import scribe

from ._common import _is_pln_model, console
from ._interactive import PlotResultCollection, plot_function
from .dispatch import _get_layouts_for_plot
from .gene_selection import _coerce_and_align_counts_to_results


def _compute_correlation_matrix(samples, n_samples):
    """Compute pairwise Pearson correlation matrix from posterior samples."""
    samples_centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    samples_std = jnp.std(samples, axis=0, keepdims=True)
    samples_std = jnp.where(samples_std == 0, 1.0, samples_std)
    samples_standardized = samples_centered / samples_std
    correlation_matrix = (samples_standardized.T @ samples_standardized) / (
        n_samples - 1
    )
    return correlation_matrix


def _select_genes_by_w_clusters(
    W: np.ndarray,
    n_take: int,
    n_clusters: int,
):
    """Cluster genes by *direction* of their W rows; pick representatives.

    The motivation is that gene-gene correlation in PLN/LNM is
    governed by the *direction* of the W rows in the latent factor
    space, not their magnitude:

        Σ_gh = W_g · W_h + δ_gh d_g
        Corr(g, h) ≈ cos(angle(W_g, W_h)) (modulo d_g, d_h variance)

    Sorting by ``||W_g||`` therefore picks genes that participate in
    the latent structure but tells us nothing about *which*
    direction they participate in. If one direction dominates (a
    common pathology — typically a "library-size" or "global mean"
    factor), every high-norm gene aligns with it and the heatmap
    collapses to a uniform block of high positive correlation.

    Clustering unit-norm W rows finds groups of genes that share a
    direction. Picking top genes from each cluster gives a heatmap
    where each within-cluster block is internally highly correlated
    and between-cluster pairs reveal the latent geometry.

    Parameters
    ----------
    W : np.ndarray
        Shape ``(G_eff, k)`` factor-loading matrix.
    n_take : int
        Total number of genes to return.
    n_clusters : int
        Number of W-direction clusters to extract.

    Returns
    -------
    selected_idx : np.ndarray
        Selected gene indices, ordered by cluster id (so the
        heatmap shows block-diagonal structure when row/col
        clustering is disabled).
    cluster_id : np.ndarray
        Cluster id (1-indexed) for each selected gene, parallel to
        ``selected_idx``.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    G_eff = W.shape[0]
    n_clusters_eff = max(1, min(int(n_clusters), G_eff))

    # Unit-normalize each row. Genes with effectively-zero W row
    # don't participate in the latent factors; we keep them in W_hat
    # but they will form their own cluster (unit ε vector points
    # nowhere meaningful, which the linkage will detect as outliers).
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    W_hat = W / safe_norms

    # Euclidean distance on unit vectors: ‖u − v‖² = 2 − 2 cos θ, so
    # this is monotone in angular distance with cleaner numerics.
    # ``method="average"`` gives the most stable cuts for cosine-like
    # geometry; ``ward`` would also work but assumes squared
    # Euclidean distortion which is fine here. We use average.
    Z = linkage(W_hat, method="average", metric="euclidean")
    cluster_id_full = fcluster(
        Z, t=n_clusters_eff, criterion="maxclust"
    )

    # From each cluster, take the top genes by ||W_g|| (so we
    # surface the genes that load most strongly *within* each
    # direction). Distribute n_take roughly evenly across clusters.
    base_per_cluster = max(1, n_take // n_clusters_eff)

    selected_per_cluster = []
    for cid in range(1, n_clusters_eff + 1):
        members = np.where(cluster_id_full == cid)[0]
        if members.size == 0:
            continue
        member_norms = np.linalg.norm(W[members], axis=1)
        order = members[np.argsort(member_norms)[::-1]]
        selected_per_cluster.append((cid, order[:base_per_cluster]))

    # If we under-filled (clusters smaller than ``base_per_cluster``)
    # top up by global ||W_g|| ranking on the unselected genes.
    used = np.concatenate([sel for _, sel in selected_per_cluster])
    short = n_take - used.size
    if short > 0:
        remaining = np.setdiff1d(np.arange(G_eff), used)
        rem_norms = np.linalg.norm(W[remaining], axis=1)
        topup = remaining[np.argsort(rem_norms)[::-1][:short]]
        # Assign them cluster id 0 ("other") for the colorbar strip.
        selected_per_cluster.append((0, topup))

    # Concatenate in cluster-id order so the resulting block-diagonal
    # ordering shows the geometry directly when clustermap row/col
    # clustering is disabled.
    selected_per_cluster.sort(key=lambda kv: kv[0])
    selected_idx = np.concatenate(
        [sel for _, sel in selected_per_cluster]
    )
    cluster_id = np.concatenate(
        [np.full(sel.shape[0], cid, dtype=int)
         for cid, sel in selected_per_cluster]
    )
    return selected_idx, cluster_id


def _plot_laplace_correlation_heatmap(
    results,
    *,
    ctx,
    n_genes_to_plot: int,
    figsize,
    cmap,
    base_fname: str,
    output_format: str,
    gene_selection: str = "w_clusters",
    n_clusters: int = 8,
    subtract_direction: Optional[str] = None,
    n_pcs_to_remove: int = 1,
):
    """Render a clustered correlation heatmap from a Laplace fit's globals.

    For PLN/LNM Laplace results, the model's correlation structure
    is *parameterised* directly: ``Sigma = W W' + diag(d)``. We can
    therefore compute the gene-gene Pearson correlation in closed
    form rather than via Monte-Carlo PPC samples, which is both
    cheaper and free of the encoder's aggregate-posterior drift.

    Two gene-selection strategies are supported:

    - ``"w_clusters"`` (default): hierarchical-cluster genes by the
      *direction* of their W rows (unit-normalized) and pick top
      genes from each cluster. Surfaces multi-block structure even
      when one latent direction dominates the L2 norm.
    - ``"w_norm"``: pick the top ``n_genes`` by ``||W_g||``. Quick,
      but collapses to a single block when one direction dominates.

    Optional library-size / nuisance-direction removal:

    - ``subtract_direction="library_size"``: project out the
      column-space-of-W direction whose image is closest to
      $\\mathbf{1}_G$ before computing correlations. Recommended when
      the model has absorbed library-size variation into one of
      its latent factors and the resulting "sea of red" heatmap
      hides the secondary block structure. See
      :meth:`ScribeLaplaceResults.get_correlation_residual`.
    - ``subtract_direction="pc"``: project out the top
      ``n_pcs_to_remove`` principal components of $W^\\top W$.
      A model-agnostic fallback when the library-size direction
      is not well-aligned with $\\mathbf{1}_G$.
    - ``subtract_direction=None`` (default): plot the full
      model-implied correlation matrix.
    """
    bm = getattr(results.model_config, "base_model", None)
    bm_str = str(getattr(bm, "value", bm) or "").lower()

    if subtract_direction is None:
        correlation_matrix = np.asarray(results.get_correlation())
    elif subtract_direction in ("library_size", "pc"):
        correlation_matrix = np.asarray(
            results.get_correlation_residual(
                method=subtract_direction,
                n_components=int(n_pcs_to_remove),
                include_diagonal_d=False,
            )
        )
    else:
        raise ValueError(
            f"subtract_direction must be None, 'library_size', or 'pc'; "
            f"got {subtract_direction!r}"
        )
    W = np.asarray(results.W)

    n_total = int(correlation_matrix.shape[0])
    n_take = min(int(n_genes_to_plot), n_total)

    cluster_id_per_gene: np.ndarray | None = None
    if gene_selection == "w_norm":
        # Sort-by-magnitude path. Useful when the user wants the
        # single most-loaded block; loses information when one
        # latent direction dominates.
        w_norm = np.linalg.norm(W, axis=1)
        top_idx = np.argsort(w_norm)[-n_take:]
        top_idx = np.sort(top_idx)
        selection_label = "‖W_g‖ (top by magnitude)"
    elif gene_selection == "w_clusters":
        # Direction-cluster path. See _select_genes_by_w_clusters
        # for the geometric justification.
        top_idx, cluster_id_per_gene = _select_genes_by_w_clusters(
            W, n_take=n_take, n_clusters=n_clusters
        )
        selection_label = (
            f"W-direction clusters (k={n_clusters}, top genes per cluster)"
        )
    else:
        raise ValueError(
            f"gene_selection must be 'w_norm' or 'w_clusters'; "
            f"got {gene_selection!r}"
        )

    correlation_subset = correlation_matrix[np.ix_(top_idx, top_idx)]

    # Diagnostics: report the off-diagonal correlation range so the
    # user can quickly tell whether the model found block structure
    # at all (range [-eps, +eps] ⇒ no structure; range close to ±1
    # ⇒ strongly correlated blocks).
    off_diag = correlation_subset.copy()
    np.fill_diagonal(off_diag, 0.0)
    console.print(
        f"[dim]Selected {n_take} genes via {gene_selection!r} "
        f"(out of {n_total} {'ALR' if bm_str in ('lnm', 'lnmvcp') else 'log-rate'} dims)[/dim]"
    )
    if cluster_id_per_gene is not None:
        unique_cids, counts_per_cid = np.unique(
            cluster_id_per_gene, return_counts=True
        )
        console.print(
            f"[dim]Cluster sizes: "
            f"{dict(zip(unique_cids.tolist(), counts_per_cid.tolist()))}[/dim]"
        )
    console.print(
        f"[dim]Off-diagonal correlation range: "
        f"[{off_diag.min():.3f}, {off_diag.max():.3f}][/dim]"
    )

    space_label = (
        "ALR composition logits"
        if bm_str in ("lnm", "lnmvcp")
        else "log-rate latents"
    )
    space_extra = (
        "\n(reference gene excluded; ALR space has G-1 dims by construction)"
        if bm_str in ("lnm", "lnmvcp")
        else ""
    )

    # When using direction clustering, pre-order rows/cols by cluster
    # id and disable clustermap's own dendrogram-based reordering so
    # the user sees the direction blocks directly. Add a colored
    # row strip so each cluster is visually distinct. When using
    # w_norm we let clustermap re-cluster as before.
    clustermap_kwargs = dict(
        cmap=cmap, center=0, vmin=-1, vmax=1,
        figsize=figsize, dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.83, 0.03, 0.15),
        linewidths=0, xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Pearson Correlation"},
    )
    if cluster_id_per_gene is not None:
        # Build a per-row color strip so the clusters are easy to
        # see at a glance.
        unique_cids = np.unique(cluster_id_per_gene)
        palette = sns.color_palette("tab10", n_colors=max(10, len(unique_cids)))
        cid_to_color = {
            int(cid): palette[i % len(palette)]
            for i, cid in enumerate(unique_cids)
        }
        row_colors = [
            cid_to_color[int(cid)] for cid in cluster_id_per_gene
        ]
        clustermap_kwargs.update(
            row_cluster=False,
            col_cluster=False,
            row_colors=row_colors,
            col_colors=row_colors,
        )

    console.print(
        "[dim]Creating clustered heatmap (analytic correlation)...[/dim]"
    )
    fig = sns.clustermap(correlation_subset, **clustermap_kwargs)

    if subtract_direction == "library_size":
        projection_note = "; library-size direction projected out"
    elif subtract_direction == "pc":
        projection_note = (
            f"; top-{int(n_pcs_to_remove)} W^TW PCs projected out"
        )
    else:
        projection_note = ""

    fig.fig.suptitle(
        f"Gene Correlation Structure — {space_label}{space_extra}\n"
        f"Top {n_take} genes via {selection_label}\n"
        f"(analytic: W Wᵀ + diag(d), model={bm_str.upper() or '?'}{projection_note})",
        y=1.02,
        fontsize=12,
    )

    fname = f"{base_fname}_correlation_heatmap.{output_format}"
    return ctx.finalize(
        fig.fig,
        list(fig.fig.axes),
        1,
        filename=fname if ctx.save else None,
        save_kwargs={"bbox_inches": "tight"},
        save_label="correlation heatmap",
    )


@plot_function()
def plot_correlation_heatmap(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    n_genes=None,
    n_samples=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
    gene_selection: str = "w_clusters",
    n_clusters: int = 8,
    subtract_direction: Optional[str] = None,
    n_pcs_to_remove: int = 1,
):
    """Plot clustered correlation heatmap of gene parameters from posterior samples.

    For mixture models, a separate heatmap is produced for every component.

    Genes are ranked by variance of their correlation matrix row (non-mixture)
    or by the same criterion per mixture component, then the top ``n_genes``
    (after capping by the model's gene count) are shown.

    Parameters
    ----------
    results : object
        Fitted results with posterior samples for the active parameterization.
    counts : array-like or None
        Count matrix; used when posterior samples must be generated on the fly
        (SVI-style results).
    figs_dir : str, optional
        Output directory when ``save`` resolves to ``True``.
    cfg : OmegaConf, optional
        Run configuration used for filename generation.
    viz_cfg : OmegaConf or dict or None
        May include ``heatmap_opts`` with ``n_genes``, ``n_samples``,
        ``figsize``, and ``cmap``. Used when explicit keyword arguments are
        omitted.
    n_genes : int or None, optional
        Maximum number of genes to plot (highest correlation-variance genes).
        When ``None``, uses ``viz_cfg["heatmap_opts"]["n_genes"]`` if present,
        otherwise ``1500``. When an ``int``, overrides that config entry.
    n_samples : int or None, optional
        Number of posterior draws used to compute Pearson correlations. When
        ``None`` and new samples must be drawn (SVI-style results), uses
        ``viz_cfg["heatmap_opts"]["n_samples"]`` if present, otherwise ``512``.
        When ``None`` and samples are already stored (e.g. MCMC), **all**
        stored draws are used. When an ``int``, caps correlations to the
        first that many draws and overrides the config value for **new**
        draws.
    figsize : tuple of float, optional
        ``(width, height)`` in inches; overrides ``heatmap_opts.figsize``
        (scalar side length) when given.
    fig, axes, ax : optional
        Unsupported; correlation heatmaps use ``seaborn.clustermap``.
    save, show, close : bool, optional
        Rendering controls injected by ``@plot_function``.

    Returns
    -------
    PlotResult or PlotResultCollection
        Single figure for a non-mixture model; one result per mixture component
        otherwise. Each value wraps the figure, axes, and metadata.
    """
    console.print("[dim]Plotting correlation heatmap...[/dim]")
    if counts is not None:
        counts = _coerce_and_align_counts_to_results(
            counts, results, context="plot_correlation_heatmap"
        )
    # Seaborn clustermap owns its own figure/axes objects, so custom axis
    # injection is intentionally unsupported for this entry point.
    if fig is not None or ax is not None or axes is not None:
        raise ValueError(
            "Correlation heatmap uses seaborn.clustermap and does not accept "
            "`fig`/`ax`/`axes`. Call without axes."
        )

    heatmap_opts = (
        viz_cfg.get("heatmap_opts", {}) if viz_cfg is not None else {}
    )
    # Explicit ``n_genes`` wins over viz_cfg for interactive calls; cfg alone
    # keeps backward-compatible pipeline behavior.
    _cfg_n_genes = heatmap_opts.get("n_genes", 1500)
    n_genes_to_plot = _cfg_n_genes if n_genes is None else n_genes
    _cfg_n_samples = heatmap_opts.get("n_samples", 512)
    # figsize kwarg takes priority; fall back to config, then default (12, 12).
    _cfg_figsize = heatmap_opts.get("figsize", 12)
    if figsize is None:
        figsize = (_cfg_figsize, _cfg_figsize)
    cmap = heatmap_opts.get("cmap", "RdBu_r")

    # Build the filename prefix once — both branches (Laplace and
    # SVI/MCMC) write through the same context machinery.
    base_fname = "correlation"
    if ctx.save:
        _fname_prefix = ctx.build_filename(
            "correlation_heatmap", results=results
        )
        base_fname = _fname_prefix.rsplit("_correlation_heatmap.", 1)[0]
    output_format = ctx.output_format

    # ----------------------------------------------------------------
    # Laplace path: correlation is closed-form ``W Wᵀ + diag(d)`` so we
    # bypass the entire sampling pipeline (and the PLN-skip guard,
    # which only applies to SVI/VAE PLN fits that have no per-gene
    # scalar posteriors). ``counts`` and ``n_samples`` are unused
    # here — we ignore them with a console note for clarity.
    # ----------------------------------------------------------------
    if isinstance(results, scribe.ScribeLaplaceResults):
        if n_samples is not None:
            console.print(
                "[yellow]Note: `n_samples` is ignored for Laplace fits; "
                "correlation is computed analytically from W and d.[/yellow]"
            )
        if counts is not None:
            console.print(
                "[dim]Note: `counts` is unused for analytic Laplace correlation.[/dim]"
            )
        return _plot_laplace_correlation_heatmap(
            results,
            ctx=ctx,
            n_genes_to_plot=n_genes_to_plot,
            figsize=figsize,
            cmap=cmap,
            base_fname=base_fname,
            output_format=output_format,
            gene_selection=gene_selection,
            n_clusters=n_clusters,
            subtract_direction=subtract_direction,
            n_pcs_to_remove=n_pcs_to_remove,
        )

    # ----------------------------------------------------------------
    # Sampling-based path (SVI / VAE / MCMC). Falls through to the
    # legacy implementation below.
    # ----------------------------------------------------------------
    if counts is not None:
        counts = _coerce_and_align_counts_to_results(
            counts, results, context="plot_correlation_heatmap"
        )

    # PLN does not expose NB-style per-gene scalar posteriors (r/mu) used by
    # this posterior-sample correlation heatmap implementation.
    if _is_pln_model(results):
        console.print(
            "[yellow]Skipping correlation heatmap for PLN: this plot expects "
            "NB-family posterior samples ('r' or 'mu'). Use PLN extraction "
            "methods (e.g. get_pln_correlation/get_pln_sigma) instead."
            "[/yellow]"
        )
        return

    parameterization = results.model_config.parameterization
    if parameterization in ["linked", "mean_prob", "odds_ratio", "mean_odds"]:
        param_name = "mu"
    else:
        param_name = "r"

    console.print(
        f"[dim]Using parameter '{param_name}' for correlation "
        f"(parameterization: {parameterization})[/dim]"
    )

    # Whether samples were absent before this call (affects console messaging
    # after we optionally thin the chain for correlations).
    _posterior_was_missing = results.posterior_samples is None
    if _posterior_was_missing:
        n_draw = _cfg_n_samples if n_samples is None else n_samples
        console.print(f"[dim]Generating {n_draw} posterior samples...[/dim]")
        results.get_posterior_samples(
            rng_key=random.PRNGKey(42),
            n_samples=n_draw,
            store_samples=True,
            counts=counts,
        )
    else:
        console.print("[dim]Using existing posterior samples...[/dim]")

    if param_name not in results.posterior_samples:
        console.print(
            f"[bold red]❌ ERROR:[/bold red] [red]Parameter '{param_name}' "
            "not found in posterior samples.[/red]"
        )
        console.print(
            f"[dim]   Available parameters:[/dim] "
            f"{list(results.posterior_samples.keys())}"
        )
        return

    samples = results.posterior_samples[param_name]
    console.print(f"[dim]Sample shape:[/dim] {samples.shape}")

    n_avail = int(samples.shape[0])
    # Thin to the first ``n_samples`` draws when the caller sets that cap;
    # otherwise keep every stored draw (MCMC) or the count just drawn (SVI).
    if n_samples is not None:
        n_draws_for_corr = min(int(n_samples), n_avail)
        if n_draws_for_corr < n_avail:
            console.print(
                f"[dim]Using {n_draws_for_corr} of {n_avail} posterior draws "
                "for correlation matrix...[/dim]"
            )
        samples = samples[:n_draws_for_corr]
    else:
        n_draws_for_corr = n_avail
        if not _posterior_was_missing:
            console.print(f"[dim]Found {n_avail} existing samples[/dim]")

    # ``base_fname`` and ``output_format`` are computed at the top of
    # the function and shared between the Laplace and SVI/MCMC paths.

    # Use canonical AxisLayout (keyed by "r", "mu", etc.) to determine
    # whether this is a mixture model and which axes carry semantics.
    # Posterior samples have a leading sample dim, so shift indices.
    _canonical_layouts = _get_layouts_for_plot(results)
    _layout = _canonical_layouts[param_name].with_sample_dim()
    _is_mixture = _layout.component_axis is not None

    if _is_mixture:
        n_components = samples.shape[_layout.component_axis]
        n_genes = samples.shape[_layout.gene_axis]
        n_genes_capped = min(n_genes_to_plot, n_genes)
        console.print(
            f"[dim]Detected mixture model with {n_components} components "
            f"– generating per-component heatmaps[/dim]"
        )

        correlation_matrices = []
        variance_per_component = []

        for k in range(n_components):
            comp_samples = jnp.take(samples, k, axis=_layout.component_axis)
            corr_matrix = _compute_correlation_matrix(
                comp_samples, n_draws_for_corr
            )
            correlation_matrices.append(corr_matrix)
            corr_var = jnp.var(corr_matrix, axis=1)
            variance_per_component.append(corr_var)

            console.print(
                f"[dim]  Component {k + 1}: correlation range "
                f"[{float(jnp.min(corr_matrix)):.3f}, "
                f"{float(jnp.max(corr_matrix)):.3f}][/dim]"
            )

        selected_genes_set = set()
        for k in range(n_components):
            top_indices = jnp.argsort(variance_per_component[k])[
                -n_genes_capped:
            ]
            selected_genes_set.update(np.array(top_indices).tolist())

        selected_genes = np.sort(np.array(list(selected_genes_set)))
        console.print(
            f"[dim]Union of top {n_genes_capped} genes per component: "
            f"{len(selected_genes)} unique genes[/dim]"
        )

        component_results = []
        for k in range(n_components):
            corr_subset_np = np.array(
                correlation_matrices[k][jnp.ix_(selected_genes, selected_genes)]
            )
            console.print(
                f"[dim]Creating clustered heatmap for component "
                f"{k + 1}/{n_components}...[/dim]"
            )

            fig = sns.clustermap(
                corr_subset_np,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                figsize=figsize,
                dendrogram_ratio=0.15,
                cbar_pos=(0.02, 0.83, 0.03, 0.15),
                linewidths=0,
                xticklabels=False,
                yticklabels=False,
                cbar_kws={"label": "Pearson Correlation"},
            )

            fig.fig.suptitle(
                f"Gene Correlation Structure – Component "
                f"{k + 1}/{n_components}\n"
                f"Top {len(selected_genes)} Genes by Variance "
                f"(Union Across Components)\n"
                f"Parameter: {param_name} | Samples: {n_draws_for_corr}",
                y=1.02,
                fontsize=12,
            )

            fname = (
                f"{base_fname}_correlation_heatmap_"
                f"component{k + 1}.{output_format}"
            )
            result = ctx.finalize(
                fig.fig,
                list(fig.fig.axes),
                1,
                filename=fname if ctx.save else None,
                save_kwargs={"bbox_inches": "tight"},
                save_label=f"component {k + 1} heatmap",
            )
            component_results.append(result)
        return (
            PlotResultCollection(component_results)
            if component_results
            else None
        )

    else:
        n_genes = samples.shape[_layout.gene_axis]
        n_genes_to_plot = min(n_genes_to_plot, n_genes)

        console.print("[dim]Computing pairwise Pearson correlations...[/dim]")
        correlation_matrix = _compute_correlation_matrix(
            samples, n_draws_for_corr
        )

        console.print(
            f"[dim]Correlation matrix shape:[/dim] "
            f"{correlation_matrix.shape}"
        )
        console.print(
            f"[dim]Correlation range:[/dim] "
            f"[{float(jnp.min(correlation_matrix)):.3f}, "
            f"{float(jnp.max(correlation_matrix)):.3f}]"
        )

        console.print(
            f"[dim]Selecting top {n_genes_to_plot} genes by correlation "
            f"variance...[/dim]"
        )

        correlation_variance = jnp.var(correlation_matrix, axis=1)
        top_var_indices = jnp.argsort(correlation_variance)[-n_genes_to_plot:]
        top_var_indices = jnp.sort(top_var_indices)

        correlation_subset = correlation_matrix[top_var_indices, :][
            :, top_var_indices
        ]

        console.print(
            f"[dim]Selected {n_genes_to_plot} genes with highest "
            f"correlation variance[/dim]"
        )
        console.print(
            f"[dim]Subset correlation matrix shape:[/dim] "
            f"{correlation_subset.shape}"
        )

        correlation_subset_np = np.array(correlation_subset)

        console.print("[dim]Creating clustered heatmap...[/dim]")

        fig = sns.clustermap(
            correlation_subset_np,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            figsize=figsize,
            dendrogram_ratio=0.15,
            cbar_pos=(0.02, 0.83, 0.03, 0.15),
            linewidths=0,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": "Pearson Correlation"},
        )

        fig.fig.suptitle(
            f"Gene Correlation Structure "
            f"(Top {n_genes_to_plot} by Variance)\n"
            f"Parameter: {param_name} | Samples: {n_draws_for_corr}",
            y=1.02,
            fontsize=12,
        )

        fname = f"{base_fname}_correlation_heatmap.{output_format}"
        return ctx.finalize(
            fig.fig,
            list(fig.fig.axes),
            1,
            filename=fname if ctx.save else None,
            save_kwargs={"bbox_inches": "tight"},
            save_label="correlation heatmap",
        )

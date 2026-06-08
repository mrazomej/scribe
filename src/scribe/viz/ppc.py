"""Posterior predictive check plotting."""

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
from ._interactive import (
    _create_or_validate_grid_axes,
    _resolve_ppc_grid,
    plot_function,
)
from .dispatch import (
    _get_map_like_predictive_samples_for_plot,
    _get_predictive_samples_for_plot,
)
from .gene_selection import (
    _coerce_and_align_counts_to_results,
    _coerce_counts,
    _get_gene_names,
    _resolve_explicit_genes,
    _resolve_pooled_other_idx,
    _select_genes,
)
from .ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    plot_histogram_credible_regions_adaptive,
    plot_observed_histogram_adaptive,
)


def _has_flow_params(params) -> bool:
    """Return True when a results param dict includes flow-guide weights.

    Parameters
    ----------
    params : object
        Results parameter mapping, typically ``results.params``.

    Returns
    -------
    bool
        ``True`` when flow-module parameter blocks are present.
    """
    if not isinstance(params, dict):
        return False
    return any(
        key.endswith("$params")
        and (key.startswith("flow_") or key.startswith("joint_flow_"))
        for key in params
    )


def _requires_full_gene_sampling(results) -> bool:
    """Return whether PPC sampling should run on full results space.

    Parameters
    ----------
    results : object
        Fitted model results object.

    Returns
    -------
    bool
        ``True`` when PPC sampling should run in full model gene space before
        plotting subsetting. This includes VAE-style inference and amortized
        capture subsets.
    """
    model_config = getattr(results, "model_config", None)
    inference_method = (
        str(getattr(model_config, "inference_method", "")).lower()
        if model_config is not None
        else ""
    )
    is_vae = "vae" in inference_method
    is_amortized_subset = bool(
        hasattr(results, "_uses_amortized_capture")
        and results._uses_amortized_capture()
        and getattr(results, "_original_n_genes", None) is not None
        and int(getattr(results, "_original_n_genes")) > int(results.n_genes)
    )
    return bool(is_vae or is_amortized_subset)


def _resolve_ppc_sampling_counts(results, raw_counts, aligned_counts):
    """Resolve counts matrix for PPC sampling calls.

    Parameters
    ----------
    results : object
        Fitted model results object.
    raw_counts : ndarray
        Coerced counts in caller-provided gene space.
    aligned_counts : ndarray
        Counts aligned to ``results.n_genes`` for plotting operations.

    Returns
    -------
    ndarray
        Counts used in posterior/predictive sampling.

    Raises
    ------
    ValueError
        Raised for amortized subset results when full original-gene counts are
        required but not provided.
    """
    is_amortized = bool(
        hasattr(results, "_uses_amortized_capture")
        and results._uses_amortized_capture()
    )
    is_flow_subset = bool(
        getattr(results, "_original_n_genes", None) is not None
        and int(getattr(results, "_original_n_genes")) > int(results.n_genes)
        and _has_flow_params(getattr(results, "params", None))
    )
    if not (is_amortized or is_flow_subset):
        return aligned_counts

    original_n_genes = getattr(results, "_original_n_genes", None)
    if original_n_genes is None or int(original_n_genes) == int(
        results.n_genes
    ):
        return aligned_counts

    if int(raw_counts.shape[1]) == int(original_n_genes):
        return raw_counts
    if int(aligned_counts.shape[1]) == int(original_n_genes):
        return aligned_counts
    raise ValueError(
        "PPC sampling requires full original-gene counts for amortized-capture "
        "or flow-subset results. Pass the original counts matrix with "
        f"{int(original_n_genes)} genes."
    )


def _prepare_ppc_data(
    results,
    counts,
    viz_cfg,
    *,
    counts_for_sampling=None,
    n_rows,
    n_cols,
    n_samples,
    genes=None,
    ppc_level: str = "marginal",
    map_sampling: bool = False,
):
    """Prepare gene selection and predictive samples for PPC plotting.

    Selects genes across expression bins, generates posterior
    predictive samples, and builds the gene-position mapping needed
    for histogram rendering.

    Parameters
    ----------
    results : object
        Fitted model results exposing predictive sampling.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    counts_for_sampling : array-like or None
        Count matrix used by posterior/predictive sampling calls. This can
        differ from ``counts`` when amortized subset results require original
        full-gene counts for sufficient-statistic computation.
    viz_cfg : OmegaConf or None
        Visualization configuration (used only for render options).
    n_rows : int
        Number of grid rows (already resolved).
    n_cols : int
        Number of grid columns (already resolved).
    n_samples : int
        Number of posterior predictive samples (already resolved).
    ppc_level : str, optional
        How much conditioning on observed counts enters each predictive draw
        (Laplace / applicable backends via
        ``_get_predictive_samples_for_plot``). Common values are
        ``"marginal"`` (fully generative replay; default here),
        ``"library_anchored"`` (fresh composition paired with observed
        library sizes), and ``"per_cell"`` (most conditioned on observed
        cells). Exact support depends on the results type. Ignored when
        ``map_sampling=True`` (MAP draws have no posterior to condition on).
    map_sampling : bool, default False
        If True, generate predictive samples by fixing the variational
        parameters at their MAP / posterior mean and drawing
        observations from the likelihood at that single point. This is
        a diagnostic for guide geometry: if the MAP PPC is tight but
        the full-posterior PPC is wide, the posterior is well-localized
        and the wide bands come from guide spread; if the MAP PPC is
        already wide, the issue lives in the model or likelihood
        normalization (capture, hierarchy) and a richer guide will not
        help.

    Returns
    -------
    dict
        Dictionary with keys ``n_rows``, ``n_cols``,
        ``selected_idx_sorted``, ``subset_positions``,
        ``n_genes_selected``, ``render_opts``, ``results_subset``.
    """
    render_opts = get_ppc_render_options(viz_cfg)

    console.print(
        f"[dim]Using n_rows={n_rows}, n_cols={n_cols} for PPC plot (log-spaced binning)[/dim]"
    )
    if genes is not None:
        # Explicit caller-specified selection: resolve names/indices and
        # preserve the given order (do NOT re-sort by mean) so repeated
        # calls with the same `genes` list line up panel-for-panel.
        selected_idx = _resolve_explicit_genes(genes, results, counts=counts)
        mean_counts = np.median(_coerce_counts(counts), axis=0)
        selected_idx_sorted = selected_idx
        n_genes_selected = len(selected_idx_sorted)
        console.print(
            f"[dim]Selected {n_genes_selected} caller-specified genes[/dim]"
        )
    else:
        selected_idx, mean_counts = _select_genes(
            counts,
            n_rows,
            n_cols,
            exclude_idx=_resolve_pooled_other_idx(results),
        )

        selected_means = mean_counts[selected_idx]
        sort_order = np.argsort(selected_means)
        selected_idx_sorted = selected_idx[sort_order]
        n_genes_selected = len(selected_idx_sorted)
        console.print(
            f"[dim]Selected {n_genes_selected} genes across {n_rows} expression bins[/dim]"
        )

    if counts_for_sampling is None:
        counts_for_sampling = counts

    sample_full_space = _requires_full_gene_sampling(results)
    if sample_full_space:
        sampling_results = results
        sampling_counts = counts_for_sampling
    else:
        sampling_results = results[selected_idx]
        # Flow-guided subsets need full-width counts for posterior sampling,
        # but can still sample/plot on the subset results object.
        if _has_flow_params(getattr(results, "params", None)):
            sampling_counts = counts_for_sampling
        else:
            sampling_counts = counts[:, selected_idx]

    if map_sampling:
        console.print(
            f"[dim]Generating {n_samples} MAP-anchored predictive samples...[/dim]"
        )
        # MAP-anchored draws fix the variational parameters at their
        # posterior mean and draw observations from the likelihood at
        # that point.  ``ppc_level`` is ignored: there is no posterior
        # to condition on.
        _ = _get_map_like_predictive_samples_for_plot(
            sampling_results,
            rng_key=random.PRNGKey(42),
            n_samples=n_samples,
            cell_batch_size=None,
            use_mean=True,
            counts=sampling_counts,
            store_samples=True,
            verbose=False,
        )
    else:
        console.print(
            f"[dim]Generating {n_samples} posterior predictive samples...[/dim]"
        )
        _ = _get_predictive_samples_for_plot(
            sampling_results,
            rng_key=random.PRNGKey(42),
            n_samples=n_samples,
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

    # results[selected_idx] preserves the caller-specified gene order, so
    # subset_positions maps each gene's original index to its position in
    # selected_idx (not the old sorted-original order).
    subset_positions = {
        int(gene_idx): pos for pos, gene_idx in enumerate(selected_idx)
    }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "selected_idx_sorted": selected_idx_sorted,
        "subset_positions": subset_positions,
        "n_genes_selected": n_genes_selected,
        "render_opts": render_opts,
        "results_subset": results_subset,
    }


@plot_function(
    suffix="ppc",
    save_label="PPC plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_ppc(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    n_rows=None,
    n_cols=None,
    n_genes=None,
    genes=None,
    n_samples=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
    ppc_level: str = "marginal",
    map_sampling: bool = False,
):
    """Plot posterior predictive checks for selected genes.

    Parameters
    ----------
    results : object
        Fitted result object exposing predictive sampling.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    viz_cfg : OmegaConf or None
        Visualization config containing ``ppc_opts``.  Optional in
        interactive sessions — built-in defaults are used when ``None``.
    n_rows : int, optional
        Number of grid rows.  Overrides ``viz_cfg.ppc_opts.n_rows``.
    n_cols : int, optional
        Number of grid columns.  Overrides ``viz_cfg.ppc_opts.n_cols``.
    n_genes : int, optional
        Total number of genes to display.  When given without ``n_cols``,
        derives ``n_cols = ceil(n_genes / n_rows)``.  Ignored when
        ``genes`` is provided (the count is taken from the list length).
    genes : sequence, optional
        Explicit genes to display, as gene-name strings (matched against
        ``results.var.index``) and/or integer indices into the results'
        gene axis.  When given, this overrides the default abundance-
        stratified auto-selection and the panels appear **in the order
        listed** — so two ``plot_ppc`` calls with the same ``genes`` line
        up panel-for-panel (useful for comparing models on the same
        genes).  Names excluded by ``gene_coverage`` filtering (pooled
        into ``_other``) are not selectable and raise ``ValueError``.
    n_samples : int, optional
        Number of posterior predictive samples.  Overrides
        ``viz_cfg.ppc_opts.n_samples``.
    fig : matplotlib.figure.Figure, optional
        Figure used to create/host the PPC grid.
    axes : array-like of matplotlib.axes.Axes, optional
        Explicit axis collection with exactly ``n_rows * n_cols`` axes.
    ax : matplotlib.axes.Axes, optional
        Unsupported for this multi-panel plot. Use ``fig`` or ``axes``.
    ppc_level : str, optional
        Conditioning level for posterior predictive sampling (forwarded to
        ``_get_predictive_samples_for_plot``). Default ``"marginal"`` draws a
        fully generative replay of the model. Use ``"library_anchored"`` to
        test compositional structure with observed per-cell totals, or
        ``"per_cell"`` for the most observation-conditioned draws. Support
        depends on the fitted results class (e.g. Laplace supports all three).
        Ignored when ``map_sampling=True``.
    map_sampling : bool, default False
        If True, generate predictive samples by fixing the variational
        parameters at their MAP / posterior mean and drawing
        observations from the likelihood at that single point.
        Diagnostic for guide geometry: a tight MAP PPC alongside a
        wide full-posterior PPC indicates the wide bands come from
        variational spread; a wide MAP PPC indicates the issue lives
        in the model / capture / hierarchy and a richer guide will
        not help.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.
    """
    console.print("[dim]Plotting PPC...[/dim]")
    raw_counts = _coerce_counts(counts)
    counts = _coerce_and_align_counts_to_results(
        raw_counts, results, context="plot_ppc"
    )
    counts_for_sampling = _resolve_ppc_sampling_counts(
        results, raw_counts, counts
    )
    if ax is not None:
        raise ValueError(
            "PPC requires multiple axes; provide `fig` or `axes` instead of `ax`."
        )
    # An explicit gene list fixes the panel count; let the grid size to it.
    if genes is not None:
        n_genes = len(genes)
    # Resolve grid dimensions: explicit kwargs > viz_cfg > defaults
    grid = _resolve_ppc_grid(
        n_rows=n_rows,
        n_cols=n_cols,
        n_genes=n_genes,
        n_samples=n_samples,
        viz_cfg=viz_cfg,
    )
    prep = _prepare_ppc_data(
        results,
        counts,
        viz_cfg,
        counts_for_sampling=counts_for_sampling,
        n_rows=grid["n_rows"],
        n_cols=grid["n_cols"],
        n_samples=grid["n_samples"],
        genes=genes,
        ppc_level=ppc_level,
        map_sampling=map_sampling,
    )
    n_rows = prep["n_rows"]
    n_cols = prep["n_cols"]
    selected_idx_sorted = prep["selected_idx_sorted"]
    subset_positions = prep["subset_positions"]
    n_genes_selected = prep["n_genes_selected"]
    render_opts = prep["render_opts"]
    results_subset = prep["results_subset"]
    gene_names = _get_gene_names(results)

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
            "[cyan]Plotting PPC panels...", total=n_genes_selected
        )

        for i, panel_ax in enumerate(axes_flat):
            if i >= n_genes_selected:
                panel_ax.axis("off")
                continue

            gene_idx = selected_idx_sorted[i]
            subset_pos = subset_positions[gene_idx]
            true_counts = counts[:, gene_idx]
            max_bin = compute_adaptive_max_bin(true_counts, render_opts)

            credible_regions = scribe.stats.compute_histogram_credible_regions(
                results_subset.predictive_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
                max_bin=max_bin,
            )

            hist_results = np.histogram(
                true_counts, bins=credible_regions["bin_edges"], density=True
            )
            # Plot style is selected adaptively: stairs for moderate bin counts
            # and line/fill for very large bin counts.
            render_meta = plot_histogram_credible_regions_adaptive(
                panel_ax,
                credible_regions,
                cmap="Blues",
                alpha=0.5,
                max_bin=max_bin,
                render_opts=render_opts,
            )
            plot_observed_histogram_adaptive(
                panel_ax,
                hist_results,
                max_bin=max_bin,
                render_meta=render_meta,
                label="data",
                color="black",
            )

            panel_ax.set_xlabel("counts")
            panel_ax.set_ylabel("frequency")
            actual_mean_expr = np.mean(counts[:, gene_idx])
            mean_expr_formatted = f"{actual_mean_expr:.2f}"
            title = f"$\\langle U \\rangle = {mean_expr_formatted}$"
            if gene_names is not None:
                title = f"{gene_names[gene_idx]}\n{title}"
            panel_ax.set_title(title, fontsize=8)

            progress.update(task, advance=1)

    fig.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    del results_subset
    return fig, axes_flat, n_rows * n_cols

"""Mixture model PPC plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist

from ._common import (
    console,
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from ._interactive import (
    PlotContext,
    PlotResultCollection,
    _create_or_validate_grid_axes,
    _create_or_validate_single_axis,
    _finalize_figure,
    _resolve_ppc_grid,
    plot_function,
)
from .config import _get_config_values
from .dispatch import (
    _get_map_estimates_for_plot,
    _get_map_like_predictive_samples_for_plot,
    _get_cell_assignment_probabilities_for_plot,
)
from .gene_selection import _coerce_counts, _get_gene_names
from .ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    plot_histogram_credible_regions_adaptive,
    plot_observed_histogram_adaptive,
)


def _select_divergent_genes(results, counts, n_rows, n_cols):
    """Select genes with highest divergence across components, binned by expression."""
    counts = _coerce_counts(counts)
    parameterization = results.model_config.parameterization
    if parameterization in ["linked", "mean_prob", "odds_ratio", "mean_odds"]:
        param_name = "mu"
    else:
        param_name = "r"
    map_estimates = _get_map_estimates_for_plot(
        results,
        counts=counts,
        use_mean=False,
        targets=[param_name],
    )

    param = map_estimates.get(param_name)
    if param is None:
        raise ValueError(
            f"Parameter '{param_name}' not found in MAP estimates. "
            f"Available: {list(map_estimates.keys())}"
        )

    param_clamped = jnp.clip(param, a_min=1e-10)
    lfc_range = np.array(
        jnp.log(jnp.max(param_clamped, axis=0))
        - jnp.log(jnp.min(param_clamped, axis=0))
    )

    counts_np = np.array(counts)
    median_counts = np.median(counts_np, axis=0)
    nonzero_idx = np.where(median_counts > 0)[0]

    if len(nonzero_idx) == 0:
        return np.array([], dtype=int), np.array([])

    sorted_order = np.argsort(median_counts[nonzero_idx])
    sorted_idx = nonzero_idx[sorted_order]
    sorted_medians = median_counts[sorted_idx]
    sorted_lfc = lfc_range[sorted_idx]

    min_expr = sorted_medians[0]
    max_expr = sorted_medians[-1]
    min_expr_safe = max(min_expr, 0.1)
    bin_edges = np.logspace(
        np.log10(min_expr_safe), np.log10(max_expr), num=n_rows + 1
    )
    bin_edges[0] = min_expr

    selected_set = set()
    selected_by_bin = []

    for i in range(n_rows):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        if i == n_rows - 1:
            in_bin = (sorted_medians >= bin_start) & (sorted_medians <= bin_end)
        else:
            in_bin = (sorted_medians >= bin_start) & (sorted_medians < bin_end)

        bin_indices = np.where(in_bin)[0]
        bin_selected = []

        if len(bin_indices) > 0:
            bin_lfc = sorted_lfc[bin_indices]
            if len(bin_indices) <= n_cols:
                bin_selected = list(bin_indices)
            else:
                top_lfc_order = np.argsort(bin_lfc)[::-1][:n_cols]
                bin_selected = list(bin_indices[top_lfc_order])

        selected_by_bin.append(bin_selected)
        selected_set.update(bin_selected)

    all_indices = set(range(len(sorted_idx)))
    unselected_indices = np.array(sorted(list(all_indices - selected_set)))

    if len(unselected_indices) > 0:
        unselected_lfc = sorted_lfc[unselected_indices]
        unselected_medians = sorted_medians[unselected_indices]

        for i in range(n_rows):
            bin_selected = selected_by_bin[i]
            needed = n_cols - len(bin_selected)

            if needed > 0 and len(unselected_indices) > 0:
                bin_end = bin_edges[i + 1]
                candidates_mask = unselected_medians <= bin_end
                candidates = unselected_indices[candidates_mask]
                candidate_lfc = unselected_lfc[candidates_mask]

                if len(candidates) == 0:
                    candidates = unselected_indices
                    candidate_lfc = unselected_lfc

                n_to_add = min(needed, len(candidates))
                top_lfc_order = np.argsort(candidate_lfc)[::-1][:n_to_add]
                to_add = candidates[top_lfc_order]

                bin_selected.extend(list(to_add))
                selected_set.update(to_add)

                mask = ~np.isin(unselected_indices, to_add)
                unselected_indices = unselected_indices[mask]
                unselected_lfc = unselected_lfc[mask]
                unselected_medians = unselected_medians[mask]

            selected_by_bin[i] = bin_selected

    final_selected_sorted = []
    final_lfc = []
    for bin_selected in selected_by_bin:
        for idx in bin_selected:
            final_selected_sorted.append(sorted_idx[idx])
            final_lfc.append(sorted_lfc[idx])

    return np.array(final_selected_sorted), np.array(final_lfc)


def _get_component_ppc_samples(
    results,
    component_idx,
    n_samples,
    rng_key,
    cell_batch_size=500,
    verbose=True,
    counts=None,
):
    """Generate PPC samples for a specific mixture component."""
    # Require only core NB keys; fetch optional technical-noise params
    # conditionally so non-ZI / non-VCP models continue to work.
    component_targets = ["r", "p"]
    if bool(getattr(results.model_config, "uses_zero_inflation", False)):
        component_targets.append("gate")
    if bool(getattr(results.model_config, "uses_variable_capture", False)):
        component_targets.append("p_capture")
    map_estimates = _get_map_estimates_for_plot(
        results,
        counts=counts,
        targets=component_targets,
    )

    r_all = map_estimates["r"]
    p_all = map_estimates["p"]

    r_k = r_all[component_idx]
    n_genes = r_k.shape[0]

    if jnp.ndim(p_all) == 0:
        p_k = p_all
    elif jnp.ndim(p_all) == 1:
        p_k = p_all[component_idx]
    else:
        p_k = p_all[component_idx]

    gate_k = None
    if "gate" in map_estimates:
        gate_all = map_estimates["gate"]
        if jnp.ndim(gate_all) > 1:
            gate_k = gate_all[component_idx]
        else:
            gate_k = gate_all

    p_capture = map_estimates.get("p_capture")
    has_vcp = p_capture is not None
    has_gate = gate_k is not None

    n_cells = results.n_cells
    all_samples = []
    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size

    if verbose and n_batches > 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing cell batches...", total=n_batches
            )

            for batch_idx in range(n_batches):
                start = batch_idx * cell_batch_size
                end = min(start + cell_batch_size, n_cells)
                batch_size = end - start

                rng_key, batch_key = random.split(rng_key)

                if has_vcp:
                    p_capture_batch = p_capture[start:end]
                    p_capture_reshaped = p_capture_batch[:, None]
                    p_effective = (
                        p_k
                        * p_capture_reshaped
                        / (1 - p_k * (1 - p_capture_reshaped))
                    )
                else:
                    p_effective = p_k

                nb_dist = dist.NegativeBinomialProbs(r_k, p_effective)

                if has_gate:
                    sample_dist = dist.ZeroInflatedDistribution(
                        nb_dist, gate=gate_k
                    )
                else:
                    sample_dist = nb_dist

                if has_vcp:
                    batch_samples = sample_dist.sample(batch_key, (n_samples,))
                else:
                    batch_samples = sample_dist.sample(
                        batch_key, (n_samples, batch_size)
                    )

                all_samples.append(batch_samples)
                progress.update(task, advance=1)
    else:
        for batch_idx in range(n_batches):
            start = batch_idx * cell_batch_size
            end = min(start + cell_batch_size, n_cells)
            batch_size = end - start

            rng_key, batch_key = random.split(rng_key)

            if has_vcp:
                p_capture_batch = p_capture[start:end]
                p_capture_reshaped = p_capture_batch[:, None]
                p_effective = (
                    p_k
                    * p_capture_reshaped
                    / (1 - p_k * (1 - p_capture_reshaped))
                )
            else:
                p_effective = p_k

            nb_dist = dist.NegativeBinomialProbs(r_k, p_effective)

            if has_gate:
                sample_dist = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate_k
                )
            else:
                sample_dist = nb_dist

            if has_vcp:
                batch_samples = sample_dist.sample(batch_key, (n_samples,))
            else:
                batch_samples = sample_dist.sample(
                    batch_key, (n_samples, batch_size)
                )

            all_samples.append(batch_samples)

    samples = jnp.concatenate(all_samples, axis=1)
    return samples


def _plot_ppc_figure(
    predictive_samples,
    counts,
    selected_idx,
    n_rows,
    n_cols,
    title,
    figs_dir,
    fname,
    output_format="png",
    cmap="Blues",
    render_opts=None,
    fig=None,
    axes=None,
    save=True,
    show=None,
    close=True,
    gene_names=None,
):
    """Plot a PPC figure in the standard format."""
    import scribe

    if render_opts is None:
        render_opts = {
            "hist_max_bin_quantile": 0.99,
            "hist_max_bin_floor": 10,
            "render_auto_line_bin_threshold": 1000,
            "render_line_target_points": 200,
            "render_line_interpolate": True,
        }

    n_genes_selected = len(selected_idx)

    counts_np = np.array(counts)
    selected_means = np.array(
        [np.mean(counts_np[:, idx]) for idx in selected_idx]
    )
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]

    # results[selected_idx] preserves the caller-specified gene order, so
    # subset_positions must map each gene's original index to its position in
    # selected_idx (not the old sorted-original order).
    subset_positions = {
        int(gene_idx): pos for pos, gene_idx in enumerate(selected_idx)
    }

    fig, _, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_rows,
        n_cols=n_cols,
        fig=fig,
        axes=axes,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
    )

    for i, ax in enumerate(axes_flat):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        gene_idx = selected_idx_sorted[i]
        subset_pos = subset_positions[gene_idx]
        true_counts = counts_np[:, gene_idx]
        max_bin = compute_adaptive_max_bin(true_counts, render_opts)

        credible_regions = scribe.stats.compute_histogram_credible_regions(
            predictive_samples[:, :, subset_pos],
            credible_regions=[95, 68, 50],
            max_bin=max_bin,
        )

        hist_results = np.histogram(
            true_counts, bins=credible_regions["bin_edges"], density=True
        )

        render_meta = plot_histogram_credible_regions_adaptive(
            ax,
            credible_regions,
            cmap=cmap,
            alpha=0.5,
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

        ax.set_xlabel("counts")
        ax.set_ylabel("frequency")
        actual_mean_expr = np.mean(counts_np[:, gene_idx])
        mean_expr_formatted = f"{actual_mean_expr:.2f}"
        panel_title = f"$\\langle U \\rangle = {mean_expr_formatted}$"
        if gene_names is not None:
            panel_title = f"{gene_names[gene_idx]}\n{panel_title}"
        ax.set_title(panel_title, fontsize=8)

    fig.tight_layout()
    fig.suptitle(title, y=1.02)

    # _plot_ppc_figure always creates its own figure (no caller injection)
    return _finalize_figure(
        fig=fig,
        axes=axes_flat,
        n_panels=n_rows * n_cols,
        save=save,
        show=show,
        close=close,
        figs_dir=figs_dir,
        filename=f"{fname}.{output_format}",
        save_kwargs={"bbox_inches": "tight"},
        save_label=title,
        _fig_owned=(fig is None and axes is None),
    )


def _plot_ppc_comparison_figure(
    mixture_samples,
    component_samples_list,
    counts,
    selected_idx,
    n_rows,
    n_cols,
    figs_dir,
    fname,
    output_format="png",
    component_cmaps=None,
    render_opts=None,
    fig=None,
    axes=None,
    save=True,
    show=False,
    close=True,
    gene_names=None,
):
    """Plot comparison figure with mixture and all component PPCs overlaid."""
    import scribe

    if render_opts is None:
        render_opts = {
            "hist_max_bin_quantile": 0.99,
            "hist_max_bin_floor": 10,
            "render_auto_line_bin_threshold": 1000,
            "render_line_target_points": 200,
            "render_line_interpolate": True,
        }

    if component_cmaps is None:
        component_cmaps = [
            "Greens",
            "Purples",
            "Reds",
            "Oranges",
            "YlOrBr",
            "BuGn",
        ]

    n_genes_selected = len(selected_idx)
    n_components = len(component_samples_list)

    counts_np = np.array(counts)
    selected_means = np.array(
        [np.mean(counts_np[:, idx]) for idx in selected_idx]
    )
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]

    # results[selected_idx] preserves the caller-specified gene order, so
    # subset_positions must map each gene's original index to its position in
    # selected_idx (not the old sorted-original order).
    subset_positions = {
        int(gene_idx): pos for pos, gene_idx in enumerate(selected_idx)
    }

    fig, _, axes_flat = _create_or_validate_grid_axes(
        n_rows=n_rows,
        n_cols=n_cols,
        fig=fig,
        axes=axes,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
    )

    for i, ax in enumerate(axes_flat):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        gene_idx = selected_idx_sorted[i]
        subset_pos = subset_positions[gene_idx]
        true_counts = counts_np[:, gene_idx]

        x_max = compute_adaptive_max_bin(true_counts, render_opts)
        obs_bins = np.arange(0, x_max + 2)
        hist_results = np.histogram(true_counts, bins=obs_bins, density=True)
        y_max = np.max(hist_results[0]) * 1.1

        mixture_cr = scribe.stats.compute_histogram_credible_regions(
            mixture_samples[:, :, subset_pos],
            credible_regions=[95, 68, 50],
            max_bin=x_max,
        )

        for k, comp_samples in enumerate(component_samples_list):
            comp_cr = scribe.stats.compute_histogram_credible_regions(
                comp_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
                max_bin=x_max,
            )
            cmap = component_cmaps[k % len(component_cmaps)]
            plot_histogram_credible_regions_adaptive(
                ax,
                comp_cr,
                cmap=cmap,
                alpha=0.4,
                max_bin=x_max,
                render_opts=render_opts,
            )

        render_meta = plot_histogram_credible_regions_adaptive(
            ax,
            mixture_cr,
            cmap="Blues",
            alpha=0.3,
            max_bin=x_max,
            render_opts=render_opts,
        )
        plot_observed_histogram_adaptive(
            ax,
            hist_results,
            max_bin=x_max,
            render_meta=render_meta,
            label="data",
            color="black",
            linewidth=1.5,
        )

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

        ax.set_xlabel("counts")
        ax.set_ylabel("frequency")
        actual_mean_expr = np.mean(counts_np[:, gene_idx])
        mean_expr_formatted = f"{actual_mean_expr:.2f}"
        panel_title = f"$\\langle U \\rangle = {mean_expr_formatted}$"
        if gene_names is not None:
            panel_title = f"{gene_names[gene_idx]}\n{panel_title}"
        ax.set_title(panel_title, fontsize=8)

    fig.tight_layout()
    fig.suptitle("PPC Comparison: Mixture vs Components", y=1.02)

    return _finalize_figure(
        fig=fig,
        axes=axes_flat,
        n_panels=n_rows * n_cols,
        save=save,
        show=show,
        close=close,
        figs_dir=figs_dir,
        filename=f"{fname}.{output_format}",
        save_kwargs={"bbox_inches": "tight"},
        save_label="PPC comparison",
        _fig_owned=(fig is None and axes is None),
    )


def _prepare_mixture_ppc_data(
    results, counts, viz_cfg, *, n_rows, n_cols, n_samples
):
    """Prepare data for mixture PPC visualization.

    Selects high-CV genes, generates mixture and per-component PPC
    samples, and computes MAP cell-to-component assignments.  All
    computation is backend-agnostic (no matplotlib).

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    viz_cfg : OmegaConf or None
        Visualization configuration (used for render and batch options).
    n_rows : int
        Number of grid rows (already resolved).
    n_cols : int
        Number of grid columns (already resolved).
    n_samples : int
        Number of posterior predictive samples (already resolved).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - ``n_components``: int
        - ``n_rows``, ``n_cols``: grid dimensions
        - ``n_samples``: sample count
        - ``top_gene_indices``: selected gene indices
        - ``top_lfc``: log-fold-changes
        - ``mixture_samples_np``: mixture PPC samples array
        - ``assignments``: MAP cell-to-component assignments
        - ``component_samples_list``: list of per-component sample arrays
        - ``render_opts``: PPC rendering options
        Returns ``None`` when the model is not a mixture.
    """
    from jax import random

    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping mixture PPC plot...[/yellow]"
        )
        return None

    render_opts = get_ppc_render_options(viz_cfg)
    ppc_opts = viz_cfg.get("ppc_opts", {}) if viz_cfg is not None else {}
    mixture_ppc_opts = (
        viz_cfg.get("mixture_ppc_opts", {}) if viz_cfg is not None else {}
    )
    # Use explicit assignment batching when provided to avoid OOM in
    # cell-to-component probability computation on large datasets.
    assignment_batch_size = mixture_ppc_opts.get(
        "assignment_batch_size",
        mixture_ppc_opts.get("batch_size", ppc_opts.get("batch_size", 512)),
    )
    if assignment_batch_size is not None and assignment_batch_size <= 0:
        assignment_batch_size = None
    console.print(
        f"[dim]Selecting high-CV genes from {n_rows} expression bins "
        f"({n_cols} genes/bin) across {n_components} components...[/dim]"
    )

    top_gene_indices, top_lfc = _select_divergent_genes(
        results, counts, n_rows, n_cols
    )
    top_gene_indices = np.array(top_gene_indices)
    n_genes_to_plot = len(top_gene_indices)

    console.print(
        f"[dim]Selected {n_genes_to_plot} genes. Top log-fold-changes:[/dim] "
        f"{np.array(top_lfc[:5])}"
    )

    results_subset = results[top_gene_indices]

    rng_key = random.PRNGKey(42)

    console.print(
        f"[dim]Generating mixture PPC samples ({n_samples} samples)...[/dim]"
    )
    rng_key, subkey = random.split(rng_key)
    mixture_samples = _get_map_like_predictive_samples_for_plot(
        results_subset,
        rng_key=subkey,
        n_samples=n_samples,
        cell_batch_size=500,
        store_samples=False,
        verbose=True,
        counts=counts,
    )

    mixture_samples_np = np.array(mixture_samples)
    del mixture_samples

    console.print("[dim]Computing MAP cell-to-component assignments...[/dim]")
    if assignment_batch_size is not None:
        console.print(
            f"[dim]Using assignment batch_size={assignment_batch_size}[/dim]"
        )
    assignment_probs = _get_cell_assignment_probabilities_for_plot(
        results, counts=counts, batch_size=assignment_batch_size
    )
    assignments = np.argmax(assignment_probs, axis=1)

    component_samples_list = []
    for k in range(n_components):
        console.print(
            f"[dim]Generating Component {k+1} PPC samples ({n_samples} samples)...[/dim]"
        )
        rng_key, subkey = random.split(rng_key)

        component_samples = _get_component_ppc_samples(
            results_subset,
            component_idx=k,
            n_samples=n_samples,
            rng_key=subkey,
            cell_batch_size=500,
            verbose=True,
            counts=counts,
        )

        component_samples_np = np.array(component_samples)
        component_samples_list.append(component_samples_np)
        del component_samples

    return {
        "n_components": n_components,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_samples": n_samples,
        "top_gene_indices": top_gene_indices,
        "top_lfc": top_lfc,
        "mixture_samples_np": mixture_samples_np,
        "assignments": assignments,
        "component_samples_list": component_samples_list,
        "render_opts": render_opts,
    }


def plot_mixture_ppc(
    results,
    counts,
    figs_dir=None,
    cfg=None,
    viz_cfg=None,
    *,
    n_rows=None,
    n_cols=None,
    n_genes=None,
    n_samples=None,
    fig=None,
    axes=None,
    ax=None,
    save=None,
    show=None,
    close=None,
):
    """Plot PPC for mixture models showing genes with highest divergence.

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    figs_dir : str, optional
        Output directory used when ``save`` resolves to ``True``.
    cfg : OmegaConf, optional
        Run configuration for filename generation.
    viz_cfg : OmegaConf or None
        Visualization config.  Optional in interactive sessions —
        built-in defaults are used when ``None``.
    n_rows : int, optional
        Number of grid rows.  Overrides ``viz_cfg.mixture_ppc_opts.n_rows``.
    n_cols : int, optional
        Number of grid columns.  Overrides ``viz_cfg.mixture_ppc_opts.n_cols``.
    n_genes : int, optional
        Total number of genes to display.  When given without ``n_cols``,
        derives ``n_cols = ceil(n_genes / n_rows)``.
    n_samples : int, optional
        Number of posterior predictive samples.  Overrides
        ``viz_cfg.mixture_ppc_opts.n_samples``.

    Returns
    -------
    PlotResultCollection or None
        Wrapped result containing figure payloads.
    """
    console.print(
        "[dim]Plotting mixture model PPC (genes with highest CV across components)...[/dim]"
    )
    if ax is not None:
        raise ValueError(
            "Mixture PPC requires multiple axes; provide `fig` or `axes`."
        )
    ctx = PlotContext.from_kwargs(
        figs_dir=figs_dir,
        cfg=cfg,
        viz_cfg=viz_cfg,
        fig=fig,
        ax=ax,
        axes=axes,
        save=save,
        show=show,
        close=close,
    )

    counts = _coerce_counts(counts)

    # Resolve grid dimensions: explicit kwargs > viz_cfg > defaults
    # Mixture PPC defaults to 1500 samples (heavier computation).
    grid = _resolve_ppc_grid(
        n_rows=n_rows,
        n_cols=n_cols,
        n_genes=n_genes,
        n_samples=n_samples,
        viz_cfg=viz_cfg,
        opts_key="mixture_ppc_opts",
        default_samples=1500,
    )

    prepared = _prepare_mixture_ppc_data(
        results,
        counts,
        viz_cfg,
        n_rows=grid["n_rows"],
        n_cols=grid["n_cols"],
        n_samples=grid["n_samples"],
    )
    if prepared is None:
        return

    n_components = prepared["n_components"]
    n_rows = prepared["n_rows"]
    n_cols = prepared["n_cols"]
    top_gene_indices = prepared["top_gene_indices"]
    mixture_samples_np = prepared["mixture_samples_np"]
    assignments = prepared["assignments"]
    component_samples_list = prepared["component_samples_list"]
    render_opts = prepared["render_opts"]
    gene_names = _get_gene_names(results)
    mixture_ppc_opts = (
        viz_cfg.get("mixture_ppc_opts", {}) if viz_cfg is not None else {}
    )

    if ctx.save:
        config_vals = _get_config_values(cfg, results=results)
        base_fname = (
            f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
            f"{config_vals['model_type'].replace('_', '-')}_"
            f"{config_vals['n_components']:02d}components_"
            f"{config_vals['run_size_token']}"
        )
    else:
        base_fname = "mixture"
    output_format = ctx.output_format

    figure_payloads = []

    fig_payload = _plot_ppc_figure(
        predictive_samples=mixture_samples_np,
        counts=counts,
        selected_idx=top_gene_indices,
        n_rows=n_rows,
        n_cols=n_cols,
        title="Mixture PPC (High CV Genes)",
        figs_dir=figs_dir,
        fname=f"{base_fname}_mixture_ppc",
        output_format=output_format,
        render_opts=render_opts,
        fig=fig,
        axes=axes,
        save=ctx.save,
        show=ctx.show,
        close=ctx.close,
        gene_names=gene_names,
    )
    figure_payloads.append(fig_payload)

    component_cmaps = ["Greens", "Purples", "Reds", "Oranges", "YlOrBr", "BuGn"]
    min_cells_for_info = mixture_ppc_opts.get("min_cells_for_info", 20)
    all_component_samples = component_samples_list
    counts_np = np.array(counts)

    for k in range(n_components):
        component_samples_np = component_samples_list[k]

        cell_mask_k = assignments == k
        n_cells_component = int(np.sum(cell_mask_k))
        if n_cells_component == 0:
            console.print(
                f"[yellow]Skipping Component {k+1} PPC (no assigned cells)[/yellow]"
            )
            del component_samples
            continue
        if n_cells_component < min_cells_for_info:
            console.print(
                f"[yellow]Component {k+1} has only {n_cells_component} assigned "
                f"cells; PPC may be noisy[/yellow]"
            )

        component_samples_subset = component_samples_np[:, cell_mask_k, :]
        counts_component = counts_np[cell_mask_k, :]

        cmap = component_cmaps[k % len(component_cmaps)]

        fig_payload = _plot_ppc_figure(
            predictive_samples=component_samples_subset,
            counts=counts_component,
            selected_idx=top_gene_indices,
            n_rows=n_rows,
            n_cols=n_cols,
            title=(
                f"Component {k+1} PPC (High CV Genes, "
                f"{n_cells_component} assigned cells)"
            ),
            figs_dir=figs_dir,
            fname=f"{base_fname}_component{k+1}_ppc",
            output_format=output_format,
            cmap=cmap,
            render_opts=render_opts,
            save=ctx.save,
            show=ctx.show,
            close=ctx.close,
            gene_names=gene_names,
        )
        figure_payloads.append(fig_payload)

    if n_components <= 2:
        console.print("[dim]Generating combined comparison plot...[/dim]")
        fig_payload = _plot_ppc_comparison_figure(
            mixture_samples=mixture_samples_np,
            component_samples_list=all_component_samples,
            counts=counts,
            selected_idx=top_gene_indices,
            n_rows=n_rows,
            n_cols=n_cols,
            figs_dir=figs_dir,
            fname=f"{base_fname}_ppc_comparison",
            output_format=output_format,
            component_cmaps=component_cmaps,
            render_opts=render_opts,
            save=ctx.save,
            show=ctx.show,
            close=ctx.close,
            gene_names=gene_names,
        )
        figure_payloads.append(fig_payload)
        n_plots = 2 + n_components
    else:
        console.print(
            f"[yellow]Skipping comparison plot ({n_components} components "
            f"> 2; per-component plots are generated instead)[/yellow]"
        )
        n_plots = 1 + n_components

    del mixture_samples_np, all_component_samples

    console.print(
        f"[green]✓[/green] [dim]Generated {n_plots} mixture PPC plots[/dim]"
    )
    if figure_payloads:
        return PlotResultCollection(figure_payloads)
    return None


def _resolve_weight_fractions_for_composition(
    mixing_weights, n_components, dataset_indices=None
):
    """Convert MAP mixing weights into a normalized component-fraction vector.

    Parameters
    ----------
    mixing_weights : array-like
        MAP ``mixing_weights`` parameter from fitted model.
    n_components : int
        Number of mixture components.
    dataset_indices : array-like, optional
        Per-cell dataset ids used to build a dataset-size-weighted aggregate
        when ``mixing_weights`` is dataset-specific with shape ``(D, K)``.

    Returns
    -------
    np.ndarray or None
        Normalized component fractions with shape ``(K,)`` or ``None`` if
        ``mixing_weights`` is missing.
    """
    if mixing_weights is None:
        return None

    w = np.asarray(mixing_weights, dtype=float)
    if w.ndim == 0:
        return None

    if w.ndim == 1:
        fractions = w
    elif w.ndim == 2:
        # Prefer (D, K) orientation. If not already in that orientation,
        # transpose when that produces (D, K).
        if w.shape[-1] == n_components:
            w_dk = w
        elif w.shape[0] == n_components:
            w_dk = w.T
        else:
            # Fall back to flatten+truncate behavior only when no axis
            # matches K; this keeps plots robust to unexpected shapes.
            fractions = w.reshape(-1)
            fractions = fractions[:n_components]
            fractions = np.clip(fractions, a_min=0.0, a_max=None)
            return fractions / max(np.sum(fractions), 1e-12)

        if dataset_indices is not None:
            ds_idx = np.asarray(dataset_indices)
            if ds_idx.ndim == 1 and ds_idx.size > 0:
                n_datasets = w_dk.shape[0]
                counts = np.bincount(ds_idx.astype(int), minlength=n_datasets)
                ds_weights = counts / max(int(np.sum(counts)), 1)
                fractions = np.sum(w_dk * ds_weights[:, None], axis=0)
            else:
                fractions = np.mean(w_dk, axis=0)
        else:
            fractions = np.mean(w_dk, axis=0)
    else:
        fractions = w.reshape(-1)

    fractions = np.asarray(fractions, dtype=float).reshape(-1)
    if fractions.shape[0] != n_components:
        fractions = fractions[:n_components]
    fractions = np.clip(fractions, a_min=0.0, a_max=None)
    return fractions / max(np.sum(fractions), 1e-12)


def _reconstruct_label_map_for_composition(cell_labels, component_order=None):
    """Build fallback label-to-component mapping for composition plots.

    Parameters
    ----------
    cell_labels : array-like
        Per-cell annotation labels.
    component_order : sequence of str, optional
        Explicit component order from configuration.

    Returns
    -------
    dict
        Mapping from label string to component index.
    """
    import pandas as pd

    annotations = pd.Series(cell_labels)
    labels = annotations.dropna()

    if component_order is not None:
        return {str(label): idx for idx, label in enumerate(component_order)}

    unique_sorted = sorted(str(label) for label in labels.unique())
    return {label: idx for idx, label in enumerate(unique_sorted)}


def _resolve_label_map_for_composition(results, cell_labels, cfg):
    """Resolve label-to-component mapping with training-time priority.

    The preferred source is ``results._label_map`` because it reflects the
    mapping actually used during fitting (including any filtering logic).
    """
    stored_label_map = getattr(results, "_label_map", None)
    if isinstance(stored_label_map, dict) and len(stored_label_map) > 0:
        return {
            str(label): int(idx)
            for label, idx in stored_label_map.items()
            if idx is not None
        }

    component_order = cfg.get("annotation_component_order", None)
    return _reconstruct_label_map_for_composition(
        cell_labels=cell_labels, component_order=component_order
    )


def _prepare_mixture_composition_data(
    results, counts, cfg, viz_cfg, cell_labels=None
):
    """Prepare composition data for mixture component visualization.

    Resolves mixing weights, assignment probabilities, and (optionally)
    per-label observed vs predicted fractions.  All computation is
    backend-agnostic.

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : array-like
        Observed count matrix.
    cfg : object
        Run configuration.
    viz_cfg : object
        Visualization configuration.
    cell_labels : array-like or None
        Per-cell annotation labels.

    Returns
    -------
    dict or None
        When ``cell_labels`` is provided, returns:
        - ``mode``: ``"labeled"``
        - ``labels_by_component``: list of label strings
        - ``observed_fracs``: array of observed fractions
        - ``assigned_fracs_by_label``: array of assignment fractions
        - ``weight_fracs_by_label``: array or None
        - ``n_components``: int
        When ``cell_labels`` is None, returns:
        - ``mode``: ``"global"``
        - ``component_fractions``: array
        - ``hard_counts``: array
        - ``n_components``: int
        Returns ``None`` when the model is not a mixture.
    """
    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping composition plot...[/yellow]"
        )
        return None

    try:
        map_estimates = _get_map_estimates_for_plot(
            results, counts=counts, targets=["mixing_weights"]
        )
        mixing_weights = map_estimates.get("mixing_weights")
    except ValueError:
        mixing_weights = None
    dataset_indices = getattr(results, "_dataset_indices", None)
    weight_fractions = _resolve_weight_fractions_for_composition(
        mixing_weights=mixing_weights,
        n_components=n_components,
        dataset_indices=dataset_indices,
    )

    if cell_labels is not None:
        ppc_opts = viz_cfg.get("ppc_opts", {}) if viz_cfg is not None else {}
        mixture_ppc_opts = (
            viz_cfg.get("mixture_ppc_opts", {}) if viz_cfg is not None else {}
        )
        composition_opts = (
            viz_cfg.get("mixture_composition_opts", {})
            if viz_cfg is not None
            else {}
        )
        assignment_batch_size = composition_opts.get(
            "assignment_batch_size",
            mixture_ppc_opts.get(
                "assignment_batch_size",
                mixture_ppc_opts.get(
                    "batch_size", ppc_opts.get("batch_size", 512)
                ),
            ),
        )
        if assignment_batch_size is not None and assignment_batch_size <= 0:
            assignment_batch_size = None
        if assignment_batch_size is not None:
            console.print(
                f"[dim]Using assignment batch_size={assignment_batch_size}[/dim]"
            )
        assignment_probs = _get_cell_assignment_probabilities_for_plot(
            results,
            counts=counts,
            batch_size=assignment_batch_size,
            use_mean=False,
        )
        assignments = np.argmax(assignment_probs, axis=1)
        assignment_counts = np.bincount(assignments, minlength=n_components)
        assigned_fractions = assignment_counts / max(
            1, int(np.sum(assignment_counts))
        )

        annotations = np.asarray(cell_labels, dtype=object)
        label_map = _resolve_label_map_for_composition(
            results=results,
            cell_labels=annotations,
            cfg=cfg,
        )
        labels_by_component = sorted(
            label_map.keys(), key=lambda x: label_map[x]
        )
        labels_by_component = [
            label
            for label in labels_by_component
            if 0 <= int(label_map[label]) < n_components
        ]

        observed_fracs = np.array(
            [np.mean(annotations == label) for label in labels_by_component]
        )
        assigned_fracs_by_label = np.array(
            [
                assigned_fractions[int(label_map[label])]
                for label in labels_by_component
            ]
        )
        weight_fracs_by_label = None
        if weight_fractions is not None:
            weight_fracs_by_label = np.array(
                [
                    weight_fractions[int(label_map[label])]
                    for label in labels_by_component
                ]
            )

        return {
            "mode": "labeled",
            "labels_by_component": labels_by_component,
            "observed_fracs": observed_fracs,
            "assigned_fracs_by_label": assigned_fracs_by_label,
            "weight_fracs_by_label": weight_fracs_by_label,
            "n_components": n_components,
        }

    if weight_fractions is not None:
        component_fractions = weight_fractions
        hard_counts = np.rint(
            component_fractions * int(results.n_cells)
        ).astype(int)
    else:
        ppc_opts = viz_cfg.get("ppc_opts", {}) if viz_cfg is not None else {}
        mixture_ppc_opts = (
            viz_cfg.get("mixture_ppc_opts", {}) if viz_cfg is not None else {}
        )
        composition_opts = (
            viz_cfg.get("mixture_composition_opts", {})
            if viz_cfg is not None
            else {}
        )
        assignment_batch_size = composition_opts.get(
            "assignment_batch_size",
            mixture_ppc_opts.get(
                "assignment_batch_size",
                mixture_ppc_opts.get(
                    "batch_size", ppc_opts.get("batch_size", 512)
                ),
            ),
        )
        if assignment_batch_size is not None and assignment_batch_size <= 0:
            assignment_batch_size = None
        if assignment_batch_size is not None:
            console.print(
                f"[yellow]mixing_weights missing; using assignment "
                f"batch_size={assignment_batch_size} fallback[/yellow]"
            )
        assignment_probs = _get_cell_assignment_probabilities_for_plot(
            results,
            counts=counts,
            batch_size=assignment_batch_size,
            use_mean=False,
        )
        assignments = np.argmax(assignment_probs, axis=1)
        hard_counts = np.bincount(assignments, minlength=n_components)
        component_fractions = hard_counts / max(1, int(np.sum(hard_counts)))

    return {
        "mode": "global",
        "component_fractions": component_fractions,
        "hard_counts": hard_counts,
        "n_components": n_components,
    }


@plot_function(
    suffix="mixture_composition",
    save_label="mixture composition",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_mixture_composition(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    cell_labels=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot MAP component composition for mixture models.

    This figure summarizes each component's composition as the fraction of cells
    assigned to that component under MAP-like assignment probabilities.

    When ``cell_labels`` are provided, this plot switches to a side-by-side
    comparison per label:

    - observed fraction of each label in the dataset
    - predicted fraction from model assignments mapped to that label's component

    Returns
    -------
    PlotResult or None
        Wrapped result containing the figure, axes, and metadata.
    """
    console.print("[dim]Plotting mixture component composition...[/dim]")

    prepared = _prepare_mixture_composition_data(
        results, counts, ctx.cfg, viz_cfg, cell_labels=cell_labels
    )
    if prepared is None:
        return None

    n_components = prepared["n_components"]

    fig, ax = _create_or_validate_single_axis(
        fig=fig,
        ax=ax,
        axes=axes,
        figsize=(max(6.0, 1.1 * n_components + 2.0), 4.0),
    )

    if prepared["mode"] == "labeled":
        labels_by_component = prepared["labels_by_component"]
        observed_fracs = prepared["observed_fracs"]
        assigned_fracs_by_label = prepared["assigned_fracs_by_label"]
        weight_fracs_by_label = prepared["weight_fracs_by_label"]

        x = np.arange(len(labels_by_component))
        width = 0.25
        obs_bars = ax.bar(
            x - width,
            observed_fracs,
            width=width,
            label="Observed labels",
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.6,
        )
        assigned_bars = ax.bar(
            x,
            assigned_fracs_by_label,
            width=width,
            label="Assigned MAP",
            color="#F58518",
            edgecolor="black",
            linewidth=0.6,
        )
        if weight_fracs_by_label is not None:
            weight_bars = ax.bar(
                x + width,
                weight_fracs_by_label,
                width=width,
                label="MAP mixing weights",
                color="#54A24B",
                edgecolor="black",
                linewidth=0.6,
            )
        else:
            weight_bars = []

        for bars, fractions in (
            (obs_bars, observed_fracs),
            (assigned_bars, assigned_fracs_by_label),
            (weight_bars, weight_fracs_by_label),
        ):
            if fractions is None:
                continue
            for idx, bar in enumerate(bars):
                frac = fractions[idx]
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    frac + 0.01,
                    f"{frac * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xlabel("Annotation label")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_by_component, rotation=30, ha="right")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Label Composition: Observed vs Assigned MAP vs Weights")
        y_max = max(
            0.12,
            float(np.max(observed_fracs)) if len(observed_fracs) else 0.0,
            (
                float(np.max(assigned_fracs_by_label))
                if len(assigned_fracs_by_label)
                else 0.0
            ),
            (
                float(np.max(weight_fracs_by_label))
                if weight_fracs_by_label is not None
                and len(weight_fracs_by_label)
                else 0.0
            ),
        )
        ax.set_ylim(0.0, min(1.05, y_max + 0.15))
    else:
        component_fractions = prepared["component_fractions"]
        hard_counts = prepared["hard_counts"]

        component_ids = np.arange(1, n_components + 1)
        bars = ax.bar(
            component_ids,
            component_fractions,
            color=plt.get_cmap("Blues")(np.linspace(0.45, 0.85, n_components)),
            edgecolor="black",
            linewidth=0.6,
        )

        for idx, bar in enumerate(bars):
            frac = component_fractions[idx]
            n_cells = int(hard_counts[idx])
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                frac + 0.01,
                f"{frac * 100:.1f}%\n(n={n_cells})",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel("Component")
        ax.set_xticks(component_ids)
        ax.set_title("Mixture Composition (MAP Assignments)")
        ax.set_ylim(
            0.0, min(1.05, max(0.12, np.max(component_fractions) + 0.15))
        )

    ax.set_ylabel("Cell fraction")

    return fig, [ax], 1

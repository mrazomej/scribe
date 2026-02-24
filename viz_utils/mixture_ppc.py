"""Mixture model PPC plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import numpyro.distributions as dist

from ._common import console, Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from .config import _get_config_values
from .dispatch import (
    _get_map_estimates_for_plot,
    _get_map_like_predictive_samples_for_plot,
    _get_cell_assignment_probabilities_for_plot,
)


def _select_divergent_genes(results, counts, n_rows, n_cols):
    """Select genes with highest divergence across components, binned by expression."""
    map_estimates = _get_map_estimates_for_plot(results, counts=counts)

    parameterization = results.model_config.parameterization
    if parameterization in ["linked", "mean_prob", "odds_ratio", "mean_odds"]:
        param_name = "mu"
    else:
        param_name = "r"

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
    map_estimates = _get_map_estimates_for_plot(results, counts=counts)

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
):
    """Plot a PPC figure in the standard format."""
    import scribe

    n_genes_selected = len(selected_idx)

    counts_np = np.array(counts)
    selected_means = np.array(
        [np.mean(counts_np[:, idx]) for idx in selected_idx]
    )
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]

    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        gene_idx = selected_idx_sorted[i]
        subset_pos = subset_positions[gene_idx]
        true_counts = counts_np[:, gene_idx]

        credible_regions = scribe.stats.compute_histogram_credible_regions(
            predictive_samples[:, :, subset_pos],
            credible_regions=[95, 68, 50],
        )

        hist_results = np.histogram(
            true_counts, bins=credible_regions["bin_edges"], density=True
        )

        cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
        max_bin = np.max(
            [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10]
        )

        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, credible_regions, cmap=cmap, alpha=0.5, max_bin=max_bin
        )

        max_bin_hist = (
            max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
        )
        ax.step(
            hist_results[1][:max_bin_hist],
            hist_results[0][:max_bin_hist],
            where="post",
            label="data",
            color="black",
        )

        ax.set_xlabel("counts")
        ax.set_ylabel("frequency")
        actual_mean_expr = np.mean(counts_np[:, gene_idx])
        mean_expr_formatted = f"{actual_mean_expr:.2f}"
        ax.set_title(
            f"$\\langle U \\rangle = {mean_expr_formatted}$",
            fontsize=8,
        )

    plt.tight_layout()
    fig.suptitle(title, y=1.02)

    output_path = os.path.join(figs_dir, f"{fname}.{output_format}")
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved {title} to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)


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
):
    """Plot comparison figure with mixture and all component PPCs overlaid."""
    import scribe

    if component_cmaps is None:
        component_cmaps = [
            "Greens", "Purples", "Reds", "Oranges", "YlOrBr", "BuGn",
        ]

    n_genes_selected = len(selected_idx)
    n_components = len(component_samples_list)

    counts_np = np.array(counts)
    selected_means = np.array(
        [np.mean(counts_np[:, idx]) for idx in selected_idx]
    )
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]

    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        gene_idx = selected_idx_sorted[i]
        subset_pos = subset_positions[gene_idx]
        true_counts = counts_np[:, gene_idx]

        x_max = max(int(np.percentile(true_counts, 99)), 10)
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
            scribe.viz.plot_histogram_credible_regions_stairs(
                ax, comp_cr, cmap=cmap, alpha=0.4, max_bin=x_max
            )

        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, mixture_cr, cmap="Blues", alpha=0.3, max_bin=x_max
        )

        ax.step(
            hist_results[1][:-1],
            hist_results[0],
            where="post",
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
        ax.set_title(
            f"$\\langle U \\rangle = {mean_expr_formatted}$",
            fontsize=8,
        )

    plt.tight_layout()
    fig.suptitle("PPC Comparison: Mixture vs Components", y=1.02)

    output_path = os.path.join(figs_dir, f"{fname}.{output_format}")
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved PPC comparison to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)


def plot_mixture_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """
    Plot PPC for mixture models showing genes with highest divergence across
    components.
    """
    console.print(
        "[dim]Plotting mixture model PPC (genes with highest CV across components)...[/dim]"
    )

    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping mixture PPC plot...[/yellow]"
        )
        return

    ppc_opts = viz_cfg.get("ppc_opts", {})
    mixture_ppc_opts = viz_cfg.get("mixture_ppc_opts", {})
    n_rows = mixture_ppc_opts.get("n_rows", ppc_opts.get("n_rows", 5))
    n_cols = mixture_ppc_opts.get("n_cols", ppc_opts.get("n_cols", 5))
    n_samples = mixture_ppc_opts.get(
        "n_samples", ppc_opts.get("n_samples", 1500)
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

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}"
    )

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

    _plot_ppc_figure(
        predictive_samples=mixture_samples_np,
        counts=counts,
        selected_idx=top_gene_indices,
        n_rows=n_rows,
        n_cols=n_cols,
        title="Mixture PPC (High CV Genes)",
        figs_dir=figs_dir,
        fname=f"{base_fname}_mixture_ppc",
        output_format=output_format,
    )

    del mixture_samples

    component_cmaps = ["Greens", "Purples", "Reds", "Oranges", "YlOrBr", "BuGn"]
    min_cells_for_info = mixture_ppc_opts.get("min_cells_for_info", 20)
    all_component_samples = []
    counts_np = np.array(counts)

    console.print("[dim]Computing MAP cell-to-component assignments...[/dim]")
    if assignment_batch_size is not None:
        console.print(
            f"[dim]Using assignment batch_size={assignment_batch_size}[/dim]"
        )
    assignment_probs = _get_cell_assignment_probabilities_for_plot(
        results, counts=counts, batch_size=assignment_batch_size
    )
    assignments = np.argmax(assignment_probs, axis=1)

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
        all_component_samples.append(component_samples_np)

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

        _plot_ppc_figure(
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
        )

        del component_samples

    if n_components <= 2:
        console.print("[dim]Generating combined comparison plot...[/dim]")
        _plot_ppc_comparison_figure(
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
        )
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

    del results_subset


def _reconstruct_label_map_for_composition(cell_labels, component_order=None):
    """Build label-to-component mapping used for composition comparison plots.

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


def plot_mixture_composition(
    results, counts, figs_dir, cfg, viz_cfg, cell_labels=None
):
    """Plot MAP component composition for mixture models.

    This figure summarizes each component's composition as the fraction of cells
    assigned to that component under MAP-like assignment probabilities.

    When ``cell_labels`` are provided, this plot switches to a side-by-side
    comparison per label:

    - observed fraction of each label in the dataset
    - predicted fraction from model assignments mapped to that label's component
    """
    console.print("[dim]Plotting mixture component composition...[/dim]")

    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping composition plot...[/yellow]"
        )
        return

    # Extract MAP mixture weights (global composition prior at MAP).
    map_estimates = _get_map_estimates_for_plot(results, counts=counts)
    mixing_weights = map_estimates.get("mixing_weights")
    weight_fractions = None
    if mixing_weights is not None:
        weight_fractions = np.array(mixing_weights, dtype=float).reshape(-1)
        if weight_fractions.shape[0] != n_components:
            weight_fractions = weight_fractions[:n_components]
        weight_fractions = np.clip(weight_fractions, a_min=0.0, a_max=None)
        weight_fractions = weight_fractions / max(
            np.sum(weight_fractions), 1e-12
        )

    fig, ax = plt.subplots(1, 1, figsize=(max(6.0, 1.1 * n_components + 2.0), 4.0))

    if cell_labels is not None:
        # Compute assignment-based fractions per component. These reflect
        # p(z_i | x_i, theta_map) and are the natural model-predicted label mix.
        ppc_opts = viz_cfg.get("ppc_opts", {})
        mixture_ppc_opts = viz_cfg.get("mixture_ppc_opts", {})
        composition_opts = viz_cfg.get("mixture_composition_opts", {})
        assignment_batch_size = composition_opts.get(
            "assignment_batch_size",
            mixture_ppc_opts.get(
                "assignment_batch_size",
                mixture_ppc_opts.get("batch_size", ppc_opts.get("batch_size", 512)),
            ),
        )
        if assignment_batch_size is not None and assignment_batch_size <= 0:
            assignment_batch_size = None
        if assignment_batch_size is not None:
            console.print(
                f"[dim]Using assignment batch_size={assignment_batch_size}[/dim]"
            )
        assignment_probs = _get_cell_assignment_probabilities_for_plot(
            results, counts=counts, batch_size=assignment_batch_size
        )
        assignments = np.argmax(assignment_probs, axis=1)
        assignment_counts = np.bincount(assignments, minlength=n_components)
        assigned_fractions = assignment_counts / max(
            1, int(np.sum(assignment_counts))
        )

        # Compare observed annotation proportions against model-predicted
        # proportions by mapping each annotation to its configured component.
        annotations = np.asarray(cell_labels).astype(str)
        component_order = cfg.get("annotation_component_order", None)
        label_map = _reconstruct_label_map_for_composition(
            cell_labels=annotations, component_order=component_order
        )
        labels_by_component = sorted(label_map.keys(), key=lambda x: label_map[x])
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
            float(np.max(assigned_fracs_by_label))
            if len(assigned_fracs_by_label)
            else 0.0,
            float(np.max(weight_fracs_by_label))
            if weight_fracs_by_label is not None and len(weight_fracs_by_label)
            else 0.0,
        )
        ax.set_ylim(0.0, min(1.05, y_max + 0.15))
    else:
        # Global component-only view: prefer MAP weights and fall back to
        # assignment fractions if weights are unavailable.
        if weight_fractions is not None:
            component_fractions = weight_fractions
            hard_counts = np.rint(
                component_fractions * int(results.n_cells)
            ).astype(int)
        else:
            ppc_opts = viz_cfg.get("ppc_opts", {})
            mixture_ppc_opts = viz_cfg.get("mixture_ppc_opts", {})
            composition_opts = viz_cfg.get("mixture_composition_opts", {})
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
                results, counts=counts, batch_size=assignment_batch_size
            )
            assignments = np.argmax(assignment_probs, axis=1)
            hard_counts = np.bincount(assignments, minlength=n_components)
            component_fractions = hard_counts / max(1, int(np.sum(hard_counts)))

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

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}"
    )
    output_path = os.path.join(
        figs_dir, f"{base_fname}_mixture_composition.{output_format}"
    )
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved mixture composition to[/dim] "
        f"[cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

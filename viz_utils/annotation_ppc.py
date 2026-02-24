"""Annotation-based PPC plotting."""

import numpy as np
from jax import random

from ._common import console
from .config import _get_config_values
from .dispatch import _get_map_like_predictive_samples_for_plot
from .gene_selection import _select_genes
from .mixture_ppc import _plot_ppc_figure


def _reconstruct_label_map(cell_labels, component_order=None):
    """
    Reconstruct the label-to-component-index mapping used during inference.

    Must match scribe.core.annotation_prior.build_annotation_prior_logits:
    when no explicit component_order is given, unique labels are sorted
    alphabetically; otherwise the supplied order defines the mapping.
    """
    import pandas as pd

    annotations = pd.Series(cell_labels)
    labels = annotations.dropna()

    if component_order is not None:
        label_map = {
            str(label): idx
            for idx, label in enumerate(component_order)
        }
    else:
        unique_sorted = sorted(str(l) for l in labels.unique())
        label_map = {
            label: idx for idx, label in enumerate(unique_sorted)
        }
    return label_map


def plot_annotation_ppc(results, counts, cell_labels, figs_dir, cfg, viz_cfg):
    """
    Plot per-annotation posterior predictive checks.

    For each unique annotation label, a PPC figure is generated that compares
    the observed count distribution of cells belonging to that annotation
    against the posterior predictive distribution of the corresponding mixture
    component.
    """
    import pandas as pd

    console.print("[dim]Plotting annotation PPC...[/dim]")

    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping annotation "
            "PPC...[/yellow]"
        )
        return

    ppc_opts = viz_cfg.get("ppc_opts", {})
    ann_opts = viz_cfg.get("annotation_ppc_opts", {})
    n_rows = ann_opts.get("n_rows", ppc_opts.get("n_rows", 5))
    n_cols = ann_opts.get("n_cols", ppc_opts.get("n_cols", 5))
    n_samples = ann_opts.get("n_samples", ppc_opts.get("n_samples", 1500))

    component_order = cfg.get("annotation_component_order", None)
    label_map = _reconstruct_label_map(cell_labels, component_order)

    annotations = pd.Series(cell_labels)

    console.print(
        f"[dim]Label → component mapping ({len(label_map)} labels, "
        f"{n_components} components):[/dim]"
    )
    for label, idx in label_map.items():
        n_cells_label = int((annotations.astype(str) == label).sum())
        console.print(
            f"[dim]  {label} → component {idx} "
            f"({n_cells_label} cells)[/dim]"
        )

    selected_idx, _mean_counts = _select_genes(counts, n_rows, n_cols)
    n_genes_selected = len(selected_idx)
    console.print(
        f"[dim]Selected {n_genes_selected} genes across {n_rows} "
        f"expression bins[/dim]"
    )

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_"
        f"{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}"
    )

    component_cmaps = [
        "Blues", "Greens", "Purples", "Reds", "Oranges", "YlOrBr", "BuGn",
    ]
    rng_key = random.PRNGKey(42)

    for label, component_idx in label_map.items():
        console.print(
            f"[dim]Processing annotation '{label}' "
            f"(component {component_idx})...[/dim]"
        )

        cell_mask = np.array(annotations.astype(str) == label)
        n_cells_label = int(cell_mask.sum())
        if n_cells_label == 0:
            console.print(
                f"[yellow]  Skipping '{label}' (no cells)[/yellow]"
            )
            continue

        console.print(
            f"[dim]  {n_cells_label} cells with this annotation[/dim]"
        )

        component_results = results.get_component(component_idx)
        component_subset = component_results[selected_idx]

        rng_key, subkey = random.split(rng_key)
        console.print(
            f"[dim]  Generating {n_samples} PPC samples...[/dim]"
        )
        component_samples = _get_map_like_predictive_samples_for_plot(
            component_subset,
            rng_key=subkey,
            n_samples=n_samples,
            cell_batch_size=500,
            store_samples=False,
            verbose=True,
            counts=counts,
        )

        component_samples_np = np.array(
            component_samples[:, cell_mask, :]
        )
        counts_label = counts[cell_mask, :]

        safe_label = (
            str(label)
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
        )

        cmap = component_cmaps[component_idx % len(component_cmaps)]

        _plot_ppc_figure(
            predictive_samples=component_samples_np,
            counts=counts_label,
            selected_idx=selected_idx,
            n_rows=n_rows,
            n_cols=n_cols,
            title=(
                f"Annotation PPC: {label} "
                f"(Component {component_idx}, "
                f"{n_cells_label} cells)"
            ),
            figs_dir=figs_dir,
            fname=f"{base_fname}_annotation_ppc_{safe_label}",
            output_format=output_format,
            cmap=cmap,
        )

        del component_samples, component_samples_np

    console.print(
        f"[green]✓[/green] [dim]Generated {len(label_map)} annotation "
        f"PPC plots[/dim]"
    )

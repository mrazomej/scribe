"""Correlation heatmap plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
import jax.numpy as jnp

from ._common import console
from .config import _get_config_values


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


def plot_correlation_heatmap(results, counts, figs_dir, cfg, viz_cfg):
    """
    Plot clustered correlation heatmap of gene parameters from posterior samples.

    For mixture models, a separate heatmap is produced for every component.
    """
    console.print("[dim]Plotting correlation heatmap...[/dim]")

    heatmap_opts = viz_cfg.get("heatmap_opts", {})
    n_genes_to_plot = heatmap_opts.get("n_genes", 1500)
    n_samples = heatmap_opts.get("n_samples", 512)
    figsize = heatmap_opts.get("figsize", 12)
    cmap = heatmap_opts.get("cmap", "RdBu_r")

    parameterization = results.model_config.parameterization
    if parameterization in ["linked", "mean_prob", "odds_ratio", "mean_odds"]:
        param_name = "mu"
    else:
        param_name = "r"

    console.print(
        f"[dim]Using parameter '{param_name}' for correlation "
        f"(parameterization: {parameterization})[/dim]"
    )

    if results.posterior_samples is None:
        console.print(f"[dim]Generating {n_samples} posterior samples...[/dim]")
        results.get_posterior_samples(
            rng_key=random.PRNGKey(42),
            n_samples=n_samples,
            store_samples=True,
            counts=counts,
        )
    else:
        console.print("[dim]Using existing posterior samples...[/dim]")
        n_samples = results.posterior_samples[param_name].shape[0]
        console.print(f"[dim]Found {n_samples} existing samples[/dim]")

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

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_"
        f"{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}"
    )

    if samples.ndim == 3:
        n_components = samples.shape[1]
        n_genes = samples.shape[2]
        n_genes_capped = min(n_genes_to_plot, n_genes)
        console.print(
            f"[dim]Detected mixture model with {n_components} components "
            f"– generating per-component heatmaps[/dim]"
        )

        correlation_matrices = []
        variance_per_component = []

        for k in range(n_components):
            comp_samples = samples[:, k, :]
            corr_matrix = _compute_correlation_matrix(
                comp_samples, n_samples
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

        for k in range(n_components):
            corr_subset_np = np.array(
                correlation_matrices[k][
                    jnp.ix_(selected_genes, selected_genes)
                ]
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
                figsize=(figsize, figsize),
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
                f"Parameter: {param_name} | Samples: {n_samples}",
                y=1.02,
                fontsize=12,
            )

            fname = (
                f"{base_fname}_correlation_heatmap_"
                f"component{k + 1}.{output_format}"
            )
            output_path = os.path.join(figs_dir, fname)
            fig.savefig(output_path, bbox_inches="tight")
            console.print(
                f"[green]✓[/green] [dim]Saved component {k + 1} heatmap "
                f"to[/dim] [cyan]{output_path}[/cyan]"
            )
            plt.close(fig.fig)

    else:
        n_genes = samples.shape[1]
        n_genes_to_plot = min(n_genes_to_plot, n_genes)

        console.print(
            "[dim]Computing pairwise Pearson correlations...[/dim]"
        )
        correlation_matrix = _compute_correlation_matrix(samples, n_samples)

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
        top_var_indices = jnp.argsort(correlation_variance)[
            -n_genes_to_plot:
        ]
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
            figsize=(figsize, figsize),
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
            f"Parameter: {param_name} | Samples: {n_samples}",
            y=1.02,
            fontsize=12,
        )

        fname = f"{base_fname}_correlation_heatmap.{output_format}"
        output_path = os.path.join(figs_dir, fname)
        fig.savefig(output_path, bbox_inches="tight")
        console.print(
            f"[green]✓[/green] [dim]Saved correlation heatmap to[/dim] "
            f"[cyan]{output_path}[/cyan]"
        )
        plt.close(fig.fig)

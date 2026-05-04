"""Correlation heatmap plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
import jax.numpy as jnp

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

    base_fname = "correlation"
    if ctx.save:
        _fname_prefix = ctx.build_filename(
            "correlation_heatmap", results=results
        )
        base_fname = _fname_prefix.rsplit("_correlation_heatmap.", 1)[0]
    output_format = ctx.output_format

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

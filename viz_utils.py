"""
viz_utils.py

This module provides utility and helper functions related to visualization and
configuration extraction for the SCRIBE probabilistic modeling library. It
includes routines for:

- Extracting key configuration parameters from various Hydra/OmegaConf config
  structures, supporting both legacy and modern config layouts.
- (Presumed from context) Functions and helpers that facilitate
  visualization—such as plotting model outputs, latent variables, or diagnostic
  summaries—using popular libraries like matplotlib and seaborn.
- Standardized routines to help downstream scripts and notebooks consistently
  access and visualize parameters, results, and diagnostics, regardless of model
  specification or inference method.

The module is intended to be imported by users or higher-level modules that need
unified access to configuration parsing and robust, publication-quality figures
for model diagnostics and result interpretation.
"""

import os
import math
import re
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
import scribe
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()

# ==============================================================================
# Helper functions
# ==============================================================================


def _get_config_values(cfg, results=None):
    """
    Extracts relevant configuration values from a Hydra/OmegaConf config object.

    This function is designed to handle both legacy and current config
    structures. It retrieves the following values:
        - parameterization: The parameterization type for the model.
        - n_components: Number of mixture components (default 1 if not
          specified).
        - n_steps: Number of inference steps (default 50000 if not specified).
        - method: Inference method (e.g., 'svi', 'mcmc').
        - model_type: The type of model (default 'default' if not specified).

    The function first checks for the presence of the 'inference' attribute to
    determine if the config uses the old structure. If not, it falls back to the
    new structure where keys are at the top level.

    When ``results`` is provided and has a non-None ``n_components`` attribute,
    that value takes precedence over the config.  This handles the case where
    ``n_components`` was auto-inferred from annotation labels at runtime and is
    therefore absent from the saved Hydra config.

    Parameters
    ----------
    cfg : OmegaConf.DictConfig or dict-like
        The configuration object from which to extract values.
    results : object, optional
        A results object (e.g., ``ScribeSVIResults``) that may carry the
        actual ``n_components`` used during inference.

    Returns
    -------
    dict
        A dictionary containing the extracted configuration values.
    """
    # Check if the config uses the old structure (has 'inference' attribute and
    # 'parameterization' under it)
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "parameterization"):
        # Old structure: extract values from cfg.inference

        # Model parameterization type
        parameterization = cfg.inference.parameterization
        # Number of mixture components, default to 1 if None/0
        n_components = cfg.inference.n_components or 1
        # Number of inference steps
        n_steps = cfg.inference.n_steps
        # Inference method (e.g., 'svi', 'mcmc')
        method = cfg.inference.method
    else:
        # New structure: extract values from top-level keys

        # Default to 'standard' if not present
        parameterization = cfg.get("parameterization", "standard")
        # Default to 1 if not present or falsy
        n_components = cfg.get("n_components") or 1
        # Default to 50000 if not present
        n_steps = cfg.get("n_steps", 50000)
        # Try to get method from cfg.inference if it exists, otherwise default
        # to 'svi'
        method = cfg.inference.method if hasattr(cfg, "inference") else "svi"

    # Attempt to extract model type from cfg
    # Handle both old format (cfg.model.type) and new format (cfg.model as string)
    model_attr = getattr(cfg, "model", None)
    if model_attr is None:
        # Model not set - derive from feature flags if available
        zero_inflation = (
            cfg.get("zero_inflation", False) if hasattr(cfg, "get") else False
        )
        variable_capture = (
            cfg.get("variable_capture", False) if hasattr(cfg, "get") else False
        )
        if zero_inflation and variable_capture:
            model_type = "zinbvcp"
        elif zero_inflation:
            model_type = "zinb"
        elif variable_capture:
            model_type = "nbvcp"
        else:
            model_type = "nbdm"
    elif isinstance(model_attr, str):
        # New format: model is a string directly
        model_type = model_attr
    elif hasattr(model_attr, "get"):
        # Old format: model is a dict-like with 'type' key
        model_type = model_attr.get("type", "default")
    else:
        model_type = "default"

    # If a results object carries the actual n_components (e.g. auto-inferred
    # from annotation labels), prefer that over the config value.
    if results is not None:
        res_nc = getattr(results, "n_components", None)
        if res_nc is not None:
            n_components = res_nc

    # Return all extracted values in a dictionary for easy access
    return {
        "parameterization": parameterization,
        "n_components": n_components,
        "n_steps": n_steps,
        "method": method,
        "model_type": model_type,
    }


# ------------------------------------------------------------------------------


def _sanitize_cache_token(value, max_len=48):
    """Create a filesystem-friendly token for cache filenames.

    Parameters
    ----------
    value : object
        Raw value to sanitize (typically a config field value).
    max_len : int, default=48
        Maximum token length after sanitization and trimming.

    Returns
    -------
    str
        Sanitized token containing only letters, numbers, underscores,
        hyphens, and periods. Empty values become ``"none"``.
    """
    # Convert to string and normalize whitespace.
    as_str = str(value).strip()
    if as_str == "":
        return "none"

    # Replace unsafe filename characters while preserving readability.
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", as_str)
    token = token.strip("._-")
    if token == "":
        token = "none"
    if len(token) > max_len:
        token = token[:max_len]
    return token


def _build_umap_cache_path(cfg, cache_umap):
    """Build a split-aware path for persisted UMAP reducer caches.

    The cache filename is keyed by dataset stem and split metadata
    (``subset_column`` / ``subset_value``).  A short hash suffix is included
    to avoid collisions when sanitized subset values truncate to similar
    strings.

    Parameters
    ----------
    cfg : DictConfig
        Inference configuration that includes ``data.path`` and optionally
        ``data.subset_column`` / ``data.subset_value``.
    cache_umap : bool
        Whether UMAP caching is enabled.

    Returns
    -------
    str or None
        Absolute path to the cache file when caching is enabled and
        ``cfg.data.path`` exists, otherwise ``None``.
    """
    if not cache_umap:
        return None

    data_cfg = cfg.data if hasattr(cfg, "data") else None
    data_path = data_cfg.path if data_cfg is not None else None
    if not data_path:
        return None

    data_dir = os.path.dirname(data_path)
    data_stem = os.path.splitext(os.path.basename(data_path))[0]
    subset_column = data_cfg.get("subset_column", None)
    subset_value = data_cfg.get("subset_value", None)

    # Include subset identity in cache filename so split runs do not share
    # the same reducer cache when they come from one source .h5ad file.
    if subset_column is not None and subset_value is not None:
        subset_col_token = _sanitize_cache_token(subset_column)
        subset_val_token = _sanitize_cache_token(subset_value)
        cache_identity = (
            f"{os.path.abspath(data_path)}|{subset_column}|{subset_value}"
        )
        cache_hash = hashlib.sha1(cache_identity.encode("utf-8")).hexdigest()[
            :12
        ]
        cache_name = (
            f"{data_stem}_umap_{subset_col_token}_{subset_val_token}_"
            f"{cache_hash}.pkl"
        )
    else:
        cache_name = f"{data_stem}_umap_full.pkl"

    return os.path.join(data_dir, cache_name)


# ------------------------------------------------------------------------------


def _select_genes_simple(counts, n_genes):
    """Simple gene selection for ECDF plots (linear spacing)."""
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    spaced_indices = np.linspace(0, len(sorted_idx) - 1, num=n_genes, dtype=int)
    selected_idx = sorted_idx[spaced_indices]
    return selected_idx, mean_counts


# ------------------------------------------------------------------------------


def _select_genes(counts, n_rows, n_cols):
    """Select a subset of genes for plotting using log-spaced binning.

    Divides the expression range into n_rows logarithmically-spaced bins.
    Within each bin, selects n_cols genes using logarithmic spacing.
    This ensures coverage across the full expression range while including
    some low-expression genes.

    Parameters
    ----------
    counts : array-like, shape (n_cells, n_genes)
        The count matrix where rows are cells and columns are genes.
    n_rows : int
        Number of expression bins (rows in the plot).
    n_cols : int
        Number of genes to select per bin (columns in the plot).

    Returns
    -------
    selected_idx : array
        Indices of selected genes.
    mean_counts : array
        Median expression values for all genes.
    """
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]

    if len(nonzero_idx) == 0:
        # Fallback: return empty selection
        return np.array([], dtype=int), mean_counts

    # Sort genes by expression value
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    sorted_means = mean_counts[sorted_idx]

    # Get expression range
    min_expr = sorted_means[0]
    max_expr = sorted_means[-1]

    # Create logarithmically-spaced bin edges
    # Use a small offset to avoid log(0) issues
    min_expr_safe = max(min_expr, 0.1)
    bin_edges = np.logspace(
        np.log10(min_expr_safe), np.log10(max_expr), num=n_rows + 1
    )
    # Ensure first bin edge includes the minimum
    bin_edges[0] = min_expr

    # Track selected genes to avoid duplicates
    selected_set = set()
    selected_by_bin = []

    # First pass: select genes from each bin
    for i in range(n_rows):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Find genes in this expression range
        # For the last bin, include genes at the upper edge
        if i == n_rows - 1:
            in_bin = (sorted_means >= bin_start) & (sorted_means <= bin_end)
        else:
            in_bin = (sorted_means >= bin_start) & (sorted_means < bin_end)

        bin_indices = np.where(in_bin)[0]
        bin_selected = []

        if len(bin_indices) > 0:
            if len(bin_indices) <= n_cols:
                # If bin has fewer genes than needed, use all of them
                bin_selected = list(bin_indices)
            else:
                # Log-space select n_cols genes within this bin
                bin_means = sorted_means[bin_indices]
                bin_min = bin_means[0]
                bin_max = bin_means[-1]

                # Generate log-spaced target values within this bin
                bin_min_safe = max(bin_min, 0.1)
                log_targets = np.logspace(
                    np.log10(bin_min_safe), np.log10(bin_max), num=n_cols
                )
                # Ensure first target includes the minimum
                log_targets[0] = bin_min

                # Find closest genes to each target
                for target in log_targets:
                    closest_idx = np.argmin(np.abs(bin_means - target))
                    bin_selected.append(bin_indices[closest_idx])

                # Remove duplicates
                bin_selected = list(np.unique(bin_selected))

        selected_by_bin.append(bin_selected)
        selected_set.update(bin_selected)

    # Second pass: backfill bins that are short
    # Create a pool of all unselected genes
    all_indices = set(range(len(sorted_idx)))
    unselected_indices = sorted(list(all_indices - selected_set))
    unselected_means = sorted_means[unselected_indices]

    # Fill each bin to n_cols
    final_selected = []
    for i in range(n_rows):
        bin_selected = selected_by_bin[i]
        needed = n_cols - len(bin_selected)

        if needed > 0:
            # Get bin expression range for backfilling
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = np.sqrt(bin_start * bin_end)  # Geometric mean

            # Find unselected genes closest to this bin's expression range
            # Prefer genes from previous bins (lower expression) if available
            candidates = []
            for idx in unselected_indices:
                expr = sorted_means[idx]
                # Prefer genes that are close to the bin range
                if expr <= bin_end:
                    distance = abs(expr - bin_center)
                    candidates.append((distance, idx))

            # Sort by distance and take the closest ones
            candidates.sort(key=lambda x: x[0])
            backfill_indices = [idx for _, idx in candidates[:needed]]

            # Add to bin selection
            bin_selected.extend(backfill_indices)
            # Remove from unselected pool
            for idx in backfill_indices:
                unselected_indices.remove(idx)

        # Convert to actual gene indices and add to final selection
        final_selected.extend(
            [sorted_idx[idx] for idx in bin_selected[:n_cols]]
        )

    selected_idx = np.array(final_selected, dtype=int)
    return selected_idx, mean_counts


def plot_loss(results, figs_dir, cfg, viz_cfg):
    """Plot and save the ELBO loss history."""
    console.print("[dim]Plotting loss history...[/dim]")

    # Initialize figure with two subplots side by side
    fig, (ax_log, ax_linear) = plt.subplots(1, 2, figsize=(7.0, 3))

    # Plot loss history on both subplots
    ax_log.plot(results.loss_history)
    ax_linear.plot(results.loss_history)

    # Set labels for both subplots
    ax_log.set_xlabel("step")
    ax_log.set_ylabel("ELBO loss")
    ax_linear.set_xlabel("step")
    ax_linear.set_ylabel("ELBO loss")

    # Set y-axis scales: log scale for left, linear for right
    ax_log.set_yscale("log")
    # ax_linear uses linear scale by default

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename from original config
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_"
        f"{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_loss.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved loss plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)


# ==============================================================================
# Visualization functions
# ==============================================================================


def plot_ecdf(counts, figs_dir, cfg, viz_cfg):
    """Plot and save the ECDF of selected genes."""
    console.print("[dim]Plotting ECDF...[/dim]")

    # Gene selection (simple linear spacing for ECDF)
    n_genes = viz_cfg.ecdf_opts.n_genes
    selected_idx, _ = _select_genes_simple(counts, n_genes)

    # Sort selected indices for consistency
    selected_idx = np.sort(selected_idx)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    for i, idx in enumerate(selected_idx):
        sns.ecdfplot(
            data=counts[:, idx],
            ax=ax,
            color=sns.color_palette("Blues", n_colors=n_genes)[i],
            lw=1.5,
            label=None,
        )
    ax.set_xlabel("UMI count")
    ax.set_xscale("log")
    ax.set_ylabel("ECDF")
    plt.tight_layout()

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_example_ecdf.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved ECDF plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)


# ------------------------------------------------------------------------------


def plot_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """Plot and save the posterior predictive checks."""
    console.print("[dim]Plotting PPC...[/dim]")

    # Gene selection using log-spaced binning
    n_rows = viz_cfg.ppc_opts.n_rows
    n_cols = viz_cfg.ppc_opts.n_cols
    console.print(
        f"[dim]Using n_rows={n_rows}, n_cols={n_cols} for PPC plot (log-spaced binning)[/dim]"
    )
    selected_idx, mean_counts = _select_genes(counts, n_rows, n_cols)

    # Sort selected indices by median expression (ascending)
    # This ensures genes are plotted from lowest to highest expression
    selected_means = mean_counts[selected_idx]
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]
    n_genes_selected = len(selected_idx_sorted)
    console.print(
        f"[dim]Selected {n_genes_selected} genes across {n_rows} expression bins[/dim]"
    )

    # Index results for selected genes (using original unsorted order for subsetting)
    results_subset = results[selected_idx]

    # Generate posterior predictive samples
    n_samples = viz_cfg.ppc_opts.n_samples
    console.print(
        f"[dim]Generating {n_samples} posterior predictive samples...[/dim]"
    )
    # Pass full counts matrix (not subsetted) for amortized capture probability
    # The amortizer needs total UMI count per cell, which is computed across all genes
    results_subset.get_ppc_samples(n_samples=n_samples, counts=counts)

    # Create mapping from gene index to position in subset
    # The subset preserves the original gene order (sorted by index), not the
    # order of selected_idx So we need to find where each gene appears in the
    # sorted original indices
    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    # Plotting - use n_rows and n_cols directly
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = axes.flatten()

    # Use progress bar for plotting multiple panels
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

        for i, ax in enumerate(axes):
            if i >= n_genes_selected:
                ax.axis("off")
                continue

            # Get the gene index in sorted order
            gene_idx = selected_idx_sorted[i]
            # Get the position of this gene in the results_subset
            subset_pos = subset_positions[gene_idx]

            true_counts = counts[:, gene_idx]

            credible_regions = scribe.stats.compute_histogram_credible_regions(
                results_subset.predictive_samples[:, :, subset_pos],
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
                ax, credible_regions, cmap="Blues", alpha=0.5, max_bin=max_bin
            )

            max_bin_hist = (
                max_bin
                if len(hist_results[0]) > max_bin
                else len(hist_results[0])
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
            # Compute actual mean expression (mean_counts is actually median, used
            # for selection)
            actual_mean_expr = np.mean(counts[:, gene_idx])
            mean_expr_formatted = f"{actual_mean_expr:.2f}"
            ax.set_title(
                f"$\\langle U \\rangle = {mean_expr_formatted}$",
                fontsize=8,
            )

            progress.update(task, advance=1)

    plt.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_ppc.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved PPC plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

    # Clean up results_subset to free memory (it contains samples for subset of
    # genes) The original results object doesn't have samples, so no cleanup
    # needed there
    del results_subset


# ------------------------------------------------------------------------------


def plot_umap(results, counts, figs_dir, cfg, viz_cfg, force_refit=False):
    """
    Plot UMAP projection of experimental and synthetic data.

    The embedding follows a Scanpy-style preprocessing workflow before fitting
    UMAP:

    1. Filter genes detected in fewer than ``gene_filter_min_cells`` cells.
    2. Library-size normalize to ``target_sum`` counts per cell.
    3. Apply ``log1p`` transform.
    4. Select highly variable genes (HVGs) via Scanpy (optional).
    5. Run PCA on the selected genes (optional), then fit UMAP.

    The same learned preprocessing (gene mask, HVG mask, and PCA model) is then
    applied to synthetic counts before projection with ``umap_reducer.transform``.
    All preprocessing is done on local copies and does not modify ``counts`` or
    the results object.

    Parameters
    ----------
    results : ScribeSVIResults
        Inference results object.
    counts : array-like, shape (n_cells, n_genes)
        Raw count matrix (integers).
    figs_dir : str
        Directory in which to save the figure.
    cfg : DictConfig
        Original Hydra configuration.
    viz_cfg : DictConfig
        Visualization configuration.
    force_refit : bool, optional
        When ``True``, always refit UMAP even if a cache file exists. If cache
        is enabled, the cache file is overwritten with the newly fitted reducer.
    """
    import pickle

    console.print("[dim]Plotting UMAP projection...[/dim]")

    umap_opts = viz_cfg.get("umap_opts", {})

    try:
        import umap
    except ImportError:
        console.print(
            "[bold red]❌ ERROR:[/bold red] [red]umap-learn is not installed.[/red]"
            " [yellow]Install it with: pip install umap-learn[/yellow]"
        )
        return

    # Get UMAP parameters from config
    n_neighbors = umap_opts.get("n_neighbors", 15)
    min_dist = umap_opts.get("min_dist", 0.1)
    n_components = umap_opts.get("n_components", 2)
    random_state = umap_opts.get("random_state", 42)
    data_color = umap_opts.get("data_color", "dark_blue")
    synthetic_color = umap_opts.get("synthetic_color", "dark_red")
    cache_umap = umap_opts.get("cache_umap", True)
    target_sum = umap_opts.get("target_sum", 1e4)
    gene_filter_min_cells = int(umap_opts.get("gene_filter_min_cells", 3))
    use_hvg = bool(umap_opts.get("use_hvg", True))
    hvg_n_top_genes = int(umap_opts.get("hvg_n_top_genes", 2000))
    hvg_flavor = umap_opts.get("hvg_flavor", "seurat")
    use_pca = bool(umap_opts.get("use_pca", True))
    pca_n_comps = int(umap_opts.get("pca_n_comps", 50))
    n_ppc_samples = int(umap_opts.get("n_ppc_samples", 50))

    if force_refit:
        console.print(
            "[yellow]⚠️[/yellow] [yellow]Overwrite requested: forcing UMAP refit and cache overwrite[/yellow]"
        )

    try:
        import scanpy as sc
        import anndata as ad

        has_scanpy = True
    except ImportError:
        has_scanpy = False
        sc = None
        ad = None
        if use_hvg:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]scanpy/anndata not available; skipping HVG selection and using all retained genes[/yellow]"
            )

    try:
        from sklearn.decomposition import PCA

        has_sklearn = True
    except ImportError:
        has_sklearn = False
        PCA = None
        if use_pca:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]scikit-learn not available; skipping PCA and fitting UMAP directly on gene space[/yellow]"
            )

    def _normalize_log1p(matrix):
        """Apply library-size normalization followed by log1p."""
        matrix = np.array(matrix, dtype=np.float64)
        lib_sizes = matrix.sum(axis=1, keepdims=True)
        lib_sizes = np.where(lib_sizes == 0, 1.0, lib_sizes)
        return np.log1p(matrix / lib_sizes * target_sum)

    # Get colors from scribe.viz if available
    try:
        colors = scribe.viz.colors
        # Convert color names to actual color values if using scribe colors
        if hasattr(colors, data_color):
            data_color = getattr(colors, data_color)
        if hasattr(colors, synthetic_color):
            synthetic_color = getattr(colors, synthetic_color)
    except (AttributeError, ImportError):
        # Fallback to matplotlib color names
        pass

    # Determine split-aware cache path (same data directory, subset-specific).
    cache_path = _build_umap_cache_path(cfg, cache_umap)
    data_cfg = cfg.data if hasattr(cfg, "data") else None
    subset_column = data_cfg.get("subset_column", None) if data_cfg else None
    subset_value = data_cfg.get("subset_value", None) if data_cfg else None

    # Current UMAP parameters for cache validation
    current_params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": n_components,
        "random_state": random_state,
        "target_sum": target_sum,
        "gene_filter_min_cells": gene_filter_min_cells,
        "use_hvg": use_hvg,
        "hvg_n_top_genes": hvg_n_top_genes,
        "hvg_flavor": hvg_flavor,
        "use_pca": use_pca,
        "pca_n_comps": pca_n_comps,
        "n_cells": counts.shape[0],
        "n_genes": counts.shape[1],
        "subset_column": subset_column,
        "subset_value": subset_value,
    }

    # Try to load cached UMAP
    umap_reducer = None
    umap_embedding = None
    gene_mask = None
    hvg_mask = None
    pca_model = None

    if cache_path and os.path.exists(cache_path) and not force_refit:
        console.print(
            f"[dim]Found cached UMAP at:[/dim] [cyan]{cache_path}[/cyan]"
        )
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Validate cached parameters match current parameters
            cached_params = cached.get("params", {})
            params_match = all(
                cached_params.get(k) == v for k, v in current_params.items()
            )
            has_preprocessing = "preprocessing" in cached

            if params_match and has_preprocessing:
                cached_pre = cached["preprocessing"]
                gene_mask = cached_pre.get("gene_mask")
                hvg_mask = cached_pre.get("hvg_mask")
                pca_model = cached_pre.get("pca_model")

                valid_masks = (
                    gene_mask is not None
                    and hvg_mask is not None
                    and len(gene_mask) == counts.shape[1]
                    and int(np.sum(gene_mask)) == len(hvg_mask)
                )

                if not valid_masks:
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]Cached preprocessing metadata invalid - will re-fit UMAP[/yellow]"
                    )
                else:
                    console.print(
                        "[green]✅[/green] [dim]Cache valid - loading UMAP from cache...[/dim]"
                    )
                    umap_reducer = cached["reducer"]
                    umap_embedding = cached["embedding"]
            elif params_match and not has_preprocessing:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Cache is from an older format (missing preprocessing metadata) - will re-fit UMAP[/yellow]"
                )
            else:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Cache parameters mismatch - will re-fit UMAP[/yellow]"
                )
                console.print(f"[dim]   Cached:[/dim] {cached_params}")
                console.print(f"[dim]   Current:[/dim] {current_params}")
        except Exception as e:
            console.print(
                f"[yellow]⚠️[/yellow] [yellow]Failed to load cache: {e} - will re-fit UMAP[/yellow]"
            )

    # Fit UMAP if not loaded from cache
    if umap_reducer is None:
        console.print(
            "[dim]Preparing experimental data with Scanpy-style preprocessing...[/dim]"
        )
        counts_np = np.array(counts, dtype=np.float64)

        # 1) Filter genes by detection frequency
        detected_cells_per_gene = np.sum(counts_np > 0, axis=0)
        gene_mask = detected_cells_per_gene >= gene_filter_min_cells
        n_genes_retained = int(np.sum(gene_mask))
        if n_genes_retained == 0:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]Gene filtering removed all genes; reverting to all genes[/yellow]"
            )
            gene_mask = np.ones(counts_np.shape[1], dtype=bool)
            n_genes_retained = counts_np.shape[1]

        console.print(
            f"[dim]Gene filter: kept {n_genes_retained}/{counts_np.shape[1]} genes "
            f"(min_cells={gene_filter_min_cells})[/dim]"
        )

        # 2) Normalize + log1p on filtered matrix
        counts_gene_filtered = counts_np[:, gene_mask]
        counts_norm = _normalize_log1p(counts_gene_filtered)

        # 3) HVG selection (optional)
        hvg_mask = np.ones(counts_norm.shape[1], dtype=bool)
        if use_hvg and counts_norm.shape[1] > 1 and has_scanpy:
            n_top_genes = min(hvg_n_top_genes, counts_norm.shape[1])
            if n_top_genes >= 1:
                console.print(
                    f"[dim]Selecting HVGs with scanpy.pp.highly_variable_genes "
                    f"(flavor={hvg_flavor}, n_top_genes={n_top_genes})...[/dim]"
                )
                adata_hvg = ad.AnnData(X=counts_norm.copy())
                sc.pp.highly_variable_genes(
                    adata_hvg,
                    n_top_genes=n_top_genes,
                    flavor=hvg_flavor,
                    inplace=True,
                )
                hvg_mask = np.asarray(
                    adata_hvg.var["highly_variable"], dtype=bool
                )
                if np.sum(hvg_mask) == 0:
                    console.print(
                        "[yellow]⚠️[/yellow] [yellow]HVG selection returned no genes; using all retained genes[/yellow]"
                    )
                    hvg_mask = np.ones(counts_norm.shape[1], dtype=bool)
        elif use_hvg and counts_norm.shape[1] <= 1:
            console.print(
                "[yellow]⚠️[/yellow] [yellow]Too few genes for HVG selection; using retained genes directly[/yellow]"
            )

        counts_embed = counts_norm[:, hvg_mask]
        console.print(
            f"[dim]Embedding feature space: {counts_embed.shape[1]} genes[/dim]"
        )

        # 4) PCA (optional)
        embedding_input = counts_embed
        pca_model = None
        if use_pca and has_sklearn and counts_embed.shape[1] > 1:
            max_pcs = min(
                pca_n_comps,
                counts_embed.shape[1],
                max(1, counts_embed.shape[0] - 1),
            )
            if max_pcs >= 2:
                console.print(
                    f"[dim]Running PCA before UMAP (n_components={max_pcs})...[/dim]"
                )
                pca_model = PCA(
                    n_components=max_pcs, random_state=random_state
                )
                embedding_input = pca_model.fit_transform(counts_embed)
            else:
                console.print(
                    "[yellow]⚠️[/yellow] [yellow]Too few components/cells for PCA; fitting UMAP directly on gene space[/yellow]"
                )

        console.print(
            f"[dim]Fitting UMAP on experimental data "
            f"(n_neighbors={n_neighbors}, min_dist={min_dist})...[/dim]"
        )

        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        umap_embedding = umap_reducer.fit_transform(embedding_input)

        # Save to cache if caching is enabled
        if cache_path:
            console.print(
                f"[dim]💾 Saving UMAP cache to:[/dim] [cyan]{cache_path}[/cyan]"
            )
            try:
                cache_data = {
                    "reducer": umap_reducer,
                    "embedding": umap_embedding,
                    "params": current_params,
                    "preprocessing": {
                        "gene_mask": gene_mask,
                        "hvg_mask": hvg_mask,
                        "pca_model": pca_model,
                    },
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                console.print(
                    "[green]✅[/green] [dim]UMAP cache saved successfully[/dim]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]⚠️[/yellow] [yellow]Failed to save cache: {e}[/yellow]"
                )

    console.print(
        f"[dim]Generating {n_ppc_samples} predictive sample(s) for "
        f"synthetic dataset...[/dim]"
    )

    # Get batch_size from config for memory-efficient sampling
    batch_size = umap_opts.get("batch_size", None)

    # Gene-subset the results to only the genes that pass the detection
    # filter.  This avoids generating counts for genes that are immediately
    # discarded, keeping peak memory at (1, n_cells, n_gene_mask_genes)
    # regardless of n_ppc_samples.
    gene_mask_indices = np.where(gene_mask)[0]
    results_sub = results[gene_mask_indices]

    console.print(
        f"[dim]Gene-subsetted results to {int(gene_mask.sum())} detected "
        f"genes (out of {len(gene_mask)})[/dim]"
    )

    # Generate n_ppc_samples MAP-based PPC draws one at a time.  Each
    # sample is immediately preprocessed (normalize → HVG → PCA) and
    # projected through the fitted UMAP so only the 2-D coordinates are
    # kept, bounding memory to a single (n_cells, n_gene_mask_genes) array.
    all_umap_synthetic = []
    for i in range(n_ppc_samples):
        sample_arr = results_sub.get_map_ppc_samples(
            rng_key=random.PRNGKey(42 + i),
            n_samples=1,
            cell_batch_size=batch_size or 1000,
            use_mean=True,
            store_samples=False,
            verbose=(i == 0),
            counts=counts,
        )

        # (1, n_cells, n_gene_mask_genes) → (n_cells, n_gene_mask_genes)
        synth_i = np.array(sample_arr[0, :, :], dtype=np.float64)

        # Matched preprocessing: library-size normalize + log1p, HVG, PCA
        synth_norm = _normalize_log1p(synth_i)
        synth_embed = synth_norm[:, hvg_mask]
        if pca_model is not None:
            synth_embed = pca_model.transform(synth_embed)

        all_umap_synthetic.append(umap_reducer.transform(synth_embed))

        if (i + 1) % max(1, n_ppc_samples // 5) == 0 or i == n_ppc_samples - 1:
            console.print(
                f"[dim]  PPC sample {i + 1}/{n_ppc_samples} projected[/dim]"
            )

    # (n_ppc_samples * n_cells, 2)
    umap_synthetic = np.concatenate(all_umap_synthetic, axis=0)

    console.print("[dim]Creating overlay plot...[/dim]")

    # Scale scatter alpha and size for denser multi-sample synthetic cloud
    synth_alpha = 0.6 if n_ppc_samples == 1 else max(0.05, 0.6 / n_ppc_samples)
    synth_size = 1 if n_ppc_samples == 1 else max(0.2, 1.0 / np.sqrt(n_ppc_samples))

    # Create overlay plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot experimental data
    ax.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=data_color,
        alpha=0.6,
        s=1,
        label="experimental data",
    )

    # Plot synthetic data (alpha/size reduced for multi-sample overlays)
    ax.scatter(
        umap_synthetic[:, 0],
        umap_synthetic[:, 1],
        c=synthetic_color,
        alpha=synth_alpha,
        s=synth_size,
        label=f"synthetic data ({n_ppc_samples} sample{'s' if n_ppc_samples > 1 else ''})",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Projection: Experimental vs Synthetic Data")
    ax.legend(loc="best")

    plt.tight_layout()

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_umap.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved UMAP plot to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)


# ------------------------------------------------------------------------------


def _compute_correlation_matrix(samples, n_samples):
    """
    Compute pairwise Pearson correlation matrix from posterior samples.

    Centres and standardises the samples (z-scores) and then computes the
    sample Pearson correlation as :math:`(Z^T Z) / (n - 1)`.  Genes with
    zero standard deviation across samples are assigned unit self-correlation
    and zero cross-correlation.

    Parameters
    ----------
    samples : jnp.ndarray
        Posterior samples with shape ``(n_samples, n_genes)``.
    n_samples : int
        Number of posterior samples (must equal ``samples.shape[0]``).

    Returns
    -------
    jnp.ndarray
        Pearson correlation matrix with shape ``(n_genes, n_genes)``.
    """
    import jax.numpy as jnp

    # Centre (subtract column means)
    samples_centered = samples - jnp.mean(samples, axis=0, keepdims=True)

    # Standard deviations (avoid division by zero for constant genes)
    samples_std = jnp.std(samples, axis=0, keepdims=True)
    samples_std = jnp.where(samples_std == 0, 1.0, samples_std)

    # Standardise (z-scores)
    samples_standardized = samples_centered / samples_std

    # Correlation: (Z^T @ Z) / (n - 1)
    correlation_matrix = (samples_standardized.T @ samples_standardized) / (
        n_samples - 1
    )
    return correlation_matrix


# ------------------------------------------------------------------------------


def plot_correlation_heatmap(results, counts, figs_dir, cfg, viz_cfg):
    """
    Plot clustered correlation heatmap of gene parameters from posterior
    samples.

    Computes pairwise Pearson correlations between genes based on posterior
    samples of the gene expression parameter (``mu`` for linked / odds_ratio
    parameterizations, ``r`` for standard / canonical).  The top genes by
    correlation variance are selected and displayed as a hierarchically
    clustered heatmap with dendrograms.

    For **mixture models** (posterior samples with shape
    ``(n_samples, n_components, n_genes)``), a separate heatmap is produced
    for every mixture component.  Gene selection uses the *union* of the
    top-variance genes across all components so that every heatmap shares
    the same gene set, enabling direct visual comparison of the per-component
    correlation structures.

    Parameters
    ----------
    results : ScribeSVIResults
        Results object containing model parameters and posterior samples.
    counts : array-like, shape (n_cells, n_genes)
        Observed count matrix.  Required for amortized capture probability
        when generating posterior samples.
    figs_dir : str
        Directory in which to save the figure(s).
    cfg : DictConfig
        Original Hydra inference configuration.
    viz_cfg : DictConfig
        Visualization configuration.  Recognised keys inside
        ``viz_cfg.heatmap_opts``:

        * ``n_genes`` (int, default 1500) – number of genes to select per
          component (union may be larger).
        * ``n_samples`` (int, default 512) – number of posterior samples to
          draw when none exist yet.
        * ``figsize`` (int, default 12) – side length of the square heatmap.
        * ``cmap`` (str, default ``'RdBu_r'``) – colour map.
    """
    from jax import random
    import jax.numpy as jnp

    console.print("[dim]Plotting correlation heatmap...[/dim]")

    # Get options from config
    heatmap_opts = viz_cfg.get("heatmap_opts", {})
    n_genes_to_plot = heatmap_opts.get("n_genes", 1500)
    n_samples = heatmap_opts.get("n_samples", 512)
    figsize = heatmap_opts.get("figsize", 12)
    cmap = heatmap_opts.get("cmap", "RdBu_r")

    # Determine which parameter to use based on parameterization
    parameterization = results.model_config.parameterization
    # (standard, canonical) = r
    # (linked, mean_prob, odds_ratio, mean_odds) = mu
    if parameterization in ["linked", "mean_prob", "odds_ratio", "mean_odds"]:
        param_name = "mu"
    else:
        param_name = "r"

    console.print(
        f"[dim]Using parameter '{param_name}' for correlation "
        f"(parameterization: {parameterization})[/dim]"
    )

    # Generate posterior samples if they don't exist
    if results.posterior_samples is None:
        console.print(f"[dim]Generating {n_samples} posterior samples...[/dim]")
        # Pass full counts matrix for amortized capture probability
        # The amortizer needs total UMI count per cell, which is computed across all genes
        results.get_posterior_samples(
            rng_key=random.PRNGKey(42),
            n_samples=n_samples,
            store_samples=True,
            counts=counts,
        )
    else:
        console.print("[dim]Using existing posterior samples...[/dim]")
        # Update n_samples to match existing samples
        n_samples = results.posterior_samples[param_name].shape[0]
        console.print(f"[dim]Found {n_samples} existing samples[/dim]")

    # Get the posterior samples for the selected parameter
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

    # Common naming components
    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_"
        f"{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps"
    )

    # ------------------------------------------------------------------
    # Mixture models: one heatmap per component, shared gene set
    # ------------------------------------------------------------------
    if samples.ndim == 3:
        n_components = samples.shape[1]
        n_genes = samples.shape[2]
        n_genes_capped = min(n_genes_to_plot, n_genes)
        console.print(
            f"[dim]Detected mixture model with {n_components} components "
            f"– generating per-component heatmaps[/dim]"
        )

        # Compute correlation matrix and row-variance for each component
        console.print(
            "[dim]Computing per-component correlation matrices...[/dim]"
        )
        correlation_matrices = []
        variance_per_component = []

        for k in range(n_components):
            comp_samples = samples[:, k, :]  # (n_samples, n_genes)
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

        # Union of top-variance genes across all components
        selected_genes_set: set[int] = set()
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

        # Plot one heatmap per component using the shared gene set
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

    # ------------------------------------------------------------------
    # Single-component models: original behaviour
    # ------------------------------------------------------------------
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

        # Compute variance of each gene's correlation pattern
        correlation_variance = jnp.var(correlation_matrix, axis=1)

        # Get indices of top genes by variance
        top_var_indices = jnp.argsort(correlation_variance)[
            -n_genes_to_plot:
        ]
        top_var_indices = jnp.sort(top_var_indices)

        # Subset the correlation matrix
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

        # Convert to numpy for seaborn
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


# ------------------------------------------------------------------------------


def _select_divergent_genes(results, counts, n_rows, n_cols):
    """
    Select genes with highest divergence across components, binned by
    expression level.

    Uses the same log-spaced binning strategy as the standard PPC, but within
    each expression bin, selects genes with the highest maximum pairwise
    log-fold-change across mixture components.  This metric identifies genes
    where at least two components differ the most, which produces the most
    visually informative PPC panels regardless of the number of components.

    For two components the log-fold-change is monotonically equivalent to the
    coefficient of variation (CV), so behaviour is unchanged in the
    two-component case.

    Parameters
    ----------
    results : ScribeSVIResults
        Results object containing model parameters.
    counts : array-like
        Count matrix (n_cells, n_genes) for computing median expression.
    n_rows : int
        Number of expression bins.
    n_cols : int
        Number of genes per bin (max).

    Returns
    -------
    selected_idx : np.ndarray
        Indices of selected genes.
    divergence_values : np.ndarray
        Max pairwise log-fold-change values for selected genes.
    """
    import jax.numpy as jnp

    # Get MAP estimates
    # Pass full counts matrix for amortized capture probability
    # The amortizer needs total UMI count per cell, which is computed across all genes
    map_estimates = results.get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts
    )

    # Determine which parameter to use based on parameterization
    parameterization = results.model_config.parameterization
    # (standard, canonical) = r
    # (linked, mean_prob, odds_ratio, mean_odds) = mu
    if parameterization in ["linked", "mean_prob", "odds_ratio", "mean_odds"]:
        param_name = "mu"
    else:
        param_name = "r"

    # Get the parameter - should have shape (n_components, n_genes)
    param = map_estimates.get(param_name)
    if param is None:
        raise ValueError(
            f"Parameter '{param_name}' not found in MAP estimates. "
            f"Available: {list(map_estimates.keys())}"
        )

    # Compute max pairwise log-fold-change across components for each gene.
    # log(max_k / min_k) equals the largest absolute log-fold-change between
    # any pair of components.  Clamp the minimum to avoid log(0).
    param_clamped = jnp.clip(param, a_min=1e-10)
    lfc_range = np.array(
        jnp.log(jnp.max(param_clamped, axis=0))
        - jnp.log(jnp.min(param_clamped, axis=0))
    )

    # Compute median expression for binning
    counts_np = np.array(counts)
    median_counts = np.median(counts_np, axis=0)
    nonzero_idx = np.where(median_counts > 0)[0]

    if len(nonzero_idx) == 0:
        return np.array([], dtype=int), np.array([])

    # Sort genes by expression value
    sorted_order = np.argsort(median_counts[nonzero_idx])
    sorted_idx = nonzero_idx[sorted_order]
    sorted_medians = median_counts[sorted_idx]
    sorted_lfc = lfc_range[sorted_idx]

    # Get expression range
    min_expr = sorted_medians[0]
    max_expr = sorted_medians[-1]

    # Create logarithmically-spaced bin edges
    min_expr_safe = max(min_expr, 0.1)
    bin_edges = np.logspace(
        np.log10(min_expr_safe), np.log10(max_expr), num=n_rows + 1
    )
    bin_edges[0] = min_expr

    # Track selected genes
    selected_set = set()
    selected_by_bin = []

    # First pass: select top CV genes from each bin
    for i in range(n_rows):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]

        # Find genes in this expression range
        if i == n_rows - 1:
            in_bin = (sorted_medians >= bin_start) & (sorted_medians <= bin_end)
        else:
            in_bin = (sorted_medians >= bin_start) & (sorted_medians < bin_end)

        bin_indices = np.where(in_bin)[0]
        bin_selected = []

        if len(bin_indices) > 0:
            # Get log-fold-changes for genes in this bin
            bin_lfc = sorted_lfc[bin_indices]

            if len(bin_indices) <= n_cols:
                # Use all genes in bin
                bin_selected = list(bin_indices)
            else:
                # Select top n_cols genes by LFC within this bin
                top_lfc_order = np.argsort(bin_lfc)[::-1][:n_cols]
                bin_selected = list(bin_indices[top_lfc_order])

        selected_by_bin.append(bin_selected)
        selected_set.update(bin_selected)

    # Second pass: backfill bins that are short from previous bins
    # Create pool of unselected genes sorted by LFC
    all_indices = set(range(len(sorted_idx)))
    unselected_indices = np.array(sorted(list(all_indices - selected_set)))

    if len(unselected_indices) > 0:
        unselected_lfc = sorted_lfc[unselected_indices]
        unselected_medians = sorted_medians[unselected_indices]

        # Fill each bin to n_cols
        for i in range(n_rows):
            bin_selected = selected_by_bin[i]
            needed = n_cols - len(bin_selected)

            if needed > 0 and len(unselected_indices) > 0:
                bin_end = bin_edges[i + 1]

                # Find candidates: prefer genes from this or previous bins
                # (lower or equal expression)
                candidates_mask = unselected_medians <= bin_end
                candidates = unselected_indices[candidates_mask]
                candidate_lfc = unselected_lfc[candidates_mask]

                if len(candidates) == 0:
                    # If no candidates below, use all unselected
                    candidates = unselected_indices
                    candidate_lfc = unselected_lfc

                # Select top LFC candidates
                n_to_add = min(needed, len(candidates))
                top_lfc_order = np.argsort(candidate_lfc)[::-1][:n_to_add]
                to_add = candidates[top_lfc_order]

                bin_selected.extend(list(to_add))
                selected_set.update(to_add)

                # Remove from unselected pool
                mask = ~np.isin(unselected_indices, to_add)
                unselected_indices = unselected_indices[mask]
                unselected_lfc = unselected_lfc[mask]
                unselected_medians = unselected_medians[mask]

            selected_by_bin[i] = bin_selected

    # Flatten and convert back to original gene indices
    final_selected_sorted = []
    final_lfc = []
    for bin_selected in selected_by_bin:
        for idx in bin_selected:
            final_selected_sorted.append(sorted_idx[idx])
            final_lfc.append(sorted_lfc[idx])

    return np.array(final_selected_sorted), np.array(final_lfc)


# ------------------------------------------------------------------------------


def _get_component_ppc_samples(
    results,
    component_idx,
    n_samples,
    rng_key,
    cell_batch_size=500,
    verbose=True,
    counts=None,
):
    """
    Generate PPC samples for a specific mixture component.

    Samples as if all cells came from the specified component,
    returning shape (n_samples, n_cells, n_genes).

    Parameters
    ----------
    results : ScribeSVIResults
        Results object (should be subset to selected genes)
    component_idx : int
        Index of the component to sample from
    n_samples : int
        Number of samples to generate
    rng_key : jax.random.PRNGKey
        Random number generator key
    cell_batch_size : int, default=500
        Batch size for cell processing
    verbose : bool, default=True
        Print progress messages
    counts : array-like, optional
        Full count matrix (n_cells, n_genes). Required for amortized capture
        probability. Should be the full counts matrix, not subsetted.

    Returns
    -------
    jnp.ndarray
        Samples with shape (n_samples, n_cells, n_genes)
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist

    # Get MAP estimates
    # Pass full counts matrix for amortized capture probability
    # The amortizer needs total UMI count per cell, which is computed across all genes
    map_estimates = results.get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts
    )

    # Extract component-specific parameters
    r_all = map_estimates["r"]  # (n_components, n_genes)
    p_all = map_estimates["p"]

    r_k = r_all[component_idx]  # (n_genes,)
    n_genes = r_k.shape[0]

    # Handle p parameter
    if jnp.ndim(p_all) == 0:
        p_k = p_all  # scalar
    elif jnp.ndim(p_all) == 1:
        p_k = p_all[component_idx]  # scalar
    else:
        p_k = p_all[component_idx]  # (n_genes,)

    # Handle gate parameter if present (ZINB models)
    gate_k = None
    if "gate" in map_estimates:
        gate_all = map_estimates["gate"]
        if jnp.ndim(gate_all) > 1:
            gate_k = gate_all[component_idx]  # (n_genes,)
        else:
            gate_k = gate_all

    # Handle VCP
    p_capture = map_estimates.get("p_capture")
    has_vcp = p_capture is not None
    has_gate = gate_k is not None

    n_cells = results.n_cells
    all_samples = []
    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size

    # Use progress bar for batch processing if verbose and multiple batches
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

                # Compute effective p for this batch
                if has_vcp:
                    p_capture_batch = p_capture[start:end]  # (batch_size,)
                    p_capture_reshaped = p_capture_batch[
                        :, None
                    ]  # (batch_size, 1)
                    # p_k could be scalar or (n_genes,)
                    p_effective = (
                        p_k
                        * p_capture_reshaped
                        / (1 - p_k * (1 - p_capture_reshaped))
                    )  # (batch_size, n_genes) or (batch_size, 1)
                else:
                    p_effective = p_k

                # Create base NB distribution
                nb_dist = dist.NegativeBinomialProbs(r_k, p_effective)

                # Apply zero-inflation if present
                if has_gate:
                    sample_dist = dist.ZeroInflatedDistribution(
                        nb_dist, gate=gate_k
                    )
                else:
                    sample_dist = nb_dist

                # Sample counts
                if has_vcp:
                    # p_effective has shape (batch_size, n_genes), so sample gives
                    # (n_samples, batch_size, n_genes)
                    batch_samples = sample_dist.sample(batch_key, (n_samples,))
                else:
                    # Need to explicitly request batch dimension
                    batch_samples = sample_dist.sample(
                        batch_key, (n_samples, batch_size)
                    )

                all_samples.append(batch_samples)
                progress.update(task, advance=1)
    else:
        # No progress bar for single batch or non-verbose mode
        for batch_idx in range(n_batches):
            start = batch_idx * cell_batch_size
            end = min(start + cell_batch_size, n_cells)
            batch_size = end - start

            rng_key, batch_key = random.split(rng_key)

            # Compute effective p for this batch
            if has_vcp:
                p_capture_batch = p_capture[start:end]  # (batch_size,)
                p_capture_reshaped = p_capture_batch[:, None]  # (batch_size, 1)
                # p_k could be scalar or (n_genes,)
                p_effective = (
                    p_k
                    * p_capture_reshaped
                    / (1 - p_k * (1 - p_capture_reshaped))
                )  # (batch_size, n_genes) or (batch_size, 1)
            else:
                p_effective = p_k

            # Create base NB distribution
            nb_dist = dist.NegativeBinomialProbs(r_k, p_effective)

            # Apply zero-inflation if present
            if has_gate:
                sample_dist = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate_k
                )
            else:
                sample_dist = nb_dist

            # Sample counts
            if has_vcp:
                # p_effective has shape (batch_size, n_genes), so sample gives
                # (n_samples, batch_size, n_genes)
                batch_samples = sample_dist.sample(batch_key, (n_samples,))
            else:
                # Need to explicitly request batch dimension
                batch_samples = sample_dist.sample(
                    batch_key, (n_samples, batch_size)
                )

            all_samples.append(batch_samples)

    # Concatenate along cell dimension
    samples = jnp.concatenate(all_samples, axis=1)
    return samples


# ------------------------------------------------------------------------------


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
    """
    Plot a PPC figure in the standard format (same as plot_ppc).

    Parameters
    ----------
    predictive_samples : jnp.ndarray
        Samples with shape (n_samples, n_cells, n_genes_subset)
    counts : array-like
        Full count matrix (n_cells, n_genes_total)
    selected_idx : array-like
        Indices of selected genes in the full count matrix
    n_rows, n_cols : int
        Grid dimensions
    title : str
        Figure title
    figs_dir : str
        Directory to save figure
    fname : str
        Filename
    output_format : str
        Output format (png, pdf, etc.)
    cmap : str, default="Blues"
        Colormap for credible region shading
    """
    n_genes_selected = len(selected_idx)

    # Sort selected indices by median expression
    counts_np = np.array(counts)
    selected_means = np.array(
        [np.mean(counts_np[:, idx]) for idx in selected_idx]
    )
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]

    # Create mapping from gene index to position in subset
    # IMPORTANT: results[selected_idx] preserves the original gene order
    # (sorted by index), not the order of selected_idx. So we need to find
    # where each gene appears in the sorted original indices.
    # (Same logic as the original plot_ppc function)
    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    # Plotting
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        # Get the gene index in sorted order
        gene_idx = selected_idx_sorted[i]
        # Get the position of this gene in the samples
        subset_pos = subset_positions[gene_idx]

        true_counts = counts_np[:, gene_idx]

        # Compute credible regions from samples
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


# ------------------------------------------------------------------------------


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
    """
    Plot a comparison figure with mixture and all component PPCs overlaid.

    Parameters
    ----------
    mixture_samples : np.ndarray
        Mixture PPC samples with shape (n_samples, n_cells, n_genes_subset)
    component_samples_list : list of np.ndarray
        List of component PPC samples, each with shape (n_samples, n_cells,
        n_genes_subset)
    counts : array-like
        Full count matrix (n_cells, n_genes_total)
    selected_idx : array-like
        Indices of selected genes in the full count matrix
    n_rows, n_cols : int
        Grid dimensions
    figs_dir : str
        Directory to save figure
    fname : str
        Filename
    output_format : str
        Output format (png, pdf, etc.)
    component_cmaps : list of str, optional
        Colormaps for each component
    """
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

    # Sort selected indices by median expression
    counts_np = np.array(counts)
    selected_means = np.array(
        [np.mean(counts_np[:, idx]) for idx in selected_idx]
    )
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]

    # Create mapping from gene index to position in subset
    selected_idx_sorted_by_original = np.sort(selected_idx)
    subset_positions = {
        gene_idx: pos
        for pos, gene_idx in enumerate(selected_idx_sorted_by_original)
    }

    # Plotting
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows)
    )
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        # Get the gene index in sorted order
        gene_idx = selected_idx_sorted[i]
        # Get the position of this gene in the samples
        subset_pos = subset_positions[gene_idx]

        true_counts = counts_np[:, gene_idx]

        # Determine x-axis limit based on observed data (99th percentile)
        x_max = max(int(np.percentile(true_counts, 99)), 10)

        # Compute histogram for observed data with integer bins up to x_max
        obs_bins = np.arange(0, x_max + 2)  # +2 to include x_max bin
        hist_results = np.histogram(true_counts, bins=obs_bins, density=True)
        y_max = np.max(hist_results[0]) * 1.1  # 10% padding

        # Compute credible regions for mixture
        mixture_cr = scribe.stats.compute_histogram_credible_regions(
            mixture_samples[:, :, subset_pos],
            credible_regions=[95, 68, 50],
            max_bin=x_max,
        )

        # Plot each component PPC first (in background)
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

        # Plot mixture PPC (Blues) on top of components
        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, mixture_cr, cmap="Blues", alpha=0.3, max_bin=x_max
        )

        # Plot observed data histogram on top of everything
        ax.step(
            hist_results[1][:-1],  # bin edges (left edges)
            hist_results[0],
            where="post",
            label="data",
            color="black",
            linewidth=1.5,
        )

        # Set axis limits based on observed data only
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


# ------------------------------------------------------------------------------


def plot_mixture_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """
    Plot PPC for mixture models showing genes with highest divergence across
    components.

    Genes are selected by maximum pairwise log-fold-change across mixture
    components, binned by expression level.  This highlights genes where at
    least two components differ the most, producing the most visually
    informative PPC panels regardless of the number of components.

    Generates separate plots:

    - One for the combined mixture PPC.
    - One per component.
    - A comparison overlay (mixture + all components) **only when
      n_components <= 2**, since the limited colour-palette makes the overlay
      unreadable for more components.

    All plots use the same format as the standard PPC (credible regions).

    Parameters
    ----------
    results : ScribeSVIResults
        Results object containing model parameters
    counts : array-like
        Observed count data (n_cells, n_genes)
    figs_dir : str
        Directory to save the figure
    cfg : DictConfig
        Original inference configuration
    viz_cfg : DictConfig
        Visualization configuration
    """
    console.print(
        "[dim]Plotting mixture model PPC (genes with highest CV across components)...[/dim]"
    )

    # Check if this is a mixture model
    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping mixture PPC plot...[/yellow]"
        )
        return

    # Get options from config - use same defaults as regular PPC
    ppc_opts = viz_cfg.get("ppc_opts", {})
    mixture_ppc_opts = viz_cfg.get("mixture_ppc_opts", {})
    # Use ppc_opts as default, allow mixture_ppc_opts to override if provided
    n_rows = mixture_ppc_opts.get("n_rows", ppc_opts.get("n_rows", 5))
    n_cols = mixture_ppc_opts.get("n_cols", ppc_opts.get("n_cols", 5))
    n_samples = mixture_ppc_opts.get(
        "n_samples", ppc_opts.get("n_samples", 1500)
    )
    console.print(
        f"[dim]Selecting high-CV genes from {n_rows} expression bins "
        f"({n_cols} genes/bin) across {n_components} components...[/dim]"
    )

    # Select genes with highest divergence (max pairwise log-fold-change)
    # within expression bins
    top_gene_indices, top_lfc = _select_divergent_genes(
        results, counts, n_rows, n_cols
    )
    top_gene_indices = np.array(top_gene_indices)
    n_genes_to_plot = len(top_gene_indices)

    console.print(
        f"[dim]Selected {n_genes_to_plot} genes. Top log-fold-changes:[/dim] "
        f"{np.array(top_lfc[:5])}"
    )

    # Subset results to selected genes for memory efficiency
    results_subset = results[top_gene_indices]

    # Get output format and config values
    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps"
    )

    rng_key = random.PRNGKey(42)

    # --- Plot 1: Combined Mixture PPC ---
    console.print(
        f"[dim]Generating mixture PPC samples ({n_samples} samples)...[/dim]"
    )
    rng_key, subkey = random.split(rng_key)
    # Pass full counts matrix (not subsetted) for amortized capture probability
    # The amortizer needs total UMI count per cell, which is computed across all genes
    mixture_samples = results_subset.get_map_ppc_samples(
        rng_key=subkey,
        n_samples=n_samples,
        cell_batch_size=500,
        store_samples=False,
        verbose=True,
        counts=counts,
    )

    # Store mixture samples for comparison plot
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

    del mixture_samples  # Free JAX array

    # --- Plot 2+: Per-Component PPCs ---
    # Different colormaps for each component (cycle if more components than
    # colors)
    component_cmaps = ["Greens", "Purples", "Reds", "Oranges", "YlOrBr", "BuGn"]
    min_cells_for_info = mixture_ppc_opts.get("min_cells_for_info", 20)
    all_component_samples = []
    counts_np = np.array(counts)

    # Compute hard cell assignments once using MAP probabilities.
    console.print("[dim]Computing MAP cell-to-component assignments...[/dim]")
    assignment_probs = results.cell_type_probabilities_map(
        counts=counts, verbose=False
    )["probabilities"]
    assignments = np.argmax(np.array(assignment_probs), axis=1)

    for k in range(n_components):
        console.print(
            f"[dim]Generating Component {k+1} PPC samples ({n_samples} samples)...[/dim]"
        )
        rng_key, subkey = random.split(rng_key)

        # Pass full counts matrix (not subsetted) for amortized capture probability
        # The amortizer needs total UMI count per cell, which is computed across all genes
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
        # Keep full-cell samples for optional comparison plot.
        all_component_samples.append(component_samples_np)

        # Subset both observed and predicted data to cells assigned to component k.
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

        # Cycle through colormaps if more components than colors
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

        del component_samples  # Free JAX array

    # --- Plot 3: Combined Comparison Plot (only for <= 2 components) ---
    # With >2 components the overlaid colormaps become unreadable (limited
    # palette) and the per-component plots already convey the information.
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

    # Clean up
    del mixture_samples_np, all_component_samples

    console.print(
        f"[green]✓[/green] [dim]Generated {n_plots} mixture PPC plots[/dim]"
    )

    # Clean up
    del results_subset


# ------------------------------------------------------------------------------


def _reconstruct_label_map(cell_labels, component_order=None):
    """
    Reconstruct the label-to-component-index mapping used during inference.

    This must match the logic in
    ``scribe.core.annotation_prior.build_annotation_prior_logits``: when no
    explicit *component_order* is given, unique labels are sorted
    alphabetically; otherwise the supplied order defines the mapping.

    Parameters
    ----------
    cell_labels : array-like, shape (n_cells,)
        Annotation label for every cell (strings).  ``NaN`` / ``None``
        entries are treated as unlabelled and excluded from the mapping.
    component_order : list of str or None, optional
        Explicit label ordering.  Position *i* in the list maps to
        component index *i*.

    Returns
    -------
    dict[str, int]
        Mapping from label string to component index.
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


# ------------------------------------------------------------------------------


def plot_annotation_ppc(results, counts, cell_labels, figs_dir, cfg, viz_cfg):
    """
    Plot per-annotation posterior predictive checks.

    For each unique annotation label, a PPC figure is generated that compares
    the observed count distribution of cells belonging to that annotation
    against the posterior predictive distribution of the corresponding mixture
    component.  This enables visual assessment of how well each component
    captures the expression profile of its assigned cell population.

    Component ordering follows the same convention used during inference:
    labels are sorted alphabetically by default, or according to
    ``annotation_component_order`` if provided in the configuration.

    Parameters
    ----------
    results : ScribeSVIResults
        Inference results for a mixture model.
    counts : np.ndarray, shape (n_cells, n_genes)
        Observed count matrix.
    cell_labels : np.ndarray or pd.Series, shape (n_cells,)
        Annotation label for each cell.  Must match the labels used during
        inference (same column(s) and same ordering convention).
    figs_dir : str
        Directory in which to save the figure(s).
    cfg : DictConfig
        Original Hydra inference configuration.
    viz_cfg : DictConfig
        Visualization configuration.  Recognised keys inside
        ``viz_cfg.annotation_ppc_opts`` (falls back to ``ppc_opts``):

        * ``n_rows`` (int) – number of expression bins.
        * ``n_cols`` (int) – number of genes per bin.
        * ``n_samples`` (int) – number of PPC samples per component.
    """
    from jax import random

    console.print("[dim]Plotting annotation PPC...[/dim]")

    # Validate mixture model
    n_components = results.n_components
    if n_components is None or n_components <= 1:
        console.print(
            "[yellow]Not a mixture model, skipping annotation "
            "PPC...[/yellow]"
        )
        return

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------
    ppc_opts = viz_cfg.get("ppc_opts", {})
    ann_opts = viz_cfg.get("annotation_ppc_opts", {})
    n_rows = ann_opts.get("n_rows", ppc_opts.get("n_rows", 5))
    n_cols = ann_opts.get("n_cols", ppc_opts.get("n_cols", 5))
    n_samples = ann_opts.get("n_samples", ppc_opts.get("n_samples", 1500))

    # ------------------------------------------------------------------
    # Reconstruct label → component mapping
    # ------------------------------------------------------------------
    component_order = cfg.get("annotation_component_order", None)
    label_map = _reconstruct_label_map(cell_labels, component_order)

    import pandas as pd

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

    # ------------------------------------------------------------------
    # Gene selection (standard expression-based)
    # ------------------------------------------------------------------
    selected_idx, _mean_counts = _select_genes(counts, n_rows, n_cols)
    n_genes_selected = len(selected_idx)
    console.print(
        f"[dim]Selected {n_genes_selected} genes across {n_rows} "
        f"expression bins[/dim]"
    )

    # ------------------------------------------------------------------
    # Filename components
    # ------------------------------------------------------------------
    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    base_fname = (
        f"{config_vals['method']}_"
        f"{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps"
    )

    # ------------------------------------------------------------------
    # Per-annotation PPC
    # ------------------------------------------------------------------
    component_cmaps = [
        "Blues", "Greens", "Purples", "Reds", "Oranges", "YlOrBr", "BuGn",
    ]
    rng_key = random.PRNGKey(42)

    for label, component_idx in label_map.items():
        console.print(
            f"[dim]Processing annotation '{label}' "
            f"(component {component_idx})...[/dim]"
        )

        # Cell mask for this annotation
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

        # Extract single-component view and subset to selected genes
        component_results = results.get_component(component_idx)
        component_subset = component_results[selected_idx]

        # Generate PPC samples for this component (all cells)
        rng_key, subkey = random.split(rng_key)
        console.print(
            f"[dim]  Generating {n_samples} PPC samples...[/dim]"
        )
        component_samples = component_subset.get_map_ppc_samples(
            rng_key=subkey,
            n_samples=n_samples,
            cell_batch_size=500,
            store_samples=False,
            verbose=True,
            counts=counts,
        )

        # Subset PPC samples and observed counts to this annotation's cells
        component_samples_np = np.array(
            component_samples[:, cell_mask, :]
        )
        counts_label = counts[cell_mask, :]

        # Sanitize label for safe filenames
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

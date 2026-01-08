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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jax import random
import scribe

# ==============================================================================
# Helper functions
# ==============================================================================


def _get_config_values(cfg):
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

    Parameters
    ----------
    cfg : OmegaConf.DictConfig or dict-like
        The configuration object from which to extract values.

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

    # Attempt to extract model type from cfg.model.type if present, otherwise
    # default to 'default' getattr returns the value of 'type' in cfg.model if
    # it exists, else 'default'
    model_type = getattr(cfg, "model", {}).get("type", "default")

    # Return all extracted values in a dictionary for easy access
    return {
        "parameterization": parameterization,
        "n_components": n_components,
        "n_steps": n_steps,
        "method": method,
        "model_type": model_type,
    }


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
    print("Plotting loss history...")

    # Initialize figure
    fig, ax = plt.subplots(figsize=(3.5, 3))
    # Plot loss history
    ax.plot(results.loss_history)

    # Set labels
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO loss")

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename from original config
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_loss.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved loss plot to {output_path}")
    plt.close(fig)


# ==============================================================================
# Visualization functions
# ==============================================================================


def plot_ecdf(counts, figs_dir, cfg, viz_cfg):
    """Plot and save the ECDF of selected genes."""
    print("Plotting ECDF...")

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
    print(f"Saved ECDF plot to {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------------------


def plot_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """Plot and save the posterior predictive checks."""
    print("Plotting PPC...")

    # Gene selection using log-spaced binning
    n_rows = viz_cfg.ppc_opts.n_rows
    n_cols = viz_cfg.ppc_opts.n_cols
    print(
        f"Using n_rows={n_rows}, n_cols={n_cols} for PPC plot (log-spaced binning)"
    )
    selected_idx, mean_counts = _select_genes(counts, n_rows, n_cols)

    # Sort selected indices by median expression (ascending)
    # This ensures genes are plotted from lowest to highest expression
    selected_means = mean_counts[selected_idx]
    sort_order = np.argsort(selected_means)
    selected_idx_sorted = selected_idx[sort_order]
    n_genes_selected = len(selected_idx_sorted)
    print(f"Selected {n_genes_selected} genes across {n_rows} expression bins")

    # Index results for selected genes (using original unsorted order for subsetting)
    results_subset = results[selected_idx]

    # Generate posterior predictive samples
    n_samples = viz_cfg.ppc_opts.n_samples
    print(f"Generating {n_samples} posterior predictive samples...")
    results_subset.get_ppc_samples(n_samples=n_samples)

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

    for i, ax in enumerate(axes):
        if i >= n_genes_selected:
            ax.axis("off")
            continue

        print(f"Plotting gene {i} PPC...")

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
        # Compute actual mean expression (mean_counts is actually median, used
        # for selection)
        actual_mean_expr = np.mean(counts[:, gene_idx])
        mean_expr_formatted = f"{actual_mean_expr:.2f}"
        ax.set_title(
            f"$\\langle U \\rangle = {mean_expr_formatted}$",
            fontsize=8,
        )

    plt.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_ppc.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved PPC plot to {output_path}")
    plt.close(fig)

    # Clean up results_subset to free memory (it contains samples for subset of
    # genes) The original results object doesn't have samples, so no cleanup
    # needed there
    del results_subset


# ------------------------------------------------------------------------------


def plot_umap(results, counts, figs_dir, cfg, viz_cfg):
    """Plot UMAP projection of experimental and synthetic data."""
    import pickle

    print("Plotting UMAP projection...")

    umap_opts = viz_cfg.get("umap_opts", {})

    try:
        import umap
    except ImportError:
        print(
            "❌ ERROR: umap-learn is not installed."
            " Install it with: pip install umap-learn"
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

    # Determine cache path (same directory as data file)
    data_path = cfg.data.path if hasattr(cfg, "data") else None
    cache_path = None
    if data_path and cache_umap:
        data_dir = os.path.dirname(data_path)
        cache_path = os.path.join(data_dir, "2d_umap.pkl")

    # Current UMAP parameters for cache validation
    current_params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "n_components": n_components,
        "random_state": random_state,
        "n_cells": counts.shape[0],
        "n_genes": counts.shape[1],
    }

    # Try to load cached UMAP
    umap_reducer = None
    umap_embedding = None

    if cache_path and os.path.exists(cache_path):
        print(f"Found cached UMAP at: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)

            # Validate cached parameters match current parameters
            cached_params = cached.get("params", {})
            params_match = all(
                cached_params.get(k) == v for k, v in current_params.items()
            )

            if params_match:
                print("✅ Cache valid - loading UMAP from cache...")
                umap_reducer = cached["reducer"]
                umap_embedding = cached["embedding"]
            else:
                print("⚠️  Cache parameters mismatch - will re-fit UMAP")
                print(f"   Cached: {cached_params}")
                print(f"   Current: {current_params}")
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e} - will re-fit UMAP")

    # Fit UMAP if not loaded from cache
    if umap_reducer is None:
        print(
            f"Fitting UMAP on experimental data "
            f"(n_neighbors={n_neighbors}, min_dist={min_dist})..."
        )

        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        umap_embedding = umap_reducer.fit_transform(counts)

        # Save to cache if caching is enabled
        if cache_path:
            print(f"💾 Saving UMAP cache to: {cache_path}")
            try:
                cache_data = {
                    "reducer": umap_reducer,
                    "embedding": umap_embedding,
                    "params": current_params,
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                print("✅ UMAP cache saved successfully")
            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")

    print("Generating single predictive sample for synthetic dataset...")

    # Get batch_size from config for memory-efficient sampling
    batch_size = umap_opts.get("batch_size", None)

    # For UMAP, we need samples for ALL genes (not just the subset used in PPC
    # plots). Use the memory-efficient MAP-based sampling method.
    if results.predictive_samples is None:
        print("Using MAP estimates for memory-efficient predictive sampling...")
        if batch_size is not None:
            print(f"Using cell_batch_size={batch_size} for cell batching...")

        # Use the new memory-efficient method that samples using MAP estimates
        # and processes cells in batches to avoid OOM for VCP models
        predictive_samples = results.get_map_ppc_samples(
            rng_key=random.PRNGKey(42),
            n_samples=1,
            cell_batch_size=batch_size or 1000,
            use_mean=True,
            store_samples=True,
            verbose=True,
        )
        # Extract single sample: shape is (1, n_cells, n_genes) -> (n_cells, n_genes)
        synthetic_data = predictive_samples[0, :, :]
    else:
        # Use existing predictive samples
        print("Using existing predictive samples...")
        # Extract single sample: shape should be (n_cells, n_genes)
        # predictive_samples shape is (n_samples, n_cells, n_genes)
        if results.predictive_samples.ndim == 3:
            synthetic_data = results.predictive_samples[0, :, :]
        else:
            # If shape is (n_cells, n_genes), use directly
            synthetic_data = results.predictive_samples

    # Convert to numpy array (CPU) to avoid memory issues and for UMAP
    # compatibility
    if hasattr(synthetic_data, "block_until_ready"):
        # JAX array - convert to numpy
        synthetic_data = np.array(synthetic_data)
    elif not isinstance(synthetic_data, np.ndarray):
        # Other array type - convert to numpy
        synthetic_data = np.array(synthetic_data)

    print("Projecting synthetic data onto UMAP space...")

    # Project synthetic data onto the same UMAP space
    umap_synthetic = umap_reducer.transform(synthetic_data)

    print("Creating overlay plot...")

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

    # Plot synthetic data
    ax.scatter(
        umap_synthetic[:, 0],
        umap_synthetic[:, 1],
        c=synthetic_color,
        alpha=0.6,
        s=1,
        label="synthetic data",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Projection: Experimental vs Synthetic Data")
    ax.legend(loc="best")

    plt.tight_layout()

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_umap.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved UMAP plot to {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------------------


def plot_correlation_heatmap(results, figs_dir, cfg, viz_cfg):
    """
    Plot clustered correlation heatmap of gene parameters from posterior
    samples.

    This function computes pairwise Pearson correlations between genes based on
    posterior samples of the gene expression parameter (mu for linked/odds_ratio
    parameterization, r for standard parameterization). It then selects the top
    genes by correlation variance and displays them as a hierarchically
    clustered heatmap with dendrograms.

    Parameters
    ----------
    results : ScribeSVIResults
        Results object containing model parameters and posterior samples
    figs_dir : str
        Directory to save the figure
    cfg : DictConfig
        Original inference configuration
    viz_cfg : DictConfig
        Visualization configuration containing heatmap_opts
    """
    from jax import random
    import jax.numpy as jnp

    print("Plotting correlation heatmap...")

    # Get options from config
    heatmap_opts = viz_cfg.get("heatmap_opts", {})
    n_genes_to_plot = heatmap_opts.get("n_genes", 500)
    n_samples = heatmap_opts.get("n_samples", 256)
    figsize = heatmap_opts.get("figsize", 12)
    cmap = heatmap_opts.get("cmap", "RdBu_r")

    # Determine which parameter to use based on parameterization
    parameterization = results.model_config.parameterization
    if parameterization in ["linked", "odds_ratio"]:
        param_name = "mu"
    else:
        param_name = "r"

    print(
        f"Using parameter '{param_name}' for correlation "
        f"(parameterization: {parameterization})"
    )

    # Generate posterior samples if they don't exist
    if results.posterior_samples is None:
        print(f"Generating {n_samples} posterior samples...")
        results.get_posterior_samples(
            rng_key=random.PRNGKey(42),
            n_samples=n_samples,
            store_samples=True,
        )
    else:
        print("Using existing posterior samples...")
        # Update n_samples to match existing samples
        n_samples = results.posterior_samples[param_name].shape[0]
        print(f"Found {n_samples} existing samples")

    # Get the posterior samples for the selected parameter
    if param_name not in results.posterior_samples:
        print(
            f"❌ ERROR: Parameter '{param_name}' "
            "not found in posterior samples."
        )
        print(
            f"   Available parameters: "
            f"{list(results.posterior_samples.keys())}"
        )
        return

    samples = results.posterior_samples[param_name]

    # Handle mixture models - samples may have shape
    # (n_samples, n_components, n_genes)
    # For now, take the first component or flatten
    if samples.ndim == 3:
        print(f"Detected mixture model with {samples.shape[1]} components")
        print("Using first component for correlation heatmap...")
        samples = samples[:, 0, :]

    print(f"Sample shape: {samples.shape}")
    n_genes = samples.shape[1]

    # Ensure we don't try to plot more genes than available
    n_genes_to_plot = min(n_genes_to_plot, n_genes)

    print("Computing pairwise Pearson correlations...")

    # Center the data (subtract column means)
    samples_centered = samples - jnp.mean(samples, axis=0, keepdims=True)

    # Compute standard deviations
    samples_std = jnp.std(samples, axis=0, keepdims=True)

    # Avoid division by zero for constant genes
    samples_std = jnp.where(samples_std == 0, 1.0, samples_std)

    # Standardize the data (z-scores)
    samples_standardized = samples_centered / samples_std

    # Compute correlation matrix: (X^T @ X) / (n_samples - 1)
    correlation_matrix = (samples_standardized.T @ samples_standardized) / (
        n_samples - 1
    )

    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(
        f"Correlation range: [{float(jnp.min(correlation_matrix)):.3f}, "
        f"{float(jnp.max(correlation_matrix)):.3f}]"
    )

    print(f"Selecting top {n_genes_to_plot} genes by correlation variance...")

    # Compute variance of each gene's correlation pattern
    correlation_variance = jnp.var(correlation_matrix, axis=1)

    # Get indices of top genes by variance
    top_var_indices = jnp.argsort(correlation_variance)[-n_genes_to_plot:]
    top_var_indices = jnp.sort(
        top_var_indices
    )  # Sort for easier interpretation

    # Subset the correlation matrix
    correlation_subset = correlation_matrix[top_var_indices, :][
        :, top_var_indices
    ]

    print(f"Selected {n_genes_to_plot} genes with highest correlation variance")
    print(f"Subset correlation matrix shape: {correlation_subset.shape}")

    # Convert to numpy for seaborn
    correlation_subset_np = np.array(correlation_subset)

    print("Creating clustered heatmap...")

    # Create clustered heatmap with dendrograms
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

    # Add title
    fig.fig.suptitle(
        f"Gene Correlation Structure (Top {n_genes_to_plot} by Variance)\n"
        f"Parameter: {param_name} | Samples: {n_samples}",
        y=1.02,
        fontsize=12,
    )

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_correlation_heatmap.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved correlation heatmap to {output_path}")
    plt.close(fig.fig)


# ------------------------------------------------------------------------------


def _select_divergent_genes(results, counts, n_rows, n_cols):
    """
    Select genes with highest CV across components, binned by expression level.

    Uses the same log-spaced binning strategy as the standard PPC, but within
    each expression bin, selects genes with highest coefficient of variation
    (CV) across mixture components instead of log-spacing.

    Parameters
    ----------
    results : ScribeSVIResults
        Results object containing model parameters
    counts : array-like
        Count matrix (n_cells, n_genes) for computing median expression
    n_rows : int
        Number of expression bins
    n_cols : int
        Number of genes per bin (max)

    Returns
    -------
    selected_idx : np.ndarray
        Indices of selected genes
    cv_values : np.ndarray
        CV values for selected genes
    """
    import jax.numpy as jnp

    # Get MAP estimates
    map_estimates = results.get_map(
        use_mean=True, canonical=True, verbose=False
    )

    # Determine which parameter to use based on parameterization
    parameterization = results.model_config.parameterization
    if parameterization in ["linked", "odds_ratio"]:
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

    # Compute CV across components for each gene
    param_std = jnp.std(param, axis=0)
    param_mean = jnp.mean(param, axis=0)
    param_mean = jnp.where(param_mean == 0, 1e-10, param_mean)
    cv_all = np.array(param_std / param_mean)

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
    sorted_cvs = cv_all[sorted_idx]

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
            # Get CVs for genes in this bin
            bin_cvs = sorted_cvs[bin_indices]

            if len(bin_indices) <= n_cols:
                # Use all genes in bin
                bin_selected = list(bin_indices)
            else:
                # Select top n_cols genes by CV within this bin
                top_cv_order = np.argsort(bin_cvs)[::-1][:n_cols]
                bin_selected = list(bin_indices[top_cv_order])

        selected_by_bin.append(bin_selected)
        selected_set.update(bin_selected)

    # Second pass: backfill bins that are short from previous bins
    # Create pool of unselected genes sorted by CV
    all_indices = set(range(len(sorted_idx)))
    unselected_indices = np.array(sorted(list(all_indices - selected_set)))

    if len(unselected_indices) > 0:
        unselected_cvs = sorted_cvs[unselected_indices]
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
                candidate_cvs = unselected_cvs[candidates_mask]

                if len(candidates) == 0:
                    # If no candidates below, use all unselected
                    candidates = unselected_indices
                    candidate_cvs = unselected_cvs

                # Select top CV candidates
                n_to_add = min(needed, len(candidates))
                top_cv_order = np.argsort(candidate_cvs)[::-1][:n_to_add]
                to_add = candidates[top_cv_order]

                bin_selected.extend(list(to_add))
                selected_set.update(to_add)

                # Remove from unselected pool
                mask = ~np.isin(unselected_indices, to_add)
                unselected_indices = unselected_indices[mask]
                unselected_cvs = unselected_cvs[mask]
                unselected_medians = unselected_medians[mask]

            selected_by_bin[i] = bin_selected

    # Flatten and convert back to original gene indices
    final_selected_sorted = []
    final_cvs = []
    for bin_selected in selected_by_bin:
        for idx in bin_selected:
            final_selected_sorted.append(sorted_idx[idx])
            final_cvs.append(sorted_cvs[idx])

    return np.array(final_selected_sorted), np.array(final_cvs)


# ------------------------------------------------------------------------------


def _get_component_ppc_samples(
    results,
    component_idx,
    n_samples,
    rng_key,
    cell_batch_size=500,
    verbose=True,
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

    Returns
    -------
    jnp.ndarray
        Samples with shape (n_samples, n_cells, n_genes)
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist

    # Get MAP estimates
    map_estimates = results.get_map(
        use_mean=True, canonical=True, verbose=False
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

    for batch_idx in range(n_batches):
        start = batch_idx * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        batch_size = end - start

        if verbose and n_batches > 1 and batch_idx % 5 == 0:
            print(f"    Processing cells {start}-{end} of {n_cells}...")

        rng_key, batch_key = random.split(rng_key)

        # Compute effective p for this batch
        if has_vcp:
            p_capture_batch = p_capture[start:end]  # (batch_size,)
            p_capture_reshaped = p_capture_batch[:, None]  # (batch_size, 1)
            # p_k could be scalar or (n_genes,)
            p_effective = (
                p_k * p_capture_reshaped / (1 - p_k * (1 - p_capture_reshaped))
            )  # (batch_size, n_genes) or (batch_size, 1)
        else:
            p_effective = p_k

        # Create base NB distribution
        nb_dist = dist.NegativeBinomialProbs(r_k, p_effective)

        # Apply zero-inflation if present
        if has_gate:
            sample_dist = dist.ZeroInflatedDistribution(nb_dist, gate=gate_k)
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
    print(f"Saved {title} to {output_path}")
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

        # Compute credible regions for mixture
        mixture_cr = scribe.stats.compute_histogram_credible_regions(
            mixture_samples[:, :, subset_pos],
            credible_regions=[95, 68, 50],
        )

        # Compute histogram for observed data using mixture bins
        hist_results = np.histogram(
            true_counts, bins=mixture_cr["bin_edges"], density=True
        )

        cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
        max_bin = np.max(
            [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10]
        )

        # Plot mixture PPC (Blues) - in background
        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, mixture_cr, cmap="Blues", alpha=0.3, max_bin=max_bin
        )

        # Plot each component PPC overlaid with reduced alpha
        for k, comp_samples in enumerate(component_samples_list):
            comp_cr = scribe.stats.compute_histogram_credible_regions(
                comp_samples[:, :, subset_pos],
                credible_regions=[95, 68, 50],
            )
            cmap = component_cmaps[k % len(component_cmaps)]
            scribe.viz.plot_histogram_credible_regions_stairs(
                ax, comp_cr, cmap=cmap, alpha=0.4, max_bin=max_bin
            )

        # Plot observed data histogram on top
        max_bin_hist = (
            max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
        )
        ax.step(
            hist_results[1][:max_bin_hist],
            hist_results[0][:max_bin_hist],
            where="post",
            label="data",
            color="black",
            linewidth=1.5,
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
    fig.suptitle("PPC Comparison: Mixture vs Components", y=1.02)

    output_path = os.path.join(figs_dir, f"{fname}.{output_format}")
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved PPC comparison to {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------------------


def plot_mixture_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """
    Plot PPC for mixture models showing genes with highest CV across components.

    Generates separate plots:
    - One for the combined mixture PPC
    - One per component

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
    print(
        "Plotting mixture model PPC (genes with highest CV across components)..."
    )

    # Check if this is a mixture model
    n_components = results.n_components
    if n_components is None or n_components <= 1:
        print("Not a mixture model, skipping mixture PPC plot...")
        return

    # Get options from config
    mixture_ppc_opts = viz_cfg.get("mixture_ppc_opts", {})
    n_rows = mixture_ppc_opts.get("n_rows", 4)
    n_cols = mixture_ppc_opts.get("n_cols", 4)
    n_samples = mixture_ppc_opts.get("n_samples", 1500)
    print(
        f"Selecting high-CV genes from {n_rows} expression bins "
        f"({n_cols} genes/bin) across {n_components} components..."
    )

    # Select genes with highest CV within expression bins
    top_gene_indices, top_cvs = _select_divergent_genes(
        results, counts, n_rows, n_cols
    )
    top_gene_indices = np.array(top_gene_indices)
    n_genes_to_plot = len(top_gene_indices)

    print(f"Selected {n_genes_to_plot} genes. Top CVs: {np.array(top_cvs[:5])}")

    # Subset results to selected genes for memory efficiency
    results_subset = results[top_gene_indices]

    # Get output format and config values
    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg)
    base_fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps"
    )

    rng_key = random.PRNGKey(42)

    # --- Plot 1: Combined Mixture PPC ---
    print(f"\nGenerating mixture PPC samples ({n_samples} samples)...")
    rng_key, subkey = random.split(rng_key)
    mixture_samples = results_subset.get_map_ppc_samples(
        rng_key=subkey,
        n_samples=n_samples,
        cell_batch_size=500,
        store_samples=False,
        verbose=True,
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
    all_component_samples = []

    for k in range(n_components):
        print(
            f"\nGenerating Component {k+1} PPC samples ({n_samples} samples)..."
        )
        rng_key, subkey = random.split(rng_key)

        component_samples = _get_component_ppc_samples(
            results_subset,
            component_idx=k,
            n_samples=n_samples,
            rng_key=subkey,
            cell_batch_size=500,
            verbose=True,
        )

        component_samples_np = np.array(component_samples)
        all_component_samples.append(component_samples_np)

        # Cycle through colormaps if more components than colors
        cmap = component_cmaps[k % len(component_cmaps)]

        _plot_ppc_figure(
            predictive_samples=component_samples_np,
            counts=counts,
            selected_idx=top_gene_indices,
            n_rows=n_rows,
            n_cols=n_cols,
            title=f"Component {k+1} PPC (High CV Genes)",
            figs_dir=figs_dir,
            fname=f"{base_fname}_component{k+1}_ppc",
            output_format=output_format,
            cmap=cmap,
        )

        del component_samples  # Free JAX array

    # --- Plot 3: Combined Comparison Plot ---
    print("\nGenerating combined comparison plot...")
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

    # Clean up
    del mixture_samples_np, all_component_samples

    print(f"\nGenerated {2 + n_components} mixture PPC plots")

    # Clean up
    del results_subset

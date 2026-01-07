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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
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
        # Format mean expression to one significant figure
        mean_expr = mean_counts[gene_idx]
        mean_expr_formatted = f"{mean_expr:.2f}"
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

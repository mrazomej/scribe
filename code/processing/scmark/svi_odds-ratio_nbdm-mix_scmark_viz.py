# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import pickle

# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import library for reading 10x Genomics data
import anndata as ad

# Set plotting style
scribe.viz.matplotlib_style()

# %% ---------------------------------------------------------------------------

# Define model_type
model_type = "nbdm_mix"
# Define parameterization
parameterization = "odds_ratio"

# Define training parameters
n_steps = 50_000
batch_size = 2048

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"/app/data/scmark_v2/scmark_v2/"

# Define model directory
MODEL_DIR = f"{scribe.utils.git_root()}/output/scmark/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/scmark/{model_type}"

# Create figure directory if it does not exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# List all files in the data directory
files = sorted(glob.glob(f"{DATA_DIR}/*.h5ad", recursive=True))

# %% ---------------------------------------------------------------------------

# Loop through each file
for file in files:

    # Load data
    data = ad.read_h5ad(file)

    # Find genes with all zero counts
    zero_counts = np.all(data.X.toarray() == 0, axis=0)

    # Remove all zero genes
    data = data[:, ~zero_counts]

    # Extract counts
    counts = data.X.toarray()

    # Extract dataset name from file name
    dataset_name = file.split("/")[-1].split(".")[0]

    print(f"Processing {dataset_name}")

    # Define number of components
    n_components = len(data.obs["standard_true_celltype"].unique())

    # Define output file name
    file_name = f"{MODEL_DIR}/" \
                f"svi_{model_type.replace('_', '-')}_" \
                f"{parameterization.replace('_', '-')}_" \
                f"{n_components}components_" \
                f"{n_steps}steps_" \
                f"{batch_size}batch_" \
                f"{dataset_name}.pkl"

    # Check if the file exists
    if not os.path.exists(file_name):
        print(f"Skipping {file} because it does not exist")
        continue

    print("Loading model...")
    # Load model
    with open(file_name, "rb") as f:
        svi_results = pickle.load(f)
    
    # --------------------------------------------------------------------------

    print("Plotting loss...")
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Plot loss history
    ax.plot(svi_results.loss_history)

    # Set axis labels
    ax.set_xlabel('step')
    ax.set_ylabel('loss')

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/" \
        f"svi_{dataset_name.replace('_', '-')}_" \
        f"{parameterization.replace('_', '-')}_" \
        f"{n_steps}steps_loss.png", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print("Plotting ECDF...")
    print("Plotting ECDF for representative genes...")
    # Define number of genes to select
    n_genes = 36

    # Compute the median expression of each gene
    median_counts = np.median(counts, axis=0)

    # Compute the mean expression of each gene
    mean_counts = np.mean(counts, axis=0)

    # Get indices where median counts > 0
    nonzero_idx = np.where(mean_counts > 0.5)[0]

    # Sort the nonzero medians and get sorting indices
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    sorted_medians = mean_counts[sorted_idx]

    # Find unique median values to avoid duplicates
    unique_medians, unique_indices = np.unique(sorted_medians, return_index=True)

    # Generate logarithmically spaced indices across the unique median values
    if len(unique_medians) >= n_genes:
        # Use log-spaced indices across unique medians, avoiding log(0)
        log_spaced_indices = np.logspace(
            np.log10(1), np.log10(len(unique_medians)), num=n_genes, dtype=int
        )
        # Ensure we don't exceed array bounds and adjust for 0-based indexing
        log_spaced_indices = np.clip(
            log_spaced_indices - 1, 0, len(unique_medians) - 1
        )
        # Remove duplicates and ensure we get unique indices
        log_spaced_indices = np.unique(log_spaced_indices)
        # If we have fewer unique indices than desired, add more
        if len(log_spaced_indices) < n_genes:
            # Add more indices to reach desired number
            remaining_indices = np.setdiff1d(
                np.arange(len(unique_medians)), log_spaced_indices
            )
            if len(remaining_indices) > 0:
                additional_needed = n_genes - len(log_spaced_indices)
                additional_indices = remaining_indices[:additional_needed]
                log_spaced_indices = np.concatenate(
                    [log_spaced_indices, additional_indices]
                )
        # Get the actual gene indices for unique medians
        selected_idx = sorted_idx[unique_indices[log_spaced_indices]]
    else:
        # If we have fewer unique medians than desired genes, use all unique ones
        selected_idx = sorted_idx[unique_indices]
        
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Define step size for ECDF
    step = 1

    # Loop throu each gene
    for (i, gene) in enumerate(selected_idx):
        # Plot the ECDF for each column in the DataFrame
        sns.ecdfplot(
            data=counts[:, gene],
            ax=ax,
            color=sns.color_palette('Blues', n_colors=n_genes)[i],
            label=np.round(median_counts[gene], 0).astype(int),
            lw=1.5
        )

    # Add axis labels
    ax.set_xlabel('UMI count')
    ax.set_ylabel('ECDF')

    # Add title
    ax.set_title(dataset_name.replace("_", " "))

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/" \
        f"svi_{dataset_name.replace('_', '-')}_" \
        f"{parameterization.replace('_', '-')}_" \
        f"{n_steps}steps_ECDF.png", 
        bbox_inches="tight"
    )

    # Sort selected indices
    selected_idx = np.sort(selected_idx)

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print("Keeping subset of inference...")
    # Keep subset of inference from log_indices
    model_subset = svi_results[selected_idx]

    # --------------------------------------------------------------------------

    # Generate predictive samples
    print("Generating predictive samples...")
    model_subset.get_ppc_samples(n_samples=500)

    # --------------------------------------------------------------------------

    print("Plotting PPC for multiple example genes...")

    # Sort genes by mean UMI count for better visualization
    gene_means = np.mean(counts[:, selected_idx], axis=0)
    sorted_gene_order = np.argsort(gene_means)

    # Single plot example
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))

    # Flatten axes
    axes = axes.flatten()

    # Loop through each gene in sorted order
    for i, gene_idx in enumerate(sorted_gene_order):
        print(f"Plotting gene {i} PPC...")

        # Extract true counts for this gene
        true_counts = counts[:, selected_idx[gene_idx]]

        # Compute credible regions
        credible_regions = scribe.stats.compute_histogram_credible_regions(
            model_subset.predictive_samples[:, :, gene_idx],
            credible_regions=[95, 68, 50],
            # max_bin=true_counts.max()
        )

        # Compute histogram of the real data
        hist_results = np.histogram(
            true_counts, bins=credible_regions["bin_edges"], density=True
        )

        # Get indices where cumsum <= 0.999
        cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
        # If no indices found (all values > 0.999), use first bin
        max_bin = np.max([cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10])

        # Plot credible regions
        scribe.viz.plot_histogram_credible_regions_stairs(
            axes[i], credible_regions, cmap="Blues", alpha=0.5, max_bin=max_bin
        )

        # Define max_bin for histogram
        max_bin_hist = (
            max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
        )
        # Plot histogram of the real data as step plot
        axes[i].step(
            hist_results[1][:max_bin_hist],
            hist_results[0][:max_bin_hist],
            where="post",
            label="data",
            color="black",
        )

        # Set axis labels
        axes[i].set_xlabel("counts")
        axes[i].set_ylabel("frequency")

        # Set title with mean UMI count
        axes[i].set_title(
            f"$\\langle U \\rangle = {true_counts.mean():.2f}$",
            fontsize=8,
        )

        # Remove y-axis tick labels
        axes[i].set_yticklabels([])

    plt.tight_layout()

    # Set global title
    fig.suptitle("Example PPC", y=1.02)

    # Set global title
    fig.suptitle(dataset_name.replace("_", " "), y=1.02)

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/" \
        f"svi_{dataset_name.replace('_', '-')}_" \
        f"{parameterization.replace('_', '-')}_" \
        f"{n_steps}steps_example_ppc.png", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

# %% ---------------------------------------------------------------------------

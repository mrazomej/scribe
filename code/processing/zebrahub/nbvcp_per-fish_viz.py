# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle
import glob
import warnings

# Import JAX-related libraries
from jax import random
import jax.numpy as jnp
# Import numpy for array manipulation
import numpy as np
# Import scipy for statistical functions
import scipy.stats as stats
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Import scribe
import scribe
# Import scanpy for reading 10x h5 files
import scanpy as sc
# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

# Define number of steps for scribe
n_steps = 25_000

# Define model type
model_type = "nbvcp"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/zebrahub/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/zebrahub/{model_type}"

# Define dataset directory
DATA_DIR = f"/app/data/zebrahub/count_matrices/*/"

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# List all files in the output directory
model_files = sorted(glob.glob(f"{OUTPUT_DIR}/*.pkl"))

# List all datasets
data_files = sorted(glob.glob(f"{DATA_DIR}/*bc_matrix.h5", recursive=True))

# %% ---------------------------------------------------------------------------

# Loop over all model files
for i, model_file in enumerate(model_files):
    # Extract dataset name
    dataset_name = model_file.split("_")[-1].replace(".pkl", "")

    print(f"Plotting results for dataset {dataset_name} ({i+1} of {len(model_files)})...")

    # Load data
    with open(model_file, "rb") as f:
        results = pickle.load(f)

    # Search for the corresponding data file
    data_file = [f for f in data_files if dataset_name in f][0]

    # Silence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Load data
        data = sc.read_10x_h5(data_file).to_df()

    # --------------------------------------------------------------------------

    print("Plotting loss history...")

    # Initialize figure
    fig, ax = plt.subplots(figsize=(3.5, 3))

    # Plot loss history
    ax.plot(results.loss_history)

    # Set labels
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO loss")

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{dataset_name}_loss_history.pdf", bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print("Plotting ECDF...")
    # Define number of genes to select
    n_genes = 9

    # Compute the mean expression of each gene
    mean_counts = np.median(data.values, axis=0)

    # Get indices where mean counts > 0
    nonzero_idx = np.where(mean_counts > 0)[0]

    # Sort the nonzero means and get sorting indices
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]

    # Generate evenly spaced indices across the sorted nonzero genes
    spaced_indices = np.linspace(0, len(sorted_idx)-1, num=n_genes, dtype=int)

    # Get the actual gene indices after sorting
    selected_idx = sorted_idx[spaced_indices]

    # Initialize figure with extra space for legends
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

    # Define step size for ECDF
    step = 1

    # Plot shared genes
    for j, idx in enumerate(selected_idx):
        # Plot the ECDF for shared genes
        sns.ecdfplot(
            data=data.values[:, idx],
            ax=ax,
            color=sns.color_palette('Blues', n_colors=n_genes)[j],
            label=np.round(mean_counts[idx], 0).astype(int),
            lw=1.5
        )

    # Add axis labels and titles
    ax.set_xlabel('UMI count')
    ax.set_xscale('log')
    ax.set_ylabel('ECDF')

    # Add legends outside plots
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        title=r"$\langle U \rangle$",
        frameon=False
    )

    plt.tight_layout()

    # Save figure with extra space for legends
    fig.savefig(
        f"{FIG_DIR}/{dataset_name}_example_ECDF.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    # Index results for shared genes
    results_subset = results[np.sort(selected_idx)]

    # --------------------------------------------------------------------------

    # Define number of samples
    n_samples = 150

    print("Generating posterior predictive samples...")
    # Generate posterior predictive samples
    results_subset.get_ppc_samples(n_samples=n_samples)

    # --------------------------------------------------------------------------

    print("Plotting PPC for multiple example genes...")

    # Single plot example
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    # Flatten axes
    axes = axes.flatten()

    # Loop through each gene
    for j, ax in enumerate(axes):
        print(f"Plotting gene {j} PPC...")

        # Extract true counts for this gene
        true_counts = data.values[:, np.sort(selected_idx)[j]]

        # Compute credible regions
        credible_regions = scribe.stats.compute_histogram_credible_regions(
            results_subset.posterior_samples["predictive_samples"][:, :, j],
            credible_regions=[95, 68, 50],
            max_bin=true_counts.max()
        )

        # Compute histogram of the real data
        hist_results = np.histogram(
            true_counts,
            bins=credible_regions['bin_edges'],
            density=True
        )

        # Get indices where cumsum <= 0.999
        cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.95)[0]
        # If no indices found (all values > 0.999), use first bin
        max_bin = np.max([
            cumsum_indices[-1] if len(cumsum_indices) > 0 else 0,
            10
        ])

        # Plot credible regions
        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, 
            credible_regions,
            cmap='Blues',
            alpha=0.5,
            max_bin=max_bin
        )

        # Define max_bin for histogram
        max_bin_hist = max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
        # Plot histogram of the real data as step plot
        ax.step(
            hist_results[1][:max_bin_hist],
            hist_results[0][:max_bin_hist],
            where='post',
            label='data',
            color='black',
        )

        # Set axis labels
        ax.set_xlabel('counts')
        ax.set_ylabel('frequency')

        # Set title
        ax.set_title(
            f"$\\langle U \\rangle = {np.round(mean_counts[np.sort(selected_idx)[i]], 0).astype(int)}$", fontsize=8)

    plt.tight_layout()

    # Set global title
    fig.suptitle("Example PPC", y=1.02)

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{dataset_name}_example_ppc.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

# %% ---------------------------------------------------------------------------

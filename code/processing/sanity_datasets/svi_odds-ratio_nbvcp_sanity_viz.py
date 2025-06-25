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

# Set plotting style
scribe.viz.matplotlib_style()

# %% ---------------------------------------------------------------------------

# Define model_type
model_type = "nbvcp"
# Define parameterization
parameterization = "odds_ratio"

# Define training parameters
n_steps = 50_000
batch_size_max = 2048

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"/app/data/sanity"

# Define model directory
MODEL_DIR = f"{scribe.utils.git_root()}/output/sanity/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/sanity/{model_type}"

# Create figure directory if it does not exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# List all files in the data directory
files = glob.glob(f"{DATA_DIR}/*counts.txt.gz", recursive=True)

# %% ---------------------------------------------------------------------------

# Loop through each file
for file in files:

    # Load data
    df = pd.read_csv(file, sep="\t", index_col=0, compression="gzip")

    # Extract dataset name from file name
    dataset_name = file.split("/")[-1].replace("_counts.txt.gz", "")

    print(f"Processing {dataset_name}")

    # Define output file name
    file_name = f"{MODEL_DIR}/" \
                f"svi_{parameterization.replace('_', '-')}_" \
                f"{model_type.replace('_', '-')}_" \
                f"{n_steps}steps_" \
                f"{dataset_name}.pkl"

    # Check if the file exists
    if not os.path.exists(file_name):
        print(f"Skipping {file} because it does not exist")
        continue

    print("Loading model...")
    # Load model
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    
    # Add list of genes to model
    model.var = pd.DataFrame(df.index)

    # --------------------------------------------------------------------------

    print("Plotting loss...")
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Plot loss history
    ax.plot(model.loss_history)

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
    # Define number of genes to select
    n_genes = 25

    # Compute the median for each gene
    median_counts = df.median(axis=1)

    # Get median counts > 0
    median_counts_nonzero = median_counts[median_counts > 0]

    # Sort median counts
    sorted_idx = np.argsort(median_counts_nonzero)
    
    # Generate logarithmically spaced positions
    log_positions = np.unique(np.logspace(
        0, np.log10(len(sorted_idx) - 1), num=n_genes, dtype=int
    ))

    # If we got fewer positions than requested due to uniqueness, fill in the 
    # gaps with random positions
    if len(log_positions) < n_genes:
        available_positions = np.setdiff1d(
            np.arange(len(sorted_idx)), 
            log_positions
        )
        additional_positions = np.random.choice(
            available_positions,
            size=n_genes - len(log_positions),
            replace=False
        )
        selected_positions = np.sort(
            np.concatenate([log_positions, additional_positions])
        )
    else:
        selected_positions = log_positions

    # Map positions back to original array indices
    selected_genes = sorted_idx.iloc[selected_positions].index.values
    
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Define step size for ECDF
    step = 1

    # Loop throu each gene
    for (i, gene) in enumerate(selected_genes):
        # Plot the ECDF for each column in the DataFrame
        sns.ecdfplot(
            data=df.loc[gene, :],
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

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print("Keeping subset of inference...")
    # Keep subset of inference from log_indices
    model_subset = model[
        model.var.isin(selected_genes).values.flatten()]

    # --------------------------------------------------------------------------

    # Generate predictive samples
    print("Generating predictive samples...")
    model_subset.get_ppc_samples(n_samples=500)

    # --------------------------------------------------------------------------

    # Single plot example
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    # Flatten axes
    axes = axes.flatten()

    # Loop through each gene
    for i in range(model_subset.n_genes):
        print(f"Plotting gene {i} PPC...")

        # Extract axis
        ax = axes[i]

        # Extract gene from results to plot
        gene = model_subset.var.iloc[i].values[0]
    
        # Extract true counts for this gene
        true_counts = df.loc[gene, :]

        # Compute credible regions
        credible_regions = scribe.stats.compute_histogram_credible_regions(
            model_subset.predictive_samples[:, :, i],
            credible_regions=[95, 68, 50],
            max_bin=None
        )

        # Compute histogram of the real data
        hist_results = np.histogram(
            true_counts,
            bins=np.arange(0, true_counts.max() + 1),
            density=True
        )


        # Get indices where cumsum <= 0.999
        cumsum_indices_data = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
        

        # If no indices found (all values > 0.99), use first bin
        max_bin = (
            cumsum_indices_data[-1] 
            if len(cumsum_indices_data) > 1 
            else 5
        )

        # Plot credible regions
        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, 
            credible_regions,
            cmap='Blues',
            alpha=0.5,
            max_bin=max_bin
        )

        
        # Plot histogram of the real data as step plot
        ax.step(
            hist_results[1][:-1],
            hist_results[0],
            where='post',
            label='data',
            color='black',
            lw=2
        )

        # Set axis labels
        ax.set_xlabel('counts')
        ax.set_ylabel('frequency')
        # Set xlim
        ax.set_xlim(true_counts.min() - 0.05, np.max([max_bin, 3]) + 0.05)
        # Set ylim
        # ax.set_ylim(0, hist_results[0].max() * 1.05)

        # Set title
        ax.set_title(gene, fontsize=8)

    plt.tight_layout()

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

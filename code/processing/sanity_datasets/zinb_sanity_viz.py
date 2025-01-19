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
model_type = "zinb"

# Define training parameters
n_steps = 25_000

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
    output_file = f"{MODEL_DIR}/{dataset_name}_{n_steps}steps.pkl"

    print("Loading model...")
    # Load model
    with open(output_file, "rb") as f:
        model = pickle.load(f)
    
    # Add list of genes to model
    model.gene_metadata = pd.DataFrame(df.index)

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
        f"{FIG_DIR}/{dataset_name.replace('_', '-')}_loss.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print("Plotting ECDF...")
    # Define number of genes to select
    n_genes = 9

    # Compute the mean expression of each gene and sort them
    df_mean = df.mean(axis=1).sort_values(ascending=False)

    # Generate logarithmically spaced indices
    log_indices = np.logspace(
        0, np.log10(len(df_mean) - 1), num=n_genes, dtype=int
    )

    # Select genes using the logarithmically spaced indices
    genes = df_mean.iloc[log_indices].index

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Define step size for ECDF
    step = 1

    # Loop throu each gene
    for (i, gene) in enumerate(genes):
        # Plot the ECDF for each column in the DataFrame
        sns.ecdfplot(
            data=df.loc[gene, :],
            ax=ax,
            color=sns.color_palette('Blues', n_colors=n_genes)[i],
            label=np.round(df_mean[gene], 0).astype(int),
            lw=1.5
        )

    # Add axis labels
    ax.set_xlabel('UMI count')
    ax.set_ylabel('ECDF')

    # Add title
    ax.set_title(dataset_name.replace("_", " "))

    # Add legend
    ax.legend(
        loc='lower right', 
        fontsize=8, 
        title=r"$\langle U \rangle$", 
        frameon=False
    )

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{dataset_name.replace('_', '-')}_example_ECDF.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print("Keeping subset of inference...")
    # Keep subset of inference from log_indices
    gene_indices = model.gene_metadata.isin(genes.values).values.flatten()
    model_subset = model[gene_indices]

    # --------------------------------------------------------------------------

    # Generate predictive samples
    print("Generating predictive samples...")
    model_subset.get_ppc_samples(n_samples=500)

    # --------------------------------------------------------------------------

    # Single plot example
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    # Flatten axes
    axes = axes.flatten()

    # Loop through each gene
    for i, ax in enumerate(axes):
        print(f"Plotting gene {i} PPC...")
    
        # Extract true counts for this gene
        true_counts = df.loc[model_subset.gene_metadata.iloc[i], :]

        # Compute credible regions
        credible_regions = scribe.stats.compute_histogram_credible_regions(
            model_subset.posterior_samples["predictive_samples"][:, :, i],
            credible_regions=[95, 68, 50],
            max_bin=true_counts.values.max()
        )

        # Compute histogram of the real data
        hist_results = np.histogram(
            true_counts,
            bins=credible_regions['bin_edges'],
            density=True
        )

        # Get indices where cumsum <= 0.999
        cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
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
        ax.set_title(model_subset.gene_metadata.iloc[i].values[0], fontsize=8)

    plt.tight_layout()

    # Set global title
    fig.suptitle(dataset_name.replace("_", " "), y=1.02)

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{dataset_name.replace('_', '-')}_example_ppc.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

# %% ---------------------------------------------------------------------------

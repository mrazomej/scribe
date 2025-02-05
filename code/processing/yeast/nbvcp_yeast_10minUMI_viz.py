# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import pickle

# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import scanpy for single-cell data manipulation
import scanpy as sc
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

# Define training parameters
n_steps = 25_000
batch_size = 4096

# Minimum UMI threshold
min_umi = 10

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"/app/data/yeast"

# Define model directory
MODEL_DIR = f"{scribe.utils.git_root()}/output/yeast/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/yeast/{model_type}/{n_steps}steps"

# Create figure directory if it does not exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)
# %% ---------------------------------------------------------------------------

print("Loading data...")

# List all files in the data directory
file = glob.glob(f"{DATA_DIR}/*h5ad", recursive=True)

# Load dataset
data = sc.read_h5ad(file[0])

# Group by `batch` and extract index for each batch
df_group = data.obs.groupby('batch')

# Extract index for each batch
idxs = df_group.indices

# Extract data for each batch
data_batch = {k: data[v] for k, v in idxs.items()}

# %% ---------------------------------------------------------------------------

print("Calculating total UMI counts per cell...")

# Initialize dictionary to store total UMI counts per cell
umi_counts = {}

# Loop through each batch
for batch in data_batch.keys():
    # Get total UMI counts per cell and flatten from 1 x n matrix to array
    umi_counts[batch] = jnp.ravel(data_batch[batch].X.sum(axis=1))

# %% ---------------------------------------------------------------------------

# Loop through each batch
for batch in data_batch.keys():
    print(f"Processing batch {batch}...")
    # Select cells with at least `min_umi` UMI
    mask = umi_counts[batch] >= min_umi
    df = data_batch[batch][mask.tolist()].to_df()

    # Define output file name
    output_file = f"{MODEL_DIR}/{batch}_" \
                f"{model_type}_" \
                f"{min_umi}minUMI_" \
                f"{n_steps}steps_" \
                f"{batch_size}batch.pkl"

    print(" - Loading model...")
    # Load model
    with open(output_file, "rb") as f:
        model = pickle.load(f)
        
    print(" - Plotting loss...")
    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Plot loss history
    ax.plot(model.loss_history)

    # Set axis labels
    ax.set_xlabel('step')
    ax.set_ylabel('loss')

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{batch}_loss.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print(" - Plotting ECDF...")
    # Define number of genes to select
    n_genes = 25

    # Calculate metrics for each gene
    gene_metrics = pd.DataFrame({
        'mean_nonzero': df.apply(lambda x: x[x > 0].mean() if x[x > 0].size > 0 else 0),
        'detection_rate': (df > 0).mean()
    })

    # Filter for genes that are detected in at least 15% of cells
    min_detection_rate = 0.15
    gene_metrics = gene_metrics[
        gene_metrics['detection_rate'] >= min_detection_rate
    ]

    # Create a composite score combining expression level and detection rate
    gene_metrics['score'] = gene_metrics['mean_nonzero'] * \
                            gene_metrics['detection_rate']

    # Sort genes by composite score
    sorted_idx = gene_metrics['score'].sort_values().index

    # Generate logarithmically spaced positions
    log_positions = np.unique(np.logspace(
        0, np.log10(len(sorted_idx) - 1), num=n_genes, dtype=int
    ))

    # If we got fewer positions than requested due to uniqueness, fill in the
    # gaps
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
    selected_genes = sorted_idx[selected_positions]

    # Initialize figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    # Define step size for ECDF
    step = 1

    # Loop throu each gene
    for (i, gene) in enumerate(selected_genes):
        # Plot the ECDF for each column in the DataFrame
        sns.ecdfplot(
            data=df.loc[:, gene],
            ax=ax,
            color=sns.color_palette('Blues', n_colors=n_genes)[i],
            label=np.round(gene_metrics['mean_nonzero'][gene], 0).astype(int),
            lw=1.5
        )

    # Add axis labels
    ax.set_xlabel('UMI count')
    ax.set_ylabel('ECDF')

    # Add title
    ax.set_title(batch)

    # Set x-axis limits
    ax.set_xlim(0, 25)

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{batch}_example_ECDF.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)

    # --------------------------------------------------------------------------

    print(" - Keeping subset of inference...")
    # Keep subset of inference from log_indices
    model_subset = model[
        pd.Series(
            model.gene_metadata.index.values
        ).isin(selected_genes).values.flatten()]

    # --------------------------------------------------------------------------

    print(" - Generating predictive samples...")
    # Generate predictive samples
    model_subset.get_ppc_samples(n_samples=500)

    # --------------------------------------------------------------------------

    print(" - Plotting predictive samples...")

    # Single plot example
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    # Flatten axes
    axes = axes.flatten()

    # Loop through each gene
    for i in range(model_subset.n_genes):
        print(f"    - Plotting gene {i} PPC...")

        # Extract axis
        ax = axes[i]

        # Extract gene from results to plot
        gene = model_subset.gene_metadata.index.values[i]

        # Extract true counts for this gene
        true_counts = df.loc[:, gene]

        # Compute credible regions
        credible_regions = scribe.stats.compute_histogram_credible_regions(
            model_subset.posterior_samples["predictive_samples"][:, :, i],
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
            if len(cumsum_indices_data) > 1 and cumsum_indices_data[-1] > 5
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

        # Set title
        ax.set_title(gene, fontsize=8)

    plt.tight_layout()

    # Set global title
    fig.suptitle(batch, y=1.02)

    # Save figure
    fig.savefig(
        f"{FIG_DIR}/{batch}_example_ppc.pdf", 
        bbox_inches="tight"
    )

    # Close figure
    plt.close(fig)
    # --------------------------------------------------------------------------

print("Done!")
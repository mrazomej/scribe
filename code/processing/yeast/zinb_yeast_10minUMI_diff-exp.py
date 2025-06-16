# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import pickle
import itertools

# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import scipy for statistical operations
import scipy as sp
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

# Set color palette
color_palette = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

# Define model_type
model_type = "zinb"

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

# Initialize dictionary to store fraction of transcripts per gene
frac_transcripts = {}

# Loop through each batch
for batch in data_batch.keys():

    print(f"Processing {batch}...")

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
    # --------------------------------------------------------------------------

    print(" - Generating posterior samples...")
    # Generate posterior samples
    model.get_posterior_samples(n_samples=500)

    # --------------------------------------------------------------------------

    print(" - Sampling from Dirichlet distribution...")

    with jax.experimental.enable_x64():
        # Extract r parameter samples as float64
        r_samples = model.posterior_samples["parameter_samples"]["r"].astype(np.float64)

        # Sample from dirichlet distribution given r parameter samples
        dirichlet_samples = scribe.stats.sample_dirichlet_from_parameters(
            r_samples
        )

    # --------------------------------------------------------------------------

    print(" - Fitting Dirichlet distribution...")
    with jax.experimental.enable_x64():
        # Fit Dirichlet distribution to samples
        dirichlet_fit = scribe.stats.fit_dirichlet_minka(dirichlet_samples)

    # --------------------------------------------------------------------------

    print(" - Storing results...")
    # Store Dirichlet samples and fit
    frac_transcripts[batch] = {
        "dirichlet_samples": dirichlet_samples,
        "dirichlet_fit": dirichlet_fit
    }

# %% ---------------------------------------------------------------------------

# Initialize dictionary to store KL divergence for pairwise comparisons
kl_div = {}
hellinger_dist = {}

# List all unique pairs of batches
batch_pairs = sorted(
    set(tuple(sorted(pair)) 
        for pair in itertools.combinations(data_batch.keys(), 2))
)

# Loop through each batch pair
for batch_pair in batch_pairs:
    # Extract batches
    batch1, batch2 = batch_pair

    print(f"Processing {batch1} vs {batch2}...")

    with jax.experimental.enable_x64():
        # Extract Beta distribution parameters for each batch
        alpha1 = frac_transcripts[batch1]["dirichlet_fit"]
        alpha2 = frac_transcripts[batch2]["dirichlet_fit"]
        beta1 = jnp.sum(alpha1) - alpha1
        beta2 = jnp.sum(alpha2) - alpha2

        # Get KL divergence for pairwise comparisons
        kl_div[batch_pair] = scribe.stats.kl_beta(alpha1, beta1, alpha2, beta2)
        hellinger_dist[batch_pair] = scribe.stats.sq_hellinger_beta(
            alpha1, beta1, alpha2, beta2
        )

# %% ---------------------------------------------------------------------------

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

# Loop through each batch pair with corresponding color
for batch_pair in batch_pairs:
    # Extract KL divergence
    hellinger = hellinger_dist[batch_pair]

    # Plot histogram with color based on batch names
    if 'Standard' in batch_pair[0] and 'Standard' in batch_pair[1]:
        color = color_palette['dark_blue']
    elif 'EthGly' in batch_pair[0] or 'EthGly' in batch_pair[1]:
        color = color_palette['dark_green'] 
    else:
        color = color_palette['dark_red']
    # Plot KL divergence ECDF with specific color
    sns.ecdfplot(jnp.sqrt(hellinger), ax=ax, label=batch_pair, color=color)

# Set labels
ax.set_xlabel("Hellinger distance")
ax.set_ylabel("ECDF")

# Add legend outside plot on the right
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

# Save figure
fig.savefig(f"{FIG_DIR}/hellinger_dist_ecdf.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/hellinger_dist_ecdf.png", dpi=300, bbox_inches="tight")

# %% ---------------------------------------------------------------------------

# Get unique batches
unique_batches = sorted(data_batch.keys())
n_batches = len(unique_batches)

# Initialize figure with a grid of subplots
fig, axes = plt.subplots(
    n_batches, 
    n_batches, 
    figsize=(2.5*n_batches, 2.5*n_batches),
    sharex=True,
)

# Loop through all combinations of batches
for i, batch1 in enumerate(unique_batches):
    for j, batch2 in enumerate(unique_batches):
        # Skip diagonal and upper triangle
        if i <= j:
            # Remove axis from diagonal
            axes[i, j].remove()
            continue
            
        # Get the batch pair in sorted order to match our hellinger_dist
        # dictionary
        pair = tuple(sorted([batch1, batch2]))
        
        # Extract Hellinger distance
        hellinger = jnp.sqrt(hellinger_dist[pair])
        
        # Plot histogram with color based on batch names
        if 'Standard' in batch1 and 'Standard' in batch2:
            color = color_palette['dark_blue']
        elif 'EthGly' in batch1 or 'EthGly' in batch2:
            color = color_palette['dark_green'] 
        else:
            color = color_palette['dark_red']
        # Plot density
        sns.histplot(hellinger, ax=axes[i, j], color=color, edgecolor='none')
        
        # Set labels
        if i == n_batches-1:  # Bottom row
            axes[i, j].set_xlabel(batch2)
        if j == 0:  # Leftmost column
            axes[i, j].set_ylabel(batch1)
        
        # Remove redundant axis labels
        if i != n_batches-1:
            axes[i, j].set_xlabel('')
        if j != 0:
            axes[i, j].set_ylabel('')

# Adjust layout to prevent overlap
plt.tight_layout()

fig.savefig(f"{FIG_DIR}/hellinger_dist_corner.pdf", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/hellinger_dist_corner.png", dpi=300, bbox_inches="tight")

# %% ---------------------------------------------------------------------------

# # Initialize figure
# fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# # Define batch
# batch = "Glucose"

# # Extract samples and fit
# dirichlet_samples = frac_transcripts[batch]["dirichlet_samples"]
# dirichlet_fit = frac_transcripts[batch]["dirichlet_fit"]

# # Compute sample means
# sample_means = dirichlet_samples.mean(axis=0)

# # Compute distribution means
# dist_means = sp.stats.dirichlet(dirichlet_fit).mean()

# # Plot identity line within range of sample means
# ax.plot(
#     [0, max(sample_means)],
#     [0, max(dist_means)],
#     color="black",
#     lw=1,
#     ls="--",
# )

# # Plot sample mean vs distribution mean
# ax.scatter(
#     sample_means,
#     dist_means,
#     alpha=0.75,
#     s=10
# )

# # Set as log-log scale
# ax.set_xscale("log")
# ax.set_yscale("log")

# # Set labels
# ax.set_xlabel("Sample mean")
# ax.set_ylabel("Fit mean")

# # %% ---------------------------------------------------------------------------

# # Define number of genes to plot
# n_genes = 25

# # Define sorted Dirichlet parameters
# sorted_idx = np.argsort(dirichlet_fit)

# # Select spaced genes on their Dirichlet parameters
# lin_positions = np.unique(np.linspace(
#     0, len(sorted_idx) - 1, num=n_genes, dtype=int
# ))

# # %% ---------------------------------------------------------------------------

# # Single plot example
# fig, axes = plt.subplots(5, 5, figsize=(12, 12))

# # Flatten axes
# axes = axes.flatten()

# # Define quantile range
# q = (0.0001, 0.999)

# # Loop through each gene
# for i, idx in enumerate(lin_positions):
#     # Extract axis
#     ax = axes[i]

#     # Extract first Beta distributionparameter for this gene
#     alpha_gene = dirichlet_fit[sorted_idx[idx]]

#     # Define second Beta distribution parameter as sum of all other parameters
#     # minus the parameter for this gene
#     beta_gene = jnp.sum(dirichlet_fit) - alpha_gene

#     # Define scipy beta distribution
#     beta_dist = sp.stats.beta(alpha_gene, beta_gene)

#     # Plot posterior distribution
#     scribe.viz.plot_posterior(
#         ax,
#         beta_dist,
#         plot_quantiles=(q[0], q[1]),
#         n_points=1_500,
#         log_scale=True
#     )

#     # Add samples as rug plot
#     sns.rugplot(
#         dirichlet_samples[sorted_idx[idx]],
#         ax=ax,
#         color="black",
#         alpha=0.5
#     )
    
#     # Set x-axis limits to quantile range
#     # ax.set_xlim(beta_dist.ppf(q[0]), beta_dist.ppf(q[1]))

#     # Set x-axis scale to logarithmic
#     ax.set_xscale("log")

# plt.tight_layout()
    


# # %% ---------------------------------------------------------------------------

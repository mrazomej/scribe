# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import scribe
import scribe
# Import scanpy for data manipulation
import scanpy as sc
# Import numpy for array manipulation
import numpy as np
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import seaborn for plotting
import seaborn as sns
# Import umap-learn for UMAP
import umap

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

print("Defining parameterization...")

# Define parameterization
parameterization = "odds_ratio"
# Define model type
model_type = "nbvcp"
# Define latent dimension
latent_dim = 2
# Define number of steps
n_steps = 25_000

print("Defining directories...")

# Define data directory
DATA_DIR = (
    f"{scribe.utils.git_root()}/data/10xGenomics/50-50_Jurkat-293T_mixture"
)

# Define output directory
OUTPUT_DIR = (
    f"{scribe.utils.git_root()}/output/10xGenomics/50-50_Jurkat-293T_mixture/"
    f"{model_type}"
)

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/supplementary"

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load CSV file
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# Define data
counts = data.X.toarray()

# Define number of cells
n_cells = data.shape[0]
# Define number of genes
n_genes = data.shape[1]

# %% ---------------------------------------------------------------------------

print("Loading dpVAE results...")

vae_results = pickle.load(
    open(
        f"{OUTPUT_DIR}/dpvae_{parameterization.replace('_', '-')}_"
        f"{model_type.replace('_', '-')}-unconstrained_"
        f"{latent_dim:02d}latentdim_"
        f"{n_steps}steps.pkl",
        "rb",
    )
)

# %% ---------------------------------------------------------------------------

print("Plotting loss...")
# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

# Plot loss history
ax.plot(vae_results.loss_history)

# Set axis labels
ax.set_xlabel('step')
ax.set_ylabel('loss')

# Set title
ax.set_title(
    f"dpVAE loss for NBVCP model\n unconstrained odds-ratio parameterization"
)

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_loss.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_loss.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting latent embeddings...")

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Get latent embeddings
latent_embeddings = vae_results.get_latent_embeddings(counts)

# Plot latent embeddings
ax.scatter(
    latent_embeddings[:, 0],
    latent_embeddings[:, 1],
    alpha=0.5,
    s=5,
    color=colors["dark_blue"],
)

# Set axis scale to be equal
ax.set_aspect('equal', adjustable='datalim')

# Label axes
ax.set_xlabel('latent dimension 1')
ax.set_ylabel('latent dimension 2')

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_latent_embeddings.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_latent_embeddings.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting latent samples from learned prior...")

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Get latent samples
vae_results.get_latent_samples(n_samples=10_000)

# Plot latent samples
ax.scatter(
    vae_results.latent_samples[:, 0],
    vae_results.latent_samples[:, 1],
    alpha=0.5,
    s=5,
    color=colors["dark_red"],
)

# Set axis scale to be equal
ax.set_aspect('equal', adjustable='datalim')

# Label axes
ax.set_xlabel('latent dimension 1')
ax.set_ylabel('latent dimension 2')

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_latent_samples.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_latent_samples.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting density of latent samples...")

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Plot density for latent samples
sns.kdeplot(
    x=vae_results.latent_samples[:, 0],
    y=vae_results.latent_samples[:, 1],
    fill=True,
    color=colors["dark_red"],
)

# Plot latent embeddings
ax.scatter(
    latent_embeddings[:, 0],
    latent_embeddings[:, 1],
    alpha=0.25,
    s=5,
    color=colors["light_blue"],
)

# Set axis scale to be equal
ax.set_aspect('equal', adjustable='datalim')

# Label axes
ax.set_xlabel('latent dimension 1')
ax.set_ylabel('latent dimension 2')

# Save figure

fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_latent_samples_density.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_"
    f"{parameterization.replace('_', '-')}_"
    f"{latent_dim:02d}latentdim_"
    f"{model_type.replace('_', '-')}-unconstrained_latent_samples_density.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Selecting genes...")
# Define number of genes to select
n_genes = 36

# Compute the median expression of each gene
median_counts = np.median(counts, axis=0)

# Compute the mean expression of each gene
mean_counts = np.mean(counts, axis=0)

# Get indices where median counts > 0
nonzero_idx = np.where(mean_counts > 0.1)[0]

# Sort the nonzero medians and get sorting indices
sorted_idx = nonzero_idx[np.argsort(median_counts[nonzero_idx])]
sorted_medians = median_counts[sorted_idx]

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

# Index results for selected genes
results_subset = vae_results[selected_idx]

# Sort selected indices
selected_idx = np.sort(selected_idx)

# Sort genes by mean UMI count for better visualization
gene_means = np.mean(counts[:, selected_idx], axis=0)
sorted_gene_order = np.argsort(gene_means)
# %% ---------------------------------------------------------------------------

print("Generating predictive samples...")
# Generate predictive samples
results_subset.get_ppc_samples(n_samples=500)
# Convert to canonical parameters (i.e. r = mu * theta)
results_subset._convert_to_canonical()

# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Single plot example
fig, axes = plt.subplots(6, 6, figsize=(12, 12))

# Flatten axes
axes = axes.flatten()

# Loop through each gene in sorted order
for i, gene_idx in enumerate(sorted_gene_order):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data.X.toarray()[:, selected_idx[gene_idx]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.predictive_samples[:, :, gene_idx],
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
    max_bin = np.max(
        [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10])

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
fig.suptitle("Posterior Predictive Checks", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-unconstrained_ppc.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-unconstrained_ppc.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Generating predictive samples conditioned on data...")

# Generate posterior samples
results_subset.get_posterior_samples_conditioned_on_data(
    counts,
    n_samples=100,
    canonical=False,
)
# Generate posterior predictive samples
results_subset.get_predictive_samples()
# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Single plot example
fig, axes = plt.subplots(6, 6, figsize=(12, 12))

# Flatten axes
axes = axes.flatten()

# Loop through each gene in sorted order
for i, gene_idx in enumerate(sorted_gene_order):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data.X.toarray()[:, selected_idx[gene_idx]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.predictive_samples[:, :, gene_idx],
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
    max_bin = np.max(
        [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10])

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
fig.suptitle("Posterior Predictive Checks\n conditioned on data", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-unconstrained_ppc_conditioned.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-unconstrained_ppc_conditioned.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Generate single predictive sample for full dataset...")

# Generate single predictive sample for full dataset
ppc = vae_results.get_ppc_samples(n_samples=1)["predictive_samples"]

# Generate posterior samples conditioned on data
vae_results.get_posterior_samples_conditioned_on_data(
    counts,
    n_samples=1,
    canonical=False,
)
# Generate posterior predictive samples conditioned on data
ppc_conditioned = vae_results.get_predictive_samples()

# %% ---------------------------------------------------------------------------

print("Fit UMAP for data...")

# Initialize UMAP
umap_counts = umap.UMAP(n_components=2).fit_transform(counts)

print("Fit UMAP for PPC...")

# Initialize UMAP
umap_ppc = umap.UMAP(n_components=2).fit_transform(
    ppc[0, :, :]
)

print("Fit UMAP for PPC conditioned on data...")

# Initialize UMAP
umap_ppc_conditioned = umap.UMAP(n_components=2).fit_transform(
    ppc_conditioned[0, :, :]
)

# %% ---------------------------------------------------------------------------

print("Plotting UMAP...")


# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Plot UMAP for data
ax.scatter(
    umap_counts[:, 0],
    umap_counts[:, 1],
    s=1,
    label="data",
    color=colors["dark_blue"],
)

# Plot UMAP for PPC
ax.scatter(
    umap_ppc[:, 0],
    umap_ppc[:, 1],
    s=1,
    label="PPC",
    color=colors["dark_red"],
)

# Plot UMAP for PPC conditioned on data
ax.scatter(
    umap_ppc_conditioned[:, 0],
    umap_ppc_conditioned[:, 1],
    s=1,
    label="PPC conditioned",
    color=colors["dark_green"],
)

# Add legend outside of plot on the right side
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# Label axes
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

# Turn off ticks in axis
ax.set_xticks([])
ax.set_yticks([])

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-unconstrained_umap.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_10x50-50mix_dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-unconstrained_umap.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

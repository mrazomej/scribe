# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import numpy for array manipulation
import numpy as np

# Import plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Import scribe
import scribe

# Import scanpy for loading data
import scanpy as sc

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

# Define model type
model_type = "nbvcp"

# Define parameterization
parameterization = "linked"

# Define if unconstrained
unconstrained = True

# Define number of steps for scribe
n_steps = 25_000

# Define latent dimension
latent_dim = 3

# Define data directory
DATA_DIR = (
    f"{scribe.utils.git_root()}/data/" f"10xGenomics/50-50_Jurkat-293T_mixture"
)

# Define output directory
OUTPUT_DIR = (
    f"{scribe.utils.git_root()}/output/"
    f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"
)

# Define figure directory
FIG_DIR = (
    f"{scribe.utils.git_root()}/fig/"
    f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"
)

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# Convert data to dense array
counts = data.X.toarray()

# %% ---------------------------------------------------------------------------

print("Loading scribe results...")

vae_results = pickle.load(
    open(
        f"{OUTPUT_DIR}/"
        f"dpvae_{parameterization.replace('_', '-')}_"
        f"{model_type.replace('_', '-')}-"
        f"unconstrained_"
        f"{latent_dim:02d}latentdim_"
        f"{n_steps}steps.pkl",
        "rb",
    )
)

# %% ---------------------------------------------------------------------------

print("Plotting loss history...")

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot loss history
ax.plot(vae_results.loss_history)

# Set labels
ax.set_xlabel("step")
ax.set_ylabel("ELBO loss")

# Set y-axis to log scale
# ax.set_yscale("log")

# Save figure
fig.savefig(
    f"{FIG_DIR}/dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"unconstrained_"
    f"{latent_dim:02d}latentdim_"
    f"{n_steps}steps_loss.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

print("Plotting latent embeddings (3D)...")

# Initialize 3D figure
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="3d")

# Get latent embeddings
latent_embeddings = vae_results.get_latent_embeddings(counts)

# Plot latent embeddings in 3D
ax.scatter(
    latent_embeddings[:, 0],
    latent_embeddings[:, 1],
    latent_embeddings[:, 2],
    alpha=0.5,
    s=5,
)

# Label axes
ax.set_xlabel("latent dimension 1")
ax.set_ylabel("latent dimension 2")
ax.set_zlabel("latent dimension 3")

plt.tight_layout()

# Save figure
fig.savefig(
    f"{FIG_DIR}/dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"unconstrained_"
    f"{latent_dim:02d}latentdim_"
    f"{n_steps}steps_latent_embeddings.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

print("Plot latent samples...")

# Initialize 3D figure
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="3d")

# Get latent samples
vae_results.get_latent_samples(n_samples=1_000)

# Plot latent samples in 3D
ax.scatter(
    vae_results.latent_samples[:, 0],
    vae_results.latent_samples[:, 1],
    vae_results.latent_samples[:, 2],
    alpha=0.5,
    s=5,
)

# Label axes
ax.set_xlabel("latent dimension 1")
ax.set_ylabel("latent dimension 2")
ax.set_zlabel("latent dimension 3")

# Save figure
fig.savefig(
    f"{FIG_DIR}/dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"unconstrained_"
    f"{latent_dim:02d}latentdim_"
    f"{n_steps}steps_latent_samples.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

print("Plotting ECDF...")
# Define number of genes to select
n_genes = 25

# Compute the mean expression of each gene
mean_counts = np.median(counts, axis=0)

# Get indices where mean counts > 0
nonzero_idx = np.where(mean_counts > 0)[0]

# Sort the nonzero means and get sorting indices
sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]

# Generate evenly spaced indices across the sorted nonzero genes
spaced_indices = np.linspace(0, len(sorted_idx) - 1, num=n_genes, dtype=int)

# Get the actual gene indices after sorting
selected_idx = sorted_idx[spaced_indices]

# Initialize figure with extra space for legends
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

# Define step size for ECDF
step = 1

# Plot shared genes
for i, idx in enumerate(selected_idx):
    # Plot the ECDF for shared genes
    sns.ecdfplot(
        data=counts[:, idx],
        ax=ax,
        color=sns.color_palette("Blues", n_colors=n_genes)[i],
        lw=1.5,
        label=None,
    )

# Add axis labels and titles
ax.set_xlabel("UMI count")
ax.set_xscale("log")
ax.set_ylabel("ECDF")

plt.tight_layout()

# Save figure with extra space for legends
fig.savefig(f"{FIG_DIR}/example_ECDF_{n_steps}steps.png", bbox_inches="tight")

# Sort selected indices
selected_idx = np.sort(selected_idx)

# %% ---------------------------------------------------------------------------

print("Indexing results for example genes...")

# Index results for shared genes
results_subset = vae_results[selected_idx]

# %% ---------------------------------------------------------------------------

print("Generating posterior samples for full dataset...")

# Generate posterior samples
results_subset.get_posterior_samples(n_samples=100, canonical=True)

print("Generating posterior predictive samples for example genes...")
# Generate posterior predictive samples
results_subset.get_predictive_samples()

# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Single plot example
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data.X.toarray()[:, selected_idx[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.predictive_samples[:, :, i],
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
        ax, credible_regions, cmap="Blues", alpha=0.5, max_bin=max_bin
    )

    # Define max_bin for histogram
    max_bin_hist = (
        max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
    )
    # Plot histogram of the real data as step plot
    ax.step(
        hist_results[1][:max_bin_hist],
        hist_results[0][:max_bin_hist],
        where="post",
        label="data",
        color="black",
    )

    # Set axis labels
    ax.set_xlabel("counts")
    ax.set_ylabel("frequency")

    # Set title
    ax.set_title(
        f"$\\langle U \\rangle = {np.round(mean_counts[selected_idx[i]], 0).astype(int)}$",
        fontsize=8,
    )

plt.tight_layout()

# Set global title
fig.suptitle("Example PPC", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"unconstrained_"
    f"{latent_dim:02d}latentdim_"
    f"{n_steps}steps_ppc.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

print("Generating predictive samples conditioned on data...")

# Generate posterior samples
results_subset.get_posterior_samples_conditioned_on_data(
    counts,
    n_samples=100,
    # batch_size=1000,
    canonical=False,
)

# %% ---------------------------------------------------------------------------

print("Generating posterior predictive samples for example genes...")
# Generate posterior predictive samples
results_subset.get_predictive_samples()
# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Single plot example
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data.X.toarray()[:, selected_idx[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.predictive_samples[:, :, i],
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
        ax, credible_regions, cmap="Blues", alpha=0.5, max_bin=max_bin
    )

    # Define max_bin for histogram
    max_bin_hist = (
        max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
    )
    # Plot histogram of the real data as step plot
    ax.step(
        hist_results[1][:max_bin_hist],
        hist_results[0][:max_bin_hist],
        where="post",
        label="data",
        color="black",
    )

    # Set axis labels
    ax.set_xlabel("counts")
    ax.set_ylabel("frequency")

    # Set title
    ax.set_title(
        f"$\\langle U \\rangle = {np.round(mean_counts[selected_idx[i]], 0).astype(int)}$",
        fontsize=8,
    )

plt.tight_layout()

# Set global title
fig.suptitle("Example PPC", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-"
    f"unconstrained_"
    f"conditioned_"
    f"{latent_dim:02d}latentdim_"
    f"{n_steps}steps_ppc.png",
    bbox_inches="tight",
)
# %% ---------------------------------------------------------------------------

# Generate p-capture samples conditioned on data
p_capture_samples = vae_results.get_p_capture_samples_conditioned_on_data(
    counts,
    n_samples=100,
)

# %% ---------------------------------------------------------------------------

# Plot histogram of mean p-capture samples
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Plot histogram of mean p-capture samples
sns.histplot(p_capture_samples.mean(axis=0), ax=ax)

# Set axis labels
ax.set_xlabel("mean p-capture")
ax.set_ylabel("frequency")

# %% ---------------------------------------------------------------------------

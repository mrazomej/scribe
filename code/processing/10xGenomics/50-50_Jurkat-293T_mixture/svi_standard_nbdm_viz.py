# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import numpy for array manipulation
import numpy as np

# Import plotting libraries
import matplotlib.pyplot as plt
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
model_type = "nbdm"

# Define parameterization
parameterization = "standard"

# Define number of steps for scribe
n_steps = 40_000

# Define number of components in mixture model
n_components = 1

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

svi_results = pickle.load(
    open(
        f"{OUTPUT_DIR}/svi_{parameterization.replace('-', '_')}_"
        f"{model_type.replace('_', '-')}_"
        f"{n_components:02d}components_"
        f"{n_steps}steps.pkl",
        "rb",
    )
)

# %% ---------------------------------------------------------------------------

print("Plotting loss history...")

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot loss history
ax.plot(svi_results.loss_history)

# Set labels
ax.set_xlabel("step")
ax.set_ylabel("ELBO loss")

# Set y-axis to log scale
ax.set_yscale("log")

# Save figure
fig.savefig(
    f"{FIG_DIR}/svi_"
    f"{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components:02d}components_"
    f"{n_steps}steps_loss.png",
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
fig.savefig(
    f"{FIG_DIR}/svi_"
    f"{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components:02d}components_"
    f"{n_steps}steps_ECDF.png",
    bbox_inches="tight",
)

# Sort selected indices
selected_idx = np.sort(selected_idx)

# %% ---------------------------------------------------------------------------

# Index results for shared genes
results_subset = svi_results[selected_idx]

# %% ---------------------------------------------------------------------------

# Define number of samples
n_samples = 1_500

print("Generating posterior predictive samples...")
# Generate posterior predictive samples
results_subset.get_ppc_samples(n_samples=n_samples)

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
    f"{FIG_DIR}/svi_"
    f"{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components:02d}components_"
    f"{n_steps}steps_ppc.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

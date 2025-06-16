# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import JAX-related libraries
import jax
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
# Import scanpy for loading data
import scanpy as sc

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

# Define model type
model_type = "nbvcp_log_mix"

# Define number of steps for scribe
n_steps = 30_000

# Define number of components in mixture model
n_components = 2

# Define r_distribution
r_distribution = "lognormal"

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/" \
           f"10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/" \
             f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/" \
          f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# Extract counts
counts = jnp.array(data.X.toarray())

# %% ---------------------------------------------------------------------------

print("Loading scribe results...")

results = pickle.load(open(f"{OUTPUT_DIR}/"
                           f"scribe_{model_type}_r-{r_distribution}_results_"
                           f"{n_components:02d}components_"
                           f"{n_steps}steps.pkl", "rb"))

# %% ---------------------------------------------------------------------------

print("Plotting loss history...")

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot loss history
ax.plot(results.loss_history)

# Set labels
ax.set_xlabel("step")
ax.set_ylabel("ELBO loss")

# Save figure
fig.savefig(f"{FIG_DIR}/loss_history_{n_steps}steps.png", bbox_inches="tight")

plt.show()
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
spaced_indices = np.linspace(0, len(sorted_idx)-1, num=n_genes, dtype=int)

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
        color=sns.color_palette('Blues', n_colors=n_genes)[i],
        lw=1.5,
        label=None
    )

# Add axis labels and titles
ax.set_xlabel('UMI count')
ax.set_xscale('log')
ax.set_ylabel('ECDF')

plt.tight_layout()

# Sort selected_idx by mean counts
selected_idx = np.sort(selected_idx)

# Save figure with extra space for legends
fig.savefig(f"{FIG_DIR}/example_ECDF_{n_steps}steps.png", bbox_inches="tight")
# %% ---------------------------------------------------------------------------

# Index results for shared genes
results_subset = results[selected_idx]

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
    true_counts = counts[:, selected_idx[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.predictive_samples[:, :, i],
        credible_regions=[95, 68, 50],
        # max_bin=true_counts.max()
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
    ax.set_title(
        f"$\\langle U \\rangle = {np.round(mean_counts[selected_idx[i]], 0).astype(int)}$", fontsize=8)

plt.tight_layout()

# Set global title
fig.suptitle("Example PPC", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_ppc_{n_steps}steps.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting KL divergence for r parameters...")

# Compute KL divergence
kl_divergence = results.kl_divergence()

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot KL divergence
sns.ecdfplot(kl_divergence, ax=ax)

# Set labels
ax.set_xlabel(r"$D_{KL}(P_{\text{type}_1}||Q_{\text{type}_2})$")
ax.set_ylabel("ECDF")

# Set y-axis limits
ax.set_ylim(-0.01, 1.01)

# Set x-scale to log
ax.set_xscale('log')

# Save figure
fig.savefig(f"{FIG_DIR}/kl_divergence_{n_steps}steps.png", bbox_inches="tight")

# %% ---------------------------------------------------------------------------

print("Computing cell type assignments MAP...")
# Compute cell assignments with mean substitution
cell_types_map = results.cell_type_assignments_map(
    counts=jnp.array(counts),
    batch_size=100,
    temperature=100.0,
    use_mean=True
)

# Define cell assignment as class with highest probability
cell_assignments_map = np.argmax(cell_types_map["probabilities"], axis=1)

# %% ---------------------------------------------------------------------------

print("Sampling from full posterior distribution...")
# Sample from full posterior distribution
results.get_posterior_samples(n_samples=1_500)

# %% ---------------------------------------------------------------------------

print("Computing cell type assignments...")
# Use posterior samples to assign cell types
cell_types = results.cell_type_assignments(
    counts=jnp.array(counts),
    ignore_nans=True,
    temperature=100.0
)

# Extract mean prob assignments
mean_assignments = cell_types["mean_probabilities"]

# Define cell assignment as class with highest probability
cell_assignments = np.argmax(mean_assignments, axis=1)


# %% ---------------------------------------------------------------------------

print("Plotting Hellinger distance for r parameters...")

# Compute Hellinger distance
hellinger_distance = results.hellinger_distance()["0_1"]

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot Hellinger distance
sns.ecdfplot(hellinger_distance, ax=ax)

# Set labels
ax.set_xlabel(r"$H(P_{\text{type}_1}, Q_{\text{type}_2})$")
ax.set_ylabel("ECDF")

# Set y-axis limits
ax.set_ylim(-0.01, 1.01)

# Save figure
fig.savefig(
    f"{FIG_DIR}/hellinger_distance_{n_steps}steps.png", bbox_inches="tight"
)


# %% ---------------------------------------------------------------------------

print("Plotting ECDF...")
# Define number of genes to select
n_genes = 25

# Sort genes by Hellinger distance in descending order
sorted_idx = np.argsort(-hellinger_distance)  # Negative to sort in descending order

# Select top n_genes with highest Hellinger distance
selected_idx = np.sort(sorted_idx[:n_genes])

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
        color=sns.color_palette('Blues', n_colors=n_genes)[i],
        lw=1.5,
        label=None
    )

# Add axis labels and titles
ax.set_xlabel('UMI count')
ax.set_xscale('log')
ax.set_ylabel('ECDF')

plt.tight_layout()

# Save figure with extra space for legends
fig.savefig(
    f"{FIG_DIR}/example_ECDF_hellinger_{n_steps}steps.png", bbox_inches="tight"
) 
# %% ---------------------------------------------------------------------------

# Index results for shared genes
results_subset_hellinger = results[selected_idx]

# %% ---------------------------------------------------------------------------

# Define number of samples
n_samples = 1_500

print("Generating posterior predictive samples...")
# Generate posterior predictive samples
results_subset_hellinger.get_ppc_samples(
    n_samples=n_samples, resample_parameters=True
)

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
    true_counts = counts[:, selected_idx[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset_hellinger.predictive_samples[:, :, i],
        credible_regions=[95, 68, 50],
        # max_bin=true_counts.max()
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
    ax.set_title(
        f"$\\langle U \\rangle = {np.round(mean_counts[selected_idx[i]], 0).astype(int)}$", fontsize=8)

plt.tight_layout()

# Set global title
fig.suptitle("Example PPC", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_ppc_hellinger_{n_steps}steps.png", 
    bbox_inches="tight"
)
# %% ---------------------------------------------------------------------------

print("Computing cell type assignments MAP with hellinger distance weights...")

# Compute cell assignments with mean substitution
cell_types_hellinger_map = results.cell_type_assignments_map(
    counts=jnp.array(counts),
    batch_size=100,
    use_mean=True,
    temperature=100.0,
    weights=jnp.log(hellinger_distance),
    weight_type="additive"
)

# Define cell assignment as class with highest probability
cell_assignments_hellinger_map = np.argmax(
    cell_types_hellinger_map["probabilities"], axis=1)

# %% ---------------------------------------------------------------------------

# Get index of genes with hellinger distance > 0.7
idx_hellinger = np.where(hellinger_distance > 0.7)[0]

# Extract results for these genes
results_subset_hellinger = results[idx_hellinger]

# Extract data for these genes
data_subset_hellinger = jnp.array(counts[:, idx_hellinger])

# Compute cell type assignments for these genes
cell_types_hellinger_weights = results_subset_hellinger.cell_type_assignments_map(
    counts=data_subset_hellinger,
    batch_size=100,
    use_mean=True,
    temperature=len(idx_hellinger),
    # weights=jnp.log(hellinger_distance),
    # weight_type="additive"
)

# Define cell assignment as class with highest probability
cell_assignments_hellinger_weights = np.argmax(
    cell_types_hellinger_weights["probabilities"], axis=1)

# %% ---------------------------------------------------------------------------

# Compute component entropy
entropy = results.assignment_entropy(
    counts=jnp.array(counts),
    ignore_nans=True,
    return_by="gene",
)

# Compute the mean entropy over samples
mean_entropy = jnp.mean(entropy, axis=0)
sns.ecdfplot(mean_entropy)

# Compute weights based on mean entropy
# weights = 1 - mean_entropy / jnp.log(results.n_components)

# %% ---------------------------------------------------------------------------

# Compute entropy per gene
entropy_map = results.assignment_entropy_map(counts, batch_size=100)

sns.ecdfplot(entropy_map)

# %% ---------------------------------------------------------------------------

# Find index of genes with entropy > 0.3
idx_entropy = np.where(mean_entropy < 0.06)[0]

# Extract results for these genes
results_subset_entropy = results[idx_entropy]

# Extract data for these genes
data_subset_entropy = jnp.array(counts[:, idx_entropy])

# Compute cell type assignments for these genes
cell_types_entropy = results_subset_entropy.cell_type_assignments(
    counts=data_subset_entropy,
    ignore_nans=True,
    temperature=10.0,
    batch_size=100
)

# Define cell assignment as class with highest probability
cell_assignments_entropy = jnp.argmax(
    cell_types_entropy["mean_probabilities"], axis=1)

# %% ---------------------------------------------------------------------------


# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

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

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

# Define number of steps for scribe
n_steps = 50_000
# Define number of cells
n_cells = 10_000
# Define number of shared genes
n_shared_genes = 10_000
# Define number of unique genes
n_unique_genes = 10_000
# Define number of genes
n_genes = n_shared_genes + n_unique_genes
# Define number of components
n_components = 2

# Define model type
model_type = "zinb_mix"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/sim/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/sim/{model_type}"

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Load data
with open(
    f"{OUTPUT_DIR}/data_"
    f"{n_cells}cells_"
    f"{n_genes}genes_"
    f"{n_shared_genes}shared_"
    f"{n_unique_genes}unique_"
    f"{n_components:02d}components.pkl",
    "rb"
) as f:
    data = pickle.load(f)

# Load scribe results
with open(
    f"{OUTPUT_DIR}/scribe_zinb_mix_results_"
    f"{n_cells}cells_"
    f"{n_genes}genes_"
    f"{n_shared_genes}shared_"
    f"{n_unique_genes}unique_"
    f"{n_components:02d}components_"
    f"{n_steps}steps.pkl",
    "rb"
) as f:
    results = pickle.load(f)

# %% ---------------------------------------------------------------------------

# Plot loss history

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot loss history
ax.plot(results.loss_history)

# Set labels
ax.set_xlabel("step")
ax.set_ylabel("ELBO loss")

# Save figure
fig.savefig(f"{FIG_DIR}/loss_history.png", bbox_inches="tight")

plt.show()

# %% ---------------------------------------------------------------------------

# Split shared and unique genes
shared_genes = data["counts"][:, 0:n_shared_genes]
unique_genes = data["counts"][:, n_shared_genes:]

# %% ---------------------------------------------------------------------------

print("Plotting ECDF...")
# Define number of genes to select
n_genes = 9

# Compute the mean expression of each gene and sort them
mean_shared = shared_genes.mean(axis=0)
mean_unique = unique_genes.mean(axis=0)

# Sort means and get sorting indices
sorted_idx_shared = np.argsort(mean_shared)
sorted_idx_unique = np.argsort(mean_unique)

# Generate logarithmically spaced indices
log_indices_shared = np.logspace(
    0, np.log10(len(mean_shared) - 1), num=n_genes, dtype=int
)
log_indices_unique = np.logspace(
    0, np.log10(len(mean_unique) - 1), num=n_genes, dtype=int
)

# Get the actual gene indices after sorting
selected_idx_shared = sorted_idx_shared[log_indices_shared]
selected_idx_unique = sorted_idx_unique[log_indices_unique]

# Initialize figure with extra space for legends
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Define step size for ECDF
step = 1

# Plot shared genes
for i, idx in enumerate(selected_idx_shared):
    # Plot the ECDF for shared genes
    sns.ecdfplot(
        data=shared_genes[:, idx],
        ax=axes[0],
        color=sns.color_palette('Blues', n_colors=n_genes)[i],
        label=np.round(mean_shared[idx], 0).astype(int),
        lw=1.5
    )

# Plot unique genes
for i, idx in enumerate(selected_idx_unique):
    # Plot the ECDF for unique genes
    sns.ecdfplot(
        data=unique_genes[:, idx],
        ax=axes[1],
        color=sns.color_palette('Reds', n_colors=n_genes)[i],
        label=np.round(mean_unique[idx], 0).astype(int),
        lw=1.5
    )

# Add axis labels and titles
for ax in axes:
    ax.set_xlabel('UMI count')
    ax.set_xscale('log')
    ax.set_ylabel('ECDF')

axes[0].set_title('Shared Genes')
axes[1].set_title('Unique Genes')

# Add legends outside plots
axes[0].legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=8,
    title=r"$\langle U \rangle$",
    frameon=False
)
axes[1].legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=8,
    title=r"$\langle U \rangle$",
    frameon=False
)

plt.tight_layout()

# Save figure with extra space for legends
fig.savefig(f"{FIG_DIR}/example_ECDF.png", bbox_inches="tight")

# %% ---------------------------------------------------------------------------

# Index results for shared genes
results_shared = results[selected_idx_shared]

# %% ---------------------------------------------------------------------------

# Index results for unique genes
results_unique = results[selected_idx_unique + n_shared_genes]

# %% ---------------------------------------------------------------------------

# Define number of samples
n_samples = 500

print("Generating posterior predictive samples for shared genes...")
# Generate posterior predictive samples
results_shared.get_ppc_samples(n_samples=500)

print("Generating posterior predictive samples for unique genes...")
# Generate posterior predictive samples
results_unique.get_ppc_samples(n_samples=500)

# %% ---------------------------------------------------------------------------

print("Plotting PPC for shared genes...")

# Single plot example
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data["counts"][:, np.sort(selected_idx_shared)[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_shared.posterior_samples["predictive_samples"][:, :, i],
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
        f"$\langle U \\rangle = {np.round(mean_shared[np.sort(selected_idx_shared)[i]], 0).astype(int)}$", fontsize=8)

plt.tight_layout()

# Set global title
fig.suptitle("Shared Genes", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_ppc_shared.png", 
    bbox_inches="tight"
)

# # Close figure
# plt.close(fig)
# %% ---------------------------------------------------------------------------

print("Plotting PPC for unique genes...")

# Single plot example
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data["counts"][:, np.sort(selected_idx_unique)[i] + n_shared_genes]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_unique.posterior_samples["predictive_samples"][:, :, i],
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
        cmap='Reds',
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
        f"$\langle U \\rangle = {np.round(mean_unique[np.sort(selected_idx_unique)[i]], 0).astype(int)}$", fontsize=8)

plt.tight_layout()

# Set global title
fig.suptitle("Unique Genes", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_ppc_unique.png", 
    bbox_inches="tight"
)

# # Close figure
# plt.close(fig)
# %% ---------------------------------------------------------------------------

# Compute the KL divergence between the posterior Gamma distributions for the
# r parameters

alpha_r_1 = results.params["alpha_r"][0, :]
beta_r_1 = results.params["beta_r"][0, :]

alpha_r_2 = results.params["alpha_r"][1, :]
beta_r_2 = results.params["beta_r"][1, :]

kl_divergence = scribe.stats.kl_gamma(
    alpha_r_1, beta_r_1, alpha_r_2, beta_r_2
)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot KL divergence ecdf for shared genes
sns.ecdfplot(kl_divergence[0:n_shared_genes], ax=ax, label='Shared Genes')

# Plot KL divergence ecdf for unique genes
sns.ecdfplot(kl_divergence[n_shared_genes:], ax=ax, label='Unique Genes')

# Set labels
ax.set_xlabel(r"$D_{KL}(P_{\text{type}_1}||Q_{\text{type}_2})$")
ax.set_ylabel("ECDF")
ax.set_title(r"KL divergence for $r$ parameters")

# Set y-axis limits
ax.set_ylim(-0.01, 1.01)

# Add legend bottom right
ax.legend(
    loc='lower right',
    fontsize=10
)

# Set x-scale to log
ax.set_xscale('log')

# Save figure
fig.savefig(
    f"{FIG_DIR}/kl_divergence_r.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(3, 3, figsize=(9.5, 9))

# Flatten axes
ax = ax.flatten()

fig.suptitle(r"Shared Genes $r$ posterior distributions", y=1.005, fontsize=18)

# Loop through each gene in shared genes
for i, ax in enumerate(ax):
    # Extract distribution for first type
    distribution_first = stats.gamma(
        results.params["alpha_r"][0, selected_idx_shared[i]],
        loc=0,
        scale=1 / results.params["beta_r"][0, selected_idx_shared[i]]
    )

    # Extract distribution for second type
    distribution_second = stats.gamma(
        results.params["alpha_r"][1, selected_idx_shared[i]],
        loc=0,
        scale=1 / results.params["beta_r"][1, selected_idx_shared[i]]
    )

    # Plot distribution
    scribe.viz.plot_posterior(
        ax,
        distribution_first,
        ground_truth=data["r"][0, selected_idx_shared[i]],
        ground_truth_color="black",
        color=scribe.viz.colors()["dark_blue"],
        fill_color=scribe.viz.colors()["light_blue"],
    )

    scribe.viz.plot_posterior(
        ax,
        distribution_second,
        color=scribe.viz.colors()["dark_red"],
        fill_color=scribe.viz.colors()["light_red"],
    )

    ax.set_title(f"$\langle U \\rangle = {np.round(mean_shared[selected_idx_shared[i]], 0).astype(int)}$")

plt.tight_layout()

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_r_posterior_shared.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(3, 3, figsize=(9.5, 9))

# Flatten axes
ax = ax.flatten()

fig.suptitle(r"Unique Genes $r$ posterior distributions", y=1.005, fontsize=18)

# Loop through each gene in shared genes
for i, ax in enumerate(ax):
    # Extract distribution for first type
    distribution_first = stats.gamma(
        results.params["alpha_r"][0, selected_idx_unique[i] + n_shared_genes],
        loc=0,
        scale=1 / results.params["beta_r"][0, selected_idx_unique[i] + n_shared_genes]
    )

    # Extract distribution for second type
    distribution_second = stats.gamma(
        results.params["alpha_r"][1, selected_idx_unique[i] + n_shared_genes],
        loc=0,
        scale=1 / results.params["beta_r"][1, selected_idx_unique[i] + n_shared_genes]
    )

    # Plot distribution
    scribe.viz.plot_posterior(
        ax,
        distribution_first,
        ground_truth=data["r"][0, selected_idx_unique[i] + n_shared_genes],
        ground_truth_color=scribe.viz.colors()["dark_red"],
        color=scribe.viz.colors()["dark_blue"],
        fill_color=scribe.viz.colors()["light_blue"],
    )

    scribe.viz.plot_posterior(
        ax,
        distribution_second,
        ground_truth=data["r"][1, selected_idx_unique[i] + n_shared_genes],
        ground_truth_color=scribe.viz.colors()["dark_blue"],
        color=scribe.viz.colors()["dark_red"],
        fill_color=scribe.viz.colors()["light_red"],
    )

    ax.set_title(f"$\langle U \\rangle = {np.round(mean_unique[selected_idx_unique[i]], 0).astype(int)}$")

plt.tight_layout()

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_r_posterior_unique.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

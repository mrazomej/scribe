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
n_steps = 25_000
# Define number of cells
n_cells = 10_000
# Define number of genes
n_genes = 20_000

# Define model type
model_type = "zinbvcp"

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
    f"{n_genes}genes.pkl",
    "rb"
) as f:
    data = pickle.load(f)

# Load scribe results
with open(
    f"{OUTPUT_DIR}/scribe_zinbvcp_results_"
    f"{n_cells}cells_"
    f"{n_genes}genes_"
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

print("Plotting ECDF...")
# Define number of genes to select
n_genes = 9

# Compute the mean expression of each gene and sort them
# mean_counts = data["counts"].mean(axis=0)
mean_counts = np.median(data["counts"], axis=0)

# Sort means and get sorting indices
sorted_idx = np.argsort(mean_counts)

# Generate logarithmically spaced indices
log_indices = np.logspace(
    0, np.log10(len(mean_counts) - 1), num=n_genes, dtype=int
)

# Get the actual gene indices after sorting
selected_idx = sorted_idx[log_indices]

# Initialize figure with extra space for legends
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

# Define step size for ECDF
step = 1

# Plot shared genes
for i, idx in enumerate(selected_idx):
    # Plot the ECDF for shared genes
    sns.ecdfplot(
        data=data["counts"][:, idx],
        ax=ax,
        color=sns.color_palette('Blues', n_colors=n_genes)[i],
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
fig.savefig(f"{FIG_DIR}/example_ECDF.png", bbox_inches="tight")

# Close figure
# plt.close(fig)
# %% ---------------------------------------------------------------------------

# Index results for shared genes
results_subset = results[np.sort(selected_idx)]

# %% ---------------------------------------------------------------------------

# Define number of samples
n_samples = 500

print("Generating posterior predictive samples...")
# Generate posterior predictive samples
results_subset.get_ppc_samples(n_samples=n_samples)

# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Single plot example
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data["counts"][:, np.sort(selected_idx)[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.posterior_samples["predictive_samples"][:, :, i],
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
        f"$\\langle U \\rangle = {np.round(mean_counts[np.sort(selected_idx)[i]], 0).astype(int)}$", fontsize=8)

plt.tight_layout()

# Set global title
fig.suptitle("Example PPC", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_ppc.png", 
    bbox_inches="tight"
)

# # Close figure
# plt.close(fig)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(3, 3, figsize=(9.5, 9))

# Flatten axes
ax = ax.flatten()

fig.suptitle(r"$r$ parameter posterior distributions", y=1.005, fontsize=18)

# Loop through each gene in shared genes
for i, ax in enumerate(ax):
    # Extract distribution for first type
    distribution = stats.gamma(
        results.params["alpha_r"][np.sort(selected_idx)[i]],
        loc=0,
        scale=1 / results.params["beta_r"][np.sort(selected_idx)[i]]
    )

    # Plot distribution
    scribe.viz.plot_posterior(
        ax,
        distribution,
        ground_truth=data["r"][np.sort(selected_idx)[i]],
        ground_truth_color="black",
        color=scribe.viz.colors()["dark_blue"],
        fill_color=scribe.viz.colors()["light_blue"],
    )

    ax.set_title(f"$\\langle U \\rangle = {np.round(mean_counts[np.sort(selected_idx)[i]], 0).astype(int)}$")

plt.tight_layout()

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_r_posterior.png", 
    bbox_inches="tight"
)


# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

# Extract p posterior distribution
distribution = stats.beta(results.params["alpha_p"], results.params["beta_p"])

# Plot distribution
scribe.viz.plot_posterior(
    ax,
    distribution,
    ground_truth=data["p"],
    ground_truth_color="black",
    color=scribe.viz.colors()["dark_blue"],
    fill_color=scribe.viz.colors()["light_blue"],
)

# Label axis
ax.set_xlabel("p")
ax.set_ylabel("posterior density")

# Set title
ax.set_title(r"Posterior distribution of $p$")

# Save figure
fig.savefig(
    f"{FIG_DIR}/example_p_posterior.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------
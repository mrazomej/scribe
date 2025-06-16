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
n_steps = 20_000
# Define number of cells
n_cells = 3_000
# Define number of genes
n_genes = 20_000

# Define model type
model_type = "nbdm"
# Define r_distribution
r_distribution = "lognormal"
# Define suffix
suffix = f"r-{r_distribution}_{n_steps}steps"

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
    f"{OUTPUT_DIR}/scribe_nbdm_r-{r_distribution}_results_"
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
fig.savefig(
    f"{FIG_DIR}/loss_history_{suffix}.png", 
    bbox_inches="tight"
)

plt.show()

# %% ---------------------------------------------------------------------------

print("Plotting ECDF...")
# Define number of genes to select
n_genes = 25

# Compute the mean expression of each gene
mean_counts = np.median(data["counts"], axis=0)

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

plt.tight_layout()

# Save figure with extra space for legends
fig.savefig(
    f"{FIG_DIR}/example_ECDF_{suffix}.png", 
    bbox_inches="tight"
)

# Close figure
# plt.close(fig)
# %% ---------------------------------------------------------------------------

# Index results for shared genes
results_subset = results[np.sort(selected_idx)]

# %% ---------------------------------------------------------------------------

# Define number of samples
n_samples = 1_500

print("Generating posterior predictive samples...")
# Generate posterior predictive samples
results_subset.get_ppc_samples(n_samples=n_samples)

# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Single plot example
fig, axes = plt.subplots(5, 5, figsize=(13, 13))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene
    true_counts = data["counts"][:, np.sort(selected_idx)[i]]

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        results_subset.predictive_samples[:, :, i],
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
    f"{FIG_DIR}/example_ppc_{suffix}.png", 
    bbox_inches="tight"
)

# # Close figure
# plt.close(fig)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(5, 5, figsize=(9.5, 9))

# Flatten axes
ax = ax.flatten()

fig.suptitle(r"$r$ parameter posterior distributions", y=1.005, fontsize=18)

# Loop through each gene in shared genes
for i, ax in enumerate(ax):
    # Extract distribution for first type
    distribution = results[int(np.sort(selected_idx)[i])].get_distributions()["r"]

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
    f"{FIG_DIR}/example_r_posterior_{suffix}.png", 
    bbox_inches="tight"
)


# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

# Extract p posterior distribution
distribution = results.get_distributions()["p"]

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
    f"{FIG_DIR}/example_p_posterior_{suffix}.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

# Get distribution of r
r_dist = results.get_distributions()["r"]

# Compute 68% confidence interval quantiles
lower_quantile = r_dist.ppf(0.16)  # 16th percentile
upper_quantile = r_dist.ppf(0.84)  # 84th percentile
median = r_dist.ppf(0.5)  # median

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Plot error bars
ax.errorbar(
    data["r"], 
    median, 
    yerr=[median - lower_quantile, upper_quantile - median],
    color=colors["light_blue"],
    alpha=0.25,
    zorder=0
)

# Plot median
ax.scatter(
    data["r"],
    median,
    color=colors["blue"],
    alpha=0.25
)

# Plot identity line
ax.plot(
    [data["r"].min(), data["r"].max()],
    [data["r"].min(), data["r"].max()],
    color="black",
    linestyle="--",
    zorder=50_000
)

# Set axis labels
ax.set_xlabel(r"ground truth $r$")
ax.set_ylabel(r"posterior $r$")

# Save figure
fig.savefig(
    f"{FIG_DIR}/r_posterior_vs_ground_truth_{suffix}.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

# Extract ground truth parameters
r_true = data["r"]
p_true = data["p"]

# Generate posterior samples
results.get_posterior_samples(n_samples=500)

# Extract parameter samples
r_samples = results.posterior_samples["r"]
p_samples = results.posterior_samples["p"]

# %% ---------------------------------------------------------------------------

# Compute ground truth negative binomial mean
mean_true = r_true * p_true / (1 - p_true)
# Compute ground truth negative binomial variance
var_true = r_true * p_true / (1 - p_true) ** 2

# Compute posterior mean of negative binomial mean
mean_post = r_samples.T * p_samples / (1 - p_samples)
# Compute posterior variance of negative binomial mean
var_post = r_samples.T * p_samples / (1 - p_samples) ** 2

# %% ---------------------------------------------------------------------------

# Compute sample median and quantiles
mean_post_median = jnp.median(mean_post, axis=1)
mean_post_lower = jnp.quantile(mean_post, 0.16, axis=1)
mean_post_upper = jnp.quantile(mean_post, 0.84, axis=1)

# Compute sample median and quantiles
var_post_median = jnp.median(var_post, axis=1)
var_post_lower = jnp.quantile(var_post, 0.16, axis=1)
var_post_upper = jnp.quantile(var_post, 0.84, axis=1)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(6, 3))

# Plot error bars
ax[0].errorbar(
    mean_true, 
    mean_post_median, 
    yerr=[mean_post_median - mean_post_lower, mean_post_upper - mean_post_median],
    color=colors["light_blue"],
    alpha=0.25,
    zorder=0
)


# Plot median vs ground truth
ax[0].scatter(
    mean_true,
    mean_post_median,
    color=colors["blue"],
    alpha=0.25
)

# Plot identity line
ax[0].plot(
    [mean_true.min(), mean_true.max()],
    [mean_true.min(), mean_true.max()],
    color="black",
    linestyle="--",
    zorder=50_000
)

# Label axis
ax[0].set_xlabel(r"ground truth mean counts $\langle u \rangle$")
ax[0].set_ylabel(r"posterior mean counts $\langle u \rangle$")
ax[0].set_title(r"Posterior mean counts $\langle u \rangle$")

# Plot variance error bars
ax[1].errorbar(
    var_true, 
    var_post_median, 
    yerr=[var_post_median - var_post_lower, var_post_upper - var_post_median],
    color=colors["light_blue"],
    alpha=0.25,
    zorder=0
)

# Plot variance median vs ground truth
ax[1].scatter(
    var_true,
    var_post_median,
    color=colors["blue"],
    alpha=0.25
)

# Plot identity line
ax[1].plot(
    [var_true.min(), var_true.max()],
    [var_true.min(), var_true.max()],
    color="black",
    linestyle="--",
    zorder=50_000
)

# Label axis
ax[1].set_xlabel(r"ground truth variance $\sigma^2(u)$")
ax[1].set_ylabel(r"posterior variance $\sigma^2(u)$")
ax[1].set_title(r"Posterior variance $\sigma^2(u)$")

plt.tight_layout()

# Save figure
fig.savefig(
    f"{FIG_DIR}/mean_counts_posterior_vs_ground_truth_{suffix}.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp

# Import NumPyro related libraries
import numpyro.distributions as dist

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

print("Defining directories...")

# Define model type
model_type = "nbvcp_mix"

# Define parameterization
parameterization = "odds_ratio"

# Define number of steps for scribe
n_steps = 50_000

# Define number of components in mixture model
n_components = 2

# Define batch size
batch_size = 1024

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
        f"{OUTPUT_DIR}/"
        f"svi_{parameterization.replace('_', '-')}_"
        f"{model_type.replace('_', '-')}-split_"
        f"{n_components:02d}components_"
        f"{batch_size}batch_"
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
# ax.set_yscale("log")

# Save figure
fig.savefig(
    f"{FIG_DIR}/svi_{parameterization}_{n_steps}steps_loss.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

print("Plotting ECDF for representative genes...")
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

# Index results for shared genes
results_subset = svi_results[selected_idx]

# %% ---------------------------------------------------------------------------

# Define number of samples
n_samples = 1_000

print("Generating posterior predictive samples...")
# Generate posterior predictive samples
results_subset.get_ppc_samples(n_samples=n_samples)

# %% ---------------------------------------------------------------------------

print("Plotting PPC for multiple example genes...")

# Sort genes by mean UMI count for better visualization
gene_means = np.mean(counts[:, selected_idx], axis=0)
sorted_gene_order = np.argsort(gene_means)

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
    max_bin = np.max([cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10])

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
fig.suptitle("Example PPC", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/svi_{parameterization}_{n_steps}steps_ppc.png",
    bbox_inches="tight",
)

# %% ---------------------------------------------------------------------------

print("Plotting KL divergence for mu parameters...")

# Extract mu parameters
mu_loc_1 = svi_results.params["mu_loc"][0, :]
mu_loc_2 = svi_results.params["mu_loc"][1, :]

mu_scale_1 = svi_results.params["mu_scale"][0, :]
mu_scale_2 = svi_results.params["mu_scale"][1, :]

# Compute KL divergence
kl_divergence = scribe.stats.kl_lognormal(
    mu_loc_1, mu_scale_1, mu_loc_2, mu_scale_2
)

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot KL divergence
sns.ecdfplot(kl_divergence, ax=ax)

# Set labels
ax.set_xlabel(r"$D_{KL}(P_{\text{type}_1}||Q_{\text{type}_2})$")
ax.set_ylabel("ECDF")

# Set y-axis limits
ax.set_ylim(-0.01, 1.01)

# Set x-axis as log scale
ax.set_xscale("log")

# Save figure
fig.savefig(
    f"{FIG_DIR}/kl-divergence-mu_{parameterization}_{n_steps}steps.png",
    bbox_inches="tight",
)
# %% ---------------------------------------------------------------------------

print("Sampling from full posterior distribution...")
# Sample from full posterior distribution
svi_results.get_posterior_samples(n_samples=250, canonical=True)

# %% ---------------------------------------------------------------------------

print("Computing cell type assignments...")
# Use posterior samples to assign cell types
cell_types = svi_results.cell_type_probabilities(
    counts=counts,
    batch_size=1024,
    fit_distribution=False,
)

# %% ---------------------------------------------------------------------------

gene_entropy = results.compute_component_entropy(
    counts=jnp.array(counts),
    return_by="gene",
    ignore_nans=True,
    normalize=False,
)


# %% ---------------------------------------------------------------------------

# Evaluate log-likelihood for each sample
gene_log_like = results.compute_log_likelihood(
    counts=jnp.array(counts),
    return_by="gene",
    ignore_nans=True,
    split_components=True,
)

# %% ---------------------------------------------------------------------------

# Normalize log-likelihood by number of cells
gene_log_like_norm = gene_log_like  # / results.n_cells

# Compute log-sum-exp for each sample
gene_log_sum_exp = jsp.special.logsumexp(
    gene_log_like_norm, axis=-1, keepdims=True
)

# Compute log probabilities
gene_log_probs = gene_log_like_norm - gene_log_sum_exp

# Compute entropy per gene
gene_entropy_ = -jnp.sum(gene_log_probs * jnp.exp(gene_log_probs), axis=-1)

# Average entropy over samples
gene_entropy_avg = jnp.mean(gene_entropy_, axis=0)

weights = 1 - gene_entropy_avg / jnp.log(results.n_components)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(figsize=(3.5, 3))

# Plot entropy
sns.ecdfplot(gene_entropy_avg, ax=ax)

# Set labels
ax.set_xlabel("⟨entropy⟩")
ax.set_ylabel("ECDF")

# %% ---------------------------------------------------------------------------

# Compute log-sum-exp for each sample
log_sum_exp = jsp.special.logsumexp(log_likelihood, axis=-1, keepdims=True)

# Compute probabilities
probs = jnp.exp(log_likelihood - log_sum_exp)

# %% ---------------------------------------------------------------------------


print("Computing cell type assignments...")
# Use posterior samples to assign cell types
cell_types = results.compute_cell_type_assignments(
    counts=jnp.array(counts), ignore_nans=True, weigh_by_entropy=True
)

# %% ---------------------------------------------------------------------------

# with jax.experimental.enable_x64():
#     # Use posterior samples to assign cell types
#     cell_types = results.compute_cell_type_assignments(
#         counts=jnp.array(data.X.toarray()),
#         ignore_nans=True,
#         dtype=jnp.float64,
#         batch_size=100
#     )

# Extract mean prob assignments
mean_assignments = cell_types["mean_probabilities"]

# Define cell assignment as class with highest probability
cell_assignments = np.argmax(mean_assignments, axis=1)


# %% ---------------------------------------------------------------------------

# Get posterior samples
results.get_posterior_samples(n_samples=1_500)

# %% ---------------------------------------------------------------------------

# Sample from Dirichlet distribution
dirichlet_samples_first = scribe.stats.sample_dirichlet_from_parameters(
    results.posterior_samples["parameter_samples"]["r"][:, 0, :],
)

dirichlet_samples_second = scribe.stats.sample_dirichlet_from_parameters(
    results.posterior_samples["parameter_samples"]["r"][:, 1, :],
)

# %% ---------------------------------------------------------------------------

# Fit Dirichlet distribution to samples
dirichlet_first_fit = scribe.stats.fit_dirichlet_minka(dirichlet_samples_first)
dirichlet_second_fit = scribe.stats.fit_dirichlet_minka(
    dirichlet_samples_second
)

# %% ---------------------------------------------------------------------------

with jax.experimental.enable_x64():
    # Build Dirichlet distribution from fitted parameters
    dirichlet_first_dist = dist.Dirichlet(
        jnp.array(dirichlet_first_fit).astype(jnp.float64)
    )
    dirichlet_second_dist = dist.Dirichlet(
        jnp.array(dirichlet_second_fit).astype(jnp.float64)
    )

    # Normalize counts by cell
    counts_normalized = jnp.array(
        (counts + 1)
        / ((counts + 1).sum(axis=1, keepdims=True)).astype(jnp.float64)
    )

    # Evaluate log-likelihood for each sample
    log_likelihood_first = dirichlet_first_dist.log_prob(counts_normalized)
    log_likelihood_second = dirichlet_second_dist.log_prob(counts_normalized)

# %% ---------------------------------------------------------------------------

# Compute log-likelihood for each sample
with jax.experimental.enable_x64():
    log_likelihood = results.compute_log_likelihood(
        counts=jnp.array(counts), dtype=jnp.float64, ignore_nans=True
    )

# %% ---------------------------------------------------------------------------

# Compute cell type assignments
with jax.experimental.enable_x64():
    cell_types = results.compute_cell_type_assignments(
        counts=jnp.array(counts), dtype=jnp.float64, fit_distribution=False
    )

# %% ---------------------------------------------------------------------------

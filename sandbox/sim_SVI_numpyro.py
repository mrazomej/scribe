# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import JAX-related libraries
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
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

# Set plotting style
scribe.viz.matplotlib_style()

# %% ---------------------------------------------------------------------------


print("Setting up the simulation...")


# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define number of cells and genes
n_cells = 1_000
n_genes = 31_000

# Define parameters for prior
r_alpha = 2
r_beta = 1
r_prior = (r_alpha, r_beta)

# Split keys for different random operations
key1, key2, key3 = random.split(rng_key, 3)

# Sample true r parameters using JAX's random
r_true = random.gamma(key1, r_alpha, shape=(n_genes,)) / r_beta

# Define prior for p parameter
p_prior = (1, 1)
# Sample true p parameter using JAX's random
p_true = random.beta(key2, p_prior[0], p_prior[1])

# Create negative binomial distribution
nb_dist = dist.NegativeBinomialProbs(r_true, p_true)

# Sample from the distribution
counts_true = nb_dist.sample(key3, sample_shape=(n_cells,))


# %% ---------------------------------------------------------------------------

# Define number of genes to select
n_genes = 9

# Compute the mean expression of each gene and sort them
gene_means = counts_true.mean(axis=0)
sorted_indices = np.argsort(gene_means)[::-1]  # Sort in descending order
gene_means = gene_means[sorted_indices]

# Remove all genes with mean expression less than 1
mask = gene_means > 1
gene_means = gene_means[mask]
valid_indices = sorted_indices[mask]

# Generate logarithmically spaced indices
log_indices = np.logspace(
    0, np.log10(len(gene_means) - 1), num=n_genes, dtype=int
)

# Select genes using the logarithmically spaced indices
selected_indices = valid_indices[log_indices]

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

# Loop through each selected gene
for i, gene_idx in enumerate(selected_indices):
    # Plot the ECDF for each gene
    sns.ecdfplot(
        data=counts_true[:, gene_idx],
        ax=ax,
        color=sns.color_palette('Blues', n_colors=n_genes)[i],
        label=np.round(gene_means[log_indices[i]], 0).astype(int),
        lw=1.5
    )

# Set x-axis to log scale
ax.set_xscale('log')

# Add axis labels
ax.set_xlabel('UMI count')
ax.set_ylabel('ECDF')

# Add legend
ax.legend(loc='lower right', fontsize=8, title=r"$\langle U \rangle$")


# %% ---------------------------------------------------------------------------

# Define file name
file_name = "./output/sim_scribe_result.pkl"

# Check if the file exists
if os.path.exists(file_name):
    # Load the results, the true values, and the counts
    with open(file_name, "rb") as f:
        scribe_results = pickle.load(f)
        counts_true = pickle.load(f)
        r_true = pickle.load(f)
        p_true = pickle.load(f)
else:
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        counts=counts_true,
        n_steps=100_000,
        batch_size=512,
        p_prior=p_prior,
        r_prior=r_prior
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)
        pickle.dump(counts_true, f)
        pickle.dump(r_true, f)
        pickle.dump(p_true, f)

# %% ---------------------------------------------------------------------------

print("Plotting the ELBO loss...")

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
# Plot the ELBO loss
ax.plot(scribe_results.loss_history)
ax.set_xlabel('iteration')
ax.set_ylabel('ELBO loss')

# Set log scale
ax.set_yscale('log')

plt.tight_layout()

# %% ---------------------------------------------------------------------------


print("Plotting parameter posteriors...")

# Plot parameter posteriors
fig = scribe.viz.plot_parameter_posteriors(
    scribe_results,
    p_true,
    r_true,
    n_rows=4,
    n_cols=4
)

# %% ---------------------------------------------------------------------------

# Generate variational posterior samples
scribe_results.sample_posterior()

# %% ---------------------------------------------------------------------------

# Keep subset of inference from log_indices
scribe_results_subset = scribe_results[selected_indices]

# %% ---------------------------------------------------------------------------

# Generate predictive samples
scribe_results_subset.ppc_samples(n_samples=200, resample_parameters=True)

# %% ---------------------------------------------------------------------------

# Plot parameter posteriors
fig = scribe.viz.plot_parameter_posteriors(
    scribe_results_subset,
    p_true,
    r_true[selected_indices.sort()],
    n_rows=3,
    n_cols=3
)

# %% ---------------------------------------------------------------------------

# Single plot example
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        scribe_results_subset.posterior_samples["predictive_samples"][:, :, i],
        credible_regions=[95, 68, 50]
    )

    # Plot credible regions
    scribe.viz.plot_histogram_credible_regions(
        ax, 
        credible_regions,
        cmap='Blues',
        alpha=0.5
    )

    # Extract index for this gene
    gene_idx = selected_indices.sort()[i]

    # Extract true counts for this gene
    true_counts = counts_true[:, gene_idx]

    # Compute histogram of the real data
    hist_results = np.histogram(
        true_counts,
        bins=credible_regions['bin_edges'],
        density=True
    )

    # Plot histogram of the real data as step plot
    ax.step(
        hist_results[1][:-1],
        hist_results[0],
        where='post',
        label='data',
        color='black',
    )

    ax.set_xlabel('counts')
    ax.set_ylabel('frequency')

plt.tight_layout()

# %% ---------------------------------------------------------------------------

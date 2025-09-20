"""
Quickstart
==========

This tutorial will walk you through simulating data from a Negative
Binomial-Dirichlet Multinomial (NBDM) model, fitting it with SCRIBE, and
visualizing the results. See all the available models in the
:doc:`../models/models` documentation.
"""

# %%
# Setup
# -----
#
# First, let's import the necessary libraries and set up our directories:

import os
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scribe
import numpyro.distributions as dist

# Define directories
OUTPUT_DIR = "output"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# %%
# Simulating Data
# ---------------
#
# Next, we'll simulate data from the NBDM model. We'll generate data for 10,000
# cells and 20,000 genes. For memory efficiency, we'll use batching when
# generating samples, generating the samples in the GPU using `JAX
# <https://jax.readthedocs.io/en/latest/>`_ and then moving the samples to the CPU
# using `NumPy <https://numpy.org/>`_.
#
# First, let's set up our simulation parameters:

# Setup random seed
rng_key = random.PRNGKey(42)

# Define dimensions
n_cells = 10_000
n_genes = 20_000
batch_size = 4096

# Define prior parameters
r_prior = (0.0, 1.0)  # Loc and scale for LogNormal prior on r
p_prior = (1, 1)  # Alpha and beta for Beta prior on p

# %%
# Now we'll sample the true parameters from their respective prior distributions:

# Split random keys
key1, key2 = random.split(rng_key, 2)

# Sample true parameters
r_true = dist.LogNormal(r_prior[0], r_prior[1]).sample(
    key1, sample_shape=(n_genes,)
)
p_true = random.beta(key2, p_prior[0], p_prior[1])

# %%
# With our true parameters in hand, we can generate the count data. We'll do this
# in batches to manage memory usage:

counts_true = np.zeros((n_cells, n_genes))

# Sample in batches
for i in range(0, n_cells, batch_size):
    current_batch_size = min(batch_size, n_cells - i)
    key_batch = random.fold_in(rng_key, i)

    # Sample from Negative Binomial distribution
    batch_samples = dist.NegativeBinomialProbs(r_true, p_true).sample(
        key_batch, sample_shape=(current_batch_size,)
    )

    counts_true[i : i + current_batch_size] = np.array(batch_samples)

# %%
# Fitting the Model
# -----------------
#
# Now that we have our simulated data, we can fit it using SCRIBE. We'll run the
# inference for 25,000 steps using the base NBDM model:

n_steps = 25_000

# Run SCRIBE inference with the NBDM model (default settings)
results = scribe.run_scribe(
    counts=counts_true,
    inference_method="svi",  # Use stochastic variational inference
    zero_inflated=False,  # NBDM model has no zero-inflation
    variable_capture=False,  # NBDM model has no variable capture probabilities
    mixture_model=False,  # Not using a mixture model
    r_prior=r_prior,  # Using our defined priors
    p_prior=p_prior,
    n_steps=n_steps,
    batch_size=batch_size,
    seed=42,
)

# %%
# Visualizing Results
# -------------------
#
# Let's create some visualizations to assess our model fit. First, let's look at
# the ELBO loss history:

# Plot loss history
fig, ax = plt.subplots(figsize=(3.5, 3))
ax.plot(results.loss_history)
ax.set_xlabel("step")
ax.set_ylabel("ELBO loss")
plt.tight_layout()
plt.show()

# %%
# We can also compare our inferred parameters to the true values. Let's look at
# the posterior distribution for p:

fig, ax = plt.subplots(figsize=(3.5, 3))

distributions = results.get_distributions()
distribution = distributions["p"]

scribe.viz.plot_posterior(
    ax,
    distribution,
    ground_truth=p_true,
    ground_truth_color="black",
    color=scribe.viz.colors()["dark_blue"],
    fill_color=scribe.viz.colors()["light_blue"],
)

ax.set_xlabel("p")
ax.set_ylabel("posterior density")
plt.tight_layout()
plt.show()

# %%
# Let's generate a similar plot for various examples of the inferred :math:`r`
# parameter:

# Select a few genes to visualize
selected_idx = np.random.choice(n_genes, 9, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(9.5, 9))
axes = axes.flatten()
fig.suptitle(r"$r$ parameter posterior distributions", y=1.005, fontsize=18)

for i, ax in enumerate(axes):
    r_distributions = results.get_distributions(split=True)["r"]
    gene_dist = r_distributions[selected_idx[i]]

    scribe.viz.plot_posterior(
        ax,
        gene_dist,
        ground_truth=r_true[selected_idx[i]],
        ground_truth_color="black",
        color=scribe.viz.colors()["dark_blue"],
        fill_color=scribe.viz.colors()["light_blue"],
    )

    ax.set_xlabel(f"Gene {selected_idx[i]}")

plt.tight_layout()
plt.show()

# %%
# Finally, we can generate posterior predictive checks (PPCs) to assess model fit:

# Generate PPC samples
n_samples = 500
ppc = results.ppc_samples(n_samples=n_samples, rng_key=random.PRNGKey(43))

# Select a gene to visualize
gene_idx = 0

# Plot PPCs for the selected gene
fig, ax = plt.subplots(figsize=(3.5, 3))

credible_regions = scribe.stats.compute_histogram_credible_regions(
    ppc["predictive_samples"][:, :, gene_idx], credible_regions=[95, 68, 50]
)

scribe.viz.plot_histogram_credible_regions_stairs(
    ax, credible_regions, cmap="Blues", alpha=0.5
)

# Plot observed counts for this gene
bin_edges = credible_regions["bin_edges"]
hist, _ = np.histogram(counts_true[:, gene_idx], bins=bin_edges)
hist = hist / hist.sum()  # Normalize
ax.stairs(hist, bin_edges, color="black", alpha=0.8, label="Observed")

ax.set_xlabel(f"Counts for Gene {gene_idx}")
ax.set_ylabel("Probability")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Comparing Models
# ---------------
#
# Let's compare the basic NBDM model with a Zero-Inflated Negative Binomial (ZINB)
# model to see which fits the data better:

# Fit the ZINB model
zinb_results = scribe.run_scribe(
    counts=counts_true,
    inference_method="svi",  # Use stochastic variational inference
    zero_inflated=True,  # Use zero-inflation
    variable_capture=False,
    mixture_model=False,
    r_prior=r_prior,
    p_prior=p_prior,
    gate_prior=(1, 1),  # Prior for dropout probabilities
    n_steps=n_steps,
    batch_size=batch_size,
    seed=42,
)

from scribe.model_comparison import compute_waic

# Compute WAIC for both models
nbdm_waic = compute_waic(
    results, counts_true, n_samples=100, batch_size=batch_size
)

zinb_waic = compute_waic(
    zinb_results, counts_true, n_samples=100, batch_size=batch_size
)

# Display comparison results
print(f"NBDM WAIC: {nbdm_waic['waic_2']:.2f}")
print(f"ZINB WAIC: {zinb_waic['waic_2']:.2f}")
print(f"Delta WAIC: {zinb_waic['waic_2'] - nbdm_waic['waic_2']:.2f}")

if nbdm_waic["waic_2"] < zinb_waic["waic_2"]:
    print("NBDM model fits better (lower WAIC)")
else:
    print("ZINB model fits better (lower WAIC)")

# %%
# Working with the Results
# -----------------------
#
# The results object provides several ways to access and work with the fitted
# model:
#
# 1. **Accessing Parameters**: Get direct access to model parameters

# Access p parameters (Beta distribution)
p_alpha = results.params["p_alpha"]
p_beta = results.params["p_beta"]
print(f"p_alpha: {p_alpha}, p_beta: {p_beta}")

# Access r parameters (LogNormal distribution)
r_loc = results.params["r_loc"]
r_scale = results.params["r_scale"]
print(f"r_loc shape: {r_loc.shape}, r_scale shape: {r_scale.shape}")

# %%
# 2. **Working with Subsets of Genes**: You can use indexing to focus on specific
#    genes

# Get results for the first gene
gene0_results = results[0]

# Get results for genes 0, 10, and 20
selected_genes = results[[0, 10, 20]]

# Boolean indexing also works
highly_variable = np.random.choice([True, False], size=n_genes, p=[0.1, 0.9])
hv_results = results[highly_variable]
print(f"Selected results for {hv_results.n_genes} highly variable genes.")

# %%
# 3. **Computing Log Likelihoods**: Use the log likelihood function to evaluate model fit

# Compute log likelihoods for each cell
log_liks = results.log_likelihood(
    counts_true, return_by="cell", batch_size=batch_size
)
print(f"Log likelihoods computed per cell, shape: {log_liks.shape}")

# Compute log likelihoods for each gene
gene_log_liks = results.log_likelihood(
    counts_true, return_by="gene", batch_size=batch_size
)
print(f"Log likelihoods computed per gene, shape: {gene_log_liks.shape}")

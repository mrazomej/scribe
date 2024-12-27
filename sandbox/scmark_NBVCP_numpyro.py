# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import gc
import pickle

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
from numpyro.optim import Adam
import numpyro

# Import numpy for array manipulation
import numpy as np
import scipy.stats as stats
# Import library to load h5ad files
import anndata as ad
# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Set plotting style
scribe.viz.matplotlib_style()

# Import colors
colors = scribe.viz.colors()

# Change to sandbox directory if not already there
if os.path.basename(os.getcwd()) != "/app/sandbox":
    os.chdir("./sandbox")

# %% ---------------------------------------------------------------------------

# Define file index
FILE_IDX = 1

# Define group to keep
GROUP = "train"

print("Loading data...")

# Get the repository root directory (assuming the script is anywhere in the repo)
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# List all files in the data directory
files = glob.glob(
    os.path.join(
        repo_root, "scrappy", "data", "scmark_v2", "scmark_v2", "*.h5ad"
    )
)

# Read the data
data = ad.read_h5ad(files[FILE_IDX])

# Keep only cells in "test" group
# data = data[data.obs.split == GROUP]

# Extract filename by splitting the path by / and taking the last element
filename = os.path.basename(files[FILE_IDX]).split("/")[-1].split(".")[0]

# %% ---------------------------------------------------------------------------

# Extract the counts into a pandas dataframe
df_counts = pd.DataFrame(
    data.X.toarray(),
    columns=data.var.gene,
    index=data.obs.index
)
# %% ---------------------------------------------------------------------------

# Define number of genes to select
n_genes = 9

# Compute the mean expression of each gene and sort them
df_mean = df_counts.mean().sort_values(ascending=False)

# Remove all genes with mean expression less than 1
df_mean = df_mean[df_mean > 1]

# Generate logarithmically spaced indices
log_indices = np.logspace(
    0, np.log10(len(df_mean) - 1), num=n_genes, dtype=int
)

# Select genes using the logarithmically spaced indices
genes = df_mean.iloc[log_indices].index

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

# Define step size for ECDF
step = 1

# Loop throu each gene
for (i, gene) in enumerate(genes):
    # Plot the ECDF for each column in the DataFrame
    sns.ecdfplot(
        data=df_counts,
        x=gene,
        ax=ax,
        color=sns.color_palette('Blues', n_colors=n_genes)[i],
        label=np.round(df_mean[gene], 0).astype(int),
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
file_name = f"./output/{filename}_{GROUP}_scribe_nbvcp.pkl"

# Check if the file exists
if os.path.exists(file_name):
    # Load the results
    with open(file_name, "rb") as f:
        scribe_result = pickle.load(f)
else:
    # Run SCRIBE
    scribe_result = scribe.svi.run_scribe(
        model_type="zinbvcp",
        counts=data, 
        n_steps=100_000,
        batch_size=1_024,
        optimizer=Adam(step_size=0.001),
        loss=numpyro.infer.TraceMeanField_ELBO(),
        prior_params={
            "p_prior": (1, 1),
            "r_prior": (2, 0.01),
            "p_capture_prior": (1, 1)
        }
    )

    # Clear JAX caches
    jax.clear_caches()
    # Clear memory
    gc.collect()

    # Save the results
    with open(file_name, "wb") as f:
        pickle.dump(scribe_result, f)

# %% ---------------------------------------------------------------------------

# Plot loss_history

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

# Plot loss history
plt.plot(scribe_result.loss_history)

# Add axis labels
ax.set_xlabel('iteration')
ax.set_ylabel('ELBO')

# Set y-scale to log
ax.set_yscale('log')

# %% ---------------------------------------------------------------------------

print("Plotting parameter posteriors...")

# Plot parameter posteriors
fig = scribe.viz.plot_parameter_posteriors(
    scribe_result,
    n_rows=4,
    n_cols=4
)

# %% ---------------------------------------------------------------------------

# Generate variational posterior samples
scribe_result.sample_posterior(n_samples=500)

# %% ---------------------------------------------------------------------------

# Keep subset of inference from log_indices
scribe_result_subset = scribe_result[log_indices]

# %% ---------------------------------------------------------------------------

# scribe_results_subset.sample_posterior(n_samples=250)
# Generate predictive samples
scribe_result_subset.ppc_samples(n_samples=250, resample_parameters=True)

# %% ---------------------------------------------------------------------------

# Plot parameter posteriors
fig = scribe.viz.plot_parameter_posteriors(
    scribe_result_subset,
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
        scribe_result_subset.posterior_samples["predictive_samples"][:, :, i],
        credible_regions=[99.5, 95, 68, 50]
    )

    # Plot credible regions
    scribe.viz.plot_histogram_credible_regions_stairs(
        ax, 
        credible_regions,
        cmap='Blues',
        alpha=0.75
    )

    # Extract true counts for this gene
    true_counts = data.X[:, log_indices[i]].toarray().flatten()

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
        lw=1
    )

    ax.set_xlabel('counts')
    ax.set_ylabel('frequency')

    # Set x-ax limit
    ax.set_xlim(7e-1, 1e1)
    # Set x-ax on log scale
    ax.set_xscale('log')

plt.tight_layout()

# %% ---------------------------------------------------------------------------

# Sample from Dirichlet distribution given the r parameter posterior samples
frac_samples = scribe.stats.sample_dirichlet_from_parameters(
    scribe_result.posterior_samples["parameter_samples"]["r"],
)

# %% ---------------------------------------------------------------------------

# Fit Dirichlet distribution to the samples
dirichlet_fit = scribe.stats.fit_dirichlet_mle(frac_samples)

# %% ---------------------------------------------------------------------------

# Single plot example
fig, axes = plt.subplots(3, 3, figsize=(7.5, 7))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    # Plot fraction samples
    ax.hist(
        frac_samples[:, log_indices[i]],
        bins=100,
        density=True,
        alpha=0.5
    )
    # Extract dirichlet fit for this gene
    alpha_p = dirichlet_fit[log_indices[i]]
    # Sum all other dirichlet fits
    beta_p = jnp.sum(dirichlet_fit) - alpha_p
    # Determine range of x values from quantiles
    # lower_bound = stats.beta.ppf(0.001, alpha_p, beta_p)
    # upper_bound = stats.beta.ppf(0.999, alpha_p, beta_p)
    lower_bound = 0
    upper_bound = 5e-6
    
    # print(upper_bound - lower_bound)
    # # If the range is too small, set the bounds to 0 and 0.0002
    # if upper_bound - lower_bound < 1e-5:
    #     lower_bound = 0
    #     upper_bound = 1e-6


    # Define x values for plotting
    x_p = np.linspace(lower_bound, upper_bound, 100)
    # Define posterior p
    posterior_p = stats.beta.pdf(x_p, alpha_p, beta_p)
    # Plot posterior p
    ax.plot(
        x_p,
        posterior_p,
        color=colors['dark_blue'],
        label='fit'
    )

    ax.set_xlabel('fraction')
    ax.set_ylabel('density')

plt.tight_layout()

# %% ---------------------------------------------------------------------------

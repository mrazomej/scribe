# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro.distributions as dist
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe
# Import pandas for data manipulation
import pandas as pd
# Import Arviz for MCMC diagnostics
import arviz as az
# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

# Define model type
model_type = "nbdm"

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/singer/"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/singer/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/singer/{model_type}"

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Load CSV file
df = pd.read_csv(f"{DATA_DIR}/singer_transcript_counts.csv", comment="#")

# Define data
data = jnp.array(df.to_numpy()).astype(jnp.float64)

# Define number of cells
n_cells = data.shape[0]
# Define number of genes
n_genes = data.shape[1]

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define MCMC burn-in samples
n_mcmc_burnin = 1_000
# Define MCMC samples
n_mcmc_samples = 5_000

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"mcmc_unconstrained_{model_type}_results_" \
        f"{n_cells}cells_" \
        f"{n_genes}genes_" \
        f"{n_mcmc_burnin}burnin_" \
        f"{n_mcmc_samples}samples.pkl"

# Load MCMC results
with open(file_name, "rb") as f:
    mcmc_results = pickle.load(f)

# %% ---------------------------------------------------------------------------

# Get ppc_samples
mcmc_results.get_ppc_samples()


# %% ---------------------------------------------------------------------------

# Build Arviz object
mcmc_az = az.from_numpyro(
    mcmc_results,
)
# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(2, 2, figsize=(12, 6))


# Plot trace
az.plot_trace(
    mcmc_az,
    var_names=["p", "r"],
    axes=ax,
)

plt.tight_layout()

# Save figure
fig.savefig(f"{FIG_DIR}/mcmc_unconstrained_trace.png", bbox_inches="tight")

# %% ---------------------------------------------------------------------------

# Initialize figure
# fig, ax = plt.subplots(2, 2, figsize=(12, 6))

# Plot pair
az.plot_pair(
    mcmc_az,
    var_names=["p", "r"],
    divergences=True,
    textsize=25,
)

plt.tight_layout()

# Get the current figure and save it
fig = plt.gcf()
fig.savefig(f"{FIG_DIR}/mcmc_unconstrained_pairplot.png", bbox_inches="tight")


# %% ---------------------------------------------------------------------------

print("Plotting PPC for 4 genes...")

# Create 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Flatten axes for easier iteration
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene from the dataframe
    true_counts = df.iloc[:, i].values

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        mcmc_results.predictive_samples[:, :, i],
        credible_regions=[95, 68, 50],
        max_bin=true_counts.max()
    )

    # Compute histogram of the real data
    hist_results = np.histogram(
        true_counts,
        bins=credible_regions['bin_edges'],
        density=True
    )

    # Get indices where cumsum <= 0.99
    cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
    # If no indices found (all values > 0.99), use first bin
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

    # Set title with gene name
    ax.set_title(df.columns[i], fontsize=10)

plt.tight_layout()

# Set global title
fig.suptitle("Posterior Predictive Checks", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/mcmc_unconstrained_ppc.png", 
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting ECDF credible regions for 4 genes...")

# Create 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Flatten axes for easier iteration
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene from the dataframe
    true_counts = df.iloc[:, i].values
    # Define max bin
    max_bin = true_counts.max()

    # Compute credible regions
    credible_regions = scribe.stats.compute_ecdf_credible_regions(
        mcmc_results.predictive_samples[:, :, i],
        credible_regions=[95, 68, 50],
        max_bin=max_bin
    )

    # Get x_values - need to add an extra point for stairs function
    x_values = credible_regions['bin_edges']
    
    # For stairs, we need to add an extra point at the end for proper edges
    x_edges = np.append(x_values[0] - 1, x_values)
    
    # Plot credible regions
    scribe.viz.plot_ecdf_credible_regions_stairs(
        ax, 
        credible_regions,
        cmap='Blues',
        alpha=0.5,
        max_bin=max_bin
    )
    
    # Compute empirical CDF for the true data
    ecdf_values = np.zeros(len(x_values))
    for j, x in enumerate(x_values):
        ecdf_values[j] = np.mean(true_counts <= x)
    
    # Plot ECDF of the real data as stairs
    ax.stairs(
        ecdf_values,
        x_edges,  # Use the same extended edges
        label='data',
        color='black',
        linewidth=1.5
    )

    # Set axis labels
    ax.set_xlabel('counts')
    ax.set_ylabel('cumulative probability')

    # Set title with gene name
    ax.set_title(df.columns[i], fontsize=10)
    # Set x-axis limits
    ax.set_xlim(x_values[0] - 1, max_bin - 0.1)

    # Add legend
    ax.legend()

plt.tight_layout()

# Set global title
fig.suptitle("Posterior Predictive Checks", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/mcmc_unconstrained_ppc_ecdf.png", 
    bbox_inches="tight"
)
# %% ---------------------------------------------------------------------------

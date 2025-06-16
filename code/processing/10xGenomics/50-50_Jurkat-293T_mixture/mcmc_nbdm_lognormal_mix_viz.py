# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
# Import NumPyro-related libraries
import numpyro
# Import numpy for numerical operations
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
model_type = "nbdm_log_mix"

# Define number of MCMC burn-in samples
n_mcmc_burnin = 5_000
# Define number of MCMC samples
n_mcmc_samples = 10_000
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

print("Loading MCMC results...")

mcmc_results = pickle.load(open(f"{OUTPUT_DIR}/"
                           f"mcmc_{model_type}_r-{r_distribution}_results_"
                           f"{n_components:02d}components_"
                           f"{n_mcmc_burnin}burnin_"
                           f"{n_mcmc_samples}samples.pkl", "rb"))

# %% ---------------------------------------------------------------------------

print("Loading scribe results...")

svi_results = pickle.load(open(f"{OUTPUT_DIR}/"
                           f"scribe_{model_type}_r-{r_distribution}_results_"
                           f"{n_components:02d}components_"
                           f"{n_steps}steps.pkl", "rb"))
# %% ---------------------------------------------------------------------------

print("Selecting genes...")
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

# %% ---------------------------------------------------------------------------

print("Extracting posterior samples...")

# Extract posterior samples
posterior_samples = mcmc_results.get_samples()

# For r parameter, index selected genes
posterior_samples["r"] = posterior_samples["r"][:, :, selected_idx]

# %% ---------------------------------------------------------------------------

print("Plotting posterior samples for p")

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# Plot histogram of posterior samples
ax.hist(posterior_samples["p"], density=True)

# Label axes
ax.set_xlabel("p")
ax.set_ylabel("density")

# %% ---------------------------------------------------------------------------

print("Generating posterior predictive samples...")

# Extract model function
model, _ = svi_results._model_and_guide()

# Define predictive model
predictive = numpyro.infer.Predictive(model, posterior_samples)

# Generate posterior predictive samples
ppc_samples = predictive(
    random.PRNGKey(0), 
    n_cells=svi_results.n_cells, 
    n_genes=len(selected_idx),
    model_config=svi_results.model_config
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
        ppc_samples["counts"][:, :, i],
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
# fig.savefig(
#     f"{FIG_DIR}/example_ppc_{n_steps}steps.png", 
#     bbox_inches="tight"
# )
# %% ---------------------------------------------------------------------------

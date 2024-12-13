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
# Import numpy for array manipulation
import numpy as np
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

# %% ---------------------------------------------------------------------------

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
data = ad.read_h5ad(files[0])

# Keep only cells in "test" group
data = data[data.obs.split == "test"]

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
file_name = "./output/scmark_scribe_result.pkl"

# Check if the file exists
if os.path.exists(file_name):
    # Load the results
    with open(file_name, "rb") as f:
        scribe_result = pickle.load(f)
else:
    # Run SCRIBE
    scribe_result = scribe.svi.run_scribe(
        counts=data, n_steps=100_000
    )

    # Clear JAX caches
    jax.clear_caches()
    # Clear memory
    gc.collect()

    # Save the results
    with open(file_name, "wb") as f:
        pickle.dump(scribe_result, f)

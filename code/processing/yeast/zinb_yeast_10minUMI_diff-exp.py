# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import pickle

# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import scipy for statistical operations
import scipy as sp
# Import scanpy for single-cell data manipulation
import scanpy as sc
# Import scribe
import scribe

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
scribe.viz.matplotlib_style()

# %% ---------------------------------------------------------------------------

# Define model_type
model_type = "zinb"

# Define training parameters
n_steps = 25_000
batch_size = 4096

# Minimum UMI threshold
min_umi = 10

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"/app/data/yeast"

# Define model directory
MODEL_DIR = f"{scribe.utils.git_root()}/output/yeast/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/yeast/{model_type}/{n_steps}steps"

# Create figure directory if it does not exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)
# %% ---------------------------------------------------------------------------

print("Loading data...")

# List all files in the data directory
file = glob.glob(f"{DATA_DIR}/*h5ad", recursive=True)

# Load dataset
data = sc.read_h5ad(file[0])

# Group by `batch` and extract index for each batch
df_group = data.obs.groupby('batch')

# Extract index for each batch
idxs = df_group.indices

# Extract data for each batch
data_batch = {k: data[v] for k, v in idxs.items()}

# %% ---------------------------------------------------------------------------

print("Calculating total UMI counts per cell...")

# Initialize dictionary to store total UMI counts per cell
umi_counts = {}

# Loop through each batch
for batch in data_batch.keys():
    # Get total UMI counts per cell and flatten from 1 x n matrix to array
    umi_counts[batch] = jnp.ravel(data_batch[batch].X.sum(axis=1))

# %% ---------------------------------------------------------------------------

# Select one batch as example (for now)
batch = list(data_batch.keys())[0]

# Define output file name
output_file = f"{MODEL_DIR}/{batch}_" \
            f"{model_type}_" \
            f"{min_umi}minUMI_" \
            f"{n_steps}steps_" \
            f"{batch_size}batch.pkl"

print(" - Loading model...")
# Load model
with open(output_file, "rb") as f:
    model = pickle.load(f)
# %% ---------------------------------------------------------------------------

# Generate posterior samples
model.get_posterior_samples(n_samples=250)

# %% ---------------------------------------------------------------------------

# Sample from dirichlet distribution given r parameter samples
dirichlet_samples = scribe.stats.sample_dirichlet_from_parameters(
    model.posterior_samples["parameter_samples"]["r"]
)

# %% ---------------------------------------------------------------------------

# Fit Dirichlet distribution to samples
dirichlet_fit = scribe.stats.fit_dirichlet_minka(dirichlet_samples)

# %% ---------------------------------------------------------------------------

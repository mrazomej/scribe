# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import gc
import pickle
import scanpy as sc

# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp

# %% ---------------------------------------------------------------------------

print("Defining inference parameters...")

# Define model_type
model_type = "zinb"

# Define training parameters
n_steps = 25_000
batch_size = 4096

# Minimum UMI threshold
min_umi = 10

# Define prior parameters
prior_params = {
    "p_prior": (1, 1),
    "r_prior": (2, 0.075),
    "gate_prior": (1, 1)
}

# %% ---------------------------------------------------------------------------

print("Setting directories...")

# Define data directory
DATA_DIR = f"/app/data/yeast"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/yeast/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Loop over files
for batch in data_batch.keys():

    print(f"Processing {batch}")

    # Define output file name
    output_file = f"{OUTPUT_DIR}/{batch}_" \
                f"{model_type}_" \
                f"{min_umi}minUMI_" \
                f"{n_steps}steps_" \
                f"{batch_size}batch.pkl"

    # Check if the file exists
    if os.path.exists(output_file):
        print(f"Skipping {file} because it already exists")
        continue

    # Select cells with at least `min_umi` UMI
    mask = umi_counts[batch] >= min_umi
    d = data_batch[batch][mask.tolist()]

    print(f"Running SCRIBE...")

    # Run SCRIBE
    scribe_result = scribe.svi.run_scribe(
        model_type=model_type,
        counts=d,
        n_steps=n_steps,
        prior_params=prior_params,
        batch_size=batch_size,
        rng_key=random.PRNGKey(42),
        stable_update=True
    )

    # Clear JAX caches
    jax.clear_caches()
    # Clear memory
    gc.collect()

    # Save the results
    with open(output_file, "wb") as f:
        pickle.dump(scribe_result, f)
# %% ---------------------------------------------------------------------------
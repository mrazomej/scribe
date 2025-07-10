# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle
import glob
import warnings

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

# Import library for reading 10x Genomics data
import scanpy as sc

# %% ---------------------------------------------------------------------------
print("Setting up the model parameters and output directory...")

# Define model type
model_type = "nbvcp"

# Define parameterization
parameterization = "odds_ratio"

# Define number of steps
n_steps = 40_000

# Define batch size for memory-efficient sampling
batch_size = 2048

# Define dataset directory
DATA_DIR = f"/app/data/zebrahub/count_matrices/*/"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/zebrahub/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Listing the datasets...")

# List all files in the data directory
files = sorted(glob.glob(f"{DATA_DIR}/*bc_matrix.h5", recursive=True))

# Print the number of datasets
print(f"Found {len(files)} datasets")

# %% ---------------------------------------------------------------------------

print(f"Inferring the {model_type} model parameters...")

# Loop over the datasets
for i, file in enumerate(files):
    # Define dataset name
    dataset_name = file.split("/")[-2]

    # Define output file
    output_file = (
        f"{OUTPUT_DIR}/svi_"
        f"{model_type}_"
        f"{parameterization.replace('_', '-')}_"
        f"{n_steps}steps_"
        f"{batch_size}batch_"
        f"{dataset_name}.pkl"
    )

    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Skipping dataset {dataset_name} ({i+1} of {len(files)})...")
        continue

    print(f"Processing dataset {dataset_name} ({i+1} of {len(files)})...")

    print(f"Loading the dataset...")
    # Ignore warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Load the dataset
        data = sc.read_10x_h5(file)

    print(f"Running the inference...")
    # Run scribe
    scribe_results = scribe.run_scribe(
        inference_method="svi",
        parameterization=parameterization,
        counts=data,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    # Save the results
    with open(output_file, "wb") as f:
        pickle.dump(scribe_results, f)


# %% ---------------------------------------------------------------------------

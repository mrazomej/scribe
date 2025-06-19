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
model_type = "nbvcp_mix"

# Define parameterization
parameterization = "odds_ratio"

# Define training parameters
n_steps = 50_000

# Define number of components in mixture model
n_components = 2

# Define batch size
batch_size = 1024

# %% ---------------------------------------------------------------------------

print("Setting directories...")

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/" \
           f"10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/" \
             f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# %% ---------------------------------------------------------------------------

print("Running inference...")


# Define file name
file_name = f"{OUTPUT_DIR}/" \
    f"svi_{parameterization.replace('_', '-')}_" \
    f"{model_type.replace('_', '-')}_" \
    f"{n_components:02d}components_" \
    f"{batch_size}batch_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.run_scribe(
        inference_method="svi",
        counts=data,
        batch_size=batch_size,
        mixture_model=True,
        variable_capture=True,
        n_steps=n_steps,
        parameterization=parameterization,
        n_components=n_components,
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)


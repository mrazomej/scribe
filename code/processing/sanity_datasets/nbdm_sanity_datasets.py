# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import gc
import pickle

# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
from numpyro.optim import Adam
import numpyro

# %% ---------------------------------------------------------------------------

# Define model_type
model_type = "nbdm"

# Define training parameters
n_steps = 25_000
optimizer = Adam(step_size=1e-3)
batch_size_max = 4096

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"/app/data/sanity"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/sanity/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# List all files in the data directory
files = glob.glob(f"{DATA_DIR}/*counts.txt.gz", recursive=True)
# %% ---------------------------------------------------------------------------

# Loop over files
for file in files:
    print(f"Processing {file}")
    # Extract dataset name from file name
    dataset_name = file.split("/")[-1].replace("_counts.txt.gz", "")

    # Define output file name
    output_file = f"{OUTPUT_DIR}/{dataset_name}_{n_steps}steps.pkl"

    # Check if the file exists
    if os.path.exists(output_file):
        print(f"Skipping {file} because it already exists")
        continue

    print(f"Loading data...")
    # Load data
    df = pd.read_csv(file, sep="\t", index_col=0, compression="gzip")

    # Define batch size based on the number of cells
    batch_size = batch_size_max if df.shape[1] >= batch_size_max else None

    # Run SCRIBE
    scribe_result = scribe.svi.run_scribe(
        model_type=model_type,
        counts=df.values,
        n_steps=n_steps,
        optimizer=optimizer,
        cells_axis=1,
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

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
model_type = "nbvcp"

# Define parameterization
parameterization = "linked"

# Define training parameters
n_steps = 25_000
batch_size_max = 2048

# Define latent dimension
latent_dim = 2

# Define if unconstrained
unconstrained = True

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
    file_name = (
        f"{OUTPUT_DIR}/"
        f"dpvae_{parameterization.replace('_', '-')}_"
        f"{model_type.replace('_', '-')}-"
        f"unconstrained_"
        f"{latent_dim}latentdim_"
        f"{n_steps}steps_"
        f"{dataset_name}.pkl"
    )

    # Check if the file exists
    if os.path.exists(file_name):
        print(f"Skipping {file} because it already exists")
        continue

    print(f"Loading data for {dataset_name}...")
    # Load data
    df = pd.read_csv(file, sep="\t", index_col=0, compression="gzip")

    # Define batch size based on the number of cells
    batch_size = batch_size_max if df.shape[1] >= batch_size_max else None

    # Clear JAX caches
    jax.clear_caches()
    # Clear memory
    gc.collect()

    print(f"Running inference for {dataset_name}...")

    # Run SCRIBE
    scribe_result = scribe.run_scribe(
        inference_method="vae",
        counts=jnp.array(df.values),
        variable_capture=True,
        parameterization=parameterization,
        unconstrained=unconstrained,
        n_steps=n_steps,
        batch_size=batch_size,
        cells_axis=1,
        vae_latent_dim=latent_dim,
        vae_hidden_dims=[128, 128, 128, 128],
        vae_activation="relu",
        vae_prior_type="decoupled",
        vae_prior_hidden_dims=[128, 128, 128],
        vae_prior_num_layers=3,
        vae_prior_activation="relu",
        vae_prior_mask_type="alternating",
        vae_standardize=False,
        seed=42,
    )

    print(f"Saving results for {dataset_name}...")

    # Save the results
    with open(file_name, "wb") as f:
        pickle.dump(scribe_result, f)


# %% ---------------------------------------------------------------------------

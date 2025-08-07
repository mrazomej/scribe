# %% ---------------------------------------------------------------------------

import os
import pickle
import gc

# Import JAX-related libraries
import jax

from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# %% ---------------------------------------------------------------------------
# Define model type
model_type = "nbvcp"

# Define parameterization type
parameterization = "linked"

# Define latent dimension
latent_dim = 2

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/singer/"
# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/singer/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Load CSV file
df = pd.read_csv(f"{DATA_DIR}/singer_transcript_counts.csv", comment="#")

# Define data
data = jnp.array(df.to_numpy())

# Define number of cells
n_cells = data.shape[0]
# Define number of genes
n_genes = data.shape[1]

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define training parameters
n_steps = 50_000

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"dpvae_{parameterization}_{model_type}_" \
        f"{n_cells}cells_" \
        f"{n_genes}genes_" \
        f"{latent_dim}latentdim_" \
        f"{n_steps}steps.pkl"

if not os.path.exists(file_name):
    # Run SVI
    vae_results = scribe.run_scribe(
        inference_method="vae",
        counts=data,
        n_steps=n_steps,
        parameterization=parameterization,
        variable_capture=True,
        vae_latent_dim=latent_dim,
        vae_hidden_dims=[128, 128, 128],  # Add encoder/decoder hidden dimensions
        vae_activation="leaky_relu",  # Add encoder/decoder activation
        vae_prior_type="decoupled",
        vae_prior_hidden_dims=[128, 128, 128],
        vae_prior_num_layers=3,
        vae_prior_activation="leaky_relu",
        vae_prior_mask_type="alternating",
        p_prior=(10, 2),
        vae_standardize=False,
    )
    # Save VAE results
    with open(file_name, "wb") as f:
        pickle.dump(vae_results, f)
# %% ---------------------------------------------------------------------------

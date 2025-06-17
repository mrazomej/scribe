# %% ---------------------------------------------------------------------------

import os
import pickle
import gc

# Import JAX-related libraries
import jax

from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
# Import numpy for array manipulation
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# %% ---------------------------------------------------------------------------
# Define model type
model_type = "nbdm_mix"

# Define prior distribution
r_dist = "gamma"

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
data = jnp.array(df.to_numpy()).astype(jnp.float64)

# Define number of cells
n_cells = data.shape[0]
# Define number of genes
n_genes = data.shape[1]

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define training parameters
n_steps = 25_000

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"svi_{r_dist}_{model_type}_results_" \
        f"{n_cells}cells_" \
        f"{n_genes}genes_" \
        f"{n_steps}steps.pkl"

if not os.path.exists(file_name):
    # Run SVI
    svi_results = scribe.svi.run_scribe(
        counts=data,
        mixture_model=True,
        n_components=2,
        mixing_prior=[1.0, 1.0],
        r_dist=r_dist,
        n_steps=n_steps,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(svi_results, f)
# %% ---------------------------------------------------------------------------

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

# Define parameterization type
parameterization = "mean_field"

# Define number of components
n_components = 2

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
n_steps = 50_000

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"svi_{parameterization}_{model_type.replace('_', '-')}_results_" \
        f"{n_cells}cells_" \
        f"{n_genes}genes_" \
        f"{n_steps}steps.pkl"

if not os.path.exists(file_name):
    # Run SVI
    svi_results = scribe.svi.run_scribe(
        counts=data,
        n_steps=n_steps,
        parameterization=parameterization,
        phi_prior=(3, 2),
        mixture_model=True,
        n_components=n_components,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(svi_results, f)
# %% ---------------------------------------------------------------------------

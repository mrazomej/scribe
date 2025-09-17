# %% ---------------------------------------------------------------------------

import os
import pickle
import gc

# Import JAX-related libraries
import jax

from jax import random
import jax.numpy as jnp

# Import pandas for data manipulation
import pandas as pd

# Import scribe
import scribe

# %% ---------------------------------------------------------------------------
# Define model type
model_type = "nbvcp_mix"
# Define parameterization type
parameterization = "odds_ratio"
# Define guide rank
guide_rank = 2
# Define number of components
n_components = 2
# Define whether to use component-specific parameters
component_specific_params = True

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
file_name = (
    f"{OUTPUT_DIR}/"
    f"svi_{parameterization}_"
    f"{model_type.replace('_', '-')}-split_"
    f"{n_components:02d}components_"
    f"{guide_rank:02d}rank_"
    f"{n_cells}cells_"
    f"{n_genes}genes_"
    f"{n_steps}steps.pkl"
)

if not os.path.exists(file_name):
    # Run SVI
    svi_results = scribe.run_scribe(
        inference_method="svi",
        mixture_model=True,
        n_components=n_components,
        variable_capture=True,
        counts=data,
        n_steps=n_steps,
        parameterization=parameterization,
        guide_rank=guide_rank,
        component_specific_params=component_specific_params,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(svi_results, f)
# %% ---------------------------------------------------------------------------

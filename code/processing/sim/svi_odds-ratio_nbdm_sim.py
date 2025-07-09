# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

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

# %% ---------------------------------------------------------------------------

print("Setting up the simulation...")

# Define model type
model_type = "nbdm"

# Define parameterization
parameterization = "odds_ratio"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/sim/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/sim/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Drawing true parameters...")

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define number of cells
n_cells = 10_000

# Define number of genes
n_genes = 20_000

# Define batch size for memory-efficient sampling
batch_size_max = 2048

# Define number of steps for scribe
n_steps = 50_000

# Define parameters for prior
r_mu = 0
r_sigma = 1
r_prior = (r_mu, r_sigma)

# Define prior for p parameter
p_prior = (1, 1)

# Split keys for different random operations
key1, key2, key3, key4 = random.split(rng_key, 4)

# Sample true r parameters using JAX's random
r_true = dist.LogNormal(r_mu, r_sigma).sample(key1, sample_shape=(n_genes,))

# Sample true p parameter using JAX's random
p_true = 1 - dist.Beta(p_prior[0], p_prior[1]).sample(key2)


# %% ---------------------------------------------------------------------------

print("Sampling counts...")

# Define output file name
output_file = f"{OUTPUT_DIR}/" \
    f"data_{n_cells}cells_" \
    f"{n_genes}genes.pkl"

# Check if output file already exists
if not os.path.exists(output_file):
    # Initialize array to store counts (using numpy for memory efficiency)
    counts_true = np.zeros((n_cells, n_genes))

    # Sample in batches
    for i in range(0, n_cells, batch_size_max):
        # Get batch size for this iteration
        current_batch_size = min(batch_size_max, n_cells - i)

        print(f"Sampling from cell index {i} to {i+current_batch_size}...")
        
        # Create new key for this batch
        key5 = random.fold_in(rng_key, i)

        # Sample only for cells belonging to this component
        batch_samples = dist.NegativeBinomialProbs(
            r_true,
            p_true
        ).sample(key5, sample_shape=(current_batch_size,))
            
        # Store batch samples
        counts_true[i:i+current_batch_size] = np.array(batch_samples)

    # Save true values and parameters to file
    with open(output_file, 'wb') as f:
        pickle.dump({
            'counts': np.array(counts_true),
            'r': np.array(r_true),
            'p': np.array(p_true)
        }, f)

# %% ---------------------------------------------------------------------------

# Load true values and parameters from file
with open(output_file, 'rb') as f:
    data = pickle.load(f)

# %% ---------------------------------------------------------------------------

print("Running scribe...")

# Define file name
file_name = f"{OUTPUT_DIR}/" \
    f"scribe_{parameterization}_" \
    f"{model_type}_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.run_scribe(
        counts=jnp.array(data['counts']),
        inference_method="svi",
        parameterization=parameterization,
        n_steps=n_steps,
        batch_size=batch_size_max if n_cells > batch_size_max else None,
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)
# %% ---------------------------------------------------------------------------
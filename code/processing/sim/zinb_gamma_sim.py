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
model_type = "zinb"

# Define r-distribution
r_distribution = "gamma"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/sim/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/sim/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define number of cells
n_cells = 3_000

# Define number of genes
n_genes = 20_000

# Define batch size for memory-efficient sampling
batch_size = 4096

# Define number of steps for scribe
n_steps = 20_000

# Define parameters for prior
r_alpha = 2
r_beta = 1
r_prior = (r_alpha, r_beta) if r_distribution == "gamma" else (1, 1)

# Define prior for p parameter
p_prior = (1, 1)

# Define gate prior
gate_prior = (0.1, 1)

# Split keys for different random operations
key1, key2, key3, key4, key5 = random.split(rng_key, 5)

# Sample true r parameters using JAX's random
r_true = random.gamma(
    key1, r_alpha, shape=(n_genes,)) / r_beta

# Sample true p parameter using JAX's random
p_true = random.beta(key2, p_prior[0], p_prior[1])

# Sample true gate parameter using JAX's random
gate_true = random.beta(key3, gate_prior[0], gate_prior[1], shape=(n_genes,))

# %% ---------------------------------------------------------------------------

# Define output file name
output_file = f"{OUTPUT_DIR}/" \
    f"data_{n_cells}cells_" \
    f"{n_genes}genes.pkl"

# Check if output file already exists
if not os.path.exists(output_file):
    # Initialize array to store counts (using numpy for memory efficiency)
    counts_true = np.zeros((n_cells, n_genes))

    # Sample in batches
    for i in range(0, n_cells, batch_size):
        # Get batch size for this iteration
        current_batch_size = min(batch_size, n_cells - i)

        print(f"Sampling from cell index {i} to {i+current_batch_size}...")
        
        # Create new key for this batch
        key5 = random.fold_in(rng_key, i)

        # Define base distribution  
        base_dist = dist.NegativeBinomialProbs(
            r_true,
            p_true
        )

        # Sample only for cells belonging to this component
        batch_samples = dist.ZeroInflatedDistribution(
            base_dist,
            gate=gate_true
        ).sample(key5, sample_shape=(current_batch_size,))
            
        # Store batch samples
        counts_true[i:i+current_batch_size] = np.array(batch_samples)

    # Save true values and parameters to file
    with open(output_file, 'wb') as f:
        pickle.dump({
            'counts': np.array(counts_true),
            'r': np.array(r_true),
            'p': np.array(p_true),
            'gate': np.array(gate_true)
        }, f)

# %% ---------------------------------------------------------------------------

# Load true values and parameters from file
with open(output_file, 'rb') as f:
    data = pickle.load(f)

# %% ---------------------------------------------------------------------------

# Define file name
file_name = f"{OUTPUT_DIR}/" \
    f"scribe_{model_type}_r-{r_distribution}_results_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        counts=jnp.array(data['counts']),
        zero_inflated=True,
        n_steps=n_steps,
        batch_size=batch_size if n_cells > batch_size else None,
        p_prior=p_prior,
        r_prior=r_prior,
        r_dist=r_distribution,
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)
# %% ---------------------------------------------------------------------------

# Load results
with open(file_name, "rb") as f:
    scribe_results = pickle.load(f)

# %% ---------------------------------------------------------------------------

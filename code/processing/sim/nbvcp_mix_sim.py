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
model_type = "nbvcp_mix"

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
n_cells = 10_000

# Define number of components
n_components = 2

# Define number of shared genes
n_shared_genes = 10_000

# Define number of unique genes
n_unique_genes = 10_000

# Define number of genes
n_genes = n_shared_genes + n_unique_genes

# Define batch size for memory-efficient sampling
batch_size = 2048

# Define number of steps for scribe
n_steps = 50_000

# Define parameters for prior
r_alpha = 2
r_beta = 1
r_prior = (r_alpha, r_beta)

# Define prior for p parameter
p_prior = (1, 1)

# Define prior for p_capture parameter
p_capture_prior = (1, 0.25)

# Split keys for different random operations
key1, key2, key3, key4, key5 = random.split(rng_key, 5)

# Sample true r parameters using JAX's random
r_true_shared = random.gamma(
    key1, r_alpha, shape=(n_shared_genes,)) / r_beta

# Sample true r parameters for unique genes
r_true_unique = random.gamma(
    key1, r_alpha, shape=(n_components, n_unique_genes)) / r_beta

# Combine shared and unique r parameters into matrix of shape (n_components, n_genes + n_unique_genes)
r_true = jnp.vstack([
    jnp.concatenate([r_true_shared, r_true_unique[i]])
    for i in range(n_components)
])

# Sample true p parameter using JAX's random
p_true_shared = random.beta(key2, p_prior[0], p_prior[1])

# Sample true p_capture parameter using JAX's random
p_capture_true = random.beta(
    key3, p_capture_prior[0], p_capture_prior[1], shape=(n_cells,)
)

# Define mixing prior
mixing_prior = (1, 1)

# Define true mixing weights
mixing_weights_true = [0.75, 0.25]

# %% ---------------------------------------------------------------------------

# Define output file name
output_file = f"{OUTPUT_DIR}/" \
    f"data_{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_shared_genes}shared_" \
    f"{n_unique_genes}unique_" \
    f"{n_components:02d}components.pkl"

# Check if output file already exists
if not os.path.exists(output_file):
    # Initialize array to store counts (using numpy for memory efficiency)
    counts_true = np.zeros((n_cells, n_genes))

    # Initialize array to store component assignments
    component_assignments = np.zeros((n_cells,))

    # Sample in batches
    for i in range(0, n_cells, batch_size):
        # Get batch size for this iteration
        current_batch_size = min(batch_size, n_cells - i)

        print(f"Sampling from cell index {i} to {i+current_batch_size}...")
        
        # Split key for component assignments
        key4, subkey = random.split(key4)

        # Get component assignments for this batch
        batch_components = random.categorical(
            subkey, jnp.array(mixing_weights_true), shape=(current_batch_size,))

        # Select p_capture for this batch
        p_capture_batch = p_capture_true[i:i+current_batch_size][:, None]
        
        # Create new key for this batch
        key6, subkey = random.split(key5)
        
        # Store component assignments
        component_assignments[i:i+current_batch_size] = np.array(batch_components)

        # Sample for each component separately to reduce memory usage   
        # Sample for each component separately to reduce memory usage
        for comp in range(n_components):
            # Get indices for this component in the current batch
            comp_mask = batch_components == comp
            if not np.any(comp_mask):
                continue
            # Compute p_hat for batch
            p_hat = p_true_shared * p_capture_batch[comp_mask] / (1 - p_true_shared * (1 - p_capture_batch[comp_mask]))

            # Sample only for cells belonging to this component
            batch_samples = dist.NegativeBinomialProbs(
                r_true[comp],
                p_hat
            ).sample(subkey, sample_shape=(1,))
            
            # Store batch samples
            counts_true[i:i+current_batch_size][comp_mask] = np.array(batch_samples)

    # Save true values and parameters to file
    with open(output_file, 'wb') as f:
        pickle.dump({
            'counts': np.array(counts_true),
            'r': np.array(r_true),
            'p': np.array(p_true_shared),
            'p_capture': np.array(p_capture_true),
            'mixing_weights': np.array(mixing_weights_true),
            'component_assignments': np.array(component_assignments)
        }, f)

# %% ---------------------------------------------------------------------------

# Load true values and parameters from file
with open(output_file, 'rb') as f:
    data = pickle.load(f)

# %% ---------------------------------------------------------------------------

# Define file name
file_name = f"{OUTPUT_DIR}/" \
    f"scribe_nbvcp_mix_results_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_shared_genes}shared_" \
    f"{n_unique_genes}unique_" \
    f"{n_components:02d}components_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        model_type="nbvcp_mix",
        counts=data['counts'],
        n_steps=n_steps,
        batch_size=batch_size,
        n_components=n_components,
        prior_params={
            "p_prior": p_prior,
            "r_prior": r_prior,
            "p_capture_prior": p_capture_prior,
            "mixing_prior": mixing_prior
        }
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)

# %% ---------------------------------------------------------------------------
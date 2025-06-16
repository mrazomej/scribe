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
model_type = "zinb_mix"

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
batch_size = 4096

# Define number of steps for scribe
n_steps = 30_000

# Define r_distribution
r_distribution = "lognormal"

# Define parameters for prior
r_alpha = 2
r_beta = 1
r_prior = (r_alpha, r_beta) if r_distribution == "gamma" else (1, 1)

# Define prior for p parameter
p_prior = (1, 1)

# Split keys for different random operations
key1, key2, key3, key4 = random.split(rng_key, 4)

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

# Define mixing prior
mixing_prior = (1, 1)

# Define true mixing weights
mixing_weights_true = [0.75, 0.25]

# Define gate prior
gate_prior = (0.1, 1)

# Sample true gate parameter using JAX's random
gate_true = random.beta(
    key3, gate_prior[0], gate_prior[1], shape=(n_components, n_genes)
)

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
    # Sample component assignments
    component_assignments = random.categorical(
        key4, jnp.array(mixing_weights_true), shape=(n_cells,))

    # Initialize array to store counts (using numpy for memory efficiency)
    counts_true = np.zeros((n_cells, n_genes))

    # Sample in batches
    for i in range(0, n_cells, batch_size):
        # Get batch size for this iteration
        current_batch_size = min(batch_size, n_cells - i)

        print(f"Sampling from cell index {i} to {i+current_batch_size}...")
        
        # Get component assignments for this batch
        batch_components = component_assignments[i:i+current_batch_size]
        
        # Create new key for this batch
        key5, subkey = random.split(key4)
        
        # Sample for each component separately to reduce memory usage
        for comp in range(n_components):
            # Get indices for this component in the current batch
            comp_mask = batch_components == comp
            if not np.any(comp_mask):
                continue
                
            # Sample only for cells belonging to this component
            base_dist = dist.NegativeBinomialProbs(
                r_true[comp],
                p_true_shared
            )
            
            # Sample only for cells belonging to this component
            batch_samples = dist.ZeroInflatedDistribution(
                base_dist,
                gate=gate_true[comp]
            ).sample(subkey, sample_shape=(comp_mask.sum(),))
            
            # Store batch samples
            counts_true[i:i+current_batch_size][comp_mask] = np.array(batch_samples)

    # Save true values and parameters to file
    with open(output_file, 'wb') as f:
        pickle.dump({
            'counts': np.array(counts_true),
            'r': np.array(r_true),
            'p': np.array(p_true_shared),
            'gate': np.array(gate_true),
            'mixing_weights': np.array(mixing_weights_true),
            'component_assignments': np.array(component_assignments),
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
    f"{n_shared_genes}shared_" \
    f"{n_unique_genes}unique_" \
    f"{n_components:02d}components_" \
    f"{batch_size}batch_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        counts=jnp.array(data['counts']),
        n_steps=n_steps,
        mixture_model=True,
        zero_inflated=True,
        batch_size=batch_size,
        n_components=n_components,
        p_prior=p_prior,
        r_prior=r_prior,
        gate_prior=gate_prior,
        mixing_prior=mixing_prior,
        r_dist=r_distribution,
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)

# %% ---------------------------------------------------------------------------
# %% ---------------------------------------------------------------------------
# Import base libraries
# Import base libraries
# Set the fraction of memory JAX is allowed to use (e.g., 90% of available RAM)
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

# Preallocate a specific amount of memory (in bytes)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Disable the memory preallocation completely
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import pickle
import gc

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe

# %% ---------------------------------------------------------------------------

print("Setting up the simulation...")

# Define model type
model_type = "zinb"
# Define r-distribution
r_distribution = "lognormal"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/sim/{model_type}"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/sim/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define MCMC burn-in samples
n_mcmc_burnin = 2
# Define MCMC samples
n_mcmc_samples = 10

# Define number of cells
n_cells = 3_000

# Define number of genes
n_genes = 1_000

# Define batch size for memory-efficient sampling
batch_size = 4096

# Define number of steps for scribe
n_steps = 20_000

# %% ---------------------------------------------------------------------------

# Define output file name
output_file = f"{OUTPUT_DIR}/" \
    f"data_{n_cells}cells_" \
    f"{n_genes}genes.pkl"

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

# Load results
with open(file_name, "rb") as f:
    scribe_results = pickle.load(f)
# %% ---------------------------------------------------------------------------

# Extract model function
model, _ = scribe_results._model_and_guide()
# Extract model configuration
model_config = scribe_results.model_config

# Extract posterior map from SVI results
param_map = scribe_results.get_map(use_mean=True)

# %% ---------------------------------------------------------------------------

with jax.default_device(jax.devices("cpu")[0]):
    # Clear caches before running
    gc.collect()
    jax.clear_caches()

    # Define output file name
    file_name = f"{OUTPUT_DIR}/" \
            f"mcmc_{model_type}_r-{r_distribution}_results_" \
            f"{n_cells}cells_" \
            f"{n_genes}genes_" \
            f"{n_mcmc_burnin}burnin_" \
            f"{n_mcmc_samples}samples.pkl"

    if not os.path.exists(file_name):
        # Define MCMC sampler with initial position from SVI results
        mcmc_results = MCMC(
            NUTS(model), 
            num_warmup=n_mcmc_burnin, 
            num_samples=n_mcmc_samples,
            chain_method="sequential",
        ) 
        # Run MCMC sampler
        mcmc_results.run(
            random.PRNGKey(0), 
            n_cells=n_cells,
            n_genes=n_genes,
            counts=jnp.array(data["counts"]), 
            model_config=model_config,
            init_params=param_map
        )
        # Save MCMC results
        # with open(file_name, "wb") as f:
        #     pickle.dump(mcmc_results, f)
# %% ---------------------------------------------------------------------------

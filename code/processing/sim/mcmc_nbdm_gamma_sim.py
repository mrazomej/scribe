# %% ---------------------------------------------------------------------------
# Import base libraries
# Set the fraction of memory JAX is allowed to use (e.g., 90% of available RAM)
import os

# Force JAX to use CPU only
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Set XLA optimization flags
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math --xla_cpu_enable_xprof_traceme'

# Set the fraction of memory JAX is allowed to use (e.g., 90% of available RAM)
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
# Enable 64-bit precision
import numpyro
numpyro.enable_x64()

from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe

# %% ---------------------------------------------------------------------------

print("Setting up the simulation...")

# Define model type
model_type = "nbdm"
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

# Define MCMC burn-in samples
n_mcmc_burnin = 1_000
# Define MCMC samples
n_mcmc_samples = 1_000

# Define number of cells
n_cells = 10

# Define number of genes
n_genes = 15

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

# Define kernel kwargs
kernel_kwargs = {
    "target_accept_prob": 0.85,
    "max_tree_depth": (10, 10),
    "step_size": jnp.array(1.0, dtype=jnp.float64),
    "find_heuristic_step_size": False,
    "dense_mass": False,
    "adapt_step_size": True, 
    "adapt_mass_matrix": True,
    "regularize_mass_matrix": False
}

if not os.path.exists(file_name):
    # Run MCMC sampling
    mcmc_results = scribe.mcmc.run_scribe(
        counts=data["counts"],
        num_warmup=n_mcmc_burnin,
        num_samples=n_mcmc_samples,
        kernel_kwargs=kernel_kwargs,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)
# %% ---------------------------------------------------------------------------

# Load MCMC results
with open(file_name, "rb") as f:
    mcmc_results = pickle.load(f)

# %% ---------------------------------------------------------------------------




# %% ---------------------------------------------------------------------------
# Import base libraries
# Set the fraction of memory JAX is allowed to use (e.g., 90% of available RAM)
import gc
import scanpy as sc
import scribe
import jax.numpy as jnp
from jax import random
import jax
import pickle
import os
# Add these near the top with other environment variables
# Use only one GPU with maximum memory efficiency
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Single GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'  # Use almost all memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Use platform allocator
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Enable double precision (Float64)
jax.config.update("jax_enable_x64", True)

# Set up multi-GPU mesh for model sharding
devices = mesh_utils.create_device_mesh((2,))  # 2 GPUs
mesh = Mesh(devices, axis_names=('data',))

print(f"Available devices: {jax.devices()}")
print(f"Device mesh: {mesh}")

# Clear caches before running
gc.collect()
jax.clear_caches()  # In newer JAX versions


# %% ---------------------------------------------------------------------------

print("Setting up MCMC parameters...")

# Setup the PRNG key
rng_key = random.PRNGKey(42)
# Define number of MCMC burn-in samples
n_mcmc_burnin = 5_000  # Restore original values
# Define number of MCMC samples
n_mcmc_samples = 2_500  # Restore original values

# %% ---------------------------------------------------------------------------

print("Setting up directory structure...")

# Define model type
model_type = "nbvcp_mix"

# Define number of components in mixture model
n_components = 2

# Define data directory
DATA_DIR = f"data/10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"/scratch/groups/dpetrov/mrazo/" \
    f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# If the output directory does not exist, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# %% ---------------------------------------------------------------------------

# Define number of genes to use
n_genes = data.n_vars
# Define number of cells to use
n_cells = data.n_obs

# Extract counts
counts = jnp.array(data.X.toarray(), dtype=jnp.float64)

# Add memory usage check and optional data subsetting
print(f"Data shape: {counts.shape}")
print(f"Data memory usage: {counts.nbytes / 1e9:.2f} GB")

# %% ---------------------------------------------------------------------------

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
    f"mcmc_{model_type}_results_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_mcmc_burnin}burnin_" \
    f"{n_mcmc_samples}samples.pkl"

print("Defining kernel kwargs...")

# More aggressive kernel settings for memory efficiency
kernel_kwargs = {
    "target_accept_prob": 0.8,  # Slightly lower
    "max_tree_depth": (6, 6),  # Much more conservative
    "step_size": jnp.array(1.0, dtype=jnp.float64),
    "find_heuristic_step_size": False,
    "dense_mass": False,
    "adapt_step_size": True,
    "adapt_mass_matrix": True,
    "regularize_mass_matrix": False
}

if not os.path.exists(file_name):
    gc.collect()
    jax.clear_caches()

    # Run MCMC sampling with single chain but distributed computation
    mcmc_results = scribe.mcmc.run_scribe(
        counts=counts,  # Use sharded data
        variable_capture=True,
        mixture_model=True,
        n_components=2,
        num_warmup=n_mcmc_burnin,
        num_samples=n_mcmc_samples,
        kernel_kwargs=kernel_kwargs,
        chain_method="vectorized",  # Single chain
    )

    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)

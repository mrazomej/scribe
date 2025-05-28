# %% ---------------------------------------------------------------------------
# Import base libraries
# Set the fraction of memory JAX is allowed to use (e.g., 90% of available RAM)
import jax.experimental.maps as maps
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
import gc
import scanpy as sc
import scribe
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
from jax import random
import jax
import pickle
import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

# Preallocate a specific amount of memory (in bytes)
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # This was duplicated
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Disable the memory preallocation completely
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Add these near the top with other environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Explicitly specify GPUs to use
# Use 90% of available GPU memory
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'

# Import JAX-related libraries
# Enable double precision (Float64)
jax.config.update("jax_enable_x64", True)

# Clear caches before running
gc.collect()
jax.clear_caches()  # In newer JAX versions


# %% ---------------------------------------------------------------------------

print("Setting up MCMC parameters...")

# Setup the PRNG key
rng_key = random.PRNGKey(42)
# Define number of MCMC burn-in samples
n_mcmc_burnin = 5_000
# Define number of MCMC samples
n_mcmc_samples = 4_000

# %% ---------------------------------------------------------------------------

print("Setting up directory structure...")

# Define model type
model_type = "nbvcp_mix"

# Define number of components in mixture model
n_components = 2

# Define data directory
DATA_DIR = f"data/10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"/home/groups/dpetrov/mrazo/" \
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

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
    f"mcmc_{model_type}_results_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_mcmc_burnin}burnin_" \
    f"{n_mcmc_samples}samples.pkl"

print("Defining kernel kwargs...")

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

print("Setting up multi-GPU configuration...")

# Get the number of available devices
n_devices = jax.device_count()
print(f"Number of available devices: {n_devices}")

# Create a mesh for device partitioning
devices = mesh_utils.create_device_mesh((n_devices,))
mesh = Mesh(devices, axis_names=('data',))

# Define the partition specification
partition_spec = PartitionSpec('data')

print("Running MCMC sampling...")

if not os.path.exists(file_name):
    # Run MCMC sampling with device mesh
    with mesh:
        mcmc_results = scribe.mcmc.run_scribe(
            counts=data,
            variable_capture=True,
            mixture_model=True,
            n_components=2,
            num_warmup=n_mcmc_burnin,
            num_samples=n_mcmc_samples,
            kernel_kwargs=kernel_kwargs,
        )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)

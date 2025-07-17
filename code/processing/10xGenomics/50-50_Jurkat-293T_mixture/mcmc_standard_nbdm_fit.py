# %% ---------------------------------------------------------------------------
# Import base libraries
import pickle
import jax
import jax.numpy as jnp
import scribe
import scanpy as sc
import gc
import os

# Set memory fraction to prevent JAX from allocating all GPU memory
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# Add these for tighter memory control
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Use only 70% of GPU memory
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['JAX_PLATFORMS'] = 'gpu'
# Force garbage collection more aggressively
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'  # Better error diagnostics

# Enable double precision (Float64)
jax.config.update("jax_enable_x64", True)

# Clear caches before running
gc.collect()
jax.clear_caches()  # In newer JAX versions

# %% ---------------------------------------------------------------------------

# Define number of MCMC burn-in samples
n_mcmc_burnin = 5_000
# Define number of MCMC samples
n_mcmc_samples = 2_500

# %% ---------------------------------------------------------------------------

# Define model type
model_type = "nbdm"

# Define parameterization
parameterization = "standard"

# Define data directory
DATA_DIR = "data/10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = "/scratch/groups/dpetrov/mrazo/" \
    f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# If the output directory does not exist, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# Define number of genes to use
n_genes = data.n_vars
# Define number of cells to use
n_cells = data.n_obs

# Extract counts
counts = jnp.array(data.X.toarray(), dtype=jnp.float64)

# %% ---------------------------------------------------------------------------

# Clear caches and force garbage collection before MCMC
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
    f"mcmc_{model_type.replace('_', '-')}_" \
    f"{parameterization.replace('_', '-')}_" \
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

print("Running MCMC sampling...")

if not os.path.exists(file_name):
    # Run MCMC sampling
    mcmc_results = scribe.run_scribe(
        inference_method="mcmc",
        parameterization=parameterization,
        counts=counts,
        n_warmup=n_mcmc_burnin,
        n_samples=n_mcmc_samples,
        mcmc_kwargs=kernel_kwargs,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)

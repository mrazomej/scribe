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
# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe
# Import scanpy for loading data
import scanpy as sc

# %% ---------------------------------------------------------------------------

# Define number of MCMC burn-in samples
n_mcmc_burnin = 5_000
# Define number of MCMC samples
n_mcmc_samples = 10_000

# %% ---------------------------------------------------------------------------

# Define model type
model_type = "nbdm_mix"

# Define number of components in mixture model
n_components = 2

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/" \
           f"10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/" \
             f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"


# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"mcmc_{model_type}_results_" \
        f"{n_components:02d}components_" \
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
        counts=data,
        mixture_model=True,
        n_components=2,
        num_warmup=n_mcmc_burnin,
        num_samples=n_mcmc_samples,
        kernel_kwargs=kernel_kwargs,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)
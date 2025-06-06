# %% ---------------------------------------------------------------------------
# Import base libraries
# Set the fraction of memory JAX is allowed to use (e.g., 90% of available RAM)
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
# Define number of MCMC samples
n_steps = 100_000

# %% ---------------------------------------------------------------------------

print("Setting up directory structure...")

# Define model type
model_type = "nbdm_mix"

# Define number of components in mixture model
n_components = 2

# Define r_distribution
r_distribution = "lognormal"

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/" \
    f"10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/" \
    f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# If the output directory does not exist, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define prior parameters
p_prior = (1, 1)
r_prior = (2, 0.1) if r_distribution == "gamma" else (1, 1)
mixing_prior = (47, 47)

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
    f"svi_{model_type}_" \
    f"r-{r_distribution}_" \
    f"{n_components:02d}components_" \
    f"{n_steps}steps.pkl"

print("Running SVI...")

if not os.path.exists(file_name):
    # Run SVI
    svi_results = scribe.svi.run_scribe(
        counts=data,
        mixture_model=True,
        n_components=n_components,
        r_prior=r_prior,
        p_prior=p_prior,
        mixing_prior=mixing_prior,
        r_dist=r_distribution,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(svi_results, f)

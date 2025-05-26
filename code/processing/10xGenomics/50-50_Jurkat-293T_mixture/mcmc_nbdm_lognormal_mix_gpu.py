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

# Define number of MCMC burn-in samples
n_mcmc_burnin = 5_000
# Define number of MCMC samples
n_mcmc_samples = 10_000

# %% ---------------------------------------------------------------------------

# Define model type
model_type = "nbdm_log_mix"

# Define number of steps for scribe
n_steps = 30_000

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


# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# %% ---------------------------------------------------------------------------

print("Loading scribe results...")

svi_results = pickle.load(open(f"{OUTPUT_DIR}/"
                               f"scribe_{model_type}_r-{r_distribution}_results_"
                               f"{n_components:02d}components_"
                               f"{n_steps}steps.pkl", "rb"))

# Define number of genes to use
n_genes = svi_results.n_genes
# Define number of cells to use
n_cells = svi_results.n_cells

# Extract counts
counts = jnp.array(data.X.toarray())

# %% ---------------------------------------------------------------------------

# Extract model function
model, _ = svi_results._model_and_guide()
# Extract model configuration
model_config = svi_results.model_config
# Set prior parameters
model_config.p_param_prior = (1, 1)
model_config.r_param_prior = (1, 1)
model_config.mixing_param_prior = (47, 47)

# Extract posterior map from SVI results
param_map = svi_results.get_map(use_mean=True)
# Keep only first entry of mixing_weights - A two-component model only needs one
# probability value
param_map["mixing_weights"] = jnp.expand_dims(
    param_map["mixing_weights"][0], axis=0
)

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
    f"mcmc_{model_type}_r-{r_distribution}_results_" \
    f"{n_components:02d}components_" \
    f"{n_mcmc_burnin}burnin_" \
    f"{n_mcmc_samples}samples.pkl"

if not os.path.exists(file_name):
    # Define MCMC sampler with initial position from SVI results
    mcmc_results = MCMC(
        NUTS(model),
        num_warmup=n_mcmc_burnin,
        num_samples=n_mcmc_samples
    )
    # Run MCMC sampler
    mcmc_results.run(
        random.PRNGKey(0),
        n_cells=n_cells,
        n_genes=n_genes,
        counts=jnp.array(counts),
        model_config=model_config,
        init_params=param_map
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)
# %% ---------------------------------------------------------------------------

# Load MCMC results
mcmc_results = pickle.load(open(file_name, "rb"))

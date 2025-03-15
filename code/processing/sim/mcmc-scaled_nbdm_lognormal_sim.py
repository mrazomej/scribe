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
import jax.scipy as jsp
# Import Pyro-related libraries
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe

from scribe.models_scaled import loc_scale_factors, nbdm_model_scaled

# %% ---------------------------------------------------------------------------

print("Setting up the simulation...")

# Define model type
model_type = "nbdm"
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
n_mcmc_burnin = 1_000
# Define MCMC samples
n_mcmc_samples = 1_000

# Define number of cells
n_cells = 3_000

# Define number of genes
n_genes = 1_000

# Define batch size for memory-efficient sampling
batch_size = 4096

# Define number of steps for scribe
n_steps = 20_000

# Define parameters for prior
r_alpha = 2
r_beta = 1
r_prior = (r_alpha, r_beta) if r_distribution == "gamma" else (1, 1)

# Define prior for p parameter
p_prior = (1, 1)

# Split keys for different random operations
key1, key2, key3, key4 = random.split(rng_key, 4)

# Sample true r parameters using JAX's random
r_true = random.gamma(
    key1, r_alpha, shape=(n_genes,)) / r_beta

# Sample true p parameter using JAX's random
p_true = random.beta(key2, p_prior[0], p_prior[1])


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

# %% ---------------------------------------------------------------------------

# Generate samples from variational posterior
samples = scribe_results.get_posterior_samples(n_samples=500)

# %% ---------------------------------------------------------------------------

# Compute location and scale factors
loc_factors, scale_factors = loc_scale_factors(samples)

# %% ---------------------------------------------------------------------------

# Extract MAP parameters
param_map = scribe_results.get_map(use_mean=True)
    
# Convert to transformed space and divide by scale
logit_p_init = jsp.special.logit(param_map['p']) / scale_factors['logit_p']
log_r_init = jnp.log(param_map['r']) / scale_factors['log_r']

# Create initial values dict
init_params = {
    'logit_p': logit_p_init,
    'log_r': log_r_init
}

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"mcmc-scaled_{model_type}_r-{r_distribution}_results_" \
        f"{n_cells}cells_" \
        f"{n_genes}genes_" \
        f"{n_mcmc_burnin}burnin_" \
        f"{n_mcmc_samples}samples.pkl"

if not os.path.exists(file_name):
    # Define MCMC sampler with initial position from SVI results
    nuts_kernel = NUTS(
        nbdm_model_scaled,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        regularize_mass_matrix=True,
        max_tree_depth=(15, 10),
    )
    # Define MCMC sampler
    mcmc_results = MCMC(
        nuts_kernel, 
        num_warmup=n_mcmc_burnin, 
        num_samples=n_mcmc_samples,
        chain_method="vectorized",
    ) 
    # Run MCMC sampler
    mcmc_results.run(
        random.PRNGKey(0), 
        n_cells=n_cells,
        n_genes=n_genes,
        counts=jnp.array(data['counts']),
        loc_factors=loc_factors,
        scale_factors=scale_factors,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)
# %% ---------------------------------------------------------------------------


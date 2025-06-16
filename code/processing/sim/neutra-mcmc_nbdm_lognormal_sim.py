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
import numpyro.optim as optim
import numpyro.infer as infer

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
n_genes = 100

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

# Extract model and guide functions
model, guide = scribe_results._model_and_guide()
# Extract model configuration
model_config = scribe_results.model_config

# %% ---------------------------------------------------------------------------

# Define Define filename for autoguide
auto_guide_file_name = f"{OUTPUT_DIR}/" \
    f"autoguide_{model_type}_r-{r_distribution}_results_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_steps}steps.pkl"

if not os.path.exists(auto_guide_file_name):
    # Define AutoNormal guide
    auto_guide = infer.autoguide.AutoDiagonalNormal(model)

    # Define SVI
    svi = infer.SVI(
        model,
        auto_guide,
        loss=infer.Trace_ELBO(),
        optim=optim.Adam(step_size=0.001),
        n_cells=n_cells,
        n_genes=n_genes,
        counts=jnp.array(data["counts"]),
        total_counts=jnp.sum(data["counts"], axis=1),
        model_config=model_config,
    )

    # Run SVI
    auto_guide_results = svi.run(
        random.PRNGKey(42),
        num_steps=n_steps,
    )

    # Save SVI results
    with open(auto_guide_file_name, "wb") as f:
        pickle.dump(auto_guide_results, f)

# Load SVI results
with open(auto_guide_file_name, "rb") as f:
    auto_guide_results = pickle.load(f)

# %% ---------------------------------------------------------------------------

# Define NeuTra reparameterizer
neutra = infer.reparam.NeuTraReparam(auto_guide, auto_guide_results.params)

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"neutra-mcmc_{model_type}_r-{r_distribution}_results_" \
        f"{n_cells}cells_" \
        f"{n_genes}genes_" \
        f"{n_mcmc_burnin}burnin_" \
        f"{n_mcmc_samples}samples.pkl"

if not os.path.exists(file_name):
    # Define MCMC sampler with initial position from SVI results
    nuts_kernel = NUTS(
        model,
        # inverse_mass_matrix=inv_diag_matrix,
        target_accept_prob=0.85,  
        max_tree_depth=(10, 10),
        step_size=1.0,
        find_heuristic_step_size=False,
        dense_mass=[("p", "r")], 
        adapt_step_size=True,
        adapt_mass_matrix=True,
        regularize_mass_matrix=True,
        # init_strategy="",
    )
    # Define HMCECS kernel
    # hmcecs_kernel = HMCECS(nuts_kernel, num_blocks=10)
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
        counts=jnp.array(data["counts"]), 
        total_counts=jnp.sum(data["counts"], axis=1),
        model_config=model_config,
        init_params=init_params,
    )
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)
# %% ---------------------------------------------------------------------------


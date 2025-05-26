# %% ---------------------------------------------------------------------------
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
from jax.scipy.special import expit, logit
# Import Pyro-related libraries
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe
# Import the non-centered model
from scribe.models_noncentered import nbdm_model_noncentered

# %% ---------------------------------------------------------------------------

print("Setting up the simulation...")

# Define model type (updated to reflect non-centered LogNormal)
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

# Define parameters for prior (original gamma prior)
r_alpha = 2
r_beta = 1

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

# Check if output file already exists
if not os.path.exists(output_file):
    # Initialize array to store counts (using numpy for memory efficiency)
    counts_true = np.zeros((n_cells, n_genes))

    # Sample in batches
    for i in range(0, n_cells, batch_size):
        # Get batch size for this iteration
        current_batch_size = min(batch_size, n_cells - i)

        print(f"Sampling from cell index {i} to {i+current_batch_size}...")
        
        # Create new key for this batch
        key5 = random.fold_in(rng_key, i)

        # Sample only for cells belonging to this component
        batch_samples = dist.NegativeBinomialProbs(
            r_true,
            p_true
        ).sample(key5, sample_shape=(current_batch_size,))
            
        # Store batch samples
        counts_true[i:i+current_batch_size] = np.array(batch_samples)

    # Save true values and parameters to file
    with open(output_file, 'wb') as f:
        pickle.dump({
            'counts': np.array(counts_true),
            'r': np.array(r_true),
            'p': np.array(p_true)
        }, f)

# %% ---------------------------------------------------------------------------

# Load true values and parameters from file
with open(output_file, 'rb') as f:
    data = pickle.load(f)

# %% ---------------------------------------------------------------------------
# For non-centered models, we don't need to load existing SVI results
# Instead, we'll use the true parameters to set up informed priors

# Calculate variance for raw parameters
p_var_raw = 1.0  # Standard normal prior for p_raw
r_var_raw = 1.0  # Standard normal prior for r_raw

# Set up diagonal mass matrix directly in the unconstrained raw space
diag_matrix = jnp.zeros((n_genes + 1, n_genes + 1))
diag_matrix = diag_matrix.at[0, 0].set(p_var_raw + 1e-2)  # p_raw variance

for i in range(n_genes):
    diag_matrix = diag_matrix.at[i + 1, i + 1].set(r_var_raw + 1e-2)  # r_raw variance

# Invert to get inverse mass matrix
inv_diag_matrix = jnp.linalg.inv(diag_matrix)

# 4. Set up initial parameters for MCMC 
# Initialize at zero (standard normal mean) with small random noise
init_params = {
    "p_raw": jnp.array(0.0) + 0.01 * random.normal(key3, shape=()),
    "r_raw": jnp.zeros(n_genes) + 0.01 * random.normal(key4, shape=(n_genes,))
}

# %% ---------------------------------------------------------------------------

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
        f"mcmc_{model_type}_r-{r_distribution}_results_" \
        f"{n_mcmc_burnin}burnin_" \
        f"{n_mcmc_samples}samples.pkl"

if not os.path.exists(file_name):
    # Define MCMC sampler with non-centered model
    nuts_kernel = NUTS(
        nbdm_model_noncentered,
        # inverse_mass_matrix=inv_diag_matrix,
        target_accept_prob=0.85,  
        max_tree_depth=10,
        step_size=0.1,  # Larger step size for standardized parameters
        find_heuristic_step_size=True,
        dense_mass=False, 
        adapt_step_size=True,
        adapt_mass_matrix=True,
        regularize_mass_matrix=True,
    )
    
    mcmc_results = MCMC(
        nuts_kernel, 
        num_warmup=n_mcmc_burnin, 
        num_samples=n_mcmc_samples,
        chain_method="sequential",  # Use sequential for better memory usage
    ) 
    
    # Run MCMC sampler with non-centered model
    mcmc_results.run(
        random.PRNGKey(0), 
        n_cells=n_cells,
        n_genes=n_genes,
        counts=jnp.array(data["counts"]), 
        total_counts=jnp.sum(data["counts"], axis=1),
        r_prior_loc=1,#r_prior_loc,
        r_prior_scale=1,#r_prior_scale,
        p_prior_alpha=1,#p_prior_alpha,
        p_prior_beta=1,#p_prior_beta,
        # init_params=init_params,
    )
    
    # Save MCMC results
    with open(file_name, "wb") as f:
        pickle.dump(mcmc_results, f)

# %% ---------------------------------------------------------------------------
# To extract original parameters from raw samples, use these transformations:
# p = expit(logit_p_mean + p_raw * logit_p_std)
# r = jnp.exp(r_prior_loc + r_raw * r_prior_scale)

if os.path.exists(file_name):
    # Load MCMC results
    with open(file_name, "rb") as f:
        mcmc_results = pickle.load(f)
    
    # Get posterior samples
    samples = mcmc_results.get_samples()
    
    # Calculate Beta parameters from priors
    p_mean = p_prior_alpha / (p_prior_alpha + p_prior_beta)
    p_var = (p_prior_alpha * p_prior_beta) / ((p_prior_alpha + p_prior_beta)**2 * (p_prior_alpha + p_prior_beta + 1))
    p_std = jnp.sqrt(p_var)
    
    # Transform to logit space
    logit_p_mean = logit(p_mean)
    logit_p_std = p_std / (p_mean * (1 - p_mean))
    
    # Convert raw parameters back to original scale
    p_samples = expit(logit_p_mean + samples["p_raw"] * logit_p_std)
    r_samples = jnp.exp(r_prior_loc + samples["r_raw"] * r_prior_scale)
    
    # Print some summary statistics of the recovered parameters
    print("\nPosterior Summary:")
    print(f"p (true={p_true:.4f}): mean={jnp.mean(p_samples):.4f}, std={jnp.std(p_samples):.4f}")
    
    r_mean = jnp.mean(r_samples, axis=0)
    print(f"r: mean across genes={jnp.mean(r_mean):.4f}")
    
    # Calculate correlation with true r values
    r_corr = jnp.corrcoef(jnp.mean(r_samples, axis=0), data["r"])[0, 1]
    print(f"Correlation between estimated and true r values: {r_corr:.4f}")
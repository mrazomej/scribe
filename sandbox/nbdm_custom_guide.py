# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro.distributions as dist
import numpyro
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe

# Switch directory to sandbox
os.chdir(f"{scribe.utils.git_root()}/sandbox")

# %% ---------------------------------------------------------------------------

print("Setting up the simulation...")

# Define model type
model_type = "nbdm"

# Define output directory
OUTPUT_DIR = f"./output/"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define number of cells
n_cells = 10_000

# Define number of genes
n_genes = 20_000

# Define batch size for memory-efficient sampling
batch_size = 4096

# Define number of steps for scribe
n_steps = 25_000

# Define parameters for prior
r_alpha = 2
r_beta = 1
r_prior = (r_alpha, r_beta)

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

def nbdm_lognormal_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (0, 1),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Numpyro model for Dirichlet-Multinomial single-cell RNA sequencing data.

    This model assumes a hierarchical structure where:
    0. Each cell has a total count drawn from a Negative Binomial distribution
    1. The counts for individual genes are drawn from a Dirichlet-Multinomial
    distribution conditioned on the total count.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p parameter.
        Default is (1, 1) for a uniform prior.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters.
        Default is (2, 0.1).
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    total_counts : array-like, optional
        Total counts per cell of shape (n_cells,).
        Required if counts is provided.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - Success probability p ~ Beta(p_prior)
        - Gene-specific dispersion r ~ Gamma(r_prior)
        - Total dispersion r_total = sum(r)

    Likelihood:
        - total_counts ~ NegativeBinomial(r_total, p)
        - counts ~ DirichletMultinomial(r, total_counts)
    """
    # Define the prior on the p parameter
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample("r", dist.LogNormal(r_prior[0], r_prior[1]).expand([n_genes]))

    # Sum of r parameters
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts
                )

            # Define plate for cells individual counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts),
                    obs=counts
                )
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx]
                )

            # Define plate for cells individual counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(
                        r, total_count=total_counts[idx]),
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make a NegativeBinomial distribution that returns a vector of
            # length n_genes
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            counts = numpyro.sample("counts", dist_nb)

# %% ---------------------------------------------------------------------------

def nbdm_lognormal_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (0, 1),  # Changed to mean, std for lognormal
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Modified guide function for NBDM model using lognormal prior for r.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p (default: (1,1))
    r_prior : tuple of float
        Parameters (mean, std) for the LogNormal prior on r (default: (0,1))
    counts : array_like, optional
        Observed counts matrix
    total_counts : array_like, optional
        Total counts per cell
    batch_size : int, optional
        Mini-batch size for stochastic optimization
    """
    # Register parameters for p (keep Beta distribution)
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(p_prior[0]),
        constraint=numpyro.distributions.constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(p_prior[1]),
        constraint=numpyro.distributions.constraints.positive
    )

    # Register parameters for r (using LogNormal)
    # LogNormal is parameterized by the mean and std of the underlying normal distribution
    mu_r = numpyro.param(
        "mu_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=numpyro.distributions.constraints.real  # mu can be any real number
    )
    sigma_r = numpyro.param(
        "sigma_r",
        jnp.ones(n_genes) * r_prior[1],
        constraint=numpyro.distributions.constraints.positive
    )

    # Sample from the variational distributions
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    # LogNormal for r
    numpyro.sample("r", dist.LogNormal(mu_r, sigma_r))

# %% ---------------------------------------------------------------------------

# Define param_spec
param_spec = {
    "alpha_p": {"type": "global"},
    "beta_p": {"type": "global"},
    "mu_r": {"type": "gene-specific"},
    "sigma_r": {"type": "gene-specific"}
}

# %% ---------------------------------------------------------------------------

# Define file name
file_name = f"{OUTPUT_DIR}/" \
    f"scribe_nbdm_custom_guide_results_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        model_type="nbdm_lognormal",
        counts=data['counts'],
        custom_model=nbdm_lognormal_model,
        custom_guide=nbdm_lognormal_guide,
        custom_args={
            "total_counts": jnp.sum(data['counts'], axis=1)
        },
        param_spec=param_spec,
        n_steps=n_steps,
        batch_size=batch_size,
        prior_params={
            "p_prior": p_prior,
            "r_prior": (0, 1) 
        }
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)
# %% ---------------------------------------------------------------------------

# Load the results
with open(file_name, "rb") as f:
    scribe_results = pickle.load(f)

# %% ---------------------------------------------------------------------------

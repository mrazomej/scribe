# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import gc
import pickle
import scanpy as sc

# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp

# Import numpyro
import numpyro

# %% ---------------------------------------------------------------------------

print("Defining inference parameters...")

# Define model_type
model_type = "nbdm_mix"

# Define training parameters
n_steps = 20_000

# Define number of components in mixture model
n_components = 2

# Define prior parameters
prior_params = {
    "p_prior": (1, 1),
    "r_prior": (2, 0.075),
    "mixing_prior": (1, 1),
}

# %% ---------------------------------------------------------------------------

print("Setting directories...")

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/" \
           f"10xGenomics/50-50_Jurkat-293T_mixture"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/" \
             f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# Extract the counts
counts = data.X.toarray()

# %% ---------------------------------------------------------------------------

print("Running inference...")


# Define file name
file_name = f"{OUTPUT_DIR}/" \
    f"scribe_{model_type}_results_" \
    f"{n_components:02d}components_" \
    f"{n_steps}steps.pkl"

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        model_type=model_type,
        counts=data,
        n_steps=n_steps,
        n_components=n_components,
        prior_params=prior_params,
        # loss=numpyro.infer.TraceEnum_ELBO(max_plate_nesting=1),
        # batch_size=64,
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)
# %% ---------------------------------------------------------------------------

# Load the results
with open(file_name, "rb") as f:
    scribe_results = pickle.load(f)

# %% ---------------------------------------------------------------------------

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate, infer_discrete
from jax import random
from typing import Union

@config_enumerate
def nbdm_classifier_model(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Modified NBDM mixture model that exposes assignments for classification.
    This version is meant to be used with parameters from a fitted model.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    n_components : int
        Number of mixture components
    mixing_probs : Optional[jnp.ndarray]
        Mixing probabilities from fitted model
    p : Optional[float]
        Success probability parameter from fitted model
    r : Optional[jnp.ndarray]
        Component and gene-specific dispersion parameters from fitted model
    batch_size : Optional[int]
        Mini-batch size for stochastic computation
    """
    # Check if mixing_prior is a tuple
    if isinstance(mixing_prior, tuple):
        if len(mixing_prior) != n_components:
            raise ValueError(
                f"Length of mixing_prior ({len(mixing_prior)}) must match "
                f"number of components ({n_components})"
            )
        mixing_concentration = jnp.array(mixing_prior)
    else:
        mixing_concentration = jnp.ones(n_components) * mixing_prior

    # Sample mixing weights from Dirichlet prior
    mixing_weights = numpyro.sample(
        "mixing_weights",
        dist.Dirichlet(mixing_concentration)
    )

    # Create mixing distribution
    mixing_dist = dist.Categorical(mixing_weights)

    # Define the prior on the p parameters 
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_components, n_genes])
    )

    with numpyro.plate("cells", n_cells):
        # Sample assignments for each cell
        assignment = numpyro.sample(
            "assignment",
            mixing_dist
        )
        
        # Sample counts conditioned on assignments
        numpyro.sample(
            "counts",
            dist.NegativeBinomialProbs(r[assignment], p).to_event(1),
            obs=counts
        )

    # # If batch_size is provided, use mini-batches
    # if batch_size is not None:
    #     with numpyro.plate(
    #         "cells", n_cells, subsample_size=batch_size) as idx:
    #         # Sample assignments for each cell
    #         assignment = numpyro.sample(
    #             "assignment",
    #             mixing_dist
    #         )
            
    #         # Sample counts conditioned on assignments
    #         numpyro.sample(
    #             "counts",
    #             dist.NegativeBinomialProbs(r[assignment[idx]], p).to_event(1),
    #             obs=counts[idx]
    #         )
    # else:
    #     with numpyro.plate("cells", n_cells):
    #         # Sample assignments for each cell
    #         assignment = numpyro.sample(
    #             "assignment",
    #             mixing_dist
    #         )
            
    #         # Sample counts conditioned on assignments
    #         numpyro.sample(
    #             "counts",
    #             dist.NegativeBinomialProbs(r[assignment], p).to_event(1),
    #             obs=counts
    #         )

# %% ---------------------------------------------------------------------------

# Define random key
rng_key = random.PRNGKey(42)

# Extract the parameters from the fitted model
params = scribe_results.params

# substitute trained params
trained_guide = handlers.substitute(
    scribe.models_mix.nbdm_mixture_guide, params
)

# Set the random key in the guide
seeded_trained_guide = handlers.seed(trained_guide, rng_key)

# record the globals
guide_trace = handlers.trace(seeded_trained_guide).get_trace(
    counts=counts,
    n_cells=scribe_results.n_cells,
    n_genes=scribe_results.n_genes,
    n_components=scribe_results.n_components,
)

# replay the globals
trained_model = handlers.replay(nbdm_classifier_model, trace=guide_trace)
# %% ---------------------------------------------------------------------------

def classifier(data, temperature=0, rng_key=random.PRNGKey(42)):
    # Split rng key
    rng_key, rng_key_infer = random.split(rng_key)
    # Infer the discrete model
    # set first_available_dim to avoid conflict with data plate
    inferred_model = infer_discrete(
        trained_model, 
        temperature=temperature, 
        rng_key=rng_key
    )
    # Seed the inferred model
    seeded_inferred_model = handlers.seed(inferred_model, rng_key_infer)
    # Get the trace
    trace = handlers.trace(seeded_inferred_model).get_trace(
        n_cells=scribe_results.n_cells,
        n_genes=scribe_results.n_genes,
        n_components=scribe_results.n_components,
        counts=counts,
    )
    # Return the assignments
    return trace["assignment"]["value"]

# %% ---------------------------------------------------------------------------

classifier(counts[1:3])




# %% ---------------------------------------------------------------------------

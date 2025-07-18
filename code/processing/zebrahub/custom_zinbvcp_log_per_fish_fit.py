# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle
import glob
import warnings

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro.distributions as dist
# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe
# Import library for reading 10x Genomics data
import scanpy as sc

if not os.getcwd().endswith("code/processing/zebrahub"):
    os.chdir("./code/processing/zebrahub")

# Import custom model and guide
from custom_zinbvcp_log_per_fish_model import zinbvcp_log_model, zinbvcp_log_guide

# %% ---------------------------------------------------------------------------

# Define param_spec for model
param_spec = {
    "alpha_p": {"type": "global"},
    "beta_p": {"type": "global"},
    "loc_r": {"type": "gene-specific"},
    "scale_r": {"type": "gene-specific"},
    "alpha_p_capture": {"type": "cell-specific"},
    "beta_p_capture": {"type": "cell-specific"},
    "alpha_gate": {"type": "gene-specific"},
    "beta_gate": {"type": "gene-specific"}
}

# %% ---------------------------------------------------------------------------
print("Setting up the model parameters and output directory...")

# Define model type
model_type = "custom_zinbvcp-log"

# Define number of steps
n_steps = 25_000

# Define batch size for memory-efficient sampling
batch_size = 2048

# Define priors
p_prior = (1, 1)
p_capture_prior = (1, 1)
r_prior = (0, 2)
gate_prior = (1, 1)

# Define dataset directory
DATA_DIR = f"/app/data/zebrahub/count_matrices/*/"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/zebrahub/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Listing the datasets...")

# List all files in the data directory
files = sorted(glob.glob(f"{DATA_DIR}/*bc_matrix.h5", recursive=True))

# Print the number of datasets
print(f"Found {len(files)} datasets")

# %% ---------------------------------------------------------------------------

print(f"Inferring the {model_type} model parameters...")

# Loop over the datasets
for i, file in enumerate(files[9:]):
    # Define dataset name
    dataset_name = file.split('/')[-2]

    # Define output file
    output_file = f"{OUTPUT_DIR}/" \
        f"{model_type}_" \
        f"{n_steps}steps_" \
        f"{batch_size}batch_" \
        f"{dataset_name}.pkl"

    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Skipping dataset {dataset_name} ({i+1} of {len(files)})...")
        continue

    print(f"Processing dataset {dataset_name} ({i+1} of {len(files)})...")


    print(f"Loading the dataset...")
    # Ignore warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Load the dataset
        data = sc.read_10x_h5(file)

    print(f"Running the inference...")
    # Run scribe
    scribe_results = scribe.svi.run_scribe(
        custom_model=zinbvcp_log_model,
        custom_guide=zinbvcp_log_guide,
        counts=data,
        n_steps=n_steps,
        batch_size=batch_size,
        param_spec=param_spec,
        prior_params={
            "p_prior": p_prior,
            "r_prior": r_prior,
            "p_capture_prior": p_capture_prior,
            "gate_prior": gate_prior
        }
    )

    # Save the results
    with open(output_file, "wb") as f:
        pickle.dump(scribe_results, f)


# %% ---------------------------------------------------------------------------

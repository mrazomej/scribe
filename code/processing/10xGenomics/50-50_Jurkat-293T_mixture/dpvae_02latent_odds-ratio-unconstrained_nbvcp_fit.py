# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import gc
import pickle
import scanpy as sc

# Import scribe
import scribe

# Import JAX-related libraries
import jax

# %% ---------------------------------------------------------------------------

print("Defining inference parameters...")

# Define model_type
model_type = "nbvcp"

# Define parameterization
parameterization = "odds_ratio"

# Define if unconstrained
unconstrained = True

# Define training parameters
n_steps = 25_000

# Define latent dimension
latent_dim = 2

# %% ---------------------------------------------------------------------------

print("Setting directories...")

# Define data directory
DATA_DIR = (
    f"{scribe.utils.git_root()}/data/" f"10xGenomics/50-50_Jurkat-293T_mixture"
)

# Define output directory
OUTPUT_DIR = (
    f"{scribe.utils.git_root()}/output/"
    f"10xGenomics/50-50_Jurkat-293T_mixture/{model_type}"
)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load data
data = sc.read_h5ad(f"{DATA_DIR}/data.h5ad")

# %% ---------------------------------------------------------------------------

print("Running inference...")

# Clear caches before running
gc.collect()
jax.clear_caches()

# Define file name
file_name = (
    f"{OUTPUT_DIR}/"
    f"dpvae_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}-"
    f"unconstrained_"
    f"{latent_dim}latentdim_"
    f"{n_steps}steps.pkl"
)

# Check if the file exists
if not os.path.exists(file_name):
    # Run scribe
    scribe_results = scribe.run_scribe(
        inference_method="vae",
        counts=data,
        n_steps=n_steps,
        parameterization=parameterization,
        unconstrained=unconstrained,
        variable_capture=True,
        vae_latent_dim=latent_dim,
        vae_hidden_dims=[128, 128, 128, 128],
        vae_activation="relu",
        vae_prior_type="decoupled",
        vae_prior_hidden_dims=[128, 128, 128],
        vae_prior_num_layers=3,
        vae_prior_activation="relu",
        vae_prior_mask_type="alternating",
        vae_standardize=False,
        batch_size=512,
    )

    # Save the results, the true values, and the counts
    with open(file_name, "wb") as f:
        pickle.dump(scribe_results, f)

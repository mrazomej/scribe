# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle
import glob
import warnings

# Import scribe
import scribe
# Import library for reading 10x Genomics data
import scanpy as sc

# %% ---------------------------------------------------------------------------
print("Setting up the model parameters and output directory...")

# Define model type
model_type = "nbvcp_mix"

# Define number of components
n_components = 2

# Define number of steps
n_steps = 30_000

# Define batch size for memory-efficient sampling
batch_size = 4096

# Define priors
p_prior = (1, 1)
p_capture_prior = (1, 1)
r_prior = (2, 1)
mixing_prior = (1, 1)

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
for i, file in enumerate(files):
    # Define dataset name
    dataset_name = file.split('/')[-2]

    # Define output file
    output_file = f"{OUTPUT_DIR}/" \
        f"{model_type}_" \
        f"{n_components}components_" \
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
        model_type=model_type,
        counts=data,
        n_steps=n_steps,
        n_components=n_components,
        batch_size=batch_size,
        prior_params={
            "p_prior": p_prior,
            "r_prior": r_prior,
            "p_capture_prior": p_capture_prior,
            "mixing_prior": mixing_prior
        }
    )

    # Save the results
    with open(output_file, "wb") as f:
        pickle.dump(scribe_results, f)


# %% ---------------------------------------------------------------------------

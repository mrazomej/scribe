# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle
import glob
import warnings

# Import scribe
import scribe
# Import library for reading 10x Genomics data
import anndata as ad

# %% ---------------------------------------------------------------------------
print("Setting up the model parameters and output directory...")

# Define model type
model_type = "nbvcp_mix"

# Define number of steps
n_steps = 50_000

# Define batch size for memory-efficient sampling
batch_size = 2048

# Define priors
p_prior = (1, 1)
p_capture_prior = (1, 1)
r_prior = (2, 0.075)

# Define dataset directory
DATA_DIR = f"/app/data/scmark_v2/scmark_v2/"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/scmark/{model_type}"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Listing the datasets...")

# List all files in the data directory
files = sorted(glob.glob(f"{DATA_DIR}/*.h5ad", recursive=True))

# Print the number of datasets
print(f"Found {len(files)} datasets")

# %% ---------------------------------------------------------------------------

# Loop over the datasets
for i, file in enumerate(files[3:]):
    # Define dataset name
    dataset_name = file.split("/")[-1].split(".")[0]

    print(f"Loading the dataset...")
    # Ignore warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Load the dataset
        data = ad.read_h5ad(file)

    # Define number of components
    n_components = len(data.obs["standard_true_celltype"].unique())

    # Define mixing prior
    mixing_prior = tuple([1] * n_components)


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
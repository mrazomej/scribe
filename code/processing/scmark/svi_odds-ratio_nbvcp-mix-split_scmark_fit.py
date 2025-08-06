# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle
import glob
import warnings

# Import numpy
import numpy as np
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

# Define parameterization
parameterization = "odds_ratio"

# Define component_specific_params
component_specific_params = True

# Define batch size for memory-efficient sampling
batch_size = 2048

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
for i, file in enumerate(files):
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

    # Find genes with all zero counts
    zero_counts = np.all(data.X.toarray() == 0, axis=0)

    # Remove all zero genes
    data = data[:, ~zero_counts]

    # Define output file
    output_file = f"{OUTPUT_DIR}/" \
        f"svi_{model_type.replace('_', '-')}-split_" \
        f"{parameterization.replace('_', '-')}_" \
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
    scribe_results = scribe.run_scribe(
        inference_method="svi",
        parameterization=parameterization,
        counts=data,
        variable_capture=True,
        mixture_model=True,
        n_components=n_components,
        n_steps=n_steps,
        batch_size=batch_size,
        component_specific_params=component_specific_params,
    )

    # Save the results
    with open(output_file, "wb") as f:
        pickle.dump(scribe_results, f)
# %% ---------------------------------------------------------------------------

# %% ---------------------------------------------------------------------------

# Import base libraries
import os
import glob
import pickle

# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import scribe
import scribe

# Import JAX-related libraries
import jax
from jax import random
import jax.numpy as jnp

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
scribe.viz.matplotlib_style()

# Allow for float64 precision
jax.config.update("jax_enable_x64", True)

# %% ---------------------------------------------------------------------------

# Define batch size
batch_size = 512

# Define max number of samples
max_n_samples = 1024

# Define min number of samples
min_n_samples = 256

# Define model_type
model_types = ["nbvcp", "zinbvcp", "zinb", "nbdm"]

# Define training parameters
n_steps = {
    "nbvcp": 25_000,
    "zinbvcp": 25_000,
    "zinb": 50_000,
    "nbdm": 50_000,
}

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/sanity"

# Define model directory
MODEL_DIRS = [
    f"{scribe.utils.git_root()}/output/sanity/{model_type}"
    for model_type in model_types
]

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/sanity/model_comparison"

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# List all files in the data directory
files = glob.glob(f"{DATA_DIR}/*counts.txt.gz", recursive=True)

# Initialize dataframe to store metadata
df_meta = pd.DataFrame()

# Loop through each file
for file in files:
    # Extract dataset name from file name
    dataset_name = file.split("/")[-1].replace("_counts.txt.gz", "")

    # Define model file names
    model_files = {
        model_type: f"{model_dir}/{dataset_name}_{n_steps[model_type]}steps.pkl"
        for model_dir, model_type in zip(MODEL_DIRS, model_types)
    }

    # Add metadata to dataframe
    df_meta = pd.concat([
        df_meta,
        pd.DataFrame(
            {
                "dataset_name": dataset_name,
                "data_file": file,
                **model_files
            },
            index=[0]
        )
    ], ignore_index=True)
# %% ---------------------------------------------------------------------------

# Initialize dataframe to store results if it does not exist
if not os.path.exists(f"{OUTPUT_DIR}/model_comparison.csv"):
    df_comparison = pd.DataFrame()
else:
    df_comparison = pd.read_csv(f"{OUTPUT_DIR}/model_comparison.csv")

# Loop through each dataset
for _, df in df_meta.iterrows():
    print(f"Processing {df['dataset_name']}...")

    # If dataframe is not empty and dataset is already in it, skip
    if not df_comparison.empty and df["dataset_name"] in df_comparison["dataset_name"].values:
        print(f"    - Skipping {df['dataset_name']}...")
        continue

    print(f"    - Loading data...")
    # Load data
    with open(df.data_file, "rb") as f:
        df_counts = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")

    # Extract number of cells
    n_cells = df_counts.shape[1]

    print(f"    - Loading models...")
    # Load models
    models = [
        pickle.load(open(df[model_type], "rb")) for model_type in model_types
    ]

    # Define number of smaples
    n_samples = max_n_samples if n_cells < 2000 else min_n_samples

    print(f"    - Computing WAIC...")
    # Compute WAIC
    waic = scribe.model_comparison.compare_models(
        models, 
        jnp.array(df_counts.values, dtype=jnp.float64), 
        n_samples=n_samples, 
        batch_size=batch_size, 
        ignore_nans=True,
        rng_key=random.PRNGKey(0),
        cells_axis=1,
        dtype=jnp.float64,
    )

    # Add column to dataframe with dataset name
    waic["dataset_name"] = df.dataset_name

    # Add number of samples
    waic["n_samples"] = n_samples

    # Add number of cells
    waic["n_cells"] = n_cells

    # Add results to dataframe
    df_comparison = pd.concat([
        df_comparison,
        waic
    ], ignore_index=True)

    # Save partial results
    df_comparison.to_csv(
        f"{OUTPUT_DIR}/model_comparison.csv", index=False
    )

# %% ---------------------------------------------------------------------------

# Save results
df_comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# %% ---------------------------------------------------------------------------

# Initialize dataframe to store results if it does not exist
if not os.path.exists(f"{OUTPUT_DIR}/model_comparison_by_gene.csv"):
    df_comparison = pd.DataFrame()
else:
    df_comparison = pd.read_csv(f"{OUTPUT_DIR}/model_comparison_by_gene.csv")

# Loop through each dataset
for _, df in df_meta.iterrows():
    print(f"Processing {df['dataset_name']}...")

    # If dataframe is not empty and dataset is already in it, skip
    if not df_comparison.empty and df["dataset_name"] in df_comparison["dataset_name"].values:
        print(f"    - Skipping {df['dataset_name']}...")
        continue

    print(f"    - Loading data...")
    # Load data
    with open(df.data_file, "rb") as f:
        df_counts = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")

    # Extract number of cells
    n_cells = df_counts.shape[1]

    print(f"    - Loading models...")
    # Load models
    models = [
        pickle.load(open(df[model_type], "rb")) for model_type in model_types
    ]

    # Define number of smaples
    n_samples = max_n_samples if n_cells < 2000 else min_n_samples

    print(f"    - Computing WAIC...")
    # Compute WAIC
    waic = scribe.model_comparison.compare_models_by_gene(
        models, 
        jnp.array(df_counts.values, dtype=jnp.float64), 
        n_samples=n_samples, 
        batch_size=batch_size, 
        ignore_nans=True,
        rng_key=random.PRNGKey(0),
        cells_axis=1,
        dtype=jnp.float64,
    )

    # Add column to dataframe with dataset name
    waic["dataset_name"] = df.dataset_name

    # Add number of samples
    waic["n_samples"] = n_samples

    # Add number of cells
    waic["n_cells"] = n_cells

    # Add results to dataframe
    df_comparison = pd.concat([
        df_comparison,
        waic
    ], ignore_index=True)

    # Save partial results
    df_comparison.to_csv(
        f"{OUTPUT_DIR}/model_comparison_by_gene.csv", index=False
    )

# %% ---------------------------------------------------------------------------

# Save results
df_comparison.to_csv(f"{OUTPUT_DIR}/model_comparison_by_gene.csv", index=False)

# %% ---------------------------------------------------------------------------

# Import base libraries
import os

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

# %% ---------------------------------------------------------------------------

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/sanity"

# Define model comparison directory
COMPARISON_DIR = f"{scribe.utils.git_root()}/output/sanity/model_comparison"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/sanity/model_comparison"

# Create figure directory if it does not exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

# Load model comparison by gene
df_comparison = pd.read_csv(f"{COMPARISON_DIR}/model_comparison.csv")

# %% ---------------------------------------------------------------------------

# Group by dataset_name
df_group = df_comparison.groupby("dataset_name")
# %% ---------------------------------------------------------------------------

# Select group (for now)

# Get list of unique dataset names
dataset_names = df_comparison["dataset_name"].unique()
# %% ---------------------------------------------------------------------------

# Define columns to plot
cols = ["lppd", "elppd_waic_1", "elppd_waic_2", "waic_1", "waic_2"]

# Define scale for each column
scales = {
    "lppd": "symlog",
    "elppd_waic_1": "symlog", 
    "elppd_waic_2": "symlog",
    "waic_1": "symlog",
    "waic_2": "symlog"
}

# Loop over each dataset
for (dataset_name, data) in df_group:
    # Initialize figure with extra width for legend
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    # Flatten axes
    axes = axes.flatten()

    # Store legend handles and labels
    legend_handles = []
    legend_labels = []

    # Loop over each column to plot
    for (i, x) in enumerate(cols):
        # Plot lppd ECDF for each model_type in data, with consistent hue order
        ecdf_plot = sns.ecdfplot(
            data=data, x=x, ax=axes[i], hue="model",
            hue_order=["nbdm", "zinb", "nbvcp", "zinbvcp"],
            palette=sns.color_palette("tab10"),
            legend=False)  # Don't show individual legends

        # Get handles and labels from first plot only
        if i == 0:
            legend_handles = ecdf_plot.get_lines()
            legend_labels = ["nbdm", "zinb", "nbvcp", "zinbvcp"]

        # Set x-scale to log
        # axes[i].set_xscale(scales[x])

        # Set x-axis label
        axes[i].set_xlabel(f"{x.replace('_', ' ').upper()}")

        # Set y-axis label
        axes[i].set_ylabel("ECDF")

        # Set subplot title
        axes[i].set_title(f"{x.replace('_', ' ').upper()}")

        # Set y-axis limit
        axes[i].set_ylim(-0.015, 1.015)

    # Set global title
    fig.suptitle(f"{dataset_name.replace('_', ' ')}")

    # Add single legend to the right of the plots
    fig.legend(legend_handles, legend_labels, 
              bbox_to_anchor=(1, 0.5),
              loc='center left',
              title="Model")

    # Adjust subplot spacing
    plt.tight_layout()

    # Save figure with extra width for legend
    fig.savefig(f"{FIG_DIR}/{dataset_name}_ecdfs.pdf", 
                bbox_inches="tight",
                dpi=300)

    # Close figure
    plt.close(fig)

# %% ---------------------------------------------------------------------------

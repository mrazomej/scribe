# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import numpy for array manipulation
import numpy as np
# Import scribe
import scribe
# Import pandas for data manipulation
import pandas as pd
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import seaborn for plotting
import seaborn as sns

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

print("Defining directories...")

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/singer/"

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/supplementary"

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)

# %% ---------------------------------------------------------------------------

print("Loading data...")

# Load CSV file
df = pd.read_csv(f"{DATA_DIR}/singer_transcript_counts.csv", comment="#")

# Define data
data = df.to_numpy()

# Define number of cells
n_cells = data.shape[0]
# Define number of genes
n_genes = data.shape[1]

# %% ---------------------------------------------------------------------------

print("Plotting histogram...")

# Initialize figure
fig, axes = plt.subplots(2, 2, figsize=(5, 5))

# Flatten axes
axes = axes.flatten()

# Loop through genes (columns of dataframe)
for i in range(n_genes):
    # Get gene expression
    gene_expression = df.iloc[:, i]

    # Plot histogram
    sns.histplot(
        gene_expression,
        ax=axes[i],
        bins=range(int(gene_expression.max()) + 1)
    )

    # Set axis labels
    axes[i].set_xlabel("mRNA count")
    axes[i].set_ylabel("count")

    # Set title as column name
    axes[i].set_title(df.columns[i])

plt.tight_layout()

# Save figure
fig.savefig(f"{FIG_DIR}/figSI_singer_hist.png", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/figSI_singer_hist.pdf", bbox_inches="tight")


# %% ---------------------------------------------------------------------------

print("Plotting ECDF...")

# Initialize figure
fig, axes = plt.subplots(2, 2, figsize=(5, 5))

# Flatten axes
axes = axes.flatten()

# Loop through genes (columns of dataframe)
for i in range(n_genes):
    # Get gene expression
    gene_expression = df.iloc[:, i]

    # Plot ECDF
    sns.ecdfplot(gene_expression, ax=axes[i])

    # Set axis labels
    axes[i].set_xlabel("mRNA count")
    axes[i].set_ylabel("ECDF")

    # Set title as column name
    axes[i].set_title(df.columns[i])

    # Set y-axis limits
    axes[i].set_ylim(-0.05, 1.05)

plt.tight_layout()

# Save figure
fig.savefig(f"{FIG_DIR}/figSI_singer_ecdf.png", bbox_inches="tight")
fig.savefig(f"{FIG_DIR}/figSI_singer_ecdf.pdf", bbox_inches="tight")


# %% ---------------------------------------------------------------------------

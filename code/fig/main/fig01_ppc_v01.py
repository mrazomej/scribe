# %% ---------------------------------------------------------------------------

# Import basic packages
import sys
from pathlib import Path
import pickle
import json

# Add parent directory to sys.path to allow importing figure utilities
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import get_fig_dir

# Import our main package
import scribe

# Import useful tools
import numpy as np
import scanpy as sc
import jax.numpy as jnp

# Import plotting packages
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

# Set our plotting style (totally optional)
scribe.viz.matplotlib_style()

# %% ---------------------------------------------------------------------------

# Load the local tutorial path configuration.
with (Path(__file__).parent.parent.parent / "data_paths.local.json").open(
    "r", encoding="utf-8"
) as _f:
    _data_root = json.load(_f)["SCRIBE_DATA_ROOT"]

# Build the dataset-specific path relative to the configured root.
data_dir = Path(_data_root).expanduser() / "5050_jurkat-293t"

# Load the data
adata = scribe.data_loader.load_and_preprocess_anndata(
    data_dir, return_jax=False
)

adata
# %% ---------------------------------------------------------------------------

# Define output directory
out_dir = data_dir / "scribe_results"
out_dir.mkdir(parents=True, exist_ok=True)

# Define parameterization
_parameterization = "mean_odds"

# Define output file path
_out_path = out_dir / f"scribe_results_nbvcp_{_parameterization}.pkl"

if _out_path.exists():
    # Load model from pkl file
    with open(_out_path, "rb") as f:
        scribe_results = pickle.load(f)
else:
    # Fit basic model to data with variable capture probability fit per cell
    scribe_results = scribe.fit(
        adata,
        variable_capture=True,
        parameterization=_parameterization,
        n_components=2,
        early_stopping={
            "enabled": True,
            "patience": 1000,
            "checkpoint_dir": str(_out_path.with_suffix("")),
        },
    )
    # Save the fitted model
    with open(_out_path, "wb") as f:
        pickle.dump(scribe_results, f)

scribe_results
# %% ---------------------------------------------------------------------------

# 1. Run the PPC plot to get the PlotResult wrapper
_fig = scribe.viz.plot_ppc(
    scribe_results,
    adata,
    genes=["RPS4X"],
    n_rows=1,
    figsize=(2.5, 2.5),
    n_samples=512,
)
# 2. Clear the figure-level title (the main title "Example PPC")
_fig.fig.suptitle("")

# 3. Iterate over each panel axis to clear panel titles and y-axis ticks
for ax in _fig.axes:
    # Remove the panel-level title/subtitle (e.g. "RPS4X" and "<U> = ...")
    ax.set_title("")

    # Remove the y-axis ticks and tick labels completely
    ax.set_yticks([])

    # Remove the grid lines
    ax.grid(False)

# 4. Define custom proxy artists
data_handle = mlines.Line2D([], [], color="black", linewidth=1.5, label="data")
model_handle = mpatches.Patch(
    color="#3182bd", alpha=0.5, label="model prediction"
)

# 5. Add the legend to the upper right corner of the plot axis, arranged in two rows (1 column)
for ax in _fig.axes:
    ax.legend(
        handles=[data_handle, model_handle],
        loc="upper right",
        ncol=1,  # Stack vertically for two rows
        frameon=False,
        fontsize=9,
    )

# 6. Optional: Re-run tight_layout to reclaim any whitespace left behind
_fig.fig.tight_layout()

# 7. Save the figure to the central gitignored figure directory in both PDF and PNG formats
fig_dir = get_fig_dir("main")
_fig.fig.savefig(fig_dir / "fig01_ppc_v01.pdf", bbox_inches="tight")

_fig.fig
# %% ---------------------------------------------------------------------------

# 1. Run the PPC plot to get the PlotResult wrapper
_fig = scribe.viz.plot_mixture_ppc_comparison(
    scribe_results,
    adata,
    genes=["RPS4X"],
    n_rows=1,
    figsize=(2.5, 2.5),
    n_samples=512,
)
# 2. Clear the figure-level title (the main title "Example PPC")
_fig.fig.suptitle("")

# 3. Iterate over each panel axis to clear panel titles and y-axis ticks
for ax in _fig.axes:
    # Remove the panel-level title/subtitle (e.g. "RPS4X" and "<U> = ...")
    ax.set_title("")

    # Remove the y-axis ticks and tick labels completely
    ax.set_yticks([])

    # Remove the grid lines
    ax.grid(False)

# 4. Legend colors match plot_mixture_ppc_comparison component colormaps
#    (Greens for component 1, Purples for component 2).
_component_cmaps = ["Greens", "Purples"]
_cell_type_1_handle = mpatches.Patch(
    color=plt.get_cmap(_component_cmaps[0])(0.8),
    alpha=0.4,
    label="cell type 1",
)
_cell_type_2_handle = mpatches.Patch(
    color=plt.get_cmap(_component_cmaps[1])(0.8),
    alpha=0.4,
    label="cell type 2",
)

# 5. Add the legend to the upper right corner of each panel
for ax in _fig.axes:
    ax.legend(
        handles=[_cell_type_1_handle, _cell_type_2_handle],
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=9,
    )

# 6. Re-run tight_layout to reclaim any whitespace left behind
_fig.fig.tight_layout()

# 7. Save the figure to the central gitignored figure directory
fig_dir = get_fig_dir("main")
_fig.fig.savefig(fig_dir / "fig01_ppc_v01_mix.pdf", bbox_inches="tight")

_fig.fig

# %% ---------------------------------------------------------------------------

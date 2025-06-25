# %% ---------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Import project package
import scribe

# %% ---------------------------------------------------------------------------

# Define figure directory
FIG_DIR = f"{scribe.utils.git_root()}/fig/supplementary"

# Create figure directory if it doesn't exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

# %% ---------------------------------------------------------------------------
# Define Gaussian peak function for schematic
# ------------------------------------------------------------------------------


def f(x1, x2):
    """Gaussian peak function"""
    return 10 * np.exp(-(x1**2 + x2**2))


# %% ---------------------------------------------------------------------------
# Evaluate function over 2D grid
# ---------------------------------------------------------------------------


# Define number of points in range
n_grid = 10
n_range = 100

# Set range of values where to evaluate function
x1_grid = x2_grid = np.linspace(-8, 8, n_grid)
x1 = x2 = np.linspace(-8, 8, n_range)

# Create meshgrid for surface plot
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Create meshgrid for wireframe (bottom grid)
X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
Z_floor = np.zeros_like(X1_grid)

# %% ---------------------------------------------------------------------------
# Plot surface
# ---------------------------------------------------------------------------

# Initialize figure
fig = plt.figure(figsize=(4.5, 3))
ax = fig.add_subplot(111, projection="3d")

# Plot surface
surface = ax.plot_surface(
    X1,
    X2,
    Z,
    cmap="magma_r",  # Reverse magma colormap
    alpha=1.0,
    shade=False,
    linewidth=0,
    antialiased=True,
)

# Plot discrete grid on bottom
wireframe = ax.plot_wireframe(
    X1_grid, X2_grid, Z_floor, color="black", linewidth=0.5
)

# Set labels
ax.set_xlabel("θ₁", labelpad=-15)
ax.set_ylabel("θ₂", labelpad=-15)
ax.set_zlabel("π(x̲,θ₁,θ₂)", labelpad=-15)

# Customize appearance to match Julia style and project theme
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Set pane colors to white
ax.xaxis.pane.set_edgecolor("white")
ax.yaxis.pane.set_edgecolor("white")
ax.zaxis.pane.set_edgecolor("white")

# Set pane face colors to white
ax.xaxis.pane.set_facecolor("white")
ax.yaxis.pane.set_facecolor("white")
ax.zaxis.pane.set_facecolor("white")

# Hide tick labels and ticks for cleaner look
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set view angle for better visualization
ax.view_init(elev=20, azim=45)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(
    f"{FIG_DIR}/figSI_gaussian_peak.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    f"{FIG_DIR}/figSI_gaussian_peak.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# %% ---------------------------------------------------------------------------

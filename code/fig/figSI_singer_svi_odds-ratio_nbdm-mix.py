# %% ---------------------------------------------------------------------------
# Import base libraries
import os
import pickle

# Import scribe
import scribe
# Import numpy for array manipulation
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import seaborn for plotting
import seaborn as sns
# Import ArviZ for plotting
import arviz as az
# Import corner
import corner

# Set plotting style
scribe.viz.matplotlib_style()

# Extract colors
colors = scribe.viz.colors()

# %% ---------------------------------------------------------------------------

print("Defining directories...")

# Define data directory
DATA_DIR = f"{scribe.utils.git_root()}/data/singer/"

# Define output directory
OUTPUT_DIR = f"{scribe.utils.git_root()}/output/singer/nbdm_mix"

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

print("Loading SVI results...")

# Define parameterization
parameterization = "odds_ratio"
# Define model type
model_type = "nbdm_mix"
# Define number of components
n_components = 2
# Define number of steps
n_steps = 50_000

# Define output file name
file_name = f"{OUTPUT_DIR}/" \
    f"svi_{parameterization.replace('_', '-')}_" \
    f"{model_type.replace('_', '-')}_" \
    f"{n_components}components_" \
    f"{n_cells}cells_" \
    f"{n_genes}genes_" \
    f"{n_steps}steps.pkl"

# Load MCMC results
with open(file_name, "rb") as f:
    svi_results = pickle.load(f)

# %% ---------------------------------------------------------------------------

print("Plotting loss...")
# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

# Plot loss history
ax.plot(svi_results.loss_history)

# Set y-axis to log scale
# ax.set_yscale('log')

# Set axis labels
ax.set_xlabel('step')
ax.set_ylabel('loss')

# Set title
ax.set_title(f"SVI loss for NBDM-mix model\n with odds-ratio parameterization")

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components}components_loss.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components}components_loss.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting posterior distributions...")

# Initialize figure
fig, axes = plt.subplots(2, 1, figsize=(6, 6))

# Split components
comp1 = svi_results.get_component(0)
comp2 = svi_results.get_component(1)

# Plot posterior for p parameter
scribe.viz.plot_posterior(
    ax=axes[0],
    distribution=comp1.get_distributions(backend="scipy")["phi"],
    color=colors["dark_blue"],
    fill_color=colors["blue"],
)

# Label axes
axes[0].set_xlabel(r"$\theta$")
axes[0].set_ylabel("density")

# Set title
axes[0].set_title(r"$q_\phi(\theta)$")

# Define colors
col = sns.color_palette("colorblind")

# Loop over genes
for i in range(n_genes):
    # Plot posterior for r parameter
    scribe.viz.plot_posterior(
        ax=axes[1],
        distribution=comp1.get_distributions(
            backend="scipy",
            split=True
        )["mu"][i],  # TODO: change to comp2
        color=col[i],
        fill_color=col[i],
    )

# Get the lines from the r parameter distribution plot and create proper legend
lines = axes[1].get_lines()
# Create legend with all 4 lines
axes[1].legend(
    lines,
    ['Rex1', 'Rest', 'Nanog', 'Prdm14'],
    loc='upper right',
)

# Label axes
axes[1].set_xlabel(r"$\mu$")
axes[1].set_ylabel("density")

# Set titlimage.pnge
axes[1].set_title(r"$q_\phi(\mu_1, \ldots, \mu_4)$")

# Set title
plt.tight_layout()

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_"
    f"{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components}components_posterior.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_"
    f"{parameterization.replace('_', '-')}_"
    f"{model_type.replace('_', '-')}_"
    f"{n_components}components_posterior.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Generating predictive samples...")
# Generate predictive samples
svi_results.get_ppc_samples(n_samples=5_000)
# Convert to canonical parameters (i.e. r = mu * theta)
svi_results._convert_to_canonical()

# %% ---------------------------------------------------------------------------

svi_az = az.from_dict(
    posterior={
        "phi": svi_results.posterior_samples["phi"],
        "mu_0": svi_results.posterior_samples["mu"][:, 0],
        "mu_1": svi_results.posterior_samples["mu"][:, 1],
        "mu_2": svi_results.posterior_samples["mu"][:, 2],
        "mu_3": svi_results.posterior_samples["mu"][:, 3],
        "p": svi_results.posterior_samples["p"],
        "r_0": svi_results.posterior_samples["r"][:, 0],
        "r_1": svi_results.posterior_samples["r"][:, 1],
        "r_2": svi_results.posterior_samples["r"][:, 2],
        "r_3": svi_results.posterior_samples["r"][:, 3],
    }
)
# %% ---------------------------------------------------------------------------

print("Plotting corner plot...")

# Define parameter labels
labels = [
    r"$\theta$ (odds-ratio)",
    r"$\mu_1$ (Rex1)",
    r"$\mu_2$ (Rest)",
    r"$\mu_3$ (Nanog)",
    r"$\mu_4$ (Prdm14)",
    r"$p$ (succ. prob.)",
    r"$r_1$ (Rex1)",
    r"$r_2$ (Rest)",
    r"$r_3$ (Nanog)",
    r"$r_4$ (Prdm14)"
]

# Create corner plot
figure = corner.corner(
    svi_az,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 12},
    figsize=(10, 10)
)

# Extract the axes for further customization
axes = np.array(figure.axes).reshape((10, 10))

# Customize tick sizes
for ax in axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)

plt.tight_layout()

# Save the corner plot
figure.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type}_corner.png",
    bbox_inches="tight"
)
figure.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type}_corner.pdf",
    bbox_inches="tight"
)
# %% ---------------------------------------------------------------------------

print("Plotting PPC for 4 genes...")

# Create 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

# Flatten axes for easier iteration
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene from the dataframe
    true_counts = df.iloc[:, i].values

    # Compute credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        svi_results.predictive_samples[:, :, i],
        credible_regions=[95, 68, 50],
        max_bin=true_counts.max()
    )

    # Compute histogram of the real data
    hist_results = np.histogram(
        true_counts,
        bins=credible_regions['bin_edges'],
        density=True
    )

    # Get indices where cumsum <= 0.99
    cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
    # If no indices found (all values > 0.99), use first bin
    max_bin = np.max([
        cumsum_indices[-1] if len(cumsum_indices) > 0 else 0,
        10
    ])

    # Plot credible regions
    scribe.viz.plot_histogram_credible_regions_stairs(
        ax,
        credible_regions,
        cmap='Blues',
        alpha=0.5,
        max_bin=max_bin
    )

    # Define max_bin for histogram
    max_bin_hist = max_bin if len(
        hist_results[0]) > max_bin else len(hist_results[0])
    # Plot histogram of the real data as step plot
    ax.step(
        hist_results[1][:max_bin_hist],
        hist_results[0][:max_bin_hist],
        where='post',
        label='data',
        color='black',
    )

    # Set axis labels
    ax.set_xlabel('counts')
    ax.set_ylabel('frequency')

    # Add legend
    ax.legend()

    # Set title with gene name
    ax.set_title(df.columns[i], fontsize=10)

plt.tight_layout()

# Set global title
fig.suptitle("Posterior Predictive Checks", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type}_ppc.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type}_ppc.pdf",
    bbox_inches="tight"
)

# %% ---------------------------------------------------------------------------

print("Plotting ECDF credible regions for 4 genes...")

# Create 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

# Flatten axes for easier iteration
axes = axes.flatten()

# Loop through each gene
for i, ax in enumerate(axes):
    print(f"Plotting gene {i} PPC...")

    # Extract true counts for this gene from the dataframe
    true_counts = df.iloc[:, i].values
    # Define max bin
    max_bin = true_counts.max()

    # Compute credible regions
    credible_regions = scribe.stats.compute_ecdf_credible_regions(
        svi_results.predictive_samples[:, :, i],
        credible_regions=[95, 68, 50],
        max_bin=max_bin
    )

    # Get x_values - need to add an extra point for stairs function
    x_values = credible_regions['bin_edges']

    # For stairs, we need to add an extra point at the end for proper edges
    x_edges = np.append(x_values[0] - 1, x_values)

    # Plot credible regions
    scribe.viz.plot_ecdf_credible_regions_stairs(
        ax,
        credible_regions,
        cmap='Blues',
        alpha=0.5,
        max_bin=max_bin
    )

    # Compute empirical CDF for the true data
    ecdf_values = np.zeros(len(x_values))
    for j, x in enumerate(x_values):
        ecdf_values[j] = np.mean(true_counts <= x)

    # Plot ECDF of the real data as stairs
    ax.stairs(
        ecdf_values,
        x_edges + 1,  # Use the same extended edges
        label='data',
        color='black',
        linewidth=1.5
    )

    # Set axis labels
    ax.set_xlabel('counts')
    ax.set_ylabel('cumulative probability')

    # Set title with gene name
    ax.set_title(df.columns[i], fontsize=10)
    # Set x-axis limits
    ax.set_xlim(x_values[0] - 1, max_bin - 1)

    # Add legend
    ax.legend()

plt.tight_layout()

# Set global title
fig.suptitle("Posterior Predictive Checks", y=1.02)

# Save figure
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type}_ppc_ecdf.png",
    bbox_inches="tight"
)
fig.savefig(
    f"{FIG_DIR}/figSI_singer_svi_{parameterization.replace('_', '-')}_"
    f"{model_type}_ppc_ecdf.pdf",
    bbox_inches="tight"
)
# %% ---------------------------------------------------------------------------

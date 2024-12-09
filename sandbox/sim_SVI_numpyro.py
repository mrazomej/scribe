# %% ---------------------------------------------------------------------------

# Import JAX-related libraries
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, Trace_ELBO, TraceMeanField_ELBO
# Import numpy for array manipulation
import numpy as np
# Import scipy for statistical functions
import scipy.stats as stats
# Import pandas for data manipulation
import pandas as pd
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
# Import the utils file
# from utils import matplotlib_style
# matplotlib_style()

# %% ---------------------------------------------------------------------------

# Set platform to use CUDA GPU
numpyro.set_platform("gpu")
# %% ---------------------------------------------------------------------------

print("Defining model...")


def model(
    n_cells,
    n_genes,
    p_prior=(1, 1),
    r_prior=(2, 2),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    # Define the prior on the p parameter
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample("r", dist.Gamma(
        r_prior[0],
        r_prior[1]
    ).expand([n_genes])
    )

    # Sum of r parameters
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts
                )

            # Define plate for cells individual counts
            with numpyro.plate("cells", n_cells, dim=-1):
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts),
                    obs=counts
                )
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx]
                )

            # Define plate for cells individual counts
            with numpyro.plate("cells", n_cells, dim=-1) as idx:
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(
                        r, total_count=total_counts[idx]),
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make a NegativeBinomial distribution that returns a vector of
            # length n_genes
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            counts = numpyro.sample("counts", dist_nb)

# %% ---------------------------------------------------------------------------


print("Defining guide...")


def guide(
    n_cells,
    n_genes,
    p_prior=(1, 1),
    r_prior=(2, 2),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    # register alpha_p and beta_p parameters for the Beta distribution in the
    # variational posterior
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(p_prior[0]),
        constraint=constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(p_prior[1]),
        constraint=constraints.positive
    )

    # register one alpha_r and one beta_r parameters for the Gamma distributions
    # for each of the n_genes categories
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones(n_genes) * r_prior[1],
        constraint=constraints.positive
    )

    # Sample from the variational posterior parameters
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))

# %% ---------------------------------------------------------------------------


print("Setting up the simulation...")


# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Define number of cells and genes
n_cells = 1_000
n_genes = 20_000

# Define parameters for prior
r_alpha = 5
r_beta = 0.1
r_prior = (r_alpha, r_beta)

# Split keys for different random operations
key1, key2, key3 = random.split(rng_key, 3)

# Sample true r parameters using JAX's random
r_true = random.gamma(key1, r_alpha, shape=(n_genes,)) / r_beta

# Define prior for p parameter
p_prior = (1, 1)
# Sample true p parameter using JAX's random
p_true = random.beta(key2, p_prior[0], p_prior[1])

# Create negative binomial distribution
nb_dist = dist.NegativeBinomialProbs(r_true, p_true)

# Sample from the distribution
counts_true = nb_dist.sample(key3, sample_shape=(n_cells,))

# %% ---------------------------------------------------------------------------

print("Setting up the optimizer...")

# Set optimizer
optimizer = numpyro.optim.Adam(step_size=0.001)

# Set the inference algorithm
svi = SVI(
    model,
    guide,
    optimizer,
    loss=TraceMeanField_ELBO()
)

# %% ---------------------------------------------------------------------------

print("Running the inference algorithm...")

# Extract counts and total counts
total_counts = counts_true.sum(axis=1)  # Sum counts across genes
n_cells = counts_true.shape[0]
n_genes = counts_true.shape[1]

# Define number of steps
n_steps = 100_000
# Define batch size
batch_size = 512

# Run the inference algorithm
svi_result = svi.run(
    rng_key,
    n_steps,
    n_cells,
    n_genes,
    p_prior=p_prior,
    r_prior=r_prior,
    counts=counts_true,
    total_counts=total_counts,
    batch_size=batch_size
)

# %% ---------------------------------------------------------------------------

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
# Plot the ELBO loss
ax.plot(svi_result.losses)
ax.set_xlabel('iteration')
ax.set_ylabel('ELBO loss')

# Set log scale
ax.set_yscale('log')

plt.tight_layout()

# %% ---------------------------------------------------------------------------

def plot_parameter_posteriors(svi_result, p_true, r_true, n_r_examples=5, n_rows=2, n_cols=3):
    """
    Plot posterior distributions vs ground truth for p and selected r parameters
    
    Args:
        svi_result: SVI results containing alpha and beta parameters
        p_true: True value of p parameter
        r_true: Array of true r parameters
        n_r_examples: Number of r parameters to plot (default 5)
        n_rows, n_cols: Grid dimensions for the subplot
    """
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    # Plot p posterior

    # Extract alpha and beta parameters from svi_result
    alpha_p = svi_result.params['alpha_p']
    beta_p = svi_result.params['beta_p']
    # Define x values for plotting
    x_p = np.linspace(0, 1, 200)
    # Define posterior p
    posterior_p = stats.beta.pdf(x_p, alpha_p, beta_p)
    
    # Plot posterior p
    axes[0].plot(x_p, posterior_p, 'b-', label='Posterior')
    # Plot true p
    axes[0].axvline(p_true, color='r', linestyle='--', label='True Value')
    # Set title
    axes[0].set_title('p parameter')
    # Set x label
    axes[0].set_xlabel('Value')
    # Set y label
    axes[0].set_ylabel('Density')
    # Add legend
    axes[0].legend()
    
    # Extract alpha and beta parameters from svi_result
    alpha_r = svi_result.params['alpha_r']
    beta_r = svi_result.params['beta_r']
    
    # Randomly select r parameters to plot
    r_indices = np.random.choice(len(r_true), size=n_r_examples, replace=False)
    
    # Loop through r parameters
    for i, idx in enumerate(r_indices, 1):
        if i >= len(axes):  # Skip if we run out of subplot space
            break

        # Define x values for plotting
        x_r = np.linspace(0, r_true[idx]*2, 1000)  # Range from 0 to 2x true value
        # Define posterior r
        posterior_r = stats.gamma.pdf(x_r, alpha_r[idx], scale=1/beta_r[idx])
        
        # Plot posterior r
        axes[i].plot(x_r, posterior_r, 'b-', label='posterior')
        # Plot true r
        axes[i].axvline(
            r_true[idx], color='r', linestyle='--', label='ground truth'
        )
        # Set title
        axes[i].set_title(f'r parameter {idx}')
        # Set x label
        axes[i].set_xlabel('parameter value')
        # Set y label
        axes[i].set_ylabel('density')
        # Add legend
        axes[i].legend()
    
    # Remove any unused subplots
    for i in range(n_r_examples + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    return fig

# %% ---------------------------------------------------------------------------

fig = plot_parameter_posteriors(
    svi_result,
    p_true,
    r_true,
    n_r_examples=5,
    n_rows=2,
    n_cols=3
)
plt.show()

# %% ---------------------------------------------------------------------------

print("Defining predictive object...")

# Define number of samples
n_samples = 500

# Define predictive object for posterior samples
predictive_param = Predictive(
    guide,
    params=svi_result.params,
    num_samples=n_samples
)

# Sample from posterior
posterior_param_samples = predictive_param(
    rng_key,
    n_cells,
    n_genes,
    counts=None,
    total_counts=None
)

# use posterior samples to make predictive
predictive = Predictive(
    model,
    posterior_param_samples,
    num_samples=n_samples
)

# Sample from predictive
post_pred = predictive(
    rng_key,
    n_cells,
    n_genes,
    counts=None,
    total_counts=None,
    batch_size=None
)
# %% ---------------------------------------------------------------------------

# Define percentiles for credible regions
percentiles = [95, 68, 50]

# Define array

# %% ---------------------------------------------------------------------------

print("Plotting credible regions...")

# Initialize figure
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# Flatten axes
axes = axes.flatten()

# Compute percentiles for each gene
# First reshape to (n_genes, n_samples, n_cells)
counts_reshaped = np.moveaxis(post_pred["counts"], -1, 0)

# Define percentiles for credible regions
percentiles = [5, 25, 50, 75, 95]

# Loop through each gene
for i, ax in enumerate(axes):
    # Get data for this gene
    gene_samples = counts_reshaped[i]  # shape: (n_samples, n_cells)

    # Sort each sample for ECDF
    gene_samples_sorted = np.sort(gene_samples, axis=1)

    # Compute percentiles across samples
    gene_percentiles = np.percentile(gene_samples_sorted, percentiles, axis=0)

    # Create x values for plotting (using the sorted true data as reference)
    x = np.sort(counts_true[:, i])
    y = np.linspace(0, 1, len(x))

    # Plot credible regions
    ax.fill_between(
        gene_percentiles[0],
        y,
        y,
        color='gray',
        alpha=0.2,
        label='90% CI'
    )
    ax.fill_between(
        gene_percentiles[1],
        y,
        y,
        color='gray',
        alpha=0.3,
        label='50% CI'
    )
    ax.plot(
        gene_percentiles[2],
        y,
        color='gray',
        alpha=0.5,
        label='median'
    )
    ax.fill_between(
        gene_percentiles[3],
        y,
        y,
        color='gray',
        alpha=0.3
    )
    ax.fill_between(
        gene_percentiles[4],
        y,
        y,
        color='gray',
        alpha=0.2
    )

    # Plot ECDF of the real data
    sns.ecdfplot(
        counts_true[:, i],
        ax=ax,
        label='data',
    )

    # Label axis
    ax.set_xlabel('counts')
    ax.set_ylabel('ECDF')
    # Set title
    ax.set_title(f'gene {i}')

    # Add legend to first plot only
    if i == 0:
        ax.legend()

plt.tight_layout()


# %% ---------------------------------------------------------------------------

# Set random seed
rng = np.random.default_rng(42)

# Initialize figure
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# Flatten axes
axes = axes.flatten()

# Loop through each gene
for (i, ax) in enumerate(axes):
    # Loop through samples
    for j in range(n_samples):
        # Plot ECDF of the posterior predictive checks total counts
        sns.ecdfplot(
            post_pred["counts"][j, :, i],
            ax=ax,
            color='gray',
            alpha=0.1
        )

    # Plot ECDF of the real data total counts
    sns.ecdfplot(
        counts_true[:, i],
        ax=ax,

        label='data',
    )
    # Label axis
    ax.set_xlabel('counts')
    ax.set_ylabel('ECDF')
    # Set title
    ax.set_title(f'gene {i}')

plt.tight_layout()

# %%

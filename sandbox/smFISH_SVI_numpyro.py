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
# Import pandas for data manipulation
import pandas as pd
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
# Import the utils file
from utils import matplotlib_style
matplotlib_style()

# %% ---------------------------------------------------------------------------

# Set the number of devices (threads) to use
numpyro.set_host_device_count(4)
# %% ---------------------------------------------------------------------------

# Import tidy dataframe with mRNA counts
df_counts = pd.read_csv('../data/singer_transcript_counts.csv', comment='#')

# %% ---------------------------------------------------------------------------


def smFISH_model(
        n_cells,
        n_genes,
        counts=None,
        total_counts=None,
):
    # Define the prior on the p parameter
    p = numpyro.sample("p", dist.Beta(1, 1))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample("r", dist.Gamma(2, 2).expand([n_genes]))

    # Sum of r parameters
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    # If we have observed data, condition on it
    if counts is not None:
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
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make a NegativeBinomial distribution that returns a vector of
            # length n_genes
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            counts = numpyro.sample("counts", dist_nb)

# %% ---------------------------------------------------------------------------


def guide(
    n_cells,
    n_genes,
    counts=None,
    total_counts=None,
):
    # register alpha_p and beta_p parameters for the Beta distribution in the
    # variational posterior
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(1.0),
        constraint=constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(1.0),
        constraint=constraints.positive
    )

    # register one alpha_r and one beta_r parameters for the Gamma distributions
    # for each of the n_genes categories
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones(n_genes),
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones(n_genes),
        constraint=constraints.positive
    )

    # Sample from the variational posterior parameters
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))

# %% ---------------------------------------------------------------------------


# Set optimizer
optimizer = numpyro.optim.Adam(step_size=0.001)

# Set the inference algorithm
svi = SVI(
    smFISH_model,
    guide,
    optimizer,
    loss=TraceMeanField_ELBO()
)

# %% ---------------------------------------------------------------------------

# Setup the PRNG key
rng_key = random.PRNGKey(42)  # Set random seed

# Extract counts and total counts
counts = df_counts.values  # Transform data to (M, N) shape
total_counts = counts.sum(axis=1)  # Sum counts across genes
n_cells = counts.shape[0]
n_genes = counts.shape[1]

# Define number of steps
n_steps = 100_000

# Run the inference algorithm
svi_result = svi.run(
    rng_key,
    n_steps,
    n_cells,
    n_genes,
    counts,
    total_counts
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
    smFISH_model,
    posterior_param_samples,
    num_samples=n_samples
)

# Sample from predictive
post_pred = predictive(
    rng_key,
    n_cells,
    n_genes,
    counts=None,
    total_counts=None
)
# %% ---------------------------------------------------------------------------

# Set random seed
rng = np.random.default_rng(42)

# Initialize figure
fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

# Loop through samples
for i in range(n_samples):
    # Plot ECDF of the posterior predictive checks total counts
    sns.ecdfplot(
        np.sum(post_pred["counts"][i, :], axis=1),
        ax=ax,
        color='gray',
        alpha=1
    )

# Plot ECDF of the real data total counts
sns.ecdfplot(
    df_counts.sum(axis=1),
    ax=ax,
    label='data',
)

# Label axis
ax.set_xlabel('total counts')
ax.set_ylabel('ECDF')

plt.tight_layout()

# %% ---------------------------------------------------------------------------

# Set random seed
rng = np.random.default_rng(42)

# Initialize figure
fig, axes = plt.subplots(2, 2, figsize=(5, 5))

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
        counts[:, i],
        ax=ax,
        label='data',
    )
    # Label axis
    ax.set_xlabel('counts')
    ax.set_ylabel('ECDF')
    # Set title
    ax.set_title(df_counts.columns[i])

plt.tight_layout()


# %% ---------------------------------------------------------------------------

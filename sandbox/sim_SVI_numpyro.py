# %% ---------------------------------------------------------------------------

# Import JAX-related libraries
from jax import random
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO
# Import numpy for array manipulation
import numpy as np
# Import scipy for statistical functions
import scipy.stats as stats
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Import scribe
import scribe

# Set plotting style
scribe.viz.matplotlib_style()

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
    """
    Numpyro model for Dirichlet-Multinomial single-cell RNA sequencing data.

    This model assumes a hierarchical structure where:
    1. Each cell has a total count drawn from a Negative Binomial distribution
    2. The counts for individual genes are drawn from a Dirichlet-Multinomial
    distribution conditioned on the total count.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float, optional
        Parameters (alpha, beta) for the Beta prior on p parameter.
        Default is (1, 1) for a uniform prior.
    r_prior : tuple of float, optional
        Parameters (shape, rate) for the Gamma prior on r parameters.
        Default is (2, 2).
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    total_counts : array-like, optional
        Total counts per cell of shape (n_cells,).
        Required if counts is provided.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.

    Returns
    -------
    None
        The function defines a probabilistic model but does not return anything.
        Samples are drawn using numpyro's sampling mechanisms.
    """
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
    """
    
    Define the variational distribution for stochastic variational inference.
    
    This guide function specifies the form of the variational distribution that
    will be optimized to approximate the true posterior. It defines a mean-field
    variational family where: - The success probability p follows a Beta
    distribution - Each gene's overdispersion parameter r follows an independent
    Gamma distribution
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float, optional
        Parameters (alpha, beta) for the Beta prior on p (default: (1,1))
    r_prior : tuple of float, optional
        Parameters (alpha, beta) for the Gamma prior on r (default: (2,2))
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    total_counts : array_like, optional
        Total counts per cell of shape (n_cells,)
    batch_size : int, optional
        Mini-batch size for stochastic optimization
        
    Notes
    -----
    The variational parameters (alpha_p, beta_p) for p and (alpha_r, beta_r) for
    each r are initialized using the prior parameters but will be optimized
    during inference.
    """
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
n_genes = 5_000

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

print("Plotting the ELBO loss...")

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

def plot_parameter_posteriors(
    svi_result,
    p_true=None,
    r_true=None,
    n_rows=2,
    n_cols=3,
    n_points=200,
    figsize=None
):
    """
    Plot posterior distributions versus ground truth values for model
    parameters.

    This function creates a grid of plots showing: 1. The Beta posterior
    distribution for the p parameter (success probability) 2. Multiple Gamma
    posterior distributions for randomly selected r parameters
       (overdispersion parameters)

    For each parameter, the posterior distribution is plotted along with a
    vertical line indicating the true value (if provided).

    Parameters
    ----------
    svi_result : numpyro.infer.SVI
        Results from stochastic variational inference containing the optimized
        variational parameters (alpha and beta) for both p and r distributions.
    p_true : float, optional
        True value of the p parameter (success probability). If provided, will
        be shown as a vertical line on the plot.
    r_true : array-like, optional
        Array of true values for the r parameters (overdispersion). If provided,
        will be shown as vertical lines on the respective plots.
    n_rows : int, default=2
        Number of rows in the subplot grid.
    n_cols : int, default=3
        Number of columns in the subplot grid.
    n_points : int, default=200
        Number of points to use when plotting the posterior distributions.
        Higher values give smoother curves but increase computation time.
    figsize : tuple of float, optional
        Figure dimensions (width, height) in inches. If None, defaults to
        (4*n_cols, 4*n_rows).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the grid of posterior plots.

    Notes
    -----
    - The first subplot always shows the p parameter posterior
    - Subsequent subplots show randomly selected r parameter posteriors
    - The posterior distributions are colored blue
    - Ground truth values (if provided) are shown as red dashed lines
    """
    # Set figure size if not provided
    if figsize is None:
        figsize = (4*n_cols, 4*n_rows)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot p posterior

    # Extract alpha and beta parameters from svi_result
    alpha_p = svi_result.params['alpha_p']
    beta_p = svi_result.params['beta_p']
    # Define x values for plotting
    x_p = np.linspace(0, 1, n_points)
    # Define posterior p
    posterior_p = stats.beta.pdf(x_p, alpha_p, beta_p)
    
    # Plot posterior p
    axes[0].plot(x_p, posterior_p, 'b-', label='posterior')
    if p_true is not None:  
        # Plot true p
        axes[0].axvline(p_true, color='r', linestyle='--', label='ground truth')
    # Set title
    axes[0].set_title('p parameter')
    # Set x label
    axes[0].set_xlabel('value')
    # Set y label
    axes[0].set_ylabel('density')
    # Add legend
    axes[0].legend()
    
    # Extract alpha and beta parameters from svi_result
    alpha_r = svi_result.params['alpha_r']
    beta_r = svi_result.params['beta_r']
    
    # Calculate number of r examples needed (total plots minus p plot)
    n_r_examples = n_rows * n_cols - 1
    
    # Randomly select r parameters to plot
    r_indices = np.random.choice(len(r_true), size=n_r_examples, replace=False)
    
    # Loop through r parameters
    for i, idx in enumerate(r_indices, 1):
        # Define x range based on the posterior mean if r_true not provided
        r_mean = alpha_r[idx] / beta_r[idx]
        x_max = r_true[idx]*2 if r_true is not None else r_mean*2
        x_r = np.linspace(0, x_max, n_points)

        # Define posterior r
        posterior_r = stats.gamma.pdf(x_r, alpha_r[idx], scale=1/beta_r[idx])
        
        # Plot posterior r
        axes[i].plot(x_r, posterior_r, 'b-', label='posterior')
        if r_true is not None:
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

    plt.tight_layout()

    return fig
# %% ---------------------------------------------------------------------------


print("Plotting parameter posteriors...")

# Plot parameter posteriors
fig = plot_parameter_posteriors(
    svi_result,
    p_true,
    r_true,
    n_rows=3,
    n_cols=3
)

# %% ---------------------------------------------------------------------------

print("Defining predictive object...")

# Define number of samples
n_samples = 100

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

def compute_histogram_percentiles(
    samples,
    percentiles=[5, 25, 50, 75, 95],
    normalize=True
):
    """
    Compute percentiles of histogram frequencies across multiple samples.
    
    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) where each row is a sample
    percentiles : list-like, optional
        List of percentiles to compute (default: [5, 25, 50, 75, 95])
    normalize : bool, optional
        Whether to normalize histograms (default: True)
        
    Returns
    -------
    bin_edges : array
        Array of bin edges (integers from min to max value + 1)
    hist_percentiles : array
        Array of shape (len(percentiles), len(bin_edges)-1) containing
        the percentiles of histogram frequencies for each bin
    """
    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())
    
    # Create bin edges (integers from min to max + 1)
    bin_edges = np.arange(global_min, global_max + 2)  # +2 because we want right edge
    
    # Initialize array to store histograms
    n_samples = samples.shape[0]
    n_bins = len(bin_edges) - 1
    all_hists = np.zeros((n_samples, n_bins))
    
    # Compute histogram for each sample
    for i in range(n_samples):
        hist, _ = np.histogram(samples[i], bins=bin_edges)
        if normalize:
            hist = hist / hist.sum()
        all_hists[i] = hist
    
    # Compute percentiles across samples for each bin
    hist_percentiles = np.percentile(all_hists, percentiles, axis=0)
    
    return bin_edges, hist_percentiles

# %% ---------------------------------------------------------------------------

def compute_histogram_credible_regions(
    samples,
    credible_regions=[95, 68, 50],
    normalize=True
):
    """
    Compute credible regions of histogram frequencies across multiple samples.
    
    Parameters
    ----------
    samples : array-like
        Array of shape (n_samples, n_points) where each row is a sample
    credible_regions : list-like, optional
        List of credible region percentages to compute (default: [95, 68, 50])
        For example, 95 will compute the 2.5 and 97.5 percentiles
    normalize : bool, optional
        Whether to normalize histograms (default: True)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'bin_edges': array of bin edges
        - 'regions': nested dictionary where each key is the credible region percentage
          and values are dictionaries containing:
            - 'lower': lower bound of the credible region
            - 'upper': upper bound of the credible region
            - 'median': median (50th percentile)
    """
    # Find global min and max across all samples
    global_min = int(samples.min())
    global_max = int(samples.max())
    
    # Create bin edges (integers from min to max + 1)
    bin_edges = np.arange(global_min, global_max + 2)  # +2 because we want right edge
    
    # Initialize array to store histograms
    n_samples = samples.shape[0]
    n_bins = len(bin_edges) - 1
    all_hists = np.zeros((n_samples, n_bins))
    
    # Compute histogram for each sample
    for i in range(n_samples):
        hist, _ = np.histogram(samples[i], bins=bin_edges)
        if normalize:
            hist = hist / hist.sum()
        all_hists[i] = hist
    
    # Compute credible regions
    results = {
        'bin_edges': bin_edges,
        'regions': {}
    }
    
    # Always compute median
    median = np.percentile(all_hists, 50, axis=0)
    
    # Loop through credible regions
    for cr in credible_regions:
        # Compute lower and upper percentiles
        lower_percentile = (100 - cr) / 2
        upper_percentile = 100 - lower_percentile
        
        # Store results
        results['regions'][cr] = {
            'lower': np.percentile(all_hists, lower_percentile, axis=0),
            'upper': np.percentile(all_hists, upper_percentile, axis=0),
            'median': median
        }
    
    return results

# %% ---------------------------------------------------------------------------

def plot_credible_regions(
    ax,
    hist_results,
    colors=None,
    cmap=None,
    alpha=0.2,
    plot_median=True,
    median_color='black',
    median_alpha=0.2,
    median_linewidth=1.5,
    label_prefix='',
):
    """
    Plot credible regions as fill_between on a given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    hist_results : dict
        Results dictionary from compute_histogram_credible_regions
    colors : list, optional
        List of colors for each credible region. Must match length of 
        credible regions. If None and cmap is None, defaults to grays.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to generate colors from. Ignored if colors is provided.
    alpha : float or list, optional
        Transparency for fill_between plots. If float, same alpha used for all
        regions. If list, must match length of credible regions.
    plot_median : bool, optional
        Whether to plot the median line (default: True)
    median_color : str, optional
        Color for median line (default: 'black')
    median_alpha : float, optional
        Transparency for median line (default: 0.8)
    median_linewidth : float, optional
        Line width for median line (default: 1.5)
    label_prefix : str, optional
        Prefix for legend labels (default: '')
        
    Returns
    -------
    matplotlib.axes.Axes
        The axis with plots added
    """
    bin_edges = hist_results['bin_edges']
    x = bin_edges[:-1]  # Use left edges for plotting
    
    # Sort credible regions from largest to smallest for proper layering
    cr_values = sorted(hist_results['regions'].keys(), reverse=True)
    n_regions = len(cr_values)
    
    # Handle colors
    if colors is None:
        if cmap is None:
            # Default to grays if no colors specified
            colors = [f'gray' for _ in range(n_regions)][::-1]
        else:
            # Generate colors from colormap
            if isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            colors = [cmap(i / (n_regions - 1)) for i in range(n_regions)][::-1]
    
    # Handle alpha
    if isinstance(alpha, (int, float)):
        alphas = [alpha] * n_regions
    else:
        alphas = alpha
        
    # Plot credible regions
    for cr, color, alpha in zip(cr_values, colors, alphas):
        region = hist_results['regions'][cr]
        ax.fill_between(
            x,
            region['lower'],
            region['upper'],
            color=color,
            alpha=alpha,
            label=f'{label_prefix}{cr}% CR'
        )
    
    # Plot median
    if plot_median:
        # Use the median from any region (they're all the same)
        median = hist_results['regions'][cr_values[0]]['median']
        ax.plot(
            x,
            median,
            color=median_color,
            alpha=median_alpha,
            linewidth=median_linewidth,
            label=f'{label_prefix}median'
        )
    
    return ax

# %% ---------------------------------------------------------------------------

# Compute credible regions for first gene
hist_results = compute_histogram_credible_regions(post_pred["counts"][:, :, 0])

# Compute histogram for first gene data
hist_results_data = np.histogram(
    counts_true[:, 0],
    bins=np.arange(np.min(counts_true[:, 0]), np.max(counts_true[:, 0]) + 2),
    density=True
)

# Single plot example
fig, ax = plt.subplots()

# Plot credible regions
plot_credible_regions(
    ax, 
    hist_results,
    cmap='Blues',
    alpha=0.5
)

# Plot data histogram as step plot
ax.step(
    hist_results_data[1][:-1], # Remove last bin edge
    hist_results_data[0],
    where='post',
    label='data',
    color='black',
)

ax.set_xlabel('counts')
ax.set_ylabel('frequency')

# # Multiple subplots example
# fig, axes = plt.subplots(2, 2)
# for ax in axes.flat:
#     plot_credible_regions(
#         ax,
#         hist_results,
#         colors=['blue', 'red', 'green'],
#         alpha=0.2
#     )

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

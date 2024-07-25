# %% ---------------------------------------------------------------------------
from scipy.stats import dirichlet
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from functools import partial
import bayesflow as bf
import pandas as pd
from utils import matplotlib_style
cor, pal = matplotlib_style()

# %% ---------------------------------------------------------------------------

# Define number of genes
n_genes = 100

# Define range of number of cells
n_cells = [100, 1000]

# %% ---------------------------------------------------------------------------

# Set the random number generator to be used
RNG = np.random.default_rng(42)


def prior_fn():
    """
    Generates prior parameter samples for the scRNA-seq model

    This function generates prior distributions for the parameters `p_hat` and
    `r_vec`. The `p_hat` parameter is drawn from a Beta(1, 1) distribution,
    representing the probability of success in a Bernoulli trial. The `r_vec`
    parameter is a vector of shape parameters for each gene, drawn from a
    log-normal distribution with mean -1 and standard deviation 1, and then
    sorted in ascending order.

    Returns:
        np.ndarray: A numpy array containing `p_hat` and `r_vec`. The first
        element is `p_hat`, a float representing the probability of success. The
        second element is `r_vec`, a sorted numpy array of shape parameters for
        each gene.

    Raises:
        NameError: If `RNG` or `n_genes` is not defined in the scope where this
        function is called.

    Example:
        >>> p_hat, r_vec = prior_fn()
        >>> print(p_hat)
        0.5
        >>> print(r_vec)
        [0.1, 0.2, 0.3, ...]

    Note:
        Ensure that `RNG` is an instance of a random number generator (e.g.,
        `numpy.random.default_rng()`) and `n_genes` is defined in the scope
        where this function is called.
    """
    # Prior on p_hat for negative binomial parameter
    p_hat = RNG.beta(1, 10)
    # Prior on r parameters for each gene
    r_vec = RNG.lognormal(-1, 0.5, n_genes)
    # Sort r_vec
    r_vec = np.sort(r_vec)[::-1]

    # Return the prior
    return np.float32(np.concatenate(([p_hat], r_vec)))

# %% ---------------------------------------------------------------------------


# Define Prior simulator function for BayesFlow
param_names = [r"$\hat{p}$"] + [f"$r_{{{i+1}}}$" for i in range(n_genes)]

prior = bf.simulation.Prior(
    prior_fun=prior_fn,
    param_names=param_names,
)

# Draw samples from the prior
prior(10)

# %% ---------------------------------------------------------------------------


def likelihood_fn(params, n_obs=100):
    """
    Computes the likelihood of the parameters given the Dirichlet-Multinomial
    model.

    This function simulates the observed counts from a Dirichlet-Multinomial
    distribution using the provided parameters. The Dirichlet-Multinomial
    distribution is a compound distribution where the probabilities of the
    Multinomial distribution are drawn from a Dirichlet distribution.

    Args:
        params (np.ndarray): Array containing the parameters. The first element
        is `p_hat`, which is the parameter for the Negative Binomial
        distribution. The remaining elements are `r_vec`, which are the
        parameters for the Dirichlet distribution.  n_obs (int, optional): The
        number of observations to generate. Defaults to 1.

    Returns:
        np.ndarray: A sample from the Dirichlet-Multinomial distribution. The
        shape of the returned array is (n_obs, len(r_vec)) if n_obs > 1,
        otherwise (len(r_vec),).

    Example:
        >>> params = np.array([0.5, 1.0, 2.0, 3.0])
        >>> likelihood_fn(params, n_obs=5)
        array([[10., 20., 30.],
               [12., 18., 25.],
               [11., 19., 29.],
               [13., 21., 31.],
               [14., 22., 32.]], dtype=float32)

    Notes:
        - The first element of `params` is `p_hat`, which is used to sample the
          total number of counts `U` from a Negative Binomial distribution.
        - The remaining elements of `params` form `r_vec`, which are used as
          parameters for the Dirichlet distribution to sample the probabilities
          for the Multinomial distribution.
        - The function returns the counts sampled from the Multinomial
          distribution as a `float32` numpy array.
    """
    # Unpack parameters
    p_hat = params[0]
    r_vec = params[1:]
    # Compute sum of r parameters
    r_o = np.sum(r_vec)
    # Sample from Negative Binomial distribution the total number of UMI counts
    U = RNG.negative_binomial(r_o, p_hat)
    # Sample probabilities from Dirichlet distribution
    dirichlet_probs = RNG.dirichlet(r_vec)
    # Sample counts from Multinomial distribution using the Dirichlet probabilities
    u_vec = RNG.multinomial(U, dirichlet_probs, size=n_obs)

    return np.float32(u_vec)

# %% ---------------------------------------------------------------------------


# Define Likelihood simulator function for BayesFlow
simulator = bf.simulation.Simulator(simulator_fun=likelihood_fn)

# Build generative model
model = bf.simulation.GenerativeModel(prior, simulator)
# %% ---------------------------------------------------------------------------

# Draw samples from the generative model
model_draws = model(500)
# %% ---------------------------------------------------------------------------

# Plot the distribution of prior draws for p_hat and a few r parameters

# Define number of rows and columns
rows = 3
cols = 3

# Initialize figure
fig, ax = plt.subplots(rows, cols, figsize=(4, 4))

# Plot prior draws for p_hat on first plot
ax[0, 0].hist(model_draws["prior_draws"][:, 0], bins=20, color=pal[0])
# Set x-axis label
ax[0, 0].set_xlabel(param_names[0])

# Select random indexes for r parameters
r_idx = np.sort(
    RNG.choice(np.arange(1, n_genes), size=(rows * cols) - 1, replace=False)
)

# Loop through the rest of the parameters
for i in range(rows * cols):
    # Calculate row and column index
    row = i // cols
    col = i % cols
    print(row, col)
    # Skip first plot
    if i == 0:
        continue

    # Plot histogram of prior draws for r parameters
    ax[row, col].hist(
        model_draws["prior_draws"][:, r_idx[i-1]], bins=20, color=pal[0]
    )
    # Set x-axis label
    ax[row, col].set_xlabel(param_names[r_idx[i-1]])

fig.tight_layout()
# %% ---------------------------------------------------------------------------

# Set up summary network

# Define summary network as a Deepset
summary_net = bf.networks.DeepSet(summary_dim=125)

# Simulate a pass through the summary network
summary_pass = summary_net(model_draws["sim_data"])

# %% ---------------------------------------------------------------------------

# Define the conditional invertible network
inference_net = bf.inference_networks.InvertibleNetwork(
    num_params=prior(1)["prior_draws"].shape[-1],
)

# Perform a forward pass through the network given the summary network embedding
z, log_det_J = inference_net(model_draws['prior_draws'], summary_pass)

print(f"Latent variables: {z.numpy()}")
print(f"Log Jacobian Determinant: {log_det_J.numpy()}")
# %% ---------------------------------------------------------------------------

# Assemble the amoratizer that combines the summary network and inference
# network
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

# Assemble the trainer with the amortizer and generative model
trainer = bf.trainers.Trainer(amortizer=amortizer, generative_model=model)
# %% ---------------------------------------------------------------------------

history = trainer.train_online(
    epochs=5,
    iterations_per_epoch=1000,
    batch_size=32,
    validation_sims=200
)
# %%

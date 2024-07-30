# %% ---------------------------------------------------------------------------
from utils import matplotlib_style
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import bayesflow as bf
import tensorflow as tf
import tensorflow_probability as tfp

cor, pal = matplotlib_style()

# %% ---------------------------------------------------------------------------

# Define number of genes
n_genes = 5_000

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

    Note:
        Ensure that `RNG` is an instance of a random number generator (e.g.,
        `numpy.random.default_rng()`) and `n_genes` is defined in the scope
        where this function is called.
    """
    # Prior on p_hat for negative binomial parameter
    p_hat = RNG.beta(1, 10)
    # Prior on r parameters for each gene
    r_vec = RNG.gamma(3, 1/5, n_genes)
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


def likelihood_fn(params, n_obs=10_000):
    # Unpack parameters
    p_hat = params[0]
    r_vec = params[1:]

    # Sample from individual negative binomial distributions for each element
    # in r_vec
    u_vec = np.transpose(
        np.array([RNG.negative_binomial(r, p_hat, n_obs) for r in r_vec])
    )

    return u_vec.astype(np.float32)
# %% ---------------------------------------------------------------------------

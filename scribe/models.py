"""
Models for single-cell RNA sequencing data.
"""

# Import JAX-related libraries
import jax.numpy as jnp
# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Callable, Dict, Tuple

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------

def nbdm_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (0, 1),
    r_prior: tuple = (1, 2),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Numpyro model for Dirichlet-Multinomial single-cell RNA sequencing data.

    This model assumes a hierarchical structure where:
    0. Each cell has a total count drawn from a Negative Binomial distribution
    1. The counts for individual genes are drawn from a Dirichlet-Multinomial
       distribution conditioned on the total count.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p parameter.
        Default is (0, 1) for a uniform prior.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters.
        Default is (1, 2).
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    total_counts : array-like, optional
        Total counts per cell of shape (n_cells,).
        Required if counts is provided.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.
    """
    # Define the prior on the p parameter
    p = numpyro.sample("p", dist.Beta(p_prior[-1], p_prior[1]))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample("r", dist.Gamma(
        r_prior[-1],
        r_prior[0]
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
            with numpyro.plate("cells", n_cells, dim=-2):
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
            with numpyro.plate("cells", n_cells, dim=-2) as idx:
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
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(0)
            counts = numpyro.sample("counts", dist_nb)


# ------------------------------------------------------------------------------
# Beta-Gamma Variational Posterior
# ------------------------------------------------------------------------------

def nbdm_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (0, 1),
    r_prior: tuple = (1, 2),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for stochastic variational inference.
    
    This guide function specifies the form of the variational distribution that
    will be optimized to approximate the true posterior. It defines a mean-field
    variational family where:
    - The success probability p follows a Beta distribution
    - Each gene's overdispersion parameter r follows an independent Gamma
    distribution
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p (default: (0,1))
    r_prior : tuple of float
        Parameters (alpha, beta) for the Gamma prior on r (default: (1,2))
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    total_counts : array_like, optional
        Total counts per cell of shape (n_cells,)
    batch_size : int, optional
        Mini-batch size for stochastic optimization
    """
    # register alpha_p and beta_p parameters for the Beta distribution in the
    # variational posterior
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(p_prior[-1]),
        constraint=constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(p_prior[0]),
        constraint=constraints.positive
    )

    # register one alpha_r and one beta_r parameters for the Gamma distributions
    # for each of the n_genes categories
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones(n_genes) * r_prior[-1],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=constraints.positive
    )

    # Sample from the variational posterior parameters
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------

def zinb_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 2),
    gate_prior: tuple = (1, 1),
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Zero-Inflated Negative Binomial single-cell RNA sequencing
    data. Uses a single shared p parameter across all genes.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on shared p parameter
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters
    gate_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic variational inference
    """
    # Single shared p parameter for all genes
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))
    
    # Sample r parameters for all genes simultaneously
    r = numpyro.sample(
        "r",
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_genes])
    )
    
    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate",
        dist.Beta(gate_prior[0], gate_prior[1]).expand([n_genes])
    )

    # Create base negative binomial distribution
    base_dist = dist.NegativeBinomialProbs(r, p)
    
    # Create zero-inflated distribution
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells
            with numpyro.plate("cells", n_cells):
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # Define plate for cells
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make the distribution return a vector of length n_genes
            dist_zinb = zinb.to_event(0)
            counts = numpyro.sample("counts", dist_zinb)

# ------------------------------------------------------------------------------
# Beta-Gamma-Beta Variational Posterior
# ------------------------------------------------------------------------------

def zinb_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 2),
    gate_prior: tuple = (1, 1),
    counts=None,
    batch_size=None,
):
    """
    Variational distribution for the Zero-Inflated Negative Binomial model.
    """
    # Variational parameters for shared p
    alpha_p = numpyro.param(
        "alpha_p",
        p_prior[0],
        constraint=constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        p_prior[1],
        constraint=constraints.positive
    )
    
    # Variational parameters for r (one per gene)
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
    
    # Variational parameters for gate (one per gene)
    alpha_gate = numpyro.param(
        "alpha_gate",
        jnp.ones(n_genes) * gate_prior[0],
        constraint=constraints.positive
    )
    beta_gate = numpyro.param(
        "beta_gate",
        jnp.ones(n_genes) * gate_prior[1],
        constraint=constraints.positive
    )

    # Sample from variational posterior parameters
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))
    numpyro.sample("gate", dist.Beta(alpha_gate, beta_gate))

# ------------------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------------------

def get_model_and_guide(model_type: str) -> Tuple[Callable, Callable]:
    """
    Get model and guide functions for a specified model type.

    This function returns the appropriate model and guide functions based on the
    requested model type. Currently supports:
    - "nbdm": Negative Binomial-Dirichlet Multinomial model
    - "zinb": Zero-Inflated Negative Binomial model

    Parameters
    ----------
    model_type : str
        The type of model to retrieve functions for. Must be one of ["nbdm", "zinb"].

    Returns
    -------
    Tuple[Callable, Callable]
        A tuple containing (model_function, guide_function) for the requested model type.

    Raises
    ------
    ValueError
        If an unsupported model type is provided.
    """
    # Handle Negative Binomial-Dirichlet Multinomial model
    if model_type == "nbdm":
        # Import model and guide functions locally to avoid circular imports
        from .models import nbdm_model, nbdm_guide
        return nbdm_model, nbdm_guide
    
    # Handle Zero-Inflated Negative Binomial model
    elif model_type == "zinb":
        # Import model and guide functions locally to avoid circular imports
        from .models import zinb_model, zinb_guide
        return zinb_model, zinb_guide
    
    # Raise error for unsupported model types
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ------------------------------------------------------------------------------
# Model default priors
# ------------------------------------------------------------------------------

def get_default_priors(model_type: str) -> Dict[str, Tuple[float, float]]:
    """
    Get default prior parameters for a specified model type.

    This function returns a dictionary of default prior parameters based on the
    requested model type. Currently supports:
    - "nbdm": Negative Binomial-Dirichlet Multinomial model
    - "zinb": Zero-Inflated Negative Binomial model

    Parameters
    ----------
    model_type : str
        The type of model to get default priors for. Must be one of ["nbdm",
        "zinb"]. For custom models, returns an empty dictionary.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A dictionary mapping parameter names to prior parameter tuples:
        - For "nbdm":
            - 'p_prior': (alpha, beta) for Beta prior on p parameter
            - 'r_prior': (shape, rate) for Gamma prior on r parameter
        - For "zinb":
            - 'p_prior': (alpha, beta) for Beta prior on p parameter  
            - 'r_prior': (shape, rate) for Gamma prior on r parameter
            - 'gate_prior': (alpha, beta) for Beta prior on gate parameter
        - For custom models: empty dictionary
    """
    if model_type == "nbdm":
        prior_params = {
            'p_prior': (1, 1),
            'r_prior': (2, 0.1)
        }
    elif model_type == "zinb":
        prior_params = {
            'p_prior': (1, 1),
            'r_prior': (2, 0.1),
            'gate_prior': (1, 1)
        }
    else:
        prior_params = {}  # Empty dict for custom models if none provided

    return prior_params
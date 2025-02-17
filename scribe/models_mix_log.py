"""
Mixture model implementations for single-cell RNA sequencing data with
log-normal priors and variational distributions
"""

import jax.numpy as jnp
import jax.scipy as jsp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Callable, Dict, Tuple, Optional, Union

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

def nbdm_log_mixture_model(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Numpyro mixture model for single-cell RNA sequencing data using Negative
    Binomial distributions with shared p parameter per component.
    
    This model assumes a hierarchical mixture structure where:
        1. Each mixture component has:
           - A shared success probability p across all genes
           - Gene-specific dispersion parameters r
        2. The mixture is handled using Numpyro's MixtureSameFamily
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    n_components : int
        Number of mixture components to fit
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p parameters.
        Default is (1, 1) for uniform priors.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters.
        Default is (2, 0.1).
    mixing_prior : Union[float, tuple]
        Concentration parameter for the Dirichlet prior on mixing weights.
        If float, uses same value for all components.
        If tuple, uses different concentration for each component.
        Default is 1.0 for a uniform prior over the simplex.
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - Mixture weights ~ Dirichlet(mixing_prior)
        - Success probability p ~ Beta(p_prior)
        - Component-specific dispersion r ~ Gamma(r_prior) per gene and component

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), NegativeBinomialProbs(r, p)
    )
    """
    # Check if mixing_prior is a tuple
    if isinstance(mixing_prior, tuple):
        if len(mixing_prior) != n_components:
            raise ValueError(
                f"Length of mixing_prior ({len(mixing_prior)}) must match "
                f"number of components ({n_components})"
            )
        mixing_concentration = jnp.array(mixing_prior)
    else:
        mixing_concentration = jnp.ones(n_components) * mixing_prior

    # Sample mixing weights from Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights",
        dist.Dirichlet(mixing_concentration)
    )
    
    # Create mixing distribution
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Define the prior on the p parameters - one for each component
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        dist.LogNormal(r_prior[0], r_prior[1]).expand([n_components, n_genes])
    )

    # Create base negative binomial distribution
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
    
    # Create mixture distribution
    mixture = dist.MixtureSameFamily(
        mixing_dist, 
        base_dist
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells
            with numpyro.plate("cells", n_cells):
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts)
        else:
            # Mini-batch version
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample counts from mixture
            numpyro.sample("counts", mixture)


# ------------------------------------------------------------------------------
# Variational Guide for Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

def nbdm_log_mixture_guide(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the NBDM mixture model. This guide function defines
    the form of the variational distribution that will be optimized to
    approximate the true posterior.

    This guide function specifies a mean-field variational family where:
        - The success probability p follows a Beta distribution
        - Each gene's overdispersion parameter r follows an independent Gamma
          distribution
        - The mixing weights follow a Dirichlet distribution
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    n_components : int
        Number of mixture components
    p_prior : tuple of float
        Parameters (alpha, beta) for Beta prior on p (default: (1,1))
    r_prior : tuple of float
        Parameters (alpha, beta) for Gamma prior on r (default: (2,0.1))
    mixing_prior : Union[float, tuple]
        Concentration parameter(s) for Dirichlet prior on mixing weights. If
        float, uses same value for all components. If tuple, uses different
        concentration for each component.
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic optimization
    """
    # Check if mixing_prior is a tuple
    if isinstance(mixing_prior, tuple):
        mixing_concentration = jnp.array(mixing_prior)
    else:
        mixing_concentration = jnp.ones(n_components) * mixing_prior

    # Variational parameters for mixing weights
    alpha_mixing = numpyro.param(
        "alpha_mixing",
        mixing_concentration,
        constraint=constraints.positive
    )

    # Variational parameters for p
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

    # Variational parameters for r (one per component and gene)
    mu_r = numpyro.param(
        "mu_r",
        jnp.ones((n_components, n_genes)) * r_prior[0],
        constraint=constraints.real
    )
    sigma_r = numpyro.param(
        "sigma_r",
        jnp.ones((n_components, n_genes)) * r_prior[1],
        constraint=constraints.positive
    )

    # Sample from variational distributions
    numpyro.sample("mixing_weights", dist.Dirichlet(alpha_mixing))
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.LogNormal(mu_r, sigma_r))
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
from typing import Callable, Dict, Tuple, Optional

# Import mixture models
from .models_mix import *

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------

def nbdm_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
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
        Default is (1, 1) for a uniform prior.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters.
        Default is (2, 0.1).
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
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample("r", dist.Gamma(r_prior[0], r_prior[1]).expand([n_genes]))

    # Sum of r parameters
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells, dim=-2):
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
                subsample_size=batch_size,
                dim=-2
            ) as idx:
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx]
                )

            # Define plate for cells individual counts
            with numpyro.plate(
                "cells",
                n_cells,
                dim=-2,
                subsample_size=batch_size
            ) as idx:
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


# ------------------------------------------------------------------------------
# Beta-Gamma Variational Posterior for Negative Binomial-Dirichlet Multinomial
# ------------------------------------------------------------------------------

def nbdm_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
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
        "r", dist.Gamma(r_prior[0], r_prior[1]).expand([n_genes]))
    
    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate", dist.Beta(gate_prior[0], gate_prior[1]).expand([n_genes]))

    # Create base negative binomial distribution
    base_dist = dist.NegativeBinomialProbs(r, p)
    
    # Create zero-inflated distribution
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells
            with numpyro.plate("cells", n_cells, dim=-2):
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # Define plate for cells
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
                dim=-2
            ) as idx:
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make the distribution return a vector of length n_genes
            counts = numpyro.sample("counts", zinb.to_event(1))

# ------------------------------------------------------------------------------
# Beta-Gamma-Beta Variational Posterior for Zero-Inflated Negative Binomial
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
    Define the variational distribution for stochastic variational inference of
    the Zero-Inflated Negative Binomial model.
    
    This guide function specifies the form of the variational distribution that
    will be optimized to approximate the true posterior. It defines a mean-field
    variational family where:
        - The success probability p follows a Beta distribution
        - Each gene's overdispersion parameter r follows an independent Gamma
          distribution
        - Each gene's dropout probability (gate) follows an independent Beta
          distribution
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p (default: (1,1))
    r_prior : tuple of float
        Parameters (alpha, beta) for the Gamma prior on r (default: (2,2))
    gate_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on gate (default: (1,1))
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic variational inference
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
# Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

def nbvcp_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Negative Binomial with variable mRNA capture probability.

    This model assumes that each gene's mRNA count follow a Negative Binomial
    distribution, but with a cell-specific mRNA capture probability that
    modifies the success probability parameter. The model structure is:
        1. Each gene has a base success probability p and dispersion r
        2. Each cell has a capture probability p_capture
        3. The effective success probability for each gene in each cell is
           computed as p_hat = p / (p_capture + p * (1 - p_capture)). This comes
           from the composition of a negative binomial distribution with a
           binomial distribution.
        4. Counts are drawn from NB(r, p_hat)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on base success probability
        p. Default is (1, 1) for a uniform prior.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on dispersion parameters.
        Default is (2, 0.1).
    p_capture_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on capture probabilities.
        Default is (1, 1) for a uniform prior.
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.
    """
    # Define global parameters
    # Sample base success probability
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Sample gene-specific dispersion parameters
    r = numpyro.sample(
        "r", 
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_genes])
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_prior[0], p_capture_prior[1])
                )

                # Reshape p_capture for broadcasting and compute effective
                # probability given shared p
                p_capture_reshaped = p_capture[:, None]
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
                )

                # Condition on observed counts
                numpyro.sample(
                    "counts", 
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1), 
                    obs=counts
                )
        else:
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_prior[0], p_capture_prior[1])
                )

                # Reshape p_capture for broadcasting and compute effective probability
                p_capture_reshaped = p_capture[:, None]
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
                )

                # Condition on observed counts
                numpyro.sample(
                    "counts", 
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1), 
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probabilities
            p_capture = numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_prior[0], p_capture_prior[1])
            )

            # Reshape p_capture for broadcasting and compute effective probability
            p_capture_reshaped = p_capture[:, None]
            p_hat = numpyro.deterministic(
                "p_hat",
                p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
            )

            # Sample counts
            numpyro.sample(
                "counts",
                dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            )

# ------------------------------------------------------------------------------
# Beta-Gamma-Beta Variational Posterior for Negative Binomial with variable
# capture probability
# ------------------------------------------------------------------------------

def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    counts=None,
    batch_size=None,
):
    """
    Variational guide (approximate posterior) for the Negative Binomial model
    with variable capture probability.

    This guide specifies a factorized variational distribution that approximates
    the true posterior:
        1. A Beta distribution for the global success probability p
        2. Independent Gamma distributions for each gene's dispersion parameter
           r
        3. Independent Beta distributions for each cell's capture probability
           p_capture

    The variational parameters are:
        - alpha_p, beta_p: Parameters for p's Beta distribution
        - alpha_r, beta_r: Parameters for each gene's Gamma distribution (shape:
          n_genes)
        - alpha_p_capture, beta_p_capture: Parameters for each cell's Beta
          distribution (shape: n_cells)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float, default=(1, 1)
        Parameters (alpha, beta) for initializing the Beta variational
        distribution of the global success probability p
    r_prior : tuple of float, default=(2, 0.1)
        Parameters (shape, rate) for initializing the Gamma variational
        distributions of gene-specific dispersion parameters
    p_capture_prior : tuple of float, default=(1, 1)
        Parameters (alpha, beta) for initializing the Beta variational
        distributions of cell-specific capture probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration
    """
    # Variational parameters for base success probability p (global)
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

    # Variational parameters for r (global per gene)
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

    # Sample global parameters outside the plate
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))

    # Initialize p_capture parameters for all cells
    alpha_p_capture = numpyro.param(
        "alpha_p_capture",
        jnp.ones(n_cells) * p_capture_prior[0],
        constraint=constraints.positive
    )
    beta_p_capture = numpyro.param(
        "beta_p_capture",
        jnp.ones(n_cells) * p_capture_prior[1],
        constraint=constraints.positive
    )

    # Use plate for handling local parameters (p_capture)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture",
                dist.Beta(alpha_p_capture, beta_p_capture)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(alpha_p_capture[idx], beta_p_capture[idx])
            )

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

def zinbvcp_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    gate_prior: tuple = (1, 1),  # Added for zero-inflation
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Zero-Inflated Negative Binomial with variable mRNA capture
    probability.

    This model combines the zero-inflation mechanism with variable capture
    probability. The model structure is:
        1. Each gene has a base success probability p and dispersion r
        2. Each cell has a capture probability p_capture
        3. Each gene has a dropout probability (gate)
        4. The effective success probability for each gene in each cell is
           computed as p_hat = p / (p_capture + p * (1 - p_capture)). This comes
           from the composition of a negative binomial distribution with a
           binomial distribution.  
        5. Counts are drawn from ZINB(r, p_hat, gate)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on base success probability
        p
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on dispersion parameters
    p_capture_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on capture probabilities
    gate_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic variational inference
    """
    # Define global parameters
    # Sample base success probability
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Sample gene-specific dispersion parameters
    r = numpyro.sample(
        "r", 
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_genes])
    )

    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate", 
        dist.Beta(gate_prior[0], gate_prior[1]).expand([n_genes])
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_prior[0], p_capture_prior[1])
                )

                # Reshape p_capture for broadcasting and compute effective probability
                p_capture_reshaped = p_capture[:, None]
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
                )

                # Create base negative binomial distribution with adjusted probabilities
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts)
        else:
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_prior[0], p_capture_prior[1])
                )

                # Reshape p_capture for broadcasting and compute effective probability
                p_capture_reshaped = p_capture[:, None]
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
                )

                # Create base negative binomial distribution with adjusted probabilities
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probabilities
            p_capture = numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_prior[0], p_capture_prior[1])
            )

            # Reshape p_capture for broadcasting and compute effective
            # probability
            p_capture_reshaped = p_capture[:, None]
            p_hat = numpyro.deterministic(
                "p_hat",
                p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
            )

            # Create base negative binomial distribution with adjusted
            # probabilities
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            # Sample counts
            numpyro.sample("counts", zinb)

# ------------------------------------------------------------------------------
# Beta-Gamma-Beta Variational Posterior for Zero-Inflated Negative Binomial with
# variable capture probability
# ------------------------------------------------------------------------------

def zinbvcp_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    gate_prior: tuple = (1, 1),
    counts=None,
    batch_size=None,
):
    """
    Variational guide (approximate posterior) for the Zero-Inflated Negative
    Binomial model with variable capture probability.

    This guide specifies a factorized variational distribution that approximates
    the true posterior:
        1. A Beta distribution for the global success probability p
        2. Independent Gamma distributions for each gene's dispersion parameter
           r
        3. Independent Beta distributions for each gene's dropout probability
           (gate)
        4. Independent Beta distributions for each cell's capture probability
           p_capture

    The variational parameters are:
        - alpha_p, beta_p: Parameters for p's Beta distribution
        - alpha_r, beta_r: Parameters for each gene's Gamma distribution (shape:
          n_genes)
        - alpha_gate, beta_gate: Parameters for each gene's Beta distribution
          (shape: n_genes)
        - alpha_p_capture, beta_p_capture: Parameters for each cell's Beta
          distribution (shape: n_cells)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float, default=(1, 1)
        Parameters (alpha, beta) for initializing the Beta variational
        distribution of the global success probability p
    r_prior : tuple of float, default=(2, 0.1)
        Parameters (shape, rate) for initializing the Gamma variational
        distributions of gene-specific dispersion parameters
    p_capture_prior : tuple of float, default=(1, 1)
        Parameters (alpha, beta) for initializing the Beta variational
        distributions of cell-specific capture probabilities
    gate_prior : tuple of float, default=(1, 1)
        Parameters (alpha, beta) for initializing the Beta variational
        distributions of gene-specific dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration
    """
    # Variational parameters for base success probability p
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

    # Sample global parameters outside the plate
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))
    numpyro.sample("gate", dist.Beta(alpha_gate, beta_gate))

    # Initialize p_capture parameters for all cells
    alpha_p_capture = numpyro.param(
        "alpha_p_capture",
        jnp.ones(n_cells) * p_capture_prior[0],
        constraint=constraints.positive
    )
    beta_p_capture = numpyro.param(
        "beta_p_capture",
        jnp.ones(n_cells) * p_capture_prior[1],
        constraint=constraints.positive
    )

    # Use plate for handling local parameters (p_capture)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture",
                dist.Beta(alpha_p_capture, beta_p_capture)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(alpha_p_capture[idx], beta_p_capture[idx])
            )

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
        - "nbvcp": Negative Binomial with variable mRNA capture probability
        - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
          probability

    Parameters
    ----------
    model_type : str
        The type of model to retrieve functions for. Must be one of ["nbdm",
        "zinb", "nbvcp", "zinbvcp"].

    Returns
    -------
    Tuple[Callable, Callable]
        A tuple containing (model_function, guide_function) for the requested
        model type.

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
    
    # Handle Negative Binomial with variable mRNA capture probability model
    elif model_type == "nbvcp":
        # Import model and guide functions locally to avoid circular imports
        from .models import nbvcp_model, nbvcp_guide
        return nbvcp_model, nbvcp_guide
    
    # Handle Zero-Inflated Negative Binomial with variable capture probability
    elif model_type == "zinbvcp":
        # Import model and guide functions locally to avoid circular imports
        from .models import zinbvcp_model, zinbvcp_guide
        return zinbvcp_model, zinbvcp_guide
    
    # Handle Negative Binomial-Dirichlet Multinomial Mixture Model
    elif model_type == "nbdm_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import nbdm_mixture_model, nbdm_mixture_guide
        return nbdm_mixture_model, nbdm_mixture_guide

    # Handle Zero-Inflated Negative Binomial Mixture Model
    elif model_type == "zinb_mix":
        # Import model and guide functions locally to avoid circular imports
        from .models_mix import zinb_mixture_model, zinb_mixture_guide
        return zinb_mixture_model, zinb_mixture_guide
    
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
        - "nbvcp": Negative Binomial with variable mRNA capture probability
          model
        - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
          probability model

    Parameters
    ----------
    model_type : str
        The type of model to get default priors for. Must be one of ["nbdm",
        "zinb", "nbvcp", "zinbvcp"]. For custom models, returns an empty
        dictionary.

    Returns
    -------
    Dict[str, Tuple[float, float]]
        A dictionary mapping parameter names to prior parameter tuples: - For
        "nbdm":
            - 'p_prior': (alpha, beta) for Beta prior on p parameter
            - 'r_prior': (shape, rate) for Gamma prior on r parameter
        - For "zinb":
            - 'p_prior': (alpha, beta) for Beta prior on p parameter  
            - 'r_prior': (shape, rate) for Gamma prior on r parameter
            - 'gate_prior': (alpha, beta) for Beta prior on gate parameter
        - For "nbvcp":
            - 'p_prior': (alpha, beta) for Beta prior on base success
              probability p
            - 'r_prior': (shape, rate) for Gamma prior on dispersion parameters
            - 'p_capture_prior': (alpha, beta) for Beta prior on capture
              probabilities
        - For "zinbvcp":
            - 'p_prior': (alpha, beta) for Beta prior on base success
              probability p
            - 'r_prior': (shape, rate) for Gamma prior on dispersion parameters
            - 'p_capture_prior': (alpha, beta) for Beta prior on capture
              probabilities
            - 'gate_prior': (alpha, beta) for Beta prior on dropout
              probabilities
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
    elif model_type == "nbvcp":
        prior_params = {
            'p_prior': (1, 1),
            'r_prior': (2, 0.1),
            'p_capture_prior': (1, 1)
        }
    elif model_type == "zinbvcp":
        prior_params = {
            'p_prior': (1, 1),
            'r_prior': (2, 0.1),
            'p_capture_prior': (1, 1),
            'gate_prior': (1, 1)
        }
    elif model_type == "nbdm_mix":
        prior_params = {
            'mixing_weights_prior': (1, 1),
            'p_prior': (1, 1),
            'r_prior': (2, 0.1)
        }
    else:
        prior_params = {}  # Empty dict for custom models if none provided

    return prior_params

# ------------------------------------------------------------------------------
# Log Likelihood functions
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial (NBDM) likelihood
# ------------------------------------------------------------------------------

def nbdm_log_likelihood(
    counts: jnp.ndarray, 
    params: Dict, 
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for NBDM model using plates.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters: 
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities using the NBDM model (default)
            - 'gene': returns log probabilities using independent NB per gene
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    # Check return_by
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params['p'])
    r = jnp.squeeze(params['r'])
    r_total = jnp.sum(r)
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    if return_by == 'cell':
        # Compute total counts for each cell
        total_counts = jnp.sum(counts, axis=1)
        
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Compute log probability for total counts
            log_prob_total = dist.NegativeBinomialProbs(
                r_total, p).log_prob(total_counts)
            # Compute log probability for each gene
            log_prob_genes = dist.DirichletMultinomial(
                r, total_count=total_counts).log_prob(counts)
            # Return sum of log probabilities
            return log_prob_total + log_prob_genes
        
        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)

        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_total_counts = total_counts[start_idx:end_idx]
            
            # Compute log probability for total counts
            batch_log_prob_total = dist.NegativeBinomialProbs(
                r_total, p).log_prob(batch_total_counts)
            # Compute log probability for each gene
            batch_log_prob_genes = dist.DirichletMultinomial(r, total_count=batch_total_counts).log_prob(batch_counts)
            
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                batch_log_prob_total + batch_log_prob_genes
            )
        
        return cell_log_probs
    
    else:  # return_by == 'gene'
        # For per-gene likelihood, use independent negative binomials
        if batch_size is None:
            # Compute log probabilities for all genes at once
            return jnp.sum(
                dist.NegativeBinomialProbs(r, p).log_prob(counts),
                axis=0  # Sum over cells
            )
        
        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            
            # Create NB distribution and compute log probs for each gene
            nb_dist = dist.NegativeBinomialProbs(r, p)
            # Shape of batch_counts is (batch_size, n_genes)
            # We want log probs for each gene summed over the batch
            batch_log_probs = nb_dist.log_prob(batch_counts)  # Shape: (batch_size, n_genes)
            
            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)
        
        return gene_log_probs

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial (ZINB) likelihood
# ------------------------------------------------------------------------------

def zinb_log_likelihood(
    counts: jnp.ndarray, 
    params: Dict, 
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for Zero-Inflated Negative Binomial model.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene
            - 'gate': dropout probability parameter
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params['p'])
    r = jnp.squeeze(params['r'])
    gate = jnp.squeeze(params['gate'])
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)
       
    if return_by == 'cell':
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            # Return per-cell log probabilities
            return zinb.log_prob(counts)
        
        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                zinb.log_prob(batch_counts)
            )
        
        return cell_log_probs
    
    else:  # return_by == 'gene'
        # For per-gene likelihood
        if batch_size is None:
            # Create base distribution and compute all at once
            base_dist = dist.NegativeBinomialProbs(r, p)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            return jnp.sum(zinb.log_prob(counts), axis=0)
        
        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            # Get indices for current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            
            # Create distributions and compute log probs
            base_dist = dist.NegativeBinomialProbs(r, p)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            # Shape: (batch_size, n_genes)
            batch_log_probs = zinb.log_prob(batch_counts)  
            
            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)
        
        return gene_log_probs

# ------------------------------------------------------------------------------
# Negative Binomial with Variable Capture Probability (NBVC) likelihood
# ------------------------------------------------------------------------------

def nbvcp_log_likelihood(
    counts: jnp.ndarray, 
    params: Dict, 
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for Negative Binomial with Variable Capture Probability.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene
            - 'p_capture': cell-specific capture probabilities
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    # Check return_by
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params['p'])
    r = params['r']
    p_capture = params['p_capture']
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    if return_by == 'cell':
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Reshape p_capture to [n_cells, 1] for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute p_hat for all cells
            p_hat = p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
            # Return per-cell log probabilities
            return dist.NegativeBinomialProbs(
                r, p_hat).to_event(1).log_prob(counts)
        
        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]
            
            # Reshape batch_p_capture for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute p_hat for batch
            batch_p_hat = p / (batch_p_capture + p * (1 - batch_p_capture))
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                dist.NegativeBinomialProbs(
                    r, batch_p_hat).to_event(1).log_prob(batch_counts)
            )
        
        return cell_log_probs
    
    else:  # return_by == 'gene'
        # For per-gene likelihood
        if batch_size is None:
            # Reshape p_capture to [n_cells, 1] for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute p_hat for all cells
            p_hat = p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
            # Compute log probabilities for each gene
            return jnp.sum(
                dist.NegativeBinomialProbs(r, p_hat).log_prob(counts),
                axis=0  # Sum over cells
            )
        
        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]
            
            # Reshape batch_p_capture for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute p_hat for batch
            batch_p_hat = p / (batch_p_capture + p * (1 - batch_p_capture))
            # Compute log probabilities for batch
            batch_log_probs = dist.NegativeBinomialProbs(
                r, batch_p_hat).log_prob(batch_counts)
            
            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)
        
        return gene_log_probs

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVC)
# ------------------------------------------------------------------------------

def zinbvcp_log_likelihood(
    counts: jnp.ndarray, 
    params: Dict, 
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for Zero-Inflated Negative Binomial with Variable
    Capture Probability.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene
            - 'p_capture': cell-specific capture probabilities
            - 'gate': dropout probability parameter
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Array of log likelihood values. Shape depends on return_by:
            - 'cell': shape (n_cells,)
            - 'gene': shape (n_genes,)
    """
    # Check return_by
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters from dictionary
    p = jnp.squeeze(params['p'])
    r = jnp.squeeze(params['r'])
    p_capture = jnp.squeeze(params['p_capture'])
    gate = jnp.squeeze(params['gate'])
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)
       
    if return_by == 'cell':
        # If no batch size provided, process all cells at once
        if batch_size is None:
            # Reshape capture probabilities for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute adjusted success probability
            p_hat = p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            # Return per-cell log probabilities
            return zinb.log_prob(counts)
        
        # Initialize array to store per-cell log probabilities
        cell_log_probs = jnp.zeros(n_cells)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]
            
            # Reshape capture probabilities for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute adjusted success probability
            batch_p_hat = p / (batch_p_capture + p * (1 - batch_p_capture))
            # Create base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, batch_p_hat)
            # Create Zero-Inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                zinb.log_prob(batch_counts)
            )
        
        return cell_log_probs
    
    else:  # return_by == 'gene'
        # For per-gene likelihood
        if batch_size is None:
            # Reshape capture probabilities for broadcasting
            p_capture_reshaped = p_capture[:, None]
            # Compute adjusted success probability
            p_hat = p / (p_capture_reshaped + p * (1 - p_capture_reshaped))
            # Create base distribution and compute all at once
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            return jnp.sum(zinb.log_prob(counts), axis=0)
        
        # Initialize array to store per-gene log probabilities
        gene_log_probs = jnp.zeros(n_genes)
        
        # Process in batches
        for i in range((n_cells + batch_size - 1) // batch_size):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_cells)
            
            # Get batch data
            batch_counts = counts[start_idx:end_idx]
            batch_p_capture = p_capture[start_idx:end_idx]
            
            # Reshape capture probabilities for broadcasting
            batch_p_capture = batch_p_capture[:, None]
            # Compute adjusted success probability
            batch_p_hat = p / (batch_p_capture + p * (1 - batch_p_capture))
            # Create distributions and compute log probs
            base_dist = dist.NegativeBinomialProbs(r, batch_p_hat)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)
            batch_log_probs = zinb.log_prob(batch_counts)
            
            # Add the batch contribution to the running total
            gene_log_probs += jnp.sum(batch_log_probs, axis=0)
        
        return gene_log_probs

# ------------------------------------------------------------------------------
# Log Likelihood registry
# ------------------------------------------------------------------------------

def get_log_likelihood_fn(model_type: str) -> Callable:
    """
    Get the log likelihood function for a specified model type.
    
    Parameters
    ----------
    model_type : str
        Type of model to get likelihood function for. Must be one of:
            - 'nbdm': Negative Binomial-Dirichlet Multinomial
            - 'zinb': Zero-Inflated Negative Binomial
            - 'nbvcp': Negative Binomial with Variable Capture Probability
            - 'zinbvcp': Zero-Inflated Negative Binomial with Variable Capture
              Probability
        
    Returns
    -------
    Callable
        Log likelihood function for the specified model type. The function takes
        observed counts and model parameters as input and returns an array of
        log likelihood values.
        
    Raises
    ------
    KeyError
        If model_type is not one of the supported types.
    """
    return {
        "nbdm": nbdm_log_likelihood,
        "zinb": zinb_log_likelihood,
        "nbvcp": nbvcp_log_likelihood,
        "zinbvcp": zinbvcp_log_likelihood
    }[model_type]
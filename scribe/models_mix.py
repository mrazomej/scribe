"""
Mixture model implementations for single-cell RNA sequencing data.
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

def nbdm_mixture_model(
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
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_components, n_genes])
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

def nbdm_mixture_guide(
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
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones((n_components, n_genes)) * r_prior[0],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones((n_components, n_genes)) * r_prior[1],
        constraint=constraints.positive
    )

    # Sample from variational distributions
    numpyro.sample("mixing_weights", dist.Dirichlet(alpha_mixing))
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_model(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    gate_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Numpyro mixture model for single-cell RNA sequencing data using
    Zero-Inflated Negative Binomial distributions.
    
    This model assumes a hierarchical mixture structure where:
        1. Each mixture component has: - A shared success probability p across
           all genes - Gene-specific dispersion parameters r - Gene-specific
           dropout probabilities (gate)
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
        Parameters (alpha, beta) for the Beta prior on p parameters. Default is
        (1, 1) for uniform priors.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters. Default is
        (2, 0.1).
    gate_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on dropout probabilities.
        Default is (1, 1).
    mixing_prior : Union[float, tuple]
        Concentration parameter for the Dirichlet prior on mixing weights. If
        float, uses same value for all components. If tuple, uses different
        concentration for each component. Default is 1.0 for a uniform prior
        over the simplex.
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Mixture weights ~ Dirichlet(mixing_prior)
        - Success probability p ~ Beta(p_prior) [shared across components]
        - Component-specific dispersion r ~ Gamma(r_prior) [per component and gene]
        - Dropout probabilities gate ~ Beta(gate_prior) [per component and gene]

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), ZeroInflatedNegativeBinomial(r, p, gate)
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

    # Define the prior on the p parameters
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_components, n_genes])
    )

    # Define the prior on the gate parameters - one for each gene and component
    gate = numpyro.sample(
        "gate",
        dist.Beta(gate_prior[0], gate_prior[1]).expand([n_components, n_genes])
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Create base negative binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            
            # Create zero-inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, 
                gate=gate
            ).to_event(1)
            
            # Create mixture distribution
            mixture = dist.MixtureSameFamily(
                mixing_dist, 
                zinb
            )
            
            # Define plate for cells
            with numpyro.plate("cells", n_cells):
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts)
        else:
            # Mini-batch version
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p)
                
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                
                # Create mixture distribution
                mixture = dist.MixtureSameFamily(mixing_dist, zinb)
                
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Create base negative binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            
            # Create zero-inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            
            # Create mixture distribution
            mixture = dist.MixtureSameFamily(mixing_dist, zinb)
            
            # Sample counts from mixture
            numpyro.sample("counts", mixture)


# ------------------------------------------------------------------------------
# Variational Guide for Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    gate_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the ZINB mixture model. This guide function defines
    the form of the variational distribution that will be optimized to
    approximate the true posterior.

    This guide function specifies a mean-field variational family where:
        - The mixing weights follow a Dirichlet distribution
        - Each component's success probability p follows a Beta distribution
        - Each gene's overdispersion parameter r follows an independent Gamma
          distribution for each component
        - Each gene's dropout probability follows a Beta distribution for each
          component
    
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
    gate_prior : tuple of float
        Parameters (alpha, beta) for Beta prior on dropout probability
        (default: (1,1))
    mixing_prior : Union[float, tuple]
        Concentration parameter(s) for Dirichlet prior on mixing weights
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

    # Variational parameters for p (one per component)
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
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones((n_components, n_genes)) * r_prior[0],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones((n_components, n_genes)) * r_prior[1],
        constraint=constraints.positive
    )

    # Variational parameters for gate (one per component and gene)
    alpha_gate = numpyro.param(
        "alpha_gate",
        jnp.ones((n_components, n_genes)) * gate_prior[0],
        constraint=constraints.positive
    )
    beta_gate = numpyro.param(
        "beta_gate",
        jnp.ones((n_components, n_genes)) * gate_prior[1],
        constraint=constraints.positive
    )

    # Sample from variational distributions
    numpyro.sample("mixing_weights", dist.Dirichlet(alpha_mixing))
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))
    numpyro.sample("gate", dist.Beta(alpha_gate, beta_gate))

# ------------------------------------------------------------------------------
# Negative Binomial Mixture Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Numpyro mixture model for single-cell RNA sequencing data using Negative
    Binomial distributions with variable capture probability.
    
    This model assumes a hierarchical mixture structure where:
        1. Each mixture component has:
           - A shared success probability p across all genes
           - Gene-specific dispersion parameters r
        2. Each cell has:
           - A cell-specific capture probability p_capture (independent of components)
        3. The effective success probability for each gene in each cell is
           computed as p_hat = p / (p_capture + p * (1 - p_capture))
        4. The mixture is handled using Numpyro's MixtureSameFamily
    
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
    p_capture_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on capture probabilities.
        Default is (1, 1).
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
        - Component-specific dispersion r ~ Gamma(r_prior) per gene

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~ Beta(p_capture_prior)
        - Effective probability p_hat = p * p_capture / (1 - p * (1 - p_capture))

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), NegativeBinomial(r, p_hat)
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
    p = numpyro.sample(
        "p",
        dist.Beta(p_prior[0], p_prior[1])
    )

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_components, n_genes])
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

                # Reshape p_capture for broadcasting with components
                p_capture_reshaped = p_capture[:, None, None]  # [cells, 1, 1]
                
                # Compute effective probability for each component
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )

                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                
                # Create mixture distribution
                mixture = dist.MixtureSameFamily(mixing_dist, base_dist)
                
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts)
        else:
            # Mini-batch version
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_prior[0], p_capture_prior[1])
                )

                # Reshape p_capture for broadcasting with components
                p_capture_reshaped = p_capture[:, None, None]  # [cells, 1, 1]
                
                # Compute effective probability for each component
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )

                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                
                # Create mixture distribution
                mixture = dist.MixtureSameFamily(mixing_dist, base_dist)
                
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probabilities
            p_capture = numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_prior[0], p_capture_prior[1])
            )

            # Reshape p_capture for broadcasting with components
            p_capture_reshaped = p_capture[:, None, None]  # [cells, 1, 1]
            
            # Compute effective probability for each component
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            )

            # Create base negative binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            
            # Create mixture distribution
            mixture = dist.MixtureSameFamily(mixing_dist, base_dist)
            
            # Sample counts from mixture
            numpyro.sample("counts", mixture)

# ------------------------------------------------------------------------------
# Variational Guide for Negative Binomial Mixture Model with Variable Capture
# Probability
# ------------------------------------------------------------------------------

def nbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the NBVCP mixture model. This guide function defines
    the form of the variational distribution that will be optimized to
    approximate the true posterior.

    This guide function specifies a mean-field variational family where:
        - The mixing weights follow a Dirichlet distribution
        - Each component's success probability p follows a Beta distribution
        - Each gene's overdispersion parameter r follows an independent Gamma
          distribution for each component
        - Each cell's capture probability follows an independent Beta
          distribution
    
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
    p_capture_prior : tuple of float
        Parameters (alpha, beta) for Beta prior on capture probabilities
        (default: (1,1))
    mixing_prior : Union[float, tuple]
        Concentration parameter(s) for Dirichlet prior on mixing weights
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

    # Variational parameters for p (one per component)
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
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones((n_components, n_genes)) * r_prior[0],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones((n_components, n_genes)) * r_prior[1],
        constraint=constraints.positive
    )

    # Sample global parameters outside the plate
    numpyro.sample("mixing_weights", dist.Dirichlet(alpha_mixing))
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
# Zero-Inflated Negative Binomial Mixture Model with Variable Capture
# Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    gate_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Zero-Inflated Negative Binomial Mixture Model with Variable Capture
    Probability for single-cell RNA sequencing data.

    This model captures key characteristics of scRNA-seq data including:
        - Technical dropouts via zero-inflation
        - Cell-specific capture efficiencies
        - Overdispersion via negative binomial distribution
        - Heterogeneous cell populations via mixture components

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int 
        Number of genes in the dataset
    n_components : int
        Number of mixture components
    p_prior : tuple, default=(1, 1)
        Beta prior parameters (alpha, beta) for success probability p
    r_prior : tuple, default=(2, 0.1)
        Gamma prior parameters (shape, rate) for dispersion r
    p_capture_prior : tuple, default=(1, 1)
        Beta prior parameters for cell-specific capture probabilities
    gate_prior : tuple, default=(1, 1)
        Beta prior parameters for gene-specific dropout probabilities
    mixing_prior : float or tuple, default=1.0
        Dirichlet prior concentration parameter(s) for mixture weights
    counts : array_like, optional
        Observed count matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic inference. If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - Mixture weights ~ Dirichlet(mixing_prior)
        - Success probability p ~ Beta(p_prior)
        - Component-specific dispersion r ~ Gamma(r_prior) per gene
        - Dropout probabilities gate ~ Beta(gate_prior) per gene

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~ Beta(p_capture_prior)
        - Effective probability p_hat = p * p_capture / (1 - p * (1 - p_capture))

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), ZeroInflatedNegativeBinomial(r, p_hat,
        gate)
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

    # Define the prior on the p parameter (shared across components)
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        dist.Gamma(r_prior[0], r_prior[1]).expand([n_components, n_genes])
    )
    
    # Define the prior on the gate parameters - one for each gene
    gate = numpyro.sample(
        "gate",
        dist.Beta(gate_prior[0], gate_prior[1]).expand([n_components, n_genes])
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

                # Reshape p_capture for broadcasting with components
                p_capture_reshaped = p_capture[:, None, None]  # [cells, 1, 1]
                
                # Compute effective probability for each component
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )

                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p)
                
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                
                # Create mixture distribution
                mixture = dist.MixtureSameFamily(mixing_dist, zinb)
                
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_prior[0], p_capture_prior[1])
                )

                # Reshape p_capture for broadcasting with components
                p_capture_reshaped = p_capture[:, None, None]  # [cells, 1, 1]
                
                # Compute effective probability for each component
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )

                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p)
                
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                
                # Create mixture distribution
                mixture = dist.MixtureSameFamily(mixing_dist, zinb)
                
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probabilities
            p_capture = numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_prior[0], p_capture_prior[1])
            )

            # Reshape p_capture for broadcasting with components
            p_capture_reshaped = p_capture[:, None, None]  # [cells, 1, 1]
            
            # Compute effective probability for each component
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            )

            # Create base negative binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p)
            
            # Create zero-inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            
            # Create mixture distribution
            mixture = dist.MixtureSameFamily(mixing_dist, zinb)
            
            # Sample counts from mixture
            numpyro.sample("counts", mixture)

# ------------------------------------------------------------------------------
# Variational Guide for Zero-Inflated Negative Binomial Mixture Model with
# Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    n_components: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    p_capture_prior: tuple = (1, 1),
    gate_prior: tuple = (1, 1),
    mixing_prior: Union[float, tuple] = 1.0,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the Zero-Inflated Negative Binomial mixture model with
    variable capture probability.

    This guide specifies a mean-field variational family for approximating the
    posterior distribution of model parameters. The variational distributions
    are:

    Global Parameters:
        - Mixing weights ~ Dirichlet(alpha_mixing)
        - Success probability p ~ Beta(alpha_p, beta_p) [shared across
          components]
        - Component-specific dispersion r ~ Gamma(alpha_r, beta_r) [per
          component and gene]
        - Dropout probabilities gate ~ Beta(alpha_gate, beta_gate) [per gene]

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~ Beta(alpha_p_capture,
          beta_p_capture) [per cell]

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    n_components : int
        Number of mixture components
    p_prior : tuple, default=(1, 1)
        Beta prior parameters (alpha, beta) for success probability p
    r_prior : tuple, default=(2, 0.1)
        Gamma prior parameters (shape, rate) for dispersion r
    p_capture_prior : tuple, default=(1, 1)
        Beta prior parameters for cell-specific capture probabilities
    gate_prior : tuple, default=(1, 1)
        Beta prior parameters for gene-specific dropout probabilities
    mixing_prior : float or tuple, default=1.0
        Dirichlet prior concentration parameter(s) for mixture weights
    counts : array_like, optional
        Observed count matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.
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

    # Variational parameters for p (shared)
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
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones((n_components, n_genes)) * r_prior[0],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones((n_components, n_genes)) * r_prior[1],
        constraint=constraints.positive
    )
    # Variational parameters for gate (one per component and gene)
    alpha_gate = numpyro.param(
        "alpha_gate",
        jnp.ones((n_components, n_genes)) * gate_prior[0],
        constraint=constraints.positive
    )
    beta_gate = numpyro.param(
        "beta_gate",
        jnp.ones((n_components, n_genes)) * gate_prior[1],
        constraint=constraints.positive
    )

    # Sample global parameters outside the plate
    numpyro.sample("mixing_weights", dist.Dirichlet(alpha_mixing))
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
# Log Likelihood functions
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Negative Binomial Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

def nbdm_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    split_components: bool = False,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for NBDM mixture model using independent negative
    binomials.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene and component
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component. If
        False, returns the log probability of the mixture.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells, n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes, n_components)
    """
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, 1, n_genes)
    counts = counts[:, None, :]
    # r: (n_components, n_genes) -> (1, n_components, n_genes)
    r = r[None, :, :]
    # p: scalar -> (1, n_components, 1) for broadcasting
    p = jnp.array(p)[None, None, None]  # First convert scalar to array, then add dimensions

    # Create base NB distribution vectorized over cells, components, genes
    # r: (1, n_components, n_genes)
    # p: (1, n_components, 1) or scalar
    # counts: (n_cells, 1, n_genes)
    # This will broadcast to: (n_cells, n_components, n_genes)
    nb_dist = dist.NegativeBinomialProbs(r, p)

    if return_by == 'cell':
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=-1) + jnp.log(mixing_weights)
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=-1) + jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            gene_log_probs = nb_dist.log_prob(counts)
            # Sum over cells and add mixing weights
            log_probs = (jnp.sum(gene_log_probs, axis=0).T + 
                         jnp.log(mixing_weights))  # (n_genes, n_components)
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(
                    counts[start_idx:end_idx, None, :]
                )  # (batch_size, n_components, n_genes)

                # Sum over batch
                log_probs += jnp.sum(batch_log_probs, axis=0).T
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights)
            
    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    split_components: bool = False,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for ZINB mixture model.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': success probability parameter
            - 'r': dispersion parameters for each gene and component
            - 'gate': dropout probabilities for each gene and component
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component.
        If False, returns the log probability of the mixture.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells, n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes, n_components)
    """
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    gate = jnp.squeeze(params['gate']).astype(dtype)  # shape (n_components, n_genes)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, 1, n_genes)
    counts = counts[:, None, :]
    # r: (n_components, n_genes) -> (1, n_components, n_genes)
    r = r[None, :, :]
    # gate: (n_components, n_genes) -> (1, n_components, n_genes)
    gate = gate[None, :, :]
    # p: scalar -> (1, n_components, 1) for broadcasting
    p = jnp.array(p)[None, None, None]  # First convert scalar to array, then add dimensions

    # Create base NB distribution vectorized over cells, components, genes
    # r: (1, n_components, n_genes)
    # p: (1, n_components, 1) or scalar
    # counts: (n_cells, 1, n_genes)
    # This will broadcast to: (n_cells, n_components, n_genes)
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Create zero-inflated distribution for each component
    # This will broadcast to: (n_cells, n_components, n_genes)
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)

    if return_by == 'cell':
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=-1) + jnp.log(mixing_weights)
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=-1) + jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)
            # Sum over cells and add mixing weights
            log_probs = (jnp.sum(gene_log_probs, axis=0).T + 
                         jnp.log(mixing_weights))  # (n_genes, n_components)
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                batch_log_probs = zinb.log_prob(
                    counts[start_idx:end_idx, None, :]
                )  # (batch_size, n_components, n_genes)

                # Sum over batch
                log_probs += jnp.sum(batch_log_probs, axis=0).T
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights)
            
    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)

# ------------------------------------------------------------------------------
# Negative Binomial Mixture Model with Capture Probabilities
# ------------------------------------------------------------------------------

def nbvcp_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    split_components: bool = False,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for NBVCP mixture model.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene and component
            - 'p_capture': cell-specific capture probabilities
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component. If
        False, returns the log probability of the mixture.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells,
              n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes,
              n_components)
    """
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    p_capture = jnp.squeeze(params['p_capture']).astype(dtype)  # shape (n_cells,)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, 1, n_genes)
    counts = counts[:, None, :]
    # r: (n_components, n_genes) -> (1, n_components, n_genes)
    r = r[None, :, :]
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture_reshaped = p_capture[:, None, None]
    # p: scalar -> (1, 1, 1) for broadcasting
    p = jnp.array(p)[None, None, None]
    # Compute effective probability for each cell
    # This will broadcast to shape (n_cells, 1, 1)
    p_hat = p / (p_capture_reshaped + p * (1 - p_capture_reshaped))

    if return_by == 'cell':
        if batch_size is None:
            # Create base NB distribution vectorized over cells, components, genes
            # r: (1, n_components, n_genes)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, 1, n_genes)
                # This will broadcast to: (n_cells, n_components, n_genes)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)

            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=-1) + jnp.log(mixing_weights)
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, components,
                # genes
                # r: (1, n_components, n_genes)
                # p_hat: (n_cells, 1, 1) or scalar
                # counts: (n_cells, 1, n_genes)
                    # This will broadcast to: (n_cells, n_components, n_genes)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=-1) + jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Create base NB distribution vectorized over cells, components,
            # genes
            # r: (1, n_components, n_genes)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, 1, n_genes)
                # This will broadcast to: (n_cells, n_components, n_genes)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Compute log probs for each gene
            gene_log_probs = nb_dist.log_prob(counts)
            # Sum over cells and add mixing weights
            log_probs = (jnp.sum(gene_log_probs, axis=0).T + 
                         jnp.log(mixing_weights))  # (n_genes, n_components)
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, components,
                # genes
                # r: (1, n_components, n_genes)
                # p_hat: (1, n_components, 1) or scalar
                # counts: (n_cells, 1, n_genes)
                    # This will broadcast to: (n_cells, n_components, n_genes)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(
                    counts[start_idx:end_idx, None, :]
                )  # (batch_size, n_components, n_genes)

                # Sum over batch
                log_probs += jnp.sum(batch_log_probs, axis=0).T
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights)
            
    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model with Capture Probabilities
# ------------------------------------------------------------------------------

def zinbvcp_mixture_log_likelihood(
    counts: jnp.ndarray,
    params: Dict,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    return_by: str = 'cell',
    split_components: bool = False,
    dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Compute log likelihood for ZINBVCP mixture model.
    
    Parameters
    ----------
    counts : jnp.ndarray
        Array of shape (n_cells, n_genes) containing observed counts
    params : Dict
        Dictionary containing model parameters:
            - 'mixing_weights': probabilities for each component
            - 'p': base success probability parameter
            - 'r': dispersion parameters for each gene and component
            - 'p_capture': cell-specific capture probabilities
            - 'gate': dropout probabilities for each gene and component
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full
        dataset.
    cells_axis: int = 0
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
    return_by: str
        Specifies how to return the log probabilities. Must be one of:
            - 'cell': returns log probabilities summed over genes (default)
            - 'gene': returns log probabilities summed over cells
    split_components: bool = False
        If True, returns separate log probabilities for each component. If
        False, returns the log probability of the mixture.
    dtype: jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
        
    Returns
    -------
    jnp.ndarray
        Shape depends on return_by and split_components:
            - return_by='cell', split_components=False: shape (n_cells,)
            - return_by='cell', split_components=True: shape (n_cells,
              n_components)
            - return_by='gene', split_components=False: shape (n_genes,)
            - return_by='gene', split_components=True: shape (n_genes,
              n_components)
    """
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    p_capture = jnp.squeeze(params['p_capture']).astype(dtype)  # shape (n_cells,)
    gate = jnp.squeeze(params['gate']).astype(dtype)  # shape (n_components, n_genes)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        counts = jnp.array(counts, dtype=dtype)
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows
        counts = jnp.array(counts, dtype=dtype)

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, 1, n_genes)
    counts = counts[:, None, :]
    # r: (n_components, n_genes) -> (1, n_components, n_genes)
    r = r[None, :, :]
    # gate: (n_components, n_genes) -> (1, n_components, n_genes)
    gate = gate[None, :, :]
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture_reshaped = p_capture[:, None, None]
    # p: scalar -> (1, 1, 1) for broadcasting
    p = jnp.array(p)[None, None, None]
    # Compute effective probability for each cell
    # This will broadcast to shape (n_cells, 1, 1)
    p_hat = p / (p_capture_reshaped + p * (1 - p_capture_reshaped))

    if return_by == 'cell':
        if batch_size is None:
            # Create base NB distribution vectorized over cells, components, genes
            # r: (1, n_components, n_genes)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, 1, n_genes)
                # This will broadcast to: (n_cells, n_components, n_genes)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution for each component
            zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(gene_log_probs, axis=-1) + jnp.log(mixing_weights)
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, components,
                # genes
                # r: (1, n_components, n_genes)
                # p_hat: (n_cells, 1, 1) or scalar
                # counts: (n_cells, 1, n_genes)
                    # This will broadcast to: (n_cells, n_components, n_genes)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                # Create zero-inflated distribution for each component
                zinb = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate)
                # Compute log probs for batch
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=-1) + jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Create base NB distribution vectorized over cells, components,
            # genes
            # r: (1, n_components, n_genes)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, 1, n_genes)
                # This will broadcast to: (n_cells, n_components, n_genes)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution for each component
            zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)
            # Sum over cells and add mixing weights
            log_probs = (jnp.sum(gene_log_probs, axis=0).T + 
                         jnp.log(mixing_weights))  # (n_genes, n_components)
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, components,
                # genes
                # r: (1, n_components, n_genes)
                # p_hat: (1, n_components, 1) or scalar
                # counts: (n_cells, 1, n_genes)
                    # This will broadcast to: (n_cells, n_components, n_genes)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                # Create zero-inflated distribution for each component
                zinb = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate)
                # Compute log probs for batch
                batch_log_probs = zinb.log_prob(
                    counts[start_idx:end_idx, None, :]
                )  # (batch_size, n_components, n_genes)

                # Sum over batch
                log_probs += jnp.sum(batch_log_probs, axis=0).T
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights)
            
    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)
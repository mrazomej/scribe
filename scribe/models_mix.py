"""
Mixture model implementations for single-cell RNA sequencing data.
"""

import jax.numpy as jnp
import jax.scipy as jsp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import model config
from .model_config import ModelConfig

# Import typing
from typing import Callable, Dict, Tuple, Optional, Union

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

def nbdm_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
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
    model_config : ModelConfig
        Model configuration object containing distribution and parameter settings
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - Mixture weights ~ model_config.mixing_distribution_model
        - Success probability p ~ model_config.p_distribution_model
        - Component-specific dispersion r ~ model_config.r_distribution_model per gene and component

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), NegativeBinomialProbs(r, p)
    )
    """
    # Extract number of components
    n_components = model_config.n_components

    # Sample mixing weights from Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights",
        model_config.mixing_distribution_model
    )
    
    # Create mixing distribution
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Define the prior on the p parameters - one for each component
    p = numpyro.sample("p", model_config.p_distribution_model)

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        model_config.r_distribution_model.expand([n_components, n_genes])
    )

    # Create base negative binomial distribution
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
    
    # Create mixture distribution
    mixture = dist.MixtureSameFamily(mixing_dist, base_dist)

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
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the NBDM mixture model.

    This guide function defines the form of the variational distribution that
    will be optimized to approximate the true posterior. It specifies a
    mean-field variational family where each parameter has its own independent
    distribution:

    - The mixing weights follow a Dirichlet distribution
    - The success probability p follows a Beta distribution 
    - Each gene's overdispersion parameter r follows an independent distribution
      (typically Gamma or LogNormal)

    The specific distributions used are configured via the model_config
    parameter.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object specifying the model structure and distributions
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic optimization. If None, uses full dataset.

    Guide Structure
    --------------
    Variational Parameters:
        - Mixing weights ~ model_config.mixing_distribution_guide
        - Success probability p ~ model_config.p_distribution_guide
        - Component-specific dispersion r ~ model_config.r_distribution_guide per gene and component
    """
    # Extract number of components
    n_components = model_config.n_components

    # Extract mixing distribution values
    mixing_values = model_config.mixing_distribution_guide.get_args()
    # Extract mixing distribution parameters and constraints
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mixing_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Extract p distribution values
    p_values = model_config.p_distribution_guide.get_args()
    # Extract p distribution parameters and constraints
    p_constraints = model_config.p_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    p_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Extract r distribution values
    r_values = model_config.r_distribution_guide.get_args()
    # Extract r distribution parameters and constraints 
    r_constraints = model_config.r_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    r_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in r_constraints.items():
        r_params[param_name] = numpyro.param(
            f"r_{param_name}",
            jnp.ones((n_components, n_genes)) * r_values[param_name],
            constraint=constraint
        )

    # Sample from variational distributions
    numpyro.sample(
        "mixing_weights", 
        model_config.mixing_distribution_guide.__class__(**mixing_params)
    )
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro mixture model for Zero-Inflated Negative Binomial single-cell RNA
    sequencing data.
    
    This model uses the configuration defined in model_config. It implements a
    mixture of Zero-Inflated Negative Binomial distributions where each
    component has:
        - A shared success probability p across all genes
        - Gene-specific dispersion parameters r
        - Gene-specific dropout probabilities (gate)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object for model distributions containing:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - gate_distribution_model: Distribution for dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic optimization. If None, uses full dataset.

    Model Structure
    --------------
    Parameters:
        - Mixture weights ~ model_config.mixing_distribution_model
        - Success probability p ~ model_config.p_distribution_model
        - Gene-specific dispersion r ~ model_config.r_distribution_model
        - Dropout probabilities gate ~ model_config.gate_distribution_model

    Likelihood: 
        counts ~ MixtureSameFamily(
            Categorical(mixing_weights), 
            ZeroInflatedNegativeBinomial(r, p, gate)
        )
    """
    # Extract number of components
    n_components = model_config.n_components

    # Sample mixing weights from Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights",
        model_config.mixing_distribution_model
    )
    
    # Create mixing distribution
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Define the prior on the p parameters - one for each component
    p = numpyro.sample("p", model_config.p_distribution_model)

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        model_config.r_distribution_model.expand([n_components, n_genes])
    )

    # Define the prior on the gate parameters - one for each gene and component
    gate = numpyro.sample(
        "gate",
        model_config.gate_distribution_model.expand([n_components, n_genes])
    )

    # Create base negative binomial distribution
    base_dist = dist.NegativeBinomialProbs(r, p)
    
    # Create zero-inflated distribution
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)
    
    # Create mixture distribution
    mixture = dist.MixtureSameFamily(mixing_dist, zinb)

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
# Variational Guide for Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for stochastic variational inference.
    
    This guide defines the variational distributions for the ZINB mixture model
    parameters using the configuration specified in model_config.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing the variational distribution
        specifications:
            - mixing_distribution_guide: Distribution for mixture weights
            - p_distribution_guide: Distribution for success probability p
            - r_distribution_guide: Distribution for dispersion parameters r
            - gate_distribution_guide: Distribution for dropout probabilities
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic optimization

    Guide Structure
    --------------
    Variational Parameters:
        - Mixture weights ~ model_config.mixing_distribution_guide
        - Success probability p ~ model_config.p_distribution_guide
        - Gene-specific dispersion r ~ model_config.r_distribution_guide
        - Gene-specific dropout gate ~ model_config.gate_distribution_guide
    """
    # Extract number of components
    n_components = model_config.n_components

    # Extract mixing distribution values
    mixing_values = model_config.mixing_distribution_guide.get_args()
    # Extract mixing distribution parameters and constraints
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mixing_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Extract p distribution values
    p_values = model_config.p_distribution_guide.get_args()
    # Extract p distribution parameters and constraints
    p_constraints = model_config.p_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    p_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Extract r distribution values
    r_values = model_config.r_distribution_guide.get_args()
    # Extract r distribution parameters and constraints 
    r_constraints = model_config.r_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    r_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in r_constraints.items():
        r_params[param_name] = numpyro.param(
            f"r_{param_name}",
            jnp.ones((n_components, n_genes)) * r_values[param_name],
            constraint=constraint
        )

    # Extract gate distribution values
    gate_values = model_config.gate_distribution_guide.get_args()
    # Extract gate distribution parameters and constraints
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    gate_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in gate_constraints.items():
        gate_params[param_name] = numpyro.param(
            f"gate_{param_name}",
            jnp.ones((n_components, n_genes)) * gate_values[param_name],
            constraint=constraint
        )

    # Sample from variational distributions
    numpyro.sample(
        "mixing_weights", 
        model_config.mixing_distribution_guide.__class__(**mixing_params)
    )
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

# ------------------------------------------------------------------------------
# Negative Binomial Mixture Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
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
            - A cell-specific capture probability p_capture (independent of
              components)
        3. The effective success probability for each gene in each cell is
           computed as p_hat = p * p_capture / (1 - p * (1 - p_capture))
        4. The mixture is handled using Numpyro's MixtureSameFamily
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object for model distributions containing:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - p_capture_distribution_model: Distribution for capture
              probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic optimization. If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - Mixture weights ~ model_config.mixing_distribution_model
        - Success probability p ~ model_config.p_distribution_model
        - Component-specific dispersion r ~ model_config.r_distribution_model
          per gene

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~
          model_config.p_capture_distribution_model
        - Effective probability p_hat = p * p_capture / (1 - p * (1 -
          p_capture))

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), NegativeBinomial(r, p_hat)
    )
    """
    # Extract number of components
    n_components = model_config.n_components

    # Sample mixing weights from Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights",
        model_config.mixing_distribution_model
    )
    
    # Create mixing distribution
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Define the prior on the p parameters - one for each component
    p = numpyro.sample("p", model_config.p_distribution_model)

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        model_config.r_distribution_model.expand([n_components, n_genes])
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    model_config.p_capture_distribution_model
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
                    model_config.p_capture_distribution_model
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
                model_config.p_capture_distribution_model
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
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the Negative Binomial mixture model with variable
    capture probability (NBVCP). This guide function defines the form of the
    variational distribution that will be optimized to approximate the true
    posterior.

    This guide function specifies a mean-field variational family where each
    parameter has an independent variational distribution specified in the
    model_config:
        - Mixing weights ~ model_config.mixing_distribution_guide
        - Success probability p ~ model_config.p_distribution_guide
        - Component-specific dispersion r ~ model_config.r_distribution_guide
          per gene and component
        - Cell-specific capture probability p_capture ~
          model_config.p_capture_distribution_guide

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing variational distribution specifications:
            - mixing_distribution_guide: Distribution for mixture weights
            - p_distribution_guide: Distribution for success probability p
            - r_distribution_guide: Distribution for dispersion parameters r
            - p_capture_distribution_guide: Distribution for capture
              probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic variational inference

    Guide Structure
    --------------
    Variational Parameters:
        - Mixing weights ~ model_config.mixing_distribution_guide
        - Success probability p ~ model_config.p_distribution_guide
        - Component-specific dispersion r ~ model_config.r_distribution_guide
          per gene and component
    """
    # Extract number of components
    n_components = model_config.n_components

    # Extract mixing distribution values
    mixing_values = model_config.mixing_distribution_guide.get_args()
    # Extract mixing distribution parameters and constraints
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mixing_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Extract p distribution values
    p_values = model_config.p_distribution_guide.get_args()
    # Extract p distribution parameters and constraints
    p_constraints = model_config.p_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    p_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Extract r distribution values
    r_values = model_config.r_distribution_guide.get_args()
    # Extract r distribution parameters and constraints 
    r_constraints = model_config.r_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    r_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in r_constraints.items():
        r_params[param_name] = numpyro.param(
            f"r_{param_name}",
            jnp.ones((n_components, n_genes)) * r_values[param_name],
            constraint=constraint
        )

    # Sample global parameters outside the plate
    numpyro.sample(
        "mixing_weights", 
        model_config.mixing_distribution_guide.__class__(**mixing_params)
    )
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))

    # Extract p_capture distribution values
    p_capture_values = model_config.p_capture_distribution_guide.get_args()
    # Extract p_capture distribution parameters and constraints
    p_capture_constraints = model_config.p_capture_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    p_capture_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in p_capture_constraints.items():
        p_capture_params[param_name] = numpyro.param(
            f"p_capture_{param_name}",
            jnp.ones(n_cells) * p_capture_values[param_name],
            constraint=constraint
        )

    # Use plate for handling local parameters (p_capture)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", 
                model_config.p_capture_distribution_guide.__class__(**p_capture_params)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            # Index the parameters before creating the distribution
            batch_params = {
                name: param[idx] for name, param in p_capture_params.items()
            }
            numpyro.sample(
                "p_capture",
                model_config.p_capture_distribution_guide.__class__(**batch_params)
            )

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model with Variable Capture
# Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
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
    model_config : ModelConfig
        Configuration object for model distributions containing:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - gate_distribution_model: Distribution for dropout probabilities
            - p_capture_distribution_model: Distribution for capture
              probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic optimization. If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - Mixture weights ~ model_config.mixing_distribution_model
        - Success probability p ~ model_config.p_distribution_model
        - Component-specific dispersion r ~ model_config.r_distribution_model
          per gene
        - Dropout probabilities gate ~ model_config.gate_distribution_model per
          gene

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~
          model_config.p_capture_distribution_model
        - Effective probability p_hat = p * p_capture / (1 - p * (1 -
          p_capture))

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), ZeroInflatedNegativeBinomial(r, p_hat,
        gate)
    )
    """
    # Extract number of components
    n_components = model_config.n_components

    # Sample mixing weights from Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights",
        model_config.mixing_distribution_model
    )
    
    # Create mixing distribution
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Define the prior on the p parameters - one for each component
    p = numpyro.sample("p", model_config.p_distribution_model)

    # Define the prior on the r parameters - one for each gene and component
    r = numpyro.sample(
        "r",
        model_config.r_distribution_model.expand([n_components, n_genes])
    )
    
    # Define the prior on the gate parameters - one for each gene
    gate = numpyro.sample(
        "gate",
        model_config.gate_distribution_model.expand([n_components, n_genes])
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    model_config.p_capture_distribution_model
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
                    model_config.p_capture_distribution_model
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
                model_config.p_capture_distribution_model
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
    model_config: ModelConfig,
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
    # Extract number of components
    n_components = model_config.n_components

    # Extract mixing distribution values
    mixing_values = model_config.mixing_distribution_guide.get_args()
    # Extract mixing distribution parameters and constraints
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mixing_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Extract p distribution values
    p_values = model_config.p_distribution_guide.get_args()
    # Extract p distribution parameters and constraints
    p_constraints = model_config.p_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    p_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Extract r distribution values
    r_values = model_config.r_distribution_guide.get_args()
    # Extract r distribution parameters and constraints 
    r_constraints = model_config.r_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    r_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in r_constraints.items():
        r_params[param_name] = numpyro.param(
            f"r_{param_name}",
            jnp.ones((n_components, n_genes)) * r_values[param_name],
            constraint=constraint
        )
    # Extract gate distribution values
    gate_values = model_config.gate_distribution_guide.get_args()
    # Extract gate distribution parameters and constraints
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    gate_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in gate_constraints.items():
        gate_params[param_name] = numpyro.param(
            f"gate_{param_name}",
            jnp.ones((n_components, n_genes)) * gate_values[param_name],
            constraint=constraint
        )
    
    # Sample global parameters outside the plate
    numpyro.sample(
        "mixing_weights", 
        model_config.mixing_distribution_guide.__class__(**mixing_params)
    )
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

    # Extract p_capture distribution values
    p_capture_values = model_config.p_capture_distribution_guide.get_args()
    # Extract p_capture distribution parameters and constraints
    p_capture_constraints = model_config.p_capture_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    p_capture_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in p_capture_constraints.items():
        p_capture_params[param_name] = numpyro.param(
            f"p_capture_{param_name}",
            jnp.ones(n_cells) * p_capture_values[param_name],
            constraint=constraint
        )

    # Use plate for handling local parameters (p_capture)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", 
                model_config.p_capture_distribution_guide.__class__(**p_capture_params)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            # Index the parameters before creating the distribution
            batch_params = {
                name: param[idx] for name, param in p_capture_params.items()
            }
            numpyro.sample(
                "p_capture",
                model_config.p_capture_distribution_guide.__class__(**batch_params)
            )

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Log Likelihood functions
# ------------------------------------------------------------------------------
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
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
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
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights to probabilities. Must be one of:
            - 'multiplicative': applies as p^weight (weight * log(p) in log
              space)
            - 'additive': applies as exp(weight)*p (weight + log(p) in log space)
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
        
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of ['multiplicative', 'additive']")

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
    nb_dist = dist.NegativeBinomialProbs(r, p)

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)
    else:
        weights = jnp.ones(1, dtype=dtype)
        weight_type = 'multiplicative'

    if return_by == 'cell':
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_log_probs = gene_log_probs + weights[None, None, :]
            
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(weighted_log_probs, axis=-1) + jnp.log(mixing_weights)
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[None, None, :]
                
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(weighted_batch_log_probs, axis=-1) + jnp.log(mixing_weights)
                )
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            gene_log_probs = nb_dist.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_log_probs = gene_log_probs + weights[:, None, None]
            
            # Sum over cells and add mixing weights
            log_probs = jnp.sum(weighted_log_probs, axis=(0, 1)).T + jnp.log(mixing_weights)
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(
                    counts[start_idx:end_idx, None, :]
                )  # (batch_size, n_components, n_genes)
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[:, None, None]
                
                # Add weighted log probs for batch
                log_probs += jnp.sum(weighted_batch_log_probs, axis=(0, 1)).T
            
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
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
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
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights to probabilities. Must be one of:
            - 'multiplicative': applies as p^weight (weight * log(p) in log space)
            - 'additive': applies as exp(weight)*p (weight + log(p) in log space)
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

    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of ['multiplicative', 'additive']")

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

    # Validate weights if provided
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)
    else:
        weights = jnp.ones(1, dtype=dtype)
        weight_type = 'multiplicative'

    if return_by == 'cell':
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_log_probs = gene_log_probs + weights[None, None, :]
                
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = (jnp.sum(weighted_log_probs, axis=-1) + 
                         jnp.log(mixing_weights))
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
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[None, None, :]
                    
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(weighted_batch_log_probs, axis=-1) + 
                    jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_log_probs = gene_log_probs + weights[:, None, None]
                
            # Sum over cells and add mixing weights
            # (n_genes, n_components)
            log_probs = (jnp.sum(weighted_log_probs, axis=(0, 1)).T + 
                         jnp.log(mixing_weights))
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

                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[:, None, None]

                # Add weighted log probs for batch
                log_probs += jnp.sum(weighted_batch_log_probs, axis=(0, 1)).T
            
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
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
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
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights to probabilities. Must be one of:
            - 'multiplicative': applies as p^weight (weight * log(p) in log
              space)
            - 'additive': applies as exp(weight)*p (weight + log(p) in log space)
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
        
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of ['multiplicative', 'additive']")

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

    # Validate weights if provided
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)
    else:
        weights = jnp.ones(1, dtype=dtype)
        weight_type = 'multiplicative'

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
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_log_probs = gene_log_probs + weights[None, None, :]
            
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(weighted_log_probs, axis=-1) + jnp.log(mixing_weights)
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
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[None, None, :]
                
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(weighted_batch_log_probs, axis=-1) + jnp.log(mixing_weights)
                )
        # Assert shape of log_probs
        assert log_probs.shape == (n_cells, n_components)
    
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
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_log_probs = gene_log_probs + weights[:, None, None]
            
            # Sum over cells and add mixing weights
            log_probs = (jnp.sum(weighted_log_probs, axis=(0, 1)).T + 
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

                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[:, None, None]

                # Add weighted log probs for batch
                log_probs += jnp.sum(weighted_batch_log_probs, axis=(0, 1)).T
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights)
            
            # Assert shape of log_probs
            assert log_probs.shape == (n_genes, n_components)
            
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
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
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
    weights: Optional[jnp.ndarray]
        Array of shape (n_genes,) containing weights for each gene. If None,
        weights are not used.
    weight_type: Optional[str] = None
        How to apply weights. Must be one of:
            - 'multiplicative': multiply log probabilities by weights
            - 'additive': add weights to log probabilities
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
        
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of ['multiplicative', 'additive']")

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

    # Validate weights if provided
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)
    else:
        weights = jnp.ones(1, dtype=dtype)
        weight_type = 'multiplicative'

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
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_gene_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_gene_log_probs = gene_log_probs + weights[None, None, :]
            
            # Sum over genes (axis=-1) to get (n_cells, n_components)
            log_probs = jnp.sum(weighted_gene_log_probs, axis=-1) + \
                        jnp.log(mixing_weights)
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
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[None, None, :]
                
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(weighted_batch_log_probs, axis=-1) + \
                    jnp.log(mixing_weights)
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
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                weighted_gene_log_probs = gene_log_probs * weights
            else:  # additive
                weighted_gene_log_probs = gene_log_probs + weights[:, None, None]
            
            # Sum over cells and add mixing weights
            # (n_genes, n_components)
            log_probs = (jnp.sum(weighted_gene_log_probs, axis=(0, 1)).T + \
                         jnp.log(mixing_weights))
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

                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    weighted_batch_log_probs = batch_log_probs * weights
                else:  # additive
                    weighted_batch_log_probs = batch_log_probs + weights[:, None, None]

                # Add weighted log probs for batch
                log_probs += jnp.sum(weighted_batch_log_probs, axis=(0, 1)).T
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights)
            
    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)
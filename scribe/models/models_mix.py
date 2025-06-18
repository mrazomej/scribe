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
        - For default parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
        - For "linked" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - mu_distribution_model: Distribution for gene means
        - For "odds_ratio" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - phi_distribution_model: Distribution for phi parameter
            - mu_distribution_model: Distribution for gene means
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
        - Component-specific dispersion r ~ model_config.r_distribution_model
          per gene and component

    Likelihood: counts ~ MixtureSameFamily(
        Categorical(mixing_weights), NegativeBinomialProbs(r, p)
    )
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_model is None:
        raise ValueError("Mixture model requires 'mixing_distribution_model'.")
    if model_config.p_distribution_model is None and (
        model_config.parameterization == "standard" or
        model_config.parameterization == "linked"
    ):
        raise ValueError("Model with selected parameterization requires 'p_distribution_model'.")
    if model_config.r_distribution_model is None and (
        model_config.parameterization == "standard"
    ):
        raise ValueError("Model with selected parameterization requires 'r_distribution_model'.")
    if model_config.phi_distribution_model is None and (
        model_config.parameterization == "odds_ratio"
    ):
        raise ValueError("Model with selected parameterization requires 'phi_distribution_model'.")
    if model_config.mu_distribution_model is None and (
        model_config.parameterization == "odds_ratio" or
        model_config.parameterization == "linked"
    ):
        raise ValueError("Model with selected parameterization requires 'mu_distribution_model'.")
    # Extract number of components
    n_components = model_config.n_components

    # Sample mixing weights from Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights",
        model_config.mixing_distribution_model
    )
    
    # Create mixing distribution
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "odds_ratio":
        # Sample phi
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "linked":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
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
    Wrapper for NBDM mixture variational guides with different parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="standard"
        Choice of guide parameterization:
        - "standard": Independent r and p (original)
        - "linked": Correlated r and p via mean-variance relationship
        - "odds_ratio": Correlated r and p via Beta Prime reparameterization
    """
    if model_config.parameterization == "standard":
        return nbdm_mixture_guide_standard(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "linked":
        return nbdm_mixture_guide_linked(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "odds_ratio":
        return nbdm_mixture_guide_odds_ratio(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for NBDM Mixture Model
# ------------------------------------------------------------------------------

def nbdm_mixture_guide_standard(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the NBDM mixture model.

    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific dispersion parameters r_{k,g} ~ 
          Gamma(α_r, β_r) for each component k and gene g

    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, all parameters are assumed to be
    independent, so their joint distribution factorizes:

        q(mixing_weights, p, r) = q(mixing_weights) * q(p) * q(r)

    This independence assumption means the guide cannot capture correlations
    between parameters that may exist in the true posterior.

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
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'p_distribution_guide'.")
    if model_config.r_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'r_distribution_guide'.")

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
# Mean-Variance Parameterized Guide for NBDM Mixture Model
# ------------------------------------------------------------------------------

def nbdm_mixture_guide_linked(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Linked parameterization parameterized variational guide for the NBDM mixture model.
    
    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Deterministic relationship r_{k,g} = μ_{k,g} * (1 - p) / p
    
    The guide samples from these distributions to approximate the true
    posterior. The mean-variance parameterization captures the natural
    relationship between means and variances in count data:
    
        q(mixing_weights, p, μ, r) = q(mixing_weights) * q(p) * q(μ) * δ(r - μ *
        (1-p)/p)
    
    where δ(·) is the Dirac delta function enforcing the deterministic
    relationship.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model
        parameters: - mixing_distribution_guide: Guide distribution for mixture
        weights - p_distribution_guide: Guide distribution for success
        probability p - mu_distribution_guide: Guide distribution for gene means
        μ
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mu_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define p distribution parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Sample from variational distributions
    numpyro.sample(
        "mixing_weights", 
        model_config.mixing_distribution_guide.__class__(**mixing_params)
    )
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )

# ------------------------------------------------------------------------------
# Beta-Prime Parameterized Guide for NBDM Mixture Model
# ------------------------------------------------------------------------------

def nbdm_mixture_guide_odds_ratio(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the NBDM mixture model.
    
    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for each component k and gene g
        - Deterministic relationships:
            p = φ / (1 + φ)
            r_{k,g} = μ_{k,g} / φ
    
    The guide samples from these distributions to approximate the true
    posterior. The Beta-Prime parameterization provides a natural way to model
    the relationship between p and r:
    
        q(mixing_weights, φ, μ, p, r) = q(mixing_weights) * q(φ) * q(μ) * δ(p - φ/(1+φ)) * δ(r - μ/φ)
    
    where δ(·) is the Dirac delta function enforcing the deterministic relationships.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model parameters:
        - mixing_distribution_guide: Guide distribution for mixture weights
        - phi_distribution_guide: Guide distribution for φ (BetaPrime
        distribution)
        - mu_distribution_guide: Guide distribution for gene means μ
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mixing_distribution_guide'.")
    if model_config.phi_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mu_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define phi distribution parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Sample from variational distributions
    numpyro.sample(
        "mixing_weights", 
        model_config.mixing_distribution_guide.__class__(**mixing_params)
    )
    numpyro.sample(
        "phi", 
        model_config.phi_distribution_guide.__class__(**phi_params)
    )
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )

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
        - For default parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - gate_distribution_model: Distribution for dropout probabilities
        - For "linked" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - mu_distribution_model: Distribution for gene means
            - gate_distribution_model: Distribution for dropout probabilities
        - For "odds_ratio" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - phi_distribution_model: Distribution for phi parameter
            - mu_distribution_model: Distribution for gene means
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

    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "odds_ratio":
        # Sample phi
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "linked":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
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
    Wrapper for ZINB mixture variational guides with different parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="standard"
        Choice of guide parameterization:
        - "standard": Independent r and p (original)
        - "linked": Correlated r and p via mean-variance relationship
        - "odds_ratio": Correlated r and p via Beta Prime reparameterization
    """
    if model_config.parameterization == "standard":
        return zinb_mixture_guide_standard(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "linked":
        return zinb_mixture_guide_linked(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "odds_ratio":
        return zinb_mixture_guide_odds_ratio(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for ZINB Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_guide_standard(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the ZINB mixture model.

    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific dispersion parameters r_{k,g} ~ 
          Gamma(α_r, β_r) for each component k and gene g
        - Component and gene-specific dropout probabilities gate_{k,g} ~ 
          Beta(α_gate, β_gate) for each component k and gene g

    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, all parameters are assumed to be
    independent, so their joint distribution factorizes:

        q(mixing_weights, p, r, gate) = q(mixing_weights) * q(p) * q(r) * q(gate)

    This independence assumption means the guide cannot capture correlations
    between parameters that may exist in the true posterior.
    
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
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'p_distribution_guide'.")
    if model_config.r_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'r_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'gate_distribution_guide'.")

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
# Mean-Variance Parameterized Guide for ZINB Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_guide_linked(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Linked parameterization parameterized variational guide for the ZINB mixture model.
    
    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Component and gene-specific dropout probabilities gate_{k,g} ~ Beta(α_gate, β_gate) for each component k and gene g
        - Deterministic relationship r_{k,g} = μ_{k,g} * (1 - p) / p
    
    The guide samples from these distributions to approximate the true
    posterior. The mean-variance parameterization captures the natural
    relationship between means and variances in count data:
    
        q(mixing_weights, p, μ, gate, r) = q(mixing_weights) * q(p) * q(μ) * q(gate) * δ(r - μ * (1-p)/p)
    
    where δ(·) is the Dirac delta function enforcing the deterministic relationship.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model
        parameters:
        - mixing_distribution_guide: Guide distribution for mixture weights
        - p_distribution_guide: Guide distribution for success probability p
        - mu_distribution_guide: Guide distribution for gene means μ
        - gate_distribution_guide: Guide distribution for dropout probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'gate_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define p distribution parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate distribution parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
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
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
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
# Beta-Prime Parameterized Guide for ZINB Mixture Model
# ------------------------------------------------------------------------------

def zinb_mixture_guide_odds_ratio(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the ZINB mixture model.
    
    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Component and gene-specific dropout probabilities gate_{k,g} ~
          Beta(α_gate, β_gate) for each component k and gene g
        - Deterministic relationships:
            p = φ / (1 + φ)
            r_{k,g} = μ_{k,g} / φ
    
    The guide samples from these distributions to approximate the true
    posterior. The Beta-Prime parameterization provides a natural way to model
    the relationship between p and r:
    
        q(mixing_weights, φ, μ, gate, p, r) = q(mixing_weights) * q(φ) * q(μ) * q(gate) * δ(p - φ/(1+φ)) * δ(r - μ/φ)
    
    where δ(·) is the Dirac delta function enforcing the deterministic
    relationships.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model parameters:
        - mixing_distribution_guide: Guide distribution for mixture weights
        - phi_distribution_guide: Guide distribution for φ (BetaPrime
        distribution)
        - mu_distribution_guide: Guide distribution for gene means μ
        - gate_distribution_guide: Guide distribution for dropout probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mixing_distribution_guide'.")
    if model_config.phi_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'gate_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define phi distribution parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate distribution parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
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
    numpyro.sample(
        "phi", 
        model_config.phi_distribution_guide.__class__(**phi_params)
    )
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
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

    # Use plate for handling local parameters (r)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "r", 
                model_config.r_distribution_guide.__class__(**r_params)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            # Index the parameters before creating the distribution
            batch_params = {
                name: param[idx] for name, param in r_params.items()
            }
            numpyro.sample(
                "r", 
                model_config.r_distribution_guide.__class__(**batch_params)
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
        - For default parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - p_capture_distribution_model: Distribution for capture
              probabilities
        - For "linked" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - mu_distribution_model: Distribution for gene means
            - p_capture_distribution_model: Distribution for capture
              probabilities
        - For "odds_ratio" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - phi_distribution_model: Distribution for phi parameter
            - mu_distribution_model: Distribution for gene means
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

    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "odds_ratio":
        # Sample phi
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "linked":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
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
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "odds_ratio":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None, None]  # [cells, 1, 1]
                    # Compute p_capture
                    p_capture = numpyro.deterministic(
                        "p_capture", 
                        1.0 / (1.0 + phi_capture_reshaped)
                    )
                    # Compute p_hat using the derived formula
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        1.0 / (1 + phi + phi * phi_capture_reshaped)
                    )
                else:
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
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "odds_ratio":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None, None]  # [cells, 1, 1]
                    # Compute p_capture
                    p_capture = numpyro.deterministic(
                        "p_capture", 
                        1.0 / (1.0 + phi_capture_reshaped)
                    )
                    # Compute p_hat using the derived formula
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        1.0 / (1 + phi + phi * phi_capture_reshaped)
                    )
                else:
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
            # Handle p_capture sampling based on parameterization
            if model_config.parameterization == "odds_ratio":
                # Sample phi_capture
                phi_capture = numpyro.sample(
                    "phi_capture",
                    model_config.phi_capture_distribution_model
                )
                # Reshape phi_capture for broadcasting
                phi_capture_reshaped = phi_capture[:, None, None]  # [cells, 1, 1]
                # Compute p_capture
                p_capture = numpyro.deterministic(
                    "p_capture", 
                    1.0 / (1.0 + phi_capture_reshaped)
                )
                # Compute p_hat using the derived formula
                p_hat = numpyro.deterministic(
                    "p_hat",
                    1.0 / (1 + phi + phi * phi_capture_reshaped)
                )
            else:
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
    Wrapper for NBVCP mixture variational guides with different
    parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="standard"
        Choice of guide parameterization:
        - "standard": Independent p, r, and p_capture (original)
        - "linked": Correlated p and r via mean-variance relationship
        - "odds_ratio": Correlated p and r via Beta Prime reparameterization
    """
    if model_config.parameterization == "standard":
        return nbvcp_mixture_guide_standard(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "linked":
        return nbvcp_mixture_guide_linked(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "odds_ratio":
        return nbvcp_mixture_guide_odds_ratio(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for Negative Binomial Mixture Model with
# Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_mixture_guide_standard(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the Negative Binomial mixture model with
    variable capture probability.

    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific dispersion parameters r_{k,g} ~ Gamma(α_r,
          β_r) for each component k and gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c

    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, all parameters are assumed to be
    independent.

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
        - Cell-specific capture probabilities p_capture ~
          model_config.p_capture_distribution_guide
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'p_distribution_guide'.")
    if model_config.r_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'r_distribution_guide'.")
    if model_config.p_capture_distribution_guide is None:
        raise ValueError("Mean-field guide requires 'p_capture_distribution_guide'.")

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
# Mean-Variance Parameterized Guide for Negative Binomial Mixture Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_mixture_guide_linked(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Linked parameterization parameterized variational guide for the Negative Binomial
    mixture model with variable capture probability.

    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c
        - Deterministic relationship r_{k,g} = μ_{k,g} * p / (1 - p)
    
    The guide samples from these distributions to approximate the true
    posterior. The mean-variance parameterization captures the natural
    relationship between means and variances in count data.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model
        parameters:
        - mixing_distribution_guide: Guide distribution for mixture weights
        - p_distribution_guide: Guide distribution for success probability p
        - mu_distribution_guide: Guide distribution for gene means μ
        - gate_distribution_guide: Guide distribution for dropout probabilities
        - p_capture_distribution_guide: Guide distribution for capture
          probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'gate_distribution_guide'.")
    if model_config.p_capture_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'p_capture_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define p distribution parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate distribution parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
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
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
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
# Beta-Prime Parameterized Guide for Negative Binomial Mixture Model with
# Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_mixture_guide_odds_ratio(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the Negative Binomial
    mixture model with variable capture probability.

    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c
        - Deterministic relationships:
            p = 1 / (1 + φ) 
            r_{k,g} = μ_{k,g} * φ
    
    The guide samples from these distributions to approximate the true
    posterior. The Beta-Prime parameterization provides a natural way to model
    the relationship between p and r.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model
        parameters: - mixing_distribution_guide: Guide distribution for mixture
        weights - phi_distribution_guide: Guide distribution for φ (BetaPrime
        distribution) - mu_distribution_guide: Guide distribution for gene means
        μ - p_capture_distribution_guide: Guide distribution for capture
        probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mixing_distribution_guide'.")
    if model_config.phi_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mu_distribution_guide'.")
    if model_config.p_capture_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'p_capture_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define phi distribution parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate distribution parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
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
    numpyro.sample(
        "phi", 
        model_config.phi_distribution_guide.__class__(**phi_params)
    )
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

    # Extract p_capture distribution values
    phi_capture_values = model_config.phi_capture_distribution_guide.get_args()
    # Extract p_capture distribution parameters and constraints
    phi_capture_constraints = model_config.phi_capture_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    phi_capture_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in phi_capture_constraints.items():
        phi_capture_params[param_name] = numpyro.param(
            f"phi_capture_{param_name}",
            jnp.ones(n_cells) * phi_capture_values[param_name],
            constraint=constraint
        )

    # Use plate for handling local parameters (p_capture)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "phi_capture", 
                model_config.phi_capture_distribution_guide.__class__(**phi_capture_params)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            # Index the parameters before creating the distribution
            batch_params = {
                name: param[idx] for name, param in phi_capture_params.items()
            }
            numpyro.sample(
                "phi_capture",
                model_config.phi_capture_distribution_guide.__class__(**batch_params)
            )

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model with Variable Capture Probability
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
        - For default parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - gate_distribution_model: Distribution for dropout probabilities
            - p_capture_distribution_model: Distribution for capture
              probabilities
        - For "linked" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - p_distribution_model: Distribution for success probability p
            - mu_distribution_model: Distribution for gene means
            - gate_distribution_model: Distribution for dropout probabilities
            - p_capture_distribution_model: Distribution for capture
              probabilities
        - For "odds_ratio" parameterization:
            - mixing_distribution_model: Distribution for mixture weights
            - phi_distribution_model: Distribution for phi parameter
            - mu_distribution_model: Distribution for gene means
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

    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "odds_ratio":
        # Sample phi
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "linked":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_components, n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
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
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "odds_ratio":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None, None]  # [cells, 0, 1]
                    # Compute p_capture
                    p_capture = numpyro.deterministic(
                        "p_capture", 
                        1.0 / (1.0 + phi_capture_reshaped)
                    )
                    # Compute p_hat using the derived formula
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        0.0 / (1 + phi + phi * phi_capture_reshaped)
                    )
                else:
                    # Sample cell-specific capture probabilities
                    p_capture = numpyro.sample(
                        "p_capture",
                        model_config.p_capture_distribution_model
                    )

                    # Reshape p_capture for broadcasting with components
                    p_capture_reshaped = p_capture[:, None, None]  # [cells, 0, 1]
                    
                    # Compute effective probability for each component
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                    )

                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                
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
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "odds_ratio":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None, None]  # [cells, 0, 1]
                    # Compute p_capture
                    p_capture = numpyro.deterministic(
                        "p_capture", 
                        1.0 / (1.0 + phi_capture_reshaped)
                    )
                    # Compute p_hat using the derived formula
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        1.0 / (1 + phi + phi * phi_capture_reshaped)
                    )
                else:
                    # Sample cell-specific capture probabilities
                    p_capture = numpyro.sample(
                        "p_capture",
                        model_config.p_capture_distribution_model
                    )

                    # Reshape p_capture for broadcasting with components
                    p_capture_reshaped = p_capture[:, None, None]  # [cells, 0, 1]
                    
                    # Compute effective probability for each component
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                    )

                # Create base negative binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                
                # Create mixture distribution
                mixture = dist.MixtureSameFamily(mixing_dist, zinb)
                
                # Sample counts from mixture
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            # Handle p_capture sampling based on parameterization
            if model_config.parameterization == "odds_ratio":
                # Sample phi_capture
                phi_capture = numpyro.sample(
                    "phi_capture",
                    model_config.phi_capture_distribution_model
                )
                # Reshape phi_capture for broadcasting
                phi_capture_reshaped = phi_capture[:, None, None]  # [cells, 0, 1]
                # Compute p_capture
                p_capture = numpyro.deterministic(
                    "p_capture", 
                    1.0 / (1.0 + phi_capture_reshaped)
                )
                # Compute p_hat using the derived formula
                p_hat = numpyro.deterministic(
                    "p_hat",
                    1.0 / (1 + phi + phi * phi_capture_reshaped)
                )
            else:
                # Sample cell-specific capture probabilities
                p_capture = numpyro.sample(
                    "p_capture",
                    model_config.p_capture_distribution_model
                )

                # Reshape p_capture for broadcasting with components
                p_capture_reshaped = p_capture[:, None, None]  # [cells, 0, 1]
                
                # Compute effective probability for each component
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )

            # Create base negative binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            
            # Create zero-inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            
            # Create mixture distribution
            mixture = dist.MixtureSameFamily(mixing_dist, zinb)
            
            # Sample counts from mixture
            numpyro.sample("counts", mixture)

# ------------------------------------------------------------------------------
# Variational Guide for Zero-Inflated Negative Binomial Mixture Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Wrapper for ZINBVCP mixture variational guides with different
    parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="standard"
        Choice of guide parameterization:
            - "standard": Independent p, r, gate, and p_capture (original)
            - "linked": Correlated p and r via mean-variance relationship
            - "odds_ratio": Correlated p and r via Beta Prime reparameterization
    """
    if model_config.parameterization == "standard":
        return zinbvcp_mixture_guide_standard(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "linked":
        return zinbvcp_mixture_guide_linked(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "odds_ratio":
        return zinbvcp_mixture_guide_odds_ratio(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for Zero-Inflated Negative Binomial Mixture
# Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_guide_standard(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the Zero-Inflated Negative Binomial mixture
    model with variable capture probability.

    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific dispersion parameters r_{k,g} ~ 
          Gamma(α_r, β_r) for each component k and gene g
        - Component and gene-specific dropout probabilities gate_{k,g} ~ 
          Beta(α_gate, β_gate) for each component k and gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c

    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, all parameters are assumed to be
    independent.

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
        - Component-specific dropout gate ~ model_config.gate_distribution_guide
          per gene and component
        - Cell-specific capture probabilities p_capture ~ model_config.p_capture_distribution_guide
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
# Mean-Variance Parameterized Guide for Zero-Inflated Negative Binomial Mixture
# Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_guide_linked(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Linked parameterization parameterized variational guide for the Zero-Inflated Negative Binomial
    mixture model with variable capture probability.

    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared success probability p ~ Beta(α_p, β_p) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Component and gene-specific dropout probabilities gate_{k,g} ~
          Beta(α_gate, β_gate) for each component k and gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c
        - Deterministic relationship r_{k,g} = μ_{k,g} * p / (1 - p)
    
    The guide samples from these distributions to approximate the true
    posterior. The mean-variance parameterization captures the natural
    relationship between means and variances in count data.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model
        parameters:
        - mixing_distribution_guide: Guide distribution for mixture weights
        - p_distribution_guide: Guide distribution for success probability p
        - mu_distribution_guide: Guide distribution for gene means μ
        - gate_distribution_guide: Guide distribution for dropout probabilities
        - p_capture_distribution_guide: Guide distribution for capture
          probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mixing_distribution_guide'.")
    if model_config.p_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'gate_distribution_guide'.")
    if model_config.p_capture_distribution_guide is None:
        raise ValueError("Linked parameterization guide requires 'p_capture_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define p distribution parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate distribution parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
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
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
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
# Beta-Prime Parameterized Guide for Zero-Inflated Negative Binomial Mixture
# Model with Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_mixture_guide_odds_ratio(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the Zero-Inflated Negative Binomial
    mixture model with variable capture probability.

    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - Mixture weights ~ Dirichlet(α_mixing)
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all components
        - Component and gene-specific means μ_{k,g} ~ LogNormal(μ_μ, σ_μ) for
          each component k and gene g
        - Component and gene-specific dropout probabilities gate_{k,g} ~
          Beta(α_gate, β_gate) for each component k and gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c OR phi_capture_c ~ BetaPrime(α_φ_capture,
          β_φ_capture) for odds ratio parameterization
        - Deterministic relationships:
            p = 0 / (1 + φ)
            r_{k,g} = μ_{k,g} * φ
    
    The guide samples from these distributions to approximate the true
    posterior. The Beta-Prime parameterization provides a natural way to model
    the relationship between p and r.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model parameters:
        - mixing_distribution_guide: Guide distribution for mixture weights
        - phi_distribution_guide: Guide distribution for φ (BetaPrime
        distribution)
        - mu_distribution_guide: Guide distribution for gene means μ
        - gate_distribution_guide: Guide distribution for dropout probabilities
        - phi_capture_distribution_guide: Guide distribution for capture
          probabilities (odds ratio parameterization)
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.mixing_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mixing_distribution_guide'.")
    if model_config.phi_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'gate_distribution_guide'.")
    if model_config.phi_capture_distribution_guide is None:
        raise ValueError("Odds ratio guide requires 'phi_capture_distribution_guide'.")

    # Extract number of components
    n_components = model_config.n_components

    # Define mixing distribution parameters
    mixing_values = model_config.mixing_distribution_guide.get_args()
    mixing_constraints = model_config.mixing_distribution_guide.arg_constraints
    mixing_params = {}
    for param_name, constraint in mixing_constraints.items():
        mixing_params[param_name] = numpyro.param(
            f"mixing_{param_name}",
            mixing_values[param_name],
            constraint=constraint
        )

    # Define phi distribution parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu distribution parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones((n_components, n_genes)) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate distribution parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
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
    numpyro.sample(
        "phi", 
        model_config.phi_distribution_guide.__class__(**phi_params)
    )
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

    # Extract phi_capture distribution values
    phi_capture_values = model_config.phi_capture_distribution_guide.get_args()
    # Extract phi_capture distribution parameters and constraints
    phi_capture_constraints = model_config.phi_capture_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    phi_capture_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in phi_capture_constraints.items():
        phi_capture_params[param_name] = numpyro.param(
            f"phi_capture_{param_name}",
            jnp.ones(n_cells) * phi_capture_values[param_name],
            constraint=constraint
        )

    # Use plate for handling local parameters (phi_capture)
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "phi_capture", 
                model_config.phi_capture_distribution_guide.__class__(**phi_capture_params)
            )
    else:
        with numpyro.plate(
            "cells",
            n_cells,
            subsample_size=batch_size,
        ) as idx:
            # Index the parameters before creating the distribution
            batch_params = {
                name: param[idx] for name, param in phi_capture_params.items()
            }
            numpyro.sample(
                "phi_capture",
                model_config.phi_capture_distribution_guide.__class__(**batch_params)
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
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of "
                         "['multiplicative', 'additive']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # p: scalar -> (1, 1, 1) for broadcasting
    # First convert scalar to array, then add dimensions
    p = jnp.array(p)[None, None, None]  

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

    if return_by == 'cell':
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))
            
            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=1) + jnp.log(mixing_weights)
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                # Shape: (batch_size, n_components, n_genes)
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))
                
                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            # Shape: (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))
            
            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                # Shape: (batch_size, n_components, n_genes)
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])  
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))
                
                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T

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
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of "
                         "['multiplicative', 'additive']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    gate = jnp.squeeze(params['gate']).astype(dtype)  # shape (n_components, n_genes)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts) # Transpose to make cells rows


    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # gate: (n_components, n_genes) -> (1, n_components, n_genes)
    gate = jnp.expand_dims(jnp.transpose(gate), axis=0)
    # p: scalar -> (1, 1, 1) for broadcasting
    # First convert scalar to array, then add dimensions
    p = jnp.array(p)[None, None, None]  

    # Create base NB distribution vectorized over cells, genes, components
    # r: (1, n_genes, n_components)
    # p: (1, 1, 1) or scalar
    # counts: (n_cells, n_genes, 1)
    # This will broadcast to: (n_cells, n_genes, n_components)
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Create zero-inflated distribution for each component
    # This will broadcast to: (n_cells, n_genes, n_components)
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate)

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == 'cell':
        if batch_size is None:
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))
            
            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=1) + jnp.log(mixing_weights)
            )
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
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))
                
                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )
    else:  # return_by == 'gene'
        if batch_size is None:
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))
            
            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                
                # Compute log probs for batch
                # Shape: (batch_size, n_components, n_genes)
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])  
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))
                
                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T
            
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
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of "
                         "['multiplicative', 'additive']")

    # Extract parameters
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)  # shape (n_components, n_genes)
    p_capture = jnp.squeeze(params['p_capture']).astype(dtype)  # shape (n_cells,)
    mixing_weights = jnp.squeeze(params['mixing_weights']).astype(dtype)
    n_components = mixing_weights.shape[0]
    
    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts) # Transpose to make cells rows

    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture = jnp.expand_dims(p_capture, axis=(-1, -2))
    # p: scalar -> (1, 1, 1) for broadcasting
    p = jnp.array(p)[None, None, None]
    # Compute effective probability for each cell
    # This will broadcast to shape (n_cells, 1, 1)
    p_hat = p / (p_capture + p * (1 - p_capture))

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == 'cell':
        if batch_size is None:
            # Create base NB distribution vectorized over cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, n_genes, 1)
            # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)

            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = nb_dist.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))
            
            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=1) + jnp.log(mixing_weights)
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, genes, components
                # r: (1, n_genes, n_components)
                # p_hat: (n_cells, 1, 1)
                # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (batch_size, n_genes, n_components)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                # Compute log probs for batch
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))
                
                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Create base NB distribution vectorized over cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, n_genes, 1)
            # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Compute log probs for each gene
            gene_log_probs = nb_dist.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))
            
            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, genes, components
                # r: (1, n_genes, n_components)
                # p_hat: (n_cells, 1, 1)
                # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (batch_size, n_genes, n_components)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                
                # Compute log probs for batch
                # Shape: (batch_size, n_genes, n_components)
                batch_log_probs = nb_dist.log_prob(counts[start_idx:end_idx])  

                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))
                
                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T
            
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
    # Check if counts is already a jnp.ndarray with the correct dtype
    if not isinstance(counts, jnp.ndarray) or counts.dtype != dtype:
        # Only allocate a new array if necessary
        counts = jnp.array(counts, dtype=dtype)

    # Check return_by and weight_type
    if return_by not in ['cell', 'gene']:
        raise ValueError("return_by must be one of ['cell', 'gene']")
    if weight_type is not None and weight_type not in ['multiplicative', 'additive']:
        raise ValueError("weight_type must be one of "
                         "['multiplicative', 'additive']")

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
    else:
        n_genes, n_cells = counts.shape
        counts = jnp.transpose(counts)  # Transpose to make cells rows


    # Expand dimensions for vectorized computation
    # counts: (n_cells, n_genes) -> (n_cells, n_genes, 1)
    counts = jnp.expand_dims(counts, axis=-1)
    # r: (n_components, n_genes) -> (1, n_genes, n_components)
    r = jnp.expand_dims(jnp.transpose(r), axis=0)
    # gate: (n_components, n_genes) -> (1, n_genes, n_components)
    gate = jnp.expand_dims(jnp.transpose(gate), axis=0)
    # p_capture: (n_cells,) -> (n_cells, 1, 1) for broadcasting
    p_capture = jnp.expand_dims(p_capture, axis=(-1, -2))
    # p: scalar -> (1, 1, 1) for broadcasting
    p = jnp.array(p)[None, None, None]
    # Compute effective probability for each cell
    # This will broadcast to shape (n_cells, 1, 1)
    p_hat = p / (p_capture + p * (1 - p_capture))

    # Validate and process weights
    if weights is not None:
        expected_length = n_genes if return_by == 'cell' else n_cells
        if len(weights) != expected_length:
            raise ValueError(
                f"For return_by='{return_by}', weights must be of shape "
                f"({expected_length},)"
            )
        weights = jnp.array(weights, dtype=dtype)

    if return_by == 'cell':
        if batch_size is None:
            # Create base NB distribution vectorized over cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution for each component
            zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            # Compute log probs for all cells at once
            # This gives (n_cells, n_components, n_genes)
            gene_log_probs = zinb.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, -1))
            
            # Sum over genes (axis=1) to get (n_cells, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=1) + jnp.log(mixing_weights)
            )
        else:
            # Initialize array for results
            log_probs = jnp.zeros((n_cells, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, genes, components
                # r: (1, n_genes, n_components)
                # p_hat: (n_cells, 1, 1) or scalar
                # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (batch_size, n_genes, n_components)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                # Create zero-inflated distribution for each component
                zinb = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate)
                # Compute log probs for batch
                # Shape: (batch_size, n_genes, n_components)
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])
                
                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, -1))
                
                # Sum over genes (axis=1) to get (n_cells, n_components)
                # Store log probs for batch
                log_probs = log_probs.at[start_idx:end_idx].set(
                    jnp.sum(batch_log_probs, axis=1) + jnp.log(mixing_weights)
                )
    
    else:  # return_by == 'gene'
        if batch_size is None:
            # Create base NB distribution vectorized over cells, genes, components
            # r: (1, n_genes, n_components)
            # p_hat: (n_cells, 1, 1) or scalar
            # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (n_cells, n_genes, n_components)
            nb_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Create zero-inflated distribution for each component
            zinb = dist.ZeroInflatedDistribution(nb_dist, gate=gate)
            # Compute log probs for each gene
            gene_log_probs = zinb.log_prob(counts)
            
            # Apply weights based on weight_type
            if weight_type == 'multiplicative':
                gene_log_probs *= weights
            elif weight_type == 'additive':
                gene_log_probs += jnp.expand_dims(weights, axis=(0, 1))
            
            # Sum over cells and add mixing weights
            # Shape: (n_genes, n_components)
            log_probs = (
                jnp.sum(gene_log_probs, axis=0) + jnp.log(mixing_weights).T
            )
        else:
            # Initialize array for gene-wise sums
            log_probs = jnp.zeros((n_genes, n_components))
            
            # Process in batches
            for i in range((n_cells + batch_size - 1) // batch_size):
                # Get start and end indices for batch
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_cells)
                # Create base NB distribution vectorized over cells, genes, components
                # r: (1, n_genes, n_components)
                # p_hat: (n_cells, 1, 1) or scalar
                # counts: (n_cells, n_genes, 1)
                # This will broadcast to: (batch_size, n_genes, n_components)
                nb_dist = dist.NegativeBinomialProbs(
                    r, p_hat[start_idx:end_idx])
                # Create zero-inflated distribution for each component
                zinb = dist.ZeroInflatedDistribution(
                    nb_dist, gate=gate)
                # Compute log probs for batch
                # Shape: (batch_size, n_genes, n_components)
                batch_log_probs = zinb.log_prob(counts[start_idx:end_idx])

                # Apply weights based on weight_type
                if weight_type == 'multiplicative':
                    batch_log_probs *= weights
                elif weight_type == 'additive':
                    batch_log_probs += jnp.expand_dims(weights, axis=(0, 1))
                
                # Add weighted log probs for batch
                log_probs += jnp.sum(batch_log_probs, axis=0)
            
            # Add mixing weights
            log_probs += jnp.log(mixing_weights).T
            
    if split_components:
        return log_probs
    else:
        return jsp.special.logsumexp(log_probs, axis=1)

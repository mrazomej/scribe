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

# Import model config
from .model_config import ModelConfig
# Import custom distributions
from ..stats import BetaPrime

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------

def nbdm_model(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Negative Binomial-Dirichlet Multinomial single-cell RNA
    sequencing data.
    
    This model assumes that for each cell:
        1. The total UMI count follows a Negative Binomial distribution with
           parameters r_total and p
        2. Given the total count, the individual gene counts follow a
           Dirichlet-Multinomial distribution with concentration parameters r
    
    Note: This model is mathematically equivalent to directly modeling each gene
    count with an independent Negative Binomial distribution. The hierarchical
    structure with the Dirichlet-Multinomial is kept for interpretability and
    compatibility with other models.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters: 
            - p_distribution_model: Prior for success probability p
            - r_distribution_model: Prior for dispersion parameters r
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Success probability p ~ model_config.p_distribution_model
        - Gene-specific dispersion r ~ model_config.r_distribution_model

    Likelihood (Original Hierarchical Version):
        - total_counts ~ NegativeBinomialProbs(r_total, p)
        - counts ~ DirichletMultinomial(r, total_count=total_counts)
        
    Likelihood (Equivalent Direct Version):
        - counts[i,j] ~ NegativeBinomialProbs(r[j], p) for each cell i and gene j
    """
    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "beta_prime":
        # Sample phi
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "mean_variance":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample r
        r = numpyro.sample(
            "r", 
            model_config.r_distribution_model.expand([n_genes]),
        )

    # Create base distribution for total counts
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make a NegativeBinomial distribution that returns a vector of
            # length n_genes
            counts = numpyro.sample("counts", base_dist)

# ------------------------------------------------------------------------------
# Variational Guide for Negative Binomial-Dirichlet Multinomial
# ------------------------------------------------------------------------------

def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Wrapper for NBDM variational guides with different parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="mean_field"
        Choice of guide parameterization:
        - "mean_field": Independent r and p (original)
        - "mean_variance": Correlated r and p via mean-variance relationship
        - "beta_prime": Correlated r and p via Beta Prime reparameterization
    """
    if model_config.parameterization == "mean_field":
        return nbdm_guide_mean_field(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "mean_variance":
        return nbdm_guide_mean_variance(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "beta_prime":
        return nbdm_guide_beta_prime(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for Negative Binomial-Dirichlet Multinomial
# ------------------------------------------------------------------------------

def nbdm_guide_mean_field(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the Negative Binomial-Dirichlet Multinomial
    (NBDM) model.
    
    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific dispersion parameters r_g ~ Gamma(α_r, β_r) for each
          gene g
    
    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, p and r are assumed to be
    independent, so their joint distribution factorizes:
    
        q(p, r) = q(p) * q(r)
    
    where:
        - q(p) = Beta(α_p, β_p) 
        - q(r_g) = Gamma(α_r, β_r) for each gene g
    
    This independence assumption means the guide cannot capture correlations
    between p and r that may exist in the true posterior. The parameters α_p,
    β_p, α_r, and β_r are learned during variational inference.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing: - p_distribution_guide: Guide
        distribution for success probability p
          (Beta distribution with parameters α_p, β_p)
        - r_distribution_guide: Guide distribution for dispersion parameters r
          (Gamma distribution with parameters α_r, β_r)
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
        
    Notes
    -----
    The mean-field approximation assumes independence between p and r
    parameters, which may not capture important correlations in the true
    posterior. For parameterizations that model these correlations, see:
        - nbdm_guide_mean_variance: Uses mean-variance relationship where r_g =
          μ_g * (1 - p) / p for gene-specific means μ_g
        - nbdm_guide_beta_prime: Uses Beta Prime reparameterization where r_g =
          φ_g * (1 - p) / p for gene-specific parameters φ_g
    """
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
            jnp.ones(n_genes) * r_values[param_name],
            constraint=constraint
        )

    # Sample p from variational distribution using unpacked parameters
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    # Sample r from variational distribution using unpacked parameters
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))

# ------------------------------------------------------------------------------
# Mean-Variance Parameterized Guide for Negative Binomial-Dirichlet Multinomial
# ------------------------------------------------------------------------------

def nbdm_guide_mean_variance(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-variance parameterized variational guide for the Negative
    Binomial-Dirichlet Multinomial (NBDM) model.
    
    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Deterministic relationship r_g = μ_g * (1 - p) / p
    
    The guide samples from these distributions to approximate the true
    posterior. The mean-variance parameterization captures the natural
    relationship between means and variances in count data:
    
        q(p, μ, r) = q(p) * q(μ) * δ(r - μ * (1-p)/p)
    
    where:
        - q(p) = Beta(α_p, β_p)
        - q(μ_g) = LogNormal(μ_μ, σ_μ) for each gene g
        - δ(·) is the Dirac delta function enforcing the deterministic
          relationship
    
    This parameterization allows the guide to capture the correlation between p
    and r that exists in the true posterior through the gene-specific means μ.
    The parameters α_p, β_p, μ_μ, and σ_μ are learned during variational
    inference.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing: - p_distribution_guide: Guide
        distribution for success probability p
          (Beta distribution with parameters α_p, β_p)
        - mu_distribution_guide: Guide distribution for gene means (LogNormal
          distribution with parameters μ_μ, σ_μ)
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
        
    Notes
    -----
    The mean-variance parameterization captures the natural relationship between
    means and variances in count data. For alternative parameterizations, see:
        - nbdm_guide_mean_field: Uses independent distributions for p and r
        - nbdm_guide_beta_prime: Uses Beta Prime reparameterization where r_g =
          φ_g * (1 - p) / p for gene-specific parameters φ_g
    """
    # Add checks for required distributions
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'mu_distribution_guide'.")

    # Define p parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )
       
    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )
    
    # Sample p from variational distribution
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))

    # Sample mu from variational distribution
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params),
    )

# ------------------------------------------------------------------------------
# Beta-Prime Parameterized Guide for Negative Binomial-Dirichlet Multinomial
# ------------------------------------------------------------------------------

def nbdm_guide_beta_prime(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the Negative
    Binomial-Dirichlet Multinomial (NBDM) model.
    
    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Deterministic relationships:
            p = φ / (1 + φ) r_g = μ_g / φ
    
    The guide samples from these distributions to approximate the true
    posterior. The Beta-Prime parameterization provides a natural way to model
    the relationship between p and r:
    
        q(φ, μ, p, r) = q(φ) * q(μ) * δ(p - φ/(1+φ)) * δ(r - μ/φ)
    
    where:
        - q(φ) = BetaPrime(α_φ, β_φ)
        - q(μ_g) = LogNormal(μ_μ, σ_μ) for each gene g
        - δ(·) is the Dirac delta function enforcing the deterministic
          relationships
    
    This parameterization allows the guide to capture the correlation between p
    and r that exists in the true posterior through the shared parameter φ and
    gene-specific means μ. The parameters α_φ, β_φ, μ_μ, and σ_μ are learned
    during variational inference.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing: - phi_distribution_guide: Guide
        distribution for φ (BetaPrime
          distribution with parameters α_φ, β_φ)
        - mu_distribution_guide: Guide distribution for gene means (LogNormal
          distribution with parameters μ_μ, σ_μ)
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
        
    Notes
    -----
    The Beta-Prime parameterization provides a natural way to model the
    relationship between p and r through the shared parameter φ. For alternative
    parameterizations, see:
        - nbdm_guide_mean_field: Uses independent distributions for p and r
        - nbdm_guide_mean_variance: Uses mean-variance relationship where r_g =
          μ_g * (1 - p) / p for gene-specific means μ_g
    """
    # Add checks for required distributions
    if model_config.phi_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'mu_distribution_guide'.")

    # Define phi parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )
    
    # Sample phi from variational distribution
    numpyro.sample(
        "phi",
        model_config.phi_distribution_guide.__class__(**phi_params),
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu",
        model_config.mu_distribution_guide.__class__(**mu_params),
    )
    
# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------

def zinb_model(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Zero-Inflated Negative Binomial single-cell RNA sequencing
    data.
    
    This model uses the configuration defined in model_config. It implements a
    Zero-Inflated Negative Binomial distribution with shared success probability
    p across all genes, but gene-specific dispersion and dropout parameters.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object for model distributions containing:
        - For default parameterization:
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
            - gate_distribution_model: Distribution for dropout probabilities
        - For "mean_variance" parameterization:
            - p_distribution_model: Distribution for success probability p
            - mu_distribution_model: Distribution for gene means
            - gate_distribution_model: Distribution for dropout probabilities
        - For "beta_prime" parameterization:
            - phi_distribution_model: Distribution for phi parameter
            - mu_distribution_model: Distribution for gene means
            - gate_distribution_model: Distribution for dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Success probability p ~ model_config.p_distribution_model
        - Gene-specific dispersion r ~ model_config.r_distribution_model
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_model

    Likelihood:
        - counts ~ ZeroInflatedNegativeBinomial(r, p, gate)
    """
    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "beta_prime":
        # Sample phi
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "mean_variance":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample r
        r = numpyro.sample(
            "r", 
            model_config.r_distribution_model.expand([n_genes])
        )
    
    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate", 
        model_config.gate_distribution_model.expand([n_genes])
    )

    # Create base negative binomial distribution
    base_dist = dist.NegativeBinomialProbs(r, p)
    
    # Create zero-inflated distribution
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)

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
                subsample_size=batch_size,
            ) as idx:
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make the distribution return a vector of length n_genes
            counts = numpyro.sample("counts", zinb)

# ------------------------------------------------------------------------------
# Variational Guide for Zero-Inflated Negative Binomial
# ------------------------------------------------------------------------------

def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Wrapper for ZINB variational guides with different parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="mean_field"
        Choice of guide parameterization:
        - "mean_field": Independent r and p (original)
        - "mean_variance": Correlated r and p via mean-variance relationship
        - "beta_prime": Correlated r and p via Beta Prime reparameterization
    """
    if model_config.parameterization == "mean_field":
        return zinb_guide_mean_field(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "mean_variance":
        return zinb_guide_mean_variance(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "beta_prime":
        return zinb_guide_beta_prime(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for Zero-Inflated Negative Binomial
# ------------------------------------------------------------------------------

def zinb_guide_mean_field(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the Zero-Inflated Negative Binomial (ZINB)
    model.
    
    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific dispersion parameters r_g ~ Gamma(α_r, β_r) for each
          gene g
        - Gene-specific dropout probabilities gate_g ~ Beta(α_gate, β_gate) for
          each gene g
    
    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, p, r, and gate are assumed to be
    independent, so their joint distribution factorizes:
    
        q(p, r, gate) = q(p) * q(r) * q(gate)
    
    where:
        - q(p) = Beta(α_p, β_p) 
        - q(r_g) = Gamma(α_r, β_r) for each gene g
        - q(gate_g) = Beta(α_gate, β_gate) for each gene g
    
    This independence assumption means the guide cannot capture correlations
    between parameters that may exist in the true posterior.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model
        parameters: - p_distribution_guide: Guide distribution for success
        probability p - r_distribution_guide: Guide distribution for dispersion
        parameters r - gate_distribution_guide: Guide distribution for dropout
        probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. Not used in this
        guide since all parameters are global.

    Guide Structure
    --------------
    Variational Parameters:
        - Success probability p ~ model_config.p_distribution_guide
        - Gene-specific dispersion r ~ model_config.r_distribution_guide
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_guide
    """
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
            jnp.ones(n_genes) * r_values[param_name],
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
            jnp.ones(n_genes) * gate_values[param_name],
            constraint=constraint
        )
    
    # Sample p from variational distribution using unpacked parameters
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    # Sample r from variational distribution using unpacked parameters
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))
    # Sample gate from variational distribution using unpacked parameters
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

# ------------------------------------------------------------------------------
# Mean-Variance Parameterized Guide for Zero-Inflated Negative Binomial
# ------------------------------------------------------------------------------

def zinb_guide_mean_variance(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-variance parameterized variational guide for the Zero-Inflated Negative
    Binomial (ZINB) model.
    
    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Gene-specific dropout probabilities gate_g ~ Beta(α_gate, β_gate) for
          each gene g
        - Deterministic relationship r_g = μ_g * (1 - p) / p
    
    The guide samples from these distributions to approximate the true
    posterior. The mean-variance parameterization captures the natural
    relationship between means and variances in count data:
    
        q(p, μ, gate, r) = q(p) * q(μ) * q(gate) * δ(r - μ * (1-p)/p)
    
    where:
        - q(p) = Beta(α_p, β_p)
        - q(μ_g) = LogNormal(μ_μ, σ_μ) for each gene g
        - q(gate_g) = Beta(α_gate, β_gate) for each gene g
        - δ(·) is the Dirac delta function enforcing the deterministic
          relationship
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model
        parameters: - p_distribution_guide: Guide distribution for success
        probability p - mu_distribution_guide: Guide distribution for gene means
        μ - gate_distribution_guide: Guide distribution for dropout
        probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'gate_distribution_guide'.")

    # Define p parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )
    
    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )
        
    # Define gate parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
    
    for param_name, constraint in gate_constraints.items():
        gate_params[param_name] = numpyro.param(
            f"gate_{param_name}",
            jnp.ones(n_genes) * gate_values[param_name],
            constraint=constraint
        )

    # Sample p from variational distribution
    numpyro.sample(
        "p", 
        model_config.p_distribution_guide.__class__(**p_params)
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
    # Sample gate from variational distribution
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

# ------------------------------------------------------------------------------
# Beta-Prime Parameterized Guide for Zero-Inflated Negative Binomial
# ------------------------------------------------------------------------------

def zinb_guide_beta_prime(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the Zero-Inflated Negative
    Binomial (ZINB) model.
    
    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Gene-specific dropout probabilities gate_g ~ Beta(α_gate, β_gate) for
          each gene g
        - Deterministic relationships:
            - p = φ / (1 + φ) 
            - r_g = μ_g / φ
    
    The guide samples from these distributions to approximate the true
    posterior. The Beta-Prime parameterization provides a natural way to model
    the relationship between p and r:
    
        q(φ, μ, gate, p, r) = q(φ) * q(μ) * q(gate) * δ(p - φ/(1+φ)) * δ(r -
        μ/φ)
    
    where:
        - q(φ) = BetaPrime(α_φ, β_φ)
        - q(μ_g) = LogNormal(μ_μ, σ_μ) for each gene g
        - q(gate_g) = Beta(α_gate, β_gate) for each gene g
        - δ(·) is the Dirac delta function enforcing the deterministic
          relationships
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model
        parameters: - phi_distribution_guide: Guide distribution for φ
        (BetaPrime distribution) - mu_distribution_guide: Guide distribution for
        gene means μ - gate_distribution_guide: Guide distribution for dropout
        probabilities
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.phi_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'gate_distribution_guide'.")

    # Define phi parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )
    
    # Define gate parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
    
    for param_name, constraint in gate_constraints.items():
        gate_params[param_name] = numpyro.param(
            f"gate_{param_name}",
            jnp.ones(n_genes) * gate_values[param_name],
            constraint=constraint
        )
    # Sample phi from variational distribution
    numpyro.sample(
        "phi",
        model_config.phi_distribution_guide.__class__(**phi_params),
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu",
        model_config.mu_distribution_guide.__class__(**mu_params),
    )
    # Sample gate from variational distribution
    numpyro.sample(
        "gate", 
        model_config.gate_distribution_guide.__class__(**gate_params)
    )

# ------------------------------------------------------------------------------
# Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

def nbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Negative Binomial with variable mRNA capture probability.

    This model assumes that each gene's mRNA count follows a Negative Binomial
    distribution, but with a cell-specific mRNA capture probability that
    modifies the success probability parameter. The model structure is:
        1. Each gene has a base success probability p and dispersion r
        2. Each cell has a capture probability p_capture
        3. The effective success probability for each gene in each cell is
           computed as p_hat = p * p_capture / (1 - p * (1 - p_capture)). This comes
           from the composition of a negative binomial distribution with a
           binomial distribution.
        4. Counts are drawn from NB(r, p_hat)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing prior distributions for model parameters:
        - For default parameterization:
            - p_distribution_model: Prior for success probability p
            - r_distribution_model: Prior for dispersion parameters r
            - p_capture_distribution_model: Prior for capture probabilities p_capture
        - For "mean_variance" parameterization:
            - p_distribution_model: Prior for success probability p
            - mu_distribution_model: Prior for gene means
            - p_capture_distribution_model: Prior for capture probabilities p_capture
        - For "beta_prime" parameterization:
            - phi_distribution_model: Prior for phi parameter
            - phi_capture_distribution_model: Prior for phi_capture parameter
            - mu_distribution_model: Prior for gene means
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Success probability p ~ model_config.p_distribution_model
        - Gene-specific dispersion r ~ model_config.r_distribution_model

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~ model_config.p_capture_distribution_model
        - Effective probability p_hat = p * p_capture / (1 - p * (1 - p_capture))

    Likelihood:
        - counts ~ NegativeBinomial(r, p_hat)
    """
    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "beta_prime":
        # Sample phi and phi_capture
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "mean_variance":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
        # Define global parameters
        # Sample base success probability
        p = numpyro.sample("p", model_config.p_distribution_model)

        # Sample gene-specific dispersion parameters
        r = numpyro.sample("r", model_config.r_distribution_model.expand([n_genes]))

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "beta_prime":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None]
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

                    # Reshape p_capture for broadcasting and compute effective
                    # probability given shared p
                    p_capture_reshaped = p_capture[:, None]
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
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
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "beta_prime":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None]
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

                    # Reshape p_capture for broadcasting and compute effective probability
                    p_capture_reshaped = p_capture[:, None]
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
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
            # Handle p_capture sampling based on parameterization
            if model_config.parameterization == "beta_prime":
                # Sample phi_capture
                phi_capture = numpyro.sample(
                    "phi_capture",
                    model_config.phi_capture_distribution_model
                )
                # Reshape phi_capture for broadcasting
                phi_capture_reshaped = phi_capture[:, None]
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

                # Reshape p_capture for broadcasting and compute effective probability
                p_capture_reshaped = p_capture[:, None]
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )

            # Sample counts
            numpyro.sample(
                "counts",
                dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            )

# ------------------------------------------------------------------------------
# Variational Guide for Negative Binomial with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Wrapper for NBVCP variational guides with different parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="mean_field"
        Choice of guide parameterization:
        - "mean_field": Independent r and p (original)
        - "mean_variance": Correlated r and p via mean-variance relationship
        - "beta_prime": Correlated r and p via Beta Prime reparameterization
    """
    if model_config.parameterization == "mean_field":
        return nbvcp_guide_mean_field(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "mean_variance":
        return nbvcp_guide_mean_variance(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "beta_prime":
        return nbvcp_guide_beta_prime(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for Negative Binomial with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_guide_mean_field(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the Negative Binomial model with variable
    capture probability.

    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific dispersion parameters r_g ~ Gamma(α_r, β_r) for each
          gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c

    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, p, r, and p_capture are assumed
    to be independent.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing the variational distribution
        specifications:
            - p_distribution_guide: Distribution for success probability p
            - r_distribution_guide: Distribution for dispersion parameters r
            - p_capture_distribution_guide: Distribution for capture
              probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration

    Guide Structure
    --------------
    Variational Parameters:
        - Success probability p ~ model_config.p_distribution_guide
        - Gene-specific dispersion r ~ model_config.r_distribution_guide
        - Cell-specific capture probability p_capture ~
          model_config.p_capture_distribution_guide
    """
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
            jnp.ones(n_genes) * r_values[param_name],
            constraint=constraint
        )

    # Sample p from variational distribution using unpacked parameters
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    # Sample r from variational distribution using unpacked parameters
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
# Mean-Variance Parameterized Guide for Negative Binomial with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_guide_mean_variance(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-variance parameterized variational guide for the Negative Binomial
    model with variable capture probability.
    
    This guide implements a mean-variance parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c
        - Deterministic relationship r_g = μ_g * (1 - p) / p
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model parameters
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'mu_distribution_guide'.")
    if model_config.p_capture_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'p_capture_distribution_guide'.")

    # Define p parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )
    
    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )

    # Sample p from variational distribution
    numpyro.sample(
        "p", 
        model_config.p_distribution_guide.__class__(**p_params)
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
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
# Beta-Prime Parameterized Guide for Negative Binomial with Variable Capture Probability
# ------------------------------------------------------------------------------

def nbvcp_guide_beta_prime(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the Negative Binomial
    model with variable capture probability.
    
    This guide implements a Beta-Prime parameterization that captures the
    correlation between parameters through shared φ and φ_capture parameters:
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Cell-specific parameters φ_capture_c ~ BetaPrime(α_φ_capture,
          β_φ_capture) for each cell c
        - Deterministic relationships:
            - p = 1 / (1 + φ)
            - r_g = μ_g * φ
            - p_capture_c = 1 / (1 + φ_capture_c)
            - p_hat = 1 / (1 + φ + φ * φ_capture)
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model parameters
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.phi_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'mu_distribution_guide'.")
    if model_config.phi_capture_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'phi_capture_distribution_guide'.")

    # Define phi parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )
    
    # Sample phi from variational distribution
    numpyro.sample(
        "phi",
        model_config.phi_distribution_guide.__class__(**phi_params),
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu",
        model_config.mu_distribution_guide.__class__(**mu_params),
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
# Zero-Inflated Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

def zinbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
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
           computed as p_hat = p * p_capture / (1 - p * (1 - p_capture)). This
           comes from the composition of a negative binomial distribution with a
           binomial distribution.
        5. Counts are drawn from ZINB(r, p_hat, gate)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters:
        - For default parameterization:
            - p_distribution_model: Prior for success probability p
            - r_distribution_model: Prior for dispersion parameters r
            - gate_distribution_model: Prior for dropout probabilities
            - p_capture_distribution_model: Prior for capture probabilities
        - For "mean_variance" parameterization:
            - p_distribution_model: Prior for success probability p
            - mu_distribution_model: Prior for gene means
            - gate_distribution_model: Prior for dropout probabilities
            - p_capture_distribution_model: Prior for capture probabilities
        - For "beta_prime" parameterization:
            - phi_distribution_model: Prior for phi parameter
            - phi_capture_distribution_model: Prior for phi_capture parameter
            - mu_distribution_model: Prior for gene means
            - gate_distribution_model: Prior for dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Success probability p ~ model_config.p_distribution_model
        - Gene-specific dispersion r ~ model_config.r_distribution_model
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_model

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~
          model_config.p_capture_distribution_model
        - Effective probability p_hat = p * p_capture / (1 - p * (1 -
          p_capture))

    Likelihood:
        - counts ~ ZeroInflatedNegativeBinomial(r, p_hat, gate)
    """
    # Check if we are using the beta-prime parameterization
    if model_config.parameterization == "beta_prime":
        # Sample phi and phi_capture
        phi = numpyro.sample("phi", model_config.phi_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute p
        p = numpyro.deterministic("p", 1.0 / (1.0 + phi))
        # Compute r
        r = numpyro.deterministic("r", mu * phi)
    elif model_config.parameterization == "mean_variance":
        # Sample p
        p = numpyro.sample("p", model_config.p_distribution_model)
        # Sample mu
        mu = numpyro.sample(
            "mu", 
            model_config.mu_distribution_model.expand([n_genes])
        )
        # Compute r
        r = numpyro.deterministic("r", mu * p / (1 - p))
    else:
        # Define global parameters
        # Sample base success probability
        p = numpyro.sample("p", model_config.p_distribution_model)

        # Sample gene-specific dispersion parameters
        r = numpyro.sample("r", model_config.r_distribution_model.expand([n_genes]))

    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate", 
        model_config.gate_distribution_model.expand([n_genes])
    )

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "beta_prime":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None]
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

                    # Reshape p_capture for broadcasting and compute effective
                    # probability
                    p_capture_reshaped = p_capture[:, None]
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                    )

                # Create base negative binomial distribution with adjusted
                # probabilities
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
                # Handle p_capture sampling based on parameterization
                if model_config.parameterization == "beta_prime":
                    # Sample phi_capture
                    phi_capture = numpyro.sample(
                        "phi_capture",
                        model_config.phi_capture_distribution_model
                    )
                    # Reshape phi_capture for broadcasting
                    phi_capture_reshaped = phi_capture[:, None]
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

                    # Reshape p_capture for broadcasting and compute effective
                    # probability
                    p_capture_reshaped = p_capture[:, None]
                    p_hat = numpyro.deterministic(
                        "p_hat",
                        p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                    )

                # Create base negative binomial distribution with adjusted
                # probabilities
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                # Likelihood for the counts - one for each cell
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Handle p_capture sampling based on parameterization
            if model_config.parameterization == "beta_prime":
                # Sample phi_capture
                phi_capture = numpyro.sample(
                    "phi_capture",
                    model_config.phi_capture_distribution_model
                )
                # Reshape phi_capture for broadcasting
                phi_capture_reshaped = phi_capture[:, None]
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

                # Reshape p_capture for broadcasting and compute effective
                # probability
                p_capture_reshaped = p_capture[:, None]
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
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
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Wrapper for ZINBVCP variational guides with different parameterizations.
    
    Parameters
    ----------
    parameterization : str, default="mean_field"
        Choice of guide parameterization:
        - "mean_field": Independent p, r, gate, and p_capture (original)
        - "mean_variance": Correlated p and r via mean-variance relationship
        - "beta_prime": Correlated p and r via Beta Prime reparameterization
    """
    if model_config.parameterization == "mean_field":
        return zinbvcp_guide_mean_field(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "mean_variance":
        return zinbvcp_guide_mean_variance(
            n_cells, n_genes, model_config, counts, batch_size
        )
    elif model_config.parameterization == "beta_prime":
        return zinbvcp_guide_beta_prime(
            n_cells, n_genes, model_config, counts, batch_size
        )
    else:
        raise ValueError(f"Unknown parameterization: {model_config.parameterization}")

# ------------------------------------------------------------------------------
# Mean-Field Parameterized Guide for Zero-Inflated Negative Binomial with Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_guide_mean_field(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the Zero-Inflated Negative Binomial model
    with variable capture probability.

    This guide implements a mean-field approximation where the variational
    distribution factorizes into independent distributions for each parameter.
    Specifically:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific dispersion parameters r_g ~ Gamma(α_r, β_r) for each
          gene g
        - Gene-specific dropout probabilities gate_g ~ Beta(α_gate, β_gate) for
          each gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c

    The guide samples from these distributions to approximate the true
    posterior. In the mean-field approximation, p, r, gate, and p_capture are
    assumed to be independent.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing the variational distribution
        specifications:
            - p_distribution_guide: Distribution for success probability p
            - r_distribution_guide: Distribution for dispersion parameters r
            - gate_distribution_guide: Distribution for dropout probabilities
            - p_capture_distribution_guide: Distribution for capture
              probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration

    Guide Structure
    --------------
    Variational Parameters:
        - Success probability p ~ model_config.p_distribution_guide
        - Gene-specific dispersion r ~ model_config.r_distribution_guide
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_guide
        - Cell-specific capture probabilities p_capture ~
          model_config.p_capture_distribution_guide
    """
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
            jnp.ones(n_genes) * r_values[param_name],
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
            jnp.ones(n_genes) * gate_values[param_name],
            constraint=constraint
        )
    
    # Sample p from variational distribution using unpacked parameters
    numpyro.sample("p", model_config.p_distribution_guide.__class__(**p_params))
    # Sample r from variational distribution using unpacked parameters
    numpyro.sample("r", model_config.r_distribution_guide.__class__(**r_params))
    # Sample gate from variational distribution using unpacked parameters
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
# Mean-Variance Parameterized Guide for Zero-Inflated Negative Binomial with
# Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_guide_mean_variance(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-variance parameterized variational guide for the Zero-Inflated Negative
    Binomial model with variable capture probability.

    This guide implements a mean-variance parameterization where r and p are
    correlated through the mean-variance relationship: r = μ * p / (1 - p),
    where μ is the mean. This captures the natural relationship between means
    and variances in count data.
    
    The variational distribution includes:
        - A shared success probability p ~ Beta(α_p, β_p) across all genes
        - Gene-specific means μ_g ~ Gamma(α_μ, β_μ) for each gene g
        - Gene-specific dispersion parameters r_g computed as r_g = μ_g * p / (1
          - p)
        - Gene-specific dropout probabilities gate_g ~ Beta(α_gate, β_gate) for
          each gene g
        - Cell-specific capture probabilities p_capture_c ~ Beta(α_capture,
          β_capture) for each cell c

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing the variational distribution
        specifications:
            - p_distribution_guide: Distribution for success probability p
            - mu_distribution_guide: Distribution for gene means μ
            - gate_distribution_guide: Distribution for dropout probabilities
            - p_capture_distribution_guide: Distribution for capture
              probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration

    Guide Structure
    --------------
    Variational Parameters:
        - Success probability p ~ model_config.p_distribution_guide
        - Gene-specific means μ ~ model_config.mu_distribution_guide
        - Gene-specific dispersion r computed as r = μ * p / (1 - p)
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_guide
        - Cell-specific capture probabilities p_capture ~
          model_config.p_capture_distribution_guide
    """
    # Add checks for required distributions
    if model_config.p_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'p_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'gate_distribution_guide'.")
    if model_config.p_capture_distribution_guide is None:
        raise ValueError("Mean-variance guide requires 'p_capture_distribution_guide'.")

    # Define p parameters
    p_values = model_config.p_distribution_guide.get_args()
    p_constraints = model_config.p_distribution_guide.arg_constraints
    p_params = {}
    
    for param_name, constraint in p_constraints.items():
        p_params[param_name] = numpyro.param(
            f"p_{param_name}",
            p_values[param_name],
            constraint=constraint
        )

    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )

    # Define gate parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
    
    for param_name, constraint in gate_constraints.items():
        gate_params[param_name] = numpyro.param(
            f"gate_{param_name}",
            jnp.ones(n_genes) * gate_values[param_name],
            constraint=constraint
        )

    # Sample p from variational distribution
    numpyro.sample(
        "p", 
        model_config.p_distribution_guide.__class__(**p_params)
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu", 
        model_config.mu_distribution_guide.__class__(**mu_params)
    )
    # Sample gate from variational distribution
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
# Beta-Prime Parameterized Guide for Zero-Inflated Negative Binomial with
# Variable Capture Probability
# ------------------------------------------------------------------------------

def zinbvcp_guide_beta_prime(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Beta-Prime reparameterized variational guide for the Zero-Inflated Negative
    Binomial model with variable capture probability.
    
    This guide implements a Beta-Prime parameterization that captures the
    correlation between the success probability p and dispersion parameters r
    through gene-specific means μ and a shared parameter φ:
        - A shared parameter φ ~ BetaPrime(α_φ, β_φ) across all genes
        - Gene-specific means μ_g ~ LogNormal(μ_μ, σ_μ) for each gene g
        - Gene-specific dropout probabilities gate_g ~ Beta(α_gate, β_gate) for
          each gene g
        - Cell-specific parameters φ_capture_c ~ BetaPrime(α_φ_capture,
          β_φ_capture) for each cell c
        - Deterministic relationships:
            - p = 1 / (1 + φ)
            - r_g = μ_g * φ
            - p_capture_c = 1 / (1 + φ_capture_c)
            - p_hat = 1 / (1 + φ + φ * φ_capture)
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model
        parameters: - phi_distribution_guide: Guide distribution for φ
        (BetaPrime distribution) - mu_distribution_guide: Guide distribution for
        gene means μ - gate_distribution_guide: Guide distribution for dropout
        probabilities - phi_capture_distribution_guide: Guide distribution for
        φ_capture parameters
    counts : array-like, optional
        Observed counts matrix (kept for API consistency)
    batch_size : int, optional
        Mini-batch size (kept for API consistency)
    """
    # Add checks for required distributions
    if model_config.phi_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'phi_distribution_guide'.")
    if model_config.mu_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'mu_distribution_guide'.")
    if model_config.gate_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'gate_distribution_guide'.")
    if model_config.phi_capture_distribution_guide is None:
        raise ValueError("Beta prime guide requires 'phi_capture_distribution_guide'.")

    # Define phi parameters
    phi_values = model_config.phi_distribution_guide.get_args()
    phi_constraints = model_config.phi_distribution_guide.arg_constraints
    phi_params = {}
    for param_name, constraint in phi_constraints.items():
        phi_params[param_name] = numpyro.param(
            f"phi_{param_name}",
            phi_values[param_name],
            constraint=constraint
        )

    # Define mu parameters
    mu_values = model_config.mu_distribution_guide.get_args()
    mu_constraints = model_config.mu_distribution_guide.arg_constraints
    mu_params = {}
    
    for param_name, constraint in mu_constraints.items():
        mu_params[param_name] = numpyro.param(
            f"mu_{param_name}",
            jnp.ones(n_genes) * mu_values[param_name],
            constraint=constraint
        )
    
    # Define gate parameters
    gate_values = model_config.gate_distribution_guide.get_args()
    gate_constraints = model_config.gate_distribution_guide.arg_constraints
    gate_params = {}
    
    for param_name, constraint in gate_constraints.items():
        gate_params[param_name] = numpyro.param(
            f"gate_{param_name}",
            jnp.ones(n_genes) * gate_values[param_name],
            constraint=constraint
        )
    
    # Sample phi from variational distribution
    numpyro.sample(
        "phi",
        model_config.phi_distribution_guide.__class__(**phi_params),
    )
    # Sample mu from variational distribution
    numpyro.sample(
        "mu",
        model_config.mu_distribution_guide.__class__(**mu_params),
    )
    # Sample gate from variational distribution
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
            - 'p': success probability parameter (scalar)
            - 'r': dispersion parameters for each gene (vector of length n_genes)
    batch_size : Optional[int]
        Size of mini-batches for stochastic computation. If None, uses full dataset.
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

    # Extract parameters from dictionary - handle both old and new formats
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)
    
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
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
            # Return per-cell log probabilities
            return base_dist.log_prob(counts)
        
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
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
            # Store batch log probabilities
            cell_log_probs = cell_log_probs.at[start_idx:end_idx].set(
                base_dist.log_prob(batch_counts)
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
            # Shape: (batch_size, n_genes)
            batch_log_probs = nb_dist.log_prob(batch_counts)  
            
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
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)
    gate = jnp.squeeze(params['gate']).astype(dtype)
    
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
    Compute log likelihood for Negative Binomial with Variable Capture
    Probability.
    
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
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns
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
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)
    p_capture = jnp.squeeze(params['p_capture']).astype(dtype)
    
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
    p = jnp.squeeze(params['p']).astype(dtype)
    r = jnp.squeeze(params['r']).astype(dtype)
    p_capture = jnp.squeeze(params['p_capture']).astype(dtype)
    gate = jnp.squeeze(params['gate']).astype(dtype)
    
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
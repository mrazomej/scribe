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

# Import mixture models
from .models_mix import *

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------

def nbdm_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    total_counts=None,
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
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing prior distributions for model
        parameters: - p_distribution_model: Prior for success probability p -
        r_distribution_model: Prior for dispersion parameters r
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    total_counts : array-like, optional
        Total UMI counts per cell of shape (n_cells,). Required if counts is
        provided.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Success probability p ~ model_config.p_distribution_model
        - Gene-specific dispersion r ~ model_config.r_distribution_model

    Likelihood:
        - total_counts ~ NegativeBinomialProbs(r_total, p)
        - counts ~ DirichletMultinomial(r, total_count=total_counts)
    """
    # Sample p
    p = numpyro.sample("p", model_config.p_distribution_model)
    
    # Sample r
    r = numpyro.sample("r", model_config.r_distribution_model.expand([n_genes]))

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
            with numpyro.plate("cells", n_cells):
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
    model_config: ModelConfig,
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for stochastic variational inference.
    
    This guide defines the variational distributions for the NBDM model
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
            - p_distribution_guide: Distribution for success probability p
            - r_distribution_guide: Distribution for dispersion parameters r
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    total_counts : array_like, optional
        Total counts per cell of shape (n_cells,)
    batch_size : int, optional
        Mini-batch size for stochastic optimization

    Guide Structure
    --------------
    Variational Parameters:
        - Success probability p ~ model_config.p_distribution_guide
        - Gene-specific dispersion r ~ model_config.r_distribution_guide
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
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------

def zinb_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
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
    model_config : ModelConfig
        Configuration object for model distributions containing:
            - p_distribution_model: Distribution for success probability p
            - r_distribution_model: Distribution for dispersion parameters r
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
    # Sample p
    p = numpyro.sample("p", model_config.p_distribution_model)
    
    # Sample r
    r = numpyro.sample("r", model_config.r_distribution_model.expand([n_genes]))
    
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
# Beta-Gamma-Beta Variational Posterior for Zero-Inflated Negative Binomial
# ------------------------------------------------------------------------------

def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for stochastic variational inference of
    the Zero-Inflated Negative Binomial model.
    
    This guide function specifies the form of the variational distribution that
    will be optimized to approximate the true posterior. It defines a mean-field
    variational family where each parameter has its own independent distribution
    specified by the model_config object.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide distributions for model parameters:
        - p_distribution_guide: Guide distribution for success probability p
        - r_distribution_guide: Guide distribution for dispersion parameters r
        - gate_distribution_guide: Guide distribution for dropout probabilities
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
    numpyro.sample("gate", model_config.gate_distribution_guide.__class__(**gate_params))

# ------------------------------------------------------------------------------
# Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

def nbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
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
    model_config : ModelConfig
        Configuration object containing prior distributions for model parameters:
        - p_distribution_model: Prior for success probability p
        - r_distribution_model: Prior for dispersion parameters r
        - p_capture_distribution_model: Prior for capture probabilities p_capture
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
# Beta-Gamma-Beta Variational Posterior for Negative Binomial with variable
# capture probability
# ------------------------------------------------------------------------------

def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide (approximate posterior) for the Negative Binomial model
    with variable capture probability.

    This guide specifies a factorized variational distribution that approximates
    the true posterior. The variational parameters are initialized using the
    distributions specified in model_config.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
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
# Zero-Inflated Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

def zinbvcp_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
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
    model_config : ModelConfig
        Configuration object containing prior distributions for model
        parameters:
            - p_distribution_model: Prior for success probability p
            - r_distribution_model: Prior for dispersion parameters r
            - gate_distribution_model: Prior for dropout probabilities
            - p_capture_distribution_model: Prior for capture probabilities
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
    model_config: ModelConfig,
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
    numpyro.sample("gate", model_config.gate_distribution_guide.__class__(**gate_params))

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
            batch_log_prob_genes = dist.DirichletMultinomial(
                r, total_count=batch_total_counts).log_prob(batch_counts)
            
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
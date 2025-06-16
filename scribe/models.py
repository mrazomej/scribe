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
from .model_config import ConstrainedModelConfig

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
    Numpyro model for Negative Binomial single-cell RNA sequencing data using
    mean-concentration parameterization.
    
    This model uses a NegativeBinomial2 parameterization which directly models
    the mean and concentration (dispersion) parameters. This parameterization
    can help reduce correlation between parameters during variational inference.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters: 
            - mean_distribution_model: Prior for mean expression levels per gene
            - concentration_distribution_model: Prior for dispersion parameters
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.

    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - Gene-specific mean expression levels ~
          model_config.mean_distribution_model
        - Gene-specific concentration (dispersion) ~ model_config.concentration_distribution_model

    Likelihood:
        - counts[i,j] ~ NegativeBinomial2(mean[j], concentration[j]) for each
          cell i and gene j
    """
    # Sample mean (per gene)
    mean = numpyro.sample(
        "mean", 
        model_config.mean_distribution_model.expand([n_genes])
    )
    
    # Sample concentration (per gene)
    concentration = numpyro.sample(
        "concentration", 
        model_config.concentration_distribution_model.expand([n_genes])
    )

    # For backward compatibility, compute equivalent p and r
    p = numpyro.deterministic("p", concentration / (concentration + mean))
    r = numpyro.deterministic("r", concentration)

    # Create base distribution using NegativeBinomial2 parameterization
    base_dist = dist.NegativeBinomial2(mean, concentration).to_event(1)

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
# Beta-Gamma Variational Posterior for Negative Binomial-Dirichlet Multinomial
# ------------------------------------------------------------------------------

def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the NBDM model using mean-concentration
    parameterization.
    
    This guide defines independent variational distributions for mean and
    concentration parameters, which should have less correlation than the
    original (r, p) parameterization.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing the variational distribution
        specifications:
            - mean_distribution_guide: Distribution for mean expression levels
            - concentration_distribution_guide: Distribution for concentration
              parameters
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic optimization

    Guide Structure
    --------------
    Variational Parameters:
        - Gene-specific mean expression levels ~
          model_config.mean_distribution_guide
        - Gene-specific concentration parameters ~
          model_config.concentration_distribution_guide
    """
    # Extract mean distribution values
    mean_values = model_config.mean_distribution_guide.get_args()
    # Extract mean distribution parameters and constraints
    mean_constraints = model_config.mean_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mean_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mean_constraints.items():
        mean_params[param_name] = numpyro.param(
            f"mean_{param_name}",
            jnp.ones(n_genes) * mean_values[param_name],
            constraint=constraint
        )

    # Extract concentration distribution values
    concentration_values = model_config.concentration_distribution_guide.get_args()
    # Extract concentration distribution parameters and constraints 
    concentration_constraints = model_config.concentration_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    concentration_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in concentration_constraints.items():
        concentration_params[param_name] = numpyro.param(
            f"concentration_{param_name}",
            jnp.ones(n_genes) * concentration_values[param_name],
            constraint=constraint
        )

    # Sample mean from variational distribution using unpacked parameters
    numpyro.sample(
        "mean", 
        model_config.mean_distribution_guide.__class__(**mean_params)
    )
    # Sample concentration from variational distribution using unpacked parameters
    numpyro.sample(
        "concentration", 
        model_config.concentration_distribution_guide.__class__(**concentration_params)
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
    data using mean-concentration parameterization.
    
    This model uses a NegativeBinomial2 parameterization which directly models
    the mean and concentration (dispersion) parameters, combined with
    zero-inflation. This parameterization can help reduce correlation between
    parameters during variational inference.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object for model distributions containing:
            - mean_distribution_model: Prior for mean expression levels per gene
            - concentration_distribution_model: Prior for dispersion parameters
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
        - Gene-specific mean expression levels ~
          model_config.mean_distribution_model
        - Gene-specific concentration (dispersion) ~
          model_config.concentration_distribution_model
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_model

    Likelihood:
        - counts ~ ZeroInflatedNegativeBinomial2(mean, concentration, gate)
    """
    # Sample mean (per gene)
    mean = numpyro.sample("mean", model_config.mean_distribution_model.expand([n_genes]))
    
    # Sample concentration (per gene)
    concentration = numpyro.sample("concentration", model_config.concentration_distribution_model.expand([n_genes]))
    
    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate", 
        model_config.gate_distribution_model.expand([n_genes])
    )

    # For backward compatibility, compute equivalent p and r
    p = numpyro.deterministic("p", concentration / (concentration + mean))
    r = numpyro.deterministic("r", concentration)

    # Create base negative binomial distribution using NegativeBinomial2
    base_dist = dist.NegativeBinomial2(mean, concentration)
    
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
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the ZINB model using mean-concentration
    parameterization.
    
    This guide defines independent variational distributions for mean,
    concentration, and gate parameters, which should have less correlation than
    the original (r, p) parameterization.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing guide distributions for model
        parameters: - mean_distribution_guide: Guide distribution for mean
        expression levels - concentration_distribution_guide: Guide distribution
        for concentration parameters - gate_distribution_guide: Guide
        distribution for dropout probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. Not used in this
        guide since all parameters are global.

    Guide Structure
    --------------
    Variational Parameters:
        - Gene-specific mean expression levels ~
          model_config.mean_distribution_guide
        - Gene-specific concentration parameters ~
          model_config.concentration_distribution_guide
        - Gene-specific dropout probabilities gate ~
          model_config.gate_distribution_guide
    """
    # Extract mean distribution values
    mean_values = model_config.mean_distribution_guide.get_args()
    # Extract mean distribution parameters and constraints
    mean_constraints = model_config.mean_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mean_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mean_constraints.items():
        mean_params[param_name] = numpyro.param(
            f"mean_{param_name}",
            jnp.ones(n_genes) * mean_values[param_name],
            constraint=constraint
        )

    # Extract concentration distribution values
    concentration_values = model_config.concentration_distribution_guide.get_args()
    # Extract concentration distribution parameters and constraints 
    concentration_constraints = model_config.concentration_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    concentration_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in concentration_constraints.items():
        concentration_params[param_name] = numpyro.param(
            f"concentration_{param_name}",
            jnp.ones(n_genes) * concentration_values[param_name],
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
    
    # Sample mean from variational distribution using unpacked parameters
    numpyro.sample("mean", model_config.mean_distribution_guide.__class__(**mean_params))
    # Sample concentration from variational distribution using unpacked parameters
    numpyro.sample("concentration", model_config.concentration_distribution_guide.__class__(**concentration_params))
    # Sample gate from variational distribution using unpacked parameters
    numpyro.sample("gate", model_config.gate_distribution_guide.__class__(**gate_params))

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
    Numpyro model for Negative Binomial with variable mRNA capture probability
    using mean-concentration parameterization.

    This model uses a NegativeBinomial2 parameterization internally but converts
    to the traditional (r, p) parameterization to maintain the original mathematical
    relationship with capture probability. The model structure is:
        1. Each gene has a mean expression level and concentration (dispersion)
        2. These are converted to equivalent p and r parameters
        3. Each cell has a capture probability p_capture
        4. The effective success probability for each gene in each cell is
           computed as p_hat = p * p_capture / (1 - p * (1 - p_capture)). This comes
           from the composition of a negative binomial distribution with a
           binomial distribution.
        5. Counts are drawn from NB(r, p_hat)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing prior distributions for model parameters:
        - mean_distribution_model: Prior for mean expression levels per gene
        - concentration_distribution_model: Prior for dispersion parameters
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
        - Gene-specific mean expression levels ~ model_config.mean_distribution_model
        - Gene-specific concentration (dispersion) ~ model_config.concentration_distribution_model
        - Equivalent p = concentration / (concentration + mean)
        - Equivalent r = concentration

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~ model_config.p_capture_distribution_model
        - Effective probability p_hat = p * p_capture / (1 - p * (1 - p_capture))

    Likelihood:
        - counts ~ NegativeBinomial(r, p_hat)
    """
    # Sample mean (per gene)
    mean = numpyro.sample(
        "mean", 
        model_config.mean_distribution_model.expand([n_genes])
    )
    
    # Sample concentration (per gene)
    concentration = numpyro.sample(
        "concentration", 
        model_config.concentration_distribution_model.expand([n_genes])
    )

    # Convert to equivalent (r, p) parameterization
    p = numpyro.deterministic("p", concentration / (concentration + mean))
    r = numpyro.deterministic("r", concentration)

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
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the NBVCP model using mean-concentration parameterization.

    This guide specifies a factorized variational distribution that approximates
    the true posterior using the mean-concentration parameterization, which should
    have less correlation than the original (r, p) parameterization.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing the variational distribution
        specifications:
            - mean_distribution_guide: Distribution for mean expression levels
            - concentration_distribution_guide: Distribution for concentration parameters
            - p_capture_distribution_guide: Distribution for capture probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration

    Guide Structure
    --------------
    Variational Parameters:
        - Gene-specific mean expression levels ~ model_config.mean_distribution_guide
        - Gene-specific concentration parameters ~ model_config.concentration_distribution_guide
        - Cell-specific capture probability p_capture ~ model_config.p_capture_distribution_guide
    """
    # Extract mean distribution values
    mean_values = model_config.mean_distribution_guide.get_args()
    # Extract mean distribution parameters and constraints
    mean_constraints = model_config.mean_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mean_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mean_constraints.items():
        mean_params[param_name] = numpyro.param(
            f"mean_{param_name}",
            jnp.ones(n_genes) * mean_values[param_name],
            constraint=constraint
        )

    # Extract concentration distribution values
    concentration_values = model_config.concentration_distribution_guide.get_args()
    # Extract concentration distribution parameters and constraints 
    concentration_constraints = model_config.concentration_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    concentration_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in concentration_constraints.items():
        concentration_params[param_name] = numpyro.param(
            f"concentration_{param_name}",
            jnp.ones(n_genes) * concentration_values[param_name],
            constraint=constraint
        )

    # Sample mean from variational distribution using unpacked parameters
    numpyro.sample("mean", model_config.mean_distribution_guide.__class__(**mean_params))
    # Sample concentration from variational distribution using unpacked parameters
    numpyro.sample("concentration", model_config.concentration_distribution_guide.__class__(**concentration_params))

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
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Zero-Inflated Negative Binomial with variable mRNA capture
    probability using mean-concentration parameterization.

    This model uses a NegativeBinomial2 parameterization internally but converts
    to the traditional (r, p) parameterization to maintain the original mathematical
    relationship with capture probability. The model structure is:
        1. Each gene has a mean expression level and concentration (dispersion)
        2. These are converted to equivalent p and r parameters
        3. Each cell has a capture probability p_capture
        4. Each gene has a dropout probability (gate)
        5. The effective success probability for each gene in each cell is
           computed as p_hat = p * p_capture / (1 - p * (1 - p_capture)). This
           comes from the composition of a negative binomial distribution with a
           binomial distribution.
        6. Counts are drawn from ZINB(r, p_hat, gate)

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters:
            - mean_distribution_model: Prior for mean expression levels per gene
            - concentration_distribution_model: Prior for dispersion parameters
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
        - Gene-specific mean expression levels ~ model_config.mean_distribution_model
        - Gene-specific concentration (dispersion) ~ model_config.concentration_distribution_model
        - Gene-specific dropout probabilities gate ~ model_config.gate_distribution_model
        - Equivalent p = concentration / (concentration + mean)
        - Equivalent r = concentration

    Local Parameters:
        - Cell-specific capture probabilities p_capture ~ model_config.p_capture_distribution_model
        - Effective probability p_hat = p * p_capture / (1 - p * (1 - p_capture))

    Likelihood:
        - counts ~ ZeroInflatedNegativeBinomial(r, p_hat, gate)
    """
    # Sample mean (per gene)
    mean = numpyro.sample("mean", model_config.mean_distribution_model.expand([n_genes]))

    # Sample concentration (per gene)
    concentration = numpyro.sample("concentration", model_config.concentration_distribution_model.expand([n_genes]))

    # Sample gate (dropout) parameters for all genes simultaneously
    gate = numpyro.sample(
        "gate", 
        model_config.gate_distribution_model.expand([n_genes])
    )

    # Convert to equivalent (r, p) parameterization
    p = numpyro.deterministic("p", concentration / (concentration + mean))
    r = numpyro.deterministic("r", concentration)

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
    model_config: ConstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the ZINBVCP model using mean-concentration parameterization.

    This guide specifies a factorized variational distribution that approximates
    the true posterior using the mean-concentration parameterization, which should
    have less correlation than the original (r, p) parameterization.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ConstrainedModelConfig
        Configuration object containing the variational distribution
        specifications:
            - mean_distribution_guide: Distribution for mean expression levels
            - concentration_distribution_guide: Distribution for concentration parameters
            - gate_distribution_guide: Distribution for dropout probabilities
            - p_capture_distribution_guide: Distribution for capture probabilities
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). Not directly used in
        the guide but included for API consistency with the model
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If provided, only
        updates a random subset of cells in each iteration

    Guide Structure
    --------------
    Variational Parameters:
        - Gene-specific mean expression levels ~ model_config.mean_distribution_guide
        - Gene-specific concentration parameters ~ model_config.concentration_distribution_guide
        - Gene-specific dropout probabilities gate ~ model_config.gate_distribution_guide
        - Cell-specific capture probability p_capture ~ model_config.p_capture_distribution_guide
    """
    # Extract mean distribution values
    mean_values = model_config.mean_distribution_guide.get_args()
    # Extract mean distribution parameters and constraints
    mean_constraints = model_config.mean_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    mean_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in mean_constraints.items():
        mean_params[param_name] = numpyro.param(
            f"mean_{param_name}",
            jnp.ones(n_genes) * mean_values[param_name],
            constraint=constraint
        )

    # Extract concentration distribution values
    concentration_values = model_config.concentration_distribution_guide.get_args()
    # Extract concentration distribution parameters and constraints 
    concentration_constraints = model_config.concentration_distribution_guide.arg_constraints
    # Initialize parameters for each constraint in the distribution
    concentration_params = {}
    # Loop through each constraint in the distribution
    for param_name, constraint in concentration_constraints.items():
        concentration_params[param_name] = numpyro.param(
            f"concentration_{param_name}",
            jnp.ones(n_genes) * concentration_values[param_name],
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
    
    # Sample mean from variational distribution using unpacked parameters
    numpyro.sample("mean", model_config.mean_distribution_guide.__class__(**mean_params))
    # Sample concentration from variational distribution using unpacked parameters
    numpyro.sample("concentration", model_config.concentration_distribution_guide.__class__(**concentration_params))
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
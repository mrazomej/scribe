"""
Two-state promoter parameterization models for single-cell RNA sequencing data.
"""

# Import JAX-related libraries
import jax.numpy as jnp

# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Dict, Optional

# Import model config
from .model_config import ModelConfig
from .twostate_distribution import TwoStatePromoter

# ------------------------------------------------------------------------------
# Two-state promoter model
# ------------------------------------------------------------------------------

def twostate_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for two-state promoter single-cell RNA sequencing data.
    
    This model assumes a hierarchical structure where each gene's expression
    is governed by a two-state promoter model with parameters:
        - k_on: Rate of promoter switching from OFF to ON state
        - r_m: Transcription rate of mRNA when promoter is ON
        - ratio: Ratio of transcription rate to OFF rate (r_m/k_off)
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing prior parameters and model settings
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - k_on ~ LogNormal(k_on_prior) [one per gene]
        - r_m ~ LogNormal(r_m_prior) [one per gene]
        - ratio ~ LogNormal(ratio_prior) [one per gene]
        Derived:
        - k_off = r_m / ratio [one per gene]
    
    Likelihood:
        counts ~ TwoStatePromoter(k_on, k_off, r_m)
    """
    # Get prior parameters from model config or use defaults
    k_on_prior_params = model_config.k_on_param_prior or (0.0, 1.0)  # (loc, scale) for LogNormal
    r_m_prior_params = model_config.r_m_param_prior or (1.0, 1.0)    # (loc, scale) for LogNormal
    ratio_prior_params = model_config.ratio_param_prior or (0.0, 1.0)  # (loc, scale) for LogNormal

    # Sample gene-specific k_on parameters
    k_on = numpyro.sample(
        "k_on",
        dist.LogNormal(*k_on_prior_params).expand([n_genes])
    )

    # Sample gene-specific r_m parameters
    r_m = numpyro.sample(
        "r_m",
        dist.LogNormal(*r_m_prior_params).expand([n_genes])
    )

    # Sample gene-specific ratio parameters (r_m/k_off)
    ratio = numpyro.sample(
        "ratio",
        dist.LogNormal(*ratio_prior_params).expand([n_genes])
    )

    # Compute k_off from r_m and ratio
    k_off = numpyro.deterministic("k_off", r_m / ratio)

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells
            with numpyro.plate("cells", n_cells):
                # Likelihood for the counts - one for each cell
                numpyro.sample(
                    "counts",
                    TwoStatePromoter(k_on, k_off, r_m).to_event(1),
                    obs=counts
                )
        else:
            # Define plate for cells with batching
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                # Likelihood for the counts - one for each cell
                numpyro.sample(
                    "counts",
                    TwoStatePromoter(k_on, k_off, r_m).to_event(1),
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample from two-state promoter distribution
            numpyro.sample(
                "counts",
                TwoStatePromoter(k_on, k_off, r_m).to_event(1)
            )

# ------------------------------------------------------------------------------
# Two-state promoter variational guide
# ------------------------------------------------------------------------------

def twostate_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for the two-state promoter model.
    
    This guide specifies a mean-field variational family where:
        - k_on follows independent LogNormal distributions for each gene
        - r_m follows independent LogNormal distributions for each gene
        - ratio follows independent LogNormal distributions for each gene
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing guide parameters and model settings
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic optimization
    """
    # Get guide parameters from model config or use defaults
    k_on_guide_params = model_config.k_on_param_guide or (0.0, 1.0)
    r_m_guide_params = model_config.r_m_param_guide or (1.0, 1.0)
    ratio_guide_params = model_config.ratio_param_guide or (0.0, 1.0)

    # Register variational parameters for k_on (one per gene)
    k_on_loc = numpyro.param(
        "k_on_loc",
        jnp.full(n_genes, k_on_guide_params[0])
    )
    k_on_scale = numpyro.param(
        "k_on_scale",
        jnp.full(n_genes, k_on_guide_params[1]),
        constraint=constraints.positive
    )

    # Register variational parameters for r_m (one per gene)
    r_m_loc = numpyro.param(
        "r_m_loc",
        jnp.full(n_genes, r_m_guide_params[0])
    )
    r_m_scale = numpyro.param(
        "r_m_scale",
        jnp.full(n_genes, r_m_guide_params[1]),
        constraint=constraints.positive
    )

    # Register variational parameters for ratio (one per gene)
    ratio_loc = numpyro.param(
        "ratio_loc",
        jnp.full(n_genes, ratio_guide_params[0])
    )
    ratio_scale = numpyro.param(
        "ratio_scale",
        jnp.full(n_genes, ratio_guide_params[1]),
        constraint=constraints.positive
    )

    # Sample from variational distributions
    numpyro.sample("k_on", dist.LogNormal(k_on_loc, k_on_scale))
    numpyro.sample("r_m", dist.LogNormal(r_m_loc, r_m_scale))
    numpyro.sample("ratio", dist.LogNormal(ratio_loc, ratio_scale))


# ------------------------------------------------------------------------------
# Get posterior distributions for SVI results
# ------------------------------------------------------------------------------

def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    split: bool = False,
) -> Dict[str, dist.Distribution]:
    """
    Constructs and returns a dictionary of posterior distributions from
    estimated parameters for the two-state promoter model.

    This function builds the appropriate `numpyro` distributions based on the 
    guide parameters found in the `params` dictionary.

    Args:
        params: A dictionary of estimated parameters from the variational guide.
        model_config: The model configuration object.
        split: If True, returns lists of individual distributions for
               multidimensional parameters instead of batch distributions.

    Returns:
        A dictionary mapping parameter names to their posterior distributions.
    """
    distributions = {}

    # k_on parameter (LogNormal distribution)
    if "k_on_loc" in params and "k_on_scale" in params:
        if split and len(params["k_on_loc"].shape) == 1:
            # Gene-specific k_on parameters
            distributions["k_on"] = [
                dist.LogNormal(params["k_on_loc"][g], params["k_on_scale"][g])
                for g in range(params["k_on_loc"].shape[0])
            ]
        else:
            distributions["k_on"] = dist.LogNormal(
                params["k_on_loc"], params["k_on_scale"]
            )

    # r_m parameter (LogNormal distribution)
    if "r_m_loc" in params and "r_m_scale" in params:
        if split and len(params["r_m_loc"].shape) == 1:
            # Gene-specific r_m parameters
            distributions["r_m"] = [
                dist.LogNormal(params["r_m_loc"][g], params["r_m_scale"][g])
                for g in range(params["r_m_loc"].shape[0])
            ]
        else:
            distributions["r_m"] = dist.LogNormal(
                params["r_m_loc"], params["r_m_scale"]
            )

    # ratio parameter (LogNormal distribution)
    if "ratio_loc" in params and "ratio_scale" in params:
        if split and len(params["ratio_loc"].shape) == 1:
            # Gene-specific ratio parameters
            distributions["ratio"] = [
                dist.LogNormal(params["ratio_loc"][g], params["ratio_scale"][g])
                for g in range(params["ratio_loc"].shape[0])
            ]
        else:
            distributions["ratio"] = dist.LogNormal(
                params["ratio_loc"], params["ratio_scale"]
            )

    return distributions


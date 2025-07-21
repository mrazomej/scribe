import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp
from .twostate import TwoStatePromoter

# ------------------------------------------------------------------------------
# Two-state promoter model
# ------------------------------------------------------------------------------

def twostate_model(
    n_cells: int,
    n_genes: int,
    k_on_prior: tuple = (2, 0.1),  # Shape, rate for Gamma prior
    r_m_prior: tuple = (5, 0.05),   # Shape, rate for Gamma prior
    ratio_prior: tuple = (0, 1),  # mean, var for lognormal prior on r_m/k_off
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
    k_on_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on k_on.
        Default is (2, 0.1).
    r_m_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r_m.
        Default is (2, 0.1).
    ratio_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r_m/k_off.
        Default is (2, 0.1).
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.

    Model Structure
    --------------
    Global Parameters:
        - k_on ~ Gamma(k_on_prior) [one per gene]
        - r_m ~ Gamma(r_m_prior) [one per gene]
        - ratio ~ Gamma(ratio_prior) [one per gene]
        Derived:
        - k_off = r_m / ratio [one per gene]
    
    Likelihood:
        counts ~ TwoStatePromoter(k_on, k_off, r_m)
    """
    # Sample gene-specific k_on parameters
    k_on = numpyro.sample(
        "k_on",
        dist.Gamma(k_on_prior[0], k_on_prior[1]).expand([n_genes])
    )

    # Sample gene-specific r_m parameters
    r_m = numpyro.sample(
        "r_m",
        dist.Gamma(r_m_prior[0], r_m_prior[1]).expand([n_genes])
    )

    # Sample gene-specific ratio parameters (r_m/k_off)
    ratio = numpyro.sample(
        "ratio",
        dist.LogNormal(ratio_prior[0], ratio_prior[1]).expand([n_genes])
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
    k_on_prior: tuple = (2, 0.1),
    r_m_prior: tuple = (5, 0.05),
    ratio_prior: tuple = (0, 1),  # Mean and std for LogNormal prior
    counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for the two-state promoter model.
    
    This guide specifies a mean-field variational family where:
        - k_on follows independent Gamma distributions for each gene
        - r_m follows independent Gamma distributions for each gene
        - ratio follows independent LogNormal distributions for each gene
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    k_on_prior : tuple of float
        Parameters (shape, rate) for initializing the Gamma variational
        distribution of k_on
    r_m_prior : tuple of float
        Parameters (shape, rate) for initializing the Gamma variational
        distribution of r_m
    ratio_prior : tuple of float
        Parameters (loc, scale) for initializing the LogNormal variational
        distribution of the ratio
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    batch_size : int, optional
        Mini-batch size for stochastic optimization
    """
    # Register variational parameters for k_on (one per gene)
    alpha_k_on = numpyro.param(
        "alpha_k_on",
        jnp.ones(n_genes) * k_on_prior[0],
        constraint=constraints.positive
    )
    beta_k_on = numpyro.param(
        "beta_k_on",
        jnp.ones(n_genes) * k_on_prior[1],
        constraint=constraints.positive
    )

    # Register variational parameters for r_m (one per gene)
    alpha_r_m = numpyro.param(
        "alpha_r_m",
        jnp.ones(n_genes) * r_m_prior[0],
        constraint=constraints.positive
    )
    beta_r_m = numpyro.param(
        "beta_r_m",
        jnp.ones(n_genes) * r_m_prior[1],
        constraint=constraints.positive
    )

    # Register variational parameters for ratio (one per gene)
    loc_ratio = numpyro.param(
        "loc_ratio",
        jnp.ones(n_genes) * ratio_prior[0]
    )
    scale_ratio = numpyro.param(
        "scale_ratio",
        jnp.ones(n_genes) * ratio_prior[1],
        constraint=constraints.positive
    )

    # Sample from variational distributions
    numpyro.sample("k_on", dist.Gamma(alpha_k_on, beta_k_on))
    numpyro.sample("r_m", dist.Gamma(alpha_r_m, beta_r_m))
    numpyro.sample("ratio", dist.LogNormal(loc_ratio, scale_ratio))
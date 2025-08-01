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
from .twostate_distribution import (
    TwoStatePromoter,
    TwoStatePromoterQuadrature,
    TwoStatePromoterMC,
)

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
    k_on_prior_params = model_config.k_on_param_prior or (5.0, 2.0)
    r_m_prior_params = model_config.r_m_param_prior or (1.0, 1.0)
    ratio_prior_params = model_config.ratio_param_prior or (5.0, 2.0)

    # Sample gene-specific k_on parameters
    k_on = numpyro.sample(
        "k_on", dist.LogNormal(*k_on_prior_params).expand([n_genes])
    )

    # Sample gene-specific r_m parameters
    r_m = numpyro.sample(
        "r_m", dist.LogNormal(*r_m_prior_params).expand([n_genes])
    )

    # Sample gene-specific ratio parameters (r_m/k_off)
    ratio = numpyro.sample(
        "ratio", dist.LogNormal(*ratio_prior_params).expand([n_genes])
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
                    obs=counts,
                )
        else:
            # Define plate for cells with batching
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Likelihood for the counts - one for each cell
                numpyro.sample(
                    "counts",
                    TwoStatePromoter(k_on, k_off, r_m).to_event(1),
                    obs=counts[idx],
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample from two-state promoter distribution
            numpyro.sample(
                "counts", TwoStatePromoter(k_on, k_off, r_m).to_event(1)
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
    k_on_guide_params = model_config.k_on_param_guide or (5.0, 2.0)
    r_m_guide_params = model_config.r_m_param_guide or (1.0, 1.0)
    ratio_guide_params = model_config.ratio_param_guide or (5.0, 2.0)

    # Register variational parameters for k_on (one per gene)
    k_on_loc = numpyro.param(
        "k_on_loc", jnp.full(n_genes, k_on_guide_params[0])
    )
    k_on_scale = numpyro.param(
        "k_on_scale",
        jnp.full(n_genes, k_on_guide_params[1]),
        constraint=constraints.positive,
    )

    # Register variational parameters for r_m (one per gene)
    r_m_loc = numpyro.param("r_m_loc", jnp.full(n_genes, r_m_guide_params[0]))
    r_m_scale = numpyro.param(
        "r_m_scale",
        jnp.full(n_genes, r_m_guide_params[1]),
        constraint=constraints.positive,
    )

    # Register variational parameters for ratio (one per gene)
    ratio_loc = numpyro.param(
        "ratio_loc", jnp.full(n_genes, ratio_guide_params[0])
    )
    ratio_scale = numpyro.param(
        "ratio_scale",
        jnp.full(n_genes, ratio_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample from variational distributions
    numpyro.sample("k_on", dist.LogNormal(k_on_loc, k_on_scale))
    numpyro.sample("r_m", dist.LogNormal(r_m_loc, r_m_scale))
    numpyro.sample("ratio", dist.LogNormal(ratio_loc, ratio_scale))


# ------------------------------------------------------------------------------
# Two-state promoter model with variable capture probability
# ------------------------------------------------------------------------------


def twostate_variable_capture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for two-state promoter with variable mRNA capture probability.

    This model extends the basic two-state promoter model by including
    cell-specific capture probabilities that rescale the transcription rate:
        r_effective = r_m * p_capture

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

    Cell-specific Parameters:
        - p_capture ~ Beta(p_capture_prior) [one per cell]
        - r_effective = r_m * p_capture [cell-gene specific]

    Likelihood:
        counts ~ TwoStatePromoter(k_on, k_off, r_effective)
    """
    # Get prior parameters from model config or use defaults
    k_on_prior_params = model_config.k_on_param_prior or (5.0, 2.0)
    r_m_prior_params = model_config.r_m_param_prior or (1.0, 1.0)
    ratio_prior_params = model_config.ratio_param_prior or (5.0, 2.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample gene-specific k_on parameters
    k_on = numpyro.sample(
        "k_on", dist.LogNormal(*k_on_prior_params).expand([n_genes])
    )

    # Sample gene-specific r_m parameters
    r_m = numpyro.sample(
        "r_m", dist.LogNormal(*r_m_prior_params).expand([n_genes])
    )

    # Sample gene-specific ratio parameters (r_m/k_off)
    ratio = numpyro.sample(
        "ratio", dist.LogNormal(*ratio_prior_params).expand([n_genes])
    )

    # Compute k_off from r_m and ratio
    k_off = numpyro.deterministic("k_off", r_m / ratio)

    # If we have observed data, condition on it
    if counts is not None:
        if batch_size is None:
            # No batching: sample p_capture for all cells, then sample counts
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture_reshaped = p_capture[:, None]
                # Compute effective transcription rate
                r_effective = numpyro.deterministic(
                    "r_effective", r_m * p_capture_reshaped
                )
                # Sample observed counts
                numpyro.sample(
                    "counts",
                    TwoStatePromoter(k_on, k_off, r_effective).to_event(1),
                    obs=counts,
                )
        else:
            # With batching: sample p_capture and counts for a batch of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probability for the batch
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to (batch_size, n_genes)
                p_capture_reshaped = p_capture[:, None]
                # Compute effective transcription rate
                r_effective = numpyro.deterministic(
                    "r_effective", r_m * p_capture_reshaped
                )
                # Sample observed counts for the batch
                numpyro.sample(
                    "counts",
                    TwoStatePromoter(k_on, k_off, r_effective).to_event(1),
                    obs=counts[idx],
                )
    else:
        # No observed counts: sample latent counts for all cells
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probability
            p_capture = numpyro.sample(
                "p_capture", dist.Beta(*p_capture_prior_params)
            )
            # Reshape p_capture for broadcasting to (n_cells, n_genes)
            p_capture_reshaped = p_capture[:, None]
            # Compute effective transcription rate
            r_effective = numpyro.deterministic(
                "r_effective", r_m * p_capture_reshaped
            )
            # Sample latent counts
            numpyro.sample(
                "counts", TwoStatePromoter(k_on, k_off, r_effective).to_event(1)
            )


# ------------------------------------------------------------------------------
# Two-state promoter variational guide with variable capture
# ------------------------------------------------------------------------------


def twostate_variable_capture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for the two-state promoter model with
    variable capture probability.

    This guide specifies a mean-field variational family where:
        - k_on follows independent LogNormal distributions for each gene
        - r_m follows independent LogNormal distributions for each gene
        - ratio follows independent LogNormal distributions for each gene
        - p_capture follows independent Beta distributions for each cell

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
    k_on_guide_params = model_config.k_on_param_guide or (5.0, 2.0)
    r_m_guide_params = model_config.r_m_param_guide or (1.0, 1.0)
    ratio_guide_params = model_config.ratio_param_guide or (5.0, 2.0)
    p_capture_guide_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for k_on (one per gene)
    k_on_loc = numpyro.param(
        "k_on_loc", jnp.full(n_genes, k_on_guide_params[0])
    )
    k_on_scale = numpyro.param(
        "k_on_scale",
        jnp.full(n_genes, k_on_guide_params[1]),
        constraint=constraints.positive,
    )

    # Register variational parameters for r_m (one per gene)
    r_m_loc = numpyro.param("r_m_loc", jnp.full(n_genes, r_m_guide_params[0]))
    r_m_scale = numpyro.param(
        "r_m_scale",
        jnp.full(n_genes, r_m_guide_params[1]),
        constraint=constraints.positive,
    )

    # Register variational parameters for ratio (one per gene)
    ratio_loc = numpyro.param(
        "ratio_loc", jnp.full(n_genes, ratio_guide_params[0])
    )
    ratio_scale = numpyro.param(
        "ratio_scale",
        jnp.full(n_genes, ratio_guide_params[1]),
        constraint=constraints.positive,
    )

    # Set up cell-specific capture probability parameters
    p_capture_alpha = numpyro.param(
        "p_capture_alpha",
        jnp.full(n_cells, p_capture_guide_params[0]),
        constraint=constraints.positive,
    )
    p_capture_beta = numpyro.param(
        "p_capture_beta",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample from variational distributions
    numpyro.sample("k_on", dist.LogNormal(k_on_loc, k_on_scale))
    numpyro.sample("r_m", dist.LogNormal(r_m_loc, r_m_scale))
    numpyro.sample("ratio", dist.LogNormal(ratio_loc, ratio_scale))

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture",
                dist.Beta(p_capture_alpha[idx], p_capture_beta[idx]),
            )


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

    # p_capture parameter (Beta distribution) - for variable capture model
    if "p_capture_alpha" in params and "p_capture_beta" in params:
        if split and len(params["p_capture_alpha"].shape) == 1:
            # Cell-specific p_capture parameters
            distributions["p_capture"] = [
                dist.Beta(
                    params["p_capture_alpha"][c], params["p_capture_beta"][c]
                )
                for c in range(params["p_capture_alpha"].shape[0])
            ]
        else:
            distributions["p_capture"] = dist.Beta(
                params["p_capture_alpha"], params["p_capture_beta"]
            )

    return distributions


# ------------------------------------------------------------------------------
# Hierarchical Two-state promoter model (avoids ₁F₁)
# ------------------------------------------------------------------------------


def twostate_hierarchical_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Hierarchical Numpyro model for two-state promoter data that avoids ₁F₁.

    Instead of using the analytical TwoStatePromoter distribution, this model
    uses the hierarchical structure:

        1. Sample biological parameters: k_on, r_m, ratio (same as before)
        2. Convert to Beta-Poisson parameters: α=k_on, β=k_off, d=r_m
        3. For each cell, sample latent rates: p ~ Beta(α, β)
        4. Sample counts: x ~ Poisson(d * p)

    This replaces analytical marginalization with Monte Carlo marginalization
    through SVI, avoiding the numerical issues with ₁F₁.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing prior parameters and model settings
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.
    """
    # Get prior parameters from model config or use defaults
    k_on_prior_params = model_config.k_on_param_prior or (5.0, 2.0)
    r_m_prior_params = model_config.r_m_param_prior or (1.0, 1.0)
    ratio_prior_params = model_config.ratio_param_prior or (5.0, 2.0)

    # Sample gene-specific biological parameters (same as original model)
    k_on = numpyro.sample(
        "k_on", dist.LogNormal(*k_on_prior_params).expand([n_genes])
    )

    r_m = numpyro.sample(
        "r_m", dist.LogNormal(*r_m_prior_params).expand([n_genes])
    )

    ratio = numpyro.sample(
        "ratio", dist.LogNormal(*ratio_prior_params).expand([n_genes])
    )

    # Compute derived parameters
    k_off = numpyro.deterministic("k_off", r_m / ratio)

    # Hierarchical sampling structure
    if counts is not None:
        if batch_size is None:
            # Full dataset
            with numpyro.plate("cells", n_cells):
                # For each cell, sample latent rate parameters p ~ Beta(α, β)
                # Shape: (n_genes,) per cell, so full shape is (n_cells, n_genes)
                p = numpyro.sample("p", dist.Beta(k_on, k_off).to_event(1))

                # Sample observed counts: x ~ Poisson(dose * p)
                numpyro.sample("counts", dist.Poisson(r_m * p).to_event(1), obs=counts)
        else:
            # Mini-batch
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample latent rates for the batch
                p = numpyro.sample("p", dist.Beta(k_on, k_off).to_event(1))

                # Sample observed counts for the batch
                numpyro.sample(
                    "counts", dist.Poisson(r_m * p).to_event(1), obs=counts[idx]
                )
    else:
        # Predictive sampling (no observations)
        with numpyro.plate("cells", n_cells):
            p = numpyro.sample("p", dist.Beta(k_on, k_off).to_event(1))
            numpyro.sample("counts", dist.Poisson(r_m * p).to_event(1))


# ------------------------------------------------------------------------------
# Hierarchical Two-state promoter variational guide
# ------------------------------------------------------------------------------


def twostate_hierarchical_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Variational guide for the hierarchical two-state promoter model.

    This guide parameterizes:
    1. Global parameters: k_on, r_m, ratio (same as original)
    2. Local parameters: p (latent rates for each cell and gene)

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
    k_on_guide_params = model_config.k_on_param_guide or (5.0, 2.0)
    r_m_guide_params = model_config.r_m_param_guide or (1.0, 1.0)
    ratio_guide_params = model_config.ratio_param_guide or (5.0, 2.0)

    # -------------------------------------------------------------------------
    # Global parameters (same as original guide)
    # -------------------------------------------------------------------------

    # Variational parameters for k_on
    k_on_loc = numpyro.param(
        "k_on_loc", jnp.full(n_genes, k_on_guide_params[0])
    )
    k_on_scale = numpyro.param(
        "k_on_scale",
        jnp.full(n_genes, k_on_guide_params[1]),
        constraint=constraints.positive,
    )

    # Variational parameters for r_m
    r_m_loc = numpyro.param("r_m_loc", jnp.full(n_genes, r_m_guide_params[0]))
    r_m_scale = numpyro.param(
        "r_m_scale",
        jnp.full(n_genes, r_m_guide_params[1]),
        constraint=constraints.positive,
    )

    # Variational parameters for ratio
    ratio_loc = numpyro.param(
        "ratio_loc", jnp.full(n_genes, ratio_guide_params[0])
    )
    ratio_scale = numpyro.param(
        "ratio_scale",
        jnp.full(n_genes, ratio_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample global parameters
    k_on = numpyro.sample("k_on", dist.LogNormal(k_on_loc, k_on_scale))
    r_m = numpyro.sample("r_m", dist.LogNormal(r_m_loc, r_m_scale))
    ratio = numpyro.sample("ratio", dist.LogNormal(ratio_loc, ratio_scale))

    # Compute derived parameters for the latent variable guide
    k_off = r_m / ratio

    # -------------------------------------------------------------------------
    # Local latent parameters p (the key addition)
    # -------------------------------------------------------------------------

    if batch_size is None:
        # Full dataset: sample latent rates for all cells
        with numpyro.plate("cells", n_cells):
            # Sample latent rates for this cell using the same parameters as the model
            numpyro.sample("p", dist.Beta(k_on, k_off).to_event(1))

    else:
        # Mini-batch: sample latent rates for the batch
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            # Sample latent rates for this cell using the same parameters as the model
            numpyro.sample("p", dist.Beta(k_on, k_off).to_event(1))

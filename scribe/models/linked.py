"""
Linked parameterization models for single-cell RNA sequencing data.

This parameterization differes from the standard parameterization in that
the mean parameter (mu) is linked to the success probability parameter (p)
through the relationship:
    r = mu * (1 - p) / p
where r is the dispersion parameter.
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

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------


def nbdm_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Negative Binomial-Dirichlet Multinomial data.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)

    # Sample parameters
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_genes])
    )
    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # Define base distribution
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            # Without batching: sample counts for all cells
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # With batching: sample counts for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------


def nbdm_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the NBDM model.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)

    # Register p_alpha as a variational parameter with positivity constraint
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    # Register p_beta as a variational parameter with positivity constraint
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    # Sample p from the Beta distribution parameterized by p_alpha and p_beta
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register mu_loc as a variational parameter with positivity constraint
    mu_loc = numpyro.param("mu_loc", jnp.full(n_genes, mu_prior_params[0]))
    # Register mu_scale as a variational parameter with positivity constraint
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full(n_genes, mu_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample mu from the LogNormal distribution parameterized by mu_loc and
    # mu_scale, with event dimension 1
    numpyro.sample("mu", dist.LogNormal(mu_loc, mu_scale))


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
    Numpyro model for Zero-Inflated Negative Binomial data.
    """
    # Get prior parameters for p (success probability), r (dispersion), and gate
    # (zero-inflation) from model_config, or use defaults if not provided
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample p from a Beta distribution with the specified prior parameters
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample mu from a LogNormal distribution with the specified prior
    # parameters, expanded to n_genes
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_genes])
    )
    # Sample gate from a Beta distribution with the specified prior parameters,
    # expanded to n_genes
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_genes])
    )

    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # Construct the base Negative Binomial distribution using r and p
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Construct the zero-inflated distribution using the base NB and gate, and
    # set event dimension to 1
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)

    # If observed counts are provided
    if counts is not None:
        # If no batching, use a plate over all cells
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample observed counts from the zero-inflated NB distribution
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # If batching, use a plate with subsampling and get indices
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample observed counts for the batch indices
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # If no observed counts, just sample from the prior predictive
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------


def zinb_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the ZINB model.
    """
    # Define guide parameters for p, r, and gate
    # Get initial values for p's Beta distribution parameters (alpha, beta)
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    # Get initial values for mu's LogNormal distribution parameters (loc, scale)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)
    # Get initial values for gate's Beta distribution parameters (alpha, beta)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    # Sample p from the variational Beta distribution
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register variational parameters for r (dispersion)
    mu_loc = numpyro.param("mu_loc", jnp.full(n_genes, mu_prior_params[0]))
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full(n_genes, mu_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample mu from the variational LogNormal distribution (vectorized over
    # genes)
    numpyro.sample("mu", dist.LogNormal(mu_loc, mu_scale).to_event(1))

    # Register variational parameters for gate (zero-inflation probability)
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full(n_genes, gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full(n_genes, gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample gate from the variational Beta distribution (vectorized over genes)
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta).to_event(1))


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
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample global success probability p from Beta prior
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample gene-specific dispersion r from LogNormal prior (vectorized over
    # genes)
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_genes])
    )
    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # If observed counts are provided, use them as observations
    if counts is not None:
        if batch_size is None:
            # No batching: sample p_capture for all cells, then sample counts
            # for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability from Beta prior
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability p_hat for each cell/gene
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Sample observed counts from Negative Binomial with p_hat
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1),
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
                # Compute effective success probability p_hat for each cell/gene
                # in the batch
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Sample observed counts for the batch from Negative Binomial
                # with p_hat
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1),
                    obs=counts[idx],
                )
    else:
        # No observed counts: sample latent counts for all cells
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probability from Beta prior
            p_capture = numpyro.sample(
                "p_capture", dist.Beta(*p_capture_prior_params)
            )
            # Reshape p_capture for broadcasting to (n_cells, n_genes)
            p_capture_reshaped = p_capture[:, None]
            # Compute effective success probability p_hat for each cell/gene
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Sample latent counts from Negative Binomial with p_hat
            numpyro.sample(
                "counts",
                dist.NegativeBinomialProbs(r, p_hat).to_event(1),
            )


# ------------------------------------------------------------------------------


def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the NBVCP model.
    """
    # Define guide parameters for p, r, and p_capture
    # Get initial values for p's Beta distribution parameters (alpha, beta)
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    # Get initial values for mu's LogNormal distribution parameters (loc, scale)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)
    # Get initial values for p_capture's Beta distribution parameters (alpha,
    # beta)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register variational parameters for r (dispersion)
    mu_loc = numpyro.param("mu_loc", jnp.full(n_genes, mu_prior_params[0]))
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full(n_genes, mu_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("mu", dist.LogNormal(mu_loc, mu_scale).to_event(1))

    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_alpha = numpyro.param(
            "p_capture_alpha",
            jnp.full(n_cells, p_capture_prior_params[0]),
            constraint=constraints.positive,
        )
        p_capture_beta = numpyro.param(
            "p_capture_beta",
            jnp.full(n_cells, p_capture_prior_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_capture", dist.Beta(p_capture_alpha, p_capture_beta))


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
    Numpyro model for ZINB with variable mRNA capture probability.
    """
    # Get prior parameters from model_config, or use defaults if not provided
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample global success probability p (Beta prior)
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample gene-specific dispersion r (Gamma prior)
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_genes])
    )
    # Sample gene-specific zero-inflation gate (Beta prior)
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_genes])
    )

    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    if counts is not None:
        # If observed counts are provided
        if batch_size is None:
            # No batching: sample for all cells
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probability (Beta prior)
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to genes
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability for each cell/gene
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Define base Negative Binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Define zero-inflated NB distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Observe counts
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # With batching: sample for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific capture probability (Beta prior)
                p_capture = numpyro.sample(
                    "p_capture", dist.Beta(*p_capture_prior_params)
                )
                # Reshape p_capture for broadcasting to genes
                p_capture_reshaped = p_capture[:, None]
                # Compute effective success probability for each cell/gene
                p_hat = (
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                # Define base Negative Binomial distribution
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                # Define zero-inflated NB distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Observe counts for the batch
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # No observed counts: just define the generative process
        with numpyro.plate("cells", n_cells):
            # Sample cell-specific capture probability (Beta prior)
            p_capture = numpyro.sample(
                "p_capture", dist.Beta(*p_capture_prior_params)
            )
            # Reshape p_capture for broadcasting to genes
            p_capture_reshaped = p_capture[:, None]
            # Compute effective success probability for each cell/gene
            p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            # Define base Negative Binomial distribution
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            # Define zero-inflated NB distribution
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Sample counts (not observed)
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------


def zinbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the ZINBVCP model.
    """
    # Define guide parameters for p, r, and gate
    # Get initial values for p's Beta distribution parameters (alpha, beta)
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    # Get initial values for mu's LogNormal distribution parameters (loc, scale)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)
    # Get initial values for gate's Beta distribution parameters (alpha, beta)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)
    # Get initial values for p_capture's Beta distribution parameters (alpha,
    # beta)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    # Register variational parameters for r (dispersion)
    mu_loc = numpyro.param("mu_loc", jnp.full(n_genes, mu_prior_params[0]))
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full(n_genes, mu_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("mu", dist.LogNormal(mu_loc, mu_scale).to_event(1))

    # Register variational parameters for gate (zero-inflation probability)
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full(n_genes, gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full(n_genes, gate_prior_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta).to_event(1))

    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_alpha = numpyro.param(
            "p_capture_alpha",
            jnp.full(n_cells, p_capture_prior_params[0]),
            constraint=constraints.positive,
        )
        p_capture_beta = numpyro.param(
            "p_capture_beta",
            jnp.full(n_cells, p_capture_prior_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_capture", dist.Beta(p_capture_alpha, p_capture_beta))


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
    Numpyro mixture model for NBDM data.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components

    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)

    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    # Define the categorical distribution for component assignment
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_components, n_genes])
    )
    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]

    else:
        # All components share p, but have their own r
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Define the base distribution for each component (Negative Binomial)
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
    # Create the mixture distribution over components
    mixture = dist.MixtureSameFamily(mixing_dist, base_dist)

    # Sample observed or latent counts for each cell
    if counts is not None:
        if batch_size is None:
            # No batching: sample for all cells
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture, obs=counts)
        else:
            # With batching: sample for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        # No observed counts: sample latent counts for all cells
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture)


# ------------------------------------------------------------------------------


def nbdm_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the NBDM mixture model.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components

    # Get prior parameters for the mixture weights, p, and r
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)

    # Register variational parameters for the mixture weights
    mixing_conc = numpyro.param(
        "mixing_concentrations",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    # Get prior parameters for p and r
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)

    # Define parameters for r
    mu_loc = numpyro.param(
        "mu_loc",
        jnp.full((n_components, n_genes), mu_prior_params[0]),
        constraint=constraints.positive,
    )
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full((n_components, n_genes), mu_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample(
        "mu", dist.LogNormal(mu_loc, mu_scale).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))


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
    Numpyro mixture model for ZINB data.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)
    # Get prior parameters for p, r, and gate
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_components, n_genes])
    )
    # Sample the gene-specific gate from a Beta prior
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]

    else:
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # Define the base distribution for each component (Negative Binomial)
    base_dist = dist.NegativeBinomialProbs(r, p)
    # Create the zero-inflated distribution over components
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)
    # Create the mixture distribution over components
    mixture = dist.MixtureSameFamily(mixing_dist, zinb)

    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", mixture, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture)


# ------------------------------------------------------------------------------


def zinb_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the ZINB mixture model.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)
    mixing_conc = numpyro.param(
        "mixing_conc",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    # Get prior parameters for p, r, and gate
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Define parameters for mu
    mu_loc = numpyro.param(
        "mu_loc",
        jnp.full((n_components, n_genes), mu_prior_params[0]),
        constraint=constraints.positive,
    )
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full((n_components, n_genes), mu_prior_params[1]),
        constraint=constraints.positive,
    )

    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample(
        "mu", dist.LogNormal(mu_loc, mu_scale).expand([n_components, n_genes])
    )

    # Define parameters for gate
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full((n_components, n_genes), gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full((n_components, n_genes), gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific gate from a Beta prior
    numpyro.sample(
        "gate", dist.Beta(gate_alpha, gate_beta).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))


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
    Numpyro mixture model for NBVCP data.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)
    # Get prior parameters for p, r, and p_capture
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    # Define the categorical distribution for component assignment
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]
    else:
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    # Define plate context for sampling
    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )
    with plate_context as idx:
        # Sample cell-specific capture probability
        p_capture = numpyro.sample(
            "p_capture", dist.Beta(*p_capture_prior_params)
        )
        # Reshape p_capture for broadcasting with components
        p_capture_reshaped = p_capture[:, None, None]
        # Compute p_hat using the derived formula
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )

        # Define the base distribution for each component (Negative Binomial)
        base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
        # Create the mixture distribution over components
        mixture = dist.MixtureSameFamily(mixing_dist, base_dist)
        # Define observation context for sampling
        obs = counts[idx] if counts is not None else None
        # Sample the counts from the mixture distribution
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


def nbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the NBVCP mixture model.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)
    # Register variational parameters for the mixture weights
    mixing_conc = numpyro.param(
        "mixing_concentrations",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    # Sample the mixture weights from a Dirichlet prior
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    # Get prior parameters for p, r, and p_capture
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Define parameters for mu
    mu_loc = numpyro.param(
        "mu_loc", jnp.full((n_components, n_genes), mu_prior_params[0])
    )
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full((n_components, n_genes), mu_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample("mu", dist.LogNormal(mu_loc, mu_scale))

    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_alpha = numpyro.param(
            "p_capture_alpha",
            p_capture_prior_params[0],
            constraint=constraints.positive,
        )
        p_capture_beta = numpyro.param(
            "p_capture_beta",
            p_capture_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_capture", dist.Beta(p_capture_alpha, p_capture_beta))


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
    Numpyro mixture model for ZINBVCP data.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_prior is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_prior)
    # Get prior parameters for p, r, gate, and p_capture
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample the mixture weights from a Dirichlet prior
    mixing_probs = numpyro.sample(
        "mixing_weights", dist.Dirichlet(mixing_prior_params)
    )
    mixing_dist = dist.Categorical(probs=mixing_probs)

    # Sample the gene-specific dispersion r from a LogNormal prior
    mu = numpyro.sample(
        "mu", dist.LogNormal(*mu_prior_params).expand([n_components, n_genes])
    )
    # Sample the gene-specific gate from a Beta prior
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_components, n_genes])
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p = numpyro.sample(
            "p", dist.Beta(*p_prior_params).expand([n_components])
        )
        # Broadcast p to have the right shape
        p = p[:, None]
    else:
        p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Compute r
    r = numpyro.deterministic("r", mu * (1 - p) / p)

    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )
    with plate_context as idx:
        # Sample cell-specific capture probability
        p_capture = numpyro.sample(
            "p_capture", dist.Beta(*p_capture_prior_params)
        )
        # Reshape p_capture for broadcasting with components
        p_capture_reshaped = p_capture[:, None, None]
        # Compute p_hat using the derived formula
        p_hat = p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))

        # Define the base distribution for each component (Negative Binomial)
        base_dist = dist.NegativeBinomialProbs(r, p_hat)
        # Create the zero-inflated distribution over components
        zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)
        # Create the mixture distribution over components
        mixture = dist.MixtureSameFamily(mixing_dist, zinb)
        # Define observation context for sampling
        obs = counts[idx] if counts is not None else None
        # Sample the counts from the mixture distribution
        numpyro.sample("counts", mixture, obs=obs)


# ------------------------------------------------------------------------------


def zinbvcp_mixture_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the ZINBVCP mixture model.
    """
    n_components = model_config.n_components
    # Get prior parameters for the mixture weights
    if model_config.mixing_param_guide is None:
        mixing_prior_params = jnp.ones(n_components)
    else:
        mixing_prior_params = jnp.array(model_config.mixing_param_guide)
    # Get prior parameters for p, r, gate, and p_capture
    mixing_conc = numpyro.param(
        "mixing_conc",
        mixing_prior_params,
        constraint=constraints.positive,
    )
    numpyro.sample("mixing_weights", dist.Dirichlet(mixing_conc))

    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    mu_prior_params = model_config.mu_param_guide or (0.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

    # Define parameters for mu
    mu_loc = numpyro.param(
        "mu_loc", jnp.full((n_components, n_genes), mu_prior_params[0])
    )
    mu_scale = numpyro.param(
        "mu_scale",
        jnp.full((n_components, n_genes), mu_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific dispersion r from a LogNormal prior
    numpyro.sample("mu", dist.LogNormal(mu_loc, mu_scale))

    # Define parameters for gate
    gate_alpha = numpyro.param(
        "gate_alpha",
        jnp.full((n_components, n_genes), gate_prior_params[0]),
        constraint=constraints.positive,
    )
    gate_beta = numpyro.param(
        "gate_beta",
        jnp.full((n_components, n_genes), gate_prior_params[1]),
        constraint=constraints.positive,
    )
    # Sample the gene-specific gate from a Beta prior
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta))

    # Define parameters for p
    if model_config.component_specific_params:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            jnp.full(n_components, p_prior_params[0]),
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            jnp.full(n_components, p_prior_params[1]),
            constraint=constraints.positive,
        )
        # Each component has its own p
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    else:
        # Define parameters for p
        p_alpha = numpyro.param(
            "p_alpha",
            p_prior_params[0],
            constraint=constraints.positive,
        )
        p_beta = numpyro.param(
            "p_beta",
            p_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p", dist.Beta(p_alpha, p_beta))

    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_alpha = numpyro.param(
            "p_capture_alpha",
            p_capture_prior_params[0],
            constraint=constraints.positive,
        )
        p_capture_beta = numpyro.param(
            "p_capture_beta",
            p_capture_prior_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_capture", dist.Beta(p_capture_alpha, p_capture_beta))


# ------------------------------------------------------------------------------
# Get posterior distributions for SVI results
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray], model_config: ModelConfig
) -> Dict[str, dist.Distribution]:
    """
    Constructs and returns a dictionary of posterior distributions from
    estimated parameters.

    This function is specific to the 'standard' parameterization and builds the
    appropriate `numpyro` distributions based on the guide parameters found in
    the `params` dictionary. It handles both single and mixture models.

    Args:
        params: A dictionary of estimated parameters from the variational guide.
        model_config: The model configuration object.

    Returns:
        A dictionary mapping parameter names to their posterior distributions.
    """
    distributions = {}
    n_components = model_config.n_components
    component_specific = model_config.component_specific_params

    # Single models (or shared params in mixture models)
    if "p_alpha" in params and "p_beta" in params:
        distributions["p"] = dist.Beta(params["p_alpha"], params["p_beta"])

    if "mu_loc" in params and "mu_scale" in params:
        distributions["mu"] = dist.LogNormal(
            params["mu_loc"], params["mu_scale"]
        )

    if "gate_alpha" in params and "gate_beta" in params:
        # For ZINB, gate can be shared or component-specific
        if params["gate_alpha"].ndim > 0:
            distributions["gate"] = dist.Beta(
                params["gate_alpha"], params["gate_beta"]
            )
        else:
            distributions["gate"] = dist.Beta(
                params["gate_alpha"], params["gate_beta"]
            )

    if "p_capture_alpha" in params and "p_capture_beta" in params:
        distributions["p_capture"] = dist.Beta(
            params["p_capture_alpha"], params["p_capture_beta"]
        )

    # Mixture-specific parameters
    if n_components is not None:
        if "mixing_concentrations" in params:
            distributions["mixing_weights"] = dist.Dirichlet(
                params["mixing_concentrations"]
            )

    return distributions

"""
Unconstrained parameterization models for single-cell RNA sequencing data.
Uses Normal distributions on transformed parameters for MCMC inference.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import softmax

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
    Numpyro unconstrained model for Negative Binomial-Dirichlet Multinomial data.
    Parameters are sampled in unconstrained space using Normal distributions.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained", dist.Normal(*r_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Define base distribution
    base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
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
    Mean-field variational guide for the unconstrained NBDM model.
    """
    # Define guide parameters with proper defaults
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))


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
    Numpyro unconstrained model for Zero-Inflated Negative Binomial data.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained", dist.Normal(*r_prior_params).expand([n_genes])
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define distributions
    base_dist = dist.NegativeBinomialProbs(r, p)
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", zinb, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
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
    Mean-field variational guide for the unconstrained ZINB model.
    """
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register unconstrained r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))

    # Register unconstrained gate parameters
    gate_loc = numpyro.param(
        "gate_unconstrained_loc", jnp.full(n_genes, gate_guide_params[0])
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full(n_genes, gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))


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
    Numpyro unconstrained model for Negative Binomial with variable capture probability.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained", dist.Normal(*r_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                # Define distribution and sample
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
            )

            # Reshape p_capture for broadcasting to genes
            # Shape: (batch_size, 1)
            p_capture_reshaped = p_capture[:, None]

            # Compute effective probability
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
            )

            base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------


def nbvcp_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the unconstrained NBVCP model.
    """
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global unconstrained parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
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
    Numpyro unconstrained model for Zero-Inflated Negative Binomial with variable capture probability.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    r_unconstrained = numpyro.sample(
        "r_unconstrained", dist.Normal(*r_prior_params).expand([n_genes])
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Sample counts
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific capture probabilities
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                # Define distribution and sample
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                numpyro.sample("counts", zinb, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )

                # Reshape p_capture for broadcasting to genes
                # Shape: (batch_size, 1)
                p_capture_reshaped = p_capture[:, None]

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
                )

                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
            )

            # Reshape p_capture for broadcasting to genes
            # Shape: (batch_size, 1)
            p_capture_reshaped = p_capture[:, None]

            # Compute effective probability
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped)),
            )

            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
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
    Mean-field variational guide for the unconstrained ZINBVCP model.
    """
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global unconstrained parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    r_loc = numpyro.param(
        "r_unconstrained_loc", jnp.full(n_genes, r_guide_params[0])
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full(n_genes, r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))

    gate_loc = numpyro.param(
        "gate_unconstrained_loc", jnp.full(n_genes, gate_guide_params[0])
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full(n_genes, gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
            )


# ------------------------------------------------------------------------------
# Mixture Models
# ------------------------------------------------------------------------------


def nbdm_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Negative Binomial single-cell RNA
    sequencing data. Parameters are sampled in unconstrained space.
    """
    n_components = model_config.n_components

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Define mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    # Sample unconstrained r for each component and gene
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    # Deterministic transformations to constrained space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Define component distribution using logits parameterization
    base_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained
    ).to_event(1)

    # Create mixture distribution
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
    Mean-field variational guide for the unconstrained NBDM mixture model.
    """
    # Get the number of mixture components from the model configuration
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)

    # Register mixing weights parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))


# ------------------------------------------------------------------------------


def zinb_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Zero-Inflated Negative Binomial data.
    """
    n_components = model_config.n_components

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    if model_config.gate_unconstrained_prior is None:
        gate_prior_params = (0.0, 1.0)
    else:
        gate_prior_params = model_config.gate_unconstrained_prior

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Transform to constrained space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define distributions
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    base_nb_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained
    )

    zinb_comp_dist = dist.ZeroInflatedDistribution(
        base_nb_dist, gate_logits=gate_unconstrained
    ).to_event(1)

    mixture = dist.MixtureSameFamily(mixing_dist, zinb_comp_dist)

    # Model likelihood
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
    Mean-field variational guide for the unconstrained ZINB mixture model.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register mixing weights parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    if model_config.component_specific_params:
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))

    # Register gate parameters
    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full((n_components, n_genes), gate_guide_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full((n_components, n_genes), gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))


# ------------------------------------------------------------------------------


def nbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Negative Binomial with variable capture probability.
    """
    n_components = model_config.n_components

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    if model_config.p_capture_unconstrained_prior is None:
        p_capture_prior_params = (0.0, 1.0)
    else:
        p_capture_prior_params = model_config.p_capture_unconstrained_prior

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )

    # Define global mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample r unconstrained
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

    # Define plate context for sampling
    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )

    with plate_context as idx:
        # Sample unconstrained cell-specific capture probability
        p_capture_unconstrained = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )
        # Convert to constrained space
        p_capture = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained)
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
    Mean-field variational guide for the unconstrained NBVCP mixture model.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))

    if model_config.component_specific_params:
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
            )


# ------------------------------------------------------------------------------


def zinbvcp_mixture_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Zero-Inflated Negative Binomial
    with variable capture probability.
    """
    n_components = model_config.n_components

    # Define prior parameters - handle None values properly
    if model_config.mixing_logits_unconstrained_prior is None:
        mixing_prior_params = (0.0, 1.0)
    else:
        mixing_prior_params = model_config.mixing_logits_unconstrained_prior

    if model_config.p_unconstrained_prior is None:
        p_prior_params = (0.0, 1.0)
    else:
        p_prior_params = model_config.p_unconstrained_prior

    if model_config.r_unconstrained_prior is None:
        r_prior_params = (0.0, 1.0)
    else:
        r_prior_params = model_config.r_unconstrained_prior

    if model_config.gate_unconstrained_prior is None:
        gate_prior_params = (0.0, 1.0)
    else:
        gate_prior_params = model_config.gate_unconstrained_prior

    if model_config.p_capture_unconstrained_prior is None:
        p_capture_prior_params = (0.0, 1.0)
    else:
        p_capture_prior_params = model_config.p_capture_unconstrained_prior

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    # Define global mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Sample r unconstrained
    r_unconstrained = numpyro.sample(
        "r_unconstrained",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    # Sample gate unconstrained
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Sample component-specific or shared parameters depending on config
    if model_config.component_specific_params:
        # Each component has its own p
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params).expand([n_components]),
        )
        p_unconstrained = p_unconstrained[:, None]
    else:
        # All components share p, but have their own r
        p_unconstrained = numpyro.sample(
            "p_unconstrained",
            dist.Normal(*p_prior_params),
        )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Define plate context for sampling
    plate_context = (
        numpyro.plate("cells", n_cells, subsample_size=batch_size)
        if counts is not None and batch_size is not None
        else numpyro.plate("cells", n_cells)
    )

    with plate_context as idx:
        # Sample unconstrained cell-specific capture probability
        p_capture_unconstrained = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )
        # Convert to constrained space
        p_capture = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained)
        )
        # Reshape p_capture for broadcasting with components
        p_capture_reshaped = p_capture[:, None, None]
        # Compute p_hat using the derived formula
        p_hat = numpyro.deterministic(
            "p_hat", p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
        )
        # Define the base distribution for each component (Negative Binomial)
        base_dist = dist.NegativeBinomialProbs(r, p_hat)

        zinb_base_dist = dist.ZeroInflatedDistribution(
            base_dist,
            gate_logits=gate_unconstrained[None, :, :],
        ).to_event(1)
        # Create the mixture distribution over components
        mixture = dist.MixtureSameFamily(mixing_dist, zinb_base_dist)
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
    Mean-field variational guide for the unconstrained ZINBVCP mixture model.
    """
    n_components = model_config.n_components

    # Define guide parameters
    mixing_guide_params = model_config.mixing_logits_unconstrained_guide or (
        0.0,
        1.0,
    )
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    r_guide_params = model_config.r_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)
    p_capture_guide_params = model_config.p_capture_unconstrained_guide or (
        0.0,
        1.0,
    )

    # Register global parameters
    mixing_loc = numpyro.param(
        "mixing_logits_unconstrained_loc",
        jnp.full(n_components, mixing_guide_params[0]),
    )
    mixing_scale = numpyro.param(
        "mixing_logits_unconstrained_scale",
        jnp.full(n_components, mixing_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(mixing_loc, mixing_scale),
    )

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("r_unconstrained", dist.Normal(r_loc, r_scale))

    # Register gate parameters
    gate_loc = numpyro.param(
        "gate_unconstrained_loc",
        jnp.full((n_components, n_genes), gate_guide_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_scale",
        jnp.full((n_components, n_genes), gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample("gate_unconstrained", dist.Normal(gate_loc, gate_scale))

    if model_config.component_specific_params:
        # Each component has its own p
        p_loc = numpyro.param(
            "p_unconstrained_loc", jnp.full(n_components, p_guide_params[0])
        )
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            jnp.full(n_components, p_guide_params[1]),
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))
    else:
        # All components share p, but have their own r
        p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
        p_scale = numpyro.param(
            "p_unconstrained_scale",
            p_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Set up cell-specific capture probability parameters
    p_capture_loc = numpyro.param(
        "p_capture_unconstrained_loc",
        jnp.full(n_cells, p_capture_guide_params[0]),
    )
    p_capture_scale = numpyro.param(
        "p_capture_unconstrained_scale",
        jnp.full(n_cells, p_capture_guide_params[1]),
        constraint=constraints.positive,
    )

    # Sample p_capture depending on batch size
    if batch_size is None:
        with numpyro.plate("cells", n_cells):
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc, p_capture_scale),
            )
    else:
        with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
            numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(p_capture_loc[idx], p_capture_scale[idx]),
            )


# ------------------------------------------------------------------------------
# Utility function
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    split: bool = False,
) -> Dict[str, dist.Distribution]:
    """
    Constructs and returns a dictionary of posterior distributions from
    estimated parameters.

    This function is specific to the 'unconstrained' parameterization and builds the
    appropriate `numpyro` distributions based on the guide parameters found in
    the `params` dictionary. It handles both single and mixture models.

    Args:
        params: A dictionary of estimated parameters from the variational guide.
        model_config: The model configuration object.
        split: If True, returns lists of individual distributions for
        multidimensional parameters instead of batch distributions.

    Returns:
        A dictionary mapping parameter names to their posterior distributions.
    """
    distributions = {}

    # p_unconstrained parameter (Normal distribution)
    if "p_unconstrained_loc" in params and "p_unconstrained_scale" in params:
        if split and model_config.component_specific_params:
            # Component-specific p_unconstrained parameters
            distributions["p_unconstrained"] = [
                dist.Normal(
                    params["p_unconstrained_loc"][i],
                    params["p_unconstrained_scale"][i],
                )
                for i in range(params["p_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["p_unconstrained"] = dist.Normal(
                params["p_unconstrained_loc"], params["p_unconstrained_scale"]
            )

    # r_unconstrained parameter (Normal distribution)
    if "r_unconstrained_loc" in params and "r_unconstrained_scale" in params:
        if split and len(params["r_unconstrained_loc"].shape) == 1:
            # Gene-specific r_unconstrained parameters
            distributions["r_unconstrained"] = [
                dist.Normal(
                    params["r_unconstrained_loc"][c],
                    params["r_unconstrained_scale"][c],
                )
                for c in range(params["r_unconstrained_loc"].shape[0])
            ]
        elif split and len(params["r_unconstrained_loc"].shape) == 2:
            # Component and gene-specific r_unconstrained parameters
            distributions["r_unconstrained"] = [
                [
                    dist.Normal(
                        params["r_unconstrained_loc"][c, g],
                        params["r_unconstrained_scale"][c, g],
                    )
                    for g in range(params["r_unconstrained_loc"].shape[1])
                ]
                for c in range(params["r_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["r_unconstrained"] = dist.Normal(
                params["r_unconstrained_loc"], params["r_unconstrained_scale"]
            )

    # gate_unconstrained parameter (Normal distribution)
    if (
        "gate_unconstrained_loc" in params
        and "gate_unconstrained_scale" in params
    ):
        if split and len(params["gate_unconstrained_loc"].shape) == 1:
            # Gene-specific gate_unconstrained parameters
            distributions["gate_unconstrained"] = [
                dist.Normal(
                    params["gate_unconstrained_loc"][c],
                    params["gate_unconstrained_scale"][c],
                )
                for c in range(params["gate_unconstrained_loc"].shape[0])
            ]
        elif split and len(params["gate_unconstrained_loc"].shape) == 2:
            # Component and gene-specific gate_unconstrained parameters
            distributions["gate_unconstrained"] = [
                [
                    dist.Normal(
                        params["gate_unconstrained_loc"][c, g],
                        params["gate_unconstrained_scale"][c, g],
                    )
                    for g in range(params["gate_unconstrained_loc"].shape[1])
                ]
                for c in range(params["gate_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["gate_unconstrained"] = dist.Normal(
                params["gate_unconstrained_loc"],
                params["gate_unconstrained_scale"],
            )

    # p_capture_unconstrained parameter (Normal distribution)
    if (
        "p_capture_unconstrained_loc" in params
        and "p_capture_unconstrained_scale" in params
    ):
        if split and len(params["p_capture_unconstrained_loc"].shape) == 1:
            # Cell-specific p_capture_unconstrained parameters
            distributions["p_capture_unconstrained"] = [
                dist.Normal(
                    params["p_capture_unconstrained_loc"][c],
                    params["p_capture_unconstrained_scale"][c],
                )
                for c in range(params["p_capture_unconstrained_loc"].shape[0])
            ]
        else:
            distributions["p_capture_unconstrained"] = dist.Normal(
                params["p_capture_unconstrained_loc"],
                params["p_capture_unconstrained_scale"],
            )

    # mixing_logits_unconstrained parameter (Normal distribution)
    if (
        "mixing_logits_unconstrained_loc" in params
        and "mixing_logits_unconstrained_scale" in params
    ):
        mixing_dist = dist.Normal(
            params["mixing_logits_unconstrained_loc"],
            params["mixing_logits_unconstrained_scale"],
        )
        # mixing_logits_unconstrained is typically not split since it represents a single
        # probability vector
        distributions["mixing_logits_unconstrained"] = mixing_dist

    return distributions

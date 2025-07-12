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
    # Define guide parameters
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
    numpyro.sample(
        "gate_unconstrained", dist.Normal(gate_loc, gate_scale)
    )


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

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat", p * p_capture / (1 - p * (1 - p_capture))
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

                p_hat = numpyro.deterministic(
                    "p_hat", p * p_capture / (1 - p * (1 - p_capture))
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

            p_hat = numpyro.deterministic(
                "p_hat", p * p_capture / (1 - p * (1 - p_capture))
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

    # Register cell-specific capture probability parameters
    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_loc = numpyro.param(
            "p_capture_unconstrained_loc", p_capture_guide_params[0]
        )
        p_capture_scale = numpyro.param(
            "p_capture_unconstrained_scale",
            p_capture_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(p_capture_loc, p_capture_scale),
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

                # Compute effective probability
                p_hat = numpyro.deterministic(
                    "p_hat", p * p_capture / (1 - p * (1 - p_capture))
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

                p_hat = numpyro.deterministic(
                    "p_hat", p * p_capture / (1 - p * (1 - p_capture))
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

            p_hat = numpyro.deterministic(
                "p_hat", p * p_capture / (1 - p * (1 - p_capture))
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
    numpyro.sample(
        "gate_unconstrained", dist.Normal(gate_loc, gate_scale)
    )

    # Register cell-specific capture probability parameters
    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_loc = numpyro.param(
            "p_capture_unconstrained_loc", p_capture_guide_params[0]
        )
        p_capture_scale = numpyro.param(
            "p_capture_unconstrained_scale",
            p_capture_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(p_capture_loc, p_capture_scale),
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

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior or (
        0.0,
        1.0,
    )
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    # Add deterministic site for mixing_weights
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    # Sample unconstrained p for each component
    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(*p_prior_params).expand([n_components]),
    )

    # Sample unconstrained r for each component and gene
    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    # Deterministic transformations to constrained space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp))

    # Define mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Define component distribution using logits parameterization
    component_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained_comp[:, None]
    ).to_event(1)

    # Create mixture distribution
    mixture_model = dist.MixtureSameFamily(mixing_dist, component_dist)

    # Model likelihood
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture_model, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", mixture_model, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture_model)


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
        dist.Normal(mixing_loc, mixing_scale).to_event(1),
    )

    # Register p parameters
    p_loc = numpyro.param(
        "p_unconstrained_comp_loc", jnp.full(n_components, p_guide_params[0])
    )
    p_scale = numpyro.param(
        "p_unconstrained_comp_scale",
        jnp.full(n_components, p_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "p_unconstrained_comp", dist.Normal(p_loc, p_scale)
    )

    # Register r parameters
    r_loc = numpyro.param(
        "r_unconstrained_comp_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_comp_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "r_unconstrained_comp", dist.Normal(r_loc, r_scale)
    )


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

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior or (
        0.0,
        1.0,
    )
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(*p_prior_params).expand([n_components]),
    )

    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    gate_unconstrained_comp = numpyro.sample(
        "gate_unconstrained_comp",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Transform to constrained space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained_comp))

    # Define distributions
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    base_nb_dist = dist.NegativeBinomialLogits(
        total_count=r, logits=p_unconstrained_comp[:, None]
    )

    zinb_comp_dist = dist.ZeroInflatedDistribution(
        base_nb_dist, gate_logits=gate_unconstrained_comp
    ).to_event(1)

    mixture_model = dist.MixtureSameFamily(mixing_dist, zinb_comp_dist)

    # Model likelihood
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture_model, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample("counts", mixture_model, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture_model)


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

    # Register parameters
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
        dist.Normal(mixing_loc, mixing_scale).to_event(1),
    )

    p_loc = numpyro.param(
        "p_unconstrained_comp_loc", jnp.full(n_components, p_guide_params[0])
    )
    p_scale = numpyro.param(
        "p_unconstrained_comp_scale",
        jnp.full(n_components, p_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "p_unconstrained_comp", dist.Normal(p_loc, p_scale)
    )

    r_loc = numpyro.param(
        "r_unconstrained_comp_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_comp_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "r_unconstrained_comp", dist.Normal(r_loc, r_scale)
    )

    gate_loc = numpyro.param(
        "gate_unconstrained_comp_loc",
        jnp.full((n_components, n_genes), gate_guide_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_comp_scale",
        jnp.full((n_components, n_genes), gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "gate_unconstrained_comp", dist.Normal(gate_loc, gate_scale)
    )


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

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior or (
        0.0,
        1.0,
    )
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(*p_prior_params).expand([n_components]),
    )

    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained_comp))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp))

    # Define global mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Cell-specific sampling with capture probability
    plate_context = numpyro.plate(
        "cells",
        n_cells,
        subsample_size=batch_size if counts is not None else None,
    )

    with plate_context as idx:
        current_counts = counts[idx] if counts is not None else None

        # Sample unconstrained cell-specific capture probability
        p_capture_unconstrained_batch = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )

        p_capture_batch = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained_batch)
        )

        # Calculate effective probabilities
        p_hat_batch = numpyro.deterministic(
            "p_hat",
            p[None, :]
            * p_capture_batch[:, None]
            / (1 - p[None, :] * (1 - p_capture_batch[:, None])),
        )

        # Define component distribution
        component_dist_batch = dist.NegativeBinomialProbs(
            total_count=r[None, :, :],
            probs=p_hat_batch[:, :, None],
        ).to_event(1)

        mixture_model_batch = dist.MixtureSameFamily(
            mixing_dist, component_dist_batch
        )

        numpyro.sample("counts", mixture_model_batch, obs=current_counts)


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
        dist.Normal(mixing_loc, mixing_scale).to_event(1),
    )

    p_loc = numpyro.param(
        "p_unconstrained_comp_loc", jnp.full(n_components, p_guide_params[0])
    )
    p_scale = numpyro.param(
        "p_unconstrained_comp_scale",
        jnp.full(n_components, p_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "p_unconstrained_comp", dist.Normal(p_loc, p_scale)
    )

    r_loc = numpyro.param(
        "r_unconstrained_comp_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_comp_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "r_unconstrained_comp", dist.Normal(r_loc, r_scale)
    )

    # Register cell-specific capture probability parameters
    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_loc = numpyro.param(
            "p_capture_unconstrained_loc", p_capture_guide_params[0]
        )
        p_capture_scale = numpyro.param(
            "p_capture_unconstrained_scale",
            p_capture_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(p_capture_loc, p_capture_scale),
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

    # Define prior parameters
    mixing_prior_params = model_config.mixing_logits_unconstrained_prior or (
        0.0,
        1.0,
    )
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    r_prior_params = model_config.r_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(*mixing_prior_params).expand([n_components]),
    )
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(*p_prior_params).expand([n_components]),
    )

    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(*r_prior_params).expand([n_components, n_genes]),
    )

    gate_unconstrained_comp = numpyro.sample(
        "gate_unconstrained_comp",
        dist.Normal(*gate_prior_params).expand([n_components, n_genes]),
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained_comp))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained_comp))

    # Define global mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Cell-specific sampling with capture probability
    plate_context = numpyro.plate(
        "cells",
        n_cells,
        subsample_size=batch_size if counts is not None else None,
    )

    with plate_context as idx:
        current_counts = counts[idx] if counts is not None else None

        p_capture_unconstrained_batch = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(*p_capture_prior_params),
        )

        p_capture_batch = numpyro.deterministic(
            "p_capture", jsp.special.expit(p_capture_unconstrained_batch)
        )

        p_hat_batch = numpyro.deterministic(
            "p_hat",
            p[None, :]
            * p_capture_batch[:, None]
            / (1 - p[None, :] * (1 - p_capture_batch[:, None])),
        )

        # Define component distribution with zero-inflation
        base_nb_dist_for_zinb = dist.NegativeBinomialProbs(
            total_count=r[None, :, :], probs=p_hat_batch[:, :, None]
        )

        zinb_base_comp_dist = dist.ZeroInflatedDistribution(
            base_nb_dist_for_zinb,
            gate_logits=gate_unconstrained_comp[None, :, :],
        )

        component_dist_batch = zinb_base_comp_dist.to_event(1)

        mixture_model_batch = dist.MixtureSameFamily(
            mixing_dist, component_dist_batch
        )

        numpyro.sample("counts", mixture_model_batch, obs=current_counts)


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
        dist.Normal(mixing_loc, mixing_scale).to_event(1),
    )

    p_loc = numpyro.param(
        "p_unconstrained_comp_loc", jnp.full(n_components, p_guide_params[0])
    )
    p_scale = numpyro.param(
        "p_unconstrained_comp_scale",
        jnp.full(n_components, p_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "p_unconstrained_comp", dist.Normal(p_loc, p_scale).to_event(1)
    )

    r_loc = numpyro.param(
        "r_unconstrained_comp_loc",
        jnp.full((n_components, n_genes), r_guide_params[0]),
    )
    r_scale = numpyro.param(
        "r_unconstrained_comp_scale",
        jnp.full((n_components, n_genes), r_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "r_unconstrained_comp", dist.Normal(r_loc, r_scale).to_event(2)
    )

    gate_loc = numpyro.param(
        "gate_unconstrained_comp_loc",
        jnp.full((n_components, n_genes), gate_guide_params[0]),
    )
    gate_scale = numpyro.param(
        "gate_unconstrained_comp_scale",
        jnp.full((n_components, n_genes), gate_guide_params[1]),
        constraint=constraints.positive,
    )
    numpyro.sample(
        "gate_unconstrained_comp", dist.Normal(gate_loc, gate_scale).to_event(2)
    )

    # Register cell-specific capture probability parameters
    with numpyro.plate("cells", n_cells, subsample_size=batch_size):
        p_capture_loc = numpyro.param(
            "p_capture_unconstrained_loc", p_capture_guide_params[0]
        )
        p_capture_scale = numpyro.param(
            "p_capture_unconstrained_scale",
            p_capture_guide_params[1],
            constraint=constraints.positive,
        )
        numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(p_capture_loc, p_capture_scale),
        )


# ------------------------------------------------------------------------------
# Utility function
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray], model_config: ModelConfig
) -> Dict[str, dist.Distribution]:
    """
    Get posterior distributions for unconstrained parameters.
    """
    posteriors = {}

    # Handle different parameter types based on model configuration
    active_params = model_config.get_active_parameters()

    for param in active_params:
        if param.endswith("_unconstrained"):
            # For unconstrained parameters, we expect Normal distributions
            param_name = param.replace("_unconstrained", "")

            if f"{param}_loc" in params and f"{param}_scale" in params:
                posteriors[param_name] = dist.Normal(
                    params[f"{param}_loc"], params[f"{param}_scale"]
                )
            elif (
                f"{param}_comp_loc" in params
                and f"{param}_comp_scale" in params
            ):
                # For component-specific parameters
                posteriors[param_name] = dist.Normal(
                    params[f"{param}_comp_loc"], params[f"{param}_comp_scale"]
                )

    return posteriors

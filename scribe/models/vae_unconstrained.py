"""
VAE-based models for single-cell RNA sequencing data using unconstrained
parameterization.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.contrib.module import nnx_module
from typing import Dict, Optional

from .model_config import ModelConfig
from ..vae.architectures import (
    Encoder,
    EncoderVCP,
    Decoder,
    VAE,
    DecoupledPrior,
    DecoupledPriorDistribution,
)


# ------------------------------------------------------------------------------
# NBDM VAE Model
# ------------------------------------------------------------------------------


def nbdm_vae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    decoder: Decoder,  # Pre-created decoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    VAE-based Numpyro model for Negative Binomial-Dirichlet Multinomial data
    with unconstrained parameterization.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))

    # Register the pre-created decoder as NumPyro module
    decoder_module = nnx_module("decoder", decoder)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            # Without batching: sample counts for all cells
            with numpyro.plate("cells", n_cells):
                # Sample from latent space prior
                z = numpyro.sample(
                    "z",
                    dist.Normal(0, 1)
                    .expand([model_config.vae_latent_dim])
                    .to_event(1),
                )

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # With batching: sample counts for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample from latent space prior
                z = numpyro.sample(
                    "z",
                    dist.Normal(0, 1)
                    .expand([model_config.vae_latent_dim])
                    .to_event(1),
                )

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample(
                "z",
                dist.Normal(0, 1)
                .expand([model_config.vae_latent_dim])
                .to_event(1),
            )

            # Use decoder to generate r parameters from latent space
            r_unconstrained = numpyro.deterministic(
                "r_unconstrained", decoder_module(z)
            )
            r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------
# NBDM VAE Guide
# ------------------------------------------------------------------------------


def nbdm_vae_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    encoder: Encoder,  # Pre-created encoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based NBDM model with
    unconstrained parameterization.
    """
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register the pre-created encoder as NumPyro module for the guide
    encoder_module = nnx_module("encoder", encoder)

    # Sample latent variables using encoder
    if counts is not None:
        if batch_size is None:
            # Without batching: sample latent variables for all cells
            with numpyro.plate("cells", n_cells):
                # Use encoder to get mean and log variance for latent space
                z_mean, z_logvar = encoder_module(counts)
                z_std = jnp.exp(0.5 * z_logvar)

                # Sample from variational distribution
                numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))
        else:
            # With batching: sample latent variables for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Use encoder to get mean and log variance for latent space
                batch_data = counts[idx]
                z_mean, z_logvar = encoder_module(batch_data)
                z_std = jnp.exp(0.5 * z_logvar)

                # Sample from variational distribution
                numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            numpyro.sample(
                "z",
                dist.Normal(0, 1)
                .expand([model_config.vae_latent_dim])
                .to_event(1),
            )


# ------------------------------------------------------------------------------
# ZINB VAE Model
# ------------------------------------------------------------------------------


def zinb_vae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    decoder: Decoder,  # Pre-created decoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    VAE-based Numpyro model for Zero-Inflated Negative Binomial data with
    unconstrained parameterization.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)

    # Sample unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", dist.Normal(*gate_prior_params).expand([n_genes])
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Register the pre-created decoder as NumPyro module
    decoder_module = nnx_module("decoder", decoder)

    # If observed counts are provided
    if counts is not None:
        # If no batching, use a plate over all cells
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample from latent space prior
                z = numpyro.sample(
                    "z",
                    dist.Normal(0, 1)
                    .expand([model_config.vae_latent_dim])
                    .to_event(1),
                )

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Construct the base Negative Binomial distribution using r and
                # p
                base_dist = dist.NegativeBinomialProbs(r, p)
                # Construct the zero-inflated distribution using the base NB and
                # gate
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Sample observed counts from the zero-inflated NB distribution
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # If batching, use a plate with subsampling and get indices
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample from latent space prior
                z = numpyro.sample(
                    "z",
                    dist.Normal(0, 1)
                    .expand([model_config.vae_latent_dim])
                    .to_event(1),
                )

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Construct the base Negative Binomial distribution using r and
                # p
                base_dist = dist.NegativeBinomialProbs(r, p)
                # Construct the zero-inflated distribution using the base NB and
                # gate
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Sample observed counts for the batch indices
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # If no observed counts, just sample from the prior predictive
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample(
                "z",
                dist.Normal(0, 1)
                .expand([model_config.vae_latent_dim])
                .to_event(1),
            )

            # Use decoder to generate r parameters from latent space
            r_unconstrained = numpyro.deterministic(
                "r_unconstrained", decoder_module(z)
            )
            r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

            # Construct the base Negative Binomial distribution using r and p
            base_dist = dist.NegativeBinomialProbs(r, p)
            # Construct the zero-inflated distribution using the base NB and
            # gate
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Sample counts (not observed)
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------
# ZINB VAE Guide
# ------------------------------------------------------------------------------


def zinb_vae_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    encoder: Encoder,  # Pre-created encoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based ZINB model with
    unconstrained parameterization.
    """
    # Define guide parameters
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)
    gate_guide_params = model_config.gate_unconstrained_guide or (0.0, 1.0)

    # Register unconstrained p parameters
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

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

    # Register the pre-created encoder as NumPyro module for the guide
    encoder_module = nnx_module("encoder", encoder)

    # Sample latent variables using encoder
    if counts is not None:
        if batch_size is None:
            # Without batching: sample latent variables for all cells
            with numpyro.plate("cells", n_cells):
                # Use encoder to get mean and log variance for latent space
                z_mean, z_logvar = encoder_module(counts)
                z_std = jnp.exp(0.5 * z_logvar)

                # Sample from variational distribution
                numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))
        else:
            # With batching: sample latent variables for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Use encoder to get mean and log variance for latent space
                batch_data = counts[idx]
                z_mean, z_logvar = encoder_module(batch_data)
                z_std = jnp.exp(0.5 * z_logvar)

                # Sample from variational distribution
                numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            numpyro.sample(
                "z",
                dist.Normal(0, 1)
                .expand([model_config.vae_latent_dim])
                .to_event(1),
            )


# ------------------------------------------------------------------------------
# NBVCP VAE Model
# ------------------------------------------------------------------------------


def nbvcp_vae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    decoder: Decoder,  # Pre-created decoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    VAE-based Numpyro model for Negative Binomial with variable capture
    probability and unconstrained parameterization.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global unconstrained parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )

    # Transform to constrained space
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))

    # Register the pre-created decoder as NumPyro module
    decoder_module = nnx_module("decoder", decoder)

    # Sample counts
    if counts is not None:
        if batch_size is None:
            # Without batching: sample counts for all cells
            with numpyro.plate("cells", n_cells):
                # Sample from latent space prior
                z = numpyro.sample(
                    "z",
                    dist.Normal(0, 1)
                    .expand([model_config.vae_latent_dim])
                    .to_event(1),
                )

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Sample cell-specific capture probability from Beta prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture = p_capture[:, None]  # Shape: (batch_size, 1)
                # Compute effective success probability p_hat for each cell/gene
                p_hat = p * p_capture / (1 - p * (1 - p_capture))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # With batching: sample counts for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample from latent space prior
                z = numpyro.sample(
                    "z",
                    dist.Normal(0, 1)
                    .expand([model_config.vae_latent_dim])
                    .to_event(1),
                )

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Sample cell-specific capture probability from Beta prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture = p_capture[:, None]  # Shape: (batch_size, 1)
                # Compute effective success probability p_hat for each cell/gene
                p_hat = p * p_capture / (1 - p * (1 - p_capture))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample(
                "z",
                dist.Normal(0, 1)
                .expand([model_config.vae_latent_dim])
                .to_event(1),
            )

            # Use decoder to generate r parameters from latent space
            r_unconstrained = numpyro.deterministic(
                "r_unconstrained", decoder_module(z)
            )
            r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

            # Sample cell-specific capture probability from Beta prior
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(*p_capture_prior_params),
            )
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
            )
            # Reshape p_capture for broadcasting to (n_cells, n_genes)
            p_capture = p_capture[:, None]  # Shape: (batch_size, 1)
            # Compute effective success probability p_hat for each cell/gene
            p_hat = p * p_capture / (1 - p * (1 - p_capture))

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------
# NBVCP VAE Guide
# ------------------------------------------------------------------------------


def nbvcp_vae_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    encoder: EncoderVCP,  # Pre-created encoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based NBVCP model.
    """
    # Define guide parameters for p
    p_guide_params = model_config.p_unconstrained_guide or (0.0, 1.0)

    # Register p_alpha as a variational parameter with positivity constraint
    p_loc = numpyro.param("p_unconstrained_loc", p_guide_params[0])
    # Register p_beta as a variational parameter with positivity constraint
    p_scale = numpyro.param(
        "p_unconstrained_scale",
        p_guide_params[1],
        constraint=constraints.positive,
    )
    # Sample p from the Beta distribution parameterized by p_alpha and p_beta
    numpyro.sample("p_unconstrained", dist.Normal(p_loc, p_scale))

    # Register the pre-created encoder as NumPyro module for the guide
    encoder_module = nnx_module("encoder", encoder)

    # Sample latent variables using encoder
    if counts is not None:
        if batch_size is None:
            # Without batching: sample latent variables for all cells
            with numpyro.plate("cells", n_cells):
                # Use encoder to get mean and log variance for latent space
                (
                    z_mean,
                    z_logvar,
                    p_capture_loc,
                    p_capture_logscale,
                ) = encoder_module(counts)
                z_std = jnp.exp(0.5 * z_logvar)

                # Sample from variational distribution
                numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))

                # Sample cell-specific capture probability from prior
                numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(
                        p_capture_loc.squeeze(-1),
                        jnp.exp(p_capture_logscale.squeeze(-1)),
                    ),
                )
        else:
            # With batching: sample latent variables for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Use encoder to get mean and log variance for latent space
                batch_data = counts[idx]
                (
                    z_mean,
                    z_logvar,
                    p_capture_loc,
                    p_capture_logscale,
                ) = encoder_module(batch_data)
                z_std = jnp.exp(0.5 * z_logvar)

                # Sample from variational distribution
                numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))

                # Sample cell-specific capture probability from Beta prior
                numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(
                        p_capture_loc.squeeze(-1),
                        jnp.exp(p_capture_logscale.squeeze(-1)),
                    ),
                )
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            p_capture_prior_params = (
                model_config.p_capture_unconstrained_prior
                or (
                    0.0,
                    1.0,
                )
            )
            # Sample from latent space prior
            numpyro.sample(
                "z",
                dist.Normal(0, 1)
                .expand([model_config.vae_latent_dim])
                .to_event(1),
            )

            # Sample cell-specific capture probability from Beta prior
            numpyro.sample(
                "p_capture_unconstrained", dist.Normal(*p_capture_prior_params)
            )


# ==============================================================================
# dpVAE Model Functions (for decoupled prior)
# ==============================================================================

# ------------------------------------------------------------------------------
# NBDM dpVAE Model
# ------------------------------------------------------------------------------


def nbdm_dpvae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    decoder: Decoder,
    decoupled_prior: DecoupledPrior,
    counts=None,
    batch_size=None,
):
    """
    Generative model for NBDM dpVAE with unconstrained parameterization.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)

    # Sample global parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))

    # Register the decoder and decoupled prior as NumPyro modules
    decoder_module = nnx_module("decoder", decoder)
    decoupled_prior_module = nnx_module("decoupled_prior", decoupled_prior)

    # Create the decoupled prior distribution
    base_distribution = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    ).to_event(1)
    decoupled_prior_dist = DecoupledPriorDistribution(
        decoupled_prior=decoupled_prior_module,
        base_distribution=base_distribution,
    )

    # Sample latent variables and generate observations
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample z from decoupled prior (KEY DIFFERENCE from standard
                # VAE)
                z = numpyro.sample("z", decoupled_prior_dist)

                # Decode z to get r parameters
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample z from decoupled prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Decode z to get r parameters
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Sample observed counts
                batch_counts = counts[idx] if counts is not None else None
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r, p).to_event(1),
                    obs=batch_counts,
                )
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample("z", decoupled_prior_dist)

            # Use decoder to generate r parameters from latent space
            r_unconstrained = numpyro.deterministic(
                "r_unconstrained", decoder_module(z)
            )
            r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------
# ZINB dpVAE Model
# ------------------------------------------------------------------------------


def zinb_dpvae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    decoder: Decoder,
    decoupled_prior: DecoupledPrior,
    counts=None,
    batch_size=None,
):
    """
    Generative model for ZINB dpVAE with unconstrained parameterization.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    gate_prior_params = model_config.gate_unconstrained_prior or (0.0, 1.0)

    # Sample global parameters
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained",
        dist.Normal(*gate_prior_params).expand([n_genes]),
    )
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))

    # Register the decoder and decoupled prior as NumPyro modules
    decoder_module = nnx_module("decoder", decoder)
    decoupled_prior_module = nnx_module("decoupled_prior", decoupled_prior)

    # Create the decoupled prior distribution
    base_distribution = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    ).to_event(1)
    decoupled_prior_dist = DecoupledPriorDistribution(
        decoupled_prior=decoupled_prior_module,  # Use wrapped module
        base_distribution=base_distribution,
    )

    # If observed counts are provided
    if counts is not None:
        # If no batching, use a plate over all cells
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample from latent space prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Construct the base Negative Binomial distribution using r and
                # p
                base_dist = dist.NegativeBinomialProbs(r, p)
                # Construct the zero-inflated distribution using the base NB and
                # gate
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Sample observed counts from the zero-inflated NB distribution
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # If batching, use a plate with subsampling and get indices
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample from latent space prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Construct the base Negative Binomial distribution using r and
                # p
                base_dist = dist.NegativeBinomialProbs(r, p)
                # Construct the zero-inflated distribution using the base NB and
                # gate
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate
                ).to_event(1)
                # Sample observed counts for the batch indices
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # If no observed counts, just sample from the prior predictive
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample("z", decoupled_prior_dist)

            # Use decoder to generate r parameters from latent space
            r_unconstrained = numpyro.deterministic(
                "r_unconstrained", decoder_module(z)
            )
            r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

            # Construct the base Negative Binomial distribution using r and p
            base_dist = dist.NegativeBinomialProbs(r, p)
            # Construct the zero-inflated distribution using the base NB and
            # gate
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Sample counts (not observed)
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------
# NBVCP dpVAE Model
# ------------------------------------------------------------------------------


def nbvcp_dpvae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    decoder: Decoder,
    decoupled_prior: DecoupledPrior,
    counts=None,
    batch_size=None,
):
    """
    VAE-based Numpyro model for Negative Binomial with variable capture
    probability and unconstrained parameterization.
    """
    # Define prior parameters
    p_prior_params = model_config.p_unconstrained_prior or (0.0, 1.0)
    p_capture_prior_params = model_config.p_capture_unconstrained_prior or (
        0.0,
        1.0,
    )

    # Sample global success probability p from Beta prior
    p_unconstrained = numpyro.sample(
        "p_unconstrained", dist.Normal(*p_prior_params)
    )
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))

    # Register the decoder and decoupled prior as NumPyro modules
    decoder_module = nnx_module("decoder", decoder)
    decoupled_prior_module = nnx_module("decoupled_prior", decoupled_prior)

    # Create the decoupled prior distribution
    base_distribution = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    ).to_event(1)
    decoupled_prior_dist = DecoupledPriorDistribution(
        decoupled_prior=decoupled_prior_module,
        base_distribution=base_distribution,
    )

    # Sample counts
    if counts is not None:
        if batch_size is None:
            # Without batching: sample counts for all cells
            with numpyro.plate("cells", n_cells):
                # Sample z from decoupled prior (KEY DIFFERENCE from standard
                # VAE)
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Sample cell-specific capture probability from Beta prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture = p_capture[:, None]
                # Compute effective success probability p_hat for each cell/gene
                p_hat = p * p_capture / (1 - p * (1 - p_capture))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # With batching: sample counts for a subset of cells
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample z from decoupled prior (KEY DIFFERENCE from standard
                # VAE)
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate r parameters from latent space
                r_unconstrained = numpyro.deterministic(
                    "r_unconstrained", decoder_module(z)
                )
                r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

                # Sample cell-specific capture probability from Beta prior
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(*p_capture_prior_params),
                )
                p_capture = numpyro.deterministic(
                    "p_capture", jsp.special.expit(p_capture_unconstrained)
                )
                # Reshape p_capture for broadcasting to (n_cells, n_genes)
                p_capture = p_capture[:, None]  # Shape: (batch_size, 1)
                # Compute effective success probability p_hat for each cell/gene
                p_hat = p * p_capture / (1 - p * (1 - p_capture))

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample z from decoupled prior (KEY DIFFERENCE from standard VAE)
            z = numpyro.sample("z", decoupled_prior_dist)

            # Use decoder to generate r parameters from latent space
            r_unconstrained = numpyro.deterministic(
                "r_unconstrained", decoder_module(z)
            )
            r = numpyro.deterministic("r", jnp.exp(r_unconstrained))

            # Sample cell-specific capture probability from Beta prior
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(*p_capture_prior_params),
            )
            p_capture = numpyro.deterministic(
                "p_capture", jsp.special.expit(p_capture_unconstrained)
            )
            # Reshape p_capture for broadcasting to (n_cells, n_genes)
            p_capture = p_capture[:, None]  # Shape: (batch_size, 1)
            # Compute effective success probability p_hat for each cell/gene
            p_hat = p * p_capture / (1 - p * (1 - p_capture))

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            numpyro.sample("counts", base_dist)


# ==============================================================================
# dpVAE Posterior Distribution Functions
# ==============================================================================


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    vae_model: VAE,
) -> Dict[str, dist.Distribution]:
    """
    Constructs and returns a dictionary of posterior distributions from
    estimated parameters for the dpVAE-unconstrained parameterization.

    This function builds the appropriate `numpyro` distributions based on the
    guide parameters found in the `params` dictionary. All distributions are
    returned as batch distributions (no splitting or per-component/gene lists).

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        A dictionary of estimated parameters from the variational guide.
    vae_model : VAE
        The VAE model object.

    Returns
    -------
    Dict[str, dist.Distribution]
        Dictionary mapping parameter names to their posterior distributions.
    """
    distributions = {}

    # p_unconstrained parameter (Normal distribution)
    if "p_unconstrained_loc" in params and "p_unconstrained_scale" in params:
        distributions["p_unconstrained"] = dist.Normal(
            params["p_unconstrained_loc"], params["p_unconstrained_scale"]
        )

    # gate_unconstrained parameter (Normal distribution)
    if (
        "gate_unconstrained_loc" in params
        and "gate_unconstrained_scale" in params
    ):
        distributions["gate_unconstrained"] = dist.Normal(
            params["gate_unconstrained_loc"],
            params["gate_unconstrained_scale"],
        )

    # p_capture_unconstrained parameter (Normal distribution)
    if (
        "p_capture_unconstrained_loc" in params
        and "p_capture_unconstrained_scale" in params
    ):
        distributions["p_capture_unconstrained"] = dist.Normal(
            params["p_capture_unconstrained_loc"],
            params["p_capture_unconstrained_scale"],
        )

    # mixing_logits_unconstrained parameter (Normal distribution)
    if (
        "mixing_logits_unconstrained_loc" in params
        and "mixing_logits_unconstrained_scale" in params
    ):
        distributions["mixing_logits_unconstrained"] = dist.Normal(
            params["mixing_logits_unconstrained_loc"],
            params["mixing_logits_unconstrained_scale"],
        )

    # Get the decoupled prior distribution
    distributions["z"] = vae_model.get_prior_distribution()

    return distributions

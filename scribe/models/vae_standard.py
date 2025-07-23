"""
VAE-based models for single-cell RNA sequencing data using standard
parameterization.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.contrib.module import nnx_module
from typing import Dict, Optional

from .model_config import ModelConfig
from ..vae.architectures import create_vae


# ------------------------------------------------------------------------------
# NBDM VAE Model
# ------------------------------------------------------------------------------


def nbdm_vae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    VAE-based Numpyro model for Negative Binomial-Dirichlet Multinomial data.

    The VAE generates r parameters for each cell while keeping p interpretable.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)

    # Sample global success probability p from Beta prior
    p = numpyro.sample("p", dist.Beta(*p_prior_params))

    # Create VAE for generating r parameters
    vae = create_vae(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        output_activation=model_config.vae_output_activation,
    )

    # Register encoder and decoder separately as NumPyro modules
    encoder_module = nnx_module("encoder", vae.encoder)
    decoder_module = nnx_module("decoder", vae.decoder)

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
                r_params = decoder_module(z)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r_params, p).to_event(1)
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
                r_params = decoder_module(z)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r_params, p).to_event(1)
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
            r_params = decoder_module(z)

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r_params, p).to_event(1)
            numpyro.sample("counts", base_dist)


# ------------------------------------------------------------------------------
# NBDM VAE Guide
# ------------------------------------------------------------------------------


def nbdm_vae_guide(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based NBDM model.
    """
    # Define guide parameters for p
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)

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

    # Create VAE for encoding
    vae = create_vae(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        output_activation=model_config.vae_output_activation,
    )

    # Register encoder as NumPyro module for the guide
    encoder_module = nnx_module("encoder", vae.encoder)

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
            # Generate dummy data for encoder
            dummy_data = jnp.zeros((n_cells, n_genes))
            z_mean, z_logvar = encoder_module(dummy_data)
            z_std = jnp.exp(0.5 * z_logvar)

            # Sample from variational distribution
            numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))


# ------------------------------------------------------------------------------
# ZINB VAE Model
# ------------------------------------------------------------------------------


def zinb_vae_model(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    VAE-based Numpyro model for Zero-Inflated Negative Binomial data.
    """
    # Get prior parameters for p and gate
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample p from a Beta distribution with the specified prior parameters
    p = numpyro.sample("p", dist.Beta(*p_prior_params))
    # Sample gate from a Beta distribution with the specified prior parameters,
    # expanded to n_genes
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_genes])
    )

    # Create VAE for generating r parameters
    vae = create_vae(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        output_activation=model_config.vae_output_activation,
    )

    # Register encoder and decoder separately as NumPyro modules
    encoder_module = nnx_module("encoder", vae.encoder)
    decoder_module = nnx_module("decoder", vae.decoder)

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
                r_params = decoder_module(z)

                # Construct the base Negative Binomial distribution using r and p
                base_dist = dist.NegativeBinomialProbs(r_params, p)
                # Construct the zero-inflated distribution using the base NB and gate
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
                r_params = decoder_module(z)

                # Construct the base Negative Binomial distribution using r and p
                base_dist = dist.NegativeBinomialProbs(r_params, p)
                # Construct the zero-inflated distribution using the base NB and gate
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
            r_params = decoder_module(z)

            # Construct the base Negative Binomial distribution using r and p
            base_dist = dist.NegativeBinomialProbs(r_params, p)
            # Construct the zero-inflated distribution using the base NB and gate
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
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based ZINB model.
    """
    # Define guide parameters for p and gate
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Register variational parameters for p (success probability)
    p_alpha = numpyro.param(
        "p_alpha", p_prior_params[0], constraint=constraints.positive
    )
    p_beta = numpyro.param(
        "p_beta", p_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("p", dist.Beta(p_alpha, p_beta))

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
    numpyro.sample("gate", dist.Beta(gate_alpha, gate_beta))

    # Create VAE for encoding
    vae = create_vae(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        output_activation=model_config.vae_output_activation,
    )

    # Register encoder as NumPyro module for the guide
    encoder_module = nnx_module("encoder", vae.encoder)

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
            # Generate dummy data for encoder
            dummy_data = jnp.zeros((n_cells, n_genes))
            z_mean, z_logvar = encoder_module(dummy_data)
            z_std = jnp.exp(0.5 * z_logvar)

            # Sample from variational distribution
            numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))

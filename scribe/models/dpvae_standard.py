"""
dpVAE-based models for single-cell RNA sequencing data using standard
parameterization with decoupled prior.

The key difference from standard VAE is that the GUIDE uses a DecoupledPriorDistribution
instead of a standard Normal prior, while the MODEL remains the same.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.contrib.module import nnx_module
from typing import Dict, Optional

from .model_config import ModelConfig
from .vae_standard import nbdm_vae_guide, zinb_vae_guide
from ..vae.architectures import (
    create_encoder,
    create_decoder,
    Encoder,
    Decoder,
    DecoupledPrior,
    DecoupledPriorDistribution,
)


# ------------------------------------------------------------------------------
# NBDM dpVAE Model Functions (NEW - these use decoupled prior)
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
    Generative model for NBDM dpVAE.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Sample global parameters
    p = numpyro.sample("p", dist.Beta(1.0, 1.0))

    # Register the decoder and decoupled prior as NumPyro modules
    decoder_module = nnx_module("decoder", decoder)
    decoupled_prior_module = nnx_module("decoupled_prior", decoupled_prior)

    # Create the decoupled prior distribution
    base_distribution = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    ).to_event(1)
    decoupled_prior_dist = DecoupledPriorDistribution(
        decoupled_prior=decoupled_prior,  # Use original module, not wrapper
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
                r_params = decoder_module(z)

                # Use decoder to generate r parameters from latent space
                r_params = decoder_module(z)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r_params, p).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample z from decoupled prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Decode z to get r parameters
                r_params = decoder_module(z)

                # Sample observed counts
                batch_counts = counts[idx] if counts is not None else None
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialProbs(r_params, p).to_event(1),
                    obs=batch_counts,
                )
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample("z", decoupled_prior_dist)

            # Use decoder to generate r parameters from latent space
            r_params = decoder_module(z)

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r_params, p).to_event(1)
            numpyro.sample("counts", base_dist)

# ------------------------------------------------------------------------------
# ZINB dpVAE Model Functions (NEW - these use decoupled prior)
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
    Generative model for ZINB dpVAE.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Sample global parameters
    p = numpyro.sample("p", dist.Beta(1.0, 1.0))
    gate = numpyro.sample("gate", dist.Beta(1.0, 1.0).expand((n_genes,)))

    # Register the decoder and decoupled prior as NumPyro modules
    decoder_module = nnx_module("decoder", decoder)
    decoupled_prior_module = nnx_module("decoupled_prior", decoupled_prior)

    # Create the decoupled prior distribution
    base_distribution = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    ).to_event(1)
    decoupled_prior_dist = DecoupledPriorDistribution(
        decoupled_prior=decoupled_prior,  # Use original module, not wrapper
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
                z = numpyro.sample("z", decoupled_prior_dist)

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
            z = numpyro.sample("z", decoupled_prior_dist)

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
# FIXED Factory Functions
# ------------------------------------------------------------------------------


def make_nbdm_dpvae_model_and_guide(
    n_genes: int,
    model_config: ModelConfig,
):
    """
    FIXED: Construct and return dpVAE model and guide functions for NBDM model.
    """
    # Create the encoder and decoder modules once
    decoder = create_decoder(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        output_activation=model_config.vae_output_activation,
    )

    encoder = create_encoder(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
    )

    # Create the decoupled prior module
    rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
    decoupled_prior = DecoupledPrior(
        latent_dim=model_config.vae_latent_dim,
        num_layers=model_config.vae_prior_num_layers,
        hidden_dims=model_config.vae_prior_hidden_dims,
        rngs=rngs,
        activation=model_config.vae_prior_activation,
        mask_type=model_config.vae_prior_mask_type,
    )

    # Return functions that use the pre-created modules
    def configured_model(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """FIXED: Model function uses decoupled prior"""
        return nbdm_dpvae_model(
            n_cells,
            n_genes,
            model_config,
            decoder,
            decoupled_prior,
            counts,
            batch_size,
        )

    def configured_guide(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """FIXED: Guide function uses encoder outputs properly"""
        return nbdm_vae_guide(
            n_cells,
            n_genes,
            model_config,
            encoder,
            counts,
            batch_size,
        )

    return configured_model, configured_guide


def make_zinb_dpvae_model_and_guide(
    n_genes: int,
    model_config: ModelConfig,
):
    """
    FIXED: Construct and return dpVAE model and guide functions for ZINB model.
    """
    # Create the encoder and decoder modules once
    decoder = create_decoder(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        output_activation=model_config.vae_output_activation,
    )

    encoder = create_encoder(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
    )

    # Create the decoupled prior module
    rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
    decoupled_prior = DecoupledPrior(
        latent_dim=model_config.vae_latent_dim,
        num_layers=model_config.vae_prior_num_layers,
        hidden_dims=model_config.vae_prior_hidden_dims,
        rngs=rngs,
        activation=model_config.vae_prior_activation,
        mask_type=model_config.vae_prior_mask_type,
    )

    # Return functions that use the pre-created modules
    def configured_model(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """FIXED: Model function uses decoupled prior"""
        return zinb_dpvae_model(
            n_cells,
            n_genes,
            model_config,
            decoder,
            decoupled_prior,
            counts,
            batch_size,
        )

    def configured_guide(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """FIXED: Guide function uses encoder outputs properly"""
        return zinb_vae_guide(
            n_cells,
            n_genes,
            model_config,
            encoder,
            counts,
            batch_size,
        )

    return configured_model, configured_guide

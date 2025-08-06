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
from ..vae.architectures import (
    Encoder,
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
    VAE-based Numpyro model for Negative Binomial-Dirichlet Multinomial data.

    The VAE generates r parameters for each cell while keeping p interpretable.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)

    # Sample global success probability p from Beta prior
    p = numpyro.sample("p", dist.Beta(*p_prior_params))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
            log_r = numpyro.deterministic("log_r", decoder_module(z))
            r = numpyro.deterministic("r", jnp.exp(log_r))

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
    Mean-field variational guide for the VAE-based NBDM model.
    """
    # Define guide parameters for p
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)

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
    decoder: Decoder,  # Pre-created decoder passed as argument
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
                r_params = decoder_module(z)

                # Construct the base Negative Binomial distribution using r and
                # p
                base_dist = dist.NegativeBinomialProbs(r_params, p)
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
                r_params = decoder_module(z)

                # Construct the base Negative Binomial distribution using r and
                # p
                base_dist = dist.NegativeBinomialProbs(r_params, p)
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
            r_params = decoder_module(z)

            # Construct the base Negative Binomial distribution using r and p
            base_dist = dist.NegativeBinomialProbs(r_params, p)
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
            # Generate dummy data for encoder
            dummy_data = jnp.zeros((n_cells, n_genes))
            z_mean, z_logvar = encoder_module(dummy_data)
            z_std = jnp.exp(0.5 * z_logvar)

            # Sample from variational distribution
            numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))


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
    VAE-based Numpyro model for Negative Binomial-Dirichlet Multinomial with
    variable capture probability.

    The VAE generates r parameters for each cell while keeping p interpretable.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample global success probability p from Beta prior
    p = numpyro.sample("p", dist.Beta(*p_prior_params))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
            log_r = numpyro.deterministic("log_r", decoder_module(z))
            r = numpyro.deterministic("r", jnp.exp(log_r))

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
    encoder: Encoder,  # Pre-created encoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based NBVCP model.
    """
    # Define guide parameters for p
    p_prior_params = model_config.p_param_guide or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_guide or (1.0, 1.0)

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

    # Register the pre-created encoder as NumPyro module for the guide
    encoder_module = nnx_module("encoder", encoder)

    # Set up cell-specific capture probability parameters
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

                # Sample cell-specific capture probability from Beta prior
                numpyro.sample(
                    "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
                )
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

                # Sample cell-specific capture probability from Beta prior
                numpyro.sample(
                    "p_capture",
                    dist.Beta(p_capture_alpha[idx], p_capture_beta[idx]),
                )
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Generate dummy data for encoder
            dummy_data = jnp.zeros((n_cells, n_genes))
            z_mean, z_logvar = encoder_module(dummy_data)
            z_std = jnp.exp(0.5 * z_logvar)

            # Sample from variational distribution
            numpyro.sample("z", dist.Normal(z_mean, z_std).to_event(1))

            # Sample cell-specific capture probability from Beta prior
            numpyro.sample(
                "p_capture", dist.Beta(p_capture_alpha, p_capture_beta)
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
    Generative model for NBDM dpVAE.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)

    # Sample global parameters
    p = numpyro.sample("p", dist.Beta(*p_prior_params))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
            log_r = numpyro.deterministic("log_r", decoder_module(z))
            r = numpyro.deterministic("r", jnp.exp(log_r))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
            log_r = numpyro.deterministic("log_r", decoder_module(z))
            r = numpyro.deterministic("r", jnp.exp(log_r))

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
    VAE-based Numpyro model for Negative Binomial-Dirichlet Multinomial with
    variable capture probability.

    The VAE generates r parameters for each cell while keeping p interpretable.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)
    p_capture_prior_params = model_config.p_capture_param_prior or (1.0, 1.0)

    # Sample global success probability p from Beta prior
    p = numpyro.sample("p", dist.Beta(*p_prior_params))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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
                log_r = numpyro.deterministic("log_r", decoder_module(z))
                r = numpyro.deterministic("r", jnp.exp(log_r))

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

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p_hat).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample z from decoupled prior (KEY DIFFERENCE from standard VAE)
            z = numpyro.sample("z", decoupled_prior_dist)

            # Use decoder to generate r parameters from latent space
            log_r = numpyro.deterministic("log_r", decoder_module(z))
            r = numpyro.deterministic("r", jnp.exp(log_r))

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
    estimated parameters for the dpVAE-standard parameterization.

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

    # p parameter (Beta distribution)
    if "p_alpha" in params and "p_beta" in params:
        distributions["p"] = dist.Beta(params["p_alpha"], params["p_beta"])

    # gate parameter (Beta distribution)
    if "gate_alpha" in params and "gate_beta" in params:
        distributions["gate"] = dist.Beta(
            params["gate_alpha"], params["gate_beta"]
        )

    # p_capture parameter (Beta distribution)
    if "p_capture_alpha" in params and "p_capture_beta" in params:
        distributions["p_capture"] = dist.Beta(
            params["p_capture_alpha"], params["p_capture_beta"]
        )

    # mixing_weights parameter (Dirichlet distribution)
    if "mixing_concentrations" in params:
        mixing_dist = dist.Dirichlet(params["mixing_concentrations"])
        distributions["mixing_weights"] = mixing_dist

    # Get the decoupled prior distribution
    distributions["z"] = vae_model.get_prior_distribution()

    return distributions

"""
VAE-based models for single-cell RNA sequencing data using odds ratio
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
    create_encoder,
    create_decoder,
    Encoder,
    Decoder,
    VAE,
    DecoupledPrior,
    DecoupledPriorDistribution
    )
from ..stats import BetaPrime


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
    with odds ratio parameterization.

    The VAE generates mu parameters for each cell while keeping phi
    interpretable.
    """
    # Define prior parameters
    phi_prior_params = model_config.phi_param_prior or (1.0, 1.0)

    # Sample global odds ratio phi from BetaPrime prior
    phi = numpyro.sample("phi", BetaPrime(*phi_prior_params))

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

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialLogits(
                    r, -jnp.log(phi)
                ).to_event(1)
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

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialLogits(
                    r, -jnp.log(phi)
                ).to_event(1)
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

            # Use decoder to generate mu parameters from latent space
            log_mu = numpyro.deterministic("log_mu", decoder_module(z))
            mu = numpyro.deterministic("mu", jnp.exp(log_mu))

            # Compute r using the odds ratio parameterization
            r = numpyro.deterministic("r", mu * phi)

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi)).to_event(
                1
            )
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
    Mean-field variational guide for the VAE-based NBDM model with odds ratio
    parameterization.
    """
    # Define guide parameters for phi and mu
    phi_prior_params = model_config.phi_param_guide or (1.0, 1.0)

    # Register phi_alpha as a variational parameter with positivity constraint
    phi_alpha = numpyro.param(
        "phi_alpha", phi_prior_params[0], constraint=constraints.positive
    )
    # Register phi_beta as a variational parameter with positivity constraint
    phi_beta = numpyro.param(
        "phi_beta", phi_prior_params[1], constraint=constraints.positive
    )
    # Sample phi from the BetaPrime distribution parameterized by phi_alpha and
    # phi_beta
    numpyro.sample("phi", BetaPrime(phi_alpha, phi_beta))

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
            # Extract latent dimension from model config
            latent_dim = model_config.vae_latent_dim

            # Sample from variational distribution
            numpyro.sample(
                "z", dist.Normal(0, 1).expand([latent_dim]).to_event(1)
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
    odds ratio parameterization.
    """
    # Get prior parameters for phi (odds ratio), mu (mean), and gate
    phi_prior_params = model_config.phi_param_prior or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample phi from a BetaPrime distribution with the specified prior
    # parameters
    phi = numpyro.sample("phi", BetaPrime(*phi_prior_params))
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

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Construct the base Negative Binomial distribution using r and
                # phi
                base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
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

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Construct the base Negative Binomial distribution using r and
                # phi
                base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
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

            # Use decoder to generate mu parameters from latent space
            log_mu = numpyro.deterministic("log_mu", decoder_module(z))
            mu = numpyro.deterministic("mu", jnp.exp(log_mu))

            # Compute r using the odds ratio parameterization
            r = numpyro.deterministic("r", mu * phi)

            # Construct the base Negative Binomial distribution using r and phi
            base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
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
    encoder: Encoder,  # Pre-created encoder passed as argument
    counts=None,
    batch_size=None,
):
    """
    Mean-field variational guide for the VAE-based ZINB model with odds ratio
    parameterization.
    """
    # Define guide parameters for phi, mu, and gate
    phi_prior_params = model_config.phi_param_guide or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_guide or (1.0, 1.0)

    # Register variational parameters for phi (odds ratio)
    phi_alpha = numpyro.param(
        "phi_alpha", phi_prior_params[0], constraint=constraints.positive
    )
    phi_beta = numpyro.param(
        "phi_beta", phi_prior_params[1], constraint=constraints.positive
    )
    numpyro.sample("phi", BetaPrime(phi_alpha, phi_beta))

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
# dpVAE Model Functions (for decoupled prior)
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
    Generative model for NBDM dpVAE with odds ratio parameterization.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Define prior parameters
    phi_prior_params = model_config.phi_param_prior or (1.0, 1.0)

    # Sample global odds ratio phi from BetaPrime prior
    phi = numpyro.sample("phi", BetaPrime(*phi_prior_params))

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
                # Sample z from decoupled prior (KEY DIFFERENCE from standard VAE)
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialLogits(
                    r, -jnp.log(phi)
                ).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample z from decoupled prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Sample observed counts
                batch_counts = counts[idx] if counts is not None else None
                numpyro.sample(
                    "counts",
                    dist.NegativeBinomialLogits(r, -jnp.log(phi)).to_event(1),
                    obs=batch_counts,
                )
    else:
        # Without counts: for prior predictive sampling
        with numpyro.plate("cells", n_cells):
            # Sample from latent space prior
            z = numpyro.sample("z", decoupled_prior_dist)

            # Use decoder to generate mu parameters from latent space
            log_mu = numpyro.deterministic("log_mu", decoder_module(z))
            mu = numpyro.deterministic("mu", jnp.exp(log_mu))

            # Compute r using the odds ratio parameterization
            r = numpyro.deterministic("r", mu * phi)

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi)).to_event(
                1
            )
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
    Generative model for ZINB dpVAE with odds ratio parameterization.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    """
    # Get prior parameters for phi (odds ratio) and gate
    phi_prior_params = model_config.phi_param_prior or (1.0, 1.0)
    gate_prior_params = model_config.gate_param_prior or (1.0, 1.0)

    # Sample phi from a BetaPrime distribution with the specified prior
    # parameters
    phi = numpyro.sample("phi", BetaPrime(*phi_prior_params))
    # Sample gate from a Beta distribution with the specified prior parameters,
    # expanded to n_genes
    gate = numpyro.sample(
        "gate", dist.Beta(*gate_prior_params).expand([n_genes])
    )

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

    # If observed counts are provided
    if counts is not None:
        # If no batching, use a plate over all cells
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample from latent space prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Construct the base Negative Binomial distribution using r and
                # phi
                base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
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

                # Use decoder to generate mu parameters from latent space
                log_mu = numpyro.deterministic("log_mu", decoder_module(z))
                mu = numpyro.deterministic("mu", jnp.exp(log_mu))

                # Compute r using the odds ratio parameterization
                r = numpyro.deterministic("r", mu * phi)

                # Construct the base Negative Binomial distribution using r and
                # phi
                base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
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

            # Use decoder to generate mu parameters from latent space
            log_mu = numpyro.deterministic("log_mu", decoder_module(z))
            mu = numpyro.deterministic("mu", jnp.exp(log_mu))

            # Compute r using the odds ratio parameterization
            r = numpyro.deterministic("r", mu * phi)

            # Construct the base Negative Binomial distribution using r and phi
            base_dist = dist.NegativeBinomialLogits(r, -jnp.log(phi))
            # Construct the zero-inflated distribution using the base NB and
            # gate
            zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(
                1
            )
            # Sample counts (not observed)
            numpyro.sample("counts", zinb)


# ------------------------------------------------------------------------------
# Posterior Distributions for VAE-based models with odds ratio parameterization
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    vae_model: VAE,
) -> Dict[str, dist.Distribution]:
    """
    Constructs and returns a dictionary of posterior distributions from
    estimated parameters for the dpVAE-odds ratio parameterization.

    This function builds the appropriate `numpyro` distributions based on the
    guide parameters found in the `params` dictionary. All distributions are
    returned as batch distributions (no splitting or per-component/gene lists).

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        A dictionary of estimated parameters from the variational guide.
    model_config : ModelConfig
        The model configuration object.
    vae_model : VAE
        The VAE model object.

    Returns
    -------
    Dict[str, dist.Distribution]
        Dictionary mapping parameter names to their posterior distributions.
    """
    distributions = {}

    # phi parameter (BetaPrime distribution)
    if "phi_alpha" in params and "phi_beta" in params:
        distributions["phi"] = BetaPrime(
            params["phi_alpha"], params["phi_beta"]
        )

    # gate parameter (Beta distribution)
    if "gate_alpha" in params and "gate_beta" in params:
        distributions["gate"] = dist.Beta(
            params["gate_alpha"], params["gate_beta"]
        )

    # phi_capture parameter (BetaPrime distribution)
    if "phi_capture_alpha" in params and "phi_capture_beta" in params:
        distributions["phi_capture"] = BetaPrime(
            params["phi_capture_alpha"], params["phi_capture_beta"]
        )

    # mixing_weights parameter (Dirichlet distribution)
    if "mixing_concentrations" in params:
        mixing_dist = dist.Dirichlet(params["mixing_concentrations"])
        distributions["mixing_weights"] = mixing_dist

    # Get the decoupled prior distribution
    distributions["z"] = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    )

    return distributions

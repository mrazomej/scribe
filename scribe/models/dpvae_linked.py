"""
dpVAE-based models for single-cell RNA sequencing data using linked
parameterization with decoupled prior.

The key difference from standard VAE is that the GUIDE uses a
DecoupledPriorDistribution instead of a standard Normal prior, while the MODEL
remains the same.

This parameterization differs from the standard parameterization in that the
mean parameter (mu) is linked to the success probability parameter (p) through
the relationship:
    r = mu * (1 - p) / p
where r is the dispersion parameter. The VAE generates mu parameters for each
cell while keeping p interpretable.
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
from .vae_linked import nbdm_vae_guide, zinb_vae_guide
from ..vae.architectures import (
    create_encoder,
    create_decoder,
    Encoder,
    Decoder,
    DecoupledPrior,
    DecoupledPriorDistribution,
    dpVAE,
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
    Generative model for NBDM dpVAE with linked parameterization.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    The VAE generates mu parameters for each cell while keeping p interpretable.
    The relationship r = mu * (1 - p) / p links the parameters.
    """
    # Define prior parameters
    p_prior_params = model_config.p_param_prior or (1.0, 1.0)

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

    # Sample latent variables and generate observations
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample z from decoupled prior (KEY DIFFERENCE from standard VAE)
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate mu parameters from latent space
                mu = numpyro.deterministic("mu", decoder_module(z))

                # Compute r using the linked parameterization
                r = numpyro.deterministic("r", mu * (1 - p) / p)

                # Define base distribution with VAE-generated r
                base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample z from decoupled prior
                z = numpyro.sample("z", decoupled_prior_dist)

                # Use decoder to generate mu parameters from latent space
                mu = numpyro.deterministic("mu", decoder_module(z))

                # Compute r using the linked parameterization
                r = numpyro.deterministic("r", mu * (1 - p) / p)

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

            # Use decoder to generate mu parameters from latent space
            mu = numpyro.deterministic("mu", decoder_module(z))

            # Compute r using the linked parameterization
            r = numpyro.deterministic("r", mu * (1 - p) / p)

            # Define base distribution with VAE-generated r
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)
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
    Generative model for ZINB dpVAE with linked parameterization.

    KEY DIFFERENCE: Uses DecoupledPriorDistribution as the prior for z.
    The VAE generates mu parameters for each cell while keeping p interpretable.
    The relationship r = mu * (1 - p) / p links the parameters.
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
                mu = numpyro.deterministic("mu", decoder_module(z))

                # Compute r using the linked parameterization
                r = numpyro.deterministic("r", mu * (1 - p) / p)

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

                # Use decoder to generate mu parameters from latent space
                mu = numpyro.deterministic("mu", decoder_module(z))

                # Compute r using the linked parameterization
                r = numpyro.deterministic("r", mu * (1 - p) / p)

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

            # Use decoder to generate mu parameters from latent space
            mu = numpyro.deterministic("mu", decoder_module(z))

            # Compute r using the linked parameterization
            r = numpyro.deterministic("r", mu * (1 - p) / p)

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
# NBDM dpVAE Model/Guide Factory Functions
# ------------------------------------------------------------------------------


def make_nbdm_dpvae_model_and_guide(
    n_genes: int,
    model_config: ModelConfig,
):
    """
    FIXED: Construct and return dpVAE model and guide functions for NBDM model
    with linked parameterization.
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
        """FIXED: Model function uses decoupled prior with linked parameterization"""
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
        """FIXED: Guide function uses encoder outputs properly with linked parameterization"""
        return nbdm_vae_guide(
            n_cells,
            n_genes,
            model_config,
            encoder,
            counts,
            batch_size,
        )

    return configured_model, configured_guide


# ------------------------------------------------------------------------------
# ZINB dpVAE Model/Guide Factory Functions
# ------------------------------------------------------------------------------


def make_zinb_dpvae_model_and_guide(
    n_genes: int,
    model_config: ModelConfig,
):
    """
    FIXED: Construct and return dpVAE model and guide functions for ZINB model
    with linked parameterization.
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
        """FIXED: Model function uses decoupled prior with linked parameterization"""
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
        """FIXED: Guide function uses encoder outputs properly with linked parameterization"""
        return zinb_vae_guide(
            n_cells,
            n_genes,
            model_config,
            encoder,
            counts,
            batch_size,
        )

    return configured_model, configured_guide


# ------------------------------------------------------------------------------
# Get posterior distributions for SVI results
# ------------------------------------------------------------------------------


def get_posterior_distributions(
    params: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    vae_model: dpVAE,
) -> Dict[str, dist.Distribution]:
    """
    Constructs and returns a dictionary of posterior distributions from
    estimated parameters for the dpVAE-linked parameterization.

    This function builds the appropriate `numpyro` distributions based on the
    guide parameters found in the `params` dictionary. All distributions are
    returned as batch distributions (no splitting or per-component/gene lists).

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        A dictionary of estimated parameters from the variational guide.
    model_config : ModelConfig
        The model configuration object.
    vae_model : dpVAE
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
    decoupled_prior_dist = vae_model.get_decoupled_prior_distribution()
    distributions["z"] = decoupled_prior_dist

    return distributions 
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
    create_encoder,
    create_decoder,
    Encoder,
    Decoder,
    VAE,
)


# ------------------------------------------------------------------------------
# VAE Model/Guide Factory Functions
# ------------------------------------------------------------------------------


def make_nbdm_vae_model_and_guide(
    n_genes: int,
    model_config: ModelConfig,
):
    """
    Construct and return VAE model and guide functions for the Negative
    Binomial-Dirichlet Multinomial (NBDM) model, reusing the same encoder and
    decoder modules throughout the SVI optimization.

    This factory function instantiates the encoder and decoder neural network
    modules only once, using the provided model configuration and number of
    genes. It then returns two functions: a model and a guide, each of which
    accepts the number of cells, number of genes, model configuration, and
    optionally count data and batch size. These returned functions internally
    use the pre-created encoder and decoder modules, ensuring that the neural
    network parameters are not re-initialized at every SVI step, which is
    important for correct and efficient variational inference.

    Parameters
    ----------
    n_genes : int
        Number of genes (input dimension for encoder/decoder).
    model_config : ModelConfig
        Configuration object specifying VAE architecture and model
        hyperparameters.

    Returns
    -------
    configured_model : Callable
        A function implementing the NBDM VAE model, using the pre-created
        decoder.
    configured_guide : Callable
        A function implementing the NBDM VAE guide, using the pre-created
        encoder.
    """
    # Create the modules once
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

    # Return functions that use the pre-created modules
    def configured_model(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """
        Model function for the NBDM VAE, using the shared decoder module.

        Parameters
        ----------
        n_cells : int
            Number of cells in the dataset.
        n_genes : int
            Number of genes.
        model_config : ModelConfig
            Model configuration object.
        counts : Optional[jnp.ndarray]
            Observed count data, if available.
        batch_size : Optional[int]
            Batch size for mini-batch training, if applicable.

        Returns
        -------
        None
        """
        return nbdm_vae_model(
            n_cells, n_genes, model_config, decoder, counts, batch_size
        )

    def configured_guide(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """
        Guide function for the NBDM VAE, using the shared encoder module.

        Parameters
        ----------
        n_cells : int
            Number of cells in the dataset.
        n_genes : int
            Number of genes.
        model_config : ModelConfig
            Model configuration object.
        counts : Optional[jnp.ndarray]
            Observed count data, if available.
        batch_size : Optional[int]
            Batch size for mini-batch training, if applicable.

        Returns
        -------
        None
        """
        return nbdm_vae_guide(
            n_cells, n_genes, model_config, encoder, counts, batch_size
        )

    return configured_model, configured_guide


def make_zinb_vae_model_and_guide(
    n_genes: int,
    model_config: ModelConfig,
):
    """
    Construct and return VAE model and guide functions for the Zero-Inflated
    Negative Binomial (ZINB) model, reusing the same encoder and decoder modules
    throughout the SVI optimization.

    This factory function instantiates the encoder and decoder neural network
    modules only once, using the provided model configuration and number of
    genes. It then returns two functions: a model and a guide, each of which
    accepts the number of cells, number of genes, model configuration, and
    optionally count data and batch size. These returned functions internally
    use the pre-created encoder and decoder modules, ensuring that the neural
    network parameters are not re-initialized at every SVI step, which is
    important for correct and efficient variational inference.

    Parameters
    ----------
    n_genes : int
        Number of genes (input dimension for encoder/decoder).
    model_config : ModelConfig
        Configuration object specifying VAE architecture and model
        hyperparameters.

    Returns
    -------
    configured_model : Callable
        A function implementing the ZINB VAE model, using the pre-created
        decoder.
    configured_guide : Callable
        A function implementing the ZINB VAE guide, using the pre-created
        encoder.
    """
    # Create the modules once
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

    # Return functions that use the pre-created modules
    def configured_model(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """
        Model function for the ZINB VAE, using the shared decoder module.

        Parameters
        ----------
        n_cells : int
            Number of cells in the dataset.
        n_genes : int
            Number of genes.
        model_config : ModelConfig
            Model configuration object.
        counts : Optional[jnp.ndarray]
            Observed count data, if available.
        batch_size : Optional[int]
            Batch size for mini-batch training, if applicable.

        Returns
        -------
        None
        """
        return zinb_vae_model(
            n_cells, n_genes, model_config, decoder, counts, batch_size
        )

    def configured_guide(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """
        Guide function for the ZINB VAE, using the shared encoder module.

        Parameters
        ----------
        n_cells : int
            Number of cells in the dataset.
        n_genes : int
            Number of genes.
        model_config : ModelConfig
            Model configuration object.
        counts : Optional[jnp.ndarray]
            Observed count data, if available.
        batch_size : Optional[int]
            Batch size for mini-batch training, if applicable.

        Returns
        -------
        None
        """
        return zinb_vae_guide(
            n_cells, n_genes, model_config, encoder, counts, batch_size
        )

    return configured_model, configured_guide


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
    distributions["z"] = dist.Normal(
        jnp.zeros(model_config.vae_latent_dim),
        jnp.ones(model_config.vae_latent_dim),
    )

    return distributions
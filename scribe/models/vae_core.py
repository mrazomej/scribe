"""
Centralized VAE factory for creating model and guide functions.

This module provides a unified interface for creating VAE and dpVAE models
across all parameterizations, eliminating code duplication between standard VAE
and dpVAE implementations.
"""

from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from flax import nnx

from ..vae.architectures import (
    create_encoder,
    create_decoder,
    DecoupledPrior,
)
from .model_config import ModelConfig

# ------------------------------------------------------------------------------
# VAE Core Factory
# ------------------------------------------------------------------------------

def make_vae_model_and_guide(
    model_type: str,
    n_genes: int,
    model_config: ModelConfig,
    parameterization: str,
    prior_type: str = "standard",
) -> Tuple[Callable, Callable]:
    """
    Factory function to create VAE model and guide functions for any
    parameterization and prior type.

    This function centralizes the construction of model and guide functions for
    both standard VAE and dpVAE architectures, supporting multiple
    parameterizations ("standard", "linked", "odds_ratio") and prior types
    ("standard", "decoupled"). It instantiates the encoder, decoder, and (if
    needed) decoupled prior modules once, and returns model/guide callables that
    use these modules.

    Parameters
    ----------
    model_type : str
        The type of generative model ("nbdm" or "zinb").
    n_genes : int
        Number of genes (input dimension for encoder/decoder).
    model_config : ModelConfig
        Configuration object specifying VAE architecture and hyperparameters.
    parameterization : str
        The parameterization to use ("standard", "linked", or "odds_ratio").
    prior_type : str, default="standard"
        The prior type to use ("standard" or "decoupled").

    Returns
    -------
    Tuple[Callable, Callable]
        (configured_model, configured_guide): Functions implementing the VAE
        model and guide, using the pre-created modules.
    """
    # Create the decoder module for the generative model
    decoder = create_decoder(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        # Optionally pass standardization parameters if present in config
        standardize_mean=(
            model_config.standardize_mean
            if hasattr(model_config, "standardize_mean")
            else None
        ),
        standardize_std=(
            model_config.standardize_std
            if hasattr(model_config, "standardize_std")
            else None
        ),
    )

    # Determine if variable capture should be enabled based on model type
    variable_capture = (model_type == "nbvcp")

    # Create the encoder module for the inference model
    encoder = create_encoder(
        input_dim=n_genes,
        latent_dim=model_config.vae_latent_dim,
        hidden_dims=model_config.vae_hidden_dims,
        activation=model_config.vae_activation,
        # Optionally pass standardization parameters if present in config
        standardize_mean=(
            model_config.standardize_mean
            if hasattr(model_config, "standardize_mean")
            else None
        ),
        standardize_std=(
            model_config.standardize_std
            if hasattr(model_config, "standardize_std")
            else None
        ),
        variable_capture=variable_capture,
    )

    # Initialize decoupled prior if required by prior_type
    decoupled_prior = None
    if prior_type == "decoupled":
        # Create a random number generator for the decoupled prior
        rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
        # Use the configured number of layers, or a default if not set
        num_layers = model_config.vae_prior_num_layers
        if num_layers is None:
            num_layers = 2  # Default to 2 layers if not specified
        # Instantiate the decoupled prior module
        decoupled_prior = DecoupledPrior(
            latent_dim=model_config.vae_latent_dim,
            num_layers=num_layers,
            hidden_dims=model_config.vae_prior_hidden_dims,
            rngs=rngs,
            activation=model_config.vae_prior_activation,
            mask_type=model_config.vae_prior_mask_type,
        )

    # Dynamically import the parameterization module based on the argument
    if parameterization == "standard":
        from . import vae_standard as param_module
    elif parameterization == "linked":
        from . import vae_linked as param_module
    elif parameterization == "odds_ratio":
        from . import vae_odds_ratio as param_module
    elif parameterization == "unconstrained":
        from . import vae_unconstrained as param_module
    else:
        raise ValueError(f"Unsupported parameterization: {parameterization}")

    # Select the appropriate model and guide functions for the chosen prior and
    # model type
    if prior_type == "standard":
        # For standard prior, use the regular model and guide functions
        if model_type == "nbdm":
            model_fn = param_module.nbdm_vae_model
            guide_fn = param_module.nbdm_vae_guide
        elif model_type == "zinb":
            model_fn = param_module.zinb_vae_model
            guide_fn = param_module.zinb_vae_guide
        elif model_type == "nbvcp":
            model_fn = param_module.nbvcp_vae_model
            guide_fn = param_module.nbvcp_vae_guide
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:  # prior_type == "decoupled"
        # For decoupled prior, use the dpVAE model function and the same guide
        # as standard VAE
        if model_type == "nbdm":
            model_fn = param_module.nbdm_dpvae_model
            guide_fn = param_module.nbdm_vae_guide  # Guide is shared for dpVAE
        elif model_type == "zinb":
            model_fn = param_module.zinb_dpvae_model
            guide_fn = param_module.zinb_vae_guide  # Guide is shared for dpVAE
        elif model_type == "nbvcp":
            model_fn = param_module.nbvcp_dpvae_model
            guide_fn = param_module.nbvcp_vae_guide  # Guide is shared for dpVAE
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # Define a closure for the model function, injecting the created modules
    def configured_model(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """Model function using pre-created modules."""
        # For standard prior, call model_fn with decoder only
        if prior_type == "standard":
            return model_fn(
                n_cells, n_genes, model_config, decoder, counts, batch_size
            )
        else:
            # For decoupled prior, call model_fn with decoder and
            # decoupled_prior
            return model_fn(
                n_cells, n_genes, model_config, decoder, decoupled_prior,
                counts, batch_size
            )

    # Define a closure for the guide function, injecting the created encoder
    def configured_guide(
        n_cells, n_genes, model_config, counts=None, batch_size=None
    ):
        """Guide function using pre-created encoder."""
        return guide_fn(
            n_cells, n_genes, model_config, encoder, counts, batch_size
        )

    # Return the configured model and guide functions
    return configured_model, configured_guide 
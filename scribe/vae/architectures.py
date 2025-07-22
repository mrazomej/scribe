"""
Default VAE architectures for scRNA-seq data using Flax NNX.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """Configuration for VAE architecture."""

    input_dim: int
    latent_dim: int = 3
    hidden_dims: List[int] = None  # List of hidden layer dimensions
    activation: Callable = nnx.gelu

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]  # Default: 2 hidden layers of 256


# ------------------------------------------------------------------------------
# Encoder class
# ------------------------------------------------------------------------------


class Encoder(nnx.Module):
    """Encoder for VAE architecture."""

    def __init__(self, config: VAEConfig, *, rngs: nnx.Rngs):
        # Build encoder layers dynamically based on config
        self.encoder_layers = []

        # First layer: input_dim -> first hidden_dim
        self.encoder_layers.append(
            nnx.Linear(config.input_dim, config.hidden_dims[0], rngs=rngs)
        )

        # Hidden layers 
        # hidden_dims[0] -> hidden_dims[1] -> ... -> hidden_dims[-1]
        for i in range(len(config.hidden_dims) - 1):
            self.encoder_layers.append(
                nnx.Linear(
                    config.hidden_dims[i], config.hidden_dims[i + 1], rngs=rngs
                )
            )

        # Latent space projections (mean and log variance)
        last_hidden_dim = config.hidden_dims[-1]
        self.latent_mean = nnx.Linear(
            last_hidden_dim, config.latent_dim, rngs=rngs
        )
        self.latent_logvar = nnx.Linear(
            last_hidden_dim, config.latent_dim, rngs=rngs
        )

        self.config = config

    def encode(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Encode input data to latent space.

        Args:
            x: Input data of shape (batch_size, input_dim)

        Returns:
            Tuple of (mean, log_variance) for latent space
        """
        # Encoder forward pass through all hidden layers
        h = x
        for layer in self.encoder_layers:
            h = self.config.activation(layer(h))

        # Project to latent space
        mean = self.latent_mean(h)
        logvar = self.latent_logvar(h)

        return mean, logvar

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self.encode(x)


# ------------------------------------------------------------------------------
# Decoder class
# ------------------------------------------------------------------------------


class Decoder(nnx.Module):
    """Decoder for VAE architecture."""

    def __init__(self, config: VAEConfig, *, rngs: nnx.Rngs):
        # Build decoder layers dynamically based on config (reverse of encoder)
        self.decoder_layers = []

        # First layer: latent_dim -> last hidden_dim
        self.decoder_layers.append(
            nnx.Linear(config.latent_dim, config.hidden_dims[-1], rngs=rngs)
        )

        # Hidden layers (reverse order)
        for i in range(len(config.hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(
                nnx.Linear(
                    config.hidden_dims[i], config.hidden_dims[i - 1], rngs=rngs
                )
            )

        # Output layer: last hidden_dim -> input_dim
        self.decoder_output = nnx.Linear(
            config.hidden_dims[0], config.input_dim, rngs=rngs
        )

        self.config = config

    def decode(self, z: jax.Array) -> jax.Array:
        """
        Decode latent representation to output.

        Args:
            z: Latent representation of shape (batch_size, latent_dim)

        Returns:
            Decoded output of shape (batch_size, input_dim)
        """
        # Decoder forward pass through all hidden layers
        h = z
        for layer in self.decoder_layers:
            h = self.config.activation(layer(h))

        # Output layer (no activation for r parameters)
        output = self.decoder_output(h)

        return output

    def __call__(self, z: jax.Array) -> jax.Array:
        return self.decode(z)


# ------------------------------------------------------------------------------
# VAE class
# ------------------------------------------------------------------------------


class VAE(nnx.Module):
    """
    VAE architecture that combines an encoder and decoder.

    The VAE handles the reparameterization trick and coordinates the
    encoding/decoding process.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        config: VAEConfig,
        *,
        rngs: nnx.Rngs,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.rngs = rngs

    # --------------------------------------------------------------------------

    def reparameterize(
        self, mean: jax.Array, logvar: jax.Array, training: bool = True
    ) -> jax.Array:
        """
        Reparameterization trick to sample from latent distribution.

        Args:
            mean: Mean of latent distribution
            logvar: Log variance of latent distribution
            training: Whether in training mode (affects sampling)

        Returns:
            Sampled latent representation
        """
        if training:
            # Reparameterization trick
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(self.rngs.params(), mean.shape)
            z = mean + eps * std
        else:
            z = mean
        return z

    # --------------------------------------------------------------------------

    def __call__(
        self, x: jax.Array, training: bool = True
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Forward pass through the VAE.

        Args:
            x: Input data of shape (batch_size, input_dim)
            training: Whether in training mode (affects sampling)

        Returns:
            Tuple of (reconstructed_x, mean, logvar)
        """
        # Encode
        mean, logvar = self.encoder.encode(x)

        # Sample from latent space
        z = self.reparameterize(mean, logvar, training)

        # Decode
        reconstructed = self.decoder.decode(z)

        return reconstructed, mean, logvar

    # --------------------------------------------------------------------------

    def get_state(self):
        """
        Get the trained state from the VAE model using Flax NNX.

        Returns:
            nnx.State containing the model state
        """
        # Use Flax NNX's split to get the state
        _, state = nnx.split(self)
        return state

    # --------------------------------------------------------------------------

    def load_state(self, state):
        """
        Load trained state into the VAE model using Flax NNX.

        Args:
            state: nnx.State containing the model state
        """
        # Use Flax NNX's merge to load the state
        graphdef, _ = nnx.split(self)
        self.__dict__.update(nnx.merge(graphdef, state).__dict__)

# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------


def create_vae(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[Callable] = None,
) -> VAE:
    """
    Create a default VAE architecture with configurable hidden layers.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)

    Returns:
        Configured VAE instance
    """
    if activation is None:
        activation = nnx.gelu

    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )

    rngs = nnx.Rngs(params=0)

    # Create encoder and decoder
    encoder = Encoder(config=config, rngs=rngs)
    decoder = Decoder(config=config, rngs=rngs)

    # Create and return VAE
    return VAE(encoder=encoder, decoder=decoder, config=config, rngs=rngs)

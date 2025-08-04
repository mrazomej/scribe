"""
Default VAE architectures for scRNA-seq data using Flax NNX.
"""

import jax
import jax.numpy as jnp
from flax import nnx
import numpyro
from numpyro.distributions.util import validate_sample
import numpyro.distributions as dist
from typing import Tuple, Optional, Callable, List, Dict
from dataclasses import dataclass


# ------------------------------------------------------------------------------
# Dictionary of functions
# ------------------------------------------------------------------------------

ACTIVATION_FUNCTIONS = {
    "celu": nnx.celu,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
    "glu": nnx.glu,
    "hard_sigmoid": nnx.hard_sigmoid,
    "hard_silu": nnx.hard_silu,
    "hard_swish": nnx.hard_swish,
    "hard_tanh": nnx.hard_tanh,
    "leaky_relu": nnx.leaky_relu,
    "log_sigmoid": nnx.log_sigmoid,
    "log_softmax": nnx.log_softmax,
    "logsumexp": nnx.logsumexp,
    "one_hot": nnx.one_hot,
    "relu": nnx.relu,
    "relu6": nnx.relu6,
    "selu": nnx.selu,
    "sigmoid": nnx.sigmoid,
    "silu": nnx.silu,
    "soft_sign": nnx.soft_sign,
    "softmax": nnx.softmax,
    "softplus": nnx.softplus,
    "standardize": nnx.standardize,
    "swish": nnx.swish,
}

# Dictionary of input transformations
INPUT_TRANSFORMATIONS = {
    "log1p": jnp.log1p,
    "log": jnp.log,
    "sqrt": jnp.sqrt,
    "identity": lambda x: x,
}

# ------------------------------------------------------------------------------
# VAEConfig class
# ------------------------------------------------------------------------------


@dataclass
class VAEConfig:
    """
    Configuration dataclass for Variational Autoencoder (VAE) architectures.

    This class encapsulates all hyperparameters and options required to
    construct a VAE model, including encoder/decoder architecture, activation
    functions, input/output transformations, and data standardization
    statistics.

    Attributes
    ----------
    input_dim : int
        The dimensionality of the input data (e.g., number of genes for
        scRNA-seq).
    latent_dim : int, default=3
        The dimensionality of the latent space (number of latent variables).
    hidden_dims : List[int], optional
        List specifying the number of units in each hidden layer of the
        encoder/decoder. If not provided, defaults to two hidden layers of 256
        units each.
    activation : str, default="relu"
        Name of the activation function to use in hidden layers. Must be a key
        in ACTIVATION_FUNCTIONS.
    input_transformation : str, default="log1p"
        Name of the transformation to apply to input data before encoding.
        Common choices for count data include "log1p", "log", "sqrt", or
        "identity".
    standardize_mean : Optional[jnp.ndarray], default=None
        Per-feature (e.g., per-gene) mean for z-standardization. If provided,
        used to standardize input/output.
    standardize_std : Optional[jnp.ndarray], default=None
        Per-feature (e.g., per-gene) standard deviation for z-standardization.
        If provided, used to standardize input/output.

    Notes
    -----
    - This configuration is typically passed to VAE model/guide constructors to
      ensure consistent architecture and preprocessing.
    - Standardization parameters are optional, but recommended for stable
      training, especially when using count data with large dynamic range.
    """

    input_dim: int
    latent_dim: int = 3
    # List of hidden layer dimensions
    hidden_dims: List[int] = None
    activation: str = "relu"
    # Input transformation for encoder (default: log1p for scRNA-seq data)
    input_transformation: str = "log1p"
    # Standardization parameters
    standardize_mean: Optional[jnp.ndarray] = None
    standardize_std: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """
        Post-initialization to set default hidden_dims if not provided.

        If hidden_dims is None, sets it to [256, 256] (two hidden layers of 256 units each).
        """
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]

# ------------------------------------------------------------------------------
# Standardization functions
# ------------------------------------------------------------------------------


def compute_standardization_stats(
    data: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute mean and std for z-standardization per gene.

    Parameters
    ----------
    data : jnp.ndarray
        Input data of shape (n_cells, n_genes)

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Tuple of (mean, std) arrays, each of shape (n_genes,)
    """
    return jnp.mean(data, axis=0), jnp.std(data, axis=0)

# ------------------------------------------------------------------------------

def standardize_data(
    data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """
    Z-standardize data using pre-computed statistics.

    Parameters
    ----------
    data : jnp.ndarray
        Input data of shape (batch_size, n_genes)
    mean : jnp.ndarray
        Mean values per gene, shape (n_genes,)
    std : jnp.ndarray
        Standard deviation values per gene, shape (n_genes,)

    Returns
    -------
    jnp.ndarray
        Standardized data of same shape as input
    """
    return (data - mean) / (
        std + 1e-8
    )  # Add small epsilon for numerical stability

# ------------------------------------------------------------------------------


def destandardize_data(
    data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """
    Reverse z-standardization.

    Parameters
    ----------
    data : jnp.ndarray
        Standardized data of shape (batch_size, n_genes)
    mean : jnp.ndarray
        Mean values per gene, shape (n_genes,)
    std : jnp.ndarray
        Standard deviation values per gene, shape (n_genes,)

    Returns
    -------
    jnp.ndarray
        Destandardized data of same shape as input
    """
    return data * (std + 1e-8) + mean


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
        # Get activation function
        activation = ACTIVATION_FUNCTIONS[self.config.activation]
        input_transformation = INPUT_TRANSFORMATIONS[
            self.config.input_transformation
        ]

        # Apply configurable input transformation
        x = input_transformation(x)

        # Apply standardization if enabled
        if (
            self.config.standardize_mean is not None
            and self.config.standardize_std is not None
        ):
            x = standardize_data(
                x, self.config.standardize_mean, self.config.standardize_std
            )

        # Encoder forward pass through all hidden layers
        h = x
        for layer in self.encoder_layers:
            h = activation(layer(h))

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
        # Get activation function
        activation = ACTIVATION_FUNCTIONS[self.config.activation]

        # Decoder forward pass through all hidden layers
        h = z
        for layer in self.decoder_layers:
            h = activation(layer(h))

        # Output layer (no activation applied)
        output = self.decoder_output(h)

        # Apply destandardization if enabled
        if (
            self.config.standardize_mean is not None
            and self.config.standardize_std is not None
        ):
            output = destandardize_data(
                output,
                self.config.standardize_mean,
                self.config.standardize_std,
            )

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

    def get_prior_distribution(self) -> dist.Distribution:
        """
        Get the prior distribution for the VAE.
        """
        return dist.Normal(
            jnp.zeros(self.config.latent_dim),
            jnp.ones(self.config.latent_dim),
        )

# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------


def create_encoder(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
    standardize_mean: Optional[jnp.ndarray] = None,
    standardize_std: Optional[jnp.ndarray] = None,
) -> Encoder:
    """
    Create a standalone encoder for VAE.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)
        input_transformation: Input transformation function (default: log1p)
        standardize_mean: Mean values for standardization (default: None)
        standardize_std: Standard deviation values for standardization (default: None)

    Returns:
        Configured Encoder instance
    """
    if activation is None:
        activation = "relu"
    if input_transformation is None:
        input_transformation = "log1p"

    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    rngs = nnx.Rngs(params=0)
    return Encoder(config=config, rngs=rngs)


# ------------------------------------------------------------------------------


def create_decoder(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
    standardize_mean: Optional[jnp.ndarray] = None,
    standardize_std: Optional[jnp.ndarray] = None,
) -> Decoder:
    """
    Create a standalone decoder for VAE.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)
        input_transformation: Input transformation function (default: log1p)
        standardize_mean: Mean values for standardization (default: None)
        standardize_std: Standard deviation values for standardization (default: None)

    Returns:
        Configured Decoder instance
    """
    if activation is None:
        activation = "relu"
    if input_transformation is None:
        input_transformation = "log1p"

    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    rngs = nnx.Rngs(params=0)
    return Decoder(config=config, rngs=rngs)


# ------------------------------------------------------------------------------


def create_vae(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
    standardize_mean: Optional[jnp.ndarray] = None,
    standardize_std: Optional[jnp.ndarray] = None,
) -> VAE:
    """
    Create a default VAE architecture with configurable hidden layers.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)
        input_transformation: Input transformation function (default: log1p)
        standardize_mean: Mean values for standardization (default: None)
        standardize_std: Standard deviation values for standardization (default: None)

    Returns:
        Configured VAE instance
    """
    if activation is None:
        activation = "relu"
    if input_transformation is None:
        input_transformation = "log1p"

    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    rngs = nnx.Rngs(params=0)

    # Create encoder and decoder
    encoder = Encoder(config=config, rngs=rngs)
    decoder = Decoder(config=config, rngs=rngs)

    # Create and return VAE
    return VAE(encoder=encoder, decoder=decoder, config=config, rngs=rngs)


# ------------------------------------------------------------------------------
# Affine Coupling Layer for Normalizing Flows
# ------------------------------------------------------------------------------


class AffineCouplingLayer(nnx.Module):
    """
    Single invertible affine coupling block implementing Real NVP.

    This implements the affine coupling transformation from Real NVP:
    - Forward: y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁), y₁ = x₁
    - Inverse: x₂ = (y₂ - t(y₁)) ⊙ exp(-s(y₁)), x₁ = y₁
    - Log det jacobian: sum(s(x₁))

    where ⊙ denotes element-wise multiplication.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector
    hidden_dims : List[int]
        Hidden layer dimensions for the shift and scale networks
    activation : str, default="relu"
        Activation function for hidden layers
    mask_type : str, default="alternating_even"
        Type of masking: "alternating_even" or "alternating_odd"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        rngs: nnx.Rngs,
        activation: str = "relu",
        mask_type: str = "alternating_even",
    ):
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.mask_type = mask_type

        # Create binary mask for coupling - stored at initialization
        self.mask = self._create_mask()

        # Compute actual masked and unmasked dimensions from the mask
        masked_dim = int(jnp.sum(self.mask))
        unmasked_dim = input_dim - masked_dim

        # Create shared hidden layers
        self.shared_layers = []

        # Input layer: masked_dim -> first_hidden_dim
        self.shared_layers.append(
            nnx.Linear(masked_dim, self.hidden_dims[0], rngs=rngs)
        )

        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.shared_layers.append(
                nnx.Linear(
                    self.hidden_dims[i], self.hidden_dims[i + 1], rngs=rngs
                )
            )

        # Create separate output heads for shift and scale
        self.shift_head = nnx.Linear(
            self.hidden_dims[-1], unmasked_dim, rngs=rngs
        )
        self.scale_head = nnx.Linear(
            self.hidden_dims[-1], unmasked_dim, rngs=rngs
        )

    # --------------------------------------------------------------------------

    def _create_mask(self) -> jax.Array:
        """
        Create binary mask for coupling transformation.

        The mask determines which input dimensions are masked (fixed) and which
        are transformed in each coupling layer. Supported mask types:

        - "alternating_even": Creates mask [1, 0, 1, 0, ...] where even-indexed
          elements (0, 2, 4, ...) are masked (1) and odd-indexed are unmasked
          (0)
        - "alternating_odd": Creates mask [0, 1, 0, 1, ...] where odd-indexed
          elements (1, 3, 5, ...) are masked (1) and even-indexed are unmasked
          (0)

        Returns
        -------
        mask : jax.Array
            Binary mask array of shape (input_dim,), with 1s for masked and 0s
            for unmasked dimensions.

        Raises
        ------
        ValueError
            If an unknown mask_type is provided.
        """
        if self.mask_type == "alternating_even":
            # Even-indexed elements masked: [1, 0, 1, 0, ...]
            mask = jnp.arange(self.input_dim) % 2
        elif self.mask_type == "alternating_odd":
            # Odd-indexed elements masked: [0, 1, 0, 1, ...]
            mask = 1 - (jnp.arange(self.input_dim) % 2)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        return mask

    # --------------------------------------------------------------------------

    def _get_masked_unmasked(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Split the input tensor into masked and unmasked parts according to the
        current mask.

        This function uses the binary mask stored in self.mask to separate the
        input tensor `x` into two parts:
            - The masked part, which consists of the elements of `x` at
              positions where the mask is 1.
            - The unmasked part, which consists of the elements of `x` at
              positions where the mask is 0.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (..., input_dim), where input_dim matches the
            length of self.mask.

        Returns
        -------
        x_masked : jax.Array
            Tensor containing the masked elements of `x`, shape (...,
            num_masked).
        x_unmasked : jax.Array
            Tensor containing the unmasked elements of `x`, shape (...,
            num_unmasked).
        """
        # Use static indexing based on mask pattern to avoid JAX tracing issues
        input_dim = x.shape[-1]

        if self.mask_type == "alternating_even":
            # Even-indexed elements masked: [1, 0, 1, 0, ...]
            # For 3D: indices [0, 2] are masked, [1] is unmasked
            # But mask is [0, 1, 0] so indices [1] are masked, [0, 2] are unmasked
            masked_indices = jnp.arange(1, input_dim, 2)
            unmasked_indices = jnp.arange(0, input_dim, 2)
        elif self.mask_type == "alternating_odd":
            # Odd-indexed elements masked: [0, 1, 0, 1, ...]
            # For 3D: indices [1] are masked, [0, 2] are unmasked
            # But mask is [1, 0, 1] so indices [0, 2] are masked, [1] is unmasked
            masked_indices = jnp.arange(0, input_dim, 2)
            unmasked_indices = jnp.arange(1, input_dim, 2)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        # Select masked and unmasked elements using integer indexing
        x_masked = x[..., masked_indices]
        x_unmasked = x[..., unmasked_indices]

        return x_masked, x_unmasked

    # --------------------------------------------------------------------------

    def _get_shift_log_scale(
        self, x_masked: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the shift and log_scale parameters for the affine coupling
        transformation.

        This method takes the masked part of the input tensor and passes it
        through a shared neural network, then through separate output heads
        (self.shift_head and self.scale_head) to produce the shift and log_scale
        parameters required for the affine transformation of the unmasked part.

        Parameters
        ----------
        x_masked : jax.Array
            The masked part of the input tensor, of shape (..., num_masked),
            where num_masked is the number of masked features as determined by
            the mask.

        Returns
        -------
        shift : jax.Array
            The shift parameter for the affine transformation, of shape (...,
            unmasked_dim), where unmasked_dim is typically half of the input
            dimension.
        log_scale : jax.Array
            The log scale parameter for the affine transformation, of shape
            (..., unmasked_dim).
        """
        # Get activation function
        activation = ACTIVATION_FUNCTIONS[self.activation]

        # Forward pass through shared layers (manual iteration like encoder)
        h = x_masked
        for layer in self.shared_layers:
            h = activation(layer(h))

        # Apply separate output heads
        shift = self.shift_head(h)
        log_scale = self.scale_head(h)

        return shift, log_scale

    # --------------------------------------------------------------------------

    def forward(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Applies the forward affine coupling transformation to the input tensor.

        This method splits the input tensor into masked and unmasked parts
        according to the coupling mask, computes the shift and log_scale
        parameters from the masked part, and then applies an affine
        transformation to the unmasked part. The masked part remains unchanged.
        The method then reconstructs the full output tensor and computes the
        log-determinant of the Jacobian for the transformation.

        Parameters
        ----------
        x : jax.Array
            Input tensor of shape (..., input_dim), where input_dim is the total
            number of features.

        Returns
        -------
        y : jax.Array
            Output tensor of the same shape as x, after applying the coupling
            transformation.
        log_det_jacobian : jax.Array
            Log-determinant of the Jacobian of the transformation, shape (...,).
        """
        # Split the input tensor into masked and unmasked parts using the mask
        x_masked, x_unmasked = self._get_masked_unmasked(x)

        # Compute the shift and log_scale parameters from the masked part
        shift, log_scale = self._get_shift_log_scale(x_masked)

        # Apply the affine transformation to the unmasked part: scale and shift
        y_unmasked = x_unmasked * jnp.exp(log_scale) + shift

        # The masked part remains unchanged in the output
        y_masked = x_masked

        # Initialize an output tensor of zeros with the same shape as the input
        y = jnp.zeros_like(x)

        # Reconstruct the full tensor using static indexing
        input_dim = x.shape[-1]

        if self.mask_type == "alternating_even":
            masked_indices = jnp.arange(1, input_dim, 2)
            unmasked_indices = jnp.arange(0, input_dim, 2)
        elif self.mask_type == "alternating_odd":
            masked_indices = jnp.arange(0, input_dim, 2)
            unmasked_indices = jnp.arange(1, input_dim, 2)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        y = y.at[..., masked_indices].set(y_masked)
        y = y.at[..., unmasked_indices].set(y_unmasked)

        # Compute the log-determinant of the Jacobian (sum over log_scale for
        # each sample)
        log_det_jacobian = jnp.sum(log_scale, axis=-1)

        # Return the transformed tensor and the log-determinant
        return y, log_det_jacobian

    # --------------------------------------------------------------------------

    def inverse(self, y: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Perform the inverse coupling transformation: recovers the original input
        x from the transformed output y.

        This method splits the input tensor y into masked and unmasked parts
        according to the current mask, computes the shift and log_scale
        parameters from the masked part, and then applies the inverse affine
        transformation to the unmasked part. The masked part remains unchanged.
        The method reconstructs the full tensor x and computes the
        log-determinant of the inverse Jacobian, which is the negative of the
        forward log-determinant.

        Parameters
        ----------
        y : jax.Array
            Input tensor of shape (..., input_dim), where input_dim is the total
            number of features.

        Returns
        -------
        x : jax.Array
            Inverse transformed tensor of the same shape as y (original input
            before coupling transformation).
        log_det_jacobian : jax.Array
            Log-determinant of the inverse Jacobian (negative of the forward
            log-determinant), shape (...,).
        """
        # Split the input tensor y into masked and unmasked parts using the mask
        y_masked, y_unmasked = self._get_masked_unmasked(y)

        # Compute the shift and log_scale parameters from the masked part
        shift, log_scale = self._get_shift_log_scale(y_masked)

        # Apply the inverse affine transformation to the unmasked part:
        # (y_unmasked - shift) * exp(-log_scale)
        x_unmasked = (y_unmasked - shift) * jnp.exp(-log_scale)

        # The masked part remains unchanged in the inverse transformation
        x_masked = y_masked

        # Initialize an output tensor of zeros with the same shape as y
        x = jnp.zeros_like(y)

        # Reconstruct the full tensor using static indexing
        input_dim = y.shape[-1]

        if self.mask_type == "alternating_even":
            masked_indices = jnp.arange(1, input_dim, 2)
            unmasked_indices = jnp.arange(0, input_dim, 2)
        elif self.mask_type == "alternating_odd":
            masked_indices = jnp.arange(0, input_dim, 2)
            unmasked_indices = jnp.arange(1, input_dim, 2)
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        x = x.at[..., masked_indices].set(x_masked)
        x = x.at[..., unmasked_indices].set(x_unmasked)

        # Compute the log-determinant of the inverse Jacobian (negative sum over
        # log_scale for each sample)
        log_det_jacobian = -jnp.sum(log_scale, axis=-1)

        # Return the inverse-transformed tensor and the log-determinant
        return x, log_det_jacobian

    # --------------------------------------------------------------------------

    def __call__(
        self, x: jax.Array, inverse: bool = False
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Apply the coupling transformation.

        Parameters
        ----------
        x : jax.Array
            Input tensor
        inverse : bool, default=False
            Whether to apply inverse transformation

        Returns
        -------
        transformed : jax.Array
            Transformed tensor
        log_det_jacobian : jax.Array
            Log determinant of the Jacobian
        """
        if inverse:
            return self.inverse(x)
        else:
            return self.forward(x)


# ------------------------------------------------------------------------------
# Decoupled Prior for dpVAE
# ------------------------------------------------------------------------------


class DecoupledPrior(nnx.Module):
    """
    Stack of K coupling layers implementing the bijective mapping g_η.

    This class implements the decoupled prior from the dpVAE paper, which
    uses a stack of affine coupling layers to transform a simple spherical
    prior (e.g., standard normal) into a more complex prior distribution.

    The transformation is bijective, allowing both forward (prior -> complex)
    and inverse (complex -> prior) mappings, along with proper log determinant
    computation for the change of variables formula.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space
    num_layers : int
        Number of coupling layers in the stack
    hidden_dims : List[int]
        Hidden layer dimensions for each coupling layer
    activation : str, default="relu"
        Activation function for the coupling layers
    mask_type : str, default="alternating"
        Type of masking for coupling layers. If "alternating", layers will
        alternate between "alternating_even" and "alternating_odd" masks.
    rngs : nnx.Rngs
        Random number generators for weight initialization
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        hidden_dims: List[int],
        rngs: nnx.Rngs,
        activation: str = "relu",
        mask_type: str = "alternating",
    ):
        # Store parameters
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.mask_type = mask_type

        # Create stack of coupling layers
        self.coupling_layers = []

        # Create a new coupling layer for each layer in the stack
        for i in range(num_layers):
            # Determine mask type for this layer
            if mask_type == "alternating":
                # Alternate between even and odd masks
                layer_mask_type = (
                    "alternating_even" if i % 2 == 0 else "alternating_odd"
                )
            else:
                # Use the specified mask type for all layers
                layer_mask_type = mask_type

            coupling_layer = AffineCouplingLayer(
                input_dim=latent_dim,
                hidden_dims=hidden_dims,
                rngs=rngs,
                activation=activation,
                mask_type=layer_mask_type,
            )
            self.coupling_layers.append(coupling_layer)

    # --------------------------------------------------------------------------

    def forward(self, z: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Applies the full stack of affine coupling layers in the forward
        direction.

        Forward transformation: z ~ p(z) -> z' ~ p(z') with log determinant.

        This method transforms a sample from the simple prior distribution
        (e.g., standard normal) into a sample from the more complex, learned
        prior by sequentially applying each AffineCouplingLayer in the stack. It
        also accumulates the log-determinant of the Jacobian for the entire
        transformation, which is required for computing the change of variables
        in the normalizing flow.

        Parameters
        ----------
        z : jax.Array
            Input tensor sampled from the simple prior, shape (..., latent_dim)

        Returns
        -------
        z_transformed : jax.Array
            Output tensor transformed to the complex prior, shape (...,
            latent_dim)
        log_det_jacobian : jax.Array
            Total log-determinant of the full transformation, shape (...,)

        Notes
        -----
        The log-determinant is accumulated across all coupling layers and is
        required for likelihood computation in normalizing flows.
        """
        # Initialize the current latent variable with the input
        z_current = z
        # Initialize the total log-determinant accumulator
        total_log_det = 0.0

        # Sequentially apply each coupling layer in the stack
        for coupling_layer in self.coupling_layers:
            # Apply the forward transformation of the current coupling layer
            z_current, log_det = coupling_layer.forward(z_current)
            # Accumulate the log-determinant from this layer
            total_log_det += log_det

        # Return the final transformed latent variable and the total
        # log-determinant
        return z_current, total_log_det

    # --------------------------------------------------------------------------

    def inverse(self, z_transformed: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Applies the inverse transformation of the full stack of affine coupling
        layers.

        Inverse transformation: z' ~ p(z') -> z ~ p(z) with log determinant.

        This method takes a sample from the complex, learned prior distribution
        (i.e., after all coupling layers have been applied in the forward
        direction) and sequentially applies the inverse of each
        AffineCouplingLayer in the stack, in reverse order. This recovers a
        sample from the original, simple prior (e.g., standard normal). The
        method also accumulates the log-determinant of the Jacobian for the
        entire inverse transformation, which is required for likelihood
        computations and change-of-variables in normalizing flows.

        Parameters
        ----------
        z_transformed : jax.Array
            Input tensor from the complex prior, shape (..., latent_dim)

        Returns
        -------
        z : jax.Array
            Output tensor transformed back to the simple prior, shape (...,
            latent_dim)
        log_det_jacobian : jax.Array
            Total log-determinant of the inverse transformation, shape (...,)

        Notes
        -----
        The log-determinant is accumulated across all coupling layers in the
        reverse direction. This is essential for computing the correct
        likelihood under the flow-based prior.
        """
        # Initialize the current latent variable with the transformed input
        z_current = z_transformed
        # Initialize the total log-determinant accumulator
        total_log_det = 0.0

        # Sequentially apply the inverse of each coupling layer in reverse order
        for coupling_layer in reversed(self.coupling_layers):
            # Apply the inverse transformation of the current coupling layer
            z_current, log_det = coupling_layer.inverse(z_current)
            # Accumulate the log-determinant from this layer
            total_log_det += log_det

        # Return the final recovered latent variable and the total
        # log-determinant
        return z_current, total_log_det

    # --------------------------------------------------------------------------

    def __call__(
        self, z: jax.Array, inverse: bool = False
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Apply the decoupled prior transformation.

        Parameters
        ----------
        z : jax.Array
            Input tensor, shape (..., latent_dim)
        inverse : bool, default=False
            Whether to apply inverse transformation

        Returns
        -------
        z_transformed : jax.Array
            Transformed tensor, shape (..., latent_dim)
        log_det_jacobian : jax.Array
            Log determinant, shape (...)
        """
        if inverse:
            return self.inverse(z)
        else:
            return self.forward(z)


# ------------------------------------------------------------------------------
# NumPyro Distribution for Decoupled Prior
# ------------------------------------------------------------------------------


class DecoupledPriorDistribution(numpyro.distributions.Distribution):
    """
    NumPyro distribution implementing the decoupled prior from dpVAE.

    This distribution wraps a simple base distribution (e.g., standard normal)
    and applies a bijective transformation through a stack of coupling layers.
    The log_prob is computed using the change of variables formula:

        log p(z') = log p(z) + log |det(J)|

    where z' is the transformed variable, z is the base variable, and J is
    the Jacobian of the transformation.

    Parameters
    ----------
    decoupled_prior : DecoupledPrior
        The decoupled prior module containing the coupling layers
    base_distribution : numpyro.distributions.Distribution
        The base distribution (e.g., Normal(0, 1))
    validate_args : bool, optional
        Whether to validate arguments, by default None
    """

    def __init__(
        self,
        decoupled_prior: DecoupledPrior,
        base_distribution: numpyro.distributions.Distribution,
        validate_args: bool = None,
    ):
        if not isinstance(
            base_distribution, numpyro.distributions.Distribution
        ):
            raise ValueError("base_distribution must be a NumPyro Distribution")

        # Store the components
        self.decoupled_prior = decoupled_prior
        self.base_distribution = base_distribution

        # Get event shape from base distribution
        event_shape = base_distribution.event_shape
        # Use empty batch shape to allow NumPyro to handle broadcasting
        batch_shape = ()

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

        # Set the support to be the same as the base distribution
        self.support = base_distribution.support

    # --------------------------------------------------------------------------

    @validate_sample
    def log_prob(self, value: jax.Array) -> jax.Array:
        """
        Compute the log probability of the transformed value.

        This implements the change of variables formula:
            log p(z') = log p(z) + log |det(J)|

        where z' is the input value (from complex prior) and z is the
        corresponding value in the base distribution.

        Parameters
        ----------
        value : jax.Array
            The value from the complex prior distribution, shape
            (..., event_shape)

        Returns
        -------
        jax.Array
            The log probability, shape (...)
        """
        # Handle both NumPyro module wrapper and direct nnx module
        if hasattr(self.decoupled_prior, "func") and hasattr(
            self.decoupled_prior, "args"
        ):
            # NumPyro module wrapper
            # it's a callable that takes (input, inverse=True)
            z_base, log_det_jacobian = self.decoupled_prior(value, inverse=True)
        else:
            # Direct nnx module - call the inverse method directly
            z_base, log_det_jacobian = self.decoupled_prior.inverse(value)

        # Get log probability from base distribution
        base_log_prob = self.base_distribution.log_prob(z_base)

        # Apply change of variables formula
        # log_det_jacobian should already have the right shape
        log_prob = base_log_prob + log_det_jacobian

        return log_prob

    # --------------------------------------------------------------------------

    def sample(
        self, key: jax.Array, sample_shape: Tuple[int, ...] = ()
    ) -> jax.Array:
        """
        Sample from the decoupled prior distribution.

        This samples from the base distribution and then applies the
        forward transformation through the coupling layers.

        Parameters
        ----------
        key : jax.Array
            Random number generator key
        sample_shape : Tuple[int, ...], optional
            Shape of samples to generate, by default ()

        Returns
        -------
        jax.Array
            Samples from the decoupled prior, shape
            (sample_shape + batch_shape + event_shape)
        """
        # Sample from base distribution
        z_base = self.base_distribution.sample(key, sample_shape)

        # Apply forward transformation to get samples from complex prior
        # Handle both NumPyro module wrapper and direct nnx module
        if hasattr(self.decoupled_prior, "func") and hasattr(
            self.decoupled_prior, "args"
        ):
            # NumPyro module wrapper - it's a callable
            z_complex, _ = self.decoupled_prior(z_base)
        else:
            # Direct nnx module - call the forward method directly
            z_complex, _ = self.decoupled_prior.forward(z_base)

        return z_complex

    # --------------------------------------------------------------------------

    def sample_with_intermediates(
        self, key: jax.Array, sample_shape: Tuple[int, ...] = ()
    ) -> Tuple[jax.Array, Dict]:
        """
        Sample from the decoupled prior and return intermediate values.

        This is useful for debugging and understanding the transformation
        process.

        Parameters
        ----------
        key : jax.Array
            Random number generator key
        sample_shape : Tuple[int, ...], optional
            Shape of samples to generate, by default ()

        Returns
        -------
        Tuple[jax.Array, Dict]
            Samples from the decoupled prior and intermediate values
        """
        # Sample from base distribution
        z_base = self.base_distribution.sample(key, sample_shape)

        # Apply forward transformation
        # Handle both NumPyro module wrapper and direct nnx module
        if hasattr(self.decoupled_prior, "func") and hasattr(
            self.decoupled_prior, "args"
        ):
            # NumPyro module wrapper - it's a callable
            z_complex, log_det = self.decoupled_prior(z_base)
        else:
            # Direct nnx module - call the forward method directly
            z_complex, log_det = self.decoupled_prior.forward(z_base)

        # Return samples and intermediate values
        intermediates = {
            "z_base": z_base,
            "z_complex": z_complex,
            "log_det_jacobian": log_det,
        }

        return z_complex, intermediates


# ------------------------------------------------------------------------------
# dpVAE Class and Factory Function
# ------------------------------------------------------------------------------


class dpVAE(VAE):
    """
    VAE with decoupled prior - extends the standard VAE with a learned prior.

    This class inherits all functionality from the standard VAE but uses a
    decoupled prior distribution instead of a standard prior. The decoupled
    prior is implemented using a stack of affine coupling layers that transform
    a simple base distribution (e.g., standard normal) into a more complex,
    learned prior distribution.

    Parameters
    ----------
    encoder : Encoder
        The encoder module
    decoder : Decoder
        The decoder module
    config : VAEConfig
        Configuration for the VAE architecture
    decoupled_prior : DecoupledPrior
        The decoupled prior module containing the coupling layers
    rngs : nnx.Rngs
        Random number generators for weight initialization
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        config: VAEConfig,
        decoupled_prior: DecoupledPrior,
        *,
        rngs: nnx.Rngs,
    ):
        # Initialize the parent VAE class
        super().__init__(encoder, decoder, config, rngs=rngs)

        # Store the decoupled prior
        self.decoupled_prior = decoupled_prior

    def get_prior_distribution(self, base_distribution=None):
        """
        Create a DecoupledPriorDistribution for use in NumPyro models.

        Parameters
        ----------
        base_distribution : numpyro.distributions.Distribution, optional
            The base distribution to use. If None, uses a multivariate standard
            normal distribution.

        Returns
        -------
        DecoupledPriorDistribution
            A NumPyro distribution that implements the decoupled prior
        """
        if base_distribution is None:
            # Use multivariate standard normal as default base distribution
            base_distribution = dist.Normal(
                jnp.zeros(self.config.latent_dim),
                jnp.ones(self.config.latent_dim),
            )

        return DecoupledPriorDistribution(
            decoupled_prior=self.decoupled_prior,
            base_distribution=base_distribution,
        )


# ------------------------------------------------------------------------------


def create_dpvae(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
    standardize_mean: Optional[jnp.ndarray] = None,
    standardize_std: Optional[jnp.ndarray] = None,
    # Decoupled prior parameters
    prior_hidden_dims: Optional[List[int]] = None,
    prior_num_layers: Optional[int] = None,
    prior_activation: Optional[str] = None,
    prior_mask_type: str = "alternating",
) -> dpVAE:
    """
    Create a dpVAE (decoupled prior VAE) with the specified configuration.

    This function creates a VAE with a learned prior implemented using a stack
    of affine coupling layers. The prior transforms a simple base distribution
    (e.g., standard normal) into a more complex, learned prior distribution.

    Parameters
    ----------
    input_dim : int
        Input dimension (number of genes)
    latent_dim : int, default=3
        Latent space dimension
    hidden_dims : Optional[List[int]], default=None
        Hidden layer dimensions for encoder/decoder. If None, uses [256, 256]
    activation : Optional[str], default=None
        Activation function for encoder/decoder. If None, uses "relu"
    input_transformation : Optional[str], default=None
        Input transformation for encoder. If None, uses "log1p"
    standardize_mean : Optional[jnp.ndarray], default=None
        Mean values for standardization (default: None)
    standardize_std : Optional[jnp.ndarray], default=None
        Standard deviation values for standardization (default: None)
    prior_hidden_dims : Optional[List[int]], default=None
        Hidden layer dimensions for coupling layers. If None, uses [64, 64].
        The number of coupling layers is determined by the length of this list.
    prior_activation : Optional[str], default=None
        Activation function for coupling layers. If None, uses "relu"
    prior_mask_type : str, default="alternating"
        Type of masking for coupling layers. If "alternating", layers will
        alternate between "alternating_even" and "alternating_odd" masks.
        Can also specify "alternating_even" or "alternating_odd" for all layers.

    Returns
    -------
    dpVAE
        A dpVAE instance with the specified configuration
    """
    # Set default values for encoder/decoder parameters
    if hidden_dims is None:
        hidden_dims = [256, 256]
    if activation is None:
        activation = "relu"
    if input_transformation is None:
        input_transformation = "log1p"

    # Set default values for prior parameters
    if prior_hidden_dims is None:
        prior_hidden_dims = [128, 128]
    if prior_num_layers is None:
        prior_num_layers = 2
    if prior_activation is None:
        prior_activation = "relu"

    # Create VAE configuration
    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    # Create encoder and decoder
    encoder = create_encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    decoder = create_decoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    # Create decoupled prior
    rngs = nnx.Rngs(
        params=jax.random.PRNGKey(42)
    )  # Use fixed key for reproducibility
    decoupled_prior = DecoupledPrior(
        latent_dim=latent_dim,
        num_layers=prior_num_layers,
        hidden_dims=prior_hidden_dims,
        rngs=rngs,
        activation=prior_activation,
        mask_type=prior_mask_type,
    )

    # Create and return dpVAE
    return dpVAE(
        encoder=encoder,
        decoder=decoder,
        config=config,
        decoupled_prior=decoupled_prior,
        rngs=rngs,
    )

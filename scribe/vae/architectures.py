"""
Default VAE architectures for scRNA-seq data using Flax NNX.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass


# ------------------------------------------------------------------------------
# Dictionary of functions
# ------------------------------------------------------------------------------

ACTIVATION_FUNCTIONS = {
    "gelu": nnx.gelu,
    "relu": nnx.relu,
    "softplus": nnx.softplus,
    "sigmoid": nnx.sigmoid,
    "tanh": nnx.tanh,
    "elu": nnx.elu,
    "selu": nnx.selu,
    "swish": nnx.swish,
    "hard_swish": nnx.hard_swish,
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
    """Configuration for VAE architecture."""

    input_dim: int
    latent_dim: int = 3
    # List of hidden layer dimensions
    hidden_dims: List[int] = None
    activation: str = "relu"
    # Output activation for decoder
    output_activation: str = "softplus"
    # Input transformation for encoder (default: log1p for scRNA-seq data)
    input_transformation: str = "log1p"

    def __post_init__(self):
        if self.hidden_dims is None:
            # Default: 2 hidden layers of 256
            self.hidden_dims = [256, 256]


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
        output_activation = ACTIVATION_FUNCTIONS[self.config.output_activation]

        # Decoder forward pass through all hidden layers
        h = z
        for layer in self.decoder_layers:
            h = activation(layer(h))

        # Output layer with configurable activation for r parameters
        output = self.decoder_output(h)
        output = output_activation(output)

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


# ------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------


def create_encoder(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
) -> Encoder:
    """
    Create a standalone encoder for VAE.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)
        input_transformation: Input transformation function (default: log1p)

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
        output_activation="softplus",
        input_transformation=input_transformation,
    )

    rngs = nnx.Rngs(params=0)
    return Encoder(config=config, rngs=rngs)


# ------------------------------------------------------------------------------


def create_decoder(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    output_activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
) -> Decoder:
    """
    Create a standalone decoder for VAE.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)
        output_activation: Output activation function (default: softplus)
        input_transformation: Input transformation function (default: log1p)

    Returns:
        Configured Decoder instance
    """
    if activation is None:
        activation = "relu"
    if output_activation is None:
        output_activation = "softplus"
    if input_transformation is None:
        input_transformation = "log1p"

    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        input_transformation=input_transformation,
    )

    rngs = nnx.Rngs(params=0)
    return Decoder(config=config, rngs=rngs)


# ------------------------------------------------------------------------------


def create_vae(
    input_dim: int,
    latent_dim: int = 3,
    hidden_dims: Optional[List[int]] = None,
    activation: Optional[str] = None,
    output_activation: Optional[str] = None,
    input_transformation: Optional[str] = None,
) -> VAE:
    """
    Create a default VAE architecture with configurable hidden layers.

    Args:
        input_dim: Dimension of input data (number of genes)
        latent_dim: Dimension of latent space (default: 3)
        hidden_dims: List of hidden layer dimensions (default: [256, 256])
        activation: Activation function (default: gelu)
        output_activation: Output activation function (default: softplus)
        input_transformation: Input transformation function (default: log1p)

    Returns:
        Configured VAE instance
    """
    if activation is None:
        activation = "relu"
    if output_activation is None:
        output_activation = "softplus"
    if input_transformation is None:
        input_transformation = "log1p"

    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
        input_transformation=input_transformation,
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
    mask_type : str, default="alternating"
        Type of masking: "alternating" or "checkerboard"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        rngs: nnx.Rngs,
        activation: str = "relu",
        mask_type: str = "alternating",
    ):
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.mask_type = mask_type

        # Create shared hidden layers
        self.shared_layers = []

        # Input layer: masked_dim -> first_hidden_dim
        masked_dim = input_dim // 2
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
        unmasked_dim = input_dim // 2
        self.shift_head = nnx.Linear(
            self.hidden_dims[-1], unmasked_dim, rngs=rngs
        )
        self.scale_head = nnx.Linear(
            self.hidden_dims[-1], unmasked_dim, rngs=rngs
        )

        # Create binary mask for coupling
        self.mask = self._create_mask()

    # --------------------------------------------------------------------------

    def _create_mask(self) -> jax.Array:
        """
        Create binary mask for coupling transformation.

        The mask determines which input dimensions are masked (fixed) and which
        are transformed in each coupling layer. Supported mask types:

        - "alternating": Creates a 1D alternating mask pattern [1, 0, 1, 0, ...]
          across the input dimensions. Odd-indexed elements are masked (1),
          even-indexed are unmasked (0). This is commonly used for 1D data.

        - "checkerboard": Intended for 2D data, creates a checkerboard pattern.
          For 1D data, this reduces to the same alternating pattern as above.

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
        if self.mask_type == "alternating":
            # Alternating pattern: [1, 0, 1, 0, ...]
            mask = jnp.arange(self.input_dim) % 2
        elif self.mask_type == "checkerboard":
            # Checkerboard pattern for 2D data
            # For 1D data, this reduces to alternating
            mask = jnp.arange(self.input_dim) % 2
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

        This is useful for coupling layers in normalizing flows, where only a
        subset of the input is transformed at each step, and the rest is left
        unchanged.

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
        # Find indices where mask is 1 (masked positions)
        masked_indices = jnp.where(self.mask == 1)[0]
        # Find indices where mask is 0 (unmasked positions)
        unmasked_indices = jnp.where(self.mask == 0)[0]

        # Select masked elements from input tensor using advanced indexing
        x_masked = x[..., masked_indices]
        # Select unmasked elements from input tensor using advanced indexing
        x_unmasked = x[..., unmasked_indices]

        # Return the tuple of masked and unmasked tensors
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

        # Get the indices for masked and unmasked features
        masked_indices = jnp.where(self.mask == 1)[0]
        unmasked_indices = jnp.where(self.mask == 0)[0]

        # Set the masked part in the output tensor
        y = y.at[..., masked_indices].set(y_masked)
        # Set the transformed unmasked part in the output tensor
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

        # Get the indices for masked and unmasked features
        masked_indices = jnp.where(self.mask == 1)[0]
        unmasked_indices = jnp.where(self.mask == 0)[0]

        # Set the masked part in the output tensor
        x = x.at[..., masked_indices].set(x_masked)
        # Set the inverse-transformed unmasked part in the output tensor
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
        Type of masking for coupling layers
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
            coupling_layer = AffineCouplingLayer(
                input_dim=latent_dim,
                hidden_dims=hidden_dims,
                rngs=rngs,
                activation=activation,
                mask_type=mask_type,
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

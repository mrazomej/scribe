"""
Autoregressive normalizing flows.

Implements Masked Autoregressive Flow (MAF) and Inverse Autoregressive Flow
(IAF) using Masked Autoencoder for Distribution Estimation (MADE) as the
autoregressive network.

Key trade-offs:

- **MAF**: Fast density evaluation (one MADE call), slow sampling
  (D sequential calls). Preferred as a *learned prior*.
- **IAF**: Fast sampling (one MADE call), slow density evaluation
  (D sequential calls). Preferred as a *posterior flow*.

Classes
-------
MaskedDense
    Dense layer with a binary mask enforcing autoregressive ordering.
MADE
    Masked Autoencoder for Distribution Estimation.
MAF
    Masked Autoregressive Flow layer.
IAF
    Inverse Autoregressive Flow layer.

References
----------
Germain et al., "MADE: Masked Autoencoder for Distribution Estimation",
    ICML 2015.
Papamakarios et al., "Masked Autoregressive Flow for Density Estimation",
    NeurIPS 2017.
Kingma et al., "Improving Variational Inference with Inverse Autoregressive
    Flow", NeurIPS 2016.
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------------------------------------------------------
# Activation helper (shared with coupling module)
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
    "leaky_relu": jax.nn.leaky_relu,
    "softplus": jax.nn.softplus,
}


def _get_act(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose from: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]


# ===========================================================================
# MADE: Masked Autoencoder for Distribution Estimation
# ===========================================================================


def _create_masks(
    features: int,
    hidden_dims: List[int],
    n_outputs_per_dim: int = 2,
    seed: int = 0,
    context_dim: int = 0,
) -> List[jnp.ndarray]:
    """Create binary masks for MADE.

    Assigns each hidden unit a connectivity integer and builds masks
    that enforce the autoregressive property: output d depends only on
    inputs 1, ..., d-1.

    Parameters
    ----------
    features : int
        Input/output dimensionality.
    hidden_dims : List[int]
        Hidden layer sizes.
    n_outputs_per_dim : int
        Number of outputs per input dimension (2 for shift + log_scale).
    seed : int
        Random seed for connectivity assignment.

    Returns
    -------
    masks : List[jnp.ndarray]
        One mask per layer (including input→hidden and hidden→output).
    """
    rng = jax.random.PRNGKey(seed)
    D = features

    # Input degrees: 1, 2, ..., D (natural ordering)
    input_degrees = jnp.arange(1, D + 1)

    # Hidden degrees: random integers in [1, D-1] for each hidden unit
    all_degrees = [input_degrees]
    for h_dim in hidden_dims:
        rng, subkey = jax.random.split(rng)
        # Assign each hidden unit a degree in [1, D-1]
        degrees = jax.random.randint(subkey, (h_dim,), 1, D)
        all_degrees.append(degrees)

    # Output degrees: 1, 2, ..., D (repeated for each output per dim)
    output_degrees = jnp.repeat(jnp.arange(1, D + 1), n_outputs_per_dim)
    all_degrees.append(output_degrees)

    # Build masks: mask[j, i] = 1 if degree[j] >= degree[i] (hidden layers)
    #              mask[j, i] = 1 if degree[j] > degree[i] (output layer, strict)
    masks = []
    for i in range(len(all_degrees) - 1):
        if i < len(all_degrees) - 2:
            # Hidden layer: >= (non-strict)
            mask = (
                all_degrees[i + 1][:, None] >= all_degrees[i][None, :]
            ).astype(jnp.float32)
        else:
            # Output layer: > (strict, for autoregressive property)
            mask = (
                all_degrees[i + 1][:, None] > all_degrees[i][None, :]
            ).astype(jnp.float32)
        masks.append(mask)

    # Extend first mask for context dimensions (visible to all hidden units)
    if context_dim > 0:
        context_cols = jnp.ones(
            (masks[0].shape[0], context_dim), dtype=jnp.float32
        )
        masks[0] = jnp.concatenate([masks[0], context_cols], axis=1)

    return masks


# ===========================================================================
# MADE: Masked Autoencoder for Distribution Estimation
# ===========================================================================


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.

    An MLP where weight matrices are element-wise multiplied by binary
    masks that enforce autoregressive ordering: output dimension d
    depends only on input dimensions 1, ..., d-1.

    Outputs ``2 * features`` values: ``(shift, log_scale)`` for each
    input dimension.

    Parameters
    ----------
    features : int
        Input dimensionality.
    hidden_dims : List[int]
        Hidden layer sizes.
    activation : str
        Activation function.
    """

    features: int
    hidden_dims: List[int]
    activation: str = "relu"
    context_dim: int = 0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through masked network.

        Parameters
        ----------
        x : jnp.ndarray
            Input of shape ``(..., features)``.
        context : jnp.ndarray, optional
            Pre-embedded conditioning vector, shape ``(..., context_dim)``.

        Returns
        -------
        shift : jnp.ndarray
            Shift parameters, shape ``(..., features)``.
        log_scale : jnp.ndarray
            Log-scale parameters, shape ``(..., features)``.
        """
        masks = _create_masks(
            self.features, self.hidden_dims, n_outputs_per_dim=2,
            context_dim=self.context_dim,
        )
        act = _get_act(self.activation)

        # Build layer dimensions: input (+ context) → hidden → output
        input_dim = self.features + self.context_dim
        dims = [input_dim] + list(self.hidden_dims) + [2 * self.features]

        h = x
        if context is not None and self.context_dim > 0:
            h = jnp.concatenate([x, context], axis=-1)
        for i, (in_dim, out_dim, mask) in enumerate(
            zip(dims[:-1], dims[1:], masks)
        ):
            kernel = self.param(
                f"kernel_{i}",
                nn.initializers.lecun_normal(),
                (in_dim, out_dim),
            )
            bias = self.param(
                f"bias_{i}",
                nn.initializers.zeros,
                (out_dim,),
            )
            # mask shape: (out, in) → transpose to (in, out) for matmul
            masked_kernel = kernel * mask.T
            h = h @ masked_kernel + bias
            # Apply activation (except on last layer)
            if i < len(masks) - 1:
                h = act(h)

        # Split interleaved outputs into shift and log_scale
        shift = h[..., 0::2]
        log_scale = h[..., 1::2]

        return shift, log_scale


# ===========================================================================
# MAF: Masked Autoregressive Flow
# ===========================================================================


class MAF(nn.Module):
    """Masked Autoregressive Flow layer.

    Uses a MADE network to predict autoregressive affine parameters.
    Density evaluation requires one MADE call (fast); sampling requires
    D sequential calls (slow).

    Forward transform (data → latent): parallel, O(1) MADE calls.
    Inverse transform (latent → data): sequential, O(D) MADE calls.

    Parameters
    ----------
    features : int
        Input dimensionality.
    hidden_dims : List[int]
        Hidden layer sizes for the MADE conditioner.
    activation : str
        Activation function.
    context_dim : int
        Dimensionality of optional context conditioning vector.
    """

    features: int
    hidden_dims: List[int]
    activation: str = "relu"
    context_dim: int = 0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, reverse: bool = False,
        context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply MAF transform.

        Parameters
        ----------
        x : jnp.ndarray
            Input of shape ``(..., features)``.
        reverse : bool
            If True, apply the inverse (sequential, slow for MAF).
        context : jnp.ndarray, optional
            Pre-embedded conditioning vector, shape ``(..., context_dim)``.

        Returns
        -------
        y : jnp.ndarray
            Transformed output, same shape.
        log_det : jnp.ndarray
            Log-determinant Jacobian, shape ``(...)``.
        """
        made = MADE(
            features=self.features,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            context_dim=self.context_dim,
            name="made",
        )

        if not reverse:
            # Forward: x → z (parallel, one MADE call)
            shift, log_scale = made(x, context=context)
            log_scale = jnp.clip(log_scale, -5.0, 5.0)
            z = (x - shift) * jnp.exp(-log_scale)
            log_det = -jnp.sum(log_scale, axis=-1)
            return z, log_det
        else:
            # Inverse: z → x (sequential, D MADE calls)
            z = x  # input is z when reverse=True
            x_out = jnp.zeros_like(z)
            for d in range(self.features):
                shift, log_scale = made(x_out, context=context)
                log_scale = jnp.clip(log_scale, -5.0, 5.0)
                x_out = x_out.at[..., d].set(
                    z[..., d] * jnp.exp(log_scale[..., d]) + shift[..., d]
                )
            # Compute log_det with final parameters
            shift, log_scale = made(x_out, context=context)
            log_scale = jnp.clip(log_scale, -5.0, 5.0)
            log_det = jnp.sum(log_scale, axis=-1)
            return x_out, log_det


# ===========================================================================
# IAF: Inverse Autoregressive Flow
# ===========================================================================


class IAF(nn.Module):
    """Inverse Autoregressive Flow layer.

    Dual of MAF: sampling is fast (one MADE call), density evaluation
    is slow (D sequential calls). Preferred as a posterior flow in VAEs.

    Forward transform (data → latent): sequential, O(D) MADE calls.
    Inverse transform (latent → data): parallel, O(1) MADE calls.

    Parameters
    ----------
    features : int
        Input dimensionality.
    hidden_dims : List[int]
        Hidden layer sizes for the MADE conditioner.
    activation : str
        Activation function.
    context_dim : int
        Dimensionality of optional context conditioning vector.
    """

    features: int
    hidden_dims: List[int]
    activation: str = "relu"
    context_dim: int = 0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, reverse: bool = False,
        context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply IAF transform.

        Parameters
        ----------
        x : jnp.ndarray
            Input of shape ``(..., features)``.
        reverse : bool
            If True, apply the inverse (parallel, fast for IAF).
        context : jnp.ndarray, optional
            Pre-embedded conditioning vector, shape ``(..., context_dim)``.

        Returns
        -------
        y : jnp.ndarray
            Transformed output, same shape.
        log_det : jnp.ndarray
            Log-determinant Jacobian, shape ``(...)``.
        """
        made = MADE(
            features=self.features,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            context_dim=self.context_dim,
            name="made",
        )

        if not reverse:
            # Forward: x → z (sequential, D MADE calls)
            z = jnp.zeros_like(x)
            for d in range(self.features):
                shift, log_scale = made(z, context=context)
                log_scale = jnp.clip(log_scale, -5.0, 5.0)
                z = z.at[..., d].set(
                    (x[..., d] - shift[..., d]) * jnp.exp(-log_scale[..., d])
                )
            # Compute log_det with final parameters
            shift, log_scale = made(z, context=context)
            log_scale = jnp.clip(log_scale, -5.0, 5.0)
            log_det = -jnp.sum(log_scale, axis=-1)
            return z, log_det
        else:
            # Inverse: z → x (parallel, one MADE call)
            z = x  # input is z when reverse=True
            shift, log_scale = made(z, context=context)
            log_scale = jnp.clip(log_scale, -5.0, 5.0)
            x_out = z * jnp.exp(log_scale) + shift
            log_det = jnp.sum(log_scale, axis=-1)
            return x_out, log_det

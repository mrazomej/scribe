"""
Coupling-based normalizing flows.

Implements Real NVP (affine coupling) and Neural Spline Coupling flows.
In coupling flows, the input is split into two halves: one half passes
through unchanged while conditioning a transform applied to the other
half. Both forward and inverse are efficient (single network call each).

Classes
-------
AffineCoupling
    Real NVP affine coupling layer.
SplineCoupling
    Rational-quadratic spline coupling layer.
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .transforms import rqs_forward, rqs_inverse, unconstrained_to_rqs_params

# ---------------------------------------------------------------------------
# Activation helper
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


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------


def _split_by_mask(x: jnp.ndarray, mask_parity: int):
    """Split input into masked (unchanged) and unmasked (transformed) parts.

    Parameters
    ----------
    x : jnp.ndarray
        Input of shape ``(..., D)``.
    mask_parity : int
        0 → even indices are masked, odd are transformed.
        1 → odd indices are masked, even are transformed.

    Returns
    -------
    x_masked : jnp.ndarray
        Masked elements, shape ``(..., ceil(D/2))`` or ``(..., floor(D/2))``.
    x_unmasked : jnp.ndarray
        Unmasked elements, same complementary shape.
    """
    D = x.shape[-1]
    if mask_parity == 0:
        masked_idx = jnp.arange(0, D, 2)
        unmasked_idx = jnp.arange(1, D, 2)
    else:
        masked_idx = jnp.arange(1, D, 2)
        unmasked_idx = jnp.arange(0, D, 2)
    return x[..., masked_idx], x[..., unmasked_idx]


# ---------------------------------------------------------------------------


def _merge_by_mask(
    x_masked: jnp.ndarray,
    x_unmasked: jnp.ndarray,
    D: int,
    mask_parity: int,
) -> jnp.ndarray:
    """Inverse of ``_split_by_mask``: interleave masked and unmasked parts."""
    y = jnp.zeros(x_masked.shape[:-1] + (D,), dtype=x_masked.dtype)
    if mask_parity == 0:
        masked_idx = jnp.arange(0, D, 2)
        unmasked_idx = jnp.arange(1, D, 2)
    else:
        masked_idx = jnp.arange(1, D, 2)
        unmasked_idx = jnp.arange(0, D, 2)
    y = y.at[..., masked_idx].set(x_masked)
    y = y.at[..., unmasked_idx].set(x_unmasked)
    return y


# ===========================================================================
# Affine Coupling (Real NVP)
# ===========================================================================


class AffineCoupling(nn.Module):
    """Real NVP affine coupling layer.

    Splits the input by alternating mask and applies an affine transform
    ``y = x * exp(s) + t`` to the unmasked half, where ``(s, t)`` are
    predicted from the masked half by a conditioner MLP.

    Parameters
    ----------
    features : int
        Dimensionality of the input.
    hidden_dims : List[int]
        Hidden layer sizes for the conditioner MLP.
    mask_parity : int
        0 or 1 — determines which half is masked.
    activation : str
        Activation function for the conditioner.
    """

    features: int
    hidden_dims: List[int]
    mask_parity: int = 0
    activation: str = "relu"
    context_dim: int = 0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, reverse: bool = False,
        context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply affine coupling.

        Parameters
        ----------
        x : jnp.ndarray
            Input of shape ``(..., features)``.
        reverse : bool
            If True, apply the inverse transform.
        context : jnp.ndarray, optional
            Pre-embedded conditioning vector, shape ``(..., context_dim)``.

        Returns
        -------
        y : jnp.ndarray
            Transformed output, same shape.
        log_det : jnp.ndarray
            Log-determinant Jacobian, shape ``(...)``.
        """
        act = _get_act(self.activation)
        x_masked, x_unmasked = _split_by_mask(x, self.mask_parity)
        n_unmasked = x_unmasked.shape[-1]

        # Conditioner MLP: masked (+ optional context) → (shift, log_scale)
        h = x_masked
        if context is not None:
            h = jnp.concatenate([h, context], axis=-1)
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, name=f"hidden_{i}")(h)
            h = act(h)
        shift = nn.Dense(n_unmasked, name="shift")(h)
        log_scale = nn.Dense(n_unmasked, name="log_scale")(h)
        # Clamp log_scale to prevent numerical overflow in exp()
        log_scale = jnp.clip(log_scale, -5.0, 5.0)

        if reverse:
            y_unmasked = (x_unmasked - shift) * jnp.exp(-log_scale)
            log_det = -jnp.sum(log_scale, axis=-1)
        else:
            y_unmasked = x_unmasked * jnp.exp(log_scale) + shift
            log_det = jnp.sum(log_scale, axis=-1)

        y = _merge_by_mask(
            x_masked, y_unmasked, self.features, self.mask_parity
        )
        return y, log_det


# ===========================================================================
# Spline Coupling (Neural Spline Flow)
# ===========================================================================


class SplineCoupling(nn.Module):
    """Rational-quadratic spline coupling layer.

    Same split-and-transform structure as affine coupling, but uses
    a monotone rational-quadratic spline instead of an affine transform.
    Strictly more expressive per layer than affine coupling.

    Parameters
    ----------
    features : int
        Dimensionality of the input.
    hidden_dims : List[int]
        Hidden layer sizes for the conditioner MLP.
    mask_parity : int
        0 or 1 — determines which half is masked.
    activation : str
        Activation function for the conditioner.
    n_bins : int
        Number of spline bins (higher = more expressive, more parameters).
    boundary : float
        Spline domain is ``[-boundary, boundary]``; identity outside.

    References
    ----------
    Durkan et al., "Neural Spline Flows", NeurIPS 2019.
    """

    features: int
    hidden_dims: List[int]
    mask_parity: int = 0
    activation: str = "relu"
    context_dim: int = 0
    n_bins: int = 8
    boundary: float = 3.0

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, reverse: bool = False,
        context: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        act = _get_act(self.activation)
        x_masked, x_unmasked = _split_by_mask(x, self.mask_parity)
        n_unmasked = x_unmasked.shape[-1]

        # Conditioner MLP: masked (+ optional context) → raw spline params
        n_params_per_dim = 3 * self.n_bins + 1
        h = x_masked
        if context is not None:
            h = jnp.concatenate([h, context], axis=-1)
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, name=f"hidden_{i}")(h)
            h = act(h)
        raw_params = nn.Dense(
            n_unmasked * n_params_per_dim,
            name="spline_params",
            kernel_init=nn.initializers.zeros,
        )(h)

        # Reshape to (..., n_unmasked, n_params_per_dim)
        raw_params = raw_params.reshape(
            x_unmasked.shape[:-1] + (n_unmasked, n_params_per_dim)
        )

        widths, heights, derivatives = unconstrained_to_rqs_params(
            raw_params, self.n_bins, self.boundary
        )

        if reverse:
            y_unmasked, log_det = rqs_inverse(
                x_unmasked, widths, heights, derivatives, self.boundary
            )
        else:
            y_unmasked, log_det = rqs_forward(
                x_unmasked, widths, heights, derivatives, self.boundary
            )

        y = _merge_by_mask(
            x_masked, y_unmasked, self.features, self.mask_parity
        )
        return y, log_det

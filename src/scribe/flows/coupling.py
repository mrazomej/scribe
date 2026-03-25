"""
Coupling-based normalizing flows.

Implements Real NVP (affine coupling) and Neural Spline Coupling flows.
In coupling flows, the input is split into two halves: one half passes
through unchanged while conditioning a transform applied to the other
half. Both forward and inverse are efficient (single network call each).

Stability features (all on by default, user-configurable):

* ``zero_init_output`` — zero-initializes the conditioner's output Dense
  layer so the flow starts as identity.
* ``use_layer_norm`` — applies ``nn.LayerNorm`` after each hidden Dense
  layer to stabilize activations at high fan-in.
* ``use_residual`` — adds a skip connection between consecutive hidden
  layers of the same width.
* ``soft_clamp`` — replaces hard ``jnp.clip`` on log-scale with a smooth
  asymmetric ``arctan`` clamp (Andrade 2024) that preserves gradients at
  the boundary and tightly bounds per-layer expansion.

Classes
-------
AffineCoupling
    Real NVP affine coupling layer.
SplineCoupling
    Rational-quadratic spline coupling layer.
"""

import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .transforms import (
    DEFAULT_MIN_DERIVATIVE,
    rqs_forward,
    rqs_inverse,
    unconstrained_to_rqs_params,
)

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
# Asymmetric soft clamping (Andrade 2024, arXiv:2402.16408)
# ---------------------------------------------------------------------------


def _soft_clamp(
    s: jnp.ndarray,
    alpha_pos: float = 0.1,
    alpha_neg: float = 2.0,
) -> jnp.ndarray:
    """Asymmetric arctan-based soft clamp for log-scale values.

    Replaces the hard ``jnp.clip`` used in standard Real NVP.  Unlike
    hard clamping, this function is differentiable everywhere and provides
    non-zero gradients even at extreme input values.

    The asymmetry is deliberate: ``alpha_pos=0.1`` tightly bounds the
    positive (expansion) direction so each layer can expand by at most
    ~10%, while ``alpha_neg=2.0`` allows more contraction.  Over many
    layers this prevents the cumulative blowup of sample magnitudes
    that causes NaN gradients.

    Parameters
    ----------
    s : jnp.ndarray
        Raw log-scale values from the conditioner network.
    alpha_pos : float
        Saturation level for positive values. The output for large
        positive ``s`` approaches ``alpha_pos``.
    alpha_neg : float
        Saturation level for negative values. The output for large
        negative ``s`` approaches ``-alpha_neg``.

    Returns
    -------
    jnp.ndarray
        Clamped log-scale, same shape as ``s``.

    References
    ----------
    Andrade, "Stable Training of Normalizing Flows for High-dimensional
    Variational Inference", 2024. arXiv:2402.16408, Eq. 5.
    """
    pos = (2.0 / jnp.pi) * alpha_pos * jnp.arctan(s / alpha_pos)
    neg = (2.0 / jnp.pi) * alpha_neg * jnp.arctan(s / alpha_neg)
    return jnp.where(s >= 0, pos, neg)


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


# ---------------------------------------------------------------------------
# Shared conditioner MLP builder
# ---------------------------------------------------------------------------


def _conditioner_mlp(
    h: jnp.ndarray,
    hidden_dims: List[int],
    act,
    use_layer_norm: bool,
    use_residual: bool,
) -> jnp.ndarray:
    """Run a conditioner MLP with optional LayerNorm and residual connections.

    Parameters
    ----------
    h : jnp.ndarray
        Input (masked half, optionally concatenated with context).
    hidden_dims : list of int
        Hidden layer widths.
    act : callable
        Activation function.
    use_layer_norm : bool
        If True, apply ``nn.LayerNorm`` after each hidden Dense.
    use_residual : bool
        If True, add a skip connection when consecutive widths match.

    Returns
    -------
    jnp.ndarray
        Hidden representation ready for the output projection.
    """
    for i, dim in enumerate(hidden_dims):
        prev_h = h
        h = nn.Dense(dim, name=f"hidden_{i}")(h)
        if use_layer_norm:
            h = nn.LayerNorm(name=f"ln_{i}")(h)
        h = act(h)
        if use_residual and prev_h.shape[-1] == h.shape[-1]:
            h = h + prev_h
    return h


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
    context_dim : int
        Dimensionality of an optional continuous context vector.
    zero_init_output : bool
        If True (default), zero-initialize the output Dense layers so
        the flow starts as an identity transform.
    use_layer_norm : bool
        If True (default), apply ``nn.LayerNorm`` after each hidden
        Dense layer in the conditioner MLP.
    use_residual : bool
        If True (default), add residual (skip) connections between
        consecutive hidden layers of the same width.
    soft_clamp : bool
        If True (default), use a smooth asymmetric ``arctan``-based
        clamp on the log-scale (Andrade 2024) instead of hard
        ``jnp.clip``.  Preserves gradients at the boundary.
    alpha_pos : float
        Positive saturation level for the soft clamp.  Controls the
        maximum per-layer expansion (default 0.1 → ~10% expansion).
    alpha_neg : float
        Negative saturation level for the soft clamp.  Controls the
        maximum per-layer contraction (default 2.0).
    """

    features: int
    hidden_dims: List[int]
    mask_parity: int = 0
    activation: str = "relu"
    context_dim: int = 0
    zero_init_output: bool = True
    use_layer_norm: bool = True
    use_residual: bool = True
    # Smooth asymmetric arctan clamp on log_scale (Andrade 2024).
    # When False, falls back to the hard jnp.clip(-5, 5).
    soft_clamp: bool = True
    alpha_pos: float = 0.1
    alpha_neg: float = 2.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        reverse: bool = False,
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

        h = _conditioner_mlp(
            h,
            self.hidden_dims,
            act,
            self.use_layer_norm,
            self.use_residual,
        )

        # Output projection — zero-init keeps the flow as identity at init
        _init = (
            nn.initializers.zeros
            if self.zero_init_output
            else nn.initializers.lecun_normal()
        )
        shift = nn.Dense(n_unmasked, name="shift", kernel_init=_init)(h)
        log_scale = nn.Dense(
            n_unmasked,
            name="log_scale",
            kernel_init=_init,
        )(h)
        # Bound log_scale to prevent numerical overflow in exp()
        if self.soft_clamp:
            log_scale = _soft_clamp(log_scale, self.alpha_pos, self.alpha_neg)
        else:
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
# Spline identity-bias initializer
# ===========================================================================


def _spline_identity_bias_init(
    n_unmasked: int,
    n_bins: int,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """Bias initializer that makes zero-kernel spline output an exact identity.

    The raw spline parameter vector has layout
    ``[widths(K) | heights(K) | derivatives(K+1)]`` repeated ``n_unmasked``
    times. Widths and heights at zero → softmax → uniform bins, which already
    gives identity. Derivatives need ``softplus(raw_d) + min_derivative = 1``
    so that the knot slopes equal 1.0.
    """
    # softplus_inv(1.0 - min_derivative), computed with pure Python math
    # to avoid JAX tracer issues under JIT.
    raw_deriv = math.log(math.exp(1.0 - min_derivative) - 1.0)

    def init(key, shape, dtype=jnp.float32):
        # Build a single-element template: 0s for widths/heights, raw_deriv
        # for derivatives; then tile across all unmasked elements.
        template = jnp.concatenate(
            [
                jnp.zeros(2 * n_bins, dtype=dtype),
                jnp.full(n_bins + 1, raw_deriv, dtype=dtype),
            ]
        )
        return jnp.tile(template, (n_unmasked,))

    return init


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
    context_dim : int
        Dimensionality of an optional continuous context vector.
    n_bins : int
        Number of spline bins (higher = more expressive, more parameters).
    boundary : float
        Spline domain is ``[-boundary, boundary]``; identity outside.
    zero_init_output : bool
        If True (default), zero-initialize the kernel and use a custom
        bias that sets knot derivatives to 1.0, producing an exact
        identity transform (log-det = 0) at initialization.
    use_layer_norm : bool
        If True (default), apply ``nn.LayerNorm`` after each hidden
        Dense layer in the conditioner MLP.
    use_residual : bool
        If True (default), add residual (skip) connections between
        consecutive hidden layers of the same width.

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
    zero_init_output: bool = True
    use_layer_norm: bool = True
    use_residual: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        reverse: bool = False,
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

        h = _conditioner_mlp(
            h,
            self.hidden_dims,
            act,
            self.use_layer_norm,
            self.use_residual,
        )

        # Output projection — zero kernel + identity-bias keeps the spline
        # as an exact identity at init (log_det = 0).
        if self.zero_init_output:
            _kernel_init = nn.initializers.zeros
            _bias_init = _spline_identity_bias_init(
                n_unmasked,
                self.n_bins,
            )
        else:
            _kernel_init = nn.initializers.lecun_normal()
            _bias_init = nn.initializers.zeros

        raw_params = nn.Dense(
            n_unmasked * n_params_per_dim,
            name="spline_params",
            kernel_init=_kernel_init,
            bias_init=_bias_init,
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

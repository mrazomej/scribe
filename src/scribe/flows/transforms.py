"""
Monotone spline transforms for normalizing flows.

Implements the rational-quadratic spline (RQS) bijection from
Durkan et al., "Neural Spline Flows" (NeurIPS 2019). The spline
provides an expressive, analytically invertible, element-wise
transform with cheap log-det-Jacobian computation.

Numerical safety (matching the reference *nflows* implementation):

* **Minimum bin width / height** — prevents any bin from collapsing
  to near-zero, which would make the slope ``s_k = h_k / w_k``
  explode and poison the log-det sum.
* **Boundary pinning** — cumulative sums are explicitly pinned at
  ``-boundary`` and ``+boundary`` so float32 drift doesn't push
  knots outside the domain.
* **Log-space log-det** — computes
  ``log(deriv_num) - 2 * log(denom)`` instead of
  ``log(deriv_num / denom²)`` to avoid intermediate overflow.

Functions
---------
rqs_forward
    Forward rational-quadratic spline transform.
rqs_inverse
    Inverse rational-quadratic spline transform.
unconstrained_to_rqs_params
    Convert raw network outputs to valid spline parameters.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

# ------------------------------------------------------------------------------


def unconstrained_to_rqs_params(
    raw_params: jnp.ndarray,
    n_bins: int,
    boundary: float = 3.0,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert raw network outputs to valid spline parameters.

    Parameters
    ----------
    raw_params : jnp.ndarray
        Raw parameters of shape ``(..., 3 * n_bins + 1)``.
        The last axis is split into widths, heights, and derivatives.
    n_bins : int
        Number of spline bins.
    boundary : float
        The spline is defined on ``[-boundary, boundary]``.
    min_derivative : float
        Minimum derivative at knot points (ensures invertibility).
    min_bin_width : float
        Minimum proportion allocated to each bin width.  Prevents bins
        from collapsing, which would make the bin slope diverge.
    min_bin_height : float
        Minimum proportion allocated to each bin height (same purpose).

    Returns
    -------
    widths : jnp.ndarray
        Bin widths, shape ``(..., n_bins)``, positive, sum to ``2 * boundary``.
    heights : jnp.ndarray
        Bin heights, shape ``(..., n_bins)``, positive, sum to ``2 * boundary``.
    derivatives : jnp.ndarray
        Knot derivatives, shape ``(..., n_bins + 1)``, positive.
    """
    raw_widths = raw_params[..., :n_bins]
    raw_heights = raw_params[..., n_bins : 2 * n_bins]
    raw_derivatives = raw_params[..., 2 * n_bins :]

    # Widths: softmax → clamp minimum proportion → scale to [-B, B].
    # After clamping, each bin keeps at least ``min_bin_width`` of the
    # unit probability mass, so scaled width >= min_bin_width * 2B.
    widths = jax.nn.softmax(raw_widths, axis=-1)
    widths = min_bin_width + (1.0 - min_bin_width * n_bins) * widths
    widths = widths * (2.0 * boundary)

    heights = jax.nn.softmax(raw_heights, axis=-1)
    heights = min_bin_height + (1.0 - min_bin_height * n_bins) * heights
    heights = heights * (2.0 * boundary)

    # Derivatives via softplus + offset to ensure strict positivity
    derivatives = jax.nn.softplus(raw_derivatives) + min_derivative

    return widths, heights, derivatives


# ------------------------------------------------------------------------------
# Shared helpers for forward / inverse
# ------------------------------------------------------------------------------


def _build_knots(
    widths: jnp.ndarray,
    heights: jnp.ndarray,
    boundary: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute cumulative knot positions with pinned boundaries.

    Returns ``(cumwidths, cumheights, widths, heights)`` where the last
    two are recomputed from the pinned cumulative values to guarantee
    consistency (avoids float32 cumsum drift).
    """
    cumwidths = jnp.concatenate(
        [
            jnp.full(widths.shape[:-1] + (1,), -boundary),
            jnp.cumsum(widths, axis=-1) - boundary,
        ],
        axis=-1,
    )
    cumheights = jnp.concatenate(
        [
            jnp.full(heights.shape[:-1] + (1,), -boundary),
            jnp.cumsum(heights, axis=-1) - boundary,
        ],
        axis=-1,
    )

    # Pin right / top boundary to exact value
    cumwidths = cumwidths.at[..., -1].set(boundary)
    cumheights = cumheights.at[..., -1].set(boundary)

    # Recompute from pinned cumulatives so w_k, h_k, s_k are consistent
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    return cumwidths, cumheights, widths, heights


def _gather(arr: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Gather along last axis using idx."""
    return jnp.take_along_axis(arr, idx[..., None], axis=-1).squeeze(-1)


def _rqs_log_deriv(
    s_k: jnp.ndarray,
    d_k: jnp.ndarray,
    d_k1: jnp.ndarray,
    xi: jnp.ndarray,
    xi_1mxi: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:
    """Log-derivative of the RQS in log-space.

    Computes ``log(dy/dx)`` as ``log(num) - 2 * log(denom)`` rather than
    ``log(num / denom²)`` to avoid float32 overflow in the intermediate
    ratio.
    """
    deriv_numerator = (
        s_k
        * s_k
        * (d_k1 * xi * xi + 2.0 * s_k * xi_1mxi + d_k * (1.0 - xi) * (1.0 - xi))
    )
    return jnp.log(deriv_numerator) - 2.0 * jnp.log(denominator)


# ------------------------------------------------------------------------------


def rqs_forward(
    x: jnp.ndarray,
    widths: jnp.ndarray,
    heights: jnp.ndarray,
    derivatives: jnp.ndarray,
    boundary: float = 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward rational-quadratic spline transform (element-wise).

    Maps each element of ``x`` through a monotone rational-quadratic spline
    defined by the given widths, heights, and derivatives. Points outside
    ``[-boundary, boundary]`` are mapped via the identity.

    Parameters
    ----------
    x : jnp.ndarray
        Input values, shape ``(..., D)``.
    widths : jnp.ndarray
        Bin widths, shape ``(..., D, n_bins)``.
    heights : jnp.ndarray
        Bin heights, shape ``(..., D, n_bins)``.
    derivatives : jnp.ndarray
        Knot derivatives, shape ``(..., D, n_bins + 1)``.
    boundary : float
        Spline boundary.

    Returns
    -------
    y : jnp.ndarray
        Transformed values, same shape as ``x``.
    log_det : jnp.ndarray
        Log-determinant Jacobian, shape ``(...)``.
    """
    cumwidths, cumheights, widths, heights = _build_knots(
        widths, heights, boundary,
    )
    n_bins = widths.shape[-1]

    # Find which bin each x falls into
    x_expanded = x[..., None]  # (..., D, 1)
    bin_idx = jnp.sum(x_expanded >= cumwidths, axis=-1) - 1  # (..., D)
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    x_k = _gather(cumwidths, bin_idx)
    x_k1 = _gather(cumwidths, bin_idx + 1)
    y_k = _gather(cumheights, bin_idx)
    y_k1 = _gather(cumheights, bin_idx + 1)
    d_k = _gather(derivatives, bin_idx)
    d_k1 = _gather(derivatives, bin_idx + 1)

    w_k = x_k1 - x_k  # bin width
    h_k = y_k1 - y_k  # bin height
    s_k = h_k / w_k  # slope

    # Normalized position within bin: xi in [0, 1]
    xi = (x - x_k) / w_k
    xi = jnp.clip(xi, 0.0, 1.0)

    # Rational quadratic formula
    xi_1mxi = xi * (1.0 - xi)
    numerator = h_k * (s_k * xi * xi + d_k * xi_1mxi)
    denominator = s_k + (d_k1 + d_k - 2.0 * s_k) * xi_1mxi

    y_in = y_k + numerator / denominator

    # Log-determinant in log-space
    log_deriv = _rqs_log_deriv(s_k, d_k, d_k1, xi, xi_1mxi, denominator)

    # Identity outside boundary
    inside = (x >= -boundary) & (x <= boundary)
    y = jnp.where(inside, y_in, x)
    log_deriv = jnp.where(inside, log_deriv, 0.0)

    log_det = jnp.sum(log_deriv, axis=-1)
    return y, log_det


# ------------------------------------------------------------------------------


def rqs_inverse(
    y: jnp.ndarray,
    widths: jnp.ndarray,
    heights: jnp.ndarray,
    derivatives: jnp.ndarray,
    boundary: float = 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse rational-quadratic spline transform (element-wise).

    Analytically inverts the forward RQS transform by solving a quadratic
    equation for the normalized position within each bin.

    Parameters
    ----------
    y : jnp.ndarray
        Input values (in transformed space), shape ``(..., D)``.
    widths : jnp.ndarray
        Bin widths, shape ``(..., D, n_bins)``.
    heights : jnp.ndarray
        Bin heights, shape ``(..., D, n_bins)``.
    derivatives : jnp.ndarray
        Knot derivatives, shape ``(..., D, n_bins + 1)``.
    boundary : float
        Spline boundary.

    Returns
    -------
    x : jnp.ndarray
        Inverse-transformed values, same shape as ``y``.
    log_det : jnp.ndarray
        Log-determinant of the *inverse* Jacobian, shape ``(...)``.
        This equals ``-log_det_forward``.
    """
    cumwidths, cumheights, widths, heights = _build_knots(
        widths, heights, boundary,
    )
    n_bins = widths.shape[-1]

    # Find which bin each y falls into (using cumheights)
    y_expanded = y[..., None]
    bin_idx = jnp.sum(y_expanded >= cumheights, axis=-1) - 1
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    x_k = _gather(cumwidths, bin_idx)
    x_k1 = _gather(cumwidths, bin_idx + 1)
    y_k = _gather(cumheights, bin_idx)
    y_k1 = _gather(cumheights, bin_idx + 1)
    d_k = _gather(derivatives, bin_idx)
    d_k1 = _gather(derivatives, bin_idx + 1)

    w_k = x_k1 - x_k
    h_k = y_k1 - y_k
    s_k = h_k / w_k

    # Solve quadratic for xi
    y_rel = y - y_k
    a = h_k * (s_k - d_k) + y_rel * (d_k1 + d_k - 2.0 * s_k)
    b = h_k * d_k - y_rel * (d_k1 + d_k - 2.0 * s_k)
    c = -s_k * y_rel

    discriminant = b * b - 4.0 * a * c
    discriminant = jnp.maximum(discriminant, 0.0)  # numerical safety

    xi = 2.0 * c / (-b - jnp.sqrt(discriminant))
    xi = jnp.clip(xi, 0.0, 1.0)

    x_in = x_k + xi * w_k

    # Log-det in log-space (same formula as forward, negated at the end)
    xi_1mxi = xi * (1.0 - xi)
    denominator = s_k + (d_k1 + d_k - 2.0 * s_k) * xi_1mxi
    log_deriv = _rqs_log_deriv(s_k, d_k, d_k1, xi, xi_1mxi, denominator)

    # Identity outside boundary
    inside = (y >= -boundary) & (y <= boundary)
    x = jnp.where(inside, x_in, y)
    log_deriv = jnp.where(inside, log_deriv, 0.0)

    # Inverse log-det is negative of forward
    log_det = -jnp.sum(log_deriv, axis=-1)
    return x, log_det

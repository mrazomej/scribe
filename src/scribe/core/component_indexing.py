"""
Shared utilities for mixture-component indexing and renormalization.

This module centralizes parsing and validation of component selectors so SVI and
MCMC result classes behave consistently for:

- single integers (e.g. ``0``)
- slices (e.g. ``1:4``)
- integer lists/arrays (e.g. ``[0, 2, 3]``)
- boolean masks of length ``n_components``
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.nn import log_softmax


def normalize_component_indices(
    selector: Any,
    n_components: int,
) -> jnp.ndarray:
    """
    Normalize a component selector to a validated integer index array.

    Parameters
    ----------
    selector : Any
        Component selector. Supported types are:

        - ``int``: single component index.
        - ``slice``: interpreted against ``range(n_components)``.
        - sequence / array of integers.
        - boolean mask with length ``n_components``.
    n_components : int
        Total number of components in the mixture model.

    Returns
    -------
    jnp.ndarray
        One-dimensional integer array of validated component indices.

    Raises
    ------
    ValueError
        If the selector is empty, out of bounds, or invalid for
        ``n_components``.
    TypeError
        If the selector type is unsupported.
    """
    if n_components is None or n_components <= 1:
        raise ValueError(
            "Component selection requires a mixture model with multiple "
            "components"
        )

    # Normalize the selector into a 1D numpy array so we can robustly inspect
    # dtype and shape before converting to JAX.
    if isinstance(selector, int):
        indices_np = np.array([selector], dtype=int)
    elif isinstance(selector, slice):
        indices_np = np.arange(n_components)[selector].astype(int)
    elif isinstance(selector, (list, tuple, np.ndarray, jnp.ndarray)):
        selector_np = np.asarray(selector)
        if selector_np.ndim != 1:
            raise ValueError(
                "Component selector arrays must be one-dimensional."
            )
        if selector_np.dtype == bool:
            if selector_np.shape[0] != n_components:
                raise ValueError(
                    "Boolean component mask must have length "
                    f"{n_components}, got {selector_np.shape[0]}."
                )
            indices_np = np.flatnonzero(selector_np).astype(int)
        else:
            if not np.issubdtype(selector_np.dtype, np.integer):
                raise TypeError(
                    "Component selector arrays must contain integers or "
                    "booleans."
                )
            indices_np = selector_np.astype(int)
    else:
        raise TypeError(
            "Unsupported component selector type: "
            f"{type(selector)}. Expected int, slice, integer array/list, "
            "or boolean mask."
        )

    if indices_np.size == 0:
        raise ValueError("Component selector is empty.")

    if np.any(indices_np < 0) or np.any(indices_np >= n_components):
        raise ValueError(
            f"Component indices out of range [0, {n_components - 1}]: "
            f"{indices_np.tolist()}"
        )

    # Preserve first-occurrence ordering while removing duplicates.
    seen = set()
    unique_ordered = []
    for idx in indices_np.tolist():
        if idx not in seen:
            unique_ordered.append(idx)
            seen.add(idx)

    return jnp.asarray(unique_ordered, dtype=jnp.int32)


def renormalize_mixing_weights(
    weights: jnp.ndarray, axis: int = -1
) -> jnp.ndarray:
    """
    Renormalize selected mixing weights along the component axis.

    Parameters
    ----------
    weights : jnp.ndarray
        Selected mixing weights (e.g. shape ``(n_samples, k_selected)``).
    axis : int, default=-1
        Axis corresponding to components.

    Returns
    -------
    jnp.ndarray
        Renormalized weights that sum to one along ``axis``.

    Raises
    ------
    ValueError
        If any row has non-positive total mass.
    """
    mass = jnp.sum(weights, axis=axis, keepdims=True)
    if bool(jnp.any(mass <= 0)):
        raise ValueError(
            "Cannot renormalize mixing weights: selected components have "
            "non-positive total mass."
        )
    return weights / mass


def renormalize_mixing_logits(
    logits: jnp.ndarray, axis: int = -1
) -> jnp.ndarray:
    """
    Recenter selected mixing logits to represent a renormalized simplex.

    Parameters
    ----------
    logits : jnp.ndarray
        Selected logits (e.g. shape ``(n_samples, k_selected)``).
    axis : int, default=-1
        Axis corresponding to components.

    Returns
    -------
    jnp.ndarray
        Log-probabilities obtained from ``log_softmax(logits)``.
    """
    return log_softmax(logits, axis=axis)

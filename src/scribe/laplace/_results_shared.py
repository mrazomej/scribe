"""Shared helpers for Laplace results mixins.

This module centralizes cross-cutting utilities used by multiple Laplace
results mixins and private backend helpers:

1. numerical safety bounds for log-rate exponentiation,
2. default chunk-size configuration for PPC memory management,
3. base-model extraction from ``ModelConfig`` for dispatch,
4. robust metadata slicing for optional ``var`` payloads, and
5. ALR reference-axis augmentation utilities.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp

from ..models.config import ModelConfig

# Floor / ceiling used inside ``exp`` to prevent float32 overflow in
# rates. The bounds match Newton-kernel safety bounds so PPC and
# optimization use the same numerical envelope.
_LOG_RATE_MIN = -30.0
_LOG_RATE_MAX = 30.0

# Default chunk size for posterior predictive sampling. Sampling is
# intentionally chunked to keep device intermediates bounded for
# large single-cell matrices.
_PPC_DEFAULT_SAMPLE_CHUNK = 16


def _base_model(model_config: Optional[ModelConfig]) -> str:
    """Extract the base model name for dispatch decisions.

    Parameters
    ----------
    model_config : ModelConfig or None
        Fitted model configuration from the results object.

    Returns
    -------
    str
        Base-model name such as ``"pln"``, ``"lnm"``, or ``"lnmvcp"``.
        When ``model_config`` is ``None``, defaults to ``"pln"`` for
        backward compatibility with legacy hand-constructed objects.
    """
    if model_config is None:
        return "pln"
    return getattr(model_config, "base_model", "pln")


def _subset_var(var: Optional[Any], idx) -> Optional[Any]:
    """Slice ``var`` metadata when it is DataFrame-like.

    Parameters
    ----------
    var : Any or None
        AnnData-style ``var`` metadata payload.
    idx : Any
        Positional indexer accepted by ``.iloc`` on pandas-like objects.

    Returns
    -------
    Any or None
        Subsetted metadata when slicing is supported; otherwise the
        original object unchanged.
    """
    if var is None:
        return None
    try:
        return var.iloc[idx]
    except (TypeError, AttributeError):
        return var


def _augment_with_reference(
    y_alr: jnp.ndarray, alr_reference_idx: int, n_genes: int
) -> jnp.ndarray:
    """Embed ALR logits into full-gene logits by zero-reference insertion.

    Parameters
    ----------
    y_alr : jnp.ndarray
        ALR coordinates with trailing axis length ``G−1``. The leading axes
        are arbitrary (for example ``(C, G−1)`` or ``(S, C, G−1)``).
    alr_reference_idx : int
        Zero-based position of the ALR reference gene in full-gene indexing.
    n_genes : int
        Full gene count ``G``.

    Returns
    -------
    jnp.ndarray
        Full-gene logits ``y_full`` with shape ``y_alr.shape[:-1] + (G,)``.
        The inserted reference coordinate is exactly zero.

    Notes
    -----
    If ``r`` is the reference index, this function implements:

    - ``y_full[..., r] = 0``
    - ``y_full[..., j] = y_alr[..., j']`` for all non-reference genes,
      preserving original order.

    This is the inverse embedding needed before applying full-simplex
    transforms such as ``softmax`` or ``log_softmax`` over all ``G`` genes.
    """
    leading = y_alr.shape[:-1]
    full = jnp.zeros(leading + (n_genes,), dtype=y_alr.dtype)
    other = list(range(n_genes))
    other.remove(int(alr_reference_idx))
    return full.at[..., jnp.asarray(other)].set(y_alr)


"""Data-derived initializers for the TwoState-LogNormal (TSLN) Laplace fit.

This module provides empirical inits for the TSLN-Laplace path when
the user does *not* supply a cascade source from an upstream TwoState
SVI fit.  The primary use case for TSLN-Laplace is **cascade-from-SVI**
(``scribe.fit(..., cascade_source=svi_results)``), where the gene-level
globals ``(mu, burst_size, k_off)`` come from the upstream fit's MAP.
This data-init path is the fallback for ``cascade_source=None`` plus
the explicit ``allow_uncascade_fit=True`` opt-in.

Inits provided
--------------
- ``mu_init``: per-gene empirical mean expression. Reuses
  ``empirical_mean_from_counts`` from PLN data init.
- ``burst_size_init``: defaults to ``1.0`` per gene (matching the SVI
  prior median for ``burst_size ~ softplus(Normal(0, 1.5))``).
- ``k_off_init``: defaults to ``3.0`` per gene (matching the SVI prior
  median for ``k_off ~ softplus(Normal(3, 2))``).
- ``W_init``: PCA loadings on log-counts. Reuses
  ``pca_loadings_init`` from PLN data init.
- ``d_init``: small uniform unconstrained (matches NBLN).
- ``latent_loc_init``: per-cell ``log(counts + 1.0)`` — warm-starts
  ``x_c`` near the empirical log-counts.

The dispersion estimator from NBLN (method-of-moments on the marginal
variance) does **not** carry over usefully to TSLN's
``(burst_size, k_off)`` because those two parameters together determine
overdispersion via a Beta concentration, and disentangling them from
the marginal moments alone is ambiguous. The cascade pipeline solves
this by reading the upstream SVI fit's posterior; the data-init
fallback uses the prior medians (well-calibrated defaults).

Why no VAE / encoder warm-start?
--------------------------------
NBLN-VAE has a decoder bias + PCA loadings + encoder standardization
warm-start path because the VAE has those components. TSLN-Laplace is
**not** a VAE path — it's a per-cell Newton + outer Adam EM on the
globals. The latent ``x_c`` is solved per-cell, not produced by an
encoder. So the relevant inits are the global-level
``(mu, burst_size, k_off, W, d)`` and the per-cell ``x_init`` warm
start. There's no encoder to standardize.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .pln_data_init import (
    empirical_log_mean_from_counts,
    pca_loadings_init,
)

# Re-export PLN helpers for a single TSLN-init import path.
__all__ = [
    "empirical_log_mean_from_counts",
    "empirical_mean_from_counts",
    "pca_loadings_init",
    "empirical_burst_size_from_counts",
    "empirical_k_off_from_counts",
    "default_d_init",
    "latent_loc_init_from_counts",
]


# Defaults match the SVI prior medians documented in
# ``paper/_two_state_promoter.qmd`` and the TwoState SVI
# parameterizations in ``scribe.models.parameterizations``.
_BURST_SIZE_DEFAULT = 1.0
_K_OFF_DEFAULT = 3.0
_D_DEFAULT = 0.1


def empirical_mean_from_counts(counts) -> jnp.ndarray:
    """Per-gene empirical mean expression (positive scale).

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``, dtype ``float32``.
    """
    counts_np = np.asarray(counts, dtype=np.float64)
    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )
    # Floor at a small positive value so pos_inverse(...) doesn't blow up.
    mu = np.maximum(counts_np.mean(axis=0), 1e-3)
    return jnp.asarray(mu, dtype=jnp.float32)


def empirical_burst_size_from_counts(
    counts,
    default: float = _BURST_SIZE_DEFAULT,
) -> jnp.ndarray:
    """Per-gene burst-size init.

    For TSLN-Laplace data init we use the SVI prior median (default
    ``1.0``).  Disentangling ``burst_size`` from ``k_off`` using only
    marginal moments is ambiguous; the cascade-from-SVI path resolves
    this properly. See module docstring.

    Parameters
    ----------
    counts : array_like, shape ``(n_cells, n_genes)``
    default : float, default ``1.0``
        Constant per-gene value.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``, dtype ``float32``.
    """
    counts_np = np.asarray(counts)
    if counts_np.ndim != 2:
        raise ValueError(
            f"counts must be 2-D, got shape {counts_np.shape}."
        )
    G = counts_np.shape[1]
    return jnp.full((G,), float(default), dtype=jnp.float32)


def empirical_k_off_from_counts(
    counts,
    default: float = _K_OFF_DEFAULT,
) -> jnp.ndarray:
    """Per-gene k_off init.

    Defaults to the SVI prior median (``3.0``).  See
    ``empirical_burst_size_from_counts`` for the rationale.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``, dtype ``float32``.
    """
    counts_np = np.asarray(counts)
    if counts_np.ndim != 2:
        raise ValueError(
            f"counts must be 2-D, got shape {counts_np.shape}."
        )
    G = counts_np.shape[1]
    return jnp.full((G,), float(default), dtype=jnp.float32)


def default_d_init(n_genes: int, value: float = _D_DEFAULT) -> jnp.ndarray:
    """Default diagonal-variance init for the latent prior covariance.

    Matches NBLN's default: ``d_g = 0.1`` uniform.  In unconstrained
    space (where the optimizer operates), the obs model applies
    ``pos_inverse`` to recover ``d_loc``.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``, dtype ``float32``.
    """
    return jnp.full((int(n_genes),), float(value), dtype=jnp.float32)


def latent_loc_init_from_counts(counts) -> jnp.ndarray:
    """Per-cell per-gene latent-log-rate warm start.

    Returns ``log(counts + 1.0)`` as a ``float32`` array shaped
    ``(n_cells, n_genes)``.  Matches NBLN's pattern.
    """
    counts_np = np.asarray(counts, dtype=np.float64)
    if counts_np.ndim != 2:
        raise ValueError(
            f"counts must be 2-D, got shape {counts_np.shape}."
        )
    x = np.log1p(counts_np)  # log(1 + counts)
    return jnp.asarray(x, dtype=jnp.float32)

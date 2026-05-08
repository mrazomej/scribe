"""Likelihood helper backends for Laplace results mixins.

These functions implement model-specific MAP log-likelihood computations used
by the public Laplace likelihood mixin.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from ._results_shared import (
    _LOG_RATE_MAX,
    _LOG_RATE_MIN,
    _augment_with_reference,
)


def _ll_pln(
    counts: jnp.ndarray,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    return_by: str,
) -> jnp.ndarray:
    """Compute PLN Poisson log-likelihood at stored per-cell MAP state.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts with shape ``(n_cells, G)``.
    x_loc : jnp.ndarray
        Per-cell PLN MAP log-rates with shape ``(n_cells, G)``.
    eta_loc : jnp.ndarray or None
        Optional per-cell capture offsets with shape ``(n_cells,)``.
    return_by : {"cell", "gene"}
        Reduction mode for the log-likelihood matrix.

    Returns
    -------
    jnp.ndarray
        Per-cell or per-gene log-likelihood totals.

    Notes
    -----
    ``return_by="gene"`` is a straightforward reduction for PLN because the
    Poisson likelihood factorizes over genes.
    """
    from jax.scipy.special import gammaln

    u = jnp.asarray(counts, dtype=jnp.float32)
    eff = x_loc - eta_loc[:, None] if eta_loc is not None else x_loc
    eff = jnp.clip(eff, _LOG_RATE_MIN, _LOG_RATE_MAX)
    rate = jnp.exp(eff)
    log_pmf = u * eff - rate - gammaln(u + 1.0)
    if return_by == "cell":
        return log_pmf.sum(axis=-1)
    return log_pmf.sum(axis=0)


def _ll_lnm(
    counts: jnp.ndarray,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    z_loc: Optional[jnp.ndarray],
    y_alr_loc: Optional[jnp.ndarray],
    alr_reference_idx: Optional[int],
    return_by: str,
) -> jnp.ndarray:
    """Compute LNM/LNMVCP multinomial log-likelihood at MAP state.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts with shape ``(n_cells, G)``.
    mu : jnp.ndarray
        ALR-space decoder bias with shape ``(G-1,)``.
    W : jnp.ndarray
        ALR-space decoder loadings with shape ``(G-1, k)``.
    z_loc : jnp.ndarray or None
        Per-cell low-rank latent MAP values with shape ``(n_cells, k)``.
    y_alr_loc : jnp.ndarray or None
        Per-cell ALR latent MAP values with shape ``(n_cells, G-1)``.
    alr_reference_idx : int or None
        Reference gene index used in ALR coordinates.
    return_by : {"cell", "gene"}
        Reduction mode for the log-likelihood output.

    Returns
    -------
    jnp.ndarray
        Per-cell full multinomial log-likelihood values or per-gene data-term
        contributions.

    Notes
    -----
    For ``return_by="gene"``, the return value contains only the data-term
    contributions ``Σ_c u_cg log p_cg``. The multinomial normalizer couples
    genes and is therefore assigned only in the per-cell reduction path.
    """
    from jax.scipy.special import gammaln

    if alr_reference_idx is None:
        raise ValueError("LNM log-likelihood requires alr_reference_idx.")
    u = jnp.asarray(counts, dtype=jnp.float32)
    if y_alr_loc is not None:
        y_alr = y_alr_loc
    elif z_loc is not None:
        y_alr = mu[None, :] + z_loc @ W.T
    else:
        raise ValueError(
            "LNM log-likelihood requires either z_loc or y_alr_loc."
        )

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1
    log_p = jax.nn.log_softmax(
        _augment_with_reference(y_alr, alr_reference_idx, n_genes), axis=-1
    )

    n_per_cell = u.sum(axis=-1)
    log_pmf_per_cell_per_gene = u * log_p
    norm_per_cell = gammaln(n_per_cell + 1.0) - gammaln(u + 1.0).sum(axis=-1)
    if return_by == "cell":
        return log_pmf_per_cell_per_gene.sum(axis=-1) + norm_per_cell
    return log_pmf_per_cell_per_gene.sum(axis=0)

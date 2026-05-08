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
    """Compute PLN MAP log-likelihood under a Poisson observation model.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed count matrix ``u`` with shape ``(C, G)`` where ``C`` is the
        number of cells and ``G`` is the number of genes.
    x_loc : jnp.ndarray
        Per-cell MAP latent log-rate matrix ``x`` with shape ``(C, G)``.
    eta_loc : jnp.ndarray or None
        Optional per-cell capture offsets ``η`` with shape ``(C,)``. When
        provided, effective log-rates are ``x_cg - η_c``.
    return_by : {"cell", "gene"}
        Aggregation axis for the elementwise log-likelihood table.

    Returns
    -------
    jnp.ndarray
        If ``return_by="cell"``, returns ``(C,)`` with per-cell totals:

        ``ℓ_c = Σ_g log p(u_cg | λ_cg)``.

        If ``return_by="gene"``, returns ``(G,)`` with per-gene totals:

        ``ℓ_g = Σ_c log p(u_cg | λ_cg)``.

    Notes
    -----
    The Poisson mean is parameterized by

    ``log λ_cg = clip(x_cg - η_c, log_rate_min, log_rate_max)``

    (or ``log λ_cg = clip(x_cg, ...)`` when no ``η`` is available), and

    ``log p(u_cg | λ_cg) = u_cg log λ_cg - λ_cg - log(u_cg!)``.

    Because the PLN likelihood factorizes over genes, per-gene and per-cell
    reductions are both exact regroupings of the same elementwise terms.
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
    """Compute LNM-family MAP multinomial log-likelihood.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed count matrix ``u`` with shape ``(C, G)``.
    mu : jnp.ndarray
        ALR-space decoder bias ``μ`` with shape ``(G−1,)``.
    W : jnp.ndarray
        ALR-space loading matrix ``W`` with shape ``(G−1, K)``.
    z_loc : jnp.ndarray or None
        Low-rank per-cell MAP latent coordinates with shape ``(C, K)``.
    y_alr_loc : jnp.ndarray or None
        Direct ALR-space per-cell MAP logits with shape ``(C, G−1)``.
    alr_reference_idx : int or None
        Index of the ALR reference gene used for zero-insertion embedding.
    return_by : {"cell", "gene"}
        Aggregation axis for log-likelihood reporting.

    Returns
    -------
    jnp.ndarray
        If ``return_by="cell"``, returns ``(C,)`` full multinomial
        log-likelihood values:

        ``ℓ_c = log(N_c!) - Σ_g log(u_cg!) + Σ_g u_cg log p_cg``.

        If ``return_by="gene"``, returns ``(G,)`` data-term contributions:

        ``t_g = Σ_c u_cg log p_cg``.

    Notes
    -----
    Two latent parameterization branches are supported:

    - **Direct ALR branch**: ``y_alr = y_alr_loc``.
    - **Low-rank branch**: ``y_alr = μ + z_loc Wᵀ``.

    Full-gene logits are reconstructed by inserting a zero at the ALR
    reference index, then normalized with

    ``log p_c· = log_softmax(y_full,c·)``.

    For ``return_by="gene"``, only the additive data term is returned.
    The combinatorial normalizer depends on whole-cell totals and does not
    decompose into gene-attributable components.
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

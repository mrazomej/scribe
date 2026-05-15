"""Data-derived initialization for the TwoState (Poisson-Beta) family.

Mirrors the NBLN data-init pattern (``r_prior_loc``) but for the
TwoState model's gene-mean parameter ``mu``. Without this, the default
prior ``softplus(Normal(0, 1))`` puts mu ~ 0.7 with a 95% upper bound
near 5 — many orders of magnitude below realistic gene means after the
gene-coverage filter (which keeps the top-expressing genes). The
result is a Poisson rate vastly smaller than the observed counts at
SVI step 0, producing log-PMF underflow and NaN gradients that take
thousands of steps to recover from.

The fix is the same as for NBLN: compute an empirical log-mean per
gene, take the median as a global anchor, and stash it as
``mu_prior_loc`` on the priors extras. The factory reads this on the
next pass and overrides the ``mu`` spec's prior loc accordingly.

This is a *global scalar* anchor, not a per-gene one — the spec API
is scalar by contract. Per-gene initialization would require a
spec-level change; the scalar anchor is sufficient to start SVI in
the right neighborhood, and the per-gene posterior moves from there.
"""

from __future__ import annotations

import jax.numpy as jnp


__all__ = [
    "empirical_log_mean_from_counts",
    "inject_twostate_data_init",
]


def empirical_log_mean_from_counts(
    counts,
    pseudocount: float = 1.0,
) -> jnp.ndarray:
    """Compute per-gene empirical log-mean of the count matrix.

    Parameters
    ----------
    counts : array_like, shape ``(n_cells, n_genes)``
        Raw count matrix (after any gene-coverage filtering).
    pseudocount : float, default 1.0
        Additive pseudocount applied before the log to keep
        zero-expression genes finite.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``
        ``log(mean(counts[:, g]) + pseudocount)`` per gene.
    """
    counts_arr = jnp.asarray(counts, dtype=jnp.float32)
    per_gene_mean = counts_arr.mean(axis=0)
    return jnp.log(per_gene_mean + pseudocount)


def inject_twostate_data_init(model_config, counts):
    """Stash a data-derived ``mu_prior_loc`` on the priors extras.

    The factory reads ``priors.__pydantic_extra__["mu_prior_loc"]`` to
    override the default prior loc on ``mu`` for TwoState models. We
    use the *median* of the per-gene log-means rather than the mean
    because the right tail (housekeeping genes) is much longer than
    the left tail; the median is a more robust centre for the
    Normal-on-unconstrained prior.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly built configuration.
    counts : array_like, shape ``(n_cells, n_genes)``
        Count matrix after gene-coverage filtering.

    Returns
    -------
    ModelConfig
        New ``ModelConfig`` with the priors extras augmented. The
        original is not mutated.
    """
    log_means = empirical_log_mean_from_counts(counts)
    mu_prior_loc = float(jnp.median(log_means))

    from ..models.config.groups import PriorOverrides

    existing_extras = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    # PriorOverrides is a Pydantic model with model_config["extra"] =
    # "allow", so the extra fields round-trip via the constructor.
    updated = dict(existing_extras)
    updated["mu_prior_loc"] = mu_prior_loc
    # Carry forward any structured priors fields from the original
    # PriorOverrides (Beta/LogNormal tuples for p_capture, etc.).
    if model_config.priors is not None:
        for name in model_config.priors.model_fields_set:
            if name not in updated:
                updated[name] = getattr(model_config.priors, name)

    return model_config.model_copy(
        update={"priors": PriorOverrides(**updated)}
    )

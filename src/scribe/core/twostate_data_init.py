"""Data-derived initialization for the TwoState (Poisson-Beta) family.

For each gene ``g``, we anchor the prior loc of ``mu_g`` at the
unconstrained-space value whose transformed image equals the empirical
mean.  Concretely:

- For ``positive_transform="softplus"`` (default): ``loc_g = softplus_inv
  (empirical_mean_g)`` so that ``softplus(loc_g) ≈ empirical_mean_g``.
  Softplus is asymptotically linear, so ``loc_g`` numerically equals
  ``empirical_mean_g`` for moderately or strongly expressed genes.
- For ``positive_transform="exp"``: ``loc_g = log(empirical_mean_g)``,
  the classic log-mean anchor (mirrors NBLN's ``r_prior_loc`` shape).

This anchors each gene's variational ``mu_loc`` at the right *order of
magnitude* on the first SVI step — without it, highly-expressed genes
(e.g. ribosomal genes with mean ≈ 100–500) start at ``mu ≈ 1`` and
have to climb 4-5 decades against bounded SVI gradients, which the
optimizer may not finish inside a reasonable step budget.
"""

from __future__ import annotations

import jax.numpy as jnp


__all__ = [
    "empirical_log_mean_from_counts",
    "empirical_mu_anchor_from_counts",
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


def empirical_mu_anchor_from_counts(
    counts,
    transform: str = "softplus",
    floor: float = 1e-3,
    ceil: float = 1e6,
) -> jnp.ndarray:
    """Per-gene unconstrained-space anchor for ``mu``.

    Computes the empirical per-gene mean, clamps it to ``[floor, ceil]``,
    then maps through the inverse of the configured positive transform.

    Parameters
    ----------
    counts : array_like, shape ``(n_cells, n_genes)``
        Raw count matrix.
    transform : {"softplus", "exp"}, default "softplus"
        The positive transform used in the model.  The inverse used
        here is ``log(expm1(x))`` for softplus and ``log(x)`` for exp.
    floor : float, default 1e-3
        Lower clamp on the empirical mean before inversion (keeps zero-
        expression genes finite under both ``log`` and ``log·expm1``).
    ceil : float, default 1e6
        Upper clamp; protects ``expm1`` from overflowing in float32.

    Returns
    -------
    jnp.ndarray, shape ``(n_genes,)``
        Per-gene unconstrained-space loc for the ``mu`` prior.
    """
    counts_arr = jnp.asarray(counts, dtype=jnp.float32)
    per_gene_mean = counts_arr.mean(axis=0)
    per_gene_mean = jnp.clip(per_gene_mean, min=floor, max=ceil)

    if transform == "softplus":
        # Stable softplus_inv: log(expm1(x)) overflows in float32
        # for x ≳ 88, so we split on a threshold.
        # For x <= 20: log(expm1(x))   — the naive form, stable here.
        # For x  > 20: x + log1p(-exp(-x))
        #              ≡ log(exp(x) - 1) without computing exp(x).
        # The two branches agree to high precision around x=20.
        return jnp.where(
            per_gene_mean > 20.0,
            per_gene_mean + jnp.log1p(-jnp.exp(-per_gene_mean)),
            jnp.log(jnp.expm1(per_gene_mean)),
        )
    if transform == "exp":
        return jnp.log(per_gene_mean)
    raise ValueError(
        f"Unknown positive transform {transform!r}; expected "
        "'softplus' or 'exp'."
    )


def inject_twostate_data_init(model_config, counts):
    """Stash a per-gene ``mu_prior_loc`` array on the priors extras.

    The factory reads ``priors.__pydantic_extra__["mu_prior_loc"]`` and
    replaces the default scalar prior loc on ``mu`` with this array.
    Each gene's variational ``mu_loc`` then initializes at its own
    empirical mean (in unconstrained space), so the optimizer doesn't
    need to climb decades on the first SVI step.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly built configuration.  The ``positive_transform`` field
        (default "softplus") controls the inverse-transform used to
        compute the per-gene anchor.
    counts : array_like, shape ``(n_cells, n_genes)``
        Count matrix after gene-coverage filtering.

    Returns
    -------
    ModelConfig
        New ``ModelConfig`` with the priors extras augmented.  The
        original is not mutated.
    """
    transform = getattr(model_config, "positive_transform", "softplus")
    mu_prior_loc = empirical_mu_anchor_from_counts(counts, transform=transform)

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

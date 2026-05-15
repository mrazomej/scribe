"""Data-derived initialization for the TwoState (Poisson-Beta) family.

Mirrors the NBLN data-init pattern (``r_prior_loc``) but for the
TwoState model's two well-identifiable per-gene parameters: the gene
mean ``mu`` and the NB-limit burst size ``burst_size``. Without this,
the default priors put both parameters many orders of magnitude away
from realistic values for typical scRNA-seq data, and SVI either
starts with NaN losses (rate near 0 underflows the Poisson log-PMF)
or takes many thousands of steps to climb out of the bad regime.

The fix is the same as for NBLN: compute method-of-moments
estimates per gene, take the median as a global scalar anchor, and
stash them on the priors extras. The factory reads these on the
next pass and overrides the relevant specs' prior locs.

Two anchors are produced:

- ``mu_prior_loc`` = ``median(log(mean(counts) + pseudocount))``.

- ``burst_size_prior_loc`` = ``log(median(max(Fano − 1, b_floor)))``
  per gene. The NB-limit identity is ``Fano = 1 + b`` (b being the
  mean number of mRNAs per ON visit), so the Fano factor minus 1
  is a direct method-of-moments estimator for the burst size. We
  floor at ``b_floor`` per gene so that sub-Poisson or
  near-Poisson genes (rare in practice) do not pull the median
  toward 0 or negative values.

The third parameter, ``k_off``, is left at its default prior
``softplus(Normal(3, 2))``. In the NB limit it is structurally
under-identified by the data moments (Fano and mean fix only two
combinations of the three parameters), so a data-driven anchor on
``k_off`` would be approximate and not obviously better than the
default. Bursty genes will pull ``k_off`` down during SVI; NB-like
genes will leave it near the prior.

Both anchors are *global scalar* values, not per-gene — the spec
API is scalar by contract. Per-gene initialization would require a
spec-level change; the scalar anchors are sufficient to start SVI
in the right neighborhood, and per-gene posteriors move from there.
"""

from __future__ import annotations

import jax.numpy as jnp


__all__ = [
    "empirical_log_mean_from_counts",
    "empirical_burst_size_from_counts",
    "inject_twostate_data_init",
]


# Per-gene floor on the (Fano − 1) burst-size estimator. Genes
# where Fano ≤ 1 + b_floor (i.e. ~Poisson or sub-Poisson) get
# clamped to b_floor so they do not push the median toward 0.
_BURST_FLOOR = 0.05

# Numerical floor on per-gene mean when computing Fano. Zero-
# expression genes (mean = 0) have an undefined Fano factor; we
# do not include them in the burst-size estimator.
_MEAN_FLOOR = 1e-3


def _inverse_positive_transform(values, transform: str) -> jnp.ndarray:
    """Invert the configured positive transform.

    The TwoState parameterization uses Normal+softplus by default
    (configurable to Normal+exp via ``model_config.positive_transform``).
    To set an unconstrained-space prior loc that produces a target
    *post-transform* value, we have to apply the inverse transform.

    For softplus, the exact inverse is ``log(exp(y) − 1)`` (= ``log(expm1(y))``);
    for ``y >> 1`` this is ≈ ``y`` (so log(value) is a poor
    approximation in that regime — it would give roughly log(value)
    after the softplus, not value). For exp, the inverse is plain
    ``log``.

    Parameters
    ----------
    values : jnp.ndarray
        Positive-valued targets.
    transform : str
        Either ``"softplus"`` or ``"exp"``.

    Returns
    -------
    jnp.ndarray
        Unconstrained-space loc values that produce ``values`` after
        the corresponding transform.
    """
    if transform == "softplus":
        # log(exp(y) - 1); jnp.log(jnp.expm1(y)) is numerically stable
        # for y > ~1e-7 (below that, expm1(y) → 0 and the log
        # underflows). The valid-gene floor in our callers keeps us
        # safely above that.
        return jnp.log(jnp.expm1(values))
    # Default / "exp" path: the inverse of exp is log.
    return jnp.log(values)


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


def empirical_burst_size_from_counts(counts) -> jnp.ndarray:
    """Method-of-moments per-gene burst-size estimator.

    Uses the NB-limit identity ``Fano = 1 + b``:

        b_hat_g = max(var(counts_g) / mean(counts_g) − 1, _BURST_FLOOR).

    Genes with mean ≤ ``_MEAN_FLOOR`` are excluded (their Fano is
    numerically undefined and their burst size is not identified).
    The remaining per-gene estimates are returned as a 1-D array;
    callers typically reduce to a scalar with ``jnp.median``.

    Parameters
    ----------
    counts : array_like, shape ``(n_cells, n_genes)``
        Raw count matrix.

    Returns
    -------
    jnp.ndarray, shape ``(n_valid_genes,)``
        Per-gene burst-size estimates for genes with sufficient
        expression. May have fewer entries than ``n_genes``.
    """
    counts_arr = jnp.asarray(counts, dtype=jnp.float32)
    per_gene_mean = counts_arr.mean(axis=0)
    # ddof=0 (population variance) is consistent with the usual
    # method-of-moments derivation.
    per_gene_var = counts_arr.var(axis=0)
    valid = per_gene_mean > _MEAN_FLOOR
    # Compute Fano − 1 only on valid genes; floor at _BURST_FLOOR.
    safe_mean = jnp.where(valid, per_gene_mean, 1.0)
    fano_minus_one = per_gene_var / safe_mean - 1.0
    burst_per_gene = jnp.maximum(fano_minus_one, _BURST_FLOOR)
    # Mask out invalid genes by replacing with NaN, so the caller's
    # ``jnp.median`` ignores them implicitly via nanmedian. We use
    # nanmedian here to avoid forcing the caller to write the masking
    # boilerplate.
    return jnp.where(valid, burst_per_gene, jnp.nan)


def inject_twostate_data_init(model_config, counts):
    """Stash data-derived ``mu_prior_loc`` and ``burst_size_prior_loc``.

    Computes scalar method-of-moments anchors from the count matrix
    and stashes them on the priors extras. The factory reads
    ``priors.__pydantic_extra__["mu_prior_loc"]`` and
    ``priors.__pydantic_extra__["burst_size_prior_loc"]`` and uses
    each to override the corresponding spec's prior loc.

    Median (not mean) is used for both anchors: the per-gene
    distributions are right-skewed (a few housekeeping genes with
    very large means; a few highly-bursty genes with very large
    burst sizes), and the median is a more robust centre for the
    Normal-on-unconstrained prior loc.

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
    # Use the inverse of the configured positive transform so that
    # ``transform(loc) ≈ data_median`` (otherwise log(median) under
    # softplus would give a post-transform value of ~log(median),
    # which is wrong by a factor of ~e for moderate values).
    transform = getattr(model_config, "positive_transform", "softplus")

    # mu anchor: median per-gene mean (in linear space), then
    # inverse-transform to the unconstrained space.
    counts_arr = jnp.asarray(counts, dtype=jnp.float32)
    per_gene_mean = counts_arr.mean(axis=0)
    # Floor the median at a small positive value to keep the inverse
    # transform finite even when most genes are extremely sparse.
    mu_median = float(jnp.maximum(jnp.median(per_gene_mean), 1e-3))
    mu_prior_loc = float(
        _inverse_positive_transform(jnp.asarray(mu_median), transform)
    )

    # burst_size anchor: median of the per-gene Fano − 1 estimates,
    # in linear space, then inverse-transformed.
    burst_per_gene = empirical_burst_size_from_counts(counts)
    burst_median = float(jnp.nanmedian(burst_per_gene))
    # Guard against the edge case where every gene is invalid
    # (degenerate input); fall back to a sensible default of 1.0.
    if not jnp.isfinite(jnp.asarray(burst_median)):
        burst_median = 1.0
    burst_size_prior_loc = float(
        _inverse_positive_transform(jnp.asarray(burst_median), transform)
    )

    from ..models.config.groups import PriorOverrides

    existing_extras = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    # PriorOverrides is a Pydantic model with model_config["extra"] =
    # "allow", so the extra fields round-trip via the constructor.
    updated = dict(existing_extras)
    updated["mu_prior_loc"] = mu_prior_loc
    updated["burst_size_prior_loc"] = burst_size_prior_loc
    # Carry forward any structured priors fields from the original
    # PriorOverrides (Beta/LogNormal tuples for p_capture, etc.).
    if model_config.priors is not None:
        for name in model_config.priors.model_fields_set:
            if name not in updated:
                updated[name] = getattr(model_config.priors, name)

    return model_config.model_copy(
        update={"priors": PriorOverrides(**updated)}
    )

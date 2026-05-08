"""Data-derived initializers for the Negative Binomial-LogNormal (NBLN) model.

This module bundles the data-driven computations that the public API
performs once, up front, when the user requests an NBLN
(``model="nbln"``) fit. The goal is to warm-start the linear-decoder VAE
with sensible initial conditions derived from the count matrix.

Most of the VAE warm-start logic (decoder bias, PCA loadings for the
decoder kernel, encoder standardization stats) is identical to the
Poisson-LogNormal init and is reused directly from
:mod:`scribe.core.pln_data_init`. The piece unique to NBLN is the
method-of-moments estimator for the gene dispersion vector ``r_g``,
which is **not** present in PLN.

Why a method-of-moments r_g initializer?
----------------------------------------
The Negative Binomial second moment satisfies

    Var[u_g] = mean[u_g] + (mean[u_g])^2 / r_g,

so a per-gene moment estimator is

    r_g_hat = mean[u_g]^2 / max(Var[u_g] - mean[u_g], eps).

This places the variational guide for ``r`` near the data-implied scale
on the first iteration, sparing the optimizer from discovering it from
the (0, 1) log-Normal prior default. The estimator only uses the
empirical marginal moments — it ignores cross-gene correlations because
those are captured by the latent ``z``, not by ``r``.

When the empirical variance of a gene does not exceed its mean (the
sub-Poisson case, which arises for very low-count genes), the
denominator is floored so the estimator returns a large but finite ``r``
(approaching the Poisson limit, which is consistent with the bursty-NB
interpretation of small-mean genes).
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .pln_data_init import (
    empirical_log_mean_from_counts,
    pca_loadings_init,
)
from .lnm_data_init import compute_encoder_standardization

# Re-export for a single NBLN-init import path.
__all__ = [
    "empirical_log_mean_from_counts",
    "pca_loadings_init",
    "compute_encoder_standardization",
    "empirical_dispersion_from_counts",
    "inject_nbln_vae_data_init",
]


# Floor on the variance-minus-mean denominator in the moment estimator.
# When ``var <= mean`` (sub-Poisson), we fall back to this floor so the
# estimator returns a large finite ``r`` rather than ``+inf`` or NaN.
_VAR_MINUS_MEAN_FLOOR: float = 1e-3


def empirical_dispersion_from_counts(
    counts,
    var_minus_mean_floor: float = _VAR_MINUS_MEAN_FLOOR,
    pseudocount_var: float = 0.0,
) -> jnp.ndarray:
    """Marginal method-of-moments estimator for the per-gene NB dispersion ``r_g``.

    Uses the second moment of the per-gene count distribution to derive

        r_g_hat = mean[u_g]^2 / max(var[u_g] - mean[u_g], floor).

    For genes whose empirical variance does not exceed their mean
    (sub-Poisson, common for very low-count genes), the denominator is
    floored to ``var_minus_mean_floor``, yielding a large but finite
    ``r`` that approximates the Poisson limit.

    Bias note (important)
    ---------------------
    Under the NB-LogNormal generative model the *marginal* variance of
    each gene's count includes both the conditional NB variance
    (``mean + mean^2 / r_g``) and the lognormal regulatory contribution
    (``mean^2 (e^{Sigma_gg} - 1)``). The estimator above treats *all* of
    the marginal overdispersion as NB dispersion and so **systematically
    underestimates the conditional ``r_g``** when ``Sigma_gg > 0``. The
    bias direction is always "estimator < true conditional dispersion".

    For initialization purposes this is acceptable: it gives a
    finite-positive starting point in the right order of magnitude, and
    the optimizer adjusts during training. For downstream interpretation
    of ``r_g`` as a biological quantity you should use the inferred
    posterior, not this estimator.

    Parameters
    ----------
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix.
    var_minus_mean_floor : float, default 1e-3
        Lower bound on ``var - mean`` to keep the estimator finite for
        sub-Poisson genes. Smaller values produce larger ``r`` for
        low-count genes; ``1e-3`` is a reasonable default that yields
        ``r ~ mean^2 / 1e-3`` in the sub-Poisson regime.
    pseudocount_var : float, default 0.0
        Optional additive bias on the variance estimate. Set to a small
        positive value (e.g., 1e-2) if the unbiased variance estimator
        produces zero exactly for genes with very few non-zero cells.

    Returns
    -------
    jnp.ndarray, shape (n_genes,)
        Per-gene marginal method-of-moments estimate of ``r_g``. Useful
        as a free-standing diagnostic; not currently auto-consumed by
        ``build_r_spec``, which uses the standard LogNormal(0, 1) prior.
        Pass an explicit ``priors={"r": (loc, scale)}`` override if you
        want to anchor ``r_g`` to a data-driven scalar derived from this
        estimator (e.g., ``(jnp.log(jnp.median(r_hat)), 1.0)``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Synthetic NB(r=2, mean=10) counts
    >>> counts = rng.negative_binomial(n=2.0, p=2.0/12.0, size=(1000, 5))
    >>> r_hat = empirical_dispersion_from_counts(counts)
    >>> # Should be in the ballpark of r=2 for each gene.
    >>> r_hat.shape
    (5,)
    """
    counts_np = np.asarray(counts, dtype=np.float64)
    if counts_np.ndim != 2:
        raise ValueError(
            "counts must be a 2-D matrix (n_cells, n_genes), "
            f"got rank {counts_np.ndim} array with shape {counts_np.shape}."
        )

    mean = counts_np.mean(axis=0)
    var = counts_np.var(axis=0, ddof=1) + pseudocount_var

    # NB second-moment identity: Var = mean + mean^2 / r => r = mean^2 / (Var - mean)
    denom = np.maximum(var - mean, var_minus_mean_floor)
    # Guard against zero-mean genes (would yield 0/eps = 0, then a uniform
    # tiny r). Keep them at the prior anchor by defaulting r=1.0.
    r_hat = np.where(
        mean > 0,
        (mean ** 2) / denom,
        1.0,
    )
    return jnp.asarray(r_hat, dtype=jnp.float32)


def inject_nbln_vae_data_init(
    model_config,
    counts,
    latent_dim: int,
    pseudocount: float = 1.0,
):
    """Return a copy of ``model_config`` with NBLN data-init fields populated.

    This helper centralizes the transformation that ``scribe.api.fit``
    performs on a freshly-built ``ModelConfig`` when the user requests
    an NBLN model. It reuses the PLN warm-start path for the
    decoder bias, PCA-based loadings, and per-feature encoder
    standardization stats, since these are identical between PLN and
    NBLN (both use the same VAE log-rate decoder).

    The method-of-moments ``r_g`` estimator is computed and exposed on
    the returned config under the ``empirical_r_init`` extra prior key
    (read by the model factory when building the ``r`` spec, if
    plumbed). When the factory does not consume this field, the
    estimator is still available as a free-standing function for
    diagnostics and tests.

    Parameters
    ----------
    model_config : ModelConfig
        Freshly-built configuration.
    counts : array_like, shape (n_cells, n_genes)
        Raw count matrix.
    latent_dim : int
        Number of latent dimensions for PCA init.
    pseudocount : float, default=1.0
        Pseudocount for log transforms (passed through to PLN helpers).

    Returns
    -------
    ModelConfig
        New ``ModelConfig`` with data-init fields filled in.  The
        original is not mutated.
    """
    # Delegate the VAE warm-start to the PLN helper — the decoder bias,
    # PCA loadings, and encoder standardization are identical for NBLN
    # because both models share the POISSON_LOGNORMAL parameterization
    # and the same y_log_rate decoder head.
    from .pln_data_init import inject_pln_vae_data_init

    model_config = inject_pln_vae_data_init(
        model_config,
        counts,
        latent_dim=latent_dim,
        pseudocount=pseudocount,
    )

    # Compute the method-of-moments r_g estimator and reduce it to a
    # global scalar prior location for the LogNormal(loc, scale) prior
    # on ``r``. We use the *median* of the per-gene estimates rather
    # than the mean because the per-gene estimates are right-skewed
    # (the floor on ``var - mean`` produces large outliers for
    # sub-Poisson genes) and the median is the natural centre for a
    # LogNormal anchor.
    #
    # The per-gene array is intentionally NOT plumbed through, because
    # ``LogNormalSpec.default_params`` is scalar by contract. Per-gene
    # initialization would require a spec-level change; for now the
    # global scalar is the right level of plumbing. Users who want a
    # per-gene init can compute ``empirical_dispersion_from_counts``
    # themselves and pass an explicit ``priors={"r": (loc, scale)}``
    # override.
    r_init = empirical_dispersion_from_counts(counts)
    r_prior_loc = float(jnp.log(jnp.median(r_init)))

    _extra = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    _updated_priors = dict(_extra)
    _updated_priors["r_prior_loc"] = r_prior_loc

    from ..models.config.groups import PriorOverrides

    return model_config.model_copy(
        update={"priors": PriorOverrides(**_updated_priors)}
    )

"""
Sampling utilities for SCRIBE.

This module provides functions for posterior sampling, predictive sampling,
and posterior predictive checks (PPCs). It also provides:

* A **biological PPC** utility that strips technical noise parameters
  (capture probability, zero-inflation gate) and samples from the base
  Negative Binomial distribution only, reflecting the underlying biology
  without experimental artifacts.

* A **Bayesian denoising** utility that takes *observed* count matrices and
  posterior parameter estimates to compute the closed-form posterior of the
  true (pre-capture, pre-dropout) transcript counts.  See
  ``paper/_denoising.qmd`` for the full mathematical derivation.

Parameterization Convention
---------------------------
Throughout this module the canonical ``p`` follows the numpyro convention:
it is the ``probs`` argument of ``NegativeBinomialProbs``, i.e. the
probability of each Bernoulli trial producing a count.  The NB mean is
therefore ``r * p / (1 - p)``.  This is the *complement* of the paper's
``p`` (which appears as p^r in the PMF).
"""

# ---------------------------------------------------------------------------
# Re-exports: every name that was previously importable from
# ``scribe.sampling`` is re-exported here so existing imports remain
# unchanged.
# ---------------------------------------------------------------------------

# Shared helpers (used by svi/, mcmc/, viz/, and tests)
from ._helpers import (  # noqa: F401
    _build_canonical_layouts,
    _has_sample_dim,
    _slice_draw,
    _slice_posterior_draw,
    _slice_gene_axis,
)

# Variational and prior predictive sampling
from ._predictive import (  # noqa: F401
    sample_variational_posterior,
    generate_predictive_samples,
    generate_ppc_samples,
    generate_prior_predictive_samples,
)

# Biological (denoised) PPC — base NB only
from ._biological_ppc import (  # noqa: F401
    sample_biological_nb,
    _sample_biological_nb_single,
)

# Full-model posterior PPC (NB / ZINB / VCP / mixtures)
from ._posterior_ppc import (  # noqa: F401
    sample_posterior_ppc,
    _sample_posterior_ppc_single,
)

# BNB-specific quadrature/grid-sampling helpers
from ._denoising_bnb import (  # noqa: F401
    _BNB_DENOISE_EPS,
    _bnb_omega_to_alpha_kappa,
    _bnb_p_log_posterior_unnorm,
    _denoise_bnb_quadrature,
    _sample_p_posterior_bnb,
)

# Bayesian denoising core
from ._denoising import (  # noqa: F401
    _VALID_DENOISE_METHODS,
    _validate_denoise_method,
    _method_needs_rng,
    denoise_counts,
    _denoise_single,
    _denoise_standard,
    _denoise_batch,
    _compute_gate_weight,
    _denoise_mixture_marginal,
)

__all__ = [
    # helpers
    "_build_canonical_layouts",
    "_has_sample_dim",
    "_slice_draw",
    "_slice_posterior_draw",
    "_slice_gene_axis",
    # predictive
    "sample_variational_posterior",
    "generate_predictive_samples",
    "generate_ppc_samples",
    "generate_prior_predictive_samples",
    # biological PPC
    "sample_biological_nb",
    "_sample_biological_nb_single",
    # posterior PPC
    "sample_posterior_ppc",
    "_sample_posterior_ppc_single",
    # BNB helpers
    "_BNB_DENOISE_EPS",
    "_bnb_omega_to_alpha_kappa",
    "_bnb_p_log_posterior_unnorm",
    "_denoise_bnb_quadrature",
    "_sample_p_posterior_bnb",
    # denoising
    "_VALID_DENOISE_METHODS",
    "_validate_denoise_method",
    "_method_needs_rng",
    "denoise_counts",
    "_denoise_single",
    "_denoise_standard",
    "_denoise_batch",
    "_compute_gate_weight",
    "_denoise_mixture_marginal",
]

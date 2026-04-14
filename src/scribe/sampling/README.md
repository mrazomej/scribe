# SCRIBE Sampling

Posterior sampling, predictive checks, and Bayesian denoising utilities.

## Overview

This package provides the sampling layer for SCRIBE, covering three main
workflows:

1. **Predictive sampling** — draw from variational/prior predictive
   distributions via NumPyro's `Predictive`.
2. **Posterior predictive checks (PPC)** — generate replicate count data
   from posterior parameter estimates, either using the full generative model
   (NB / ZINB / VCP / BNB / mixtures) or using only the biological NB
   component (stripping capture efficiency and zero-inflation).
3. **Bayesian denoising** — compute the closed-form posterior of the true
   (pre-capture, pre-dropout) transcript counts given observed UMI counts
   and posterior parameter estimates.

## Module Layout

```
sampling/
├── __init__.py            # docstring + re-exports (backward-compatible)
├── _helpers.py            # shared layout/slicing utilities
├── _predictive.py         # variational & prior predictive sampling
├── _biological_ppc.py     # biological NB PPC (no capture/ZI)
├── _posterior_ppc.py      # full-model PPC (NB + ZINB + VCP + BNB + mixture)
├── _denoising.py          # Bayesian denoising core
└── _denoising_bnb.py      # BNB-specific quadrature helpers
```

### Internal Dependencies

```
_predictive.py        → _helpers.py
_biological_ppc.py    → _helpers.py
_posterior_ppc.py     → _helpers.py
_denoising.py         → _helpers.py, _denoising_bnb.py
```

All public and underscore names used by external code are re-exported from
`__init__.py`, so every existing `from scribe.sampling import X` continues
to work unchanged.

## Key Functions

| Function                            | Module            | Description                                           |
| ----------------------------------- | ----------------- | ----------------------------------------------------- |
| `sample_variational_posterior`      | `_predictive`     | Draw parameter samples from a trained guide           |
| `generate_predictive_samples`       | `_predictive`     | Predictive counts from posterior parameter draws      |
| `generate_ppc_samples`              | `_predictive`     | End-to-end PPC (sample params → generate counts)      |
| `generate_prior_predictive_samples` | `_predictive`     | Draw counts from the prior predictive                 |
| `sample_biological_nb`              | `_biological_ppc` | Biological NB PPC (strips technical noise)            |
| `sample_posterior_ppc`              | `_posterior_ppc`  | Full-model PPC (all noise components)                 |
| `denoise_counts`                    | `_denoising`      | Bayesian denoising of observed UMI counts             |
| `_build_canonical_layouts`          | `_helpers`        | Build `AxisLayout` dicts for canonical parameter keys |
| `_slice_posterior_draw`             | `_helpers`        | Extract a single posterior draw using layout metadata |
| `_slice_gene_axis`                  | `_helpers`        | Subset the gene dimension using layout metadata       |

## Parameterization Convention

Throughout this package the canonical `p` follows the **numpyro convention**:
it is the `probs` argument of `NegativeBinomialProbs`, i.e. the probability
of each Bernoulli trial producing a count.  The NB mean is therefore
`r * p / (1 - p)`.  This is the *complement* of the paper's p (which
appears as p^r in the PMF).

## Integration

- **SVI / MCMC results objects** call into this package for PPC, biological
  PPC, and denoising.  They always provide `param_layouts` (AxisLayout
  metadata), so the deprecated shape-heuristic fallback paths are never hit
  in normal use.
- **`core.axis_layout`** provides `AxisLayout`, `build_sample_layouts`, and
  `derive_axis_membership`, used by `_helpers.py` to build layouts from
  `ModelConfig`.
- **`models.components.likelihoods.beta_negative_binomial`** provides
  `build_count_dist`, used by PPC and denoising to construct the
  appropriate NB/BNB distribution objects.

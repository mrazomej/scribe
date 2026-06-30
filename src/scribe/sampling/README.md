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
3. **Bayesian denoising** — compute the posterior of the true (pre-capture,
   pre-dropout) transcript counts given observed UMI counts and posterior
   parameter estimates. The NB/BNB family uses closed-form Poisson–Gamma
   conjugacy; the Two-State (Poisson–Beta) family uses Gauss–Legendre
   quadrature over the latent promoter ON-fraction.

## Module Layout

```
sampling/
├── __init__.py              # docstring + re-exports (backward-compatible)
├── _helpers.py              # shared layout/slicing utilities
├── _predictive.py           # variational & prior predictive sampling
├── _biological_ppc.py       # biological NB PPC (no capture/ZI)
├── _posterior_ppc.py        # full-model PPC (NB + ZINB + VCP + BNB + mixture)
├── _denoising.py            # Bayesian denoising core (dispatch + NB closed-form)
├── _denoising_bnb.py        # BNB-specific quadrature helpers
└── _denoising_twostate.py   # Two-State (Poisson–Beta) quadrature helpers
```

### Internal Dependencies

```
_predictive.py           → _helpers.py
_biological_ppc.py       → _helpers.py
_posterior_ppc.py        → _helpers.py
_denoising.py            → _helpers.py, _denoising_bnb.py, _denoising_twostate.py
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
| `denoise_counts`                    | `_denoising`      | Bayesian denoising of observed UMI counts (NB + TwoState) |
| `_build_canonical_layouts`          | `_helpers`        | Build `AxisLayout` dicts for canonical parameter keys |
| `_slice_posterior_draw`             | `_helpers`        | Extract a single posterior draw using layout metadata |
| `_slice_gene_axis`                  | `_helpers`        | Subset the gene dimension using layout metadata       |

## Parameterization Convention

Throughout this package the canonical `p` follows the **numpyro convention**:
it is the `probs` argument of `NegativeBinomialProbs`, i.e. the probability
of each Bernoulli trial producing a count.  The NB mean is therefore
`r * p / (1 - p)`.  This is the *complement* of the paper's p (which
appears as p^r in the PMF).

## GPU Performance Optimizations

### PPC MAP Path — `vmap` over RNG Keys

The MAP (non-sample-dimension) path in both `sample_biological_nb` and
`sample_posterior_ppc` replaces a Python `for` loop over `n_samples` with
`jax.vmap`.  Since MAP parameters are constant across draws, only the PRNG key
varies per sample; the wrapper captures all other arguments and passes
`cell_batch_size=None` to let XLA fuse the full cell dimension into a single
kernel.

When `n_samples` is large enough to exceed GPU memory, the vmap call is chunked
using `_vmap_chunk_size` from `core._array_dispatch`, which queries available
device memory at runtime and splits the batch automatically.

### Denoising — `vmap` over Posterior Draws

`denoise_counts` uses the same vmap + adaptive-chunking pattern for its
multi-sample (posterior) path.  Each posterior draw's parameters are mapped via
`in_axes` based on `AxisLayout` metadata (with a runtime guard that verifies
axis-0 size matches `n_samples`, protecting against the deprecated
layout-inference path which can over-report sample dims). `cell_batch_size=None`
is passed to `_denoise_single` so XLA fuses the full cell dimension, while
memory is managed by chunking the sample axis.  `return_variance=True` is always
used inside the vmap closure to ensure a consistent return signature; the
variance array is discarded when the caller did not request it.

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
- **`models.components.likelihoods.two_state`** provides
  `_twostate_dispatch_reparam`, used by the SVI denoising mixin to
  convert posterior parameter samples into `(α, β, r̂)` for the
  Two-State quadrature denoiser.

## VAE Replay Modes and Parameter Binding

For VAE-backed models, replay must always receive the trained parameter
dictionary so `numpyro.param` / `flax_module` sites are substituted during
`Predictive` calls.

- `sample_variational_posterior` passes `params` to both the guide and
  model-replay `Predictive` steps.
- `generate_predictive_samples` accepts optional `params` and forwards
  them to `Predictive`.

Without this wiring, decoder parameters (e.g. `vae_decoder$params`) can
be re-initialized during replay, producing incorrect predictive samples.

`ScribeVAEResults.get_posterior_samples()` supports two replay modes:

1. **Encoder path (`counts` provided)**: sample from `q(z|counts)` via the
   VAE guide, then replay the model with substituted trained params.
2. **Prior path (`counts=None`)**: bypass the guide and sample `z` directly
   from the model prior while replaying with substituted trained params.
   - For LNM/LNMVCP this is the expected population-sampling behavior.
   - For VAEs with a latent flow prior, `z` is sampled from the learned
     flow prior (`FlowDistribution(flow, N(0, I))`).
   - For non-LNM VAEs without a flow prior, SCRIBE emits a warning that
     `z` is sampled from the uninformative `N(0, I)` prior.

## Context-aware site selection (`return_sites`)

`sample_variational_posterior(..., return_sites=...)` filters the **merged**
guide+model posterior dict down to a requested keep-set. This is a *post-merge*
filter: NumPyro still computes every deterministic site during the model replay
(its own `return_sites` filters post-trace), so this bounds the **stored**
posterior footprint that downstream consumers carry — not the transient
draw-time peak (use `batch_size` for that). It is applied at the merge layer
because the guide pass and the model replay are separate `Predictive` calls;
only the merged dict sees both guide sites (`*_raw`, `eta_capture`) and model
deterministics (`r`, `mu`, `p_capture`, `*_effect`, ...).

`return_sites` accepts a single site name (`str`) or an iterable. The shared
normalizer `_as_site_set` treats a bare string as one site name (`"mu"` →
`{"mu"}`, never `set("mu")` = `{"m","u"}`) and is the single source of truth
re-used by `scribe.svi._posterior_policy`. `None` (default) keeps every site,
so existing callers are unaffected.

# Likelihoods Submodule

This submodule provides likelihood components for count data models in SCRIBE.

## Module Structure

```
likelihoods/
├── __init__.py          # Re-exports all public classes
├── base.py              # Likelihood ABC + helper functions
├── negative_binomial.py # Standard NB likelihood
├── zero_inflated.py     # Zero-Inflated NB likelihood
├── vcp.py               # VCP variants (NB and ZINB with capture probability)
└── README.md            # This file
```

## Classes

| Class | File | Description |
|-------|------|-------------|
| `Likelihood` | base.py | Abstract base class for all likelihoods |
| `NegativeBinomialLikelihood` | negative_binomial.py | Standard Negative Binomial |
| `ZeroInflatedNBLikelihood` | zero_inflated.py | Zero-Inflated Negative Binomial |
| `NBWithVCPLikelihood` | vcp.py | NB with Variable Capture Probability |
| `ZINBWithVCPLikelihood` | vcp.py | ZINB with Variable Capture Probability |

## Plate Modes

All likelihoods handle three plate modes for different use cases:

| Mode | `counts` | `batch_size` | Use Case |
|------|----------|--------------|----------|
| Prior Predictive | `None` | - | Generate synthetic data |
| Full Sampling | provided | `None` | MCMC, small datasets |
| Batch Sampling | provided | specified | SVI on large datasets |

## VAE Path

When the model uses a **VAE** (encoder/decoder with `VAELatentGuide`), the model
builder passes a **`vae_cell_fn`** into each likelihood's `sample()` method. The
likelihood then:

1. Runs inside the cell plate (per-cell or per minibatch).
2. Calls `vae_cell_fn(batch_idx)` to obtain decoder-driven parameters (e.g. `r`,
   `gate`) and merges them into `param_values`.
3. Builds the observation distribution from the full `param_values` and samples
   or conditions on counts.

Decoder-driven parameters are **not** sample sites; they are produced by the
decoder from the latent `z`. Mixture detection in likelihoods uses
`"mixing_weights" in param_values` (not shape-based checks) so that VAE per-cell
decoder output is not mistaken for a mixture.

## Usage

```python
from scribe.models.components import NegativeBinomialLikelihood

# Create a likelihood
likelihood = NegativeBinomialLikelihood()

# Use in ModelBuilder
builder.with_likelihood(likelihood)
```

## Gene-Specific p Broadcasting

When using hierarchical parameterizations, `p` (or `phi`) becomes gene-specific
with shape `(n_genes,)` instead of a scalar. In mixture models this creates an
ambiguity: a 1D array could be `(n_components,)` or `(n_genes,)`. The helper
function `broadcast_param_for_mixture(p, r)` in `base.py` resolves this by checking
the shape against `r` and reshaping accordingly:

- Scalar `p` -> `(1, 1)` for broadcasting with `(n_components, n_genes)`
- 1D `p` matching `r.shape[-1]` (n_genes) -> `(1, n_genes)`
- 1D `p` matching `r.shape[0]` (n_components) -> `(n_components, 1)`
- 2D `p` -> passed through unchanged

All four likelihood classes use this helper for correct broadcasting when
building mixture distributions.

## Gene-Specific p Broadcasting

When using hierarchical parameterizations, `p` (or `phi`) becomes gene-specific
with shape `(n_genes,)` instead of a scalar. In mixture models this creates an
ambiguity: a 1D array could be `(n_components,)` or `(n_genes,)`. The helper
function `broadcast_param_for_mixture(p, r)` in `base.py` resolves this by checking
the shape against `r` and reshaping accordingly:

- Scalar `p` to `(1, 1)` for broadcasting with `(n_components, n_genes)`
- 1D `p` matching `r.shape[-1]` (n_genes) to `(1, n_genes)`
- 1D `p` matching `r.shape[0]` (n_components) to `(n_components, 1)`
- 2D `p` passed through unchanged

All four likelihood classes use this helper for correct broadcasting when
building mixture distributions.

## Per-Dataset Parameter Indexing

For multi-dataset models, per-dataset parameters (e.g. `r`, `p` with shape
`(n_datasets, ...)`) must be indexed to per-cell values. The helper
`index_dataset_params()` in `base.py` maps these parameters using
`dataset_indices` (shape `(batch,)`), mapping each cell to its dataset.

Because dataset-level indexing requires `dataset_indices`, dataset-level
hierarchical priors (`hierarchical_dataset_mu`, `hierarchical_dataset_p`,
`hierarchical_dataset_gate`) require running `scribe.fit(...)` with
`dataset_key` so each cell is assigned to a dataset. Without `dataset_key`,
SCRIBE now raises an early configuration error instead of silently treating
the model as single-dataset.

All likelihood `sample()` methods now accept an optional `dataset_indices`
parameter. When provided, per-dataset parameters are automatically indexed
to per-cell values inside the NumPyro plate context before building the
observation distribution.

For VCP likelihoods, 1D `p`/`phi` expansion to `(batch, 1)` is now guarded by
the capture vector length (`capture_value.shape[0]`). This keeps gene-specific
vectors `(n_genes,)` from being misinterpreted as per-cell vectors when
dataset indexing is enabled.

### Mixture + Dataset Axis Convention

When a parameter is **both** mixture-specific and dataset-specific, its shape
follows the layout produced by `resolve_shape`:

```
(n_components, n_datasets, base_dims...)
```

For example, `r` with `is_mixture=True, is_dataset=True` has shape
`(K, D, G)`. `index_dataset_params()` uses `ParamSpec` metadata
(`is_dataset`, `is_mixture`) to identify the correct axis:

- **Dataset-only** `(D, ...)`: index axis 0 -> `(batch, ...)`
- **Mixture + dataset** `(K, D, ...)`: index axis 1 -> transpose to
  `(batch, K, ...)` so that `MixtureSameFamily` sees the component
  dimension as the rightmost batch dimension.

The `param_specs` argument (passed from `model_config.param_specs`) enables
this spec-aware indexing. Without it, the function falls back to the legacy
heuristic (`shape[0] == n_datasets`).

When dataset-specific mixture weights are enabled, `mixing_weights` follow the
same indexing rule and become `(batch, K)` inside the cell plate. Annotation
logit nudging supports both global `(K,)` and batch-aligned `(batch, K)` inputs.

`broadcast_param_for_mixture()` also handles the extra batch dimension: when `p`
is 2-D `(batch, G)` and `r` is 3-D `(batch, K, G)`, it inserts a component
singleton to produce `(batch, 1, G)` for correct broadcasting.

### Dataset-Level Hierarchical Gate

When `hierarchical_dataset_gate=True`, the gate parameter (zero-inflation
probability) becomes per-dataset and gene-specific via a
`DatasetHierarchicalSigmoidNormalSpec`. The gate follows the same indexing
convention as `p`/`phi`: `index_dataset_params()` detects `is_dataset=True`
on the gate's `ParamSpec` and indexes it per cell. No changes to the
likelihood code are needed — the existing `_build_dist` methods receive
already-indexed gate values.

## VCP Likelihoods

The Variable Capture Probability (VCP) likelihoods model cell-specific technical
variation in capture efficiency. They support two parameterizations:

- **Canonical/mean-prob**: Uses `p_capture` (Beta distribution)
- **Mean-odds**: Uses `phi_capture` (BetaPrime distribution)

```python
# Explicit parameterization (recommended)
likelihood = NBWithVCPLikelihood(capture_param_name="phi_capture")

# With unconstrained sampling for better optimization
# The transform for phi_capture should align with ModelConfig.positive_transform
# (softplus by default; exp for legacy/backward-compatible configs).
likelihood = NBWithVCPLikelihood(
    capture_param_name="phi_capture",
    is_unconstrained=True,
    transform=SoftplusTransform(),
    constrained_name="phi_capture",
)
```

### Biology-Informed and Data-Driven Capture Priors

VCP likelihoods accept an optional `biology_informed_spec`
(`BiologyInformedCaptureSpec`) that replaces the flat capture prior with a
library-size-anchored TruncatedNormal (low=0) prior on the latent variable
`eta_c = log(M_c / L_c)`. The truncation enforces the physical constraint
`M_c >= L_c` (a cell cannot emit more molecules than it contains). The
biology-informed path activates automatically when `priors.organism`,
`priors.eta_capture`, or `priors.mu_eta` is set:

- **No capture priors + `mu_eta_prior="none"`** (or omitted): Standard flat
  prior (no eta framework).
- **`priors.eta_capture` set + `mu_eta_prior="none"`**: Fixed M_0, no shared
  parameter.
- **`priors.eta_capture` set + `mu_eta_prior` in {"gaussian", "horseshoe",
  "neg"}**: Learn **per-dataset** `mu_eta` via a non-centered hierarchical
  prior.  A population mean `mu_eta_pop ~ N(log_M0, sigma_mu)` is shared
  across all datasets, while per-dataset deviations are shrunk toward zero
  by the chosen prior (Gaussian scale, regularized Horseshoe, or NEG).
  `priors.mu_eta` controls `[center, sigma_mu]`.

When the eta framework is active:

1. **Pre-plate**: Log library sizes are computed from `counts` (or synthetic
   values during dry runs). When `mu_eta_prior` is not "none" and
   `n_datasets >= 2`, per-dataset `mu_eta` (shape `(D,)`) is sampled by
   `_sample_hierarchical_mu_eta()` in `base.py`.  For single-dataset
   fallback, a scalar `mu_eta` is sampled instead.
2. **Inside plate**: `eta_c ~ N(log_M0 - log_L_c, sigma_M^2)` (or centered on
   the per-dataset `mu_eta[d]` when hierarchical) is sampled via
   `_sample_capture_biology_informed()`, then transformed to `p_capture`
   or `phi_capture` via exact formulas.

**Guide parameterization** (`ModelConfig.eta_capture_guide`):

- `"softplus_normal"` (default): The guide samples an unconstrained Normal
  (`eta_capture_raw_loc/scale`) and maps through softplus to produce
  `eta_capture`. This induces a logit-normal on `nu_c` with smooth gradients
  and no truncation boundary.
- `"truncated_normal"` (legacy): The guide uses `TruncatedNormal(low=0)` for
  `eta_capture` directly (`eta_capture_loc/scale`). Preserved for backward
  compatibility with old checkpoints.

Key helpers in `base.py`:
- `_sample_hierarchical_mu_eta()` — dispatcher to prior-specific samplers
- `_sample_hierarchical_mu_eta_{gaussian,horseshoe,neg}()` — NCP samplers
- `_sample_capture_biology_informed()` — per-cell eta sampling and transform

## Annotation Priors for Mixture Models

Likelihoods support **cell-specific annotation priors** that modify the global
mixing weights on a per-cell basis. When the caller provides an
`annotation_prior_logits` array of shape `(n_cells, n_components)`, the
likelihood computes cell-specific mixing weights inside the cell plate:

```
pi_i = softmax(log(mixing_weights) + annotation_prior_logits[i])
```

This is implemented by the helper function `compute_cell_specific_mixing` in
`base.py` and used by all four likelihood classes.

### Mathematical derivation

Standard mixture model (all cells share global weights):

```
mixing_weights ~ Dirichlet(alpha)
x_i ~ MixtureSameFamily(Categorical(mixing_weights), F_k)
```

With annotation priors (cell-specific mixing):

```
mixing_weights ~ Dirichlet(alpha)                              # global, learned
pi_i = softmax(log(mixing_weights) + kappa * one_hot(ann_i))   # per-cell
x_i ~ MixtureSameFamily(Categorical(pi_i), F_k)
```

The posterior assignment `p(z_i = k | x_i)` is proportional to
`pi_{i,k} * f_k(x_i | theta_k)` -- the annotation is the prior, the data
likelihood is the update.

### Interaction with the three plate modes

| Mode | Annotation handling |
|------|---------------------|
| Prior Predictive (`counts=None`) | Full `annotation_prior_logits` used |
| Full Sampling (`batch_size=None`) | Full `annotation_prior_logits` used |
| Batch Sampling (`batch_size` set) | Indexed `annotation_prior_logits[idx]` |

When `annotation_prior_logits` is `None`, the standard behaviour is preserved
exactly (distribution built once outside the plate for the non-VAE path).

### Code example

```python
import scribe

# Via the high-level API -- single annotation column
result = scribe.fit(
    adata,
    model="nbdm",
    n_components=3,
    annotation_key="cell_type",       # column in adata.obs
    annotation_confidence=3.0,        # kappa
    annotation_component_order=["T", "B", "Mono"],  # optional explicit order
)

# Multiple annotation columns (composite labels: "T__ctrl", "B__stim", ...)
result = scribe.fit(
    adata,
    model="nbdm",
    n_components=6,
    annotation_key=["cell_type", "treatment"],  # forms all observed pairs
    annotation_confidence=3.0,
)

# Or via pre-built logits with run_scribe
from scribe.core import build_annotation_prior_logits
logits, label_map = build_annotation_prior_logits(
    adata, "cell_type", n_components=3, confidence=3.0
)
result = scribe.inference.run_scribe(
    adata, model_config=mc, inference_config=ic,
    annotation_prior_logits=logits,
)
```

## Numerical Stability Guards

All likelihood components clamp `p`, `phi`, and `p_hat` away from degenerate
values before constructing NB distributions.  This prevents NaN in the ELBO
during SVI training when hierarchical priors produce extreme samples (e.g.
`phi_g -> 0` causing `p_g -> 1.0`, or `phi_g -> inf` causing `p_g -> 0.0`).

A module-level constant `_P_EPS = 1e-6` is defined in both
`negative_binomial.py` and `vcp.py`.  The guards applied are:

| Likelihood | Parameter | Guard |
|------------|-----------|-------|
| `NegativeBinomialLikelihood._build_dist` | `p` | `jnp.clip(p, _P_EPS, 1 - _P_EPS)` |
| `NegativeBinomialLikelihood._build_annotated_mixture_dist` | `p` | `jnp.clip(p, _P_EPS, 1 - _P_EPS)` |
| `NBWithVCPLikelihood` (mean-odds path) | `phi` | `jnp.maximum(phi, _P_EPS)` before `log(phi * ...)` |
| `NBWithVCPLikelihood` (mean-prob path) | `p_hat` | `jnp.clip(p_hat, _P_EPS, 1 - _P_EPS)` |
| `ZINBWithVCPLikelihood` (mean-odds path) | `phi` | `jnp.maximum(phi, _P_EPS)` before `log(phi * ...)` |
| `ZINBWithVCPLikelihood` (mean-prob path) | `p_hat` | `jnp.clip(p_hat, _P_EPS, 1 - _P_EPS)` |

This mirrors the `p_floor` parameter already used in the post-hoc
log-likelihood evaluation functions in `log_likelihood.py`.  Tests for both
layers live in `tests/test_floor.py`.

## Adding New Likelihoods

To add a new likelihood:

1. Create a new file or add to an existing one
2. Inherit from `Likelihood` (in `base.py`)
3. Implement the `sample` method with signature including
   `vae_cell_fn: Optional[Callable] = None` and
   `annotation_prior_logits: Optional[jnp.ndarray] = None`, handling:
   - All three plate modes (prior predictive, full, batch)
   - Non-VAE path: `vae_cell_fn is None` -- build distribution once from
     `param_values`
   - VAE path: `vae_cell_fn` provided -- call it inside the cell plate, merge
     into `param_values`, then build distribution
   - Annotation path: if `annotation_prior_logits is not None` and this is a
     mixture, use `compute_cell_specific_mixing` inside the cell plate
4. Export the class in `__init__.py`

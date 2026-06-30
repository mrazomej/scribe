# Inference Methods

SCRIBE supports four inference backends that all share the same `scribe.fit()`
entry point. Choose the one that best fits your goals and computational budget.

## Choosing an Inference Method

| Criterion             | SVI                        | MCMC                      | VAE                        | Laplace                             |
| --------------------- | -------------------------- | ------------------------- | -------------------------- | ----------------------------------- |
| **Speed**             | Fast (minutes)             | Slow (hours)              | Moderate (tens of minutes) | Moderate (tens of minutes)          |
| **Scalability**       | Excellent (mini-batching)  | Limited (full data)       | Excellent (mini-batching)  | Good (mini-batching)                |
| **Posterior quality** | Approximate                | Exact                     | Approximate (neural)       | Approximate (Hessian)               |
| **Latent embeddings** | No                         | No                        | Yes                        | No                                  |
| **Models supported**  | All                        | NB-family                 | All                        | PLN, NBLN, LNM, LNMVCP              |
| **Best for**          | Exploration and production | Gold-standard uncertainty | Representation learning    | Correlation recovery, rigorous PPCs |

!!! tip "Default recommendation"
    Start with **SVI** for NB-family models. For PLN/NBLN/LNM/LNMVCP models,
    use **Laplace** --- it avoids encoder collapse, produces rigorous per-cell
    posteriors from the Hessian, and has no aggregate-posterior drift. Switch to
    **MCMC** when you need exact posteriors for a publication, or use **VAE**
    when you need amortized scoring of new cells or low-dimensional embeddings.

    For NBLN specifically, the recommended pipeline is **SVI-cascade +
    freeze + loadings shrinkage** — see the
    [NBLN cascade + freeze + shrinkage workflow](#nbln-cascade--freeze--shrinkage-workflow)
    section below.

---

## Stochastic Variational Inference (SVI)

SVI finds the best approximation to the posterior within a chosen variational
family using stochastic optimization. It is the default and most commonly used
inference method.

### Basic usage

```python
import scribe

# Default SVI inference (NBVCP model)
results = scribe.fit(adata)

# With custom parameters
results = scribe.fit(
    adata,
    zero_inflation=True,
    n_steps=100_000,
    batch_size=512,
    seed=0,
)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | 50,000 | Maximum optimization steps |
| `batch_size` | `None` (full batch) | Mini-batch size for stochastic optimization |
| `optimizer_config` | `None` | Custom optimizer specification (see below) |
| `stable_update` | `True` | Numerically stable parameter updates |
| `restore_best` | `False` | Track and restore the best variational parameters during training |
| `early_stopping` | `None` | Automatic convergence detection (see below) |
| `seed` | 42 | Random seed for reproducibility |

### Custom optimizer

By default SCRIBE uses Adam. Pass an `optimizer_config` dict to change the
optimizer or its learning rate:

```python
results = scribe.fit(
    adata,
    optimizer_config={"name": "clipped_adam", "step_size": 5e-4},
)
```

Supported optimizers: `"adam"`, `"clipped_adam"`, `"adagrad"`, `"rmsprop"`,
`"sgd"`, `"momentum"`.

### Best-params restoration

The `restore_best` flag tracks the lowest smoothed loss during training and
restores those parameters at the end, regardless of whether early stopping
is configured. This is especially useful for normalizing flow guides, where
the ELBO can fluctuate late in training:

```python
results = scribe.fit(
    adata,
    unconstrained=True,
    guide_flow="affine_coupling",
    restore_best=True,
    n_steps=100_000,
)
```

### Guide families

The variational guide controls the flexibility of the posterior approximation.
SCRIBE supports several families---mean-field (default), low-rank,
joint low-rank, normalizing flow, amortized, and VAE latent---each offering
different trade-offs between speed and the ability to capture correlations:

```python
# Low-rank guide for gene correlations
results = scribe.fit(adata, guide_rank=8)

# Joint low-rank across parameter groups
results = scribe.fit(
    adata, guide_rank=8, joint_params="biological",
)

# Normalizing flow guide for non-Gaussian posteriors
results = scribe.fit(
    adata, unconstrained=True,
    guide_flow="affine_coupling",
)

# Amortized capture for VCP models
results = scribe.fit(adata, variable_capture=True, amortize_capture=True)
```

**Full guide:** [Variational guide families](guide-families.md)

### Early stopping

SVI supports automatic convergence detection to avoid wasting computation:

```python
results = scribe.fit(
    adata,
    n_steps=200_000,
    early_stopping={
        "patience": 500,
        "min_delta": 1.0,
        "smoothing_window": 50,
        "restore_best": True,
    },
)
```

| Early stopping parameter | Default | Description |
|--------------------------|---------|-------------|
| `patience` | 500 | Steps without improvement before stopping |
| `min_delta` | 1.0 | Minimum loss improvement to count as progress |
| `smoothing_window` | 50 | Window size for moving-average loss |
| `restore_best` | `True` | Restore parameters from the best checkpoint |

### Results

`scribe.fit()` returns a `ScribeSVIResults` object. See the
[Results Class](results.md) page for the full API, including posterior
sampling, predictive checks, denoising, and normalization.

---

## Markov Chain Monte Carlo (MCMC)

MCMC generates samples from the true posterior distribution using the
No-U-Turn Sampler (NUTS). It provides the most accurate uncertainty
quantification but is slower than SVI.

### Basic usage

```python
import scribe

results = scribe.fit(
    adata,
    inference_method="mcmc",
    n_samples=2_000,
    n_warmup=1_000,
    n_chains=4,
)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_method` | `"svi"` | Set to `"mcmc"` for MCMC inference |
| `n_samples` | 2,000 | Posterior samples per chain |
| `n_warmup` | 1,000 | Warmup (burn-in) samples |
| `n_chains` | 1 | Number of parallel chains |

!!! note "Float64 precision"
    MCMC defaults to 64-bit floating point for numerical stability during
    Hamiltonian dynamics. This doubles memory usage compared to SVI but is
    important for reliable sampling.

### Warm-starting from SVI

A common workflow is to run SVI first for exploration, then refine with MCMC
using the SVI result as initialization. This dramatically reduces warmup time:

```python
import scribe

# Step 1: fast SVI exploration
svi_results = scribe.fit(adata, n_steps=50_000)

# Step 2: refine with MCMC, initialized from SVI
mcmc_results = scribe.fit(
    adata,
    inference_method="mcmc",
    svi_init=svi_results,
    n_samples=2_000,
    n_warmup=500,
)
```

The `svi_init` parameter handles cross-parameterization mapping automatically
-- you can initialize MCMC from an SVI result that used a different
parameterization.

### Results

MCMC returns a `ScribeMCMCResults` object with the same analysis API as SVI
results (posterior sampling, predictive checks, denoising, etc.), plus
MCMC-specific diagnostics:

```python
# NUTS diagnostics
results.print_summary()

# Chain-grouped samples for convergence analysis
chain_samples = results.get_samples(group_by_chain=True)
```

---

## Variational Autoencoder (VAE)

The VAE backend uses neural networks (Flax NNX) for amortized variational
inference. It learns a low-dimensional latent representation of each cell while
simultaneously fitting the SCRIBE probabilistic model.

### Basic usage

```python
import scribe

results = scribe.fit(
    adata,
    inference_method="vae",
    vae_latent_dim=10,
    n_steps=100_000,
    batch_size=256,
)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_method` | `"svi"` | Set to `"vae"` for VAE inference |
| `vae_latent_dim` | 10 | Dimensionality of the latent space |
| `vae_encoder_hidden_dims` | `None` | Encoder hidden layer sizes (e.g., `[512, 256]`) |
| `vae_decoder_hidden_dims` | `None` | Decoder hidden layer sizes |
| `vae_activation` | `None` | Activation function (`"relu"`, `"gelu"`, `"silu"`, etc.) |
| `vae_input_transform` | `"log1p"` | Input preprocessing (`"log1p"`, `"log"`, `"sqrt"`, `"identity"`) |

### VAE variants

**Standard VAE** -- single encoder-decoder pair with a standard normal prior.

**Decoupled Prior VAE (dpVAE)** -- separate priors for different parameter
groups, enabling more flexible modeling of parameter relationships.

### Normalizing flow priors

For more expressive latent distributions, attach a normalizing flow to the
VAE prior:

```python
results = scribe.fit(
    adata,
    inference_method="vae",
    vae_latent_dim=10,
    vae_flow_type="spline_coupling",
    vae_flow_num_layers=4,
    vae_flow_hidden_dims=[64, 64],
)
```

Available flow types: `"affine_coupling"` (fast baseline),
`"spline_coupling"` (expressive, recommended for production),
`"maf"` (fast density), `"iaf"` (fast sampling).

### Latent space analysis

VAE results provide cell embeddings that can be used for visualization and
clustering:

```python
# Cell embeddings in latent space
embeddings = results.get_latent_embeddings(data=adata.X, n_samples=100)

# Conditional posterior samples
latent_samples = results.get_latent_samples_conditioned_on_data(
    data=adata.X, n_samples=500,
)
```

---

## Laplace Approximation

The Laplace inference path finds each cell's MAP (maximum a posteriori)
latent via Newton iteration, then approximates the per-cell posterior as a
Gaussian centered at the MAP with covariance equal to the negative inverse
Hessian. The outer loop optimizes global parameters (decoder weights
\(\mu\), \(W\), \(d\)) via Adam on the Laplace-approximated ELBO. There is
**no encoder network**---each cell's posterior is computed locally.

### Basic usage

```python
import scribe

# PLN with Laplace inference
results = scribe.fit(
    adata,
    model="pln",
    inference_method="laplace",
    latent_dim=16,
    n_steps=50_000,
    batch_size=256,
)

# NBLN with Laplace inference (cascade-frozen workflow below)
results = scribe.fit(
    adata,
    model="nbln",
    inference_method="laplace",
    latent_dim=16,
    n_steps=50_000,
)

# LNMVCP with Laplace inference
results = scribe.fit(
    adata,
    model="lnmvcp",
    inference_method="laplace",
    latent_dim=16,
    n_steps=50_000,
)
```

!!! info "`latent_dim` vs `vae_latent_dim`"
    The `latent_dim` kwarg is the preferred name for the rank of the
    low-rank loadings matrix \(\underline{\underline{W}} \in
    \mathbb{R}^{G \times k}\). For backward compatibility, the legacy
    `vae_latent_dim` kwarg is still accepted (it was the original
    name when Laplace inference didn't yet exist). Passing both
    raises `ValueError`.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_method` | `"svi"` | Set to `"laplace"` for Laplace inference |
| `model` | `"nbvcp"` | Must be `"pln"`, `"nbln"`, `"lnm"`, or `"lnmvcp"` for Laplace |
| `n_steps` | `50_000` | Outer optimization steps |
| `batch_size` | `None` | Mini-batch size for stochastic gradient estimation |
| `latent_dim` | `None` | Rank \(k\) of the low-rank covariance \(\Sigma = WW^\top + \text{diag}(d)\). Legacy alias: `vae_latent_dim`. |
| `laplace_config` | `None` | Dict of Newton solver settings (see below) |
| `informative_priors_from` | `None` | Cascade source for NBLN (Phase-1 soft cascade) — see [NBLN workflow](#nbln-cascade--freeze--shrinkage-workflow) below |
| `informative_priors_freeze` | `("r", "eta")` | Cascade freeze parameters for NBLN (Phase-2). Accepts either internal short names (`"r"`, `"mu"`, `"eta"`) or their descriptive aliases (`"dispersion"`, `"expression"`/`"mean_expression"`, `"capture_efficiency"`). Both forms work identically. |
| `priors={"loadings": ...}` | `None` | Loadings-matrix shrinkage strategy spec for PLN/NBLN (Phase-3) |
| `correlation_hierarchy` | `None` | `"program_scales"` for grouped fits: share \(W\) across datasets, learn per-dataset activity \(s_d\). Requires `dataset_key`/`hierarchy` with \(\ge 2\) datasets and (v1) the legacy layout `correlate_other_column=True`. See [hierarchical correlation](#hierarchical-gene-gene-correlation-across-datasets) below |

### Laplace configuration

Fine-tune the inner Newton solver via `laplace_config`:

```python
results = scribe.fit(
    adata,
    model="lnmvcp",
    inference_method="laplace",
    laplace_config={
        "n_newton_steps": 15,       # more iterations for hard cells
        "damping": 1e-3,            # tighter Tikhonov regularization
        "newton_tolerance": 1e-3,   # relax for production fits
        "convergence_action": "warn",
    },
)
```

| Config key | Default | Description |
|------------|---------|-------------|
| `n_newton_steps` | `5` | Newton iterations per cell per outer step |
| `damping` | `1e-2` | Tikhonov regularization added to Hessian diagonal |
| `newton_tolerance` | `1e-4` | Gradient norm threshold for declaring convergence |
| `convergence_action` | `"warn"` | Action when cells don't converge: `"warn"`, `"raise"`, or `"ignore"` |

### How it works

The training loop alternates two steps per outer iteration:

1. **Inner Newton** on per-cell latents (holding globals fixed). For PLN:
   joint \((\underline{x}, \eta)\) Newton via Schur-complement
   back-substitution. For LNM/LNMVCP: composition Newton
   (\(\underline{z}\) or \(\underline{y}_\text{ALR}\)) plus scalar
   \(\eta\) Newton.

2. **Outer Adam step** on global parameters \((\mu, W, d)\) using the
   gradient of the Laplace ELBO with MAPs treated as `stop_gradient`
   constants.

Each Newton step costs \(O(Gk + k^3)\) per cell using nested Woodbury
identities on the low-rank covariance --- no \(G \times G\) matrices are
ever formed.

### When to use Laplace

| Use Laplace when... | Use SVI/VAE when... |
|---------------------|---------------------|
| You need rigorous per-cell posteriors from the Hessian | You need amortized inference for new cells |
| You suspect the encoder is collapsing on a per-cell latent | The encoder is well-calibrated |
| You want no aggregate-posterior drift | You need fast serving-time scoring |
| Your data has high cell-to-cell variability | The dataset is small enough that encoder collapse isn't a concern |

### Progress-bar diagnostics

During training, the progress bar reports per-cell Newton convergence:

```text
LNM Laplace (learned + capture):  21%|██  |
  init loss: -8.857e+07,
  avg. loss [10001-10500]: -8.896e+07,
  comp max/p99/med 1.38e+01/3.42e+00/2.51e-03;
  η    max/p99/med 1.79e-06/1.61e-06/4.92e-07
```

The `comp` and `η` lines show per-cell Newton gradient norms (max, 99th
percentile, median) for the composition and capture blocks respectively.

**Healthy fit:** `median` well below tolerance, `max` trending down.

**Problem cells:** `median` small but `max` large and bouncing --- a few
pathological cells (typically low-count) are slow to converge but don't
affect the bulk fit.

### Divergence handling

The engine has three layered defenses against single-cell explosive
divergence:

1. **Sherman--Morrison denominator floor** --- prevents catastrophic float32
   cancellation in the `y_alr` Newton step.
2. **Per-cell NaN/Inf mask** --- divergent cells are masked from the current
   step's gradient on globals.
3. **Outer-loop divergence detector** --- clean abort with diagnostic context
   if loss becomes NaN or grows by > 1000× from init.

If a divergence abort fires, typical remedies are:

- Increase `n_newton_steps` to 20--30
- Tighten `damping` to 1e-3 or below
- Pre-filter outlier cells (very low \(u_T\) or extreme compositional skew)

### NBLN cascade + freeze + shrinkage workflow

[NBLN](../theory/nb-lognormal.md) has a per-cell rigid-translation
gauge (\(C\) degrees of freedom, one per cell) that needs structural
pinning to produce a well-identified fit. SCRIBE addresses this with
a three-phase pipeline:

```python
import scribe, numpy as np

# Phase 1: SVI cascade source (NBVCP-SVI on the same data)
svi_results = scribe.fit(
    adata, model="nbvcp", parameterization="mean_odds",
    priors={"capture_efficiency": (np.log(100_000), 0.5)},
    inference_method="svi", n_steps=50_000,
)

# Phases 2+3: NBLN-Laplace with cascade freeze + loadings shrinkage
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    # Phase 1: pass the cascade source. Empirical Gaussian priors on
    # r, mu, eta from the SVI posterior are derived and injected as
    # soft priors in the Laplace loss.
    informative_priors_from=svi_results,
    informative_priors_tau=1.0,
    # Phase 2: freeze r and eta at the cascade MAPs. Pins the per-cell
    # rigid-translation gauge structurally. Frozen params route
    # through cascade_source for PPC and distributions to preserve
    # full SVI guide fidelity.
    informative_priors_freeze=("r", "eta"),       # default
    # Phase 3: loadings shrinkage. Lets latent_dim be generous and
    # picks the effective rank adaptively. See the loadings-shrinkage
    # theory page for the strategy catalog and calibration workflow.
    priors={
        "capture_efficiency": (np.log(100_000), 0.5),
        "loadings": {
            "type": "horseshoe_columnwise",
            "tau_scale": 1.0,
        },
    },
    latent_dim=16,
    n_steps=20_000,
)

# Inspect effective rank + correlation structure
print(laplace_results.w_prior_diagnostics["effective_rank"])  # adaptive rank
diag = laplace_results.get_gauge_diagnostics()
print(diag["gauge_contamination_ratio"])                      # should be < 0.05

# Gauge-invariant loadings for cross-gene correlation analysis
W_perp = laplace_results.get_W_compositional()
```

| Phase | Mechanism | Default |
|---|---|---|
| **1. Soft cascade** | SVI posterior → empirical Gaussian priors → Laplace loss | Activated by `informative_priors_from=` |
| **2. Hard freeze** | Selected cascade-derived params pinned at MAP, excluded from optimizer | `("r", "eta")` |
| **3. Loadings shrinkage** | Adaptive rank selection on the columns of \(W_\perp\) | None (opt-in) |

The three phases are **orthogonal** — you can use any subset. The
combined recipe above is the recommended production workflow for
NBLN fits.

**Theory:** [NB Log-Normal Model](../theory/nb-lognormal.md), [Loadings-Matrix Shrinkage Priors](../theory/loadings-shrinkage.md)

### Hierarchical gene-gene correlation across datasets

For a **grouped** fit (donors / conditions / batches), the correlation
models can share the low-rank regulatory programs \(W\) across datasets
while learning a per-dataset *relative* program activity \(s_d\), inducing
\(\Sigma_d = W\,\text{diag}(s_d^2)\,W^\top + \text{diag}(d)\). Turn it on
with `correlation_hierarchy="program_scales"` and a grouping:

```python
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    correlation_hierarchy="program_scales",     # shared W, per-donor s_d
    dataset_key="donor",                         # >= 2 datasets (legacy or decoupled)
    informative_priors_from=svi_results,         # cascade composes with the hierarchy
    informative_priors_freeze=("r",),            # pool dispersion across donors
    latent_dim=16, n_steps=20_000,
)
s   = laplace_results.get_program_activity()   # (D, K) relative activities s_d
tau = laplace_results.program_scale_tau        # scalar between-dataset scale τ_s
```

This is orthogonal to the three cascade phases above and **composes** with
them — the freeze pins the marginals (\(r\), \(\eta\)) from an upstream fit
while the \(s_d\) hierarchy learns the per-dataset correlation on top. The
same flag is available on the SVI/VAE path as a fast structure check. See
[Theory: Hierarchical gene-gene correlation across
datasets](../theory/nb-lognormal.md#hierarchical-gene-gene-correlation-across-datasets)
for the model, the effective-loadings collapse, and identifiability.

#### Per-donor marginal cascade (freezing \(\mu^{(d)}\))

When the cascade source is itself a **hierarchical** (multi-dataset)
independent-gene fit, you can additionally freeze a *per-donor* mean
\(\mu^{(d)}\): add `"mu"` to `informative_priors_freeze`. Each cell then uses
its donor's prior mean \(\mu^{(\sigma(c))}\) (a per-cell gather, the same
mechanism as the program scales), so the correlation hierarchy and the
marginal hierarchy compose — dataset-specific *expression levels* from the
SVI source plus dataset-specific *correlation* from \(s_d\), over a shared
\(W\):

```python
# Step 1: hierarchical independent-gene SVI source (per-donor mean)
svi_hier = scribe.fit(
    adata, model="nbvcp", unconstrained=True, dataset_key="donor",
    priors={"mean_expression": {"donor": "gaussian"}},
    inference_method="svi", n_steps=50_000,
)

# Step 2: NBLN-Laplace freezing per-donor mu^(d) + pooled r
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    correlation_hierarchy="program_scales",
    correlate_other_column=True,
    dataset_key="donor",
    informative_priors_from=svi_hier,
    informative_priors_freeze=("r", "mu"),   # pool r, freeze per-donor mu^(d)
    latent_dim=16, n_steps=20_000,
)
mu_d = laplace_results.get_gene_mean_per_dataset()  # (D, G) per-donor means
mu   = laplace_results.get_mu()                     # (G,) donor-pooled mean
```

The extractor reads the source's per-donor `get_map()["mu"]` (a hierarchical
fit already exposes it as `(D, G)`), **aligns the source leaves to the target
leaf ordering by label**, and log-transforms to the NBLN log-rate; the
per-donor dispersion is pooled to a shared \(r\). Per-donor \(\mu^{(d)}\) is
supported only alongside `correlation_hierarchy="program_scales"` (the
per-cell-\(W\) Newton path); freezing it without the hierarchy raises a clear
error. `get_gene_mean_per_dataset()` returns the `(D, G)` table; `get_mu()`
stays the donor-pooled `(G,)`.

### Results

`scribe.fit()` with `inference_method="laplace"` returns a
`ScribeLaplaceResults` object. See the [Results Class](results.md) page for
the full API, including MAP-only and Laplace-uncertainty posterior predictive
checks.

---

## Combining Inference Methods

### SVI then MCMC

The most common multi-method workflow is SVI for fast exploration followed by
MCMC for publication-quality posteriors:

```mermaid
flowchart LR
    A["SVI (fast)"] -->|"svi_init="| B["MCMC (exact)"]
    A --> C["Explore results"]
    B --> D["Final analysis"]
```

### SVI then DE / Model Comparison

SVI results feed directly into downstream analyses:

```mermaid
flowchart LR
    A["scribe.fit()"] --> B["Differential Expression"]
    A --> C["Model Comparison"]
    A --> D["Posterior Predictive Checks"]
    A --> E["Bayesian Denoising"]
```

See the [Differential Expression](differential-expression.md) and
[Model Comparison](model-comparison.md) guides for details on these
downstream analyses.

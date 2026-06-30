# The `scribe.fit()` Interface

`scribe.fit()` is the **single entry point** for all SCRIBE inference. Every
model, parameterization, prior, inference engine, and guide family is
configured through keyword arguments to this one function. This page walks
through every parameter group, explains when and why each matters, and links
to the deeper guides and theory pages for full details.

```python
import scribe

# Sensible defaults --- variable capture is on by default
results = scribe.fit(adata)

# Add a low-rank guide for gene-gene correlations
results = scribe.fit(adata, guide_rank=64)
```

!!! tip "Read order"
    If you are new to SCRIBE, read sections 1--4 below and the
    [Model Selection](model-selection.md) page. The remaining sections cover
    progressively more advanced features that you can explore as needed.

!!! info "Naming convention"
    Priors use **canonical, parameterization-independent parameter names**
    (`mean_expression`, `dispersion`, `probability`, `odds_ratio`,
    `zero_inflation`, ...) rather than single-letter math notation, and are all
    declared through the single `priors` dict --- there are no separate
    `*_prior` keyword arguments. See [Defining Priors](priors.md) for the full
    name table and routing grammar.

**Variable capture is on by default.** The default model is `"nbvcp"`, which
includes cell-specific capture probability. Use **`variable_capture=False`**
to disable it, or **`zero_inflation=True`** to add a zero-inflation gate.
The **`model`** keyword still accepts `"nbdm"`, `"nbvcp"`, `"zinb"`, and
`"zinbvcp"` for the same four NB-family combinations, plus `"lnm"`,
`"lnmvcp"`, `"pln"`, `"nbln"`, `"twostate"`, and `"twostatevcp"` for the
log-normal and Poisson-Beta families. See [Model selection](#2-model-selection)
for the full resolution table.

---

## 1. Data input

These parameters control what data SCRIBE reads and how it interprets the
count matrix.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `counts` | *(required)* | Count matrix (`jnp.ndarray`) or `AnnData` object. Shape is `(n_cells, n_genes)` when `cells_axis=0` |
| `cells_axis` | `0` | Which axis represents cells. `0` = rows are cells (the standard layout) |
| `layer` | `None` | AnnData layer to use for counts. `None` uses `.X` |
| `seed` | `42` | Random seed for reproducibility |

```python
# From a raw JAX/NumPy array
results = scribe.fit(counts_array, model="nbdm")

# From AnnData, reading a specific layer
results = scribe.fit(adata, layer="raw_counts")
```

!!! note
    When you need AnnData-specific features---annotation priors, multi-dataset
    keys, or layer selection---`counts` **must** be an `AnnData` object.

---

## 2. Model selection

All four likelihoods share the same Negative Binomial core. The default
includes **variable capture** (`model="nbvcp"`), which models cell-specific
library-size variation---the right choice for the vast majority of scRNA-seq
datasets. Use **`variable_capture`** and **`zero_inflation`** to compose the
model explicitly, or set **`model`** to a single string. If you pass both
flags and `model=`, they must agree or SCRIBE raises an error.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `variable_capture` | `None` | `True` adds cell-specific capture probability. `False` removes it. `None` defers to `model` |
| `zero_inflation` | `None` | `True` adds a per-gene zero-inflation gate. `None` defers to `model` |
| `model` | `"nbvcp"` | Likelihood short name. Default includes variable capture. Flags override this when set |

| What you pass | Same as `model=` |
|---------------|------------------|
| Default (nothing) | `"nbvcp"` |
| `variable_capture=False` | `"nbdm"` |
| `zero_inflation=True` | `"zinbvcp"` |
| `variable_capture=False, zero_inflation=True` | `"zinb"` |
| `model="twostate"` | TwoState (no capture) |
| `model="twostate", variable_capture=True` | `"twostatevcp"` |
| `model="lnm"` / `model="lnmvcp"` | LNM family |
| `model="pln"` / `model="nbln"` | Log-normal family |

!!! note "TwoState and log-normal families"
    The `variable_capture` flag is also recognized for `model="twostate"`
    (resolves to `"twostatevcp"`).  For `model="lnm"`, `"pln"`, `"nbln"`,
    use the explicit model string — the `variable_capture` flag is not
    used for these families.

```python
# Default: variable capture is already on
results = scribe.fit(adata)

# Disable variable capture (plain NB) --- only when library sizes are very tight
results = scribe.fit(adata, variable_capture=False)

# Add zero inflation on top of the default variable capture
results = scribe.fit(adata, zero_inflation=True)

# String form is still supported
results = scribe.fit(adata, model="zinbvcp")
```

!!! tip "Add a low-rank guide"
    Adding `guide_rank=64` gives SCRIBE a parameter-efficient way to capture
    gene-gene correlations that a mean-field posterior would miss. See
    [Model Selection](model-selection.md) for the full decision guide.

!!! note "Why variable capture is on by default"
    Empirically, we have **not yet encountered a dataset** that does not
    benefit from variable capture. Cell-specific capture probability accounts
    for library-size heterogeneity that is ubiquitous in scRNA-seq protocols.
    Set `variable_capture=False` if your library sizes are tightly controlled
    (less than 2x variation between cells).

**Full guide:** [Model Selection](model-selection.md) |
**Parameter cheatsheet:** [Parameter Reference](parameters.md)

---

## 3. Parameterization

How the Negative Binomial parameters are represented internally. The choice
affects optimization speed, numerical stability, and which downstream
analyses are available. This is independent of whether you select the
likelihood with **`variable_capture` / `zero_inflation`** or with a **`model=`**
string (both remain valid; see [Model selection](#2-model-selection)).

| Parameter          | Default       | Description                                                                                                              |
| ------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `parameterization` | `"canonical"` (NB family) / `"two_state_natural"` (TwoState family) | See the per-family tables below. |
| `unconstrained`    | `False`       | Use Normal + transform instead of constrained distributions. **Required** for hierarchical priors and BNB overdispersion |

**NB-family parameterizations** (`model` in `nbdm` / `zinb` / `nbvcp` / `zinbvcp`):

| Name           | Code          | Samples       | Derives                             | Best for                             |
| -------------- | ------------- | ------------- | ----------------------------------- | ------------------------------------ |
| **Canonical**  | `"canonical"` | \(p, r\)      | ---                                 | Direct interpretation                |
| **Mean probs** | `"mean_prob"` | \(p, \mu\)    | \(r = \mu(1-p)/p\)                  | Couples mean and success probability |
| **Mean odds**  | `"mean_odds"` | \(\phi, \mu\) | \(p = 1/(1+\phi)\), \(r = \mu\phi\) | Stable when \(p\) is near 1          |
| **Mean disp**  | `"mean_disp"` | \(\mu, r\)    | \(\phi = r/\mu\), \(p = \mu/(\mu+r)\) | Fisher-orthogonal mean & dispersion; faithful compositional DE |

**TwoState-family parameterizations** (`model` in `twostate` / `twostatevcp`):

| Name             | Code                       | Aliases               | Samples                              | Best for                                       |
| ---------------- | -------------------------- | --------------------- | ------------------------------------ | ---------------------------------------------- |
| **Natural**      | `"two_state_natural"`      | `natural`             | \(\mu, b, k^-\)                      | Biophysical interpretation; NUTS               |
| **Ratio**        | `"two_state_ratio"`        | `ratio`               | \(\mu, b, s = k^-/k^+\)              | Mean-field SVI across widely-varying \(\mu\)  |
| **Mean-Fano**    | `"two_state_mean_fano"`    | `mean_fano`, `fano`   | \(\mu, F = \text{Var}/\mu - 1, \kappa\) | When PPC bands are systematically wide       |
| **Moment-delta** | `"two_state_moment_delta"` | `moment_delta`, `delta` | \(\mu, F, \delta = 1/(\kappa+1) \in (0,1)\) | When \(\kappa\) posterior tracks its prior |

All four TwoState parameterizations are mean-preserving by construction;
`mean_fano` and `moment_delta` additionally preserve the Fano factor. See
the [Two-state promoter theory page](../theory/two-state-promoter.md) for
the math.

```python
# Mean odds parameterization (often converges faster) — NB family
results = scribe.fit(adata, variable_capture=True, parameterization="mean_odds")

# TwoState natural parameterization for bursty / bimodal genes
results = scribe.fit(adata, model="twostatevcp", parameterization="natural")

# TwoState moment-delta: bounded shape coordinate when the κ posterior
# tracks its prior under mean_fano
results = scribe.fit(
    adata, model="twostatevcp", parameterization="moment_delta",
    unconstrained=True,
)

# Unconstrained mode --- needed for hierarchical priors and BNB
results = scribe.fit(adata, model="nbdm", unconstrained=True)
```

!!! info "When to use `unconstrained=True`"
    You **must** set `unconstrained=True` when using any of the following:
    hierarchical priors (`priors={"mean_expression": ...}`, `"probability"`,
    `"zero_inflation"`), mean anchoring (`expression_anchor`), BNB
    overdispersion (`overdispersion="bnb"`), or dataset-level priors. SCRIBE
    will raise a `ValueError` if you forget. TwoState parameterizations
    also require `unconstrained=True`.

**Full guide:** [Model Selection > Parameterizations](model-selection.md#parameterizations) |
**Parameter cheatsheet:** [Parameter Reference](parameters.md#parameterization-mappings) |
**TwoState theory:** [Two-state promoter](../theory/two-state-promoter.md)

---

## 4. Inference method

SCRIBE supports four inference backends, all accessed through the same
`scribe.fit()` call.

| Parameter          | Default | Description                                                                                                                            |
| ------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `inference_method` | `"svi"` | `"svi"` (Stochastic Variational Inference), `"mcmc"` (NUTS), `"vae"` (Variational Autoencoder), or `"laplace"` (Laplace Approximation) |

### SVI parameters

| Parameter            | Default  | Description                                                                                                                                   |
| -------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_steps`            | `50_000` | Number of optimization steps                                                                                                                  |
| `batch_size`         | `None`   | Mini-batch size. `None` = full-batch. Recommended for > 10 K cells                                                                            |
| `stable_update`      | `True`   | Numerically stable parameter updates                                                                                                          |
| `log_progress_lines` | `False`  | Emit periodic plain-text progress lines (useful for SLURM logs)                                                                               |
| `early_stopping`     | `None`   | Dict or `EarlyStoppingConfig` for automatic convergence detection                                                                             |
| `restore_best`       | `False`  | Track the best variational parameters during training and restore them at the end                                                             |
| `optimizer_config`   | `None`   | Custom optimizer: `{"name": "adam", "step_size": 1e-3}`. Supports `"adam"`, `"clipped_adam"`, `"adagrad"`, `"rmsprop"`, `"sgd"`, `"momentum"` |

```python
# SVI with mini-batching and early stopping
results = scribe.fit(
    adata,
    variable_capture=True,
    n_steps=200_000,
    batch_size=512,
    early_stopping={
        "patience": 500,
        "min_delta": 1.0,
        "smoothing_window": 50,
        "restore_best": True,
    },
)
```

### MCMC parameters

| Parameter    | Default | Description                                                              |
| ------------ | ------- | ------------------------------------------------------------------------ |
| `n_samples`  | `2_000` | Posterior samples per chain                                              |
| `n_warmup`   | `1_000` | Warmup (burn-in) samples                                                 |
| `n_chains`   | `1`     | Number of parallel NUTS chains                                           |
| `svi_init`   | `None`  | `ScribeSVIResults` to warm-start MCMC (cross-parameterization supported) |
| `enable_x64` | `None`  | Float64 precision. Defaults to `True` for MCMC, `False` for SVI/VAE      |

```python
# MCMC warm-started from SVI
svi_results = scribe.fit(adata, model="nbdm", n_steps=50_000)
mcmc_results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="mcmc",
    svi_init=svi_results,
    n_samples=2_000,
    n_warmup=500,
    n_chains=4,
)
```

!!! tip "Cross-parameterization initialization"
    The `svi_init` parameter handles parameterization mapping automatically.
    You can run SVI with `parameterization="mean_prob"` and initialize MCMC
    with `parameterization="mean_odds"`---SCRIBE converts the MAP estimates
    internally.

### Laplace parameters

The Laplace path uses per-cell Newton iteration on latent variables with an
outer Adam loop on global parameters. It is the recommended inference method
for `model="pln"`, `"lnm"`, and `"lnmvcp"`.

| Parameter          | Default  | Description                                                           |
| ------------------ | -------- | --------------------------------------------------------------------- |
| `laplace_config`   | `None`   | Dict or `LaplaceConfig` for Newton inner-loop and outer-loop settings |
| `n_steps`          | `50_000` | Outer optimization steps (shared with SVI)                            |
| `batch_size`       | `None`   | Mini-batch size (shared with SVI)                                     |
| `optimizer_config` | `None`   | Outer Adam configuration (shared with SVI)                            |

The `laplace_config` dict controls the inner Newton solver:

| `laplace_config` key | Default  | Description                                                            |
| -------------------- | -------- | ---------------------------------------------------------------------- |
| `n_newton_steps`     | `5`      | Newton iterations per cell per outer step                              |
| `damping`            | `1e-2`   | Tikhonov regularization on the Hessian diagonal                        |
| `newton_tolerance`   | `1e-4`   | Gradient norm threshold for convergence                                |
| `convergence_action` | `"warn"` | What to do if cells don't converge: `"warn"`, `"raise"`, or `"ignore"` |

```python
# Laplace inference for PLN
results = scribe.fit(
    adata,
    model="pln",
    inference_method="laplace",
    n_steps=50_000,
    batch_size=256,
    laplace_config={
        "n_newton_steps": 10,
        "damping": 1e-3,
        "newton_tolerance": 1e-4,
        "convergence_action": "warn",
    },
)
```

!!! note "Supported models"
    Laplace inference is available for `model="pln"`, `"lnm"`, and `"lnmvcp"`.
    The NB-family models (`"nbdm"`, `"nbvcp"`, `"zinb"`, `"zinbvcp"`) use SVI,
    MCMC, or VAE.

**Full guide:** [Inference Methods](inference.md)

---

## 5. Variational guide configuration

The guide (variational family) controls how well the approximate posterior
can capture correlations between parameters.

### Low-rank Gaussian guides

| Parameter      | Default | Description                                                                                                                                                                                      |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `guide_rank`   | `None`  | Rank for low-rank guide on gene-specific parameters. `None` = mean-field (fully factorized)                                                                                                      |
| `joint_params` | `None`  | Parameter names to model jointly. Accepts shorthands (`"all"`, `"biological"`, `"mean"`, `"prob"`, `"gate"`) or an explicit list (e.g. `["mu", "phi"]`). Works with `guide_rank` or `guide_flow` |
| `dense_params` | `None`  | Subset of `joint_params` that get full cross-gene coupling. Accepts same shorthands or explicit list. Others get gene-local conditioning                                                         |

```python
# Low-rank guide (captures gene-gene correlations)
results = scribe.fit(adata, model="nbdm", guide_rank=8)

# Joint low-rank across mu and phi
results = scribe.fit(
    adata,
    model="nbdm",
    parameterization="mean_odds",
    unconstrained=True,
    guide_rank=10,
    joint_params=["mu", "phi"],
)
```

### Normalizing flow guides

Replaces the Gaussian variational family with a learned invertible
transformation, enabling multimodal, skewed, and heavy-tailed posterior
approximations. Mutually exclusive with `guide_rank`.

| Parameter                     | Default         | Description                                                                         |
| ----------------------------- | --------------- | ----------------------------------------------------------------------------------- |
| `guide_flow`                  | `None`          | Flow type: `"affine_coupling"` (recommended), `"spline_coupling"`, `"maf"`, `"iaf"` |
| `guide_flow_num_layers`       | `4`             | Number of coupling layers                                                           |
| `guide_flow_hidden_dims`      | `[64, 64]`      | Hidden sizes in the conditioner MLP                                                 |
| `guide_flow_activation`       | `"relu"`        | Activation function for conditioner MLPs                                            |
| `guide_flow_n_bins`           | `8`             | Spline bins (only for `"spline_coupling"`)                                          |
| `guide_flow_mixture_strategy` | `"independent"` | `"independent"` or `"shared"` for mixture/dataset components                        |
| `guide_flow_zero_init`        | `True`          | Identity-init via zero output layer                                                 |
| `guide_flow_layer_norm`       | `True`          | LayerNorm in conditioner MLP                                                        |
| `guide_flow_residual`         | `True`          | Residual connections in conditioner MLP                                             |
| `guide_flow_soft_clamp`       | `True`          | Smooth arctan clamp on affine log-scale (Andrade 2024)                              |
| `guide_flow_loft`             | `True`          | LOFT compression + trainable final affine                                           |
| `guide_flow_log_det_f64`      | `False`         | Float64 log-det accumulation (datacenter GPUs only)                                 |

```python
# Affine coupling flow guide (recommended for high-dimensional gene params)
results = scribe.fit(
    adata,
    model="nbdm",
    unconstrained=True,
    guide_flow="affine_coupling",
    guide_flow_num_layers=4,
)

# Joint flow across mu and phi
results = scribe.fit(
    adata,
    model="nbdm",
    parameterization="mean_odds",
    unconstrained=True,
    guide_flow="affine_coupling",
    joint_params=["mu", "phi"],
)
```

!!! warning "Use `affine_coupling` for guide-level flows"
    In scRNA-seq, gene-specific parameters live in thousands to tens of
    thousands of dimensions. Only affine coupling layers are numerically
    stable enough at this scale. Spline coupling and autoregressive flows
    are better suited for low-dimensional settings like VAE latent spaces.

**Full guide:** [Variational Guide Families](guide-families.md)

---

## 6. Capture amortization (VCP models)

When `variable_capture=True` (NBVCP and ZINBVCP), each cell has its own capture
probability. Amortization replaces per-cell variational parameters with a
small neural network that predicts them from total UMI count, reducing the
parameter count from \(O(N_{\text{cells}})\) to the network weights.

| Parameter                  | Default        | Description                                                          |
| -------------------------- | -------------- | -------------------------------------------------------------------- |
| `amortize_capture`         | `False`        | Enable neural-network amortization of capture probability            |
| `capture_hidden_dims`      | `[64, 32]`     | Hidden layer sizes for the amortizer MLP                             |
| `capture_activation`       | `"leaky_relu"` | Activation function (`"relu"`, `"gelu"`, `"silu"`, `"tanh"`, ...)    |
| `capture_output_transform` | `"softplus"`   | Output transform for positive parameters (`"softplus"` or `"exp"`)   |
| `capture_clamp_min`        | `0.1`          | Minimum clamp for MLP outputs. `None` to disable                     |
| `capture_clamp_max`        | `50.0`         | Maximum clamp for MLP outputs. `None` to disable                     |
| `capture_amortization`     | `None`         | `AmortizationConfig` or dict that overrides all six parameters above |

```python
# Amortized capture with defaults --- useful for very large datasets
results = scribe.fit(adata, variable_capture=True, amortize_capture=True)

# Custom amortizer architecture
results = scribe.fit(
    adata,
    variable_capture=True,
    amortize_capture=True,
    capture_hidden_dims=[128, 64, 32],
    capture_activation="gelu",
)
```

!!! info "When to amortize"
    Amortization is most beneficial when the number of cells is so large that
    you get out-of-memory issues. For small datasets the per-cell
    parameterization is fine and avoids the neural network overhead.

**See also:** [Variational Guide Families > Amortized](guide-families.md#amortized)

---

## 7. Hierarchical priors (gene-level)

Hierarchical priors provide **adaptive shrinkage** across mixture components
(for the mean) or across genes (for the probability / zero-inflation gate). They
share statistical strength so that most parameters stay close to a population
center while allowing true outliers to deviate. All require `unconstrained=True`
and are declared through the unified `priors` dict (see
[Defining Priors](priors.md)) by attaching a **family string** to a canonical
parameter name:

| `priors` key      | Shrinks ...                                       | Notes                        |
| ----------------- | ------------------------------------------------- | ---------------------------- |
| `mean_expression` | \(\mu\) (or \(r\)) **across mixture components**  | Requires `n_components >= 2` |
| `probability`     | \(p\) (or \(\phi\)) **across genes**              |                              |
| `zero_inflation`  | the zero-inflation gate **across genes**          | Only for ZI models           |

Each accepts a family string: `"gaussian"`, `"horseshoe"`, or `"neg"`.

- **Gaussian** --- simple Normal shrinkage. Lightest; suitable when most
  parameters are expected to differ moderately.
- **Horseshoe** --- strong shrinkage toward zero with heavy tails for true
  outliers. Good default for sparse signals.
- **NEG** (Normal-Exponential-Gamma) --- even heavier tails than Horseshoe
  with continuous adaptive shrinkage.

```python
# Horseshoe shrinkage on mu across cell types
results = scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    n_components=5,
    priors={"mean_expression": "horseshoe"},
)

# Gaussian prior on gene-specific p
results = scribe.fit(
    adata,
    model="nbdm",
    unconstrained=True,
    priors={"probability": "gaussian"},
)

# NEG prior on zero-inflation gate
results = scribe.fit(
    adata,
    zero_inflation=True,
    unconstrained=True,
    priors={"zero_inflation": "neg"},
)
```

### Hyperparameters

To fine-tune a Horseshoe or NEG family, pass a **family spec** (a dict carrying
the reserved `"type"` key) instead of a bare string. The extra keys set the
family's hyperparameters:

| Spec key | Default | Family | Description |
|----------|---------|--------|-------------|
| `tau0` | `1.0` | horseshoe | Global shrinkage scale. Smaller = stronger shrinkage |
| `slab_df` | `4` | horseshoe | Degrees of freedom of the regularizing slab |
| `slab_scale` | `2.0` | horseshoe | Scale of the regularizing slab |
| `u` | `1.0` | neg | Shape parameter \(u\) |
| `a` | `1.0` | neg | Shape parameter \(a\) |
| `tau` | `1.0` | neg | Scale parameter \(\tau\) |

```python
# Horseshoe with tighter global shrinkage
results = scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    n_components=4,
    priors={
        "mean_expression": {
            "type": "horseshoe",
            "tau0": 0.5,
            "slab_scale": 1.0,
        }
    },
)
```

**Full guide:** [Theory: Hierarchical Priors](../theory/hierarchical-priors.md)

---

## 8. Mean anchoring prior

The mean anchoring prior resolves the \(\mu\)--\(\phi\) degeneracy in the
Negative Binomial by centering each gene's biological mean on its observed
sample mean, adjusted for average capture efficiency.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expression_anchor` | `False` | Enable data-informed anchoring. Automatically sets `unconstrained=True` |
| `expression_anchor_sigma` | `0.3` | Log-scale standard deviation. `0.1`--`0.2` = tight, `0.3`--`0.5` = recommended, `> 1` = weak |

For VCP models, SCRIBE needs to estimate the average capture probability
from data. Provide biology-informed capture information via the `priors`
dictionary:

```python
# Mean anchoring with organism-informed capture
results = scribe.fit(
    adata,
    variable_capture=True,
    expression_anchor=True,
    expression_anchor_sigma=0.3,
    priors={"organism": "human"},
    amortize_capture=True,
)

# Mean anchoring with explicit capture efficiency
results = scribe.fit(
    adata,
    variable_capture=True,
    expression_anchor=True,
    priors={"capture_efficiency": (10.0, 1e5)},
)
```

!!! note
    Without variable capture (`nbdm`, `zinb`), the anchor uses the implicit
    capture \(\bar{\nu} = 1\), so no extra `priors` are needed.

**Full guide:** [Theory: Anchoring Priors](../theory/anchoring-priors.md)

---

## 9. BNB overdispersion

The **Beta Negative Binomial** extension adds a per-gene concentration
parameter \(\kappa_g\) that allows heavier tails than the standard NB. It
can be combined with any model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overdispersion` | `"none"` | `"none"` (standard NB) or `"bnb"` (Beta Negative Binomial) |

The hierarchical prior on \(\kappa_g\) is declared through `priors` under the
canonical name `overdispersion` (a family string `"horseshoe"` or `"neg"`);
it defaults to `"horseshoe"` when `overdispersion="bnb"` and no prior is given.

```python
results = scribe.fit(
    adata,
    variable_capture=True,
    overdispersion="bnb",
    unconstrained=True,
    amortize_capture=True,
    priors={"overdispersion": "neg"},   # override the default horseshoe
)
```

!!! warning "Fit variable capture first"
    What appears as heavy-tailed gene expression often reflects variable
    capture efficiency rather than genuine per-gene overdispersion. Always
    fit an NBVCP model first and check the posterior predictive distribution.
    Add BNB only when excess dispersion persists after accounting for
    capture.

**Full guide:** [Theory: Beta Negative Binomial](../theory/beta-negative-binomial.md)
| [Model Selection > BNB](model-selection.md#bnb-overdispersion)

---

## 10. Mixture models

Mixture models discover cell subpopulations by fitting \(K\) sets of
gene-specific parameters. Each cell is softly assigned to a component.

| Parameter        | Default | Description                                                                                                                                                                                                                                                             |
| ---------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_components`   | `None`  | Number of mixture components. `None` = single component (no mixture). Must be >= 2 if set                                                                                                                                                                               |
| `mixture_params` | `"all"` | Which parameters are component-specific. Accepts shorthands: `"all"` (every param incl. gate), `"biological"` (core params only — NB: `r`, `p`/`phi`, `mu`; TwoState: parameterization-specific extras), `"mean"`, `"prob"`, `"gate"`, or an explicit list like `["r"]` |

```python
# NB-family: discover 5 cell types
results = scribe.fit(
    adata,
    variable_capture=True,
    n_components=5,
    n_steps=150_000,
    amortize_capture=True,
)

# Extract assignments
assignments = results.cell_type_assignments(counts=adata.X)
```

TwoState models support the same mixture API.  All four parameterizations
work; the factory automatically resolves parameter names for the active
parameterization:

```python
# TwoState mixture: 3 components with per-component mu and shape params
ts_results = scribe.fit(
    adata,
    model="twostatevcp",
    parameterization="two_state_natural",
    n_components=3,
    mixture_params=["mu", "burst_size", "k_off"],
    n_steps=150_000,
)

# Same downstream API: assignments, denoising, log-prob decomposition
assignments = ts_results.cell_type_assignments(counts=adata.X)
denoised = ts_results.get_denoised_counts_map(counts=adata.X)
```

### Annotation priors

If you have partial or complete cell-type labels, use them as **soft priors**
on mixture assignments. This guides the model without forcing hard
assignments.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `annotation_key` | `None` | Column(s) in `adata.obs` with cell-type labels. Accepts a string or a list of strings for composite labels |
| `annotation_confidence` | `3.0` | Prior strength \(\kappa\). `0` = ignored, `3` = ~20x boost, large = near-hard assignment |
| `annotation_component_order` | `None` | Explicit label-to-component mapping. `None` sorts labels alphabetically |
| `annotation_min_cells` | `None` | Minimum cells per label. Labels below this threshold are treated as unlabeled |

```python
# Use existing annotations as soft priors
results = scribe.fit(
    adata,
    variable_capture=True,
    n_components=5,
    annotation_key="cell_type",
    annotation_confidence=3.0,
    amortize_capture=True,
)

# Composite labels from two columns (e.g. cell_type x treatment)
results = scribe.fit(
    adata,
    variable_capture=True,
    annotation_key=["cell_type", "treatment"],
    annotation_confidence=5.0,
    annotation_min_cells=20,
    amortize_capture=True,
)
```

!!! info "Automatic component inference"
    When `annotation_key` is set but `n_components` is omitted, SCRIBE
    automatically infers the number of components from the unique non-null
    labels (filtered by `annotation_min_cells` if set).

**See also:** [Results Class](results.md) (mixture assignments and
components)

---

## 11. Multi-dataset hierarchy

When your experiment spans **multiple datasets** (e.g. batches, conditions,
or labs), SCRIBE can share statistical strength across datasets via
dataset-level hierarchical priors on gene-specific parameters.

### Dataset specification

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_key` | `None` | Column in `adata.obs` identifying the dataset of each cell. A **list** of columns declares a crossed multi-factor design (see [Crossed designs](#crossed-multi-factor-designs)) |
| `hierarchy` | `None` | Structured alternative to `dataset_key`: a list of `scribe.GroupLevel(...)` declaring each grouping factor and its effect type |
| `interactions` | `None` | List of factor-name tuples adding interaction effects between declared factors |
| `n_datasets` | `None` | Number of datasets/leaves. Auto-inferred from `dataset_key`/`hierarchy` when `None` |
| `dataset_params` | `None` | Which parameters become dataset-specific (auto-determined from priors when `None`) |
| `dataset_mixing` | `None` | Dataset-specific mixture weights. `None` = auto (`True` when >= 2 datasets) |
| `auto_downgrade_single_dataset_hierarchy` | `True` | Automatically simplify hierarchy when the grouping resolves to a single dataset |

### Dataset-level priors

Each parameter that varies across genes can also vary across datasets. Declare
this through the unified `priors` dict by attaching a **`{level: family}` dict**
to a canonical parameter name --- the level keys are your grouping factors (a
bare `dataset_key="batch"` is a single factor named `"batch"`). The family
controls how much dataset-to-dataset variation is allowed. See
[Defining Priors](priors.md) for the full grammar.

| `priors` key      | Dataset/level hierarchy on ...                                                          |
| ----------------- | --------------------------------------------------------------------------------------- |
| `mean_expression` | \(\mu\) (the additive log-mean decomposition over factors)                              |
| `probability`     | \(p\)                                                                                   |
| `zero_inflation`  | the zero-inflation gate                                                                 |
| `overdispersion`  | BNB \(\kappa\) (requires `overdispersion="bnb"`)                                        |
| `regime`          | **TwoState only.** the bursting-regime coordinate (`k_off` / `switching_ratio` / ...)   |

Each family is `"gaussian"`, `"horseshoe"`, or `"neg"` (or a `{"type": ...}`
spec with hyperparameters). A few **structural** options stay as keyword
arguments --- they shape *how* the hierarchy is built, not its prior family:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prob_dataset_mode` | `"gene_specific"` | How \(p\) varies: `"scalar"`, `"gene_specific"`, or `"two_level"` |
| `regime_dataset_target` | `None` | **TwoState only.** Pin the regime coordinate; defaults to the parameterization's regime coordinate |
| `overdispersion_dataset_independent` | `True` | **TwoState only.** Leave the overdispersion coordinate (`burst_size` / `excess_fano`) free per dataset (no cross-dataset hierarchy) |
| `capture_scaling_prior` | `"none"` | Prior on per-dataset capture scaling \(\eta_d\). For VCP models |

```python
# Two-dataset comparison with horseshoe shrinkage on mu
results = scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    dataset_key="batch",
    priors={
        "mean_expression": {"batch": "horseshoe"},
        "probability":     {"batch": "gaussian"},
    },
    amortize_capture=True,
)

# TwoState two-condition comparison: link mu AND the bursting regime across
# datasets (horseshoe), leaving overdispersion free per dataset (default).
results = scribe.fit(
    adata,
    model="twostatevcp",
    parameterization="moment_delta",
    unconstrained=True,
    dataset_key="condition",
    priors={
        "mean_expression": {"condition": "horseshoe"},
        "regime":          {"condition": "horseshoe"},
    },
)
```

!!! info "Single-dataset downgrade"
    When `dataset_key` points to a column with only one unique value, SCRIBE
    automatically downgrades dataset-level priors to gene-level equivalents
    (or drops them) and emits a `UserWarning`. Disable this with
    `auto_downgrade_single_dataset_hierarchy=False`.

### Crossed multi-factor designs

When cells carry **more than one** grouping label at once — e.g. a donor *and*
a treatment — pass a **list** of columns (crossing is implicit) instead of a
single `dataset_key`. SCRIBE then gives mean expression an additive
decomposition over the factors,
\(\log\mu_g^{(\ell)} = \log\mu_g^{\mathrm{pop}} + \sum_f \alpha_g^{(f)}[\mathrm{level}_f(\ell)]\),
so the treatment effect is shared across donors while each donor's own
deviation is modelled separately (see
[Theory: crossed and nested designs](../theory/hierarchical-priors.md#crossed-and-nested-designs-multiple-grouping-factors)).

For finer control, use a structured `hierarchy=[GroupLevel(...)]` and mark the
**contrast of interest** as a fixed effect (no learned shrinkage, so the
contrast is not pulled toward zero). Set a prior family **per factor** by giving
the canonical name a `{level: family}` dict in `priors`:

```python
results = scribe.fit(
    adata,
    parameterization="mean_odds",
    variable_capture=True,
    unconstrained=True,
    hierarchy=[
        scribe.GroupLevel("perturbation", effect_type="fixed"),  # 2-level contrast
        scribe.GroupLevel("sample"),                             # 7 donors -> random
    ],
    priors={
        "mean_expression": {
            "perturbation": "gaussian",   # fixed-scale, weakly-informative
            "sample": "horseshoe",        # adaptive shrinkage across donors
        },
        "probability": {"sample": "gaussian"},  # technical p stays leaf-exchangeable
    },
)
```

| Argument | Form | Notes |
|----------|------|-------|
| `dataset_key=["treatment", "sample"]` | list of columns | crossed factors, all random |
| `hierarchy=[GroupLevel(...), ...]` | structured | per-factor `effect_type` (`"random"` \| `"fixed"`), `nested_in`, `fixed_scale` |
| `interactions=[("treatment", "sample")]` | list of tuples | add an interaction (random) effect between factors; set its prior family with the `":"`-joined key in `priors` (e.g. `priors={"mean_expression": {"treatment:sample": "horseshoe"}}`) for a random slope on the mean |
| `priors={"mean_expression": {factor: family}}` | `{level: family}` dict | per-factor prior family; add a `"base"` key for the gene-level prior |

`GroupLevel(name, nested_in=None, effect_type="random", fixed_scale=None)`. Only
the expression target (\(\mu\)/\(r\)) gets the additive decomposition; `p`, gate
and regime keep the single-axis per-leaf hierarchy. The multi-factor hierarchy is
a Python-API feature — the CLI supports a single `dataset_key`.

!!! tip "Worked example"
    The [crossed-hierarchy tutorial](../tutorials/zhao_2021_hierarchical.md) fits
    exactly this donor × condition model end to end and reads off the
    donor-averaged treatment effect with [`compare_groups`](differential-expression.md#population-differential-expression-across-grouping-factors).

**Full guide:** [Theory: Hierarchical Priors > Multiple datasets](../theory/hierarchical-priors.md#extension-to-multiple-datasets)

---

## 12. Custom prior hyperparameters

The `priors` dictionary lets you override default prior hyperparameters for
any model parameter. Values are tuples of hyperparameters whose meaning
depends on the distribution family.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `priors` | `None` | Dict mapping parameter names to hyperparameter tuples |

```python
# Override p and r priors
results = scribe.fit(
    adata,
    model="nbdm",
    priors={
        "p": (1.0, 1.0),     # Beta(1, 1) --- uniform
        "r": (0.0, 1.0),     # LogNormal(0, 1)
    },
)

# Symmetric Dirichlet for mixture weights (scalar is broadcast)
results = scribe.fit(
    adata,
    variable_capture=True,
    n_components=4,
    priors={"mixing": 5.0},  # equivalent to (5.0, 5.0, 5.0, 5.0)
)

# Biology-informed capture prior
results = scribe.fit(
    adata,
    variable_capture=True,
    priors={"organism": "human"},
)
```

---

## 13. VAE architecture

When `inference_method="vae"`, these parameters configure the encoder-decoder
neural network architecture.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vae_latent_dim` | `10` | Dimensionality of the latent space |
| `vae_encoder_hidden_dims` | `None` | Encoder hidden layer sizes (e.g. `[512, 256]`) |
| `vae_decoder_hidden_dims` | `None` | Decoder hidden layer sizes |
| `vae_activation` | `None` | Activation function (`"relu"`, `"gelu"`, `"silu"`, ...) |
| `vae_input_transform` | `"log1p"` | Input preprocessing: `"log1p"`, `"log"`, `"sqrt"`, `"identity"` |
| `vae_standardize` | `False` | Standardize transformed inputs to zero mean, unit variance |
| `vae_decoder_transforms` | `None` | Per-parameter decoder output transforms |

### Normalizing flow priors

For more expressive latent distributions, attach a normalizing flow:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vae_flow_type` | `"none"` | `"none"`, `"affine_coupling"`, `"spline_coupling"`, `"maf"`, or `"iaf"` |
| `vae_flow_num_layers` | `4` | Number of flow layers |
| `vae_flow_hidden_dims` | `None` | Hidden dimensions in each flow layer |

```python
# VAE with spline coupling flow
results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="vae",
    vae_latent_dim=15,
    vae_encoder_hidden_dims=[512, 256],
    vae_flow_type="spline_coupling",
    vae_flow_num_layers=4,
    n_steps=100_000,
    batch_size=256,
)

# Cell embeddings
embeddings = results.get_latent_embeddings(data=adata.X, n_samples=100)
```

**Full guide:** [Inference Methods > VAE](inference.md#variational-autoencoder-vae)

---

## 14. Power-user overrides

For maximum control, bypass the flat keyword interface and pass fully
constructed configuration objects. When provided, these override the
corresponding keyword arguments.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_config` | `None` | `ModelConfig` object. Overrides flat `model`, `variable_capture`, `zero_inflation`, `parameterization`, `unconstrained`, `n_components`, `mixture_params`, `joint_params`, `dense_params`, `guide_rank`, and `priors` |
| `inference_config` | `None` | `InferenceConfig` object. Overrides `inference_method`, `n_steps`, `batch_size`, `stable_update`, `log_progress_lines`, `n_samples`, `n_warmup`, and `n_chains` |

```python
from scribe.models.config import ModelConfigBuilder

# Build a model config step by step (horseshoe shrinkage on per-component means)
builder = (
    ModelConfigBuilder()
    .for_model("nbvcp")
    .with_parameterization("mean_odds")
    .unconstrained()
    .as_mixture(n_components=5)
)
builder._expression_prior = "horseshoe"
model_cfg = builder.build()

results = scribe.fit(adata, model_config=model_cfg)
```

---

## Return value

`scribe.fit()` returns a results object whose type depends on the inference
method:

| Inference method | Return type | Key capabilities |
|------------------|-------------|------------------|
| `"svi"` | `ScribeSVIResults` | Posterior samples, loss history, PPC, denoising |
| `"mcmc"` | `ScribeMCMCResults` | Chain samples, NUTS diagnostics, PPC, denoising |
| `"vae"` | `ScribeVAEResults` | Latent embeddings, posterior samples, PPC, denoising |

All result types share a common analysis API: posterior sampling, posterior
predictive checks, Bayesian denoising, and log-likelihood computation.

**Full guide:** [Results Class](results.md)

---

## Common recipes

### Typical single-dataset analysis

```python
# NBVCP with amortized capture and early stopping
results = scribe.fit(
    adata,
    variable_capture=True,
    parameterization="mean_odds",
    n_steps=100_000,
    batch_size=512,
    amortize_capture=True,
    early_stopping={"patience": 500, "restore_best": True},
)
```

### Multi-dataset with hierarchical priors

```python
# Share strength across batches
results = scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    dataset_key="batch",
    priors={
        "mean_expression": {"batch": "horseshoe"},
        "probability":     {"batch": "gaussian"},
    },
    amortize_capture=True,
    n_steps=200_000,
)
```

### Mixture model with annotations

```python
# 8 cell types, guided by partial annotations
results = scribe.fit(
    adata,
    variable_capture=True,
    n_components=8,
    annotation_key="cell_type",
    annotation_confidence=3.0,
    annotation_min_cells=50,
    amortize_capture=True,
    n_steps=150_000,
)
```

### SVI-to-MCMC warm start

```python
# Fast exploration with SVI
svi_results = scribe.fit(
    adata, model="nbdm", parameterization="mean_prob", n_steps=50_000,
)

# Gold-standard posteriors with MCMC, initialized from SVI
mcmc_results = scribe.fit(
    adata,
    model="nbdm",
    parameterization="mean_odds",
    inference_method="mcmc",
    svi_init=svi_results,
    n_samples=4_000,
    n_warmup=500,
    n_chains=4,
)
```

### Full hierarchical model with anchoring and BNB

```python
# Everything turned on: VCP, anchoring, horseshoe, BNB
results = scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    expression_anchor=True,
    expression_anchor_sigma=0.3,
    overdispersion="bnb",
    priors={
        "organism": "human",        # capture info for anchoring
        "probability": "gaussian",  # gene-level hierarchical prior on p
    },
    amortize_capture=True,
    n_steps=300_000,
    batch_size=512,
)
```

### VAE with normalizing flows

```python
# Latent representation + spline flow
results = scribe.fit(
    adata,
    model="nbdm",
    inference_method="vae",
    vae_latent_dim=15,
    vae_encoder_hidden_dims=[512, 256],
    vae_flow_type="spline_coupling",
    n_steps=100_000,
    batch_size=256,
)

# Retrieve embeddings for downstream analysis
z = results.get_latent_embeddings(data=adata.X, n_samples=100)
```

---

## Quick reference

All `scribe.fit()` parameters at a glance, grouped by function:

??? info "Complete parameter table (click to expand)"

    **Data input**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `counts` | *(required)* | `ndarray` or `AnnData` |
    | `cells_axis` | `0` | `int` |
    | `layer` | `None` | `str` |
    | `seed` | `42` | `int` |

    **Model**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `variable_capture` | `None` | `bool` |
    | `zero_inflation` | `None` | `bool` |
    | `model` | `"nbvcp"` | `str` |
    | `parameterization` | `"canonical"` | `str` |
    | `unconstrained` | `False` | `bool` |

    **Priors** --- every prior family and hierarchy (gene-level shrinkage,
    dataset/factor hierarchies, base hyperparameters, and family-spec
    hyperparameters) is declared through one argument.

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `priors` | `None` | `dict` --- see [Defining Priors](priors.md) |

    **Mean anchoring**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `expression_anchor` | `False` | `bool` |
    | `expression_anchor_sigma` | `0.3` | `float` |

    **Overdispersion**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `overdispersion` | `"none"` | `str` |

    **Mixture**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `n_components` | `None` | `int` |
    | `mixture_params` | `"all"` | `str` or `list[str]` |

    **Annotation priors**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `annotation_key` | `None` | `str` or `list[str]` |
    | `annotation_confidence` | `3.0` | `float` |
    | `annotation_component_order` | `None` | `list[str]` |
    | `annotation_min_cells` | `None` | `int` |

    **Multi-dataset**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `dataset_key` | `None` | `str` or `list[str]` |
    | `hierarchy` | `None` | `list[GroupLevel]` |
    | `interactions` | `None` | `list[tuple[str, ...]]` |
    | `n_datasets` | `None` | `int` |
    | `dataset_params` | `None` | `list[str]` |
    | `dataset_mixing` | `None` | `bool` |
    | `prob_dataset_mode` | `"gene_specific"` | `str` |
    | `regime_dataset_target` | `None` | `str` (TwoState) |
    | `overdispersion_dataset_independent` | `True` | `bool` (TwoState) |
    | `capture_scaling_prior` | `"none"` | `str` |
    | `auto_downgrade_single_dataset_hierarchy` | `True` | `bool` |

    Dataset/factor prior families themselves go in `priors` as `{level: family}`
    dicts (see [Defining Priors](priors.md)).

    **Guide (Gaussian)**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `guide_rank` | `None` | `int` |
    | `joint_params` | `None` | `str` or `list[str]` |
    | `dense_params` | `None` | `str` or `list[str]` |
    | `priors` | `None` | `dict` |

    **Guide (Normalizing Flow)**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `guide_flow` | `None` | `str` |
    | `guide_flow_num_layers` | `4` | `int` |
    | `guide_flow_hidden_dims` | `[64, 64]` | `list[int]` |
    | `guide_flow_activation` | `"relu"` | `str` |
    | `guide_flow_n_bins` | `8` | `int` |
    | `guide_flow_mixture_strategy` | `"independent"` | `str` |
    | `guide_flow_zero_init` | `True` | `bool` |
    | `guide_flow_layer_norm` | `True` | `bool` |
    | `guide_flow_residual` | `True` | `bool` |
    | `guide_flow_soft_clamp` | `True` | `bool` |
    | `guide_flow_loft` | `True` | `bool` |
    | `guide_flow_log_det_f64` | `False` | `bool` |

    **Capture amortization**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `amortize_capture` | `False` | `bool` |
    | `capture_hidden_dims` | `[64, 32]` | `list[int]` |
    | `capture_activation` | `"leaky_relu"` | `str` |
    | `capture_output_transform` | `"softplus"` | `str` |
    | `capture_clamp_min` | `0.1` | `float` |
    | `capture_clamp_max` | `50.0` | `float` |
    | `capture_amortization` | `None` | `AmortizationConfig` |

    **Inference**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `inference_method` | `"svi"` | `str` |
    | `n_steps` | `50_000` | `int` |
    | `batch_size` | `None` | `int` |
    | `optimizer_config` | `None` | `dict` |
    | `stable_update` | `True` | `bool` |
    | `log_progress_lines` | `False` | `bool` |
    | `early_stopping` | `None` | `dict` or `EarlyStoppingConfig` |
    | `restore_best` | `False` | `bool` |
    | `n_samples` | `2_000` | `int` |
    | `n_warmup` | `1_000` | `int` |
    | `n_chains` | `1` | `int` |
    | `svi_init` | `None` | `ScribeSVIResults` |
    | `enable_x64` | `None` | `bool` |

    **VAE**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `vae_latent_dim` | `10` | `int` |
    | `vae_encoder_hidden_dims` | `None` | `list[int]` |
    | `vae_decoder_hidden_dims` | `None` | `list[int]` |
    | `vae_activation` | `None` | `str` |
    | `vae_input_transform` | `"log1p"` | `str` |
    | `vae_standardize` | `False` | `bool` |
    | `vae_decoder_transforms` | `None` | `dict` |
    | `vae_flow_type` | `"none"` | `str` |
    | `vae_flow_num_layers` | `4` | `int` |
    | `vae_flow_hidden_dims` | `None` | `list[int]` |

    **Power-user overrides**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `model_config` | `None` | `ModelConfig` |
    | `inference_config` | `None` | `InferenceConfig` |

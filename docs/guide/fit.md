# The `scribe.fit()` Interface

`scribe.fit()` is the **single entry point** for all SCRIBE inference. Every
model, parameterization, prior, inference engine, and guide family is
configured through keyword arguments to this one function. This page walks
through every parameter group, explains when and why each matters, and links
to the deeper guides and theory pages for full details.

```python
import scribe

# Minimal call --- sensible defaults for everything
results = scribe.fit(adata)

# Recommended starting point for most datasets
results = scribe.fit(adata, model="nbvcp", amortize_capture=True)
```

!!! tip "Read order"
    If you are new to SCRIBE, read sections 1--4 below and the
    [Model Selection](model-selection.md) page. The remaining sections cover
    progressively more advanced features that you can explore as needed.

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

The `model` parameter picks which **likelihood** to use. All four models
share the same Negative Binomial core; the extensions add variable capture
and/or zero inflation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"nbdm"` | Likelihood family: `"nbdm"`, `"nbvcp"`, `"zinb"`, or `"zinbvcp"` |

```python
# Variable capture --- recommended default for heterogeneous library sizes
results = scribe.fit(adata, model="nbvcp")

# Zero-inflated NB with variable capture --- both mechanisms
results = scribe.fit(adata, model="zinbvcp")
```

!!! tip "Start with NBVCP"
    Unless total UMI counts are very homogeneous (within roughly a factor of
    two), begin with `model="nbvcp"`. Variable capture explains much of the
    apparent excess zeros and heavy tails in the data. See
    [Model Selection](model-selection.md) for the full decision guide.

**Full guide:** [Model Selection](model-selection.md) |
**Parameter cheatsheet:** [Parameter Reference](parameters.md)

---

## 3. Parameterization

How the Negative Binomial parameters are represented internally. The choice
affects optimization speed, numerical stability, and which downstream
analyses are available.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `parameterization` | `"canonical"` | `"canonical"` (alias `"standard"`), `"mean_prob"` (alias `"linked"`), or `"mean_odds"` (alias `"odds_ratio"`) |
| `unconstrained` | `False` | Use Normal + transform instead of constrained distributions. **Required** for hierarchical priors and BNB overdispersion |

| Name | Code | Samples | Derives | Best for |
|------|------|---------|---------|----------|
| **Canonical** | `"canonical"` | \(p, r\) | --- | Direct interpretation |
| **Mean probs** | `"mean_prob"` | \(p, \mu\) | \(r = \mu(1-p)/p\) | Couples mean and success probability |
| **Mean odds** | `"mean_odds"` | \(\phi, \mu\) | \(p = 1/(1+\phi)\), \(r = \mu\phi\) | Stable when \(p\) is near 1 |

```python
# Mean odds parameterization (often converges faster)
results = scribe.fit(adata, model="nbvcp", parameterization="mean_odds")

# Unconstrained mode --- needed for hierarchical priors and BNB
results = scribe.fit(adata, model="nbdm", unconstrained=True)
```

!!! info "When to use `unconstrained=True`"
    You **must** set `unconstrained=True` when using any of the following:
    hierarchical priors (`mu_prior`, `p_prior`, `gate_prior`), mean anchoring
    (`mu_mean_anchor`), BNB overdispersion (`overdispersion="bnb"`), or
    dataset-level priors. SCRIBE will raise a `ValueError` if you forget.

**Full guide:** [Model Selection > Parameterizations](model-selection.md#parameterizations) |
**Parameter cheatsheet:** [Parameter Reference](parameters.md#parameterization-mappings)

---

## 4. Inference method

SCRIBE supports three inference backends, all accessed through the same
`scribe.fit()` call.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_method` | `"svi"` | `"svi"` (Stochastic Variational Inference), `"mcmc"` (NUTS), or `"vae"` (Variational Autoencoder) |

### SVI parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_steps` | `50_000` | Number of optimization steps |
| `batch_size` | `None` | Mini-batch size. `None` = full-batch. Recommended for > 10 K cells |
| `stable_update` | `True` | Numerically stable parameter updates |
| `log_progress_lines` | `False` | Emit periodic plain-text progress lines (useful for SLURM logs) |
| `early_stopping` | `None` | Dict or `EarlyStoppingConfig` for automatic convergence detection |
| `restore_best` | `False` | Track the best variational parameters during training and restore them at the end |
| `optimizer_config` | `None` | Custom optimizer: `{"name": "adam", "step_size": 1e-3}`. Supports `"adam"`, `"clipped_adam"`, `"adagrad"`, `"rmsprop"`, `"sgd"`, `"momentum"` |

```python
# SVI with mini-batching and early stopping
results = scribe.fit(
    adata,
    model="nbvcp",
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | `2_000` | Posterior samples per chain |
| `n_warmup` | `1_000` | Warmup (burn-in) samples |
| `n_chains` | `1` | Number of parallel NUTS chains |
| `svi_init` | `None` | `ScribeSVIResults` to warm-start MCMC (cross-parameterization supported) |
| `enable_x64` | `None` | Float64 precision. Defaults to `True` for MCMC, `False` for SVI/VAE |

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

**Full guide:** [Inference Methods](inference.md)

---

## 5. Variational guide configuration

The guide (variational family) controls how well the approximate posterior
can capture correlations between parameters.

### Low-rank Gaussian guides

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guide_rank` | `None` | Rank for low-rank guide on gene-specific parameters. `None` = mean-field (fully factorized) |
| `joint_params` | `None` | Parameter names to model jointly (e.g. `["mu", "phi"]`). Works with `guide_rank` or `guide_flow` |
| `dense_params` | `None` | Subset of `joint_params` that get full cross-gene coupling. Others get gene-local conditioning |

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guide_flow` | `None` | Flow type: `"affine_coupling"` (recommended), `"spline_coupling"`, `"maf"`, `"iaf"` |
| `guide_flow_num_layers` | `4` | Number of coupling layers |
| `guide_flow_hidden_dims` | `[64, 64]` | Hidden sizes in the conditioner MLP |
| `guide_flow_activation` | `"relu"` | Activation function for conditioner MLPs |
| `guide_flow_n_bins` | `8` | Spline bins (only for `"spline_coupling"`) |
| `guide_flow_mixture_strategy` | `"independent"` | `"independent"` or `"shared"` for mixture/dataset components |
| `guide_flow_zero_init` | `True` | Identity-init via zero output layer |
| `guide_flow_layer_norm` | `True` | LayerNorm in conditioner MLP |
| `guide_flow_residual` | `True` | Residual connections in conditioner MLP |
| `guide_flow_soft_clamp` | `True` | Smooth arctan clamp on affine log-scale (Andrade 2024) |
| `guide_flow_loft` | `True` | LOFT compression + trainable final affine |
| `guide_flow_log_det_f64` | `False` | Float64 log-det accumulation (datacenter GPUs only) |

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

For VCP models (`nbvcp`, `zinbvcp`), each cell has its own capture
probability. Amortization replaces per-cell variational parameters with a
small neural network that predicts them from total UMI count, reducing the
parameter count from \(O(N_{\text{cells}})\) to the network weights.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `amortize_capture` | `False` | Enable neural-network amortization of capture probability |
| `capture_hidden_dims` | `[64, 32]` | Hidden layer sizes for the amortizer MLP |
| `capture_activation` | `"leaky_relu"` | Activation function (`"relu"`, `"gelu"`, `"silu"`, `"tanh"`, ...) |
| `capture_output_transform` | `"softplus"` | Output transform for positive parameters (`"softplus"` or `"exp"`) |
| `capture_clamp_min` | `0.1` | Minimum clamp for MLP outputs. `None` to disable |
| `capture_clamp_max` | `50.0` | Maximum clamp for MLP outputs. `None` to disable |
| `capture_amortization` | `None` | `AmortizationConfig` or dict that overrides all six parameters above |

```python
# Amortized capture with defaults --- recommended for large datasets
results = scribe.fit(adata, model="nbvcp", amortize_capture=True)

# Custom amortizer architecture
results = scribe.fit(
    adata,
    model="nbvcp",
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
(for `mu_prior`) or across genes (for `p_prior`, `gate_prior`). They share
statistical strength so that most parameters stay close to a population
center while allowing true outliers to deviate. All require
`unconstrained=True`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu_prior` | `"none"` | Hierarchical prior on \(\mu\) (or \(r\)) **across mixture components**. Requires `n_components >= 2` |
| `p_prior` | `"none"` | Hierarchical prior on \(p\) (or \(\phi\)) **across genes** |
| `gate_prior` | `"none"` | Hierarchical prior on zero-inflation gate **across genes**. Only for ZI models |

All three accept: `"none"`, `"gaussian"`, `"horseshoe"`, or `"neg"`.

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
    model="nbvcp",
    unconstrained=True,
    n_components=5,
    mu_prior="horseshoe",
)

# Gaussian prior on gene-specific p
results = scribe.fit(
    adata,
    model="nbdm",
    unconstrained=True,
    p_prior="gaussian",
)

# NEG prior on zero-inflation gate
results = scribe.fit(
    adata,
    model="zinb",
    unconstrained=True,
    gate_prior="neg",
)
```

### Hyperparameters

Fine-tune the behavior of Horseshoe and NEG priors:

| Parameter | Default | Prior | Description |
|-----------|---------|-------|-------------|
| `horseshoe_tau0` | `1.0` | Horseshoe | Global shrinkage scale. Smaller = stronger shrinkage |
| `horseshoe_slab_df` | `4` | Horseshoe | Degrees of freedom of the regularizing slab |
| `horseshoe_slab_scale` | `2.0` | Horseshoe | Scale of the regularizing slab |
| `neg_u` | `1.0` | NEG | Shape parameter \(u\) |
| `neg_a` | `1.0` | NEG | Shape parameter \(a\) |
| `neg_tau` | `1.0` | NEG | Scale parameter \(\tau\) |

```python
# Horseshoe with tighter global shrinkage
results = scribe.fit(
    adata,
    model="nbvcp",
    unconstrained=True,
    n_components=4,
    mu_prior="horseshoe",
    horseshoe_tau0=0.5,
    horseshoe_slab_scale=1.0,
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
| `mu_mean_anchor` | `False` | Enable data-informed anchoring. Automatically sets `unconstrained=True` |
| `mu_mean_anchor_sigma` | `0.3` | Log-scale standard deviation. `0.1`--`0.2` = tight, `0.3`--`0.5` = recommended, `> 1` = weak |

For VCP models, SCRIBE needs to estimate the average capture probability
from data. Provide biology-informed capture information via the `priors`
dictionary:

```python
# Mean anchoring with organism-informed capture
results = scribe.fit(
    adata,
    model="nbvcp",
    mu_mean_anchor=True,
    mu_mean_anchor_sigma=0.3,
    priors={"organism": "human"},
    amortize_capture=True,
)

# Mean anchoring with explicit capture efficiency
results = scribe.fit(
    adata,
    model="nbvcp",
    mu_mean_anchor=True,
    priors={"eta_capture": (10.0, 1e5)},
)
```

!!! note
    For non-VCP models (`nbdm`, `zinb`), the anchor uses the implicit
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
| `overdispersion_prior` | `"horseshoe"` | Hierarchical prior on \(\kappa_g\): `"horseshoe"` or `"neg"` |

```python
results = scribe.fit(
    adata,
    model="nbvcp",
    overdispersion="bnb",
    unconstrained=True,
    amortize_capture=True,
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | `None` | Number of mixture components. `None` = single component (no mixture). Must be >= 2 if set |
| `mixture_params` | `None` | Which parameters are component-specific. `None` = all core parameters. Example: `["r"]` makes only \(r\) component-specific |

```python
# Discover 5 cell types
results = scribe.fit(
    adata,
    model="nbvcp",
    n_components=5,
    n_steps=150_000,
    amortize_capture=True,
)

# Extract assignments
assignments = results.cell_type_assignments(counts=adata.X)
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
    model="nbvcp",
    n_components=5,
    annotation_key="cell_type",
    annotation_confidence=3.0,
    amortize_capture=True,
)

# Composite labels from two columns (e.g. cell_type x treatment)
results = scribe.fit(
    adata,
    model="nbvcp",
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
| `dataset_key` | `None` | Column in `adata.obs` identifying which dataset each cell belongs to |
| `n_datasets` | `None` | Number of datasets. Auto-inferred from `dataset_key` when `None` |
| `dataset_params` | `None` | Which parameters become dataset-specific (auto-determined from priors when `None`) |
| `dataset_mixing` | `None` | Dataset-specific mixture weights. `None` = auto (`True` when >= 2 datasets) |
| `auto_downgrade_single_dataset_hierarchy` | `True` | Automatically simplify hierarchy when `dataset_key` resolves to a single dataset |

### Dataset-level priors

Each parameter that varies across genes can also vary across datasets, with a
hierarchical prior controlling how much dataset-to-dataset variation is
allowed.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu_dataset_prior` | `"none"` | Prior on \(\mu\) across datasets: `"none"`, `"gaussian"`, `"horseshoe"`, `"neg"` |
| `p_dataset_prior` | `"none"` | Prior on \(p\) across datasets |
| `p_dataset_mode` | `"gene_specific"` | How \(p\) varies: `"scalar"`, `"gene_specific"`, or `"two_level"` |
| `gate_dataset_prior` | `"none"` | Prior on zero-inflation gate across datasets |
| `overdispersion_dataset_prior` | `"none"` | Prior on BNB \(\kappa\) across datasets. Requires `overdispersion="bnb"` |
| `mu_eta_prior` | `"none"` | Prior on per-dataset capture scaling \(\eta_d\). For VCP models |

```python
# Two-dataset comparison with horseshoe shrinkage on mu
results = scribe.fit(
    adata,
    model="nbvcp",
    unconstrained=True,
    dataset_key="batch",
    mu_dataset_prior="horseshoe",
    p_dataset_prior="gaussian",
    amortize_capture=True,
)
```

!!! info "Single-dataset downgrade"
    When `dataset_key` points to a column with only one unique value, SCRIBE
    automatically downgrades dataset-level priors to gene-level equivalents
    (or drops them) and emits a `UserWarning`. Disable this with
    `auto_downgrade_single_dataset_hierarchy=False`.

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
    model="nbvcp",
    n_components=4,
    priors={"mixing": 5.0},  # equivalent to (5.0, 5.0, 5.0, 5.0)
)

# Biology-informed capture prior
results = scribe.fit(
    adata,
    model="nbvcp",
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
| `model_config` | `None` | `ModelConfig` object. Overrides `model`, `parameterization`, `unconstrained`, `n_components`, `mixture_params`, `guide_rank`, and `priors` |
| `inference_config` | `None` | `InferenceConfig` object. Overrides `inference_method`, `n_steps`, `batch_size`, `stable_update`, `log_progress_lines`, `n_samples`, `n_warmup`, and `n_chains` |

```python
from scribe.models.config import ModelConfigBuilder
from scribe.inference import InferenceConfig

# Build a model config step by step
model_cfg = (
    ModelConfigBuilder()
    .set_model("nbvcp")
    .set_parameterization("mean_odds")
    .set_unconstrained(True)
    .set_mu_prior("horseshoe")
    .build()
)

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
    model="nbvcp",
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
    model="nbvcp",
    unconstrained=True,
    dataset_key="batch",
    mu_dataset_prior="horseshoe",
    p_dataset_prior="gaussian",
    amortize_capture=True,
    n_steps=200_000,
)
```

### Mixture model with annotations

```python
# 8 cell types, guided by partial annotations
results = scribe.fit(
    adata,
    model="nbvcp",
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
    model="nbvcp",
    unconstrained=True,
    mu_mean_anchor=True,
    mu_mean_anchor_sigma=0.3,
    priors={"organism": "human"},
    overdispersion="bnb",
    p_prior="gaussian",
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
    | `model` | `"nbdm"` | `str` |
    | `parameterization` | `"canonical"` | `str` |
    | `unconstrained` | `False` | `bool` |

    **Hierarchical priors (gene-level)**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `mu_prior` | `"none"` | `str` |
    | `p_prior` | `"none"` | `str` |
    | `gate_prior` | `"none"` | `str` |

    **Prior hyperparameters**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `horseshoe_tau0` | `1.0` | `float` |
    | `horseshoe_slab_df` | `4` | `int` |
    | `horseshoe_slab_scale` | `2.0` | `float` |
    | `neg_u` | `1.0` | `float` |
    | `neg_a` | `1.0` | `float` |
    | `neg_tau` | `1.0` | `float` |

    **Mean anchoring**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `mu_mean_anchor` | `False` | `bool` |
    | `mu_mean_anchor_sigma` | `0.3` | `float` |

    **Overdispersion**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `overdispersion` | `"none"` | `str` |
    | `overdispersion_prior` | `"horseshoe"` | `str` |

    **Mixture**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `n_components` | `None` | `int` |
    | `mixture_params` | `None` | `list[str]` |

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
    | `dataset_key` | `None` | `str` |
    | `n_datasets` | `None` | `int` |
    | `dataset_params` | `None` | `list[str]` |
    | `dataset_mixing` | `None` | `bool` |
    | `mu_dataset_prior` | `"none"` | `str` |
    | `p_dataset_prior` | `"none"` | `str` |
    | `p_dataset_mode` | `"gene_specific"` | `str` |
    | `gate_dataset_prior` | `"none"` | `str` |
    | `overdispersion_dataset_prior` | `"none"` | `str` |
    | `mu_eta_prior` | `"none"` | `str` |
    | `auto_downgrade_single_dataset_hierarchy` | `True` | `bool` |

    **Guide (Gaussian)**

    | Parameter | Default | Type |
    |-----------|---------|------|
    | `guide_rank` | `None` | `int` |
    | `joint_params` | `None` | `list[str]` |
    | `dense_params` | `None` | `list[str]` |
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

# SCRIBE SVI (Stochastic Variational Inference)

This directory contains the core implementation of Stochastic Variational
Inference (SVI) for SCRIBE models. SVI is a scalable approximate inference
method that uses optimization to find the best approximation to the posterior
distribution within a chosen variational family.

## Overview

The SVI module provides:

1. **Inference Engine**: Executes SVI optimization using NumPyro's SVI framework
2. **Early Stopping**: Automatic convergence detection to save computation time
3. **Checkpointing**: Orbax-based checkpointing for resumable training
4. **Results Management**: Comprehensive results class with analysis methods
5. **Results Factory**: Streamlined creation and packaging of results objects

For multi-dataset mixture models, post-fit empirical mixing replacement now
computes dataset-specific soft counts and updates mixing parameters per dataset
instead of collapsing all cells into one global correction.

## Key Components

### SVIInferenceEngine (`inference_engine.py`)

The main inference engine that handles SVI execution:

```python
from scribe.svi import SVIInferenceEngine
from scribe.models import ModelConfig

# Configure your model
from scribe.models.config import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("nbdm")
    .with_inference("svi")
    .build())

# Run inference
results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=50000,
    optimizer=numpyro.optim.Adam(step_size=0.001)
)
```

**Key Features:**
- Supports all SCRIBE model types and parameterizations
- Flexible optimizer and loss function selection
- Mini-batch training support for large datasets
- Per-parameter guide family configuration (mean-field, low-rank, amortized)
- Numerically stable parameter updates
- **Early stopping** with automatic convergence detection

**Parameters:**
- `model_config`: Model configuration specifying architecture
- `count_data`: Single-cell count matrix (cells × genes)
- `optimizer`: NumPyro optimizer (default: Adam with lr=0.001)
- `loss`: ELBO loss function (default: TraceMeanField_ELBO)
- `n_steps`: Maximum number of optimization steps
- `batch_size`: Mini-batch size for stochastic optimization
- `seed`: Random seed for reproducibility
- `early_stopping`: Optional `EarlyStoppingConfig` for convergence detection
- `restore_best`: If True, track and restore best (lowest smoothed loss)
  parameters at end of training, independently of early stopping

### Best-Params Restoration

The `restore_best` parameter (available at the `SVIConfig` level and as a
top-level `fit()` argument) tracks the variational parameters that achieved
the lowest smoothed loss during training and restores them at the end.  This
works **independently of early stopping** — when `restore_best=True` and no
`early_stopping` config is provided, a minimal internal training loop is
created to enable best-state tracking.

```python
# Via scribe.fit()
results = scribe.fit(counts, restore_best=True)

# Via Hydra (conf/inference/svi.yaml)
# inference.restore_best: true
```

### Early Stopping

The inference engine supports automatic early stopping when the loss converges.
This saves computation time by stopping training when no further improvement is
detected.

```python
from scribe.models.config import EarlyStoppingConfig

# Configure early stopping
early_stopping = EarlyStoppingConfig(
    enabled=True,       # Enable early stopping
    patience=500,       # Steps without improvement before stopping
    min_delta=1.0,      # Minimum improvement to count as progress
    check_every=10,     # Check convergence every N steps
    smoothing_window=50,# Window size for loss smoothing
    restore_best=True,  # Restore best parameters when stopping
)

# Run inference with early stopping
results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=100000,     # Maximum steps (may stop earlier)
    early_stopping=early_stopping,
)

# Check if early stopping triggered
if results.early_stopped:
    print(f"Stopped early at step {results.stopped_at_step}")
    print(f"Best loss: {results.best_loss}")
```

**Early Stopping Parameters:**
- `patience`: Number of steps to wait for improvement before stopping
- `min_delta`: Minimum change in smoothed loss to qualify as improvement
  (default: 1.0, suitable for ELBO values ~10^6-10^7)
- `check_every`: How often to check for convergence (reduces overhead)
- `smoothing_window`: Window size for computing moving average loss
- `restore_best`: If True, restores parameters from the best checkpoint
- `checkpoint_dir`: Directory for Orbax checkpoints (enables resumable training)
- `resume`: If True (default), resumes from checkpoint if one exists

### Checkpointing (`checkpoint.py`)

The SVI module supports Orbax-based checkpointing for resumable training. When
enabled, the best parameters are saved to disk whenever loss improves, allowing
training to resume from the last checkpoint if interrupted.

**Via Hydra (automatic):**

When using `infer.py`, checkpoints are automatically saved to the Hydra output
directory:

```bash
# Start training
python infer.py data=singer model=nbdm inference.n_steps=100000
# Checkpoints saved to: outputs/<date>/<time>/checkpoints/

# If interrupted, re-running resumes automatically
python infer.py data=singer model=nbdm inference.n_steps=100000
```

**Direct API (manual checkpoint directory):**

```python
from scribe.models.config import EarlyStoppingConfig

# Enable checkpointing with explicit directory
early_stopping = EarlyStoppingConfig(
    patience=500,
    checkpoint_dir="./my_checkpoints",  # Enable checkpointing
    resume=True,  # Resume if checkpoint exists (default)
)

results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=100000,
    early_stopping=early_stopping,
)
```

**Checkpoint utilities:**

```python
from scribe.svi import (
    checkpoint_exists,
    load_svi_checkpoint,
    save_svi_checkpoint,
    remove_checkpoint,
)

# Check if checkpoint exists
if checkpoint_exists("./my_checkpoints"):
    params, metadata, losses = load_svi_checkpoint("./my_checkpoints")
    print(f"Resumed from step {metadata.step}, best_loss={metadata.best_loss}")
```

### ScribeSVIResults (`results.py`)

Comprehensive results class that stores inference outcomes and provides analysis
methods:

```python
from scribe.svi import ScribeSVIResults

# Results are typically created by the inference engine
# but can also be constructed manually or from AnnData
results = ScribeSVIResults.from_anndata(
    adata=adata,
    params=svi_params,
    loss_history=losses,
    model_type="nbdm",
    model_config=config
)
```

#### Core Attributes

- **`params`**: Dictionary of optimized variational parameters
- **`loss_history`**: ELBO loss trajectory during optimization
- **`model_config`**: Model configuration used for inference
- **`n_cells`**, **`n_genes`**: Dataset dimensions
- **`_n_cells_per_dataset`**: Optional 1-D array of per-dataset cell counts
  (set automatically during inference for multi-dataset hierarchical models).
  When present, `get_dataset(d)` uses this to set the correct `n_cells` on the
  returned single-dataset view so that downstream PPC generation works with
  the right number of cells.
- **`_label_map`**: Optional mapping from annotation label string to component
  index, populated when fitting with `annotation_key`.
- **`_component_mapping`**: Optional multi-dataset component metadata
  (shared/exclusive structure) populated when fitting with
  `annotation_key + dataset_key`.
- **`obs`**, **`var`**, **`uns`**: Metadata from AnnData objects

#### Key Analysis Methods

**Parameter Access:**
```python
# Get MAP (maximum a posteriori) estimates
map_params = results.get_map()
# Canonical/standard MAP also includes deterministic mu:
# mu = r * p / (1 - p)

# Get only one MAP key (avoids unrelated expensive flow MAP work)
p_capture_map = results.get_map("p_capture")

# Get a subset of MAP keys
rp_map = results.get_map(["r", "p"])

# Get posterior distributions
distributions = results.get_distributions()
```

**Flow Guide MAP Estimation:**

When using normalizing flow guides (`guide_flow` parameter in `scribe.fit()` or
`guide_flow` in the Hydra config), MAP estimation requires sampling-based
strategies because flow transformations do not preserve modes. The `get_map()`
method accepts additional parameters for controlling flow MAP estimation:

Flow conditioner MLP activation can be configured with
`guide_flow_activation` (default: `"relu"`). Supported values:
`"relu"`, `"gelu"`, `"silu"`/`"swish"`, `"tanh"`, `"elu"`,
`"leaky_relu"`, `"softplus"`.

Hydra CLI example:

```bash
python infer.py data=singer model=nbdm guide_flow=spline_coupling guide_flow_activation=gelu
```

Python API example:

```python
results = scribe.fit(
    counts=data,
    model="nbdm",
    guide_flow="spline_coupling",
    guide_flow_activation="gelu",
)
```

```python
# Default: posterior mean via Monte Carlo sampling
map_params = results.get_map(flow_map_method="mean", flow_n_samples=1000)

# Empirical mode: pick the sample with highest log-density
map_params = results.get_map(flow_map_method="empirical", flow_n_samples=2000)

# Gradient-based mode finding: most accurate, starts from mean
map_params = results.get_map(
    flow_map_method="optimize",
    flow_n_samples=500,          # samples for initial mean estimate
    flow_optimize_steps=300,     # Adam steps for gradient ascent
    flow_optimize_lr=1e-3,       # learning rate
)

# For models with cell-level parameters (e.g. amortized capture),
# use flow_batch_size to avoid processing all cells at once
map_params = results.get_map(
    flow_map_method="mean",
    flow_batch_size=2048,        # mini-batch cells per guide call
)
```

Strategy summary:
- `"mean"` (default): Fast; gives the posterior mean. Best for most use cases.
- `"empirical"`: Medium cost; picks the highest-density sample as an approximate
  mode.
- `"optimize"`: Highest quality mode estimate via gradient ascent on the guide's
  log-density, initialized from the sample mean.

Non-flow parameters are unaffected by these settings and continue to use the
standard `transform(base.loc)` approach.

Selective extraction composes with flow settings: if you request only
non-flow targets, flow MAP optimization/sampling for unrelated parameters is
skipped automatically.

**Flow Guide Posterior Distributions:**

`get_distributions()` returns `FlowDistribution`-based entries for flow-guided
parameters. These are wrapped in the standard `{"base": ..., "transform": ...}`
dict format. Joint flow entries additionally include `"conditional": True` to
indicate that they represent conditional distributions (from the chain-rule
decomposition) and require guide execution for proper joint sampling.

Nondense (non-flow) parameters in a joint flow group are also marked
`"conditional": True`. Their stored `loc` is only a baseline; at inference time
the actual loc includes regression on the dense flow residuals
(`loc + alpha_r * r_residual`). Because the regression coefficients are only
evaluated inside the guide, `get_map` routes these parameters through the
sampling-based flow MAP path rather than reading the static `loc` directly.
This ensures denoised counts and biological PPC plots use the correctly
conditioned parameter values.

**Mixture models and flow guides:**

For mixture-aware flow parameters, `get_distributions()` returns
`ComponentFlowDistribution`-based entries (instead of plain `FlowDistribution`)
so the full component structure is preserved. Use `get_component(k)` to obtain a
single-component view and read per-component flow posteriors there. Sampling-
based `get_map()` strategies (`flow_map_method` `"mean"`, `"empirical"`, or
`"optimize"`) automatically handle `(K, G)` shapes alongside multi-dataset
layouts. Which flow parameters are shared vs. per-component is set at fit time
via `guide_flow_mixture_strategy`: `"independent"` (separate flows per mixture
component) or `"shared"` (one flow structure across components).

**Posterior Sampling:**
```python
# Sample from variational posterior
posterior_samples = results.get_posterior_samples(
    n_samples=1000,
    seed=42
)

# For large models where downstream operations (e.g., DE comparison)
# need GPU headroom, store posterior samples as CPU-resident JAX
# arrays.  The arrays remain jax.Array instances so jnp/vmap/NumPyro
# code continues to work transparently.
posterior_samples = results.get_posterior_samples(
    n_samples=10_000,
    batch_size=512,
    store_on_cpu=True,
)

# Generate predictive samples (uses stored posterior samples)
predictive_samples = results.get_predictive_samples()

# Posterior predictive checks
ppc_samples = results.get_ppc_samples(
    n_samples=100,
    seed=42
)

# For models with amortized capture probability, pass observed counts
# (required when using amortization.capture.enabled=true)
ppc_samples = results.get_ppc_samples(
    n_samples=100,
    seed=42,
    counts=observed_counts  # Shape: (n_cells, n_genes)
)
```

**Biological (Denoised) Posterior Predictive Checks:**

Standard PPCs include technical noise parameters (capture probability, zero-
inflation gate) when generating synthetic counts. Biological PPCs strip these
parameters and sample from the base Negative Binomial NB(r, p) only, reflecting
the underlying biology. See the Dirichlet-Multinomial derivation in the paper
supplement for the mathematical justification.

```python
# Full-posterior biological PPC (strips p_capture, gate, etc.)
bio_ppc = results.get_ppc_samples_biological(
    rng_key=rng_key,
    n_samples=100,
)
bio_counts = bio_ppc["predictive_samples"]  # (n_samples, n_cells, n_genes)

# MAP-based biological PPC (memory-efficient, supports cell batching)
bio_map_counts = results.get_map_ppc_samples_biological(
    rng_key=rng_key,
    n_samples=5,
    cell_batch_size=2048,  # Process cells in batches
)
```

For NBDM models (which have no technical parameters), biological PPCs produce
the same result as standard PPCs. For VCP and ZINB models, the biological PPC
gives a denoised view of the data by bypassing the capture probability
transformation and zero-inflation gate.

**Bayesian Denoising of Observed Counts:**

Unlike biological PPCs (which sample synthetic counts from the prior NB),
Bayesian denoising takes the *observed* count matrix and computes the posterior
distribution of the true (pre-capture, pre-dropout) transcript counts.  The
derivation exploits Poisson-Gamma conjugacy and the Poisson thinning property
(see `paper/_denoising.qmd`).

```python
# MAP-based denoising (single point estimate, fast)
denoised = results.denoise_counts_map(
    counts=observed_counts,
    method="mean",      # "mean" (default), "mode", or "sample"
    rng_key=rng_key,
)
# denoised.shape == (n_cells, n_genes)

# Full-posterior Bayesian denoising (propagates parameter uncertainty)
denoised_post = results.denoise_counts_posterior(
    counts=observed_counts,
    method="mean",
    rng_key=rng_key,
    n_samples=100,
)
# denoised_post.shape == (100, n_cells, n_genes)
# Bayesian point estimate:
denoised_avg = denoised_post.mean(axis=0)

# With variance (returns dict instead of array)
result = results.denoise_counts_map(
    counts=observed_counts,
    return_variance=True,
    rng_key=rng_key,
)
# result["denoised_counts"].shape == (n_cells, n_genes)
# result["variance"].shape == (n_cells, n_genes)
```

For NBDM models the denoised counts equal the observed counts (identity).
For VCP models the per-cell capture probability inflates counts to recover
the pre-capture expression level.  For ZINB models, zero observations are
additionally corrected for technical dropout using the gate posterior.

**Tuple Method for Independent Control of ZINB Zeros:**

The `method` parameter accepts a tuple `(general_method, zi_zero_method)` for
independent control of the denoising method at non-zero positions vs. zero
positions in ZINB models.  A single string like `"mean"` is equivalent to
`("mean", "mean")`.

```python
# Mean for non-zeros, sample replacement for ZINB dropout zeros
denoised = results.denoise_counts_map(
    counts=observed_counts,
    method=("mean", "sample"),
    rng_key=rng_key,
)
```

When `zi_zero_method="sample"`, the gate weight `w` (posterior probability that
the zero came from dropout) is used as a Bernoulli probability: if the draw says
"dropout", the zero is replaced with a sample from the biological prior NB(r, p);
otherwise the zero is kept as a genuine biological zero.

**Exporting Denoised Counts as AnnData / h5ad:**

The `get_denoised_anndata()` method combines denoising with AnnData packaging,
optionally writing the result to an h5ad file.  It copies cell/gene metadata
from the original data and records denoising provenance in `.uns`.

```python
# Single denoised dataset (default: MAP estimates)
adata_denoised = results.get_denoised_anndata(
    counts=observed_counts,
    rng_key=rng_key,
)
# adata_denoised.X           → denoised counts
# adata_denoised.layers["original_counts"] → input counts
# adata_denoised.uns["scribe_denoising"]   → provenance metadata

# Multiple datasets — first uses MAP, rest use posterior draws
adatas = results.get_denoised_anndata(
    counts=observed_counts,
    rng_key=rng_key,
    n_datasets=5,
)

# Opt in: all datasets use posterior draws (preserves cross-gene correlations)
adatas = results.get_denoised_anndata(
    counts=observed_counts,
    rng_key=rng_key,
    n_datasets=5,
    preserve_correlations=True,
)
# All adatas use posterior draws

# Pass an AnnData template to copy obs/var metadata
adata_denoised = results.get_denoised_anndata(
    adata=original_adata,      # extracts counts + copies metadata
    rng_key=rng_key,
    path="denoised.h5ad",      # write directly to disk
)
```

**Preserving Cross-Gene Correlations (`preserve_correlations`):**

By default, `preserve_correlations=False`: the first denoised dataset uses MAP
point estimates and subsequent datasets use posterior draws.  Setting
`preserve_correlations=True` makes **all** denoised datasets (including the
first) use posterior parameter draws, which propagates cross-gene correlations
encoded in the joint parameter posterior into the denoised counts.  For models
fitted with a joint low-rank guide (`joint_params` in the model config), these
draws capture both within- and across-parameter correlations.  See
`paper/_denoising.qmd` §"Cross-gene correlations in denoised counts" for the
mathematical derivation.

**Multi-dataset / concatenated results:** When results from independent fits
are combined via `ScribeSVIResults.concat`, `get_posterior_samples` (and by
extension `preserve_correlations=True`) automatically decomposes the sampling
into per-dataset calls via `get_dataset(i)`, draws posterior samples from each
single-dataset guide independently, and re-stacks the results.  This is
transparent — no extra arguments are needed.  For jointly hierarchical models
(fit with `hierarchical_dataset_*` or `horseshoe_dataset` flags), the standard
`Predictive` path is used directly since the guide was built for the
multi-dataset structure.

When `annotation_key` is used during fitting, `get_dataset(i)` also preserves
the fit-time `_label_map` and `_component_mapping` metadata. This guarantees
that per-dataset visualization and component lookups reuse the original
label-to-component assignment rather than reconstructing a dataset-local order.

**Amortized Capture Probability and PPC:**

When using amortized capture probability (enabled via
`amortization.capture.enabled=true`), the guide function uses a neural network
that takes total UMI counts as input to predict capture probabilities. During
posterior predictive checks (PPC), you must provide the observed counts so the
amortizer can compute the necessary sufficient statistics.

```python
# For models with amortized capture, counts are required for PPC
ppc_samples = results.get_ppc_samples(
    rng_key=rng_key,
    n_samples=100,
    counts=observed_counts  # Required for amortized capture
)

# For non-amortized models, counts can be omitted
ppc_samples = results.get_ppc_samples(
    rng_key=rng_key,
    n_samples=100
    # counts parameter not needed
)
```

**MAP PPC shape guarantees:** `get_map_ppc_samples()` always returns
`(n_samples, n_cells, n_genes_subset)` with an explicit gene axis, including
singleton subsets (`n_genes_subset=1`) after pipelines such as
`results.get_component(k)[gene_idx]`. This keeps annotation-PPC and targeted
gene workflows stable even when upstream MAP parameter tensors are scalar-like.

**MAP PPC batching semantics:** `cell_batch_size` controls only memory/throughput
trade-offs in cell-wise sampling. It does not change output shape contracts or
the selected gene/component semantics.

**Note:** The `counts` parameter should be the same observed data used during
inference. For non-amortized models (standard VCP models or non-VCP models), the
`counts` parameter is optional and can be omitted.

**Model Evaluation:**
```python
# Compute log-likelihood for model comparison
log_lik = results.log_likelihood()

# MAP-based log-likelihood (faster)
log_lik_map = results.log_likelihood_map()
```

**Mixture Model Analysis:**
```python
# For mixture models, analyze component assignments
component_entropy = results.mixture_component_entropy()
assignment_entropy = results.assignment_entropy_map()
cell_type_probs = results.cell_type_probabilities()

# Access single or multiple mixture components
component_0 = results.get_component(0)
components_12 = results.get_components([1, 2])  # renormalize=True by default
components_12_raw = results.get_components([1, 2], renormalize=False)
```

**Empirical Mixing Weights (auto-applied for mixture models):**

SVI-learned Dirichlet mixing weights are practically non-identifiable in high-
dimensional mixture models because per-gene log-likelihoods overwhelm the
mixing-weight contribution.  The factory automatically replaces them with data-
driven weights from the conditional posterior `Dir(alpha_0 + N_soft)`.

```python
# Mixing weights are already corrected — no flags needed
map_params = results.get_map()
print(map_params["mixing_weights"])  # empirical weights

# Original SVI-learned params are stashed for diagnostics
print(results._svi_mixing_params)         # original concentrations
print(results._mixing_weights_replaced)   # True

# You can also compute empirical weights explicitly
emp = results.compute_empirical_mixing_weights(counts=count_data)
print(emp["weights"])          # posterior mean
print(emp["concentrations"])   # full Dirichlet posterior alpha_0 + N_soft
print(emp["effective_counts"]) # soft cell counts per component

# Re-apply if needed (e.g. after modifying params)
results.apply_empirical_mixing_weights(counts=count_data)
```

**Data Subsetting:**
```python
# Subset by genes
gene_subset = results[:100]  # First 100 genes

# Two-axis indexing: results[genes, components]
gene_component_subset = results[1:4, [1, 2]]
```

**Concatenating compatible results (cell axis):**
```python
# Concatenate SVI results that share model_config/model_type and genes.
# This stacks cell-specific parameters (e.g. p_capture) across objects.
combined = ScribeSVIResults.concat([results_a, results_b])

# Faster path when you trust the fits and only need gene validation:
combined_fast = ScribeSVIResults.concat(
    [results_a, results_b],
    validation="var_only",  # skips deep equality on shared non-cell tensors
)

# Fastest trusted mode: skip gene-set/order validation too.
combined_trusted = ScribeSVIResults.concat(
    [results_a, results_b],
    validation="var_only",
    align_genes="assume_aligned",
)

# Gene order is validated via var.index when available; if order differs
# but content matches, concat reorders to the first object's gene order.
```

Concatenation constraints:
- All inputs must have matching `model_type`, `model_config`, and `prior_params`.
- Non-cell-specific parameters must be identical across inputs.
- Cell counts may differ; concat joins along the cell axis.
- If `_dataset_indices` and `_n_cells_per_dataset` are present, they are merged
  so `get_dataset(d)` keeps working on the concatenated object.
- **Dataset promotion**: when all inputs are single-dataset (no existing dataset
  metadata), concatenating two or more results automatically synthesizes
  `_n_cells_per_dataset`, `_dataset_indices`, and sets
  `model_config.n_datasets`, so `get_dataset(i)` and 3-axis indexing
  (`combined[:, :, i]`) work immediately on the concatenated result.
  Non-cell-specific parameters (e.g. gene-level `mu_loc`) are stacked along a
  new dataset axis so that `get_dataset(i)` recovers each input's original
  parameter values. At least two results are required (single-element lists
  are rejected to prevent the `res.concat([other])` classmethod footgun).

#### Advanced Features

**Normalization:**
- Automatic count normalization using posterior estimates
- Integration with SCRIBE's normalization framework

```python
# Deterministic MAP-based normalization (no posterior sampling)
norm_map = results.normalize_counts_map(
    estimator="mean",   # "mean" (recommended) or "mode" (shared-p only)
    use_mean=True,      # replace undefined MAP values with posterior means
    counts=observed_counts,  # required for amortized capture models
)
rho_hat = norm_map["mean_probabilities"]  # (n_genes,) or (n_components, n_genes)

# Posterior-sampling normalization (captures parameter uncertainty)
norm_post = results.normalize_counts(
    n_samples_dirichlet=1,
    batch_size=2048,
)
```

`normalize_counts_map(estimator="mean")` is parameterization-aware:

- **Convention note**: canonical SCRIBE extraction uses
  `mu = r * p / (1 - p)` for `(r, p)` maps. The linked inverse remains
  `r = mu * (1 - p) / p`. This code-level `p` convention is the one used
  throughout training and post-processing.
- If MAP includes `mu` (for example with `mean_prob` / `mean_odds`, and
  canonical/standard after deterministic MAP conversion),
  it uses `rho = mu / sum(mu)` directly.
- If gene-specific `p_g` or `phi_g` is detected, it applies the hierarchical
  scaling implied by the model parameterization:
  - `mu = r * p / (1 - p)` or equivalently `mu = r / phi`,
  - then `rho = mu / sum(mu)`.
- Otherwise (shared-p Dirichlet case), it uses `rho = r / sum(r)`.

`estimator="mode"` uses the Dirichlet mode
`rho = (r - 1) / (sum(r) - G)` and is only valid in shared-p Dirichlet
settings where all `r_g > 1`. For gene-specific `p_g`/`phi_g` cases, use
`estimator="mean"`.

**Statistical Analysis:**
- Hellinger and Jensen-Shannon divergence calculations
- KL divergence between distributions
- Dirichlet parameter fitting using Minka's method

**Data Integration:**
- Seamless integration with AnnData objects
- Preservation of cell and gene metadata
- Support for unstructured annotations

#### Mixin Architecture

The `ScribeSVIResults` class is implemented using a mixin-based architecture to
improve maintainability and code organization. The class inherits from 9
specialized mixins, each handling a specific domain of functionality. This
design allows for better code organization, easier maintenance, and clearer
separation of concerns.

**Architecture Overview:**

```
ScribeSVIResults
├── CoreResultsMixin          # Basic initialization
├── ModelHelpersMixin         # Model/guide access
├── ParameterExtractionMixin  # Parameter extraction
├── GeneSubsettingMixin       # Gene-based indexing
├── ComponentMixin            # Mixture component operations
├── SamplingMixin              # Posterior/predictive sampling
├── LikelihoodMixin            # Log-likelihood computation
├── MixtureAnalysisMixin       # Mixture model analysis
└── NormalizationMixin         # Count normalization
```

**Mixin Details:**

1. **CoreResultsMixin** (`_core.py`)
   - Purpose: Basic initialization and construction
   - Methods:
     - `__post_init__()`: Validates model configuration and sets n_components
     - `from_anndata()`: Classmethod to create results from AnnData objects
   - Dependencies: None (base functionality)

2. **ModelHelpersMixin** (`_model_helpers.py`)
   - Purpose: Internal helpers for model/guide access
   - Methods:
     - `_model_and_guide()`: Returns model and guide functions
     - `_parameterization()`: Returns parameterization type string
     - `_unconstrained()`: Returns whether parameterization is unconstrained
     - `_log_likelihood_fn()`: Returns log-likelihood function for model type
   - Dependencies: None (simple accessors)

3. **ParameterExtractionMixin** (`_parameter_extraction.py`)
   - Purpose: Extract parameters from variational distributions
   - Methods:
     - `get_distributions()`: Get posterior distributions (NumPyro or SciPy)
     - `get_map()`: Get maximum a posteriori (MAP) estimates
     - `_compute_canonical_parameters()`: Convert to canonical form and derive
       missing deterministic quantities (for canonical/standard MAP, includes
       `mu = r * p / (1 - p)` with shape-aware broadcasting across component,
       dataset, and gene axes)
     - `_convert_to_canonical()`: Deprecated conversion method
     - `_reconstruct_ncp_maps()`: Iterate over `model_config.param_specs` and
       dispatch to per-spec helpers for all NCP parameters (horseshoe and NEG).
       Replaces the old `_reconstruct_horseshoe_maps` / `_reconstruct_neg_maps`.
     - `_reconstruct_from_horseshoe_spec()`: Reconstruct one constrained MAP
       parameter from its horseshoe NCP `ParamSpec` fields (`raw_name`,
       `tau_name`, `lambda_name`, `c_sq_name`, `hyper_loc_name`, `transform`).
     - `_reconstruct_from_neg_spec()`: Reconstruct one constrained MAP
       parameter from its NEG NCP `ParamSpec` fields (`raw_name`, `psi_name`,
       `hyper_loc_name`, `transform`).
     - `_horseshoe_eff_scale()`: Compute regularized horseshoe effective scale
     - `_neg_eff_scale()`: Compute NEG effective scale as sqrt(psi)
   - Dependencies: Uses `ModelHelpersMixin` methods

4. **GeneSubsettingMixin** (`_gene_subsetting.py`)
   - Purpose: Subset results by gene indices
   - Methods:
     - `__getitem__()`: Enable indexing `results[:, genes]`
     - `_subset_params()`: Subset parameter dictionary by genes
     - `_subset_posterior_samples()`: Subset posterior samples by genes
     - `_subset_predictive_samples()`: Subset predictive samples by genes
     - `_create_subset()`: Create new instance with gene subset
     - `_subset_gene_params()`: Static helper for gene parameter subsetting
   - Notes:
     - Metadata-based axis detection recognizes both legacy variational keys
       (for example, `mu_loc`) and joint-guide keys
       (for example, `joint_joint_mu_loc`), so gene indexing remains correct
       under `joint_params` even when shapes are ambiguous.
     - **Flow guide awareness**: when flow params are detected (keys ending
       with `$params` prefixed by `flow_` or `joint_flow_`), `_create_subset`
       stores the original unsubsetted params dict as `_original_params`.
       This is critical for joint flow guides with nondense regression: the
       flow chain produces full-gene-dimensional output while the nondense
       array params (e.g. `joint_flow_joint_p_alpha_r`) would otherwise be
       sliced to the gene subset, causing a broadcasting mismatch.
       `_get_posterior_samples_standard` and `get_map` use `_original_params`
       when sampling at the original full dimensionality.
   - Dependencies: None (self-contained)

5. **ComponentMixin** (`_component.py`)
   - Purpose: Mixture model component operations
   - Methods:
     - `get_component()`: Extract single component view
     - `_subset_params_by_component()`: Subset params by component index
     - `_subset_posterior_samples_by_component()`: Subset samples by component
     - `_create_component_subset()`: Create component-specific instance
   - Notes:
     - Component extraction supports both standard per-parameter variational
       keys (for example, `phi_loc`) and joint low-rank guide keys (for
       example, `joint_joint_phi_loc`), so mixture-specific parameters are
       correctly reduced to a single component before posterior sampling.
   - Dependencies: Uses `GeneSubsettingMixin` for subsetting logic

6. **SamplingMixin** (`_sampling.py`)
   - Purpose: Posterior and predictive sampling
   - Methods:
     - `get_posterior_samples()`: Sample from variational posterior
       (supports `store_on_cpu=True` to keep arrays on CPU host memory)
     - `get_predictive_samples()`: Generate predictive samples
     - `get_ppc_samples()`: Posterior predictive check samples
     - `get_map_ppc_samples()`: MAP-based predictive samples with batching
     - `get_ppc_samples_biological()`: Biological (denoised) PPC — strips
       technical noise and samples from base NB(r, p) only
     - `get_map_ppc_samples_biological()`: MAP-based biological PPC with
       cell batching support
     - `denoise_counts_map()`: MAP-based Bayesian denoising of observed
       counts (posterior mean/mode/sample of true transcripts)
     - `denoise_counts_posterior()`: Full-posterior Bayesian denoising
       (propagates parameter uncertainty across posterior draws)
     - `get_denoised_anndata()`: Export denoised counts as AnnData/h5ad
       with metadata, supporting tuple method and multi-dataset generation
   - Dependencies: Uses `ModelHelpersMixin`, `ParameterExtractionMixin`

7. **LikelihoodMixin** (`_likelihood.py`)
   - Purpose: Log-likelihood computations
   - Methods:
     - `log_likelihood()`: Compute using posterior samples
     - `log_likelihood_map()`: Compute using MAP estimates
   - Dependencies: Uses `ModelHelpersMixin`, `ParameterExtractionMixin`

8. **MixtureAnalysisMixin** (`_mixture_analysis.py`)
   - Purpose: Mixture model analysis and empirical mixing weight correction
   - Methods:
     - `mixture_component_entropy()`: Entropy of component assignments
     - `assignment_entropy_map()`: MAP-based assignment entropy
     - `cell_type_probabilities()`: Component probabilities from samples
     - `cell_type_probabilities_map()`: Component probabilities from MAP
     - `compute_empirical_mixing_weights()`: Data-driven mixing weights
       via the conditional posterior `Dir(alpha_0 + N_soft)`
     - `apply_empirical_mixing_weights()`: Replace SVI-learned mixing
       params in `self.params` with empirical values (called automatically
       by the factory for mixture models)
   - Dependencies: Uses `LikelihoodMixin`, `ParameterExtractionMixin`

9. **NormalizationMixin** (`_normalization.py`)
   - Purpose: Count normalization methods
   - Methods:
     - `normalize_counts(batch_size=2048)`: Dirichlet-based normalization
     - `normalize_counts_map(estimator="mean")`: Deterministic MAP-based
       normalization with automatic parameterization-aware behavior
     - `fit_logistic_normal(batch_size=2048, svd_method="randomized")`:
       Logistic-Normal distribution fitting
   - Both methods use **batched Dirichlet sampling** to process posterior
     samples in configurable chunks, reducing Python-to-JAX dispatches from
     O(N) to O(ceil(N/batch_size)) for efficient GPU utilisation.
   - `fit_logistic_normal` uses **randomized SVD** by default for the
     low-rank covariance fit (O(NDk) instead of O(N^2 D)).  Pass
     `svd_method="full"` for the complete eigenvalue spectrum.
   - Dependencies: Uses `ParameterExtractionMixin` (for canonical conversion)

**Benefits of Mixin Architecture:**

- **Maintainability**: Each mixin is focused on a single responsibility (~200-500 lines)
- **Readability**: Easier to find and understand specific functionality
- **Testability**: Can test mixin functionality in isolation if needed
- **Extensibility**: Easy to add new functionality by creating new mixins
- **No Breaking Changes**: Public API remains identical - all methods accessible on `ScribeSVIResults`

**File Organization:**

```
src/scribe/svi/
├── results.py              # Main class (composed of mixins, ~108 lines)
├── _core.py                # CoreResultsMixin
├── _parameter_extraction.py # ParameterExtractionMixin
├── _gene_subsetting.py     # GeneSubsettingMixin
├── _component.py           # ComponentMixin
├── _model_helpers.py       # ModelHelpersMixin
├── _sampling.py            # SamplingMixin
├── _likelihood.py          # LikelihoodMixin
├── _mixture_analysis.py    # MixtureAnalysisMixin
└── _normalization.py       # NormalizationMixin
```

**Note:** Mixin files use the `_` prefix to indicate they are internal
implementation details. Users should interact with `ScribeSVIResults` directly;
the mixin structure is transparent to the public API.

### SVIResultsFactory (`results_factory.py`)

Factory class for creating and packaging SVI results:

```python
from scribe.svi import SVIResultsFactory

# Package raw SVI results into ScribeSVIResults
results = SVIResultsFactory.create_results(
    svi_results=raw_svi_results,
    model_config=config,
    adata=adata,  # Optional
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    model_type="nbdm",
    n_components=None,
    prior_params=priors
)
```

## Usage Examples

### Basic SVI Inference

```python
import jax.numpy as jnp
from scribe.models import ModelConfig
from scribe.svi import SVIInferenceEngine

# Prepare data
count_data = jnp.array(your_count_matrix)  # cells × genes
n_cells, n_genes = count_data.shape

# Configure model
from scribe.models.config import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("zinb")
    .with_parameterization("linked")
    .unconstrained()
    .build())

# Run inference
results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=count_data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=75000,
    batch_size=512  # For large datasets
)
```

### Mixture Model Analysis

```python
# Configure mixture model
from scribe.models.config import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("nbdm")
    .as_mixture(n_components=3)
    .build())

# Run inference
results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=count_data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=100000
)

# Analyze mixture components
for i in range(3):
    component = results.get_component(i)
    print(f"Component {i} parameters:", component.get_map())

# Multi-component selection with optional renormalization
components = results.get_components([1, 2])  # renormalize=True default
components_no_renorm = results.get_components([1, 2], renormalize=False)

# Tuple indexing: first genes, second components
subset = results[1:4, [1, 2]]

# Get cell type assignments
cell_probs = results.cell_type_probabilities()
```

### Model Comparison

```python
# Compare different models using log-likelihood
models = ["nbdm", "zinb", "nbvcp"]
log_likelihoods = {}

for model_type in models:
    config = (ModelConfigBuilder()
        .for_model(model_type)
        .build())
    results = SVIInferenceEngine.run_inference(
        model_config=config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_steps=50000
    )
    log_likelihoods[model_type] = results.log_likelihood_map()

best_model = max(log_likelihoods, key=log_likelihoods.get)
```

### Working with AnnData

```python
import scanpy as sc
from scribe.svi import SVIInferenceEngine

# Load data with scanpy
adata = sc.read_h5ad("data.h5ad")

# Extract count data
count_data = jnp.array(adata.X.toarray())

# Run inference
from scribe.models.config import ModelConfigBuilder

config = (ModelConfigBuilder()
    .for_model("zinb")
    .build())
results = SVIInferenceEngine.run_inference(
    model_config=config,
    count_data=count_data,
    n_cells=adata.n_obs,
    n_genes=adata.n_vars,
    n_steps=50000
)

# Create results with metadata
from scribe.svi import SVIResultsFactory
final_results = SVIResultsFactory.create_results(
    svi_results=results,
    model_config=config,
    adata=adata,  # Preserves obs, var, uns
    count_data=count_data,
    n_cells=adata.n_obs,
    n_genes=adata.n_vars,
    model_type="zinb",
    n_components=None,
    prior_params={}
)
```

## Optimization Tips

### Convergence
- **Use early stopping** to automatically detect convergence and save time
- Monitor loss history: `results.loss_history`
- Use longer training for complex models (100k+ steps)
- Try different optimizers: Adam, AdamW, RMSprop
- Adjust learning rates based on loss behavior
- Tune `min_delta` based on your typical ELBO scale

### Memory Management
- Use mini-batches for large datasets (`batch_size` parameter)
- Consider unconstrained parameterizations for better optimization
- Monitor memory usage with JAX profiling tools

### Numerical Stability
- Enable `stable_update=True` (default)
- Use unconstrained parameterizations when convergence is difficult
- Check for NaN values in loss history
- Progress-bar rolling mean loss ignores NaN/Inf values to keep reporting stable

## Integration with Other Modules

- **Models**: Automatically retrieves model/guide functions via registry
- **Sampling**: Provides posterior and predictive sampling capabilities  
- **Stats**: Integrates statistical analysis and divergence measures
- **Core**: Uses normalization and preprocessing utilities
- **Viz**: Results can be visualized using the visualization module

## Dependencies

- **NumPyro**: Core SVI implementation and probabilistic programming
- **JAX**: Automatic differentiation and compilation
- **Pandas**: Metadata handling and data structures
- **AnnData**: Single-cell data format integration (optional)
- **SciPy**: Statistical functions and distributions

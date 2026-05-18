# Results Class

The SCRIBE package provides a unified `ScribeResults` class that works
consistently across all model types. This class encapsulates all functionality
for handling model inference outputs, parameter access, and downstream analysis.

## Base Structure

The `ScribeResults` class provides core functionality for:

- Accessing variational model parameters and posterior distributions
- Indexing results by gene (single gene, ranges, boolean indexing)
- Selecting specific mixture components in mixture models
- Generating posterior samples and predictive samples
- Computing log likelihoods and model comparisons
- Handling metadata from `AnnData` objects

## Basic Usage

After running inference with `scribe.fit()`, you'll get a results object:

```python
import scribe

results = scribe.fit(adata, n_steps=10_000)
```

The results object contains several key attributes:

- `params` — dictionary of learned variational parameters
- `loss_history` — array of ELBO values during training
- `n_cells`, `n_genes` — dataset dimensions
- `model_type` — string indicating the type of model
- `model_config` — configuration object with model architecture and priors
- `n_components` — number of components in mixture models (`None` for
  non-mixture models)
- `obs`, `var`, `uns` — optional metadata if using `AnnData`

## Common Operations

### Accessing Parameters and Posterior Distributions

The `ScribeResults` class provides several methods to access the learned model
parameters, either as raw variational parameters, probability distributions, or
point estimates:

```python
# Get raw parameters for variational posterior
params = results.params

# Get posterior distributions for parameters
# (returns scipy.stats distributions by default)
distributions = results.get_distributions()

# Get posterior distributions as numpyro distributions
distributions_numpyro = results.get_distributions(backend="numpyro")

# Get maximum a posteriori (MAP) estimates
map_estimates = results.get_map()
```

### Descriptive parameter names

By default, SCRIBE's internal parameter keys use compact math-style names
(`r`, `p`, `mu`, `phi`, `gate`, `p_capture`, etc.). For more readable output,
pass `descriptive_names=True` to any parameter-access method. This renames the
keys to self-documenting equivalents:

| Internal key | Descriptive name |
|-------------|-----------------|
| `r` | `dispersion` |
| `p` | `prob` |
| `mu` | `expression` |
| `phi` | `odds` |
| `gate` | `zero_inflation` |
| `p_capture` | `capture_prob` |
| `phi_capture` | `capture_odds` |
| `eta_capture` | `capture_efficiency` |
| `mu_eta` | `capture_scaling` |

Suffixed keys are handled automatically (e.g., `r_0` becomes `dispersion_0`).

```python
# Default internal names
map_estimates = results.get_map()
# >>> dict_keys(['r', 'p', 'p_capture', ...])

# Human-readable names
map_estimates = results.get_map(descriptive_names=True)
# >>> dict_keys(['dispersion', 'prob', 'capture_prob', ...])
```

The `descriptive_names` option is supported by `get_map()`,
`get_distributions()`, `get_posterior_samples()`, and
`sample_posterior_parameters()`.

!!! tip
    Use `descriptive_names=True` in notebooks and exploratory analysis for
    clarity. Stick with the default internal names when passing parameters to
    other SCRIBE functions that expect them.

### Subsetting Genes

The `ScribeResults` object supports indexing operations to extract results for
specific genes of interest. You can use integer indexing, slicing, or boolean
masks to subset the results:

```python
# Get results for first gene
gene_results = results[0]

# Get results for a set of genes
subset_results = results[0:10]  # First 10 genes

# Boolean indexing
highly_variable = results.var["highly_variable"]
if highly_variable is not None:
    hv_results = results[highly_variable]
```

### Working with Mixture Components

For mixture models, you can access specific components:

```python
# Get results for the first component
component_results = results.get_component(0)

# The component results are a non-mixture ScribeResults object
print(component_results.model_type)  # e.g., "nbdm" instead of "nbdm_mix"
```

### Posterior Sampling

The `ScribeResults` class provides several methods for generating different
types of samples:

1. **Posterior Parameter Samples**: Draw samples directly from the fitted
   parameter distributions using `get_posterior_samples()`. These samples
   represent uncertainty in the model parameters as sampled from the
   variational posterior distribution.

2. **Predictive Samples**: Generate new data from the model using
   `get_predictive_samples()`. This simulates new count data using the MAP
   parameter estimates.

3. **Posterior Predictive Check (PPC) Samples**: Combine both operations with
   `get_ppc_samples()` to generate data for model validation.

```python
# Draw 1000 samples from the posterior distributions of parameters
posterior_samples = results.get_posterior_samples(n_samples=1000)

# Generate new count data using MAP estimates
predictive_samples = results.get_predictive_samples()

# Generate posterior predictive samples for model checking
ppc_samples = results.get_ppc_samples(n_samples=1000)
```

!!! note
    Generating posterior predictive samples requires simulating entire datasets,
    which can be computationally intensive. For large datasets, we recommend:

    - Reducing the number of samples
    - Subsetting to fewer genes
    - Using GPU acceleration if available
    - Running sampling in batches

#### Guide-geometry diagnostic: `map_sampling=True`

`scribe.viz.plot_ppc(..., map_sampling=True)` generates posterior
predictive samples by fixing the variational parameters at their
posterior mean (MAP-anchored) rather than sampling from the full
guide. Compared with the default `map_sampling=False`, the two plots
together form a clean diagnostic for **where wide PPC bands come
from**:

| MAP PPC | Posterior PPC | Interpretation                                                                                       |
| ------- | ------------- | ---------------------------------------------------------------------------------------------------- |
| Tight   | Tight         | Fit is fine on these genes                                                                           |
| Tight   | Wide          | Guide geometry: the posterior is well-localized but mean-field q-spread leaks into the PPC. Try a low-rank or flow guide. |
| Wide    | Wide          | Model / likelihood / capture problem. A richer guide will not help; try a different parameterization or a tighter prior. |
| Wide    | Tight         | Vanishingly rare — usually indicates a non-converged fit. Re-check loss curves.                       |

```python
# Default: full-posterior PPC
scribe.viz.plot_ppc(results, adata, n_genes=16, n_rows=4)

# MAP-anchored PPC: same call, one flag
scribe.viz.plot_ppc(results, adata, n_genes=16, n_rows=4,
                    map_sampling=True)
```

The flag is supported for all NB-family and TwoState-family results.
For TwoState the MAP-PPC path constructs the Poisson-Beta compound at
the MAP parameters and draws `p_gc` **independently per (gene, cell)**
— sharing a single latent across cells would introduce a
replicate-level random effect the model does not have. (Compositional
PPCs do not yet expose a MAP path.)

### Log Likelihood Computation

Computing the log-likelihood of your data under the fitted model can be
valuable for several purposes:

- **Model comparison**: Compare different model fits or architectures by their
  log-likelihood scores
- **Quality control**: Identify cells or genes that are poorly explained by
  the model
- **Outlier detection**: Find data points with unusually low likelihood values
- **Model validation**: Assess how well the model captures the underlying data
  distribution

```python
log_liks = results.compute_log_likelihood(
    counts,
    return_by="cell",  # or 'gene'
    batch_size=512,
)
```

### Cell Type Assignment (Mixture Models)

For mixture models, SCRIBE provides methods to compute probabilistic cell type
assignments. These assignments quantify how likely each cell belongs to each
component (cell type) in the mixture, while also characterizing the uncertainty
in these assignments.

The computation involves three key steps:

1. For each cell, compute the likelihood that it belongs to each component
   using the full posterior distribution of model parameters
2. Convert these likelihoods into proper probability distributions over
   components
3. (Optional) Fit a Dirichlet distribution to characterize the uncertainty in
   these assignments

The resulting probabilities can be used to:

- Make soft assignments of cells to types
- Identify cells with ambiguous type assignments
- Quantify uncertainty in cell type classifications
- Study cells that may be transitioning between states

Two methods are provided:

- `compute_cell_type_assignments()` — uses the full posterior distribution to
  compute assignments and uncertainty
- `compute_cell_type_assignments_map()` — uses point estimates for faster but
  less detailed results

```python
# Compute cell type assignment probabilities
assignments = results.compute_cell_type_assignments(
    counts,
    fit_distribution=True,
)

# Get Dirichlet concentration parameters
concentrations = assignments["concentration"]

# Get mean assignment probabilities
mean_probs = assignments["mean_probabilities"]

# Get assignment probabilities for each posterior sample
sample_probs = assignments["sample_probabilities"]

# Compute using MAP estimates only (faster, less uncertainty info)
map_assignments = results.compute_cell_type_assignments_map(counts)
```

### Entropy Analysis for Mixture Models

For mixture models, SCRIBE provides methods to compute the entropy of component
assignments, which serves as a measure of assignment uncertainty. Higher entropy
values indicate more uncertainty in the assignments (the cell or gene could
belong to multiple components), while lower values indicate more confident
assignments (the cell or gene clearly belongs to one component).

The entropy calculation can be performed:

- Per cell: Measuring how confidently each cell is assigned to a component
- Per gene: Measuring how component-specific each gene's expression pattern is
- With optional normalization: Making entropy values comparable across datasets
  of different sizes

```python
entropies = results.compute_component_entropy(
    counts,
    return_by="cell",  # or 'gene'
    normalize=False,
)
```

## Model-Specific Parameters

The `ScribeResults` class works with all model types supported by SCRIBE. Each
model type has specific parameters available in the `params` dictionary based
on the distributions used.

=== "NBDM (variable_capture=False)"

    ```python
    nbdm_results = scribe.fit(adata, model="nbdm")

    # Dispersion parameters (LogNormal distribution)
    r_loc = nbdm_results.params["r_loc"]
    r_scale = nbdm_results.params["r_scale"]

    # Or (Gamma distribution)
    r_concentration = nbdm_results.params["r_concentration"]
    r_rate = nbdm_results.params["r_rate"]

    # Success probability parameters
    p_concentration1 = nbdm_results.params["p_concentration1"]  # Alpha
    p_concentration0 = nbdm_results.params["p_concentration0"]  # Beta
    ```

=== "ZINB (zero_inflation=True)"

    ```python
    zinb_results = scribe.fit(adata, model="zinb")

    # Additional dropout parameters
    gate_concentration1 = zinb_results.params["gate_concentration1"]
    gate_concentration0 = zinb_results.params["gate_concentration0"]
    ```

=== "NBVCP (variable_capture=True)"

    ```python
    nbvcp_results = scribe.fit(adata, model="nbvcp")

    # Additional capture probability parameters
    p_capture_concentration1 = nbvcp_results.params["p_capture_concentration1"]
    p_capture_concentration0 = nbvcp_results.params["p_capture_concentration0"]
    ```

=== "ZINBVCP (variable_capture=True, zero_inflation=True)"

    ```python
    zinbvcp_results = scribe.fit(adata, model="zinbvcp")

    # Additional dropout and capture probability parameters
    gate_concentration1 = zinbvcp_results.params["gate_concentration1"]
    gate_concentration0 = zinbvcp_results.params["gate_concentration0"]
    p_capture_concentration1 = zinbvcp_results.params["p_capture_concentration1"]
    p_capture_concentration0 = zinbvcp_results.params["p_capture_concentration0"]
    ```

=== "Mixture"

    ```python
    mix_results = scribe.fit(adata, model="nbdm", n_components=3)

    # Mixing weights concentration parameters
    mixing_concentration = mix_results.params["mixing_concentration"]

    # Component-specific parameters have additional dimensions
    # Shape: (n_components, n_genes)
    r_concentration = mix_results.params["r_concentration"]
    ```

=== "TwoState (twostate / twostatevcp)"

    The variational parameters depend on which of the four
    parameterizations was used and on `unconstrained`.

    **`unconstrained=True`** (Normal + transform):

    ```python
    # Natural: samples (mu, burst_size, k_off)
    ts_results = scribe.fit(adata, model="twostatevcp",
                            parameterization="two_state_natural",
                            unconstrained=True)
    mu_loc = ts_results.params["mu_loc"]          # shape: (n_genes,)
    burst_size_loc = ts_results.params["burst_size_loc"]
    k_off_loc = ts_results.params["k_off_loc"]

    # Ratio:         → switching_ratio_loc / switching_ratio_scale
    # Mean-Fano:     → excess_fano_*, concentration_*
    # Moment-delta:  → excess_fano_*, inv_concentration_* (sigmoid-Normal)
    ```

    **`unconstrained=False`** (constrained distributions, the default):

    ```python
    # Same model, constrained guides
    ts_results = scribe.fit(adata, model="twostatevcp",
                            parameterization="two_state_natural")
    # LogNormal params for positive parameters
    mu_loc = ts_results.params["mu_loc"]           # LogNormal loc
    mu_scale = ts_results.params["mu_scale"]       # LogNormal scale
    # BetaSpec params for inv_concentration (moment_delta only)
    # → inv_concentration_concentration1 / _concentration0
    ```

    **Mixture models** (`n_components=K`): component-specific parameters
    gain a leading component dimension:

    ```python
    ts_mix = scribe.fit(adata, model="twostatevcp",
                        parameterization="two_state_natural",
                        n_components=3,
                        mixture_params=["mu", "burst_size", "k_off"])
    # mu_loc shape: (n_components, n_genes)
    # mixing_concentration shape: (n_components,)
    ```

    **Posterior samples** include derived deterministics:

    ```python
    samples = ts_results.get_posterior_samples()
    # samples["mu"], samples["burst_size"], samples["k_off"],
    # samples["alpha"], samples["beta"], samples["r_hat"],
    # samples["concentration"] (derived from delta in moment_delta), ...
    ```

    See [Two-state promoter theory](../theory/two-state-promoter.md) for
    which parameters are sampled vs derived under each variant.

## Laplace Results (`ScribeLaplaceResults`)

When using `inference_method="laplace"` with PLN, LNM, or LNMVCP models,
`scribe.fit()` returns a `ScribeLaplaceResults` object. This class shares the
common API with `ScribeResults` while providing Laplace-specific functionality.

### Shared API

The following methods work identically across all results classes:

```python
results = scribe.fit(adata, model="pln", inference_method="laplace")

# MAP estimates
map_estimates = results.get_map()

# Posterior distributions
distributions = results.get_distributions()

# Log-likelihood
ll = results.compute_log_likelihood(counts=adata.X)

# Loss history
losses = results.loss_history
```

### Laplace-specific attributes

| Attribute / Method | Description |
|--------------------|-------------|
| `results.final_grad_norms` | Per-cell Newton gradient norms at convergence (array of shape `(n_cells,)`) |
| `results.x_loc` | Per-cell MAP log-rates (PLN) or factor scores |
| `results.eta_loc` | Per-cell MAP capture offsets (when applicable) |
| `results.z_loc` | Per-cell MAP composition latents (LNM with `d_mode='low_rank'`) |
| `results.y_alr_loc` | Per-cell MAP ALR logits (LNM with `d_mode='learned'`) |
| `results.p_capture_loc` | Per-cell MAP capture probabilities |

### Model-dispatching behavior

`ScribeLaplaceResults` dispatches on `model_config.base_model` to provide
model-appropriate behavior:

```python
# PLN results
pln_results = scribe.fit(adata, model="pln", inference_method="laplace")
mu = pln_results.get_mu()       # population mean log-rates
W = pln_results.get_W()         # loadings matrix (G x k)

# LNMVCP results
lnm_results = scribe.fit(adata, model="lnmvcp", inference_method="laplace")
mu = lnm_results.get_mu()       # population mean ALR logits
W = lnm_results.get_W()         # loadings matrix ((G-1) x k)
```

### Posterior predictive checks

Two PPC modes are available:

```python
# MAP-only PPC: point estimates + observation noise
map_ppc = results.get_map_ppc_samples(n_samples=100, seed=0)

# Laplace-uncertainty PPC: propagate Hessian uncertainty into predictions
laplace_ppc = results.get_per_cell_predictive_samples(n_samples=100, seed=0)
```

| Mode | What it tests | When to use |
|------|---------------|-------------|
| **MAP-only** (`get_map_ppc_samples`) | Does the likelihood shape match the data given point estimates? | Quick diagnostic; cheapest PPC |
| **Laplace-uncertainty** (`get_per_cell_predictive_samples`) | Does the full posterior-predictive distribution match observed data? | Publication-quality diagnostics; honest uncertainty |

The Laplace-uncertainty PPC samples the latent from the per-cell Gaussian
approximation \(\mathcal{N}(\hat{x}, (-H)^{-1})\), using a square-root
factorization that avoids materializing any \(G \times G\) matrix.

### Inspecting convergence

```python
import numpy as np

gn = np.asarray(results.final_grad_norms)
print(f"Cells: {gn.size}")
print(f"max: {gn.max():.3e}, p99: {np.percentile(gn, 99):.3e}, "
      f"median: {np.median(gn):.3e}")
print(f"Cells above 1e-3: {(gn > 1e-3).sum()}")
```

---

## Model Comparison

To compare models, you can use the model comparison utilities:

```python
from scribe import compare_models

# Fit multiple models
nbdm_results = scribe.fit(adata)
zinb_results = scribe.fit(adata, zero_inflation=True)

# Compare models
mc = compare_models(
    [nbdm_results, zinb_results],
    counts=adata.X,
    model_names=["NBDM", "ZINB"],
)
print(mc.summary())

# Per-gene comparison
gene_df = mc.gene_level_comparison("NBDM", "ZINB")
```

## Best Practices

1. **Memory Management**:
    - Use `batch_size` for large datasets
    - Generate posterior samples for specific gene subsets
    - Use `compute_log_likelihood` with batching for large-scale analyses

2. **Working with Parameters**:
    - Access raw parameters through `.params`
    - Use `.get_distributions()` for parameter interpretation and sampling
    - Use `.get_map()` for point estimates

3. **Model Selection**:
    - Start with the simplest model (NBDM)
    - Add complexity (zero-inflation, capture probability) as justified by data
    - Consider mixture models for heterogeneous populations
    - Use model comparison tools to select the best model

4. **Diagnostics**:
    - Check `loss_history` for convergence
    - Use posterior predictive checks to evaluate model fit
    - For mixture models, examine entropy of component assignments

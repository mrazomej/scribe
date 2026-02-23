# SCRIBE MCMC (Markov Chain Monte Carlo)

This directory contains the Markov Chain Monte Carlo (MCMC) implementation for
SCRIBE models. MCMC provides exact Bayesian inference by generating samples from
the true posterior distribution, offering the most accurate uncertainty
quantification available in SCRIBE.

## Architecture

`ScribeMCMCResults` is a **`@dataclass`** that wraps a NumPyro `MCMC` object
(composition, not inheritance) and composes analysis functionality from eight
mixin classes -- the same pattern used by the SVI results module.

### File structure

```
src/scribe/mcmc/
  __init__.py                 # Public exports
  inference_engine.py         # NUTS execution (supports init_values)
  results_factory.py          # Factory: MCMC -> ScribeMCMCResults
  results.py                  # ScribeMCMCResults @dataclass (~120 lines)
  _init_from_svi.py           # SVI-to-MCMC cross-parameterization init
  _parameter_extraction.py    # ParameterExtractionMixin
  _gene_subsetting.py         # GeneSubsettingMixin
  _component.py               # ComponentMixin
  _model_helpers.py           # ModelHelpersMixin
  _sampling.py                # SamplingMixin
  _likelihood.py              # LikelihoodMixin
  _normalization.py           # NormalizationMixin
  _mixture_analysis.py        # MixtureAnalysisMixin
```

### Composition over inheritance

Previous versions of `ScribeMCMCResults` inherited from `numpyro.infer.MCMC`
and required a separate `ScribeMCMCSubset` dataclass for gene/component
subsets. The current design wraps the MCMC object as an optional `_mcmc` field:

- Subsetting (gene or component) returns another `ScribeMCMCResults`, not a
  different type.
- `print_summary()` and `get_extra_fields()` delegate to `_mcmc` when
  available.
- `_mcmc` is `None` on subsets (MCMC diagnostics are only meaningful for
  the full run).

### Mixin diagram

```
ScribeMCMCResults(@dataclass)
 ├── ParameterExtractionMixin   # quantiles, MAP, component-specificity checks
 ├── GeneSubsettingMixin        # __getitem__ gene indexing
 ├── ComponentMixin             # get_component, get_components
 ├── ModelHelpersMixin          # _model(), _log_likelihood_fn()
 ├── SamplingMixin              # PPC, biological PPC, denoising, prior predictive
 ├── LikelihoodMixin            # log_likelihood()
 ├── NormalizationMixin         # normalize_counts()
 └── MixtureAnalysisMixin       # cell_type_probabilities()
```

## Overview

The MCMC module provides:

1. **NUTS Sampling**: No-U-Turn Sampler for efficient Hamiltonian Monte Carlo
2. **Exact Inference**: True posterior samples without variational approximation
3. **Diagnostics**: Comprehensive convergence and mixing diagnostics
4. **Uncertainty Quantification**: Full posterior distributions for all
   parameters

## Key Components

### MCMCInferenceEngine (`inference_engine.py`)

The main inference engine that handles MCMC execution using NumPyro's NUTS
sampler:

```python
from scribe.mcmc import MCMCInferenceEngine
from scribe.models import ModelConfig

# Configure your model
config = ModelConfig(
    base_model="nbdm",
    parameterization="standard",
    inference_method="mcmc"
)

# Run MCMC inference
mcmc_results = MCMCInferenceEngine.run_inference(
    model_config=config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_samples=2000,
    n_warmup=1000,
    n_chains=4
)
```

**Key Features:**
- Uses NUTS (No-U-Turn Sampler) for efficient sampling
- Automatic tuning of step size and mass matrix
- Parallel chain execution for convergence diagnostics
- Customizable NUTS hyperparameters

**Parameters:**
- `model_config`: Model configuration specifying architecture
- `count_data`: Single-cell count matrix (cells × genes)
- `n_samples`: Number of posterior samples per chain
- `n_warmup`: Number of warmup/burn-in samples
- `n_chains`: Number of parallel chains
- `mcmc_kwargs`: NUTS-specific parameters (target_accept_prob, max_tree_depth)

### ScribeMCMCResults (`results.py`)

Comprehensive results class that stores MCMC samples and provides analysis
methods:

```python
from scribe.mcmc import ScribeMCMCResults

# Results are typically created by the inference engine
# but can also be constructed manually
results = ScribeMCMCResults.from_anndata(
    adata=adata,
    mcmc=mcmc_object,
    model_type="nbdm",
    model_config=config
)
```

#### Core Attributes

- **`samples`**: Dictionary of raw posterior samples for all parameters
- **`_mcmc`**: Wrapped NumPyro MCMC object (for diagnostics; `None` on subsets)
- **`model_config`**: Model configuration used for inference
- **`n_cells`**, **`n_genes`**: Dataset dimensions
- **`obs`**, **`var`**, **`uns`**: Metadata from AnnData objects

#### Key Analysis Methods

**Posterior Access:**
```python
# Get posterior samples (already in canonical form -- derived parameters
# are registered as numpyro.deterministic sites and unconstrained specs
# sample via TransformedDistribution in constrained space)
samples = results.get_posterior_samples()

# Get samples grouped by chain for diagnostics
chain_samples = results.get_samples(group_by_chain=True)

# Get posterior quantiles
quantiles = results.get_posterior_quantiles(
    param="p",
    quantiles=(0.025, 0.5, 0.975)
)
```

**Point Estimates:**
```python
# Maximum a posteriori (MAP) estimates
map_estimates = results.get_map()
```

**Model Evaluation:**
```python
# Compute log-likelihood using posterior samples
log_lik = results.log_likelihood(counts=count_data)
```

**Predictive Analysis:**
```python
# Posterior predictive checks (includes technical noise)
ppc_samples = results.get_ppc_samples(rng_key=rng_key)

# Biological (denoised) PPC — strips capture probability and
# zero-inflation gate, sampling from the base NB(r, p) only.
bio_ppc = results.get_ppc_samples_biological(
    rng_key=rng_key,
    cell_batch_size=2048,
)

# Prior predictive samples
prior_samples = results.get_prior_predictive_samples(rng_key=rng_key)
```

**Bayesian Denoising of Observed Counts:**

Takes observed counts and computes the posterior of the true (pre-capture,
pre-dropout) transcript counts using the MCMC posterior samples.

```python
denoised = results.denoise_counts(
    counts=observed_counts,
    method="mean",
    rng_key=rng_key,
)
# denoised.shape == (n_posterior_samples, n_cells, n_genes)
denoised_avg = denoised.mean(axis=0)

# With variance for uncertainty quantification
result = results.denoise_counts(
    counts=observed_counts,
    return_variance=True,
    rng_key=rng_key,
)
# result["denoised_counts"] and result["variance"]
```

**Mixture Model Analysis:**
```python
# For mixture models, analyze cell type assignments
cell_type_probs = results.cell_type_probabilities(counts=count_data)

# Access one or multiple mixture components
component_0 = results.get_component(0)
components_12 = results.get_components([1, 2])
```

**Data Subsetting:**
```python
# Subset by genes -- always returns ScribeMCMCResults
gene_subset = results[:100]

# Two-axis indexing: results[genes, components]
gene_component_subset = results[1:4, [1, 2]]
```

**MCMC Diagnostics:**
```python
# Print summary with R-hat statistics (delegates to wrapped MCMC)
results.print_summary()

# Get extra fields (potential energy, divergences, etc.)
extra = results.get_extra_fields()
```

### MCMCResultsFactory (`results_factory.py`)

Factory class for creating and packaging MCMC results:

```python
from scribe.mcmc import MCMCResultsFactory

results = MCMCResultsFactory.create_results(
    mcmc_results=raw_mcmc,
    model_config=config,
    adata=adata,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    model_type="nbdm",
    n_components=None,
    prior_params=priors
)
```

## Usage Examples

### Basic MCMC Inference

```python
import jax.numpy as jnp
from scribe.models import ModelConfig
from scribe.mcmc import MCMCInferenceEngine

# Prepare data
count_data = jnp.array(your_count_matrix)  # cells × genes
n_cells, n_genes = count_data.shape

# Configure model
config = ModelConfig(
    base_model="zinb",
    parameterization="linked",
    inference_method="mcmc"
)

# Run MCMC inference
mcmc_results = MCMCInferenceEngine.run_inference(
    model_config=config,
    count_data=count_data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_samples=3000,
    n_warmup=1500,
    n_chains=4,
    seed=42
)
```

### Convergence Diagnostics

```python
# Print summary with R-hat statistics
mcmc_results.print_summary()

# Get diagnostic fields
extra = mcmc_results.get_extra_fields()
potential_energy = extra.get("potential_energy")
```

### Mixture Model Analysis

```python
# Configure mixture model
config = ModelConfig(
    base_model="nbdm_mix",
    n_components=3,
    parameterization="standard",
    inference_method="mcmc"
)

# Run MCMC
mcmc_results = MCMCInferenceEngine.run_inference(
    model_config=config,
    count_data=count_data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_samples=4000,
    n_warmup=2000,
    n_chains=4
)

# Analyze mixture components
cell_type_probs = mcmc_results.cell_type_probabilities(counts=count_data)

# Multi-component selection with optional renormalization
components = mcmc_results.get_components([1, 2])
components_no_renorm = mcmc_results.get_components([1, 2], renormalize=False)

# Tuple indexing: first genes, second components
subset = mcmc_results[1:4, [1, 2]]
```

## Initializing MCMC from SVI Results

MCMC chains can be initialized from SVI results using `init_to_value`, which
starts all chains near the SVI optimum. This typically improves warmup
convergence and reduces the number of warmup samples needed.

### Via `fit()` API

```python
import scribe

# Same parameterization
svi_results = scribe.fit(adata, model="nbdm", parameterization="linked")
mcmc_results = scribe.fit(
    adata, model="nbdm", parameterization="linked",
    inference_method="mcmc", svi_init=svi_results,
)

# Cross-parameterization: SVI linked -> MCMC odds_ratio
# (phi and mu are automatically derived from canonical p, r)
mcmc_results = scribe.fit(
    adata, model="nbdm", parameterization="odds_ratio",
    inference_method="mcmc", svi_init=svi_results,
)

# Cross-constrained: SVI constrained -> MCMC unconstrained
# (works seamlessly -- same site names, same constrained-space values)
mcmc_results = scribe.fit(
    adata, model="nbdm", unconstrained=True,
    inference_method="mcmc", svi_init=svi_results,
)
```

### Via engine directly (power users)

```python
from scribe.mcmc import MCMCInferenceEngine, compute_init_values

# Extract MAP and convert to target parameterization
map_values = svi_results.get_map(use_mean=True, canonical=True)
init_values = compute_init_values(map_values, target_model_config)

# Pass to the inference engine
mcmc = MCMCInferenceEngine.run_inference(
    model_config=target_model_config,
    count_data=data,
    n_cells=n_cells,
    n_genes=n_genes,
    init_values=init_values,
)
```

### How it works

1. SVI MAP estimates are extracted via `get_map(use_mean=True, canonical=True)`.
2. `compute_init_values()` ensures the init dict contains all sampled parameters
   for the target MCMC parameterization. Missing parameters are derived from
   canonical `(p, r)`:
   - `mu = r * p / (1 - p)` for mean_prob / mean_odds targets
   - `phi = (1 - p) / p` for mean_odds targets
   - `phi_capture` / `p_capture` conversion for VCP models
3. `init_to_value(values=init_values)` is set as the NUTS `init_strategy`.
   NumPyro maps constrained-space values to unconstrained space internally.
4. All chains start at the same MAP point. NUTS warmup adaptation will
   diverge the chains.

### Notes

- **Constrained vs unconstrained** is transparent -- site names and value spaces
  are identical regardless of the `unconstrained` flag.
- **Hierarchical hyperparameters** (`logit_p_loc`, `log_phi_scale`, etc.) cannot
  be reliably converted across parameterizations. They fall back to
  `init_to_uniform`.
- `use_mean=True` avoids NaN MAP values that can occur with LogNormal when
  sigma is large.

## MCMC vs SVI Comparison

| Aspect | MCMC | SVI |
|--------|------|-----|
| **Accuracy** | Exact posterior samples | Approximate posterior |
| **Uncertainty** | Full posterior distribution | Variational approximation |
| **Speed** | Slower (minutes to hours) | Faster (seconds to minutes) |
| **Scalability** | Limited by memory | Highly scalable |
| **Convergence** | Diagnostic tools available | Loss-based monitoring |
| **Use Case** | Final analysis, small-medium data | Exploration, large data |

## When to Use MCMC

**Recommended for:**
- Final, publication-quality analysis
- When exact uncertainty quantification is critical
- Model comparison and selection
- Small to medium datasets (< 10,000 cells)
- When computational time is not a constraint

**Consider SVI instead when:**
- Working with very large datasets
- Rapid prototyping and exploration
- Real-time or interactive analysis
- Computational resources are limited

## Migration Notes

**`ScribeMCMCSubset` has been removed.** All subsetting operations (gene
indexing, component selection) now return `ScribeMCMCResults` instances.
Code that previously type-checked for `ScribeMCMCSubset` should use
`ScribeMCMCResults` instead.

**`ScribeMCMCResults` no longer inherits from `numpyro.infer.MCMC`.**
Code that relied on `isinstance(results, MCMC)` should be updated.
MCMC-specific methods like `print_summary()` and `get_extra_fields()`
are still available via delegation to the wrapped `_mcmc` object.

**`_convert_to_canonical` has been removed.** MCMC samples already contain
canonical parameters (`p`, `r`, `mixing_weights`, etc.) because derived
parameters are registered as `numpyro.deterministic` sites and
unconstrained specs sample via `TransformedDistribution` in constrained
space. The `canonical` parameter has been removed from
`get_posterior_samples()`, `get_samples()`, `get_posterior_quantiles()`,
and `get_map()`. Code that passed `canonical=True` should simply drop
that argument.

## Integration with Other Modules

- **Models**: Supports all SCRIBE model types and parameterizations
- **Sampling**: Uses MCMC samples for predictive analysis
- **Stats**: Provides exact posterior distributions for statistical tests
- **Viz**: MCMC results can be visualized using the visualization module
- **Core**: Integrates with normalization and preprocessing utilities

## Dependencies

- **NumPyro**: MCMC implementation and NUTS sampler
- **JAX**: Automatic differentiation and compilation
- **Pandas**: Metadata handling and data structures
- **NumPy**: Numerical computations and array operations
- **AnnData**: Single-cell data format integration (optional)

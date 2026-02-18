# SCRIBE MCMC (Markov Chain Monte Carlo)

This directory contains the Markov Chain Monte Carlo (MCMC) implementation for
SCRIBE models. MCMC provides exact Bayesian inference by generating samples from
the true posterior distribution, offering the most accurate uncertainty
quantification available in SCRIBE.

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

- **`mcmc`**: NumPyro MCMC object with samples and diagnostics
- **`samples`**: Dictionary of posterior samples for all parameters
- **`model_config`**: Model configuration used for inference
- **`n_cells`**, **`n_genes`**: Dataset dimensions
- **`obs`**, **`var`**, **`uns`**: Metadata from AnnData objects

#### Key Analysis Methods

**Posterior Access:**
```python
# Get posterior samples in canonical form
samples = results.get_posterior_samples(canonical=True)

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
map_estimates = results.get_map(canonical=True)
```

**Model Evaluation:**
```python
# Compute log-likelihood using posterior samples
log_lik = results.log_likelihood(
    n_samples=1000,
    seed=42
)
```

**Predictive Analysis:**
```python
# Posterior predictive checks (includes technical noise)
ppc_samples = results.get_ppc_samples(
    n_samples=500,
    seed=42
)

# Biological (denoised) PPC — strips capture probability and
# zero-inflation gate, sampling from the base NB(r, p) only.
# For NBDM models this is equivalent to standard PPC.
bio_ppc = results.get_ppc_samples_biological(
    rng_key=rng_key,
    cell_batch_size=2048,  # Optional cell batching for memory
)

# Prior predictive samples
prior_samples = results.get_prior_predictive_samples(
    n_samples=500,
    seed=42
)
```

**Mixture Model Analysis:**
```python
# For mixture models, analyze cell type assignments
cell_type_probs = results.cell_type_probabilities()
```

**Data Subsetting:**
```python
# Subset results by genes or cells
gene_subset = results[:, :100]  # First 100 genes
cell_subset = results[:500, :]  # First 500 cells
```

### MCMCResultsFactory (`results_factory.py`)

Factory class for creating and packaging MCMC results:

```python
from scribe.mcmc import MCMCResultsFactory

# Package raw MCMC results into ScribeMCMCResults
results = MCMCResultsFactory.create_results(
    mcmc_results=raw_mcmc,
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

### Advanced MCMC Configuration

```python
# Custom NUTS parameters for challenging models
mcmc_kwargs = {
    "target_accept_prob": 0.9,    # Higher acceptance rate
    "max_tree_depth": 12,         # Deeper trees for complex geometry
    "adapt_step_size": True,      # Adaptive step size tuning
    "adapt_mass_matrix": True,    # Adaptive mass matrix
}

mcmc_results = MCMCInferenceEngine.run_inference(
    model_config=config,
    count_data=count_data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_samples=5000,
    n_warmup=2500,
    n_chains=6,  # More chains for better diagnostics
    mcmc_kwargs=mcmc_kwargs,
    seed=42
)
```

### Convergence Diagnostics

```python
# Access MCMC object for diagnostics
mcmc = mcmc_results.mcmc

# Print summary with R-hat statistics
mcmc.print_summary()

# Get R-hat values for manual inspection
r_hat = mcmc.get_extra_fields()['r_hat']

# Check effective sample size
n_eff = mcmc.get_extra_fields()['n_eff']

# Identify problematic parameters
problematic_params = []
for param, r_hat_val in r_hat.items():
    if jnp.any(r_hat_val > 1.1):  # R-hat > 1.1 indicates poor convergence
        problematic_params.append(param)
        print(f"Parameter {param} has poor convergence (R-hat > 1.1)")
```

### Posterior Analysis

```python
# Get posterior samples
samples = mcmc_results.get_posterior_samples(canonical=True)

# Analyze parameter distributions
import matplotlib.pyplot as plt

# Plot posterior distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Success probabilities
p_samples = samples['p']
axes[0, 0].hist(p_samples.flatten(), bins=50, alpha=0.7)
axes[0, 0].set_title('Posterior: Success Probabilities')
axes[0, 0].set_xlabel('p')

# Dispersion parameters
r_samples = samples['r']
axes[0, 1].hist(r_samples.flatten(), bins=50, alpha=0.7)
axes[0, 1].set_title('Posterior: Dispersion Parameters')
axes[0, 1].set_xlabel('r')

# Trace plots for convergence
axes[1, 0].plot(p_samples[:, 0])  # First gene
axes[1, 0].set_title('Trace Plot: p[0]')
axes[1, 0].set_xlabel('Sample')

axes[1, 1].plot(r_samples[:, 0])  # First gene
axes[1, 1].set_title('Trace Plot: r[0]')
axes[1, 1].set_xlabel('Sample')

plt.tight_layout()
plt.show()
```

### Model Comparison with MCMC

```python
# Compare models using exact Bayesian inference
models = ["nbdm", "zinb", "nbvcp"]
mcmc_results = {}
log_likelihoods = {}

for model_type in models:
    config = ModelConfig(
        base_model=model_type,
        inference_method="mcmc"
    )
    
    # Run MCMC
    result = MCMCInferenceEngine.run_inference(
        model_config=config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_samples=2000,
        n_warmup=1000,
        n_chains=4
    )
    
    mcmc_results[model_type] = result
    
    # Compute log-likelihood
    log_likelihoods[model_type] = result.log_likelihood(
        n_samples=1000,
        seed=42
    )

# Compare models
for model, ll in log_likelihoods.items():
    print(f"{model}: log-likelihood = {ll:.2f}")
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
cell_type_probs = mcmc_results.cell_type_probabilities()

# Plot cell type assignments
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(cell_type_probs.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Assignment Probability')
plt.xlabel('Cells')
plt.ylabel('Components')
plt.title('Cell Type Assignment Probabilities')
plt.show()
```

### Uncertainty Quantification

```python
# Get credible intervals for all parameters
samples = mcmc_results.get_posterior_samples(canonical=True)

# Compute 95% credible intervals
credible_intervals = {}
for param_name, param_samples in samples.items():
    lower = jnp.percentile(param_samples, 2.5, axis=0)
    upper = jnp.percentile(param_samples, 97.5, axis=0)
    credible_intervals[param_name] = {
        'lower': lower,
        'upper': upper,
        'width': upper - lower
    }

# Identify parameters with high uncertainty
high_uncertainty_genes = []
for i, width in enumerate(credible_intervals['p']['width']):
    if width > 0.5:  # Threshold for high uncertainty
        high_uncertainty_genes.append(i)

print(f"Found {len(high_uncertainty_genes)} genes with high uncertainty in p")
```

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

## Optimization and Diagnostics

### Improving Convergence

```python
# For difficult models, try these strategies:

# 1. Longer warmup
mcmc_kwargs = {
    "target_accept_prob": 0.95,  # Higher acceptance rate
    "max_tree_depth": 15,        # Deeper exploration
}

# 2. More chains and samples
n_chains = 8
n_samples = 5000
n_warmup = 3000

# 3. Parameter initialization
# (Advanced: custom initialization strategies)
```

### Diagnostic Checks

```python
# Essential diagnostic checks:

# 1. R-hat convergence diagnostic
r_hat = mcmc.get_extra_fields()['r_hat']
max_r_hat = max([jnp.max(r) for r in r_hat.values()])
print(f"Maximum R-hat: {max_r_hat:.3f}")
if max_r_hat > 1.1:
    print("Warning: Poor convergence detected!")

# 2. Effective sample size
n_eff = mcmc.get_extra_fields()['n_eff']
min_n_eff = min([jnp.min(n) for n in n_eff.values()])
print(f"Minimum effective sample size: {min_n_eff:.0f}")

# 3. Divergent transitions
divergences = mcmc.get_extra_fields().get('diverging', 0)
print(f"Number of divergent transitions: {divergences}")
```

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

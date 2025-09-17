# SCRIBE SVI (Stochastic Variational Inference)

This directory contains the core implementation of Stochastic Variational
Inference (SVI) for SCRIBE models. SVI is a scalable approximate inference
method that uses optimization to find the best approximation to the posterior
distribution within a chosen variational family.

## Overview

The SVI module provides:

1. **Inference Engine**: Executes SVI optimization using NumPyro's SVI framework
2. **Results Management**: Comprehensive results class with analysis methods
3. **Results Factory**: Streamlined creation and packaging of results objects

## Key Components

### SVIInferenceEngine (`inference_engine.py`)

The main inference engine that handles SVI execution:

```python
from scribe.svi import SVIInferenceEngine
from scribe.models import ModelConfig

# Configure your model
config = ModelConfig(
    base_model="nbdm",
    parameterization="standard",
    inference_method="svi"
)

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
- Handles both VAE and traditional SVI models
- Numerically stable parameter updates

**Parameters:**
- `model_config`: Model configuration specifying architecture
- `count_data`: Single-cell count matrix (cells × genes)
- `optimizer`: NumPyro optimizer (default: Adam with lr=0.001)
- `loss`: ELBO loss function (default: TraceMeanField_ELBO)
- `n_steps`: Number of optimization steps
- `batch_size`: Mini-batch size for stochastic optimization
- `seed`: Random seed for reproducibility

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
- **`obs`**, **`var`**, **`uns`**: Metadata from AnnData objects

#### Key Analysis Methods

**Parameter Access:**
```python
# Get MAP (maximum a posteriori) estimates
map_params = results.get_map()

# Get posterior distributions
distributions = results.get_distributions()
```

**Posterior Sampling:**
```python
# Sample from variational posterior
posterior_samples = results.get_posterior_samples(
    n_samples=1000,
    seed=42
)

# Generate predictive samples
predictive_samples = results.get_predictive_samples(
    n_samples=500,
    seed=42
)

# Posterior predictive checks
ppc_samples = results.get_ppc_samples(
    n_samples=100,
    seed=42
)
```

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

# Access individual mixture components
component_0 = results.get_component(0)
```

**Data Subsetting:**
```python
# Subset results by genes or cells
gene_subset = results[:, :100]  # First 100 genes
cell_subset = results[:500, :]  # First 500 cells
```

#### Advanced Features

**Normalization:**
- Automatic count normalization using posterior estimates
- Integration with SCRIBE's normalization framework

**Statistical Analysis:**
- Hellinger and Jensen-Shannon divergence calculations
- KL divergence between distributions
- Dirichlet parameter fitting using Minka's method

**Data Integration:**
- Seamless integration with AnnData objects
- Preservation of cell and gene metadata
- Support for unstructured annotations

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
config = ModelConfig(
    base_model="zinb",
    parameterization="linked",
    unconstrained=True
)

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
config = ModelConfig(
    base_model="nbdm_mix",
    n_components=3,
    parameterization="standard"
)

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

# Get cell type assignments
cell_probs = results.cell_type_probabilities()
```

### Model Comparison

```python
# Compare different models using log-likelihood
models = ["nbdm", "zinb", "nbvcp"]
log_likelihoods = {}

for model_type in models:
    config = ModelConfig(base_model=model_type)
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
config = ModelConfig(base_model="zinb")
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
- Monitor loss history: `results.loss_history`
- Use longer training for complex models (100k+ steps)
- Try different optimizers: Adam, AdamW, RMSprop
- Adjust learning rates based on loss behavior

### Memory Management
- Use mini-batches for large datasets (`batch_size` parameter)
- Consider unconstrained parameterizations for better optimization
- Monitor memory usage with JAX profiling tools

### Numerical Stability
- Enable `stable_update=True` (default)
- Use unconstrained parameterizations when convergence is difficult
- Check for NaN values in loss history

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

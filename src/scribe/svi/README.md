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
     - `_compute_canonical_parameters()`: Convert to canonical (p, r) form
     - `_convert_to_canonical()`: Deprecated conversion method
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
   - Dependencies: None (self-contained)

5. **ComponentMixin** (`_component.py`)
   - Purpose: Mixture model component operations
   - Methods:
     - `get_component()`: Extract single component view
     - `_subset_params_by_component()`: Subset params by component index
     - `_subset_posterior_samples_by_component()`: Subset samples by component
     - `_create_component_subset()`: Create component-specific instance
   - Dependencies: Uses `GeneSubsettingMixin` for subsetting logic

6. **SamplingMixin** (`_sampling.py`)
   - Purpose: Posterior and predictive sampling
   - Methods:
     - `get_posterior_samples()`: Sample from variational posterior
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
     - `_sample_standard_model()`: Helper for standard (non-mixture) models
     - `_sample_mixture_model()`: Helper for mixture models
   - Dependencies: Uses `ModelHelpersMixin`, `ParameterExtractionMixin`

7. **LikelihoodMixin** (`_likelihood.py`)
   - Purpose: Log-likelihood computations
   - Methods:
     - `log_likelihood()`: Compute using posterior samples
     - `log_likelihood_map()`: Compute using MAP estimates
   - Dependencies: Uses `ModelHelpersMixin`, `ParameterExtractionMixin`

8. **MixtureAnalysisMixin** (`_mixture_analysis.py`)
   - Purpose: Mixture model analysis methods
   - Methods:
     - `mixture_component_entropy()`: Entropy of component assignments
     - `assignment_entropy_map()`: MAP-based assignment entropy
     - `cell_type_probabilities()`: Component probabilities from samples
     - `cell_type_probabilities_map()`: Component probabilities from MAP
   - Dependencies: Uses `LikelihoodMixin`

9. **NormalizationMixin** (`_normalization.py`)
   - Purpose: Count normalization methods
   - Methods:
     - `normalize_counts(batch_size=2048)`: Dirichlet-based normalization
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

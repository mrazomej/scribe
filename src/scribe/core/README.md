# SCRIBE Core

This directory contains the core shared components and utilities used across all
SCRIBE inference methods. The core module provides essential functionality for
data preprocessing, normalization, and cell type analysis that is used by both
SVI and MCMC inference engines.

## Overview

The core module provides:

1. **Input Processing**: Data validation, format conversion, and preprocessing
2. **Normalization**: Posterior-based count normalization and expression
   profiling
3. **Cell Type Assignment**: Advanced methods for mixture model analysis and
   cell classification

## Key Components

### InputProcessor (`input_processor.py`)

Handles data preprocessing, input validation, and model configuration for SCRIBE
inference:

```python
from scribe.core import InputProcessor

# Process count data from various formats
count_data, adata, n_cells, n_genes = InputProcessor.process_counts_data(
    counts=your_data,  # AnnData or numpy array
    cells_axis=0,      # 0=cells as rows, 1=cells as columns
    layer=None         # AnnData layer to use (None for .X)
)

# Validate model configuration
InputProcessor.validate_model_configuration(
    zero_inflated=True,
    variable_capture=False,
    mixture_model=True,
    n_components=3
)

# Determine model type automatically
model_type = InputProcessor.determine_model_type(
    zero_inflated=True,
    variable_capture=True,
    mixture_model=False
)
# Returns: "zinbvcp"
```

**Key Features:**
- **Format Flexibility**: Handles AnnData objects, numpy arrays, and sparse
  matrices
- **Automatic Conversion**: Converts data to JAX arrays with proper orientation
- **Validation**: Comprehensive input validation and error checking
- **Model Detection**: Automatic model type determination from configuration
  flags

**Supported Input Formats:**
- AnnData objects (with optional layer specification)
- NumPy arrays
- Sparse matrices (automatically converted to dense)
- JAX arrays

### Normalization (`normalization.py`)

Provides sophisticated count normalization using posterior parameter estimates:

```python
from scribe.core import normalize_counts_from_posterior

# Normalize using posterior samples
normalized = normalize_counts_from_posterior(
    posterior_samples=posterior_dict,
    n_components=None,  # Set to number for mixture models
    n_samples_dirichlet=1000,
    fit_distribution=True,
    backend="numpyro"
)
```

**Core Functionality:**

**Single-Component Normalization:**
```python
# Basic normalization for non-mixture models
normalized = normalize_counts_from_posterior(
    posterior_samples={"r": r_samples},  # Dispersion parameters
    rng_key=jax.random.PRNGKey(42),
    n_samples_dirichlet=500,
    fit_distribution=True,
    store_samples=False
)

# Access normalized profiles
mean_expression = normalized["mean_probabilities"]  # Shape: (n_genes,)
fitted_concentrations = normalized["concentrations"]  # Shape: (n_genes,)
distribution_objects = normalized["distributions"]  # Dirichlet distribution
```

**Mixture Model Normalization:**
```python
# Normalization for mixture models (component-specific)
normalized = normalize_counts_from_posterior(
    posterior_samples={"r": r_samples},  # Shape: (n_samples, n_components, n_genes)
    n_components=3,
    n_samples_dirichlet=1000,
    fit_distribution=True,
    backend="numpyro"
)

# Access component-specific profiles
mean_expression = normalized["mean_probabilities"]  # Shape: (n_components, n_genes)
concentrations = normalized["concentrations"]  # Shape: (n_components, n_genes)
distributions = normalized["distributions"]  # List of Dirichlet objects
```

**Advanced Options:**
- **Backend Selection**: Choose between NumPyro and SciPy distributions
- **Sample Storage**: Optionally store raw Dirichlet samples
- **Concentration Access**: Return original dispersion parameters
- **Flexible Sampling**: Control number of Dirichlet samples per posterior
  sample

### Cell Type Assignment (`cell_type_assignment.py`)

Advanced utilities for analyzing mixture models and assigning cell types:

```python
from scribe.core.cell_type_assignment import (
    compute_cell_type_probabilities,
    compute_cell_type_probabilities_map,
    temperature_scaling,
    hellinger_distance_weights,
    differential_expression_weights,
    top_genes_mask
)
```

#### Core Functions

**Cell Type Probability Computation:**
```python
# Compute cell type probabilities using posterior samples
cell_probs = compute_cell_type_probabilities(
    posterior_samples=posterior_dict,
    n_components=3,
    n_cells=n_cells,
    n_genes=n_genes,
    temperature=1.0,
    weighting_method="uniform",
    weight_power=2.0,
    top_genes=None
)

# MAP-based computation (faster)
cell_probs_map = compute_cell_type_probabilities_map(
    params=map_params,
    n_components=3,
    n_cells=n_cells,
    n_genes=n_genes,
    temperature=0.8,  # Sharper assignments
    weighting_method="hellinger",
    top_genes=500
)
```

**Temperature Scaling:**
```python
# Apply temperature scaling for sharper or smoother assignments
scaled_logits = temperature_scaling(
    log_probs=raw_logits,
    temperature=0.5  # < 1 for sharper, > 1 for smoother
)
```

**Advanced Weighting Methods:**

**Hellinger Distance Weights:**
```python
# Weight genes by component separability
hellinger_weights = hellinger_distance_weights(
    params=map_params,
    n_components=3,
    n_genes=n_genes,
    power=2.0,
    normalize=True
)
```

**Differential Expression Weights:**
```python
# Weight genes by differential expression strength
de_weights = differential_expression_weights(
    params=map_params,
    n_components=3,
    n_genes=n_genes,
    method="fold_change",  # or "coefficient_variation"
    power=1.5,
    normalize=True
)
```

**Top Genes Selection:**
```python
# Create mask for most informative genes
gene_mask = top_genes_mask(
    weights=combined_weights,
    top_genes=1000,
    method="absolute"  # or "relative"
)
```

## Usage Examples

### Data Preprocessing Pipeline

```python
import jax.numpy as jnp
import scanpy as sc
from scribe.core import InputProcessor

# Load single-cell data
adata = sc.read_h5ad("data.h5ad")

# Process and validate input
count_data, adata_processed, n_cells, n_genes = InputProcessor.process_counts_data(
    counts=adata,
    cells_axis=0,  # Cells as rows (standard)
    layer="counts"  # Use raw counts layer
)

# Validate model configuration
InputProcessor.validate_model_configuration(
    zero_inflated=True,
    variable_capture=False,
    mixture_model=True,
    n_components=4
)

# Determine model type
model_type = InputProcessor.determine_model_type(
    zero_inflated=True,
    variable_capture=False,
    mixture_model=True
)
print(f"Model type: {model_type}")  # "zinb_mix"
```

### Expression Normalization Workflow

```python
from scribe.core import normalize_counts_from_posterior
import jax

# After running inference (SVI or MCMC)
# posterior_samples contains parameter estimates

# Single-component model normalization
normalized_single = normalize_counts_from_posterior(
    posterior_samples=posterior_samples,
    n_components=None,
    rng_key=jax.random.PRNGKey(123),
    n_samples_dirichlet=2000,
    fit_distribution=True,
    backend="numpyro",
    verbose=True
)

# Extract normalized expression profile
expression_profile = normalized_single["mean_probabilities"]
print(f"Expression profile shape: {expression_profile.shape}")  # (n_genes,)

# For mixture models
normalized_mixture = normalize_counts_from_posterior(
    posterior_samples=mixture_posterior,
    n_components=3,
    n_samples_dirichlet=1000,
    fit_distribution=True,
    store_samples=True  # Keep raw samples for analysis
)

# Component-specific expression profiles
for i in range(3):
    profile = normalized_mixture["mean_probabilities"][i]
    print(f"Component {i} top genes:", 
          jnp.argsort(profile)[-10:])  # Top 10 expressed genes
```

### Advanced Cell Type Analysis

```python
from scribe.core.cell_type_assignment import *

# After mixture model inference
posterior_samples = results.get_posterior_samples()
map_params = results.get_map()

# Compute cell type probabilities with advanced weighting
cell_assignments = compute_cell_type_probabilities(
    posterior_samples=posterior_samples,
    n_components=4,
    n_cells=n_cells,
    n_genes=n_genes,
    temperature=0.7,  # Sharper assignments
    weighting_method="hellinger",
    weight_power=2.5,
    top_genes=800  # Use top 800 most informative genes
)

# Analyze assignment quality
assignment_entropy = -jnp.sum(
    cell_assignments * jnp.log(cell_assignments + 1e-8), 
    axis=1
)
print(f"Mean assignment entropy: {jnp.mean(assignment_entropy):.3f}")

# Identify marker genes using differential expression weights
de_weights = differential_expression_weights(
    params=map_params,
    n_components=4,
    n_genes=n_genes,
    method="fold_change",
    power=2.0
)

# Get top marker genes for each component
n_markers = 20
for component in range(4):
    # Component-specific parameters
    component_params = {k: v[component] if v.ndim > 1 else v 
                       for k, v in map_params.items()}
    
    # Find genes with highest differential expression
    marker_indices = jnp.argsort(de_weights)[-n_markers:]
    print(f"Component {component} markers: {marker_indices}")
```

### Custom Weighting Strategies

```python
# Combine multiple weighting methods
hellinger_weights = hellinger_distance_weights(
    params=map_params,
    n_components=3,
    n_genes=n_genes,
    power=2.0
)

de_weights = differential_expression_weights(
    params=map_params,
    n_components=3,
    n_genes=n_genes,
    method="coefficient_variation",
    power=1.5
)

# Combined weighting (geometric mean)
combined_weights = jnp.sqrt(hellinger_weights * de_weights)

# Apply top genes mask
top_genes_indices = top_genes_mask(
    weights=combined_weights,
    top_genes=500,
    method="absolute"
)

# Use combined strategy for cell type assignment
cell_probs = compute_cell_type_probabilities_map(
    params=map_params,
    n_components=3,
    n_cells=n_cells,
    n_genes=n_genes,
    temperature=0.6,
    weighting_method="custom",
    custom_weights=combined_weights,
    top_genes_mask=top_genes_indices
)
```

### Integration with Visualization

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Visualize cell type assignments
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Assignment probabilities heatmap
im1 = axes[0, 0].imshow(cell_assignments.T, aspect='auto', cmap='viridis')
axes[0, 0].set_title('Cell Type Assignment Probabilities')
axes[0, 0].set_xlabel('Cells')
axes[0, 0].set_ylabel('Components')
plt.colorbar(im1, ax=axes[0, 0])

# Assignment entropy distribution
axes[0, 1].hist(assignment_entropy, bins=50, alpha=0.7)
axes[0, 1].set_title('Assignment Entropy Distribution')
axes[0, 1].set_xlabel('Entropy')
axes[0, 1].set_ylabel('Frequency')

# Gene weights visualization
axes[1, 0].scatter(hellinger_weights, de_weights, alpha=0.6)
axes[1, 0].set_xlabel('Hellinger Distance Weights')
axes[1, 0].set_ylabel('Differential Expression Weights')
axes[1, 0].set_title('Gene Weighting Strategies')

# PCA of expression profiles (if mixture model)
if 'mean_probabilities' in normalized_mixture:
    profiles = normalized_mixture['mean_probabilities']  # (n_components, n_genes)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(profiles)
    
    axes[1, 1].scatter(pca_result[:, 0], pca_result[:, 1], 
                      s=100, c=range(len(pca_result)), cmap='tab10')
    axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1, 1].set_title('Component Expression Profiles (PCA)')

plt.tight_layout()
plt.show()
```

## Advanced Features

### Custom Input Processing

The InputProcessor can be extended for custom data formats:

```python
# Handle custom sparse formats
def process_custom_sparse(sparse_data):
    count_data, adata, n_cells, n_genes = InputProcessor.process_counts_data(
        counts=sparse_data.toarray(),  # Convert to dense
        cells_axis=0
    )
    return count_data, adata, n_cells, n_genes
```

### Flexible Normalization Backends

```python
# Use different statistical backends
numpyro_normalized = normalize_counts_from_posterior(
    posterior_samples=samples,
    backend="numpyro"  # Returns NumPyro distributions
)

scipy_normalized = normalize_counts_from_posterior(
    posterior_samples=samples,
    backend="scipy"  # Returns SciPy distributions
)
```

### Logistic-Normal Fitting (`normalization_logistic.py`)

Fits a low-rank Logistic-Normal distribution to posterior samples of the
concentration parameter `r`.  This preserves the gene-gene correlation
structure discovered during inference that is lost when fitting a single
Dirichlet distribution.

```python
from scribe.core.normalization_logistic import fit_logistic_normal_from_posterior

fitted = fit_logistic_normal_from_posterior(
    posterior_samples={"r": r_samples},
    n_components=3,              # None for non-mixture models
    rank=32,                     # Low-rank covariance rank
    n_samples_dirichlet=1,       # Draws per posterior sample
    batch_size=2048,             # Posterior samples per GPU batch
    verbose=True,
)

# Returns: loc, cov_factor, cov_diag, mean_probabilities, distribution(s)
```

### Performance Optimization

Both `normalize_counts_from_posterior` and `fit_logistic_normal_from_posterior`
use **batched Dirichlet sampling** to balance GPU throughput against memory
usage.  The `batch_size` parameter (default 2048) controls how many posterior
samples are processed in each JAX dispatch.  This is critical for large-scale
runs:

| Scenario | Python-to-JAX dispatches (before) | Dispatches (after, batch_size=2048) |
|---|---|---|
| 10 000 posterior samples, 1 component | 10 000 | 5 |
| 10 000 posterior samples, 5 components | 50 000 | 25 |

```python
# Memory-constrained GPU: reduce batch_size
normalized = normalize_counts_from_posterior(
    posterior_samples=samples,
    n_samples_dirichlet=1,
    batch_size=64,       # ~5 MB per batch at D=20 000
    store_samples=True,
)

# Large-memory GPU: increase batch_size for maximum throughput
fitted = fit_logistic_normal_from_posterior(
    posterior_samples=samples,
    rank=32,
    batch_size=1024,     # ~80 MB per batch at D=20 000
)
```

#### Architecture: `_fit_low_rank_mvn_core`

The SVD-based low-rank MVN fitting has been factored into a **pure-JAX
core** (`_fit_low_rank_mvn_core`) that contains no Python side-effects.
This makes it suitable for use inside `jax.jit` or `jax.vmap`.  The
verbose wrapper (`_fit_low_rank_mvn`) delegates to the core and adds
diagnostic printing.

#### Future Work: Component-Level `vmap`

For mixture models, the current implementation loops over components
sequentially.  With `_fit_low_rank_mvn_core` in place, a future
optimisation could `jax.vmap` the entire per-component pipeline
(Dirichlet sampling → ALR → SVD → embedding) to run all components
in parallel on GPU, eliminating the component loop entirely.

## Annotation Priors (`annotation_prior.py`)

This module provides utilities for injecting per-cell prior beliefs about
mixture component assignments into SCRIBE mixture models.  The key functions
are:

### `build_annotation_prior_logits`

```python
from scribe.core import build_annotation_prior_logits

# Single column
logits, label_map = build_annotation_prior_logits(
    adata,                      # AnnData with adata.obs[obs_key]
    obs_key="cell_type",        # column in adata.obs
    n_components=3,             # number of mixture components K
    confidence=3.0,             # kappa: prior strength
    component_order=None,       # optional list mapping labels to indices
)
# logits: jnp.ndarray, shape (n_cells, K) — additive logit offsets
# label_map: dict, e.g. {"T": 0, "B": 1, "Mono": 2}

# Multiple columns (composite labels)
logits, label_map = build_annotation_prior_logits(
    adata,
    obs_key=["cell_type", "treatment"],   # list of column names
    n_components=6,
    confidence=3.0,
)
# label_map: e.g. {"B__ctrl": 0, "B__stim": 1, "T__ctrl": 2, "T__stim": 3}
```

When `obs_key` is a **list**, composite labels are formed by joining per-column
values with `"__"` (double underscore).  For example, a cell with
`cell_type="T"` and `treatment="ctrl"` gets the composite label `"T__ctrl"`.
A cell is considered unlabeled if **any** of the specified columns has a
missing value.

#### How to prepare `adata.obs`

Each annotation column should be a string/categorical column in `adata.obs`.
Cells without annotations should have `NaN` (or `None`) — they receive
all-zero logits, meaning the model treats them identically to the standard
(unannotated) case.

```python
import pandas as pd
adata.obs["cell_type"] = pd.Categorical(["T", "B", None, "T", "Mono", ...])
adata.obs["treatment"] = pd.Categorical(["ctrl", "stim", "ctrl", None, ...])
```

#### The `confidence` parameter (kappa)

The confidence parameter controls how strongly the annotation influences the
per-cell prior:

| kappa | Effect |
|-------|--------|
| `0` | Annotations are completely ignored (all-zero logits, standard model) |
| `3` (default) | Annotated component gets ~exp(3) ≈ 20× prior weight boost |
| `5` | Annotated component gets ~exp(5) ≈ 150× boost |
| `→ ∞` | Hard assignment to the annotated component |

The data can always override the prior: the posterior assignment is
`p(z_i = k | x_i) ∝ pi_{i,k} · f_k(x_i | θ_k)`.

### `validate_annotation_prior_logits`

```python
from scribe.core import validate_annotation_prior_logits

# Raises ValueError if shape or finiteness checks fail
validate_annotation_prior_logits(logits, n_cells=1000, n_components=3)
```

### Usage via `scribe.fit()`

The simplest way to use annotation priors is via the high-level API:

```python
import scribe

# Single annotation column
result = scribe.fit(
    adata,
    model="nbdm",
    n_components=3,
    n_steps=50000,
    annotation_key="cell_type",       # reads from adata.obs
    annotation_confidence=3.0,        # kappa
    annotation_component_order=["T", "B", "Mono"],  # optional
)

# Multiple annotation columns (all observed combinations become labels)
result = scribe.fit(
    adata,
    model="nbdm",
    n_components=6,
    n_steps=50000,
    annotation_key=["cell_type", "treatment"],  # forms composite labels
    annotation_confidence=3.0,
    annotation_component_order=[                # optional explicit order
        "T__ctrl", "T__stim", "B__ctrl", "B__stim", "Mono__ctrl", "Mono__stim"
    ],
)
```

### Future Extensibility: Auxiliary Observation Model

The annotation prior system is designed to be extensible.  Currently it uses
a **logit-nudging** approach that modifies the mixing weights additively in
log-space.  A future **auxiliary observation model** could:

1. Replace `compute_cell_specific_mixing` with explicit discrete `z_i` sampling
   + `config_enumerate` for marginalization.
2. Add a confusion matrix parameter and an auxiliary `numpyro.sample("annotation", ...)`
   site that treats annotations as a noisy observation of the latent assignment.
3. Select the strategy via a new `annotation_strategy="logit_nudge"|"auxiliary_observation"`
   parameter — the API (`annotation_key`, `annotation_confidence`) would remain
   the same.

The `build_annotation_prior_logits` function and the label-to-component mapping
are reusable by the auxiliary observation model.

## Integration with Other Modules

- **Models**: Core utilities are used by all model types and parameterizations
- **SVI/MCMC**: Both inference engines rely on core preprocessing and
  normalization
- **Stats**: Normalization integrates with statistical analysis functions
- **Viz**: Normalized profiles and cell assignments can be visualized
- **Sampling**: Core functions support posterior and predictive sampling

## Dependencies

- **JAX**: Core numerical computations and array operations
- **NumPyro**: Probabilistic distributions and statistical functions
- **SciPy**: Statistical distributions and sparse matrix handling
- **Pandas**: Metadata handling (via AnnData integration)
- **AnnData**: Single-cell data format support (optional)

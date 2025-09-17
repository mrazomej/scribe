# SCRIBE VAE (Variational Autoencoder)

This directory contains the Variational Autoencoder (VAE) implementation for
SCRIBE models. VAEs provide a neural network-based approach to variational
inference, enabling representation learning and capturing complex non-linear
relationships in single-cell RNA sequencing data.

## Overview

The VAE module provides:

1. **Neural Network Architectures**: Flexible encoder-decoder architectures
   using Flax NNX
2. **Specialized Models**: Standard VAE and Decoupled Prior VAE (dpVAE) variants
3. **Configuration System**: Comprehensive configuration for architecture and
   training
4. **Results Analysis**: VAE-specific results class with latent space analysis

## Key Components

### VAE Architectures (`architectures.py`)

The module implements several neural network architectures for variational
inference:

#### Core Classes

**VAEConfig**: Configuration dataclass for VAE hyperparameters
```python
from scribe.vae import VAEConfig

config = VAEConfig(
    input_dim=2000,           # Number of genes
    latent_dim=10,            # Latent space dimensions
    hidden_dims=[512, 256],   # Hidden layer sizes
    activation="relu",        # Activation function
    input_transformation="log1p",  # Input preprocessing
    variable_capture=True     # Enable VCP modeling
)
```

**Encoder**: Neural network encoder that maps input to latent space
```python
from scribe.vae import create_encoder

encoder = create_encoder(
    config=vae_config,
    rngs=nnx.Rngs(42)
)
```

**Decoder**: Neural network decoder that reconstructs from latent space
```python
from scribe.vae import create_decoder

decoder = create_decoder(
    config=vae_config,
    rngs=nnx.Rngs(42)
)
```

**VAE**: Complete VAE model combining encoder and decoder
```python
from scribe.vae import create_vae

vae_model = create_vae(
    config=vae_config,
    rngs=nnx.Rngs(42)
)
```

**dpVAE**: Decoupled Prior VAE with separate modeling of different parameter
groups
```python
from scribe.vae import create_dpvae

dpvae_model = create_dpvae(
    config=vae_config,
    rngs=nnx.Rngs(42)
)
```

#### Specialized Components

**EncoderVCP**: Encoder with Variable Capture Probability (VCP) modeling
- Outputs both latent variables and capture probability parameters
- Useful for modeling technical dropout in single-cell data

**CaptureEncoder**: Specialized encoder for capture probability estimation
- Smaller network focused on modeling dropout patterns
- Integrates with main encoder for comprehensive modeling

**DecoupledPrior**: Advanced prior model for dpVAE
- Allows different parameter groups to have separate priors
- Enables more flexible modeling of parameter relationships

**AffineCouplingLayer**: Normalizing flow component for enhanced expressiveness
- Can be used to create more complex posterior approximations
- Supports invertible transformations for improved inference

### Configuration Options

#### Architecture Configuration
```python
config = VAEConfig(
    # Core architecture
    input_dim=2000,
    latent_dim=10,
    hidden_dims=[512, 256, 128],
    
    # Activation functions
    activation="relu",  # Options: relu, gelu, silu, etc.
    
    # Data preprocessing
    input_transformation="log1p",  # log1p, log, sqrt, identity
    standardize_mean=gene_means,   # Per-gene standardization
    standardize_std=gene_stds,
    
    # Variable capture modeling
    variable_capture=True,
    variable_capture_hidden_dims=[64, 32],
    variable_capture_activation="relu"
)
```

#### Available Activation Functions
- **ReLU family**: `relu`, `relu6`, `leaky_relu`
- **Modern activations**: `gelu`, `silu`, `swish`
- **Smooth activations**: `elu`, `selu`, `celu`
- **Sigmoid family**: `sigmoid`, `hard_sigmoid`, `log_sigmoid`
- **Advanced**: `softplus`, `hard_swish`, `hard_tanh`

#### Input Transformations
- **`log1p`**: log(1 + x) - handles zeros gracefully
- **`log`**: Natural logarithm - requires positive values
- **`sqrt`**: Square root transformation
- **`identity`**: No transformation

### ScribeVAEResults (`results.py`)

Specialized results class extending ScribeSVIResults with VAE-specific
functionality:

```python
from scribe.vae import ScribeVAEResults

# Results typically created by inference engine
# but can be constructed manually
results = ScribeVAEResults.from_svi_results(
    svi_results=base_results,
    vae_model=trained_vae,
    prior_type="standard"
)
```

#### Core Attributes

- **`vae_model`**: Trained VAE or dpVAE model
- **`latent_samples`**: Samples from latent space
- **`cell_embeddings`**: Cell representations in latent space
- **`prior_type`**: "standard" or "decoupled"
- **`standardize_mean/std`**: Data standardization parameters

#### Latent Space Analysis

**Latent Embeddings:**
```python
# Get cell embeddings in latent space
embeddings = results.get_latent_embeddings(
    data=count_data,
    n_samples=100,
    seed=42
)

# Conditional latent samples given data
latent_samples = results.get_latent_samples_conditioned_on_data(
    data=count_data,
    n_samples=500,
    seed=42
)
```

**Variable Capture Analysis:**
```python
# For VCP models, analyze capture probabilities
p_capture = results.get_p_capture_samples_conditioned_on_data(
    data=count_data,
    n_samples=100,
    seed=42
)
```

**Posterior Sampling:**
```python
# Sample from VAE posterior conditioned on data
posterior_samples = results.get_posterior_samples_conditioned_on_data(
    data=count_data,
    n_samples=1000,
    seed=42
)

# Standard posterior sampling
posterior_samples = results.get_posterior_samples(
    n_samples=1000,
    seed=42
)
```

## Usage Examples

### Basic VAE Setup

```python
import jax.numpy as jnp
from scribe.vae import VAEConfig, create_vae
from scribe.models import ModelConfig
from flax import nnx

# Prepare data
count_data = jnp.array(your_count_matrix)  # cells × genes
n_cells, n_genes = count_data.shape

# Configure VAE architecture
vae_config = VAEConfig(
    input_dim=n_genes,
    latent_dim=10,
    hidden_dims=[512, 256],
    activation="relu",
    input_transformation="log1p"
)

# Create VAE model
vae_model = create_vae(
    config=vae_config,
    rngs=nnx.Rngs(42)
)

# Configure for SCRIBE inference
model_config = ModelConfig(
    base_model="nbdm",
    parameterization="standard",
    inference_method="vae",
    vae_prior_type="standard",
    vae_config=vae_config
)
```

### Variable Capture Probability (VCP) Modeling

```python
# Configure VAE with VCP
vae_config = VAEConfig(
    input_dim=n_genes,
    latent_dim=15,
    hidden_dims=[512, 256, 128],
    activation="gelu",
    input_transformation="log1p",
    
    # Enable VCP modeling
    variable_capture=True,
    variable_capture_hidden_dims=[64, 32],
    variable_capture_activation="relu"
)

# Use with NBVCP or ZINBVCP models
model_config = ModelConfig(
    base_model="nbvcp",
    parameterization="linked",
    inference_method="vae",
    vae_config=vae_config
)
```

### Decoupled Prior VAE (dpVAE)

```python
# Configure dpVAE for more flexible priors
vae_config = VAEConfig(
    input_dim=n_genes,
    latent_dim=20,
    hidden_dims=[1024, 512, 256],
    activation="silu"
)

model_config = ModelConfig(
    base_model="zinb",
    parameterization="odds_ratio",
    inference_method="vae",
    vae_prior_type="decoupled",  # Use decoupled priors
    vae_config=vae_config
)
```

### Data Standardization

```python
import numpy as np

# Compute standardization statistics
gene_means = np.mean(count_data, axis=0)
gene_stds = np.std(count_data, axis=0)

# Configure VAE with standardization
vae_config = VAEConfig(
    input_dim=n_genes,
    latent_dim=10,
    hidden_dims=[512, 256],
    input_transformation="log1p",
    
    # Add standardization
    standardize_mean=jnp.array(gene_means),
    standardize_std=jnp.array(gene_stds)
)
```

### Latent Space Analysis

```python
# After training VAE model
from scribe.svi import SVIInferenceEngine

# Run VAE inference
results = SVIInferenceEngine.run_inference(
    model_config=model_config,
    count_data=count_data,
    n_cells=n_cells,
    n_genes=n_genes,
    n_steps=50000
)

# Convert to VAE results for latent analysis
vae_results = ScribeVAEResults.from_svi_results(
    svi_results=results,
    vae_model=vae_model,
    prior_type="standard"
)

# Get latent embeddings for visualization
embeddings = vae_results.get_latent_embeddings(
    data=count_data,
    n_samples=100,
    seed=42
)

# Analyze in 2D for plotting
if embeddings.shape[1] > 2:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
```

### Model Comparison: VAE vs Standard SVI

```python
# Compare VAE and SVI approaches
models_to_compare = [
    {
        "name": "SVI_MeanField",
        "config": ModelConfig(
            base_model="nbdm",
            parameterization="standard",
            inference_method="svi"
        )
    },
    {
        "name": "VAE_Standard", 
        "config": ModelConfig(
            base_model="nbdm",
            parameterization="standard",
            inference_method="vae",
            vae_prior_type="standard",
            vae_config=vae_config
        )
    },
    {
        "name": "VAE_Decoupled",
        "config": ModelConfig(
            base_model="nbdm", 
            parameterization="standard",
            inference_method="vae",
            vae_prior_type="decoupled",
            vae_config=vae_config
        )
    }
]

results = {}
for model_spec in models_to_compare:
    result = SVIInferenceEngine.run_inference(
        model_config=model_spec["config"],
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_steps=50000
    )
    results[model_spec["name"]] = result
```

## Advanced Features

### Custom Architectures

The VAE framework is highly extensible. You can create custom architectures by:

1. **Custom Encoders**: Subclass `Encoder` for specialized encoding
2. **Custom Decoders**: Subclass `Decoder` for specialized decoding  
3. **Custom VAE Models**: Subclass `VAE` or `dpVAE` for novel architectures
4. **Normalizing Flows**: Use `AffineCouplingLayer` for flow-based models

### Integration with SCRIBE Models

VAEs integrate seamlessly with all SCRIBE model types:

- **NBDM**: Basic negative binomial modeling
- **ZINB**: Zero-inflated negative binomial
- **NBVCP**: With variable capture probability
- **ZINBVCP**: Zero-inflated with VCP
- **Mixture Models**: All models support mixture variants

### Memory and Performance

**Optimization Tips:**
- Use appropriate batch sizes for your hardware
- Consider gradient checkpointing for large models
- Use mixed precision training when available
- Monitor memory usage with JAX profiling

**Architecture Guidelines:**
- Start with 2-3 hidden layers
- Use 128-512 units per layer for most datasets
- Latent dimensions of 10-50 work well for most applications
- Enable VCP for datasets with high dropout

## Dependencies

- **Flax NNX**: Neural network implementation
- **JAX**: Automatic differentiation and compilation  
- **NumPyro**: Probabilistic programming integration
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions (optional)
- **scikit-learn**: Dimensionality reduction for visualization (optional)

## Integration with Other Modules

- **Models**: VAE models are registered in the model registry
- **SVI**: VAE inference uses the same SVI framework
- **Sampling**: Supports all SCRIBE sampling methods
- **Stats**: Compatible with statistical analysis functions
- **Viz**: Latent embeddings can be visualized using the viz module

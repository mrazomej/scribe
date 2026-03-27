# Quickstart

This tutorial walks you through a basic SCRIBE analysis. For a deeper
understanding of the models and methods, see the
[Quick Overview](quick-overview.md) and the
[Model Selection](../guide/model-selection.md) guide.

## Basic Workflow

```python
import scribe
import anndata as ad

# Load your scRNA-seq data
adata = ad.read_h5ad("data.h5ad")

# Run inference with the default NBDM model
results = scribe.fit(adata, n_steps=100_000)

# Inspect convergence
print(results.loss_history[-10:])

# Get posterior parameter distributions
distributions = results.get_distributions()

# Generate posterior predictive samples for model validation
ppc_samples = results.get_ppc_samples(n_samples=100)

# Visualize results
scribe.viz.plot_parameter_posteriors(results)
```

!!! tip "Choosing an inference method"
    Start with `inference_method="svi"` for fast iteration. Once you are
    satisfied with the model choice, switch to `inference_method="mcmc"` for
    publication-quality posterior distributions (requires GPU with double
    precision support).

## Next Steps

- Explore the [Model Selection](../guide/model-selection.md) guide to find the
  right model for your data
- Learn about the [Results class](../guide/results.md) for downstream analysis
- See the [`scribe.fit()` Interface](../guide/fit.md) for full parameter
  documentation

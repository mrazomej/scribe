# Custom Models

SCRIBE provides a flexible framework for implementing and working with custom
models while maintaining compatibility with the package's infrastructure. This
tutorial walks you through the process of creating and using custom models,
using a real example of modifying the [NBDM](../models/nbdm.md) model to use a
[LogNormal](https://en.wikipedia.org/wiki/Log-normal_distribution) prior.

## Overview

Creating a custom model in SCRIBE involves several key components:

1. Defining the model function
2. Defining the guide function
3. Specifying parameter types (either `global`, `gene-specific`, or
   `cell-specific`)
4. Running inference using `run_scribe`
5. Working with the results

Let's go through each step in detail. First, we begin with the needed imports:

```python
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro
import scribe
```

## Defining the Model

The model function defines your probabilistic model using NumPyro primitives.
For this tutorial, we will modify the [NBDM](../models/nbdm.md) model to use a
LogNormal prior for the dispersion parameters. The function must have the
following signature:

- `n_cells` — the number of cells
- `n_genes` — the number of genes
- `param_prior` — the parameters used for the prior distribution of the
  parameters (define one entry per parameter; in our case `p_prior` and
  `r_prior`)
- `counts` — the count data
- `custom_arg` — any additional arguments needed by the model (define one entry
  per argument; in our case `total_counts`)
- `batch_size` — the batch size for mini-batch training

```python
def nbdm_lognormal_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (0, 1),  # Changed to mean, std for lognormal
    counts=None,
    total_counts=None,
    batch_size=None,
):
    # Define success probability prior (unchanged)
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define dispersion prior using LogNormal instead of Gamma
    r = numpyro.sample(
        "r",
        dist.LogNormal(r_prior[0], r_prior[1]).expand([n_genes]),
    )

    # Define the total dispersion parameter
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts,
                )
            with numpyro.plate("cells", n_cells):
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts),
                    obs=counts,
                )
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx],
                )
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(
                        r, total_count=total_counts[idx]
                    ),
                    obs=counts[idx],
                )
    else:
        with numpyro.plate("cells", n_cells):
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            counts = numpyro.sample("counts", dist_nb)
```

### Dissecting the Model

**Prior definitions**: We define the prior for the success probability `p` as a
Beta distribution and the dispersion parameter `r` as a LogNormal distribution.
Since `r` is a gene-specific parameter, we use `.expand([n_genes])` to create
one dispersion parameter per gene, all sharing the same prior.

**Total dispersion**: We define `r_total` as the deterministic sum of all
individual dispersion parameters using `numpyro.deterministic`.

**Likelihood**: The model handles three cases:

1. **Observed data, no batching** — condition on the entire dataset each step
2. **Observed data with batching** — use `subsample_size` in `numpyro.plate`
   for memory-efficient mini-batch training
3. **No observed data** — return the predictive distribution for posterior
   predictive sampling

The likelihood uses `numpyro.plate("cells", n_cells)` to declare \(N\)
i.i.d. observations:

\[
\pi(U_1, \ldots, U_{n_\text{cells}} \mid r_i, p) =
\prod_{i=1}^{n_\text{cells}} \pi(U_i \mid r_i, p)
\tag{1}
\]

For the predictive case, `.to_event(1)` tells NumPyro that the \(G\)
independent Negative Binomial distributions represent a single cell's worth
of counts (a "multivariate" distribution).

!!! note
    The count data must be in shape `(n_cells, n_genes)` for mini-batch
    indexing to work correctly.

### Key Requirements

- Must accept `n_cells` and `n_genes` as first arguments
- Should handle both training (`counts is not None`) and predictive
  (`counts is None`) cases
- Must use NumPyro primitives for all random variables
- Should support mini-batch training through `batch_size`

## Defining the Guide

The guide function defines the variational distribution used to approximate the
posterior. We use a **mean-field** approximation, meaning each parameter's
posterior is independent of the others.

!!! info "Why mean-field?"
    A full covariance structure for ~20k genes would require ~400M parameters.
    The mean-field approximation trades off correlation structure for
    computational tractability.

The guide must have the **same signature** as the model function:

```python
def nbdm_lognormal_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (0, 1),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    # Parameters for p (using Beta)
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(p_prior[0]),
        constraint=numpyro.distributions.constraints.positive,
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(p_prior[1]),
        constraint=numpyro.distributions.constraints.positive,
    )

    # Parameters for r (using LogNormal)
    mu_r = numpyro.param(
        "mu_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=numpyro.distributions.constraints.real,
    )
    sigma_r = numpyro.param(
        "sigma_r",
        jnp.ones(n_genes) * r_prior[1],
        constraint=numpyro.distributions.constraints.positive,
    )

    # Sample from variational distributions
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.LogNormal(mu_r, sigma_r))
```

Parameters are registered using `numpyro.param` with appropriate constraints
(e.g., `positive` for concentration parameters, `real` for location
parameters). The sample names must match those in the model exactly.

!!! tip
    You are free to choose any distribution for the variational posterior. The
    distributions chosen here are natural choices given the parameter
    constraints, but alternatives are valid.

### Key Points

- Must match the model's signature exactly
- Parameters should be registered using `numpyro.param`
- Use appropriate constraints for parameters
- Sample from variational distributions using same names as model

## Specifying Parameter Types

SCRIBE needs to know how to handle different parameters for correct indexing:

```python
param_spec = {
    "alpha_p": {"type": "global"},
    "beta_p": {"type": "global"},
    "mu_r": {"type": "gene-specific"},
    "sigma_r": {"type": "gene-specific"},
}
```

Each parameter must be categorized as one of:

- `"global"` — single value shared across all cells/genes
- `"gene-specific"` — one value per gene
- `"cell-specific"` — one value per cell

!!! note
    For mixture models, add `"component_specific": True` to parameters that
    vary by component.

## Running Inference

Once the model, guide, and param_spec are defined, pass them to `run_scribe`:

```python
results = scribe.run_scribe(
    counts=counts,
    custom_model=nbdm_lognormal_model,
    custom_guide=nbdm_lognormal_guide,
    custom_args={"total_counts": jnp.sum(counts, axis=1)},
    param_spec=param_spec,
    n_steps=10_000,
    batch_size=512,
    prior_params={"p_prior": (1, 1), "r_prior": (0, 1)},
)
```

Key arguments:

- `custom_model` — your model function
- `custom_guide` — your guide function
- `custom_args` — additional arguments needed by your model/guide
- `param_spec` — parameter type specification
- `prior_params` — prior parameters for your model

## Working with Results

Results from custom models are returned as `CustomResults` objects with the
same interface as built-in models:

```python
params = results.params
distributions = results.get_distributions()
samples = results.get_posterior_samples(n_samples=1000)
predictions = results.get_predictive_samples()
```

## Optional Extensions

### Custom Distribution Access

Define a function that maps raw variational parameters to distribution objects:

```python
def get_distributions_fn(params, backend="scipy"):
    if backend == "scipy":
        return {
            "p": stats.beta(params["alpha_p"], params["beta_p"]),
            "r": stats.lognorm(
                s=params["sigma_r"], scale=np.exp(params["mu_r"])
            ),
        }
    elif backend == "numpyro":
        return {
            "p": dist.Beta(params["alpha_p"], params["beta_p"]),
            "r": dist.LogNormal(params["mu_r"], params["sigma_r"]),
        }

results = scribe.run_scribe(..., get_distributions_fn=get_distributions_fn)
```

!!! warning
    Sometimes the parameterization between `scipy` and NumPyro differs. Check
    the documentation for the distribution you are using to ensure the correct
    parameterization.

### Custom Model Arguments

```python
def get_model_args_fn(results):
    return {
        "n_cells": results.n_cells,
        "n_genes": results.n_genes,
        "my_custom_arg": results.custom_value,
    }

results = scribe.run_scribe(..., get_model_args_fn=get_model_args_fn)
```

### Custom Log Likelihood

```python
def custom_log_likelihood_fn(counts, params):
    # Compute log likelihood
    return log_prob

results = scribe.run_scribe(
    ..., custom_log_likelihood_fn=custom_log_likelihood_fn
)
```

## Best Practices

1. **Model Design**: Start from existing models when possible. Keep track of
   dimensionality (cells vs genes). Use appropriate constraints. Support both
   training and prediction modes.

2. **Guide Design**: Match model parameters exactly. Initialize variational
   parameters sensibly. Use mean-field approximation when possible. Consider
   parameter constraints carefully.

3. **Parameter Specification**: Be explicit about parameter types. Consider
   dimensionality requirements. Document parameter relationships. Test with
   small datasets first.

4. **Testing**: Verify model runs with small datasets. Check parameter ranges
   make sense. Test both training and prediction. Validate results against
   known cases.

## Common Issues

- **Dimension Mismatch**: Check parameter shapes, verify broadcast operations,
  ensure mini-batch handling is correct.
- **Memory Issues**: Use appropriate batch sizes, avoid unnecessary parameter
  expansion, monitor device memory usage.
- **Numerical Stability**: Use appropriate parameter constraints, consider
  log-space computations, initialize parameters carefully.
- **Convergence Problems**: Check learning rate and optimization settings,
  monitor loss during training, verify parameter updates occur.

## See Also

- [NBDM model](../models/nbdm.md) — details on the base NBDM model
- [Results class](results.md) — working with result objects
- [NumPyro documentation](https://num.pyro.ai/en/stable/) — distribution
  details

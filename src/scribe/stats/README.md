# Statistics Module

The `stats` module provides statistical functions and custom probability
distributions for the SCRIBE package.

## Module Structure

```
stats/
├── __init__.py          # Public API exports
├── README.md            # This file
├── histogram.py         # Histogram functions
├── ecdf.py              # ECDF functions
├── dirichlet.py         # Dirichlet fitting functions
├── distributions.py     # Custom distributions
├── divergences.py       # KL, Jensen-Shannon, Hellinger functions
└── patches.py           # Mode property patches
```

## Submodules

### histogram.py

Functions for computing histogram statistics across posterior samples:

- **`compute_histogram_percentiles(samples, percentiles, normalize,
  sample_axis)`**: Compute percentiles of histogram frequencies across multiple
  samples.

- **`compute_histogram_credible_regions(samples, credible_regions, normalize,
  sample_axis, batch_size, max_bin)`**: Compute credible regions of histogram
  frequencies across multiple samples.

### ecdf.py

Functions for computing empirical cumulative distribution functions (ECDFs):

- **`compute_ecdf_percentiles(samples, percentiles, sample_axis)`**: Compute
  percentiles of ECDF values across multiple samples of integers.

- **`compute_ecdf_credible_regions(samples, credible_regions, sample_axis,
  batch_size, max_bin)`**: Compute credible regions of ECDF values across
  multiple samples.

### dirichlet.py

Functions for working with Dirichlet distributions:

- **`sample_dirichlet_from_parameters(parameter_samples, n_samples_dirichlet,
  rng_key)`**: Samples from a Dirichlet distribution given an array of parameter
  samples.  This function already supports batched input: passing an
  `(N, D)` array of concentration parameters creates `N` Dirichlet distributions
  and draws from all of them in a single JAX dispatch.  The
  `core.normalization_logistic._batched_dirichlet_sample` helper wraps this
  with chunked iteration for memory-safe processing of very large `N`.

- **`fit_dirichlet_mle(samples, max_iter, tol, sample_axis)`**: Fit a Dirichlet
  distribution to samples using Maximum Likelihood Estimation (Newton's method).

- **`fit_dirichlet_minka(samples, max_iter, tol, sample_axis)`**: Fit a
  Dirichlet distribution using Minka's fixed-point iteration (generally more
  stable).

- **`digamma_inv(y, num_iters)`**: Approximate the inverse of the digamma
  function using Newton iterations.

### distributions.py

Custom probability distribution classes:

- **`BetaPrime(concentration1, concentration0)`**: Beta Prime distribution with
  odds-of-Beta convention. Used for modeling odds ratios in the SCRIBE model.

- **`LowRankLogisticNormal(loc, cov_factor, cov_diag)`**: Low-rank
  Logistic-Normal distribution for compositional data using the Additive
  Log-Ratio (ALR) transformation.
  - Supports `log_prob()` evaluation
  - Uses last component as reference (asymmetric)
  - Memory efficient: O((D-1) × rank) instead of O((D-1)²)
  - Ideal for large gene sets (30K+)

- **`SoftmaxNormal(loc, cov_factor, cov_diag)`**: Softmax-Normal distribution
  for compositional data with symmetric treatment of all components.
  - Does NOT support `log_prob()` (softmax is singular)
  - Treats all components equally (symmetric)
  - Memory efficient: O(D × rank) instead of O(D²)
  - Use for sampling and visualization

### divergences.py

Functions for computing divergences and distances between distributions:

#### KL Divergence

- **`_kl_betaprime(p, q)`**: KL divergence between BetaPrime distributions
- **`_kl_lognormal(p, q)`**: KL divergence between LogNormal distributions

These are registered with NumPyro's `kl_divergence` dispatcher.

#### Jensen-Shannon Divergence

Symmetric divergence measure: JS(P||Q) = 0.5 * (KL(P||Q) + KL(Q||P))

- **`jensen_shannon(p, q)`**: Overloaded for Beta, BetaPrime, Normal, LogNormal

#### Hellinger Distance

Bounded distance metric between probability distributions:

- **`sq_hellinger(p, q)`**: Squared Hellinger distance (overloaded for Beta,
  BetaPrime, Normal, LogNormal)
- **`hellinger(p, q)`**: Hellinger distance (overloaded for Beta, BetaPrime,
  Normal, LogNormal)

### patches.py

Mode property patches for NumPyro distributions:

- **`apply_distribution_mode_patches()`**: Adds `mode` properties to Beta,
  LogNormal, and Normal distributions.

## Usage Examples

### Working with Compositional Data

```python
from jax import random
import jax.numpy as jnp
from scribe.stats import LowRankLogisticNormal, SoftmaxNormal

# Create a low-rank logistic-normal for gene expression (5 genes)
D = 5
rank = 2
loc = jnp.zeros(D - 1)  # ALR uses D-1 dimensions
cov_factor = jnp.ones((D - 1, rank)) * 0.1
cov_diag = jnp.ones(D - 1) * 0.5

# ALR-based distribution (can evaluate log_prob)
alr_dist = LowRankLogisticNormal(loc, cov_factor, cov_diag)
samples = alr_dist.sample(random.PRNGKey(0), (100,))
log_probs = alr_dist.log_prob(samples)  # Works!

# Softmax-based distribution (symmetric, no log_prob)
loc_softmax = jnp.zeros(D)  # Softmax uses D dimensions
cov_factor_softmax = jnp.ones((D, rank)) * 0.1
cov_diag_softmax = jnp.ones(D) * 0.5

softmax_dist = SoftmaxNormal(loc_softmax, cov_factor_softmax, cov_diag_softmax)
samples = softmax_dist.sample(random.PRNGKey(0), (100,))
# softmax_dist.log_prob(samples)  # Raises NotImplementedError!
```

### Fitting Dirichlet Distributions

```python
from scribe.stats import fit_dirichlet_minka
import jax.numpy as jnp

# Generate some compositional data (e.g., gene expression proportions)
samples = jnp.array([
    [0.2, 0.3, 0.5],
    [0.25, 0.35, 0.4],
    [0.15, 0.45, 0.4],
])

# Fit Dirichlet using Minka's method
alpha = fit_dirichlet_minka(samples)
print(f"Fitted concentration parameters: {alpha}")
```

### Computing Divergences

```python
from scribe.stats import jensen_shannon, hellinger
from numpyro.distributions import Beta

p = Beta(2.0, 5.0)
q = Beta(3.0, 4.0)

js_div = jensen_shannon(p, q)  # Jensen-Shannon divergence
h_dist = hellinger(p, q)  # Hellinger distance
```

## Mathematical Background

### Logistic-Normal Distribution

The Logistic-Normal distribution is a natural choice for modeling compositional
data (data on the simplex). It allows for correlation between components, unlike
the Dirichlet distribution.

#### ALR Transformation

For D-dimensional simplex data, the Additive Log-Ratio (ALR) transformation maps
to (D-1)-dimensional unconstrained space:

```
y_i = log(x_i / x_D)  for i = 1, ..., D-1
```

where x_D is the reference component.

The inverse transformation is:

```
x_i = exp(y_i) / (1 + Σ_j exp(y_j))  for i = 1, ..., D-1
x_D = 1 / (1 + Σ_j exp(y_j))
```

The Jacobian correction for the log probability is:

```
log|det(J)| = -Σ_{i=1}^D log(x_i)
```

#### Low-Rank Covariance

For large D (e.g., 30K genes), storing a full covariance matrix is prohibitive.
We use a low-rank plus diagonal structure:

```
Σ = WW^T + diag(D)
```

where W is (D-1) × rank. This reduces memory from O(D²) to O(D × rank).

### Dirichlet Fitting

The Minka method uses the relation:

```
ψ(α_j) - ψ(α_0) = ⟨ln x_j⟩
```

where ψ is the digamma function and α_0 = Σ_k α_k, leading to the fixed-point
update:

```
α_j ← ψ^(-1)(ψ(α_0) + ⟨ln x_j⟩)
```

This is generally more stable than moment matching or MLE via gradient descent.

## References

- Aitchison, J. (1986). The Statistical Analysis of Compositional Data. Chapman
  & Hall.
- Minka, T. (2000). Estimating a Dirichlet distribution. Technical report, MIT.


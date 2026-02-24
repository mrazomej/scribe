# Bayesian Differential Expression Analysis

This module provides a comprehensive framework for Bayesian differential
expression (DE) analysis of compositional data. It is designed to work with the
logistic-normal distributions fitted by `fit_logistic_normal_from_posterior`.

## Overview

The differential expression framework is:

- **Compositional**: Works in CLR (Centered Log-Ratio) or ILR (Isometric
  Log-Ratio) space, ensuring reference-invariant results
- **Correlation-aware**: Leverages low-rank covariance structure to account for
  gene-gene correlations
- **Fully Bayesian**: Provides exact posterior probabilities (not frequentist
  p-values) under the Gaussian assumption
- **Computationally efficient**: All operations are O(kG) or O(k²G) where k <<
  G, avoiding O(G³) dense matrix operations

## Quick Start

```python
from scribe.de import compare
import jax.numpy as jnp

# Fit logistic-normal models for two conditions
model_A = results_A.fit_logistic_normal(rank=16)
model_B = results_B.fit_logistic_normal(rank=16)

# Create structured comparison
de = compare(model_A, model_B,
             gene_names=adata.var_names.tolist(),
             label_A="Treatment", label_B="Control")

# Gene-level DE
results = de.gene_level(tau=jnp.log(1.1))

# Call DE genes (Bayesian decision)
is_de = de.call_genes(lfsr_threshold=0.05, prob_effect_threshold=0.95)
print(f"Found {is_de.sum()} DE genes")

# Check Bayesian FDR
pefp = de.compute_pefp(threshold=0.05)
print(f"Expected false discovery proportion: {pefp:.3f}")

# Display top genes
print(de.summary(sort_by='lfsr', top_n=20))
```

## API Reference

### `compare()` factory

The recommended entry point.  Returns a `ScribeParametricDEResults`,
`ScribeEmpiricalDEResults`, or `ScribeShrinkageDEResults` depending on
`method=`:

```python
# Results-object interface (recommended for empirical / shrinkage).
# Automatically extracts r, p (for hierarchical models), and gene names.
de = compare(results_bleo, results_ctrl,
             method="empirical",
             component_A=0, component_B=0)

# Shrinkage with results objects (runs CLR sampling from scratch)
de = compare(results_bleo, results_ctrl,
             method="shrinkage",
             component_A=0, component_B=0)

# Preferred: run empirical first, then wrap with .shrink() (zero-copy)
de_emp = compare(results_bleo, results_ctrl,
                 method="empirical",
                 component_A=0, component_B=0)
de_shrink = de_emp.shrink()  # reuses delta_samples — no extra GPU memory

# Parametric (requires pre-fitted logistic-normal models)
de = compare(model_A, model_B, gene_names=names)

# Raw-array interface (still supported for full control)
de = compare(r_A, r_B, method="empirical", component_A=0, component_B=0,
             gene_names=names, p_samples_A=p_A, p_samples_B=p_B)
```

When results objects are passed, `compare()`:

- Extracts `r` samples from `posterior_samples["r"]`
- Auto-detects hierarchical models (gene-specific `p`) via
  `model_config.is_hierarchical` and extracts `posterior_samples["p"]`
- Infers gene names from `results.var.index`
- Silently drops `gene_mask` when gene-specific `p` is present
  (the two are mutually exclusive)

**Common methods (all subclasses):**

| Method | Description |
|--------|-------------|
| `de.gene_level(tau)` | Per-gene posterior summaries (returns `lfsr` and `lfsr_tau`) |
| `de.call_genes(tau, lfsr_threshold, prob_effect_threshold)` | Bayesian gene calling (tau-aware caching) |
| `de.test_contrast(contrast, tau)` | Custom linear contrast |
| `de.test_gene_set(indices, tau)` | Pathway enrichment via ILR balance (parametric: Gaussian; empirical: Monte Carlo) |
| `de.compute_pefp(threshold, tau, use_lfsr_tau)` | Posterior expected FDP |
| `de.find_threshold(target_pefp, tau, use_lfsr_tau)` | Find lfsr threshold for PEFP control |
| `de.summary(tau, sort_by, top_n)` | Formatted results table |

**Empirical/Shrinkage-only methods:**

| Method | Description |
|--------|-------------|
| `de.test_pathway_perturbation(indices, n_permutations)` | Within-pathway compositional perturbation test |
| `de.test_multiple_gene_sets(gene_sets, tau, target_pefp)` | Batch pathway testing with PEFP control |

> **Note on `tau`-aware caching**: All methods that depend on gene-level results
> accept a `tau` parameter. Results are cached and automatically recomputed when
> `tau` changes. This prevents the stale-cache bug where calling
> `de.call_genes(tau=0.5)` after `de.gene_level(tau=0.1)` would silently use
> results from the wrong tau.

### Full Pipeline Example

```python
from scribe.de import compare
import jax.numpy as jnp

# 1. Create comparison
de = compare(model_A, model_B,
             gene_names=gene_names,
             label_A="WT", label_B="KO")

# 2. Gene-level analysis with practical significance threshold
results = de.gene_level(tau=jnp.log(1.1))

# 3. Find threshold controlling Bayesian FDR
threshold = de.find_threshold(target_pefp=0.05)
print(f"lfsr threshold for 5% PEFP: {threshold:.3f}")

# 4. Call genes at that threshold
is_de = de.call_genes(lfsr_threshold=threshold)

# 5. Test specific pathways
pathway = jnp.array([10, 25, 42, 101, 200])
pathway_result = de.test_gene_set(pathway, tau=jnp.log(1.1))
print(f"Pathway effect: {pathway_result['delta_mean']:.3f}")

# 6. Summary
print(de.summary(sort_by='lfsr', top_n=30))
```

### Standalone Functions

For users who prefer functional style over the results class:

```python
from scribe.de import (
    # Gene-level
    differential_expression,
    call_de_genes,
    # Set-level (parametric)
    test_gene_set,
    test_contrast,
    build_balance_contrast,
    # Set-level (empirical)
    empirical_test_gene_set,
    empirical_test_pathway_perturbation,
    empirical_test_multiple_gene_sets,
    # Transformations
    build_ilr_balance,
    build_pathway_sbp_basis,
    # Error control & utilities
    compute_pefp,
    find_lfsr_threshold,
    format_de_table,
    extract_alr_params,
)
```

## Key Concepts

### Bayesian Error Control

Unlike frequentist methods that use FDR (False Discovery Rate), this module
uses:

- **lfsr (Local False Sign Rate)**: Posterior probability of having the wrong
  sign for each gene. Exact under the Gaussian assumption.
- **lfsr_tau (Modified Local False Sign Rate)**: Incorporates practical
  significance as defined in the paper:

  ```
  lfsr_g(tau) = 1 - max(P(Delta_g > tau | data), P(Delta_g < -tau | data))
  ```

  When `tau=0` this reduces to the standard `lfsr`. Use
  `de.compute_pefp(use_lfsr_tau=True)` or
  `de.find_threshold(use_lfsr_tau=True)` to base error control on `lfsr_tau`.
  The default remains the standard `lfsr` with a separate two-threshold decision
  rule (lfsr + prob_effect).
- **PEFP (Posterior Expected False Discovery Proportion)**: Bayesian analogue of
  FDR. Computed as the average lfsr (or lfsr_tau) of called genes.

These are true posterior probabilities computed from the data, not frequentist
error rates.

### Compositional Data Analysis

Gene expression data is compositional (counts sum to a total). Standard analysis
methods can give misleading results. This module:

1. Works in CLR space where log-ratios are centered
2. Uses ALR→CLR transformations that preserve the correlation structure
3. Provides gene-set analysis via compositional balances

### Coordinate Systems

- **ALR (Additive Log-Ratio)**: `z_i = log(ρ_i) - log(ρ_G)` for i=1,...,G-1
  - Asymmetric (depends on reference gene G)
  - Used internally for parameter fitting
  - Dimension: G-1

- **CLR (Centered Log-Ratio)**: `z_i = log(ρ_i) - (1/G)Σ_j log(ρ_j)`
  - Symmetric (no reference gene)
  - Used for reporting gene-level effects
  - Dimension: G (constrained, sums to 0)

- **ILR (Isometric Log-Ratio)**: Orthonormal basis of CLR subspace
  - Dimension: G-1
  - Useful for gene-set analysis

### Exact Transformations

The ALR→CLR transformation preserves the low-rank structure:
- Mean: `μ_clr = H·μ_alr` (where H embeds and centers)
- Low-rank factor: `W_clr = H·W_alr`
- Diagonal: `d_clr` computed exactly via centering formula

### Posterior Distribution

Under the fitted logistic-normal models:
- `z_A ~ N(μ_A, Σ_A)` in CLR space
- `z_B ~ N(μ_B, Σ_B)` in CLR space

The difference is also Gaussian:
- `Δ = z_A - z_B ~ N(μ_A - μ_B, Σ_A + Σ_B)`

This gives exact analytic posteriors for:
- Each gene: `Δ_g ~ N(μ_A[g] - μ_B[g], σ²_A[g] + σ²_B[g])`
- Any contrast: `c^T Δ ~ N(c^T(μ_A - μ_B), c^T(Σ_A + Σ_B)c)`

## Gene-Specific p: Gamma-Based Composition Sampling

When using hierarchical parameterizations that produce gene-specific `p_g`
values, the standard `Dirichlet(r)` sampling for compositions is no longer
correct. The `compare()` function and `compute_clr_differences()` accept
optional `p_samples_A` and `p_samples_B` arrays of shape `(N, K, D)` (or
`(N, D)` for non-mixture models). When provided, compositions are generated
via scaled Gamma variates instead of Dirichlet:

```
lambda_g ~ Gamma(r_g, rate=1) * p_g / (1 - p_g)
rho_g = lambda_g / sum_j lambda_j
```

This is implemented by `_batched_gamma_normalize()` in `_empirical.py`. When
all `p_g` are equal, this reduces exactly to `Dirichlet(r)` sampling, so the
method is a strict generalization.

```python
from scribe.de import compare

de = compare(
    r_bleo, r_ctrl,
    p_samples_A=results_bleo.posterior_samples.get("p"),
    p_samples_B=results_ctrl.posterior_samples.get("p"),
    method="empirical",
    component_A=0, component_B=0,
    gene_names=gene_names,
)
```

**Note**: Combining `gene_mask` with `p_samples` raises a `ValueError` because
gene aggregation under gene-specific probabilities is ill-defined.

## Gene Expression Filter (`gene_mask`)

Low-expression genes can appear spuriously DE due to compositional artefacts:
when a dominant gene changes, the CLR geometric mean shifts, making every other
gene look different.  Both DE paths support a `gene_mask` parameter that
aggregates filtered genes into a single "other" pseudo-gene before Dirichlet
sampling.

### Quick start

```python
from scribe.de import compare, compute_expression_mask

# Build a boolean mask from MAP mean expression (mu)
mask = compute_expression_mask(
    results_A, results_B,
    component_A=0, component_B=0,
    min_mean_expression=1.0,
)

# Empirical path — pass gene_mask to compare()
de = compare(
    r_A, r_B,
    method="empirical",
    component_A=0, component_B=0,
    gene_names=gene_names,
    gene_mask=mask,
)

# Parametric path — pass gene_mask at fit_logistic_normal time
fitted_A = results_A.fit_logistic_normal(gene_mask=mask)
fitted_B = results_B.fit_logistic_normal(gene_mask=mask)
de = compare(fitted_A, fitted_B, gene_names=gene_names, gene_mask=mask)
```

### How it works

1. Genes marked `False` in `gene_mask` have their Dirichlet concentrations
   **summed** into a single "other" pseudo-gene, preserving total concentration.
2. Dirichlet sampling, ALR/CLR transformation, and all DE statistics operate on
   the reduced `(D_kept + 1)`-simplex.
3. The "other" column is dropped from final results; only `D_kept` genes appear
   in the output.

### `compute_expression_mask()`

Builds a mask from MAP mean expression (`mu`):
- A gene passes if `mu >= threshold` in **either** condition (preserving
  genuinely condition-specific genes).
- Alternatively, users can pass any boolean mask (e.g. based on raw count
  quantiles) directly to `compare()`.

## Empirical (Non-Parametric) DE

When the Gaussian assumption fails (as indicated by Gaussianity diagnostics),
the **empirical** DE path computes all statistics directly from posterior samples
via Monte Carlo counting — no distributional assumptions required.

### Quick start

```python
from scribe.de import compare

# Independent models — empirical path
de = compare(
    posterior_samples_bleo["r"],   # (N, K, D) concentration samples
    posterior_samples_ctrl["r"],
    method="empirical",
    component_A=0, component_B=0,  # which mixture component in each
    gene_names=gene_names,
    label_A="Bleomycin", label_B="Control",
)

# Same interface as parametric
results = de.gene_level(tau=jnp.log(1.1))
is_de = de.call_genes(lfsr_threshold=0.05)
print(de.summary(sort_by="lfsr", top_n=20))
```

### Within-mixture comparison (paired)

When comparing two components from the **same** mixture model, the posterior
samples are correlated (they come from the same variational draw).  Use
`paired=True` to preserve this correlation:

```python
de = compare(
    posterior_samples["r"],        # same array for both
    posterior_samples["r"],
    method="empirical",
    component_A=0, component_B=1,  # compare component 0 vs 1
    paired=True,
    gene_names=gene_names,
)
```

### How it works

1. **Composition sampling**: Draw `rho ~ Dirichlet(r)` from the concentration
   parameters in batches (GPU-friendly). When gene-specific `p_samples` are
   provided, uses Gamma-based sampling instead (see above).
2. **CLR transform**: `CLR(rho) = log(rho) - mean(log(rho))`.
3. **Pair and difference**: `Delta = CLR(rho_A) - CLR(rho_B)`.
4. **Count**: Estimate all statistics by vectorized counting:
   - `lfsr_g = min(P(Delta_g > 0), P(Delta_g < 0))`
   - `prob_effect_g = P(|Delta_g| > tau)`
   - etc.

### Validity of pairing

For **independent models**, the joint posterior factorises:
`pi(rho_A, rho_B | data_A, data_B) = pi(rho_A | data_A) * pi(rho_B | data_B)`,
so any pairing of samples is valid.

For **within-mixture**, paired indices preserve the joint posterior structure.

### Resolution

With N = 10,000 posterior samples, lfsr resolves to 1/N = 0.0001.  The
standard error is `SE(lfsr) = sqrt(lfsr * (1 - lfsr) / N)` ≈ 0.001 for
lfsr = 0.01.

### Class hierarchy

```
ScribeDEResults (base)
├── ScribeParametricDEResults   — analytic Gaussian (loc, W, d)
├── ScribeEmpiricalDEResults    — Monte Carlo counting (delta_samples)
└── ScribeShrinkageDEResults    — empirical Bayes shrinkage (extends Empirical)
```

The `compare()` factory returns the appropriate subclass based on `method=`.
All shared methods (`call_genes`, `compute_pefp`, `find_threshold`, `summary`)
work identically on all three.

## Empirical Bayes Shrinkage

When using the empirical path, the per-gene lfsr values are computed
independently for each gene.  The **shrinkage** method improves on this by
learning a genome-wide effect-size distribution (a scale mixture of normals)
and using it to update each gene's posterior.  The result is adaptive shrinkage:
noisy effect estimates are pulled toward zero, with the degree of shrinkage
determined by the data.

### Quick start

```python
from scribe.de import compare

de = compare(
    posterior_samples_bleo["r"],
    posterior_samples_ctrl["r"],
    method="shrinkage",
    component_A=0, component_B=0,
    gene_names=gene_names,
    label_A="Bleomycin", label_B="Control",
)

results = de.gene_level(tau=jnp.log(1.1))
print(f"Estimated null proportion: {de.null_proportion:.2%}")
print(de.summary(sort_by="lfsr", top_n=20))
```

### How it works

1. Runs the full empirical pipeline (Dirichlet sampling, CLR transform,
   paired differences) to produce `delta_samples`.
2. Computes `delta_mean` and `delta_sd` per gene from the samples.
3. Fits a scale mixture of normals prior via EM:
   `pi(beta) = sum_k w_k N(0, sigma_k^2)` where the grid `sigma_k` is fixed
   and the weights `w_k` are estimated.
4. Computes the shrinkage posterior per gene (a mixture of Gaussians) and
   derives shrunk lfsr values from the mixture CDF.

### When to use it

- When you suspect that most genes are not DE and want the analysis to
  account for this (adaptive shrinkage toward zero).
- When you want a data-driven estimate of the null proportion (`w_0`).
- The computational overhead is negligible relative to the upstream sampling.

### Compatibility

The shrunk lfsr values are fully compatible with the existing PEFP
error-control machinery.  All methods (`call_genes`, `compute_pefp`,
`find_threshold`, `summary`) work identically.

### Mathematical details

See Section 10 of the paper (`paper/_diffexp10.qmd`) for the full
derivation.

## Empirical Pathway Enrichment

The empirical DE path supports pathway-level analysis via ILR (Isometric
Log-Ratio) balances. Three complementary tests are available:

1. **Single-balance test** (`test_gene_set`): Tests whether a pathway shifts
   up or down as a whole, yielding a pathway-level lfsr.
2. **Perturbation test** (`test_pathway_perturbation`): Detects coordinated
   within-pathway rearrangement even when the average balance is near zero.
3. **Batch test** (`test_multiple_gene_sets`): Runs the single-balance test
   for multiple pathways with PEFP control.

### Quick start

```python
from scribe.de import compare
import jax.numpy as jnp

# Create empirical comparison
de = compare(
    results_bleo, results_ctrl,
    method="empirical",
    component_A=0, component_B=0,
)

# Single pathway balance test
pathway = jnp.array([10, 25, 42, 101, 200])
result = de.test_gene_set(pathway, tau=jnp.log(1.1))
print(f"Balance: {result['balance_mean']:.3f} ± {result['balance_sd']:.3f}")
print(f"lfsr: {result['lfsr']:.4f}")

# Multivariate within-pathway perturbation test
perturb = de.test_pathway_perturbation(pathway, n_permutations=999)
print(f"Perturbation T: {perturb['t_obs']:.4f}, p={perturb['p_value']:.4f}")

# Batch testing with PEFP control
gene_sets = [
    jnp.array([10, 25, 42, 101, 200]),
    jnp.array([5, 8, 15, 33]),
    jnp.array([60, 70, 80, 90, 100]),
]
batch = de.test_multiple_gene_sets(gene_sets, target_pefp=0.05)
for i, sig in enumerate(batch["significant"]):
    print(f"Pathway {i}: lfsr={batch['lfsr'][i]:.4f}, significant={sig}")
```

### Mathematical details

See Section 9 of the paper (`paper/_diffexp09.qmd`, "Empirical Pathway
Enrichment via ILR Balances") for the full derivation, including:

- ILR balance vector construction and normalization proofs
- Pathway-aware sequential binary partition (SBP) basis
- Multivariate within-pathway perturbation statistic
- Compatibility with PEFP error control

## Module Layout

```
de/
├── __init__.py          # Public API
├── results.py           # Class hierarchy + compare() factory
├── _extract.py          # Dimension-aware parameter extraction
├── _transforms.py       # ALR ↔ CLR ↔ ILR transformations
├── _gene_level.py       # Per-gene DE (analytic Gaussian)
├── _empirical.py        # Per-gene DE (empirical Monte Carlo)
├── _shrinkage.py        # Empirical Bayes shrinkage (scale mixture of normals)
├── _set_level.py        # Gene-set/pathway analysis
├── _error_control.py    # Bayesian error control, formatting
├── _gaussianity.py      # Gaussianity diagnostics
└── README.md            # This file
```

## Practical Significance

There are two approaches to incorporating practical significance thresholds:

1. **Two-threshold approach (default)**: Gene-level results include both `lfsr`
   (sign confidence) and `prob_effect` (probability that `|Delta| > tau`).
   `call_de_genes` requires both `lfsr < lfsr_threshold` AND
   `prob_effect > prob_effect_threshold`.

2. **Paper-aligned `lfsr_tau`**: The `lfsr_tau` field implements the paper's
   modified local false sign rate that directly incorporates `tau`:

   ```python
   # Use lfsr_tau for error control
   results = de.gene_level(tau=jnp.log(1.1))
   threshold = de.find_threshold(target_pefp=0.05, tau=jnp.log(1.1),
                                  use_lfsr_tau=True)
   pefp = de.compute_pefp(threshold=threshold, tau=jnp.log(1.1),
                           use_lfsr_tau=True)
   ```

Both approaches are valid; the two-threshold method is more conservative and
is the default.

## Gaussianity Diagnostics

The DE framework assumes that the marginal ALR distribution of each gene
is well-approximated by a Gaussian.  The `gaussianity_diagnostics`
function checks this assumption by computing per-feature (per-gene)
summary statistics in a single vectorized GPU pass:

```python
from scribe.de import gaussianity_diagnostics

# On any (N, D) sample matrix (e.g. ALR-transformed posterior samples)
diag = gaussianity_diagnostics(alr_samples)
# diag["skewness"]    : (D,)  — third standardised moment (Gaussian: 0)
# diag["kurtosis"]    : (D,)  — excess kurtosis (Gaussian: 0)
# diag["jarque_bera"] : (D,)  — JB test statistic
# diag["jb_pvalue"]   : (D,)  — asymptotic chi2(2) p-value
```

### Automatic computation inside `fit_logistic_normal`

When you call `results.fit_logistic_normal(...)`, the diagnostics are
computed automatically on the ALR samples **before** the SVD fit and
returned as `fitted["gaussianity"]`:

```python
fitted = results.fit_logistic_normal(rank=32)
gd = fitted["gaussianity"]      # dict with skewness, kurtosis, etc.
# For mixture models: gd["skewness"].shape == (K, D-1)
# For non-mixture:    gd["skewness"].shape == (D-1,)
```

### Interpretation and suggested thresholds

| Statistic | Gaussian value | Flag if |
|---|---|---|
| \|skewness\| | 0 | > 0.5 |
| \|excess kurtosis\| | 0 | > 1.0 |
| Jarque-Bera | 0 | continuous score (use for ranking) |

The skewness and kurtosis thresholds are **descriptive flags** — not
frequentist hypothesis tests.  Since the DE analysis is fully Bayesian,
multiple-testing corrections (BH, Bonferroni, etc.) are not appropriate
here.  The JB statistic combines skewness and kurtosis into a single
continuous score useful for ranking genes by departure from Gaussianity.

Genes that fail the skewness/kurtosis thresholds may have poorly
calibrated lfsr/PEFP values under the Gaussian assumption.  Consider
filtering them from the DE results or using a non-parametric alternative
for those genes.

## Performance Notes

All operations are memory-efficient:
- No materialization of G×G matrices
- Low-rank operations: O(kG) or O(k²G) where k ~ 50, G ~ 30,000
- Exact computation (no approximations beyond the Gaussian fit)

## References

- Aitchison, J. (1982). "The statistical analysis of compositional data." JRSS
  B.
- Aitchison, J. & Shen, S.M. (1980). "Logistic-normal distributions."
  Biometrika.

## Notes

- The lfdr function (`compute_lfdr`) is an empirical Bayes approximation. For
  production use, consider fitting a proper two-group mixture model. The lfsr
  (local false sign rate) is exact under the Gaussian assumption and is the
  recommended primary error measure.
- `find_lfsr_threshold` uses a cumulative-sum algorithm that runs in O(D log D)
  time, making it efficient for large gene sets.
- `build_balance_contrast` validates that numerator and denominator indices are
  disjoint, raising `ValueError` if they overlap.
- An epsilon guard (`1e-30` floor) is applied to `delta_sd` in both gene-level
  and contrast-level computations to prevent `NaN`/`Inf` from near-zero
  variance genes.

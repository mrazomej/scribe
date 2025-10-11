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

## Key Concepts

### Bayesian Error Control

Unlike frequentist methods that use FDR (False Discovery Rate), this module
uses:

- **lfsr (Local False Sign Rate)**: Posterior probability of having the wrong
  sign for each gene
- **PEFP (Posterior Expected False Discovery Proportion)**: Bayesian analogue of
  FDR

These are true posterior probabilities computed from the data, not frequentist
error rates.

### Compositional Data Analysis

Gene expression data is compositional (counts sum to a total). Standard analysis
methods can give misleading results. This module:

1. Works in CLR space where log-ratios are centered
2. Uses ALR→CLR transformations that preserve the correlation structure
3. Provides gene-set analysis via compositional balances

## Quick Start

```python
from scribe.core import de
import jax.numpy as jnp

# Fit logistic-normal models for two conditions
model_A = results_A.fit_logistic_normal(rank=16)
model_B = results_B.fit_logistic_normal(rank=16)

# Global divergence
kl = de.kl_divergence(model_A, model_B)
js = de.jensen_shannon(model_A, model_B)

# Gene-level DE
de_results = de.differential_expression(
    model_A, 
    model_B,
    tau=jnp.log(1.1),  # 10% fold-change threshold
    gene_names=gene_names
)

# Call DE genes (Bayesian decision)
is_de = de.call_de_genes(
    de_results,
    lfsr_threshold=0.05,       # Max false sign rate
    prob_effect_threshold=0.95  # Min probability of effect
)

print(f"Found {is_de.sum()} DE genes")

# Check Bayesian FDR
pefp = de.compute_pefp(de_results['lfsr'], threshold=0.05)
print(f"Expected false discovery proportion: {pefp:.3f}")

# Display top genes
table = de.format_de_table(de_results, sort_by='lfsr', top_n=20)
print(table)
```

## Gene-Set Analysis

Test pathways using compositional balances:

```python
# Test a pathway (indices of genes in the pathway)
pathway_indices = jnp.array([10, 25, 42, 101, ...])

result = de.test_gene_set(model_A, model_B, pathway_indices, tau=jnp.log(1.1))

print(f"Pathway effect: {result['delta_mean']:.3f} ± {result['delta_sd']:.3f}")
print(f"P(enriched): {result['prob_positive']:.3f}")
print(f"P(effect > tau): {result['prob_effect']:.3f}")
```

## Module Structure

```
de/
├── __init__.py          # Public API
├── transformations.py   # ALR ↔ CLR ↔ ILR transformations
├── gene_level.py        # Per-gene differential expression
├── set_level.py         # Gene-set/pathway analysis
└── utils.py            # Bayesian error control, formatting
```

## API Reference

### Global Divergences

- `kl_divergence(p, q)`: KL(p || q) between two distributions
- `jensen_shannon(p, q)`: Symmetric JS divergence
- `mahalanobis(p, q)`: Squared Mahalanobis distance with pooled covariance

All support both `LowRankLogisticNormal` and `SoftmaxNormal` distributions via
multipledispatch.

### Gene-Level Analysis

- `differential_expression(model_A, model_B, tau, ...)`: Compute posterior for
  each gene
  - Returns: delta_mean, delta_sd, prob_positive, prob_effect, lfsr
- `call_de_genes(de_results, lfsr_threshold, prob_effect_threshold)`: Bayesian
  decision rule

### Set-Level Analysis

- `test_contrast(model_A, model_B, contrast, tau)`: Test a linear contrast c^T Δ
- `test_gene_set(model_A, model_B, gene_set_indices, tau)`: Test gene set
  enrichment
- `build_balance_contrast(num_indices, den_indices, D)`: Build compositional
  balance

### Bayesian Error Control

- `compute_lfdr(delta_mean, delta_sd, prior_null_prob)`: Local false discovery
  rate
- `compute_pefp(lfsr, threshold)`: Posterior expected false discovery proportion
- `find_lfsr_threshold(lfsr, target_pefp)`: Find threshold to control PEFP

### Utilities

- `format_de_table(de_results, sort_by, top_n)`: Format results as table

## Mathematical Details

### Coordinate Systems

- **ALR (Additive Log-Ratio)**: z_i = log(ρ_i) - log(ρ_G) for i=1,...,G-1
  - Asymmetric (depends on reference gene G)
  - Used internally for parameter fitting
  - Dimension: G-1

- **CLR (Centered Log-Ratio)**: z_i = log(ρ_i) - (1/G)Σ_j log(ρ_j)
  - Symmetric (no reference gene)
  - Used for reporting gene-level effects
  - Dimension: G (constrained, sums to 0)

- **ILR (Isometric Log-Ratio)**: Orthonormal basis of CLR subspace
  - Dimension: G-1
  - Useful for gene-set analysis

### Exact Transformations

The ALR→CLR transformation preserves the low-rank structure:
- Mean: μ_clr = H·μ_alr (where H embeds and centers)
- Low-rank factor: W_clr = H·W_alr
- Diagonal: d_clr computed exactly via centering formula

### Posterior Distribution

Under the fitted logistic-normal models:
- z_A ~ N(μ_A, Σ_A) in CLR space
- z_B ~ N(μ_B, Σ_B) in CLR space

The difference is also Gaussian:
- Δ = z_A - z_B ~ N(μ_A - μ_B, Σ_A + Σ_B)

This gives exact analytic posteriors for:
- Each gene: Δ_g ~ N(μ_A[g] - μ_B[g], σ²_A[g] + σ²_B[g])
- Any contrast: c^T Δ ~ N(c^T(μ_A - μ_B), c^T(Σ_A + Σ_B)c)

## Performance

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

The lfdr function (`compute_lfdr`) is an empirical Bayes approximation. For
production use, consider fitting a proper two-group mixture model. The lfsr
(local false sign rate) is exact under the Gaussian assumption and is the
recommended primary error measure.


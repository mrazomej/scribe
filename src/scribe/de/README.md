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

### `ScribeDEResults` (via `compare()`)

The recommended entry point. Creates a structured results object from two fitted
models:

```python
de = compare(model_A, model_B, gene_names=names)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `de.gene_level(tau)` | Per-gene posterior summaries |
| `de.call_genes(lfsr_threshold, prob_effect_threshold)` | Bayesian gene calling |
| `de.test_gene_set(indices, tau)` | Pathway enrichment via balances |
| `de.test_contrast(contrast, tau)` | Custom linear contrast |
| `de.compute_pefp(threshold)` | Posterior expected FDP |
| `de.find_threshold(target_pefp)` | Find lfsr threshold for PEFP control |
| `de.summary(sort_by, top_n)` | Formatted results table |

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
    differential_expression,
    call_de_genes,
    test_gene_set,
    test_contrast,
    build_balance_contrast,
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
- **PEFP (Posterior Expected False Discovery Proportion)**: Bayesian analogue of
  FDR. Computed as the average lfsr of called genes.

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

## Module Layout

```
de/
├── __init__.py          # Public API
├── results.py           # ScribeDEResults dataclass + compare()
├── _extract.py          # Dimension-aware parameter extraction
├── _transforms.py       # ALR ↔ CLR ↔ ILR transformations
├── _gene_level.py       # Per-gene differential expression
├── _set_level.py        # Gene-set/pathway analysis
├── _error_control.py    # Bayesian error control, formatting
└── README.md            # This file
```

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

The lfdr function (`compute_lfdr`) is an empirical Bayes approximation. For
production use, consider fitting a proper two-group mixture model. The lfsr
(local false sign rate) is exact under the Gaussian assumption and is the
recommended primary error measure.

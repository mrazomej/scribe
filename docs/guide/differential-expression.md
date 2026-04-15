# Differential Expression

SCRIBE provides a fully Bayesian framework for differential expression (DE)
analysis that works in compositional space, accounts for gene-gene
correlations, and provides true posterior probabilities instead of frequentist
p-values.

## Overview

The DE framework is:

- **Compositional** -- works in CLR (Centered Log-Ratio) or ILR (Isometric
  Log-Ratio) space, ensuring reference-invariant results
- **Correlation-aware** -- leverages low-rank covariance structure to account
  for gene-gene correlations
- **Fully Bayesian** -- provides exact posterior probabilities under the
  Gaussian assumption (parametric) or via Monte Carlo counting (empirical)
- **Computationally efficient** -- all operations are \(O(kG)\) or
  \(O(k^2 G)\) where \(k \ll G\), avoiding \(O(G^3)\) dense matrix
  operations

---

## Quick Start

```python
from scribe.de import compare

# Fit models for two conditions
results_A = scribe.fit(adata_treatment, model="nbdm")
results_B = scribe.fit(adata_control, model="nbdm")

# Create comparison (empirical method, recommended)
de = compare(
    results_A, results_B,
    method="empirical",
    component_A=0, component_B=0,
)

# Gene-level analysis with practical significance threshold
results = de.gene_level(tau=jnp.log(1.1))

# Call DE genes (Bayesian decision)
is_de = de.call_genes(lfsr_threshold=0.05, prob_effect_threshold=0.95)
print(f"Found {is_de.sum()} DE genes")

# Summary table
print(de.summary(sort_by="lfsr", top_n=20))
```

---

## Three DE Methods

The `compare()` factory returns different result types depending on `method=`:

| Method | When to use | How it works |
|--------|-------------|--------------|
| **Parametric** | Gaussian assumption holds | Analytic posteriors from fitted logistic-normal models |
| **Empirical** | General use (recommended) | Monte Carlo counting on Dirichlet-sampled compositions |
| **Shrinkage** | Most genes are not DE | Empirical Bayes shrinkage on top of empirical results |

### Parametric

Requires pre-fitted logistic-normal models. All statistics are computed
analytically:

```python
# Fit logistic-normal models first
model_A = results_A.fit_logistic_normal(rank=16)
model_B = results_B.fit_logistic_normal(rank=16)

de = compare(model_A, model_B, gene_names=adata.var_names.tolist())
```

### Empirical (recommended)

Works directly with posterior samples via Monte Carlo counting -- no
distributional assumptions required:

```python
de = compare(
    results_A, results_B,
    method="empirical",
    component_A=0, component_B=0,
)
```

When results objects are passed, `compare()` automatically:

- Extracts `r` samples from `posterior_samples["r"]`
- Detects hierarchical models (gene-specific \(p\)) and extracts \(p\) samples
- Infers gene names from `results.var.index`

### Shrinkage

Improves on empirical DE by learning a genome-wide effect-size distribution
and adaptively shrinking noisy estimates toward zero:

```python
de = compare(
    results_A, results_B,
    method="shrinkage",
    component_A=0, component_B=0,
)

results = de.gene_level(tau=jnp.log(1.1))
print(f"Estimated null proportion: {de.null_proportion:.2%}")
```

!!! tip "Zero-copy upgrade"
    You can upgrade an existing empirical result to shrinkage without
    recomputing the expensive Dirichlet sampling:

    ```python
    de_emp = compare(results_A, results_B, method="empirical", ...)
    de_shrink = de_emp.shrink()
    ```

---

## Gene-Level Analysis

All DE result types share the same analysis API:

| Method | Description |
|--------|-------------|
| `de.gene_level(tau)` | Per-gene posterior summaries (lfsr and lfsr_tau) |
| `de.call_genes(tau, lfsr_threshold, prob_effect_threshold)` | Bayesian gene calling |
| `de.summary(tau, sort_by, top_n)` | Formatted results table |
| `de.to_dataframe(tau, target_pefp)` | Export to pandas DataFrame |

### Exporting results

```python
# Basic export
df = de.to_dataframe(tau=0.5)

# With automatic PEFP-controlled DE calls
df = de.to_dataframe(tau=0.5, target_pefp=0.05)
de_genes = df[df["clr_is_de"]]

# Include biological metrics (empirical/shrinkage only)
df = de.to_dataframe(metrics="all", tau=0.5)
```

---

## Bayesian Error Control

Unlike frequentist methods that use FDR, SCRIBE uses true posterior
probabilities:

| Metric | Description |
|--------|-------------|
| **lfsr** | Local false sign rate -- posterior probability of having the wrong sign |
| **lfsr_tau** | Modified lfsr incorporating practical significance threshold \(\tau\) |
| **PEFP** | Posterior expected false discovery proportion -- Bayesian analogue of FDR |

### Controlling the false discovery rate

```python
import jax.numpy as jnp

# Gene-level analysis with practical significance
results = de.gene_level(tau=jnp.log(1.1))

# Find the lfsr threshold that controls PEFP at 5%
threshold = de.find_threshold(target_pefp=0.05)
print(f"lfsr threshold for 5% PEFP: {threshold:.3f}")

# Call genes at that threshold
is_de = de.call_genes(lfsr_threshold=threshold)

# Verify the Bayesian FDR
pefp = de.compute_pefp(threshold=0.05)
print(f"Expected false discovery proportion: {pefp:.3f}")
```

---

## Biological-Level DE

While CLR-based metrics operate in the compositional simplex, biological-level
DE computes metrics directly on the denoised Negative Binomial distribution.
This is especially valuable for lowly expressed genes where compositional
artifacts dominate.

| Metric | What it captures |
|--------|------------------|
| **Biological LFC** | Mean expression shift: \(\log(\mu_A / \mu_B)\) |
| **Log-variance ratio** | Dispersion shift: \(\log(\text{var}_A / \text{var}_B)\) |
| **Gamma Jeffreys divergence** | Full distributional shift via symmetrized KL |

```python
# Biological-level DE (empirical/shrinkage only)
bio = de.biological_level(
    tau_lfc=jnp.log(1.5),
    tau_var=jnp.log(2.0),
    tau_kl=0.5,
)

bio["lfc_mean"]     # posterior mean biological LFC per gene
bio["lfc_lfsr"]     # local false sign rate for LFC
bio["lvr_mean"]     # posterior mean log-variance ratio
bio["kl_mean"]      # posterior mean Jeffreys divergence
```

!!! info "Recommended workflow"
    1. **Screen with CLR**: use CLR-based lfsr as the primary filter
    2. **Validate with biological LFC**: filter out compositional artifacts
    3. **Detect variance changes**: inspect log-variance ratio for genes with
       small LFC but high distributional shift
    4. **Flag distributional shifts**: use Jeffreys divergence as a catch-all

---

## Mixture-Weighted DE

When a cell type is modeled as a multi-component mixture (e.g., to capture
distinct cellular states), you can perform **population-level** DE that
marginalises over the mixture instead of comparing individual components.

The pipeline samples compositions from all K components and averages them on the
simplex using the posterior mixture weights, then feeds the result into the
standard CLR machinery.

```python
# Auto-extract mixing_weights from results objects
de = compare(
    results_A, results_B,
    method="empirical",
    mixture_weighted=True,
)

results = de.gene_level(tau=jnp.log(1.1))
```

For raw arrays, provide weights explicitly:

```python
de = compare(
    r_samples_A,  # (N, K, D)
    r_samples_B,  # (N, K, D)
    method="empirical",
    mixture_weighted=True,
    mixture_weights_A=weights_A,  # (N, K)
    mixture_weights_B=weights_B,  # (N, K)
)
```

| Scenario | Approach |
|----------|----------|
| Compare biologically distinct states | `component_A=`, `component_B=` |
| Population-level change of a multi-component cell type | `mixture_weighted=True` |
| Single-component model | Standard `compare()` (mixture weighting is a no-op) |

!!! note
    `mixture_weighted=True` is mutually exclusive with `component_A` /
    `component_B`. The parametric method is not supported because the CLR
    of a mixture of Dirichlets is not Gaussian.

Shrinkage works on top of the mixture-weighted empirical result:

```python
de_shrink = compare(
    results_A, results_B,
    method="shrinkage",
    mixture_weighted=True,
)
```

Biological-level metrics (LFC, LVR, Jeffreys divergence) are computed
from mixture-weighted NB parameters when `compute_biological=True`.

---

## Gene Expression Filter

Low-expression genes can appear spuriously DE due to compositional artifacts.
The `gene_mask` parameter aggregates filtered genes into a single "other"
pseudo-gene before Dirichlet sampling:

```python
from scribe.de import compare, compute_expression_mask

# Build a mask from MAP mean expression
mask = compute_expression_mask(
    results_A, results_B,
    component_A=0, component_B=0,
    min_mean_expression=1.0,
)

# Pass to compare()
de = compare(
    results_A, results_B,
    method="empirical",
    component_A=0, component_B=0,
    gene_mask=mask,
)
```

### Interactive mask exploration

After the initial comparison, you can change the expression mask without
re-running the expensive Dirichlet sampling:

```python
# Explore a different threshold
de.set_expression_threshold(min_expression=3.0)
df2 = de.to_dataframe(tau=0.5)

# Apply a custom mask
de.set_gene_mask(my_custom_mask)

# Restore all genes
de.clear_mask()
```

---

## Pathway Analysis

The empirical DE path supports pathway-level analysis via ILR balances:

### Single pathway test

```python
import jax.numpy as jnp

# Test whether a pathway shifts as a whole
pathway = jnp.array([10, 25, 42, 101, 200])
result = de.test_gene_set(pathway, tau=jnp.log(1.1))
print(f"Balance: {result['balance_mean']:.3f} +/- {result['balance_sd']:.3f}")
print(f"lfsr: {result['lfsr']:.4f}")
```

### Within-pathway perturbation test

Detects coordinated rearrangement within a pathway even when the average
balance is near zero:

```python
perturb = de.test_pathway_perturbation(pathway, n_permutations=999)
print(f"Perturbation T: {perturb['t_obs']:.4f}, p={perturb['p_value']:.4f}")
```

### Batch testing with PEFP control

```python
gene_sets = [
    jnp.array([10, 25, 42, 101, 200]),
    jnp.array([5, 8, 15, 33]),
    jnp.array([60, 70, 80, 90, 100]),
]
batch = de.test_multiple_gene_sets(gene_sets, target_pefp=0.05)
```

---

## Multi-Dataset Comparisons

For multi-dataset models, `compare_datasets()` is a convenience wrapper that
slices per-dataset views and preserves within-posterior correlation:

```python
from scribe.de import compare_datasets

de = compare_datasets(results, dataset_A=0, dataset_B=1)
de = compare_datasets(results, 0, 1, component=0, method="shrinkage")
```

### Label-based component matching

When mixture models are fit with `annotation_key`, you can look up components
by label (e.g., cell type name) instead of tracking indices manually:

```python
from scribe.de import compare, match_components_by_label, get_shared_labels

# Find component indices for "Fibroblast" in both results
idx_A, idx_B = match_components_by_label(results_A, results_B, "Fibroblast")

de = compare(
    results_A, results_B,
    method="empirical",
    component_A=idx_A, component_B=idx_B,
)

# Discover labels shared across results
labels = get_shared_labels(results_A, results_B)
```

---

## Class Hierarchy

```
ScribeDEResults (base)
├── ScribeParametricDEResults   -- analytic Gaussian
├── ScribeEmpiricalDEResults    -- Monte Carlo counting
└── ScribeShrinkageDEResults    -- empirical Bayes shrinkage (extends Empirical)
```

The `compare()` factory returns the appropriate subclass based on `method=`.
All shared methods (`call_genes`, `compute_pefp`, `find_threshold`, `summary`)
work identically on all three.

For the full API, see the [API Reference](../reference/scribe/de/).

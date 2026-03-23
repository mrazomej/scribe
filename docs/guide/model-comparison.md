# Model Comparison

SCRIBE provides scalable, principled Bayesian model comparison that works with
any fitted model -- no refitting required. It implements **WAIC** and
**PSIS-LOO** on top of the existing posterior infrastructure, plus stacking
weights and per-gene goodness-of-fit diagnostics.

!!! info "Model comparison vs. differential expression"
    These answer different questions:

    | Module | Question |
    |--------|----------|
    | [Differential Expression](differential-expression.md) | Do genes change **between conditions** (same model)? |
    | Model Comparison | Does one **model** fit the data better than another? |

---

## Quick Start

```python
from scribe.mc import compare_models

# Fit two models on the same data
results_nbdm = scribe.fit(adata, model="nbdm")
results_zinb = scribe.fit(adata, model="zinb")

# Compare
mc = compare_models(
    [results_nbdm, results_zinb],
    counts=adata.X,
    model_names=["NBDM", "ZINB"],
    gene_names=adata.var_names.tolist(),
    compute_gene_liks=True,
)

# Ranked summary table
print(mc.summary())

# PSIS-LOO diagnostics
print(mc.diagnostics())

# Ranked DataFrame (ready for plotting)
df = mc.rank()

# Gene-level comparison
gene_df = mc.gene_level_comparison("NBDM", "ZINB")
print(gene_df.head(20))

# Stacking weights (optimal ensemble)
w = mc.stacking_weights()
print(w)
```

---

## WAIC vs. PSIS-LOO

Both methods estimate the **expected log predictive density (elpd)** -- the
fundamental quantity for comparing predictive performance:

| Criterion | WAIC | PSIS-LOO |
|-----------|------|----------|
| **Speed** | Very fast (JIT-compiled) | Moderate (SciPy) |
| **Reliability** | Good for well-behaved posteriors | More robust, recommended |
| **Diagnostic** | None | Per-observation \(\hat{k}\) |

!!! tip "Default recommendation"
    Use **PSIS-LOO** by default. Use **WAIC** for quick exploratory
    comparisons or when data is very large and PSIS-LOO is too slow.

---

## Diagnostics

The per-observation \(\hat{k}\) from PSIS-LOO indicates the reliability of
the importance sampling approximation:

| \(\hat{k}\) range | Interpretation |
|--------------------|----------------|
| < 0.5 | Excellent -- reliable estimate |
| 0.5 -- 0.7 | OK -- estimate is usable |
| >= 0.7 | Problematic -- IS approximation is unreliable |

In single-cell data, high \(\hat{k}\) values typically correspond to cells
with unusually high or low total UMI counts, or rare cell types that are
highly influential for the posterior.

---

## Gene-Level Comparison

Beyond the global model ranking, you can compare models at the per-gene level
to understand *where* one model outperforms another:

```python
gene_df = mc.gene_level_comparison("NBDM", "ZINB")
```

This returns a DataFrame with per-gene elpd differences, letting you identify
specific genes that drive the overall model preference.

---

## Stacking Weights

Model stacking finds the optimal ensemble weights that maximize
cross-validated predictive performance:

```python
w = mc.stacking_weights()
# e.g., {"NBDM": 0.35, "ZINB": 0.65}
```

This is useful when no single model dominates and you want to combine
predictions.

---

## Goodness-of-Fit Diagnostics

While WAIC and PSIS-LOO compare models *relative* to each other, SCRIBE also
provides absolute goodness-of-fit diagnostics to assess whether a single model
adequately describes each gene.

### Randomized Quantile Residuals (RQR)

Under a correctly specified model, the residuals are standard normal for every
gene regardless of expression level:

```python
from scribe.mc import compute_gof_mask

# Build a boolean mask: True for well-fit genes
mask = compute_gof_mask(counts, results, max_variance=1.5)
```

### PPC-Based Scoring

When RQR diagnostics are ambiguous, PPC-based scoring provides a
higher-resolution alternative by generating full posterior predictive count
samples and comparing against observed histograms:

```python
from scribe.mc import compute_ppc_gof_mask

mask, scores = compute_ppc_gof_mask(
    counts, results,
    n_ppc_samples=500,
    gene_batch_size=50,
    max_calibration_failure=0.5,
    return_scores=True,
)

# Per-gene scores
print(scores["calibration_failure"][:10])
print(scores["l1_distance"][:10])
```

| Scenario | Recommended |
|----------|-------------|
| Quick initial screen | RQR (`compute_gof_mask`) |
| Ambiguous RQR results or systematic MAP bias | PPC (`compute_ppc_gof_mask`) |
| Very large datasets where PPC is too slow | RQR |
| Comparing histogram-level shape fit | PPC |

### Integration with DE

Goodness-of-fit masks integrate directly with the DE pipeline -- pass
well-fit genes as `gene_mask` to focus DE analysis on genes that the model
describes adequately:

```python
from scribe.de import compare

de = compare(
    results_A, results_B,
    method="empirical",
    gene_mask=mask,
    component_A=0, component_B=0,
)
```

---

For the full API, see the [API Reference](../reference/scribe/mc/).

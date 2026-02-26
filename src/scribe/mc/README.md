# `scribe.mc` — Bayesian Model Comparison

This module provides scalable, principled Bayesian model comparison for SCRIBE
models.  It implements **WAIC** and **PSIS-LOO** on top of the existing
posterior inference infrastructure, with no need to refit any model.

---

## Mathematical background

See `paper/_model_comparison.qmd` for full derivations.  The short version:

**Expected log predictive density (elpd)** — the fundamental quantity:

```
elpd = E_{p_t(ỹ)}[log p(ỹ | y)]
```

**WAIC** — analytical LOO-CV approximation:

```
lppd    = Σᵢ log (1/S Σₛ exp(ℓᵢˢ))
p_waic₂ = Σᵢ var_s(ℓᵢˢ)
WAIC    = -2 (lppd - p_waic₂)
```

**PSIS-LOO** — Pareto-smoothed IS, with per-observation reliability diagnostic
k̂:

```
log w_s = -ℓᵢˢ      (raw IS weights)
→ Pareto-smooth top-M weights per observation
→ elpd_loo = Σᵢ log IS-weighted average
k̂ < 0.5 = excellent, 0.5–0.7 = OK, ≥ 0.7 = problematic
```

---

## Module structure

```
mc/
├── __init__.py            Public API
├── README.md              This file
├── results.py             ScribeModelComparisonResults + compare_models()
├── _waic.py               JAX-accelerated WAIC
├── _psis_loo.py           PSIS-LOO with Pareto fitting (NumPy/SciPy)
├── _gene_level.py         Per-gene elpd differences
├── _stacking.py           Model stacking weight optimization
└── _goodness_of_fit.py    Randomized quantile residuals & per-gene GoF
```

---

## Quick start

```python
from scribe.mc import compare_models

# Fit two models on the same data
results_nbdm        = ...   # ScribeSVIResults (NBDM, shared p)
results_hierarchical = ...  # ScribeSVIResults (hierarchical, gene-specific p)

# Compare
mc = compare_models(
    [results_nbdm, results_hierarchical],
    counts=counts,
    model_names=["NBDM", "Hierarchical"],
    gene_names=gene_names,
    compute_gene_liks=True,   # enable gene-level comparison
)

# Ranked summary table
print(mc.summary())

# PSIS-LOO k̂ diagnostics
print(mc.diagnostics())

# Ranked DataFrame (ready for plotting)
df = mc.rank()

# Gene-level comparison
gene_df = mc.gene_level_comparison("NBDM", "Hierarchical")
print(gene_df.head(20))

# Stacking weights (optimal ensemble)
w = mc.stacking_weights()
print(w)
```

---

## Choosing between WAIC and PSIS-LOO

| Criterion  | Speed    | Reliability     | Diagnostic |
|------------|----------|-----------------|------------|
| WAIC2      | Very fast (JIT) | Good for well-behaved posteriors | None |
| PSIS-LOO   | Moderate (SciPy) | More robust, recommended | k̂ per obs |

Use **PSIS-LOO** by default.  Use **WAIC** for quick exploratory comparisons
or when data is very large and PSIS-LOO is slow.

---

## Diagnostics

The per-observation k̂ from PSIS-LOO is printed by `mc.diagnostics()`.
Observations with k̂ ≥ 0.7 are "problematic" — the IS approximation is
unreliable for them and the LOO estimate may be noisy.

In single-cell data, high k̂ values typically correspond to:
- Cells with unusually high or low total UMI counts.
- Cells that are highly influential for the posterior (e.g., rare cell types).

---

## Per-gene goodness-of-fit (RQR)

The `_goodness_of_fit.py` module implements **randomized quantile residuals**
(Dunn & Smyth, 1996) for assessing absolute model fit at the gene level.
Unlike WAIC/PSIS-LOO (which compare models *relative* to each other), RQR
checks whether a *single* model adequately describes each gene's distribution.

Under a correctly specified model, the residuals are standard normal for every
gene regardless of expression level — providing an expression-scale-invariant
diagnostic.

```python
from scribe.mc import compute_gof_mask

# Build a boolean mask: True for well-fit genes
mask = compute_gof_mask(counts, results, max_variance=1.5)

# Use in DE pipeline
from scribe.de import compare
de = compare(results_A, results_B, gene_mask=mask, method="empirical", ...)
```

See `paper/_goodness_of_fit.qmd` for the full mathematical derivation.

---

## Per-gene goodness-of-fit (PPC-based)

When RQR diagnostics are ambiguous — e.g., systematic bias in shared parameters
inflates KS distances uniformly — the **PPC-based scoring** provides a
higher-resolution alternative.  It generates full posterior predictive count
samples, builds pointwise credible bands around the predicted histogram, and
measures how often (and how severely) the observed histogram falls outside.

Two metrics are produced per gene:

- **Calibration failure rate**: fraction of non-empty histogram bins outside
  the credible band.
- **L1 density distance**: sum of absolute differences between observed and
  PPC median densities.

```python
from scribe.mc import compute_ppc_gof_mask

# Build a PPC-based mask (slower, but more discriminative)
mask, scores = compute_ppc_gof_mask(
    counts, results,
    n_ppc_samples=500,
    gene_batch_size=50,
    max_calibration_failure=0.5,
    return_scores=True,
)

# Inspect per-gene scores
print(scores["calibration_failure"][:10])
print(scores["l1_distance"][:10])
```

**When to use PPC vs RQR:**

| Scenario | Recommended |
|----------|-------------|
| Quick initial screen | RQR (`compute_gof_mask`) |
| Ambiguous RQR results / systematic MAP bias | PPC (`compute_ppc_gof_mask`) |
| Very large datasets where PPC is too slow | RQR |
| Comparing histogram-level shape fit | PPC |

See `paper/_goodness_of_fit.qmd` §PPC-based goodness-of-fit diagnostics for
the full mathematical derivation.

---

## Relationship to the `de/` module

The `mc/` module is designed to answer a different question from `de/`:

| Module | Question |
|--------|----------|
| `de/`  | Do genes change **between conditions** (same model)? |
| `mc/`  | Does one **model** fit the data better than another? |

Both modules use the same `ScribeSVIResults`/`ScribeMCMCResults` infrastructure
and follow the same design pattern (results class + factory function + private
sub-modules).

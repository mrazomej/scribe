# Bayesian Model Comparison

SCRIBE offers several model families---NBDM, hierarchical, ZINB, VCP,
mixtures---and a key practical question is: **which model should I use for
my data?** Rather than relying on heuristics or fixed recipes, SCRIBE
provides a principled Bayesian framework that ranks models by their
**out-of-sample predictive accuracy**, quantifies the uncertainty in that
ranking, and can even construct optimal model ensembles.

This page develops the theory behind WAIC, PSIS-LOO, pairwise comparison,
and model stacking. For API usage and code examples, see the
[Model Comparison guide](../guide/model-comparison.md).

---

## The central quantity: expected log predictive density

All model comparison in SCRIBE revolves around the **expected log
predictive density** (elpd), which measures how well a fitted model
predicts new, unseen data drawn from the true data-generating process
\(p_t\):

\[
\text{elpd} = \int \log p(\tilde{y} \mid y)\, p_t(\tilde{y})\, d\tilde{y},
\]

where the **posterior predictive distribution** averages the likelihood
over the posterior:

\[
p(\tilde{y} \mid y) = \int p(\tilde{y} \mid \theta)\,
p(\theta \mid y)\, d\theta.
\]

Since \(p_t\) is unknown, we estimate elpd from the observed data.

### Leave-one-out cross-validation

The gold standard is **leave-one-out cross-validation** (LOO-CV):

\[
\text{elpd}_{\text{loo}} = \sum_{i=1}^{n} \log p(y_i \mid y_{-i}),
\]

where \(y_{-i}\) is the dataset with observation \(i\) removed. Each term
is the log predictive density for the held-out observation:

\[
p(y_i \mid y_{-i}) = \int p(y_i \mid \theta)\,
p(\theta \mid y_{-i})\, d\theta.
\]

Exact LOO-CV requires \(n\) separate model fits---prohibitive for
single-cell datasets with thousands of cells. SCRIBE therefore uses two
efficient approximations that compute elpd from a **single** posterior
inference.

---

## WAIC: analytical LOO approximation

The Widely Applicable Information Criterion (WAIC; Watanabe, 2010) is a
fast, fully analytical approximation to LOO-CV that requires only the
posterior samples already available after fitting.

### Log pointwise predictive density

The **log pointwise predictive density** (lppd) sums the predictive
probability of each observed data point, averaged over the posterior:

\[
\text{lppd} = \sum_{i=1}^{n} \log\!\left(
\frac{1}{S}\sum_{s=1}^{S} p(y_i \mid \theta^{(s)})
\right),
\]

where \(\{\theta^{(s)}\}_{s=1}^{S}\) are the posterior samples. In
practice, the inner sum is evaluated with log-sum-exp for numerical
stability.

### Effective number of parameters

WAIC penalizes model complexity through an **overfitting correction**
\(p_{\text{waic}}\) that estimates the effective number of parameters.
The preferred version (Gelman et al., 2014) uses the variance of the
log-likelihood across posterior draws:

\[
p_{\text{waic}} = \sum_{i=1}^{n}
\widehat{\text{Var}}_s\!\bigl(\log p(y_i \mid \theta^{(s)})\bigr).
\]

Each term measures how much the posterior disagrees about observation
\(i\): high variance means the model is "spending" parameters on that
data point.

### The WAIC criterion

\[
\text{elppd}_{\text{waic}} = \text{lppd} - p_{\text{waic}},
\qquad
\text{WAIC} = -2\,\text{elppd}_{\text{waic}}.
\]

**Lower WAIC is better.** The \(-2\) scaling places WAIC on the same
deviance scale as AIC for easy comparison.

!!! note "When WAIC can mislead"
    WAIC is asymptotically equivalent to LOO-CV under mild regularity
    conditions. However, it can be unreliable when the posterior has heavy
    tails or is dominated by a few influential observations---precisely
    the situation flagged by PSIS-LOO's built-in diagnostic.

---

## PSIS-LOO: the recommended criterion

**Pareto-Smoothed Importance Sampling LOO** (PSIS-LOO; Vehtari et al.,
2017) is more robust than WAIC and comes with a per-observation
reliability diagnostic.

### Importance sampling for LOO

The key insight is that the LOO posterior \(p(\theta \mid y_{-i})\) can
be obtained from the full-data posterior via importance weighting:

\[
p(\theta \mid y_{-i}) \propto
\frac{p(\theta \mid y)}{p(y_i \mid \theta)}.
\]

This gives raw importance weights \(w_i^{(s)} = 1 / p(y_i \mid
\theta^{(s)})\) that reweight the full posterior to approximate the LOO
posterior---without refitting.

### The heavy-tail problem

Some observations are so influential that the raw weights are extremely
variable, making the IS estimator unreliable. PSIS stabilizes this by
fitting a **generalized Pareto distribution** (GPD) to the upper tail of
the weight distribution and replacing the noisy empirical tail with
smooth GPD quantiles.

### The Pareto k diagnostic

The GPD shape parameter \(\hat{k}\) directly measures how heavy the tail
is for each observation:

| \(\hat{k}\) range | Interpretation |
|--------------------|----------------|
| < 0.5 | Excellent --- reliable estimate |
| 0.5 -- 0.7 | Acceptable --- usable but worth monitoring |
| >= 0.7 | Problematic --- IS approximation unreliable |

In single-cell data, high \(\hat{k}\) values typically correspond to
cells with unusually high or low total UMI counts, or rare cell types
that are highly influential for the posterior.

### PSIS-LOO elpd

After Pareto smoothing the weights, the LOO elpd is:

\[
\text{elpd}_{\text{psis-loo}} = \sum_{i=1}^{n}
\log \hat{p}_{\text{psis}}(y_i \mid y_{-i}),
\]

with associated criterion \(\text{LOO-IC} = -2\,\text{elpd}_{\text{psis-loo}}\).

!!! tip "Default recommendation"
    Use **PSIS-LOO** by default. It achieves lower bias than WAIC for
    realistic posterior geometries and provides per-observation
    diagnostics. Use WAIC only for quick exploratory comparisons or when
    data is very large and PSIS-LOO is too slow.

---

## Pairwise comparison with uncertainty

Given two models \(M_A\) and \(M_B\) evaluated on the same data, define
the **pointwise elpd difference**:

\[
d_i = l_{i,A} - l_{i,B},
\]

where \(l_{i,\cdot}\) is the pointwise LOO log predictive density. The
total difference and its standard error are:

\[
\Delta\text{elpd}_{AB} = \sum_{i=1}^{n} d_i,
\qquad
\widehat{\text{SE}} = \sqrt{\sum_{i=1}^{n}(d_i - \bar{d})^2}.
\]

The ratio \(z_{AB} = \Delta\text{elpd}_{AB}\, /\, \widehat{\text{SE}}\)
gives a scale-free signal-to-noise measure:

| \(|z_{AB}|\) | Interpretation |
|---------------|----------------|
| Much greater than 1 | Strong evidence for one model |
| Close to 1 | Difference is not practically meaningful |

!!! warning "Not a p-value"
    The z-score is a descriptive signal-to-noise ratio, not a
    frequentist test statistic. It does not have a calibrated Type I
    error interpretation.

---

## Gene-level comparison

Beyond the cell-level ranking, SCRIBE computes **per-gene elpd
differences** by evaluating the gene-level log-likelihood:

\[
\ell_g^{(s)} = \sum_{c=1}^{C} \log p(u_{gc} \mid \theta^{(s)}).
\]

Applying WAIC or PSIS-LOO to these gene-level log-likelihoods reveals
which genes benefit most from one model's added flexibility. For example,
when comparing NBDM to the hierarchical model, genes with highly variable
total counts across cells typically show the largest improvement.

---

## Model stacking

Instead of choosing one winner, **model stacking** (Yao et al., 2018)
constructs an optimal predictive ensemble. The stacking weights
\(\underline{w}^* \in \Delta^{K-1}\) maximize the LOO log-score of the
mixture:

\[
\underline{w}^* = \underset{\underline{w}\,\in\,\Delta^{K-1}}{\arg\max}
\sum_{i=1}^{n}\log\sum_{k=1}^{K} w_k\,\hat{p}(y_i \mid y_{-i}, M_k).
\]

The objective is **concave** (log of a linear function), so the unique
optimum is found efficiently by standard convex solvers. Stacking weights
differ from Bayesian model averaging: they optimize predictive
performance on held-out data, making them more robust when models are
misspecified or structurally similar.

For comparison, SCRIBE also provides **pseudo-BMA weights**:

\[
w_k^{\text{BMA}} \propto \exp\!\bigl(-\tfrac{1}{2}\,\text{WAIC}_k\bigr),
\]

which mimic the classical AIC weight formula.

---

## Application to SCRIBE's model families

### Pointwise log-likelihood

For the standard NBDM model with shared success probability \(\hat{p}\),
the per-cell log-likelihood under posterior sample \(s\) factorizes into
a total-count term and a composition term:

\[
\ell_c^{(s)} = \log \text{NB}(u_{T,c} \mid r_T^{(s)}, \hat{p}^{(s)})
+ \log \text{DM}(\underline{u}_c \mid u_{T,c}, \underline{r}^{(s)}),
\]

where \(u_{T,c} = \sum_g u_{gc}\) is the total UMI count and
\(r_T^{(s)} = \sum_g r_g^{(s)}\).

For the hierarchical model with gene-specific \(p_g\), the NB-DM
factorization breaks down (see the
[Hierarchical *p* theory page](hierarchical-p.md)), and the per-cell
log-likelihood becomes a sum over genes:

\[
\ell_c^{(s)} = \sum_{g=1}^{G}
\log \text{NB}(u_{gc} \mid r_g^{(s)}, p_g^{(s)}).
\]

Both are computed automatically by SCRIBE's `log_likelihood` method and
stored as an \(S \times C\) matrix for downstream comparison.

### What PSIS-LOO detects

Comparing NBDM to the hierarchical model with PSIS-LOO answers: **does
allowing gene-specific \(p_g\) improve out-of-sample predictions at the
cell level?** A positive \(\Delta\text{elpd}\) with \(|z| > 2\)
indicates meaningful improvement. Conversely, when the posterior of the
hierarchical model concentrates \(\sigma_p\) near zero, both models
predict equally well.

---

## Summary of key quantities

| Quantity | Formula | What it measures |
|----------|---------|------------------|
| lppd | \(\sum_i \log \frac{1}{S}\sum_s \exp(\ell_i^{(s)})\) | In-sample predictive fit |
| \(p_{\text{waic}}\) | \(\sum_i \widehat{\text{Var}}_s(\ell_i^{(s)})\) | Effective number of parameters |
| WAIC | \(-2(\text{lppd} - p_{\text{waic}})\) | Penalized predictive fit (lower is better) |
| elpd (PSIS-LOO) | \(\sum_i \log \hat{p}_{\text{psis}}(y_i \mid y_{-i})\) | Out-of-sample predictive fit |
| \(\hat{k}_i\) | GPD shape on IS weights | Per-observation reliability |
| \(\Delta\text{elpd}_{AB}\) | \(\sum_i (l_{i,A} - l_{i,B})\) | Pairwise model difference |
| Stacking \(\underline{w}^*\) | Maximizes LOO mixture log-score | Optimal ensemble weights |

---

## Using model comparison in SCRIBE

### Quick example

```python
from scribe.mc import compare_models

# Fit two models on the same data
results_nbdm = scribe.fit(adata)
results_hier = scribe.fit(adata, prob_prior="horseshoe")

# Compare
mc = compare_models(
    [results_nbdm, results_hier],
    counts=adata.X,
    model_names=["NBDM", "Hierarchical"],
    gene_names=adata.var_names.tolist(),
    compute_gene_liks=True,
)

# Ranked summary table
print(mc.summary())

# PSIS-LOO diagnostics (k-hat values)
print(mc.diagnostics())

# Per-gene comparison
gene_df = mc.gene_level_comparison("NBDM", "Hierarchical")
print(gene_df.head(20))

# Stacking weights for optimal ensemble
w = mc.stacking_weights()
print(w)
```

### Choosing a criterion

| Scenario | Recommended |
|----------|-------------|
| General use | PSIS-LOO |
| Quick exploratory comparison | WAIC |
| Very large dataset, PSIS-LOO too slow | WAIC |
| Need per-observation diagnostics | PSIS-LOO (\(\hat{k}\) values) |
| No single model dominates | Model stacking |

### API quick reference

| Function | What it does |
|----------|-------------|
| [`compare_models`][scribe.mc.compare_models] | High-level entry point: fits, ranks, and returns results |
| [`waic`][scribe.mc.waic] | JAX-accelerated WAIC from log-likelihood matrix |
| [`compute_psis_loo`][scribe.mc.compute_psis_loo] | PSIS-LOO with Pareto fitting and \(\hat{k}\) diagnostics |
| [`gene_level_comparison`][scribe.mc.gene_level_comparison] | Per-gene elpd differences with SE and z-scores |
| [`compute_stacking_weights`][scribe.mc.compute_stacking_weights] | Optimal ensemble weights via convex optimization |
| [`pseudo_bma_weights`][scribe.mc.pseudo_bma_weights] | AIC-style approximate model weights |

For full API details, see the
[Model Comparison guide](../guide/model-comparison.md) and the
[API Reference](../reference/scribe/mc/).

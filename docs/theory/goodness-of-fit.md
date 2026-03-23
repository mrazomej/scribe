# Goodness-of-Fit Diagnostics

Before trusting any downstream inference---differential expression, model
comparison, denoising---we need to know whether the fitted model actually
describes each gene's count distribution. Two models can be compared with
[WAIC or PSIS-LOO](../guide/model-comparison.md), but those tools only
rank models *relative* to each other. A per-gene **absolute** diagnostic
answers a different question: "Does *this* model fit gene *g* adequately?"

SCRIBE provides two complementary diagnostics:

1. **Randomized Quantile Residuals (RQR)** --- fast, expression-scale
   invariant, MAP-based.
2. **Posterior Predictive Checks (PPC)** --- slower but higher resolution,
   integrates over parameter uncertainty.

This page explains the theory behind both. For code examples, see the
[Model Comparison guide](../guide/model-comparison.md#goodness-of-fit-diagnostics).

---

## Why raw log-likelihood is not enough

A natural candidate for a per-gene fit metric is the gene-level expected log
predictive density (elpd). However, its magnitude depends on two entangled
factors: (1) how well the model fits gene \(g\), and (2) the expression
level and distributional complexity of gene \(g\). Highly expressed genes
produce larger absolute log-likelihoods simply because the NB mass function
at large counts returns different baseline values than at small counts.

This confound makes it impossible to set a single elpd threshold across
genes without an ad-hoc regression correction. What we need is a
**self-normalizing** diagnostic: one whose null distribution is the same
for every gene, regardless of expression level or dispersion.

---

## Part I -- Randomized Quantile Residuals

### The probability integral transform

The theoretical foundation is the classical **probability integral
transform** (PIT):

> If \(Y\) is a continuous random variable with CDF \(F\), then
> \(V = F(Y)\) is Uniform\((0,1)\).

This gives a universal calibration check: if our model \(F\) is correct,
the transformed values should be i.i.d. Uniform.

### The discrete complication

UMI counts are discrete, so the CDF is a step function. Applying
\(F(Y)\) to discrete data produces a set of point masses, not a
continuous uniform. To fix this, Dunn and Smyth (1996) introduced a
simple randomization.

Define the lower and upper cumulative probabilities at the observed count
\(y\):

\[
a(y) = F(y - 1) = \Pr(Y < y), \qquad
b(y) = F(y) = \Pr(Y \leq y).
\]

The **randomized PIT** draws a uniform random number within the gap:

\[
V = a(Y) + W \cdot \bigl(b(Y) - a(Y)\bigr),
\qquad W \text{ drawn Uniform}(0, 1).
\]

Under the true model, \(V\) is exactly Uniform\((0,1)\)---the
randomization "fills in" the steps.

### From uniform to normal

Applying the standard normal quantile function \(\Phi^{-1}\) gives the
**randomized quantile residual** (RQR):

\[
Q = \Phi^{-1}(V).
\]

Under the correctly specified model:

\[
Q \text{ is distributed as } \mathcal{N}(0, 1),
\]

regardless of the original distribution family, its parameters, or the
expression level of the gene. This is the key property: every gene maps
onto the same standard-normal reference.

### Application to the negative binomial

For cell \(c\) and gene \(g\) with observed count \(u_{gc}\) and NB
parameters \((r_g, \hat{p})\), the procedure is:

1. **Lower bound:** \(a_{gc} = F_{\text{NB}}(u_{gc} - 1 \mid r_g, \hat{p})\),
   with \(a_{gc} = 0\) when \(u_{gc} = 0\).
2. **Upper bound:** \(b_{gc} = F_{\text{NB}}(u_{gc} \mid r_g, \hat{p})\).
3. **Randomize:** draw \(w_{gc}\) from Uniform\((0,1)\) and set
   \(v_{gc} = a_{gc} + w_{gc}(b_{gc} - a_{gc})\).
4. **Transform:** \(q_{gc} = \Phi^{-1}(v_{gc})\).

The NB CDF is computed via the regularized incomplete beta function:

\[
F_{\text{NB}}(u \mid r_g, \hat{p})
= I_{\hat{p}}(r_g,\, u + 1),
\]

which is available as `jax.lax.betainc` and JIT-compilable over the
full \((C, G)\) count matrix.

!!! note "Numerical safeguard"
    To prevent \(\Phi^{-1}(v)\) from returning \(\pm\infty\) in
    finite-precision arithmetic, \(v_{gc}\) is clipped to
    \((\epsilon,\, 1 - \epsilon)\) with a small \(\epsilon\)
    (e.g., \(10^{-6}\)) before applying the inverse normal CDF.

### Extension to mixture models

For mixture models with \(K\) components, the relevant CDF is the
**marginal** (not component-conditional) mixture CDF:

\[
F_{gc}(u) = \sum_{k=1}^{K} \pi_k \, F_{\text{NB}}(u \mid r_{gk},
\hat{p}_k).
\]

Using a single component's CDF would condition on the unobserved latent
assignment and produce miscalibrated residuals. The marginal CDF
integrates out the latent variable.

---

## Per-gene summary statistics

Given the residual matrix \(\underline{\underline{Q}} = [q_{gc}]\) of
shape \((C, G)\), SCRIBE computes four diagnostics per gene:

### Residual mean (location miscalibration)

\[
\bar{q}_g = \frac{1}{C}\sum_{c=1}^{C} q_{gc}.
\]

Should be near zero. Values much larger than \(1/\sqrt{C}\) indicate
the model systematically over- or under-predicts gene \(g\).

### Residual variance --- the primary diagnostic

\[
s_g^2 = \frac{1}{C-1}\sum_{c=1}^{C}(q_{gc} - \bar{q}_g)^2.
\]

Under the null, \(s_g^2 \approx 1\). This is the most informative
single summary:

| \(s_g^2\) | Interpretation |
|-----------|----------------|
| Much greater than 1 | Model **underestimates** gene variability (dispersion too low, or missing zero-inflation / bimodality) |
| Much less than 1 | Model **overestimates** variability (prior too diffuse, or gene is nearly Poisson) |
| Close to 1 | Gene is well-described by the model |

### Tail excess

\[
\tau_g = \frac{1}{C}\sum_{c=1}^{C}\mathbf{1}\!\bigl[|q_{gc}| > 2\bigr]
\;-\; 0.0455,
\]

where \(0.0455 = 2(1 - \Phi(2))\) is the expected tail fraction under
\(\mathcal{N}(0,1)\). Positive values flag an excess of outlier cells.

### Kolmogorov--Smirnov distance

\[
D_g = \sup_{t}\bigl|\hat{F}_g(t) - \Phi(t)\bigr|,
\]

an omnibus measure of departure from standard normality. Because the KS
test becomes extremely powerful at large \(C\), it is better used as a
**ranking** criterion than as a formal significance test.

---

## Building a gene mask from RQR

The residual variance gives a simple, interpretable gene filter:

\[
m_g = \mathbf{1}\!\bigl[\tau_{\text{lo}} < s_g^2 < \tau_{\text{hi}}\bigr],
\]

where \(\tau_{\text{lo}}\) and \(\tau_{\text{hi}}\) bracket the
acceptable range around 1. The defaults are generous:

| Threshold | Default | What it catches |
|-----------|:-------:|-----------------|
| \(\tau_{\text{lo}}\) | 0.5 | Severe overestimation of variability |
| \(\tau_{\text{hi}}\) | 1.5 | Severe underestimation of variability |
| \(\tau_{\text{KS}}\) | optional | Omnibus shape misfit |

These can be combined:

\[
m_g = \mathbf{1}\!\bigl[\tau_{\text{lo}} < s_g^2 < \tau_{\text{hi}}\bigr]
\cdot \mathbf{1}\!\bigl[D_g < \tau_{\text{KS}}\bigr].
\]

!!! tip "Relationship to expression-based filtering"
    The GoF mask **complements** the expression-based mask from
    [`compute_expression_mask`][scribe.de.compute_expression_mask].
    Low-expression genes often have high residual variance (the NB CDF is
    coarse for sparse counts), so the two filters overlap. But the GoF
    mask also catches *highly expressed* genes that are poorly fit due to
    zero-inflation or bimodality---cases that expression filtering would
    never flag.

---

## Limitations of MAP-based residuals

RQR evaluates the model CDF at a single point estimate. Two failure modes
limit its discriminative power:

**Systematic bias from shared parameters.** Models like VCP share global
parameters (e.g., per-cell capture probability \(\nu_c\)) across all
genes. A small error in the MAP estimate biases the CDF for every gene
simultaneously, inflating the KS distance uniformly and masking
gene-specific misfit.

**Variance insensitivity to shape.** The residual variance \(s_g^2\)
captures *scale* miscalibration but is blind to *shape* deviations that
preserve the second moment. Two genes with identical \(s_g^2\) can have
very different predictive fit.

These limitations motivate a second, complementary diagnostic.

---

## Part II -- Posterior Predictive Checks

### The idea

Instead of transforming observations through a point-estimate CDF, the PPC
approach generates replicate data from the full posterior and compares the
predicted count histograms directly against the observed ones.

For each posterior draw \(s\), sample a full \((C, G)\) count matrix from
the generative model (including zero-inflation, capture probability, and
mixture assignments as applicable). Then, for each gene, build a histogram
of the replicate counts and compare it bin-by-bin against the observed
histogram.

### Calibration failure rate

For each gene \(g\), the PPC produces a pointwise credible band around
the predicted histogram. The **calibration failure rate** measures the
fraction of non-empty histogram bins where the observed density falls
outside this band:

\[
\hat{c}_g = \frac{
  \sum_{b}\mathbf{1}\!\bigl[
    \hat{f}_g^{\text{obs}}(b) \notin [L_g(b),\, U_g(b)]
  \bigr]\cdot\mathbf{1}\!\bigl[\hat{f}_g^{\text{obs}}(b) > 0\bigr]
}{
  \sum_{b}\mathbf{1}\!\bigl[\hat{f}_g^{\text{obs}}(b) > 0\bigr]
}.
\]

Under a correctly specified model with a 95% credible band, this should
be close to 5%.

### Integrated density distance

The **L1 distance** between observed and PPC-median histograms captures
the *magnitude* of misfit, not just its occurrence:

\[
d_g = \sum_{b=0}^{b_{\max}}
\bigl|\hat{f}_g^{\text{obs}}(b) - \tilde{f}_g(b)\bigr|.
\]

A gene where the model shifts the distribution by one count will have
a higher \(d_g\) than one with slightly wider tails, even if both have
the same calibration failure rate.

### RQR vs. PPC: when to use which

| Aspect | RQR | PPC |
|--------|-----|-----|
| Parameter handling | Single MAP point | Full posterior draws |
| What it tests | CDF transform to N(0,1) | Histogram shape match |
| Scale invariance | Inherent (by construction) | Approximate (via normalization) |
| Shape sensitivity | Limited (summarized by moments) | Direct (bin-level comparison) |
| Computational cost | \(O(CG)\) --- seconds | \(O(SCG)\) --- minutes |

**In practice:** use RQR for a fast initial screen, then upgrade to PPC
when RQR results are ambiguous or when systematic MAP bias is suspected.

---

## Computational considerations

### RQR (MAP-based)

The dominant cost is evaluating the NB CDF via `jax.lax.betainc` for each
cell-gene pair. For a typical dataset (5,000 cells, 2,000 genes), this is
roughly 10 million CDF evaluations --- completing in seconds on GPU.

For mixtures, the cost scales as \(O(CGK)\) where \(K\) is the number of
components.

### PPC

PPC cost has three phases:

| Phase | Cost |
|-------|------|
| Posterior sampling | \(O(S \cdot \dim(\theta))\), done once |
| Predictive generation | \(O(S \cdot C \cdot G_{\text{batch}})\) per gene batch |
| Histogram scoring | \(O(S \cdot G_{\text{batch}})\) per batch |

For \(S=500\), \(C=5{,}000\), \(G=20{,}000\), \(G_{\text{batch}}=50\):
peak memory per batch is roughly 500 MB, with 400 batches total. Estimated
wall time: 15--30 minutes depending on hardware. The implementation uses
a "sample once, predict in batches" strategy to avoid materializing the
full \((S, C, G)\) array.

---

## Using GoF diagnostics in SCRIBE

### RQR mask (fast default)

```python
from scribe.mc import compute_gof_mask

# Boolean mask: True for genes the model describes adequately
mask = compute_gof_mask(
    counts,
    results,
    min_variance=0.5,   # lower bound on residual variance
    max_variance=1.5,   # upper bound on residual variance
    max_ks=None,        # optional KS distance threshold
)
```

### Detailed RQR scores

```python
from scribe.mc import compute_quantile_residuals, goodness_of_fit_scores
import jax.random as random

# Step 1: compute residual matrix (C, G)
residuals = compute_quantile_residuals(
    counts, r, p,
    rng_key=random.PRNGKey(0),
)

# Step 2: per-gene summary statistics
scores = goodness_of_fit_scores(residuals)
print(scores["variance"][:10])    # should be near 1
print(scores["ks_distance"][:10]) # smaller is better
```

### PPC mask (higher resolution)

```python
from scribe.mc import compute_ppc_gof_mask

mask, scores = compute_ppc_gof_mask(
    counts,
    results,
    n_ppc_samples=500,
    gene_batch_size=50,
    max_calibration_failure=0.5,
    return_scores=True,
)

print(scores["calibration_failure"][:10])
print(scores["l1_distance"][:10])
```

### Feeding the mask into DE

The GoF mask plugs directly into the DE pipeline via the `gene_mask`
parameter:

```python
from scribe.de import compare

de = compare(
    results_A, results_B,
    method="empirical",
    gene_mask=mask,       # only analyze well-fit genes
    component_A=0, component_B=0,
)
```

### Quick reference

| Function | What it does | Speed |
|----------|-------------|:-----:|
| [`compute_gof_mask`][scribe.mc.compute_gof_mask] | Boolean mask from RQR variance / KS | Fast |
| [`compute_quantile_residuals`][scribe.mc.compute_quantile_residuals] | Raw residual matrix \((C, G)\) | Fast |
| [`goodness_of_fit_scores`][scribe.mc.goodness_of_fit_scores] | Per-gene mean, variance, tail excess, KS | Fast |
| [`compute_ppc_gof_mask`][scribe.mc.compute_ppc_gof_mask] | Boolean mask from PPC calibration / L1 | Slow |
| [`ppc_goodness_of_fit_scores`][scribe.mc.ppc_goodness_of_fit_scores] | Per-gene calibration failure and L1 distance | Slow |

For more on model comparison and the full API, see the
[Model Comparison guide](../guide/model-comparison.md) and the
[API Reference](../reference/scribe/mc/).

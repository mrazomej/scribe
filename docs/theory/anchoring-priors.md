# Anchoring Priors

The variable capture probability (VCP) model introduces two layers of
practical non-identifiability that can degrade inference. This page describes
two complementary anchoring priors that resolve them, each justified by a
law-of-large-numbers concentration argument. For complete derivations, see the
accompanying paper.

---

## Layer 1: The Capture Prior

### The degeneracy

The expected observed count for gene \(g\) in cell \(c\) is:

\[
\langle u_{gc} \rangle = \mu_g \cdot \nu_c,
\]

where \(\mu_g\) is the biological mean expression and \(\nu_c\) is the
cell-specific capture probability. The observed library size \(L_c = \sum_g
u_{gc}\) constrains only the **product** \(\nu_c \cdot \mu_T\) (where
\(\mu_T = \sum_g \mu_g\)), not each factor individually. With a flat prior on
\(\nu_c\), the model is free to trade off between "high capture, low total
expression" and "low capture, high total expression."

This degeneracy manifests as systematic biases when fitting multiple datasets:
even when the underlying biology is identical, the optimizer can assign
different capture efficiencies to each dataset, absorbing expression
differences into \(\nu_c\) rather than into the biological parameters.

### The fundamental relationship

The key insight is that **the capture probability is approximately the ratio
of the observed library size to the total mRNA content**:

\[
\nu_c \approx \frac{L_c}{M_c}.
\]

The observed library size \(L_c\) is known from the data. If we can constrain
\(M_c\) (the total mRNA content), we can constrain \(\nu_c\).

### Why total mRNA is nearly deterministic

Under the shared-\(p\) model, the total mRNA \(M_c = \sum_{g=1}^G m_g\) is
itself a negative binomial with aggregate dispersion \(r_T = \sum_g r_g\).
Its coefficient of variation is:

\[
\text{CV}(M_c) = \frac{1}{\sqrt{r_T \cdot p}}.
\]

For a mammalian cell with \(G \approx 20{,}000\) genes and typical per-gene
dispersions of order 1--10, the aggregate dispersion \(r_T\) exceeds
50,000, giving \(\text{CV}(M_c) \approx 0.5\%\). This is simply the
**law of large numbers over genes**: summing tens of thousands of independent
random variables produces a total that concentrates tightly around its
expectation. Within the model, total mRNA is effectively deterministic.

### The biology-informed prior

Given an organism-specific expected total mRNA \(M_0\) (e.g., \(\approx
200{,}000\) for mammalian cells), the prior on the working variable
\(\eta_c = \log(M_c / L_c)\) is:

\[
\eta_c \sim \mathcal{N}^{+}(\log M_0 - \log L_c, \; \sigma_M^2),
\]

where \(\mathcal{N}^{+}\) denotes a normal distribution truncated below at
zero (enforcing the physical constraint \(\nu_c \leq 1\)). The capture
probability and odds are recovered exactly:

\[
\nu_c = \exp(-\eta_c), \qquad \phi_c = \exp(\eta_c) - 1.
\]

Key properties:

- **Cell-specific centering**: cells with larger library sizes get a prior
  centered at higher capture efficiency, and vice versa
- **Biologically grounded scale**: \(\sigma_M \approx 0.5\) corresponds to
  roughly 1.5--2x fold variation in total mRNA, consistent with
  known biological variation
- **No approximation**: the transformations are exact for all capture regimes,
  including high-capture experiments where \(\nu_c\) approaches 0.5 or beyond

### Interpretation of \(\sigma_M\)

| \(\sigma_M\) | Behavior |
|--------------|----------|
| \(\approx 0\) | **Deterministic anchoring** -- \(\nu_c = L_c / M_0\), no degrees of freedom |
| \(\approx 0.5\) (default) | Moderate anchoring, accommodates cell-size heterogeneity |
| \(> 1\) | Weak anchoring, approaching the flat prior |

### Organism-specific values

| Organism | \(M_0\) (mRNA/cell) |
|----------|---------------------|
| Human | 200,000 |
| Mouse | 200,000 |
| Yeast (*S. cerevisiae*) | 60,000 |
| *E. coli* | 3,000 |

### Relationship to standard normalization

Standard normalization (CPM, scran, etc.) divides by library size and
multiplies by a constant: \(\tilde{u}_{gc} = u_{gc} \cdot C / L_c\). This is
structurally identical to \(\nu_c = L_c / C\), but with an arbitrary reference
constant. The biology-informed prior improves on this by:

1. Anchoring to a biologically meaningful \(M_0\) instead of an arbitrary
   constant
2. Operating within the full probabilistic model, preserving uncertainty
   quantification
3. Accounting for the nonlinear coupling between capture and expression through
   the likelihood (the effective \(\hat{p}\) depends nonlinearly on both \(p\)
   and \(\nu_c\))

---

## Layer 2: The Mean Anchoring Prior

### The residual degeneracy

Even after the capture prior pins \(\nu_c\), a second non-identifiability
remains. The expected count \(\langle u_{gc} \rangle \approx \mu_g \cdot
L_c / M_0\) constrains \(\mu_g\) through the **first moment** (the mean),
while the overdispersion \(\phi_g\) is constrained only through the **second
moment** (the variance-to-mean ratio). Because the variance is inherently
noisier to estimate than the mean, the likelihood surface exhibits a
characteristic **ridge** in \((\mu_g, \phi_g)\) space: the mean is tightly
constrained, but the overdispersion direction is broad.

This manifests as:

- Different datasets settling on different \(\phi\) values despite identical
  biology
- The hierarchical gene-specific \(\phi_g\) model failing in the mean-odds
  parameterization (where \(\phi_g\) only affects the variance, not the mean)
- Poor cross-dataset parameter correspondence despite good within-model
  posterior predictive checks

### The concentration argument

The sample mean of gene \(g\) across \(C\) cells,

\[
\bar{u}_g = \frac{1}{C} \sum_{c=1}^C u_{gc},
\]

concentrates around \(\mu_g \cdot \bar{\nu}\) (where
\(\bar{\nu} = \frac{1}{C} \sum_c \nu_c\)) with coefficient of variation:

\[
\text{CV}(\bar{u}_g) \approx \frac{1}{\sqrt{C}}
\sqrt{\frac{1}{\mu_g \bar{\nu}} + \frac{1}{r_g}}.
\]

This is the **law of large numbers over cells**, complementing the capture
prior's concentration over genes. Typical values for \(C = 1{,}000\) cells:

| Gene type | \(\mu_g\) | \(r_g\) | CV |
|-----------|-----------|---------|-----|
| Low expression, high overdispersion | 10 | 1 | ≈4.5% |
| Moderate expression | 100 | 5 | ≈1.7% |
| High expression | 1,000 | 10 | ≈1.0% |

With \(C = 5{,}000\) cells, these improve by \(\sqrt{5}\) to sub-percent
precision for all but the most extreme genes.

### The data-informed prior

The empirical anchor for each gene is:

\[
\hat{\mu}_g = \frac{\bar{u}_g + \epsilon}{\bar{\nu}},
\]

where \(\epsilon > 0\) is a small pseudocount. The prior on \(\mu_g\) is
log-normal, centered on this anchor:

\[
\log(\mu_g) \sim \mathcal{N}\!\left(\log(\hat{\mu}_g), \;
\sigma_\mu^2\right).
\]

### Interpretation of \(\sigma_\mu\)

| \(\sigma_\mu\) | Behavior |
|----------------|----------|
| 0.1--0.2 | **Tight anchoring** -- constrains \(\mu_g\) within ≈10--20% of the data-implied value |
| **0.3** (recommended default) | Moderate anchoring, accommodates finite-sample noise |
| \(> 1\) | Weak anchoring, approaching the uninformative prior |

### Why this fixes the hierarchical \(\phi_g\) failure

Without the anchor, the optimizer faces \(2G\) free parameters (\(\mu_g\)
and \(\phi_g\) per gene) with \(G\) tight constraints (means) and \(G\)
loose constraints (variances), creating \(G\) independent degeneracy
manifolds.

With the anchor, \(\mu_g\) is effectively fixed (up to \(\sigma_\mu\) slack),
leaving only \(\phi_g\) per gene. Any change in \(\phi_g\) now unambiguously
changes the overdispersion without being compensated by a shift in \(\mu_g\).
The banana-shaped ridges in \((\mu_g, \phi_g)\) space are replaced by
well-defined optima along the \(\phi_g\) axis.

---

## The Three-Layer Regularization Stack

The capture prior and mean anchor, combined with the hierarchical prior on
overdispersion, form a complete regularization stack:

\[
\boxed{
\begin{aligned}
&\text{Layer 1 (capture):} &&
\nu_c \approx L_c / M_0
&& \text{(LLN over genes)} \\
&\text{Layer 2 (mean):} &&
\mu_g \approx \bar{u}_g / \bar{\nu}
&& \text{(LLN over cells)} \\
&\text{Layer 3 (overdispersion):} &&
\phi_g \sim \text{Hierarchical prior}
&& \text{(shrinkage across genes)}
\end{aligned}
}
\]

Each layer constrains one factor in the expected count
\(\langle u_{gc} \rangle = \mu_g \cdot \nu_c\), with the overdispersion
regularized by a [hierarchical prior](hierarchical-priors.md) across genes.

---

## Using Anchoring Priors in SCRIBE

### Capture prior (Layer 1)

Enable the biology-informed capture prior by passing organism-specific total
mRNA information through the `priors` dictionary:

```python
import scribe

# Using organism shortcut (resolves M_0 and sigma_M automatically)
results = scribe.fit(
    adata,
    variable_capture=True,
    priors={"organism": "human"},
)

# Using explicit values (for other organisms or custom M_0)
import math
results = scribe.fit(
    adata,
    variable_capture=True,
    priors={"eta_capture": (math.log(200_000), 0.5)},
)
```

The `priors["organism"]` key accepts `"human"`, `"mouse"`, `"yeast"`, or
`"ecoli"` and resolves to the appropriate \(M_0\) and \(\sigma_M\). For
other organisms, pass `priors["eta_capture"]` directly as a tuple of
`(log_M_0, sigma_M)`.

| Parameter | Description |
|-----------|-------------|
| `priors["organism"]` | Shortcut: `"human"` (\(M_0=200{,}000\)), `"mouse"`, `"yeast"` (\(60{,}000\)), `"ecoli"` (\(3{,}000\)) |
| `priors["eta_capture"]` | Explicit `(log_M_0, sigma_M)` tuple |
| `capture_scaling_prior` | Data-driven shared scaling across datasets: `"none"`, `"gaussian"`, `"horseshoe"`, `"neg"` |

### Mean anchoring prior (Layer 2)

Enable the data-informed mean anchor with the `expression_anchor` flag:

```python
results = scribe.fit(
    adata,
    variable_capture=True,
    priors={"organism": "human"},
    expression_anchor=True,
    expression_anchor_sigma=0.3,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expression_anchor` | `False` | Enable data-informed anchoring prior on \(\mu_g\) |
| `expression_anchor_sigma` | 0.3 | Log-scale standard deviation (smaller = tighter anchor) |

!!! note
    Enabling `expression_anchor` automatically sets `unconstrained=True`. For
    VCP models, `priors["organism"]` or `priors["eta_capture"]` must also be
    set so that \(\bar{\nu}\) can be estimated from library sizes and \(M_0\).
    For non-VCP models, \(\bar{\nu} = 1\) is used by default.

### Combined usage (recommended for VCP models)

For the best results with variable capture models, enable both layers:

```python
results = scribe.fit(
    adata,
    variable_capture=True,
    parameterization="mean_odds",
    priors={"organism": "human"},
    expression_anchor=True,
    expression_anchor_sigma=0.3,
)
```

This resolves both the \(\nu_c \cdot \mu_T\) degeneracy (Layer 1) and the
\(\mu_g\text{--}\phi_g\) degeneracy (Layer 2), producing well-identified
parameters that are comparable across datasets and compatible with
hierarchical gene-specific \(\phi_g\) models.

For the full API details, see the [API Reference](../reference/scribe/api/).

---

!!! tip "Next steps"
    - See the [Dirichlet-Multinomial Model](dirichlet-multinomial.md) for the
      NB generative model and capture-probability derivation that these priors
      regularize.
    - See [Hierarchical Priors](hierarchical-priors.md) for Layer 3 of the
      regularization stack — adaptive shrinkage on overdispersion using
      Gaussian, Horseshoe, and NEG prior families.
    - See [Beta Negative Binomial](beta-negative-binomial.md) for an
      orthogonal overdispersion extension whose mean structure critically
      depends on the mean anchoring prior.
    - See the [Model Selection](../guide/model-selection.md) guide for
      NBVCP and ZINBVCP models where anchoring priors are most important.

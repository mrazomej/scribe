# Differential Expression

Identifying genes whose expression differs between experimental conditions is
one of the central tasks in single-cell transcriptomics. Most widely used
tools---DESeq2, edgeR, Wilcoxon tests in Scanpy---were designed for bulk
RNA-seq or treat each gene independently without accounting for the
fundamental mathematical structure of normalized expression data. SCRIBE takes
a different approach, grounded in two key insights:

1. **Normalized gene expression is compositional.** Proportions sum to one,
   so a change in one gene *mechanically* forces changes in all others. Methods
   that ignore this produce spurious results.
2. **The model already knows the uncertainty.** Having fitted a full Bayesian
   model, we can propagate posterior uncertainty directly into the DE
   analysis---no ad-hoc dispersion estimation, no permutation tests, no
   p-value hacking.

This page explains the theory behind SCRIBE's two complementary DE
frameworks: **Compositional DE** (in log-ratio space) and **Biological DE**
(in denoised count space). For API usage and code examples, see the
[Differential Expression guide](../guide/differential-expression.md).

---

## Part I -- Compositional Differential Expression

### The compositionality problem

Consider a simple example with three genes. Suppose gene 1 doubles its
absolute expression between conditions A and B, while genes 2 and 3 remain
constant. Despite genes 2 and 3 being truly unchanged, their *proportions*
must decrease to maintain the constraint
\(\rho_1 + \rho_2 + \rho_3 = 1\). A naive test would call all three genes
as differentially expressed.

This is the **closure problem**: proportions carry information only about
*relative* quantities, never absolute ones. The mathematical space they
live in---the simplex \(\Delta^{D-1}\)---is fundamentally different from
the Euclidean space that standard statistical tests assume.

!!! warning "Why standard tools can mislead"
    Methods like DESeq2 and edgeR model absolute transcript abundances via
    the negative binomial distribution. When applied to normalized
    proportions, they encounter:

    - **Spurious correlations** from the closure constraint
    - **Reference dependence** when using log-ratios to a single gene
    - **Loss of multivariate structure** by testing genes independently
    - **Scale ambiguity** in interpreting "fold-changes"

    SCRIBE addresses all four by working in proper compositional
    coordinate systems.

### Escaping the simplex: log-ratio coordinates

To perform statistical inference on compositions, we transform to coordinate
systems where standard multivariate Gaussian methods apply. SCRIBE uses two
complementary transformations:

**Centered Log-Ratio (CLR).** Each gene's coordinate is its log-expression
relative to the geometric mean of all genes:

\[
z_{\text{CLR},g} = \log\!\left(
\frac{\rho_g}{\left(\prod_{j=1}^{D} \rho_j\right)^{1/D}}
\right).
\]

The CLR treats all genes symmetrically (no arbitrary reference gene),
and each coordinate has a natural interpretation: positive means the
gene is expressed above the geometric mean; negative means below.
The CLR coordinates sum to zero by construction, reflecting the
one-dimensional constraint lost by normalization.

**Isometric Log-Ratio (ILR).** An orthonormal rotation of the CLR
coordinates into \(\mathbb{R}^{D-1}\), where standard Euclidean geometry
applies. The ILR is particularly useful for gene-set and pathway analysis,
where each coordinate can be designed to represent a biologically
meaningful *balance* between groups of genes.

### How DE works in CLR space

Given two conditions A and B, each with a fitted logistic-normal model
in CLR space:

\[
\underline{z}_A \;\sim\; \mathcal{N}(\underline{\mu}_A,\;
\underline{\underline{\Sigma}}_A), \qquad
\underline{z}_B \;\sim\; \mathcal{N}(\underline{\mu}_B,\;
\underline{\underline{\Sigma}}_B),
\]

the CLR difference is also Gaussian:

\[
\underline{\Delta} = \underline{z}_A - \underline{z}_B
\;\sim\;
\mathcal{N}(\underline{\mu}_A - \underline{\mu}_B,\;
\underline{\underline{\Sigma}}_A + \underline{\underline{\Sigma}}_B).
\]

For each gene \(g\), the marginal difference
\(\Delta_g \sim \mathcal{N}(\mu_{\Delta,g},\; \sigma_{\Delta,g}^2)\) gives
us everything we need: a posterior mean effect, a posterior standard
deviation, and tail probabilities---all in closed form.

The covariance matrices have a **low-rank plus diagonal** structure
(\(\underline{\underline{\Sigma}} = \underline{\underline{W}}\,\underline{\underline{W}}^\top + \text{diag}(\underline{d})\))
that makes every operation scale as \(O(k^2 D)\) or better, where
\(k \approx 50\) is the rank and \(D \approx 30{,}000\) is the number of
genes. A complete analysis runs in seconds.

### Three DE methods

SCRIBE provides three methods, all producing the same type of output
but differing in their assumptions:

#### Parametric (analytical Gaussian)

Fits a low-rank logistic-normal model to the posterior predictive
distribution of compositions, then computes all statistics (posterior mean,
standard deviation, lfsr, pathway balances) in closed form from the
Gaussian parameters. This is extremely fast and produces arbitrarily fine
lfsr resolution.

**Limitation:** the Gaussian approximation can break down for genes with
very low expression, multimodal posteriors, or extreme sparsity.

#### Empirical (Monte Carlo) -- recommended

Draws posterior samples from the Dirichlet predictive distribution,
transforms them to CLR space, and computes all DE statistics by **direct
counting** over the samples. No distributional assumptions are required.

For each gene \(g\) and posterior sample \(s\):

\[
\Delta_g^{(s)} =
z_{A,g,\text{CLR}}^{(s)} - z_{B,g,\text{CLR}}^{(s)},
\]

and the lfsr is simply the fraction of samples with the minority sign:

\[
\widehat{\text{lfsr}}_g = \min\!\left(
\frac{1}{N}\sum_s \mathbf{1}[\Delta_g^{(s)} > 0],\;\;
1 - \frac{1}{N}\sum_s \mathbf{1}[\Delta_g^{(s)} > 0]
\right).
\]

!!! tip "Why empirical is recommended"
    Per-gene Gaussianity diagnostics (skewness, excess kurtosis,
    Jarque-Bera statistic) consistently show that a non-trivial fraction
    of genes violate the Gaussian assumption, especially lowly expressed
    genes and those with zero-inflation. The empirical method avoids this
    issue entirely at modest computational cost (seconds on GPU).

    When the Gaussian assumption *does* hold, the empirical and parametric
    methods converge to the same answer---so the empirical method is
    strictly safer.

#### Shrinkage (empirical Bayes)

Builds on the empirical method by learning a **genome-wide effect-size
distribution** from all genes simultaneously, then updating each gene's
posterior accordingly. This is SCRIBE's implementation of the adaptive
shrinkage idea from Stephens (2016):

1. Model each gene's observed effect as
   \(\hat{\Delta}_g \mid \beta_g \sim \mathcal{N}(\beta_g, s_g^2)\).
2. Place a scale mixture of normals prior on the true effect:
   \(\beta_g \sim \sum_k w_k \,\mathcal{N}(0, \sigma_k^2)\).
3. Estimate the weights \(w_k\) by maximum marginal likelihood (EM).
4. Compute the **shrunk posterior** for each gene---a mixture of Gaussians
   with component-specific shrinkage factors.

The result is that noisy effects are pulled toward zero (especially when
most genes are not DE), while strong effects are left largely unchanged.
The estimated null proportion \(w_0\) is a useful diagnostic: it tells you
what fraction of the genome is unaffected by the perturbation.

### Measuring evidence: the local false sign rate

Rather than p-values and FDR, SCRIBE uses the **local false sign rate
(lfsr)**---the posterior probability that you would assign the wrong sign
to a gene's differential expression:

\[
\text{lfsr}_g = \min\!\big(
P(\Delta_g > 0 \mid \text{data}),\;
P(\Delta_g < 0 \mid \text{data})
\big).
\]

The lfsr has a direct Bayesian interpretation: if \(\text{lfsr}_g = 0.02\),
there is a 2% posterior probability that you are wrong about whether the
gene went up or down. This is more informative than a p-value, which only
tells you how surprising the data would be under a null hypothesis.

#### Practical significance

For many applications, tiny fold-changes are biologically irrelevant.
SCRIBE supports a **practical significance threshold** \(\tau\): rather than
asking "is the effect non-zero?", you ask "is the effect larger than
\(\tau\)?":

\[
\text{lfsr}_g(\tau) = 1 - \max\!\big(
P(\Delta_g > \tau \mid \text{data}),\;
P(\Delta_g < -\tau \mid \text{data})
\big).
\]

A common choice is \(\tau = \log(1.1) \approx 0.095\), corresponding to a
10% fold-change in relative abundance.

### Controlling the false discovery rate: PEFP

When calling hundreds of genes as DE, we need to control the overall error
rate. SCRIBE uses the **Posterior Expected False Discovery Proportion
(PEFP)**, a Bayesian analogue of FDR:

\[
\text{PEFP}(S) = \frac{1}{|S|} \sum_{g \in S} \text{lfsr}_g.
\]

This is simply the average lfsr over the set of called genes. An efficient
\(O(D \log D)\) algorithm finds the largest lfsr threshold that controls
PEFP at a desired level (e.g., 5%).

Unlike the Benjamini-Hochberg procedure, the PEFP conditions on the
observed data and uses true posterior probabilities. It adapts to the data
through the average lfsr rather than treating all hypotheses symmetrically.

### Gene-set and pathway analysis

The same framework extends naturally to gene sets. A **balance** compares
the geometric mean expression of a pathway against its complement:

\[
b(P, P^c) = \log\!\left(
\frac{\text{geom-mean}(\rho_g : g \in P)}
{\text{geom-mean}(\rho_g : g \notin P)}
\right).
\]

In CLR coordinates, this is a linear contrast, so its posterior distribution
is available analytically (parametric) or by projection of the Monte Carlo
samples (empirical). The same lfsr and PEFP machinery applies for
multiplicity control across pathways.

SCRIBE also provides a **within-pathway perturbation test** that detects
coordinated internal rearrangement (some pathway genes going up, others
going down) even when the average balance is near zero.

### How this compares to other tools

| Feature | DESeq2 / edgeR | Wilcoxon | scVI | SCRIBE |
|---------|:-:|:-:|:-:|:-:|
| Respects compositionality | No | No | No | **Yes** |
| Accounts for gene-gene correlations | No | No | Partially | **Yes** |
| Bayesian uncertainty | No | No | Partially | **Yes** |
| Pathway analysis (integrated) | No | No | No | **Yes** |
| No distributional assumption needed | No | Yes | No | **Yes** (empirical) |
| Error control | BH-FDR | BH-FDR | -- | **PEFP** |
| Computational cost | Fast | Fast | Moderate | **Fast** |

The compositional approach means SCRIBE's DE results are
**reference-invariant**: they do not depend on an arbitrary choice of
reference gene or normalization constant. Combined with the Bayesian error
control, this makes the results more reproducible and easier to interpret.

---

## Part II -- Biological Differential Expression

### Why a second framework?

The CLR-based analysis above is the correct statistical tool for testing
*compositional* hypotheses. However, it has a known artefact: the logarithm
amplifies small values, so genes with very low expression can appear
strongly DE in CLR space when their biological distributions have barely
changed.

To complement the compositional view, SCRIBE computes DE metrics directly
on the **denoised negative binomial distribution**---the biological count
distribution before technical noise is applied. These metrics are free of
the closure constraint and provide a second, independent perspective on
each gene.

### Metric 1: Biological log-fold change (LFC)

The most natural measure of mean expression change:

\[
\text{LFC}_g = \log\!\left(
\frac{\mu_g^A}{\mu_g^B}
\right),
\]

where \(\mu_g = r_g(1 - p_g)/p_g\) is the NB mean. Unlike the CLR
difference, the biological LFC is **free of compositional closure**: a gene
whose mean expression does not change will have \(\text{LFC}_g = 0\)
regardless of what other genes do.

The posterior distribution of the LFC is obtained by computing
\(\text{LFC}_g^{(s)}\) for each posterior sample \(s\) and then
applying the same counting-based statistics (mean, standard deviation,
lfsr) as in the empirical compositional pipeline.

### Metric 2: Log-variance ratio (LVR)

Two conditions can have identical means yet differ in variability. A gene
that switches from tight regulation to bursty expression may have
\(\mu_g^A \approx \mu_g^B\) but vastly different variances:

\[
\text{LVR}_g = \log\!\left(
\frac{\text{Var}^A(m_g)}{\text{Var}^B(m_g)}
\right)
= \text{LFC}_g + \log\!\left(\frac{p_g^B}{p_g^A}\right).
\]

A gene with \(\text{LFC}_g \approx 0\) but \(\text{LVR}_g \neq 0\) has
undergone a pure change in overdispersion---the "shape" of its expression
distribution changed without its centre moving.

### Metric 3: Gamma Jeffreys divergence

For a holistic measure that captures *any* distributional shift---mean,
variance, skewness, shape---SCRIBE computes the **Jeffreys divergence**
between the latent Gamma rate distributions:

\[
J_g = \text{KL}(P_g^A \| P_g^B) + \text{KL}(P_g^B \| P_g^A).
\]

This exploits the Poisson-Gamma representation of the negative binomial
(\(\lambda_g \sim \text{Gamma}(r_g, p_g/(1-p_g))\),
\(m_g \mid \lambda_g \sim \text{Poisson}(\lambda_g)\)) and the
fact that the KL divergence between two Gamma distributions has a
closed-form expression involving only \(r_g\), \(p_g\), and the digamma
function.

The Gamma KL is preferred over the NB KL because the latter has no
tractable closed form when the dispersion parameters differ between
conditions.

### Four complementary views of each gene

| Metric | Space | Captures | Closure-free? |
|--------|-------|----------|:---:|
| CLR difference | Compositional simplex | Relative abundance shift | No |
| Biological LFC | Denoised count space | Mean expression shift | Yes |
| Log-variance ratio | Denoised count space | Dispersion shift | Yes |
| Jeffreys divergence | Latent Gamma rate | Full distributional shift | Yes |

No single metric is universally "best." The recommended workflow is:

1. **Screen with CLR**: use the CLR-based lfsr as the primary filter
   (it respects compositionality).
2. **Validate with biological LFC**: filter out compositional artefacts,
   especially for lowly expressed genes.
3. **Detect variance changes**: inspect the LVR for genes with small LFC
   but high Jeffreys divergence.
4. **Flag distributional shifts**: use the Jeffreys divergence as a
   catch-all for any change in the NB distribution.

---

## Using DE in SCRIBE

### Quick example

```python
import scribe
from scribe.de import compare
import jax.numpy as jnp

# Fit models for two conditions
results_A = scribe.fit(adata_treatment, model="nbdm")
results_B = scribe.fit(adata_control, model="nbdm")

# Empirical DE (recommended)
de = compare(
    results_A, results_B,
    method="empirical",       # or "shrinkage" for adaptive shrinkage
    component_A=0, component_B=0,
)

# Gene-level analysis with 10% fold-change threshold
results = de.gene_level(tau=jnp.log(1.1))

# Call DE genes at 5% PEFP
threshold = de.find_threshold(target_pefp=0.05)
is_de = de.call_genes(lfsr_threshold=threshold)
print(f"Found {is_de.sum()} DE genes at 5% PEFP")

# Biological-level metrics
bio = de.biological_level(tau_lfc=jnp.log(1.5))
```

### Choosing a method

| Scenario | Recommended method |
|----------|-------------------|
| General use | `"empirical"` |
| Most genes are not DE, want genome-wide shrinkage | `"shrinkage"` |
| Very fast, Gaussianity verified | `"parametric"` |

!!! tip "Upgrading without recomputation"
    You can upgrade an empirical result to shrinkage without re-running
    the expensive Dirichlet sampling:

    ```python
    de_emp = compare(results_A, results_B, method="empirical", ...)
    de_shrink = de_emp.shrink()
    ```

For full API details and more examples, see the
[Differential Expression guide](../guide/differential-expression.md) and the
[API Reference](../reference/scribe/de/).

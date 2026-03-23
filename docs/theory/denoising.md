# Bayesian Denoising

This page summarizes the theoretical foundation for Bayesian denoising of
single-cell transcriptional profiles in SCRIBE. For the full derivation with
all algebraic steps, see the accompanying paper.

## Motivation

The [generative model](dirichlet-multinomial.md#model-setup) describes how
observed UMI counts arise from true transcript counts via stochastic capture.
Once the model is fitted, a natural question arises: **given an observed cell
profile, what was the most likely "true" transcriptional profile before
capture loss and dropout?**

This section shows that the denoising problem has a clean, closed-form
Bayesian solution that falls directly out of the conjugacy structure underlying
the model.

## Problem Setup

Recall the two-stage generative process for a single gene \(g\) in cell \(c\).
The true (unobserved) transcript count follows a negative binomial:

\[
m_g \sim \text{NB}(r_g, p),
\]

and the observed UMI count arises from binomial sub-sampling:

\[
u_g \mid m_g, \nu_c \sim \text{Binomial}(m_g, \nu_c),
\]

where \(\nu_c \in (0, 1)\) is the cell-specific capture probability. The
denoising problem is to compute \(\pi(m_g \mid u_g, r_g, p, \nu_c)\).

## Key Result: Denoised Count Distribution

The derivation exploits the **Poisson-Gamma representation** of the negative
binomial. Writing \(m_g = u_g + d_g\) (captured + uncaptured transcripts) and
applying the Poisson thinning property gives conditional independence of
\(u_g\) and \(d_g\) given the latent Gamma rate \(\lambda_g\). A standard
Bayesian update of the Gamma prior with the Poisson likelihood yields the
posterior:

\[
\lambda_g \mid u_g \sim \text{Gamma}\!\left(r_g + u_g, \;
\frac{p}{1 - p} + \nu_c\right).
\]

Marginalizing over \(\lambda_g\), the number of uncaptured transcripts follows
a negative binomial, giving the **denoised count distribution**:

\[
\boxed{
m_g \mid u_g \sim u_g + \text{NB}\!\left(r_g + u_g, \;\;
\nu_c + p(1 - \nu_c)\right).
}
\]

The denoised count equals the observed count \(u_g\) plus a random correction
\(d_g\) drawn from a negative binomial whose shape depends on both the prior
dispersion \(r_g\) and the observation \(u_g\).

## Point Estimates

### Posterior mean (shrinkage estimator)

The posterior mean of the denoised count has a compact closed form:

\[
\boxed{
\langle m_g \mid u_g \rangle =
\frac{u_g + r_g(1 - \nu_c)(1 - p)}{\nu_c + p(1 - \nu_c)}.
}
\]

This is a **shrinkage estimator**: the numerator combines the observed count
\(u_g\) with a prior correction \(r_g(1 - \nu_c)(1 - p)\) that pulls the
estimate toward the prior mean, while the denominator scales by the effective
capture probability.

### Limiting behavior

The posterior mean exhibits sensible behavior in several important limits:

| Limit | Result | Interpretation |
|-------|--------|----------------|
| **Perfect capture** (\(\nu_c = 1\)) | \(\langle m_g \rangle = u_g\) | No denoising needed |
| **Zero observation** (\(u_g = 0\)) | \(\langle m_g \rangle = \frac{r_g(1-\nu_c)(1-p)}{\nu_c + p(1-\nu_c)}\) | Strictly positive -- prior drives the estimate |
| **Weak prior** (\(r_g \to 0\)) | \(\langle m_g \rangle \approx \frac{u_g}{\nu_c + p(1-\nu_c)}\) | Naive rescaling by capture probability |
| **Strong signal** (\(u_g \gg r_g\)) | Same as weak prior | Data dominate |

### Posterior variance

The posterior variance quantifies denoising uncertainty:

\[
\text{Var}(m_g \mid u_g) = (r_g + u_g) \cdot
\frac{(1 - \nu_c)(1 - p)}{[\nu_c + p(1 - \nu_c)]^2}.
\]

This variance decreases as \(\nu_c \to 1\) (better capture means less
uncertainty) and increases with \(u_g\) (larger counts imply a wider range
of plausible true counts).

## Extension to Zero-Inflated Models

For ZINB models, the observed distribution includes a point mass at zero via a
technical dropout gate \(g\):

\[
u_g \sim g \cdot \delta_0 + (1 - g) \cdot \text{NB}(r_g, \hat{p}).
\]

### Non-zero observations (\(u_g > 0\))

The gate could not have produced a positive count, so the denoising is
**identical** to the pure NB case above.

### Zero observations (\(u_g = 0\))

When \(u_g = 0\), the zero may have come from either pathway. By Bayes' rule,
the posterior probability that the gate was responsible is:

\[
w = \frac{g}{g + (1 - g) \cdot \hat{p}^{\,r_g}}.
\]

The denoised count distribution is a **mixture** of two components:

\[
\pi(m_g \mid u_g = 0) = w \cdot
\underbrace{\text{NB}(r_g, p)}_{\text{gate pathway}} + (1 - w) \cdot
\underbrace{\text{NB}(r_g, \; \nu_c + p(1 - \nu_c))}_{\text{NB pathway}}.
\]

- **Gate pathway** (weight \(w\)): the cell was expressing the gene normally
  but technical dropout prevented observation. The denoised count is drawn
  from the biological prior \(\text{NB}(r_g, p)\).
- **NB pathway** (weight \(1 - w\)): the zero reflects genuine low expression
  combined with capture loss. The standard denoising formula applies with
  \(u_g = 0\).

The gate pathway mean is always **larger** than the NB pathway mean when
\(\nu_c < 1\), reflecting the intuition that dropout-induced zeros correspond
to higher expected true expression.

### Models without variable capture probability

For ZINB or NBDM models without cell-specific capture, set \(\nu_c = 1\).
Then \(d_g = 0\) almost surely, so \(m_g = u_g\) for all non-zero
observations. The zero-inflation denoising still applies for \(u_g = 0\):

\[
\langle m_g \mid u_g = 0 \rangle = w \cdot \frac{r_g(1 - \hat{p})}{\hat{p}}.
\]

## Fully Bayesian Denoising

The formulas above condition on point estimates of \((r_g, p, \nu_c, g)\).
In a fully Bayesian treatment, uncertainty is propagated by repeating the
denoising for each posterior sample:

\[
\pi(m_g \mid u_g, \text{data}) \approx
\frac{1}{S} \sum_{s=1}^{S}
\pi\!\left(m_g \mid u_g, r_g^{(s)}, p^{(s)}, \nu_c^{(s)}, g^{(s)}\right).
\]

This produces a full distribution over denoised counts that reflects both the
stochasticity of the capture process and the uncertainty in the model
parameters.

## Cross-Gene Correlations

A key insight is that **all cross-gene correlation in the denoised counts
arises from parameter uncertainty**. The law of total covariance gives:

\[
\text{Cov}(m_g, m_{g'} \mid \underline{u}) =
\text{Cov}_{\underline{\Theta}}\!\left[
    \langle m_g \mid u_g, \underline{\Theta} \rangle, \;
    \langle m_{g'} \mid u_{g'}, \underline{\Theta} \rangle
\right],
\quad g \neq g'.
\]

When parameters are fixed at their MAP estimates, denoised counts for
different genes are independent. Correlations appear only when accounting for
the joint posterior.

### Correlated denoising algorithm

In practice, cross-gene correlations are captured automatically by the
following procedure:

1. **Draw one joint parameter sample** from the posterior
2. **Denoise each gene independently** using the per-gene formulas with the
   drawn parameters
3. **Assemble** the denoised count matrix

The shared parameter draw in step 1 propagates cross-gene correlations
automatically. The quality of these correlations depends on the inference
method:

| Inference method | Cross-gene correlations |
|------------------|------------------------|
| **MCMC** | Exact (each draw is a full joint sample) |
| **SVI with joint low-rank guide** | Captured via low-rank variational family |
| **SVI with factorized guide** | Only through shared global parameters (\(p\), \(\nu_c\)) |
| **MAP point estimates** | None (genes are independent) |

!!! tip "Preserving correlations in practice"
    When exporting denoised datasets, set `preserve_correlations=True` to
    ensure all denoised matrices use individual posterior draws rather than
    the posterior mean. This preserves cross-gene correlations for downstream
    analysis.

## Using Denoising in SCRIBE

The denoising theory is implemented in the results objects returned by
`scribe.fit()`. Both SVI and MCMC results provide the same high-level API.

### The `method` parameter

The `method` parameter controls how the denoised posterior is summarized into
a point estimate for each cell-gene entry:

| Method | Description |
|--------|-------------|
| `"mean"` | Closed-form posterior mean -- the shrinkage estimator derived above. Deterministic (no RNG needed for non-ZINB models). |
| `"mode"` | Posterior mode (MAP denoised count). The integer floor of the continuous mode formula. Always less than or equal to the mean. |
| `"sample"` | One stochastic draw from the full denoised posterior distribution. Preserves the shape of the distribution, including its skewness and discreteness. |

For zero-inflated models, zeros require special treatment because they arise
from a mixture of two pathways (gate vs NB). The `method` parameter accepts
a **tuple** `(general_method, zi_zero_method)` for independent control:

- `general_method` -- applied to all non-zero positions and to all positions
  in non-ZINB models
- `zi_zero_method` -- applied exclusively to zero positions in ZINB models,
  where the gate/NB mixture posterior must be resolved

A single string `"mean"` is shorthand for `("mean", "mean")`.

!!! tip "Recommended: `method=("sample", "sample")`"
    For most downstream analyses, **sampling** is the preferred method. It
    generates realistic count matrices that faithfully represent the full
    denoised posterior, including the discrete, skewed nature of transcript
    counts. For ZINB zeros, `"sample"` uses the gate weight \(w\) as a
    Bernoulli probability: dropout zeros are replaced with draws from the
    biological prior \(\text{NB}(r_g, p)\), while genuine NB zeros are kept
    at zero. This produces the most faithful representation of what the true
    transcriptome likely looked like.

    By contrast, `"mean"` produces smooth, non-integer values that
    over-regularize the distribution, and `"mode"` tends to underestimate
    counts (especially for low-expression genes where the mode is zero).

### MAP-based denoising (fast, no cross-gene correlations)

Uses the MAP point estimates of the parameters. Fast but loses cross-gene
correlations:

```python
import jax

rng_key = jax.random.PRNGKey(0)

# SVI results
denoised = results.denoise_counts_map(
    counts=adata.X,
    method=("sample", "sample"),
    rng_key=rng_key,
)

# MCMC results
denoised = results.denoise_counts(
    counts=adata.X,
    method=("sample", "sample"),
    rng_key=rng_key,
)
```

### Full-posterior denoising (recommended, with cross-gene correlations)

Uses multiple posterior draws, automatically propagating cross-gene
correlations through the shared parameter samples:

```python
# SVI results -- uses multiple posterior draws
denoised = results.denoise_counts_posterior(
    counts=adata.X,
    method=("sample", "sample"),
    rng_key=rng_key,
)
```

### Biological posterior predictive checks

Biological PPCs strip technical noise (capture probability, zero-inflation
gate) and sample from the base \(\text{NB}(r, p)\), giving a denoised view
of the data:

```python
bio_ppc = results.get_ppc_samples_biological(
    rng_key=rng_key,
    n_samples=100,
)
```

### Exporting denoised counts as AnnData

```python
# Single denoised dataset
adata_denoised = results.get_denoised_anndata(
    counts=adata.X,
    rng_key=rng_key,
    method=("sample", "sample"),
)

# Multiple datasets preserving cross-gene correlations
adatas = results.get_denoised_anndata(
    counts=adata.X,
    rng_key=rng_key,
    n_datasets=5,
    preserve_correlations=True,
    method=("sample", "sample"),
    path="denoised.h5ad",
)
```

For the full API details, see the [Results Class](../guide/results.md) guide
and the [API Reference](../reference/scribe/svi/) for SVI or
[API Reference](../reference/scribe/mcmc/) for MCMC.

---

!!! tip "Next steps"
    - See the [Dirichlet-Multinomial Model](dirichlet-multinomial.md) for the
      NB generative model and capture-probability derivation that underpin the
      denoising result.
    - See [Beta Negative Binomial](beta-negative-binomial.md) for how
      denoising extends to power-law tail distributions via a one-dimensional
      integral over the latent Beta mixing variable.
    - See the [Model Selection](../guide/model-selection.md#zero-inflation-zinb)
      guide for practical details on the zero-inflated models.
    - See the [Results Class guide](../guide/results.md) for the full denoising
      API, including `denoise_counts`, `denoise_counts_posterior`, and
      `get_denoised_anndata`.

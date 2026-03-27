# Hierarchical Priors

SCRIBE uses **hierarchical priors** as a unifying mechanism for sharing
statistical strength across genes, mixture components, and datasets.
Rather than treating each element independently, a hierarchical prior
draws individual parameters from a shared population distribution whose
hyperparameters are themselves learned from the data. This produces
**adaptive shrinkage**: elements with limited data are regularized toward
the population center, while elements with abundant data are free to
deviate.

The same three building-block prior families---Gaussian, Horseshoe, and
Normal-Exponential-Gamma (NEG)---can be applied to different model
parameters at different levels of the hierarchy. This page introduces the
families, shows how they apply to the gene-specific success probability
\(p_g\) (see also the [dedicated derivation](hierarchical-p.md)), the
mean expression \(\mu_g\), and the zero-inflation gate \(\pi_g\), and
then extends the framework to multi-dataset models.

---

## The Three Prior Families

All three families operate in an **unconstrained space** (logit for
parameters bounded in \((0,1)\), log for positive parameters). They
share a common template: a population center plus a per-element scale
that controls how far each element can deviate.

### Gaussian (Normal hierarchy)

The simplest hierarchy assigns a single global scale to all elements:

\[
\mu_{\text{pop}} \;\sim\; \pi(\mu_{\text{pop}}), \qquad
\sigma \;\sim\; \text{Softplus}(\mathcal{N}(\mu_\sigma, s_\sigma)),
\]

\[
\theta_g \;\sim\; \mathcal{N}(\mu_{\text{pop}},\; \sigma),
\quad g = 1, \ldots, G.
\]

Every element shares the same scale \(\sigma\), so all elements
experience the same degree of shrinkage toward the population mean. This
is effective when the signal is relatively homogeneous, but suffers from
the **scale contamination problem**: if even a small fraction of elements
have genuinely different values, the posterior of \(\sigma\) inflates to
accommodate them, loosening the constraint on *all* elements---including
those that truly are identical.

### Horseshoe (Regularized)

The horseshoe prior addresses scale contamination by giving each element
its own **local scale** \(\lambda_g\):

\[
\tau \;\sim\; \text{Half-Cauchy}(\tau_0), \qquad
\lambda_g \;\sim\; \text{Half-Cauchy}(1), \qquad
c^2 \;\sim\; \text{Inv-Gamma}\!\left(\tfrac{\nu}{2},\;
\tfrac{\nu s^2}{2}\right),
\]

\[
\tilde{\lambda}_g = \frac{c\,\lambda_g}{\sqrt{c^2 + \tau^2
\lambda_g^2}}, \qquad
\theta_g \;\sim\; \mathcal{N}(\mu_{\text{pop}},\;
\tau \cdot \tilde{\lambda}_g).
\]

Most \(\lambda_g\) values cluster near zero (strong shrinkage), while a
few can be very large (escape shrinkage). The global \(\tau\) controls
the overall fraction of elements that are shrunk, and the slab
\(c^2\) bounds the maximum effective scale for numerical stability. The
regularized local scale \(\tilde{\lambda}_g\) behaves like the standard
horseshoe when \(\tau \lambda_g \ll c\) and saturates at \(c/\tau\) for
large signals.

**The challenge under SVI.** The horseshoe density has an **infinite
spike** (a pole) at zero, corresponding to Half-Cauchy local scales.
This creates a difficult gradient landscape: under mean-field or
low-rank variational families, the infinite pole often leads to
exploding gradients or variational collapse.

### Normal-Exponential-Gamma (NEG) -- recommended for SVI

The NEG prior is a member of the **Three Parameter Beta Normal (TPBN)**
mixture family, which also includes the horseshoe. The key difference is
a single index change that replaces the infinite spike with a finite
peak.

The TPBN family admits a **Gamma-Gamma hierarchical representation**:

\[
\zeta_g \;\sim\; \text{Gamma}(a,\; \tau), \qquad
\psi_g \mid \zeta_g \;\sim\; \text{Gamma}(u,\; \zeta_g),
\]

\[
\theta_g \;\sim\; \mathcal{N}(\mu_{\text{pop}},\; \sqrt{\psi_g}).
\]

Different choices of \(u\) and \(a\) recover different priors:

| Prior | \(u\) | \(a\) | Behavior |
|-------|-------|-------|----------|
| Horseshoe | 1/2 | 1/2 | Infinite spike at zero; heaviest tails |
| Strawderman-Berger | 1/2 | 1 | Infinite spike; lighter tails |
| **NEG** | **1** | \(a > 0\) | **Finite** peak; heavy tails |

Setting \(u = 1\) makes the inner Gamma an **Exponential** distribution.
This single change replaces the infinite pole at zero with a finite peak:
the density still concentrates aggressively near zero to shrink
irrelevant elements, but the gradient landscape is smooth and
well-behaved.

**Why the NEG is SVI-friendly.** Every latent variable in the Gamma-Gamma
hierarchy is either Gamma-distributed (with efficient implicit
reparameterization gradients) or Normal (trivially reparameterizable
under NCP). By contrast, the Half-Cauchy local scales in the horseshoe
lack efficient reparameterization.

**Non-centered parameterization (NCP).** In practice, all three families
use a non-centered form to avoid the funnel geometry that arises when the
scale is small:

\[
z_g \;\sim\; \mathcal{N}(0, 1), \qquad
\theta_g = \mu_{\text{pop}} + \sigma_g \cdot z_g,
\]

where \(\sigma_g\) is the family-specific scale (\(\sigma\) for
Gaussian, \(\tau \tilde{\lambda}_g\) for horseshoe, \(\sqrt{\psi_g}\)
for NEG). The variational guide always targets a unit normal for
\(z_g\)---a well-scaled problem regardless of the effective scale.

---

## Applying Hierarchical Priors to Model Parameters

### Gene-specific *p* (success probability)

The foundational application is the gene-specific success probability
\(p_g\), derived in detail on the
[Hierarchical Gene-Specific p](hierarchical-p.md) page. In brief:

\[
\text{logit}(p_g) \;\sim\; \mathcal{N}(\mu_p,\; \sigma_p),
\quad g = 1, \ldots, G,
\]

where \(\mu_p\) and \(\sigma_p\) are learned hyperparameters. The
posterior of \(\sigma_p\) serves as a diagnostic: when it is near zero,
the data are consistent with a shared \(p\), and the model effectively
recovers the standard NBDM; when it is large, gene-to-gene variation is
supported.

Any of the three prior families can replace the Gaussian hierarchy above.
With the horseshoe or NEG, each gene receives a local scale that allows
a few genes to deviate strongly while the majority are shrunk to the
population mean.

```python
# Gene-specific p with NEG prior
results = scribe.fit(
    adata,
    prob_prior="neg",
)
```

### Gene-specific mu (across mixture components)

For mixture models with \(K \geq 2\) components, SCRIBE can place a
hierarchical prior on the mean expression \(\mu_g\) (or dispersion
\(r_g\) in the canonical parameterization) across components. Each gene
has its own population center and the per-component means are drawn from
it:

\[
\mu_g^{\text{pop}} \;\sim\; \pi(\mu_g), \qquad
\log(\mu_g^{(k)}) \;\sim\; \mathcal{N}(\log(\mu_g^{\text{pop}}),\;
\sigma_g),
\quad k = 1, \ldots, K,
\]

where \(\sigma_g\) is the family-specific scale (Gaussian, horseshoe, or
NEG). This encodes the biological prior that **most genes have similar
expression across cell types**, with only a subset deviating. Each gene
has its own hyperprior because expression magnitudes vary by orders of
magnitude across genes.

```python
# Mixture model with hierarchical mu (NEG) and p (Gaussian)
results = scribe.fit(
    adata,
    zero_inflation=True,
    n_components=3,
    unconstrained=True,
    expression_prior="neg",
    prob_prior="gaussian",
)
```

| Parameter | Description |
|-----------|-------------|
| `expression_prior` | Prior family for mu across components: `"gaussian"`, `"horseshoe"`, `"neg"` |
| Requires | `n_components >= 2`, `unconstrained=True` |

### Hierarchical zero-inflation gate

Zero-inflated models (ZINB, ZINBVCP) introduce a per-gene gate
\(\pi_g \in (0,1)\) that mixes a point mass at zero with the count
distribution. In practice, model comparison reveals a characteristic
pattern: the NB model is preferred for the vast majority of genes, but a
small subset strongly favors zero inflation. A flat gate prior either
"distracts" the majority of genes or starves the few that genuinely need
it.

#### The hierarchical gate model

The hierarchical gate resolves this dilemma with the same logit-normal
structure, but with a **"default off" center**:

\[
\mu_\pi \;\sim\; \mathcal{N}(-5,\; 1), \qquad
\sigma_\pi \;\sim\; \text{Softplus}(\mathcal{N}(-2,\; 0.5)),
\]

\[
\text{logit}(\pi_g) \;\sim\; \mathcal{N}(\mu_\pi,\; \sigma_\pi),
\quad g = 1, \ldots, G.
\]

The population mean \(\mu_\pi = -5\) places the typical gate at
\(\sigma(-5) \approx 0.7\%\): in the absence of evidence, genes are
strongly regularized toward the NB model. Even one standard deviation
above the mean (\(\mu_\pi = -4\)) yields a gate of only 1.8%.

#### Two shrinkage regimes

- **When \(\sigma_\pi\) is near zero:** all gates collapse to
  \(\sigma(\mu_\pi) \approx 0\), and the zero-inflated model
  **automatically recovers the standard NB model**.
- **When \(\sigma_\pi\) is large:** individual genes can deviate
  substantially. Genes with many structural zeros (e.g., cell-type
  markers) escape the prior; genes without structural zeros remain near
  zero.

The posterior of \(\sigma_\pi\) serves as a **data-driven diagnostic**:
it tells you whether gene-to-gene variation in zero-inflation probability
is supported by the data, eliminating the need for a separate NB vs ZINB
model comparison step.

#### Gate and composition sampling

An important practical point: the gate **does not enter the standard
composition formula**. Compositions represent the fractional abundance
of each gene conditional on it being expressed. The gate modifies the
observed count distribution but not the relative expression levels. For
differential expression, the composition sampling procedure from the
[Hierarchical p page](hierarchical-p.md#composition-sampling-with-gene-specific-p_g)
uses only the \(r_g\) and \(p_g\) parameters.

```python
# ZINB with hierarchical gate (NEG) -- automatic NB recovery if ZI not needed
results = scribe.fit(
    adata,
    zero_inflation=True,
    zero_inflation_prior="neg",
)
```

### BNB overdispersion (kappa_g)

The [Beta Negative Binomial](beta-negative-binomial.md) extension adds a
per-gene concentration parameter \(\kappa_g\) that allows heavier-than-NB
tails. Since most genes are adequately described by the NB, the
hierarchical prior on \(\kappa_g\) must default to NB behaviour while
allowing a sparse subset of genes to escape.

The concentration is reparameterized as the excess dispersion fraction
\(\omega_g = (r_g + 1) / (\kappa_g - 2)\), and the horseshoe or NEG
prior is applied to \(\omega_g\) so that most genes are shrunk toward
\(\omega_g = 0\) (the NB limit). See the
[BNB page](beta-negative-binomial.md#hierarchical-prior-on-kappa_g) for
the full derivation.

```python
# BNB with horseshoe prior on kappa_g
results = scribe.fit(
    adata,
    parameterization="canonical",
    overdispersion="bnb",
    overdispersion_prior="horseshoe",   # or "neg"
)
```

| Parameter | Description |
|-----------|-------------|
| `overdispersion` | `"none"` (default) or `"bnb"` to enable the BNB |
| `overdispersion_prior` | Prior on \(\kappa_g\): `"horseshoe"` (default) or `"neg"` |

---

## Extension to Multiple Datasets

### The parameter degeneracy problem

When fitting separate models to multiple datasets (e.g., same cell line
processed with different library kits), the model has no way to
distinguish "Kit 2 captures more molecules" from "Kit 2 cells express
more." The gene-specific parameters can absorb systematic technical
differences, producing different biological parameter estimates despite
identical underlying biology.

More precisely, the expected count
\(\langle u_{gc} \rangle = \mu_g \cdot (1+\phi)^{-1} \cdot \nu_c\) is a
product of multiple terms. The data constrain only the product, not the
individual factors. This degeneracy propagates to denoised counts and
differential expression, partially or fully preserving the batch effect.

### The joint model solution

The solution is to fit all datasets **jointly** in a single model, with a
hierarchical prior linking dataset-specific parameters to a shared
population level. Biological parameters are constrained to be shared (or
nearly shared) across datasets, forcing systematic differences into the
technical parameters where they belong.

### Dataset-specific hierarchical parameters

#### Hierarchical mu across datasets

For each gene, the dataset-specific mean is drawn from a population
distribution centered on the shared biological mean:

\[
\log(\mu_g^{(d)}) \;\sim\; \mathcal{N}(\log(\mu_g),\; \tau_\mu),
\quad d = 1, \ldots, D,
\]

where \(\tau_\mu\) governs the magnitude of dataset-to-dataset variation
in log space. The NCP form samples
\(\epsilon_{g,d}\) from \(\mathcal{N}(0,1)\) and computes
\(\log(\mu_g^{(d)}) = \log(\mu_g) + \tau_\mu \cdot \epsilon_{g,d}\).

When \(\tau_\mu\) is near zero, all dataset-specific means collapse to
the shared value and the model reduces to a single-dataset fit. When
\(\tau_\mu\) is large, datasets are essentially independent.

The Gaussian hierarchy uses a single \(\tau_\mu\) for all genes. If a
small fraction of genes have genuinely different expression across
datasets, this inflates \(\tau_\mu\) for all genes (the scale
contamination problem). The **horseshoe** and **NEG** variants replace
the single scale with per-gene local scales, protecting the majority from
contamination.

#### Dataset-specific *p*

A scalar dataset-specific \(p^{(d)}\) accounts for systematic technical
differences between datasets (e.g., different kits producing different
dropout rates), while remaining shared across genes within each dataset:

\[
\text{logit}(p^{(d)}) \;\sim\; \mathcal{N}(\mu_p,\; \sigma_p),
\quad d = 1, \ldots, D.
\]

For finer control, SCRIBE also supports **gene-specific** \(p_g^{(d)}\)
in two modes:

- **Single-level:** all (gene, dataset) pairs share one
  \((\mu_p, \sigma_p)\).
- **Two-level:** each dataset draws its own \((\mu_p^{(d)},
  \sigma_p^{(d)})\) from a population distribution, then each gene draws
  from its dataset's distribution.

#### Dataset-specific gate

For zero-inflated models, each (dataset, gene) pair gets its own gate,
independently drawn from the shared population distribution. Gates are
**not pooled** across datasets---each dataset's likelihood independently
updates each gene's gate. The "default off" property still applies: most
gates are near zero unless the data demand otherwise.

### Shrinkage diagnostics

The posterior of each scale parameter serves as a diagnostic:

| Diagnostic | Near zero | Large |
|------------|-----------|-------|
| \(\tau_\mu\) | Identical biology across datasets | Genuine biological differences |
| \(\sigma_p\) | Same technical characteristics | Different kit/sequencing profiles |
| \(\sigma_\pi\) | No gene needs ZI in any dataset | Gene-specific ZI heterogeneity |

### Paired posterior samples for cross-dataset DE

A key property of the joint model is that each posterior sample provides
parameters for **all datasets simultaneously**. Parameters at the same
sample index are correlated through the shared population-level
variables. This enables **paired** CLR-difference differential expression
between datasets, where the comparison accounts for the posterior
correlation introduced by the shared hierarchy.

---

## Using Hierarchical Priors in SCRIBE

### Gene-level hierarchical priors (single dataset)

```python
import scribe

# ZINB mixture model with hierarchical priors on mu, p, and gate
results = scribe.fit(
    adata,
    zero_inflation=True,
    n_components=3,
    unconstrained=True,
    expression_prior="neg",        # across mixture components
    prob_prior="gaussian",         # gene-specific p
    zero_inflation_prior="neg",    # gene-specific ZI gate
)
```

### Multi-dataset hierarchical model

```python
# Joint model across datasets with hierarchical linking
results = scribe.fit(
    adata,
    variable_capture=True,
    zero_inflation=True,
    dataset_key="batch",
    priors={"organism": "human"},
    expression_dataset_prior="horseshoe",       # dataset-specific mu
    prob_dataset_prior="gaussian",              # dataset-specific p (gene-specific mode)
    zero_inflation_dataset_prior="neg",         # dataset-specific gate
)
```

### Parameter reference

| Parameter | Level | Accepted values | Requires |
|-----------|-------|-----------------|----------|
| `expression_prior` | Gene (across components) | `"gaussian"`, `"horseshoe"`, `"neg"` | `n_components >= 2`, `unconstrained=True` |
| `prob_prior` | Gene | `"gaussian"`, `"horseshoe"`, `"neg"` | -- |
| `zero_inflation_prior` | Gene | `"gaussian"`, `"horseshoe"`, `"neg"` | ZI model (zinb/zinbvcp) |
| `expression_dataset_prior` | Dataset | `"gaussian"`, `"horseshoe"`, `"neg"` | `dataset_key` |
| `prob_dataset_prior` | Dataset | `"gaussian"`, `"horseshoe"`, `"neg"` | `dataset_key` |
| `prob_dataset_mode` | -- | `"scalar"`, `"gene_specific"`, `"two_level"` | `prob_dataset_prior` set |
| `zero_inflation_dataset_prior` | Dataset | `"gaussian"`, `"horseshoe"`, `"neg"` | `dataset_key`, ZI model |
| `overdispersion` | -- | `"none"`, `"bnb"` | -- |
| `overdispersion_prior` | Gene | `"horseshoe"`, `"neg"` | `overdispersion="bnb"` |
| `overdispersion_dataset_prior` | Dataset | `"gaussian"`, `"horseshoe"`, `"neg"` | `overdispersion="bnb"`, `dataset_key` |

### Horseshoe and NEG hyperparameters

These are shared across all parameters that use the corresponding prior:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `horseshoe_tau0` | 1.0 | Global shrinkage scale |
| `horseshoe_slab_df` | 4 | Slab tail weight (degrees of freedom) |
| `horseshoe_slab_scale` | 2.0 | Slab location scale |
| `neg_u` | 1.0 | Inner Gamma shape (1 = NEG, 0.5 = horseshoe) |
| `neg_a` | 1.0 | Outer Gamma shape (controls concentration near zero) |
| `neg_tau` | 1.0 | Global rate for the outer Gamma |

!!! tip "Choosing a prior family"
    - **Start with `"gaussian"`** for well-behaved problems with few
      outliers (e.g., technical replicates).
    - **Use `"neg"`** (recommended default) when you expect most elements
      to be similar but a few to differ strongly---the typical case for
      gene-specific parameters. NEG provides horseshoe-like shrinkage with
      stable SVI optimization.
    - **Use `"horseshoe"`** when running MCMC, or when you specifically
      need the stronger pole at zero.

For the full API details, see the [API Reference](../reference/scribe/api/).

---

!!! tip "Next steps"
    - See [Hierarchical Gene-Specific \(p\)](hierarchical-p.md) for the
      foundational derivation of gene-specific success probabilities, the
      primary application of these prior families.
    - See [Anchoring Priors](anchoring-priors.md) for the complementary
      regularization layers — anchors constrain the mean and capture
      probability while hierarchical priors shrink the overdispersion.
    - See [Beta Negative Binomial](beta-negative-binomial.md) for how the
      Horseshoe and NEG priors are applied to the BNB concentration parameter
      \(\kappa_g\) to enforce sparsity in extra overdispersion.
    - See [Differential Expression](differential-expression.md) to see how
      joint multi-dataset hierarchical models enable paired CLR-difference DE
      that accounts for posterior correlation across datasets.
    - See the [Inference Methods guide](../guide/inference.md) for practical
      guidance on choosing SVI vs MCMC, directly relevant to the Horseshoe vs
      NEG recommendation.

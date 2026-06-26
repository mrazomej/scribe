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
    unconstrained=True,
    priors={"probability": "neg"},
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
    priors={
        "mean_expression": "neg",
        "probability": "gaussian",
    },
)
```

| `priors` key | Description |
|--------------|-------------|
| `mean_expression` | Prior family for mu across components: `"gaussian"`, `"horseshoe"`, `"neg"` |
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
    unconstrained=True,
    priors={"zero_inflation": "neg"},
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
    priors={"overdispersion": "horseshoe"},   # or "neg"
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

### Crossed and nested designs: multiple grouping factors

A single `dataset_key` flattens cells onto **one** grouping axis. Real
experiments are often *crossed*: the same set of donors is measured under
several conditions, so a cell carries two labels at once (e.g. donor **and**
treatment). Encoding that as a single axis — one leaf per present
(donor, condition) combination — loses the shared structure: the treatment
effect is no longer tied across donors, and donor variation cannot be told
apart from the treatment.

SCRIBE generalises the dataset hierarchy to an **arbitrary number of grouping
factors** by giving the (log) mean an **additive decomposition** over the
factors, evaluated on the flat *leaf* axis (the present combinations):

\[
\log \mu_g^{(\ell)} \;=\; \log \mu_g^{\mathrm{pop}}
\;+\; \sum_{f} \alpha_g^{(f)}\!\big[\,\mathrm{level}_f(\ell)\,\big],
\]

where \(\ell\) indexes leaves, \(f\) ranges over the grouping factors, and
\(\mathrm{level}_f(\ell)\) is the level of factor \(f\) at leaf \(\ell\). The
population intercept \(\log\mu_g^{\mathrm{pop}}\) is the only free baseline;
each factor contributes a per-level effect gathered onto the leaf. With one
factor whose levels *are* the leaves this reduces **exactly** to the
single-axis hierarchy above, so nothing changes for existing single-`dataset_key`
fits. Only the **expression** target (\(\mu\)/\(r\)) receives this additive
form; the technical parameters (\(p\), gate, regime) keep the single-axis
per-leaf hierarchy, leaving the per-cell likelihood unchanged.

#### Fixed vs. random effects

Each factor's effect type is chosen independently of its prior *family*, and
the distinction matters:

- A **fixed** effect uses a weakly-informative zero-mean Gaussian with a
  **fixed** scale and **no learned (adaptive) shrinkage**:
  \(\alpha_{g,k}^{(f)} = \sigma_f\, z_{g,k}^{(f)}\), \(z \sim \mathcal N(0,1)\),
  with \(\sigma_f\) a constant. Use it for the **contrast of interest** (e.g. a
  two-level treatment): a learned scale on a low-cardinality factor is weakly
  identified and would shrink the very effect you are trying to measure toward
  zero. The identified quantity is the **contrast** between two levels, e.g.
  \(\alpha_g^{(f)}[\text{treated}] - \alpha_g^{(f)}[\text{control}]\).

- A **random** effect is a zero-mean NCP term with a **learned** scale and
  shrinkage (Gaussian / regularized-horseshoe / NEG), as in the single-axis
  case. Use it for grouping factors with enough levels to estimate their own
  spread (donors, batches) and for interactions. Each level is a deviation from
  the population mean.

Identifiability is soft: random effects are zero-mean, the population intercept
is the only free baseline, and main-effect **contrasts cancel** any residual
per-factor shift. Fixed effects are identified by their prior. Interactions are
supported as additional (random) factors, but are aliased with the main effects
without a hard sum-to-zero constraint, so interaction *main-effect contrasts*
are not separately reported.

#### Population (donor-averaged) differential expression

Because each posterior sample carries every leaf, a crossed model supports a
**paired, donor-averaged** treatment contrast: for each donor present in both
arms, form the within-donor CLR difference, then average over donors. This is
the population treatment effect with donor heterogeneity differenced out — see
[Differential Expression](differential-expression.md#part-iii-population-differential-expression-across-grouping-factors)
for the estimand, and the
[crossed-hierarchy tutorial](../tutorials/zhao_2021_hierarchical.md) for a worked
example. The full generative derivation lives in the paper's hierarchical-datasets
section.

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
    priors={
        "mean_expression": "neg",       # across mixture components
        "probability": "gaussian",      # gene-specific p
        "zero_inflation": "neg",        # gene-specific ZI gate
    },
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
    priors={
        "organism": "human",
        "mean_expression": {"batch": "horseshoe"},   # dataset-specific mu
        "probability": {"batch": "gaussian"},        # dataset-specific p
        "zero_inflation": {"batch": "neg"},          # dataset-specific gate
    },
)
```

### Crossed multi-factor model

Pass a **list** of grouping columns (crossing is implicit) or a structured
`hierarchy=[GroupLevel(...)]`, and give a canonical name a `{level: family}`
dict in `priors` to set a prior family *per factor*. Mark the contrast of
interest as a fixed effect:

```python
# Donors crossed with a two-level treatment.
results = scribe.fit(
    adata,
    variable_capture=True,
    unconstrained=True,
    parameterization="mean_odds",
    hierarchy=[
        scribe.GroupLevel("perturbation", effect_type="fixed"),  # 2-level contrast
        scribe.GroupLevel("sample"),                             # donors -> random
    ],
    priors={
        "mean_expression": {
            "perturbation": "gaussian",   # fixed-scale, weakly-informative contrast
            "sample": "horseshoe",        # adaptive shrinkage across donors
        },
        "probability": {"sample": "gaussian"},  # technical p: leaf-exchangeable
    },
)
```

`GroupLevel(name, nested_in=None, effect_type="random", fixed_scale=None)`
declares one factor; `nested_in` marks a nested (rather than crossed) factor,
and `effect_type="fixed"` selects the no-learned-shrinkage contrast. The
equivalent positional form is `dataset_key=["perturbation", "sample"]` (all
factors random with a broadcast prior). The crossed/multi-factor hierarchy is a
Python-API feature; the CLI supports a single `dataset_key`.

### Parameter reference

Prior families/hierarchies are entries in the `priors` dict: a **bare family
string** is a gene-level prior; a **`{level: family}` dict** is a
dataset/factor hierarchy (add a `"base"` key for the gene-level prior too).

| `priors` key      | Gene-level (`"family"`) | Dataset/factor (`{level: family}`) | Requires                        |
| ----------------- | ----------------------- | ---------------------------------- | ------------------------------- |
| `mean_expression` | mu across components    | mu across factors                  | gene-level: `n_components >= 2` |
| `probability`     | gene-specific p         | p across factors                   | --                              |
| `zero_inflation`  | gene-specific gate      | gate across factors                | ZI model (zinb/zinbvcp)         |
| `overdispersion`  | kappa_g (BNB)           | kappa across factors               | `overdispersion="bnb"`          |
| `regime`          | --- (hierarchy only)    | regime coord across factors        | TwoState model                  |

Families are `"gaussian"`, `"horseshoe"`, `"neg"` (or a `{"type": ...}` spec).
All hierarchical priors require `unconstrained=True`; the dataset/factor forms
also require `dataset_key`/`hierarchy`. The remaining **structural** options
stay as keyword arguments:

| Keyword argument                     | Values                                       | Requires                    |
| ------------------------------------ | -------------------------------------------- | --------------------------- |
| `overdispersion`                     | `"none"`, `"bnb"`                            | --                          |
| `prob_dataset_mode`                  | `"scalar"`, `"gene_specific"`, `"two_level"` | dataset-level `probability` |
| `regime_dataset_target`              | a two-state coordinate name                  | TwoState model              |
| `overdispersion_dataset_independent` | `True` / `False`                             | TwoState model              |

!!! note "Per-factor priors in crossed designs"
    With more than one grouping factor, give the canonical name a
    `{level: family}` dict to choose the prior family per factor (e.g.
    `priors={"mean_expression": {"perturbation": "gaussian", "sample":
    "horseshoe"}}`). Add a `"base"` key to set the gene-level prior at the same
    time. Factors marked `effect_type="fixed"` use a fixed-scale Gaussian
    regardless of family and learn no shrinkage scale.

### Horseshoe and NEG hyperparameters

To tune a family, pass a **family spec** (a dict carrying `"type"`) instead of
a bare string; the extra keys set the hyperparameters (e.g.
`priors={"probability": {"type": "horseshoe", "tau0": 0.5}}`):

| Spec key     | Default | Effect                                               |
| ------------ | ------- | ---------------------------------------------------- |
| `tau0`       | 1.0     | Global shrinkage scale (horseshoe)                   |
| `slab_df`    | 4       | Slab tail weight / degrees of freedom (horseshoe)    |
| `slab_scale` | 2.0     | Slab location scale (horseshoe)                      |
| `u`          | 1.0     | Inner Gamma shape (NEG; 1 = NEG, 0.5 = horseshoe)    |
| `a`          | 1.0     | Outer Gamma shape (NEG; concentration near zero)     |
| `tau`        | 1.0     | Global rate for the outer Gamma (NEG)                |

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

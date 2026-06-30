# NB Log-Normal Model

The Negative Binomial Log-Normal (NBLN) model is the heavier-tailed
sibling of the [Poisson Log-Normal](poisson-lognormal.md) model. It
shares the same [GRN-derived multivariate Gaussian](grn-biophysics.md)
on log-rates, but replaces PLN's Poisson observation channel with a
Negative Binomial — giving each gene an explicit dispersion parameter
\(r_g\) that captures bursty transcription beyond what the Gaussian
prior alone can produce.

---

## Motivation

PLN's Poisson observation channel implicitly assumes that, conditional
on the log-rate, gene counts are equi-dispersed (variance = mean).
This is exact for the *deterministic* mass-action limit of bursty
transcription, but real scRNA-seq data shows substantially
over-dispersed per-cell counts even after the log-normal prior absorbs
gene-to-gene rate variation:

1. **Bursty transcription persists.** The Gamma-Poisson mixture
   representation of bursty gene expression
   ([Beta Negative Binomial](beta-negative-binomial.md)) survives the
   GRN coupling: each gene's per-cell count is still NB-distributed
   given its log-rate, not Poisson.

2. **PLN underfits the low-count regime.** Genes with mean expression
   \(\sim 1\) UMI per cell show heavier-than-Poisson tails. PLN tries
   to absorb this through inflated \(\Sigma_{gg}\), which corrupts
   the cross-gene covariance structure.

3. **The NB is the right per-gene channel.** Adding a per-gene
   dispersion \(r_g\) decouples gene-intrinsic burst overdispersion
   from the cross-gene regulatory signal. The two get clean separate
   parameters.

The NBLN preserves PLN's log-concave posterior geometry, biology-
informed capture handling, and Laplace inference path while restoring
the Gamma-Poisson burst kinetics at the observation channel.

---

## Definition

The NBLN distribution is a hierarchical model for a \(G\)-dimensional
count vector:

\[
\underline{x} \sim \mathcal{N}(\underline{\mu},\,
\underline{\underline{\Sigma}}), \quad
u_g \mid x_g, r_g \sim
\mathrm{NegBinomial}_{\text{mean}}(e^{x_g},\, r_g),
\quad g = 1, \ldots, G,
\]

where
\(\mathrm{NegBinomial}_{\text{mean}}(\mu, r)\) is the NB
parameterization with mean \(\mu\) and concentration (dispersion)
\(r\) — equivalently, variance \(\mu + \mu^2 / r\). The latent
\(\underline{x} \in \mathbb{R}^G\) is the log-rate vector and
\(\underline{\underline{\Sigma}}\) encodes cross-gene regulatory
correlations.

The PLN limit is \(r_g \to \infty\), where the NB collapses to its
Poisson mean and the model recovers PLN exactly. The independent-gene
NB limit is \(\underline{\underline{\Sigma}} \to 0\), where each gene
becomes a Gamma-Poisson mixture with no cross-gene coupling.

### Moments

The NBLN has tractable closed-form moments derived from the law of
total variance:

| Quantity       | Expression                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Mean**       | \(\langle u_g \rangle = \exp(\mu_g + \tfrac{1}{2}\Sigma_{gg})\)                                                          |
| **Variance**   | PLN variance \(+\;\langle u_g \rangle^2 / r_g\) — extra term is the NB-over-Poisson burst contribution                  |
| **Covariance** | \(\text{Cov}(u_g, u_{g'}) = \exp(\mu_g + \mu_{g'} + \tfrac{1}{2}(\Sigma_{gg} + \Sigma_{g'g'}))[\exp(\Sigma_{gg'}) - 1]\) |

Cross-gene covariance is identical to PLN: the per-gene dispersion
\(r_g\) only inflates within-gene variance, not between-gene
covariance.

---

## Low-rank covariance

The covariance is parameterized the same way as PLN:

\[
\underline{\underline{\Sigma}} =
\underline{\underline{W}}\,\underline{\underline{W}}^\top
+ \text{diag}(\underline{d}),
\quad W \in \mathbb{R}^{G \times k},\;
d \in \mathbb{R}_{>0}^G.
\]

The latent dimension \(k\) (set via the `latent_dim` kwarg) is the
rank of the low-rank regulatory signal. With NBLN's Phase-3
[loadings-shrinkage priors](loadings-shrinkage.md), \(k\) can be set
generously and the prior selects the effective rank adaptively.

---

## Capture as a log-rate offset

NBLN inherits PLN's clean capture handling — capture enters as an
additive log-rate offset \(\eta^{(c)}\), with the same truncated-normal
biology-informed prior:

\[
u_g^{(c)} \mid x_g^{(c)}, \eta^{(c)}, r_g \sim
\mathrm{NegBinomial}_{\text{mean}}(e^{x_g^{(c)} - \eta^{(c)}},\, r_g).
\]

The capture model is structurally identical to PLN's, just routed
through the NB likelihood instead of the Poisson likelihood.

---

## Log-concave posterior + per-cell rigid-translation gauge

The per-cell conditional posterior on \(\underline{x}\) is strictly
log-concave (same proof structure as PLN — the NB log-likelihood is
concave in \(x_g\) at fixed \(r_g\), plus the Gaussian prior
contributes another strictly concave term). This guarantees a unique
MAP and validates the Laplace approximation.

NBLN has one structural complication PLN doesn't: a **per-cell
rigid-translation gauge**. Under the transformation
\((x_c, \eta_c) \to (x_c + \Delta_c \mathbf{1}, \eta_c + \Delta_c)\)
for an arbitrary scalar \(\Delta_c\) per cell, the NB likelihood is
exactly invariant (because \(x_g - \eta\) appears in the rate and the
shift cancels). This is a \(C\)-dimensional degeneracy (one per cell)
that's broken only by the priors on \(\underline{x}\) (MVN) and
\(\eta\) (TruncatedNormal). Compositional predictions and the
gauge-invariant projection \(\underline{\underline{W}}_\perp\) are
exactly invariant under this gauge; the absolute log-rates and the raw
loadings \(\underline{\underline{W}}\) are not.

**Practical consequence:** raw \(\underline{\underline{W}}\) carries
a rank-1 all-ones-direction contamination that reflects cell-scaling
slop rather than biology. For cross-gene correlation analysis, always
use the gauge-invariant projection
\(\underline{\underline{W}}_\perp = \underline{\underline{W}} -
\overline{\underline{\underline{W}}}\) (see
[Loadings Shrinkage](loadings-shrinkage.md) for the full theorem). The
`get_W_compositional()` and `get_gauge_diagnostics()` accessors expose
this projection and quantify how much gauge slop the raw fit carries.

---

## SVI-cascade + freeze for the gauge

Because the per-cell gauge is \(C\)-dimensional rather than 1D (as in
PLN), even the MVN+TruncN priors leave a near-flat ridge in the
posterior at default hyperparameters. SCRIBE addresses this with a
**two-phase informative-prior cascade** that pins the gauge
structurally:

### Phase 1: SVI cascade as soft prior

Fit an NBVCP-SVI source first, derive empirical Gaussian priors on
\(r_g\), \(\mu_g\), and \(\eta_c\) from the SVI posterior samples,
and inject them into the NBLN-Laplace loss. The priors' *scales* (not
just locations) propagate into the Hessian:

```python
import scribe, numpy as np

# Fit NBVCP-SVI (cascade source)
svi_results = scribe.fit(
    adata, model="nbvcp", parameterization="mean_odds",
    priors={"capture_efficiency": (np.log(100_000), 0.5)},
    inference_method="svi", n_steps=50_000,
)

# NBLN-Laplace with cascade as informative prior
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_tau=1.0,        # default
    n_steps=20_000,
)
```

### Phase 2: Freeze for structural gauge fix

Promote selected cascade-derived parameters from soft priors to
**hard freezes**: the NBLN M-step does not refine them. Default
freeze is \((\underline{r}, \underline{\eta})\), which pins the
per-cell rigid-translation gauge exactly:

```python
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_freeze=("r", "eta"),   # default
    n_steps=20_000,
)
```

The freeze excludes \(r_g\) and \(\eta_c\) from the optimizer's
parameter dict, so they cannot drift. Frozen parameters retain the
cascade's full SVI posterior (not a moment-matched Gaussian summary)
— `get_distributions()` and PPC paths route through
`result.cascade_source` to preserve full guide fidelity.

| Freeze level | Freeze set | When to use |
|---|---|---|
| Level 1 | `()` | No freeze. Soft cascade only. Use for diagnostic comparison. |
| Level 2 | `("r",)` | Freeze dispersion. Use when NBVCP's \(\eta\) is suspect. |
| **Level 3** | **`("r", "eta")`** | **Default.** Pins the gauge structurally. Recommended for production. |
| Level 4 | `("r", "mu", "eta")` | NBLN learns only \(\underline{\underline{W}}\) and \(\underline{d}\) on top of NBVCP. |

!!! tip "Descriptive aliases for freeze keys"
    `informative_priors_freeze` accepts either the internal short
    names shown above or their descriptive aliases:

    - `"r"` ↔ `"dispersion"`
    - `"mu"` ↔ `"expression"` or `"mean_expression"`
    - `"eta"` ↔ `"capture_efficiency"`

    So `("dispersion", "capture_efficiency")` is equivalent to
    `("r", "eta")`. Both forms work; the descriptive form matches
    the convention used in `priors={"capture_efficiency": ...}`.
    Passing both an internal name and its alias (e.g.
    `("r", "dispersion")`) raises `ValueError`.

---

## Loadings shrinkage for adaptive rank selection

Even with the Phase-2 freeze, at generous `latent_dim` (e.g. 32) the
loadings matrix \(\underline{\underline{W}}\) can still overfit by
filling unused factors with noise — manifested as **spurious
cross-block diagonal contours** in the
[compositional corner PPC](../guide/results.md). The Phase-3
**[loadings-shrinkage prior](loadings-shrinkage.md)** adds an adaptive
rank-selection prior on the columns of
\(\underline{\underline{W}}_\perp\), letting the data pick the
effective rank:

```python
laplace_results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    informative_priors_from=svi_results,
    informative_priors_freeze=("r", "eta"),
    priors={
        "capture_efficiency": (np.log(100_000), 0.5),
        "loadings": {"type": "horseshoe_columnwise", "tau_scale": 1.0},
    },
    latent_dim=16,           # generous; prior picks effective rank
    n_steps=20_000,
)
print(laplace_results.w_prior_diagnostics["effective_rank"])    # e.g. 3-5
```

See the [Loadings Shrinkage](loadings-shrinkage.md) theory page for
the strategy catalog (`gaussian`, `horseshoe_columnwise`,
`neg_columnwise`), calibration workflow, and math details.

---

## Hierarchical gene-gene correlation across datasets

Everything above describes a *single* global covariance
\(\underline{\underline{\Sigma}} = \underline{\underline{W}}\,
\underline{\underline{W}}^\top + \text{diag}(\underline{d})\) shared by
every cell. When the data span several **datasets** — donors, conditions,
or batches (the grouped setting of
[Hierarchical Priors → Multiple datasets](hierarchical-priors.md#extension-to-multiple-datasets))
— one covariance forces a single regulatory-correlation structure on all
of them. SCRIBE's multi-dataset machinery already links the *marginal*
parameters (\(\mu_g\), \(r_g\)) across datasets; the
`correlation_hierarchy="program_scales"` option adds the analogous
hierarchy for the *correlation* itself.

The biological premise is **common regulatory programs with
dataset-specific activity**: the columns of \(\underline{\underline{W}}\)
(the \(K\) regulatory modules) are the same set of programs in every
dataset, but each dataset engages them to a different degree. This is the
single-cell analogue of Flury's *common principal components* model — it
lets a panel of donors *inform one another's* correlation structure rather
than each being estimated in isolation, which a single-donor fit can never
do.

### The model

Let \(\sigma(c) \in \{1, \ldots, D\}\) be the (observed, fixed) dataset of
cell \(c\). Each dataset \(d\) carries a vector of **relative program
activities** \(\underline{s}_d \in \mathbb{R}_{>0}^K\), and the per-cell
latent regulatory state is drawn from a dataset-specific covariance:

\[
\underline{z}_c \sim
\mathcal{N}(\underline{0},\ \underline{\underline{\Sigma}}_{\sigma(c)}),
\qquad
\underline{\underline{\Sigma}}_d =
\underline{\underline{W}}\,\text{diag}(\underline{s}_d^{\,2})\,
\underline{\underline{W}}^\top + \text{diag}(\underline{d}).
\]

The loadings \(\underline{\underline{W}}\), the residual diagonal
\(\underline{d}\), and the dispersion \(\underline{r}\) (empirically stable
across conditions) are **shared**; only \(\underline{s}_d\) varies. Setting
\(\underline{s}_d = \underline{1}\) for all \(d\) recovers the single
global \(\underline{\underline{\Sigma}}\) exactly, so this is a strict
generalization. A program with \(s_{d,k} > 1\) shows *more* cell-to-cell
covariation along module \(k\) in dataset \(d\) than the population;
\(s_{d,k} < 1\), less.

### Hierarchical prior and the sum-to-zero gauge

A non-centered hierarchical prior on the log-activities mirrors the
multi-dataset \(\mu\) construction, but on the \(D \times K\) activity
grid:

\[
\log s_{d,k} = \tau_s\,\tilde\varepsilon_{d,k},
\qquad
\tilde\varepsilon_{\cdot,k} = \varepsilon_{\cdot,k}
   - \frac{1}{D}\sum_{d'} \varepsilon_{d',k},
\qquad
\varepsilon_{d,k} \sim \mathcal{N}(0,1),
\]

with a single shared scale \(\tau_s \sim \text{Softplus}(\mathcal{N})\).
Centering \(\varepsilon_{\cdot,k}\) enforces a **sum-to-zero constraint**
\(\frac{1}{D}\sum_d \log s_{d,k} = 0\) (equivalently
\(\prod_d s_{d,k} = 1\)) for every program: each program's activities have
geometric mean 1, so \(\underline{s}_d\) is purely *relative* activity and
the absolute magnitude of program \(k\) lives in the column norm
\(\lVert \underline{\underline{W}}_{:,k} \rVert\). This is not cosmetic —
it is what makes the parameterization identifiable.

### Identifiability: two gauges, both fixed

The single-dataset \(\underline{\underline{W}}\) is identified only up to
an orthogonal rotation. The hierarchical model has *two* gauge freedoms,
and the datasets plus the sum-to-zero constraint fix both:

- **Rotation gauge.** \(\underline{\underline{\Sigma}}_d\) is
  rotation-invariant (\(\underline{\underline{W}} \mapsto
  \underline{\underline{W}}\,\underline{\underline{R}}\)) only if
  \(\underline{\underline{R}}\) commutes with
  \(\text{diag}(\underline{s}_d^{\,2})\) *for every* \(d\) simultaneously.
  When the datasets' activity profiles are sufficiently distinct, the only
  such \(\underline{\underline{R}}\) is a signed permutation — so **dataset
  heterogeneity generically breaks the rotation gauge** (the Flury
  common-principal-components result). The very non-identifiability that
  afflicts a single dataset is resolved by having several.
- **Scale gauge.** Independently, the per-column rescaling
  \(\underline{\underline{W}}_{:,k} \mapsto a_k
  \underline{\underline{W}}_{:,k},\ s_{d,k} \mapsto s_{d,k}/a_k\) leaves
  \(\underline{\underline{\Sigma}}_d\) unchanged. Dataset heterogeneity
  does **not** break this — but the sum-to-zero constraint does: it forces
  \(a_k = 1\), pinning column magnitude to \(\underline{\underline{W}}\).

Column **sign and permutation** always remain (resolved at interpretation
time, exactly as for any \(\underline{\underline{W}}\)), and the per-cell
rigid-translation gauge is unchanged by the hierarchy.

### The effective-loadings collapse: no extra inference cost

The hierarchy adds no new linear algebra. Define the **effective
loadings** of dataset \(d\), \(\underline{\underline{W}}_{\text{eff},d} =
\underline{\underline{W}}\,\text{diag}(\underline{s}_d)\). Then

\[
\underline{\underline{\Sigma}}_d =
\underline{\underline{W}}_{\text{eff},d}\,
\underline{\underline{W}}_{\text{eff},d}^\top + \text{diag}(\underline{d})
\]

is *exactly* the low-rank-plus-diagonal form, with
\(\underline{\underline{W}}_{\text{eff},d}\) in place of
\(\underline{\underline{W}}\) and the same \(\underline{d}\). Every
single-dataset result — the Woodbury inverse, the log-determinant, the
per-cell Newton/Laplace solve — carries over verbatim, with each cell using
its dataset's \(\underline{\underline{W}}_{\text{eff},\sigma(c)}\).
Internally this is a per-cell loadings gather plus a `vmap` over the
existing kernels; the shared-\(\underline{\underline{W}}\) path is
bit-identical when the hierarchy is off. The activities \(\underline{s}_d\)
receive their gradient only through the per-cell prior quadratic form and
the log-determinant of \(\underline{\underline{\Sigma}}_{\sigma(c)}\).

### Usage

```python
import scribe

# Hierarchical correlation across donors (Laplace path)
results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    correlation_hierarchy="program_scales",   # shared W, per-donor s_d
    correlate_other_column=True,              # legacy layout (v1)
    dataset_key="donor",                       # or hierarchy=[GroupLevel("donor")]
    informative_priors_from=hier_svi_source,   # freeze marginals (recommended)
    informative_priors_freeze=("r",),          # pool dispersion across donors
    latent_dim=16, n_steps=50_000,
)
s   = results.get_program_activity()   # (D, K) relative per-donor activity s_d
tau = results.program_scale_tau        # scalar between-dataset scale τ_s
```

The same `correlation_hierarchy="program_scales"` flag works on the
SVI/VAE path (per-donor variance on the \(K\)-dim latent), useful as a fast
structure check. As always for low-count NBLN-Laplace, anchoring the
marginals from a well-identified upstream fit (the
[cascade + freeze](#svi-cascade-freeze-for-the-gauge) workflow) is the
robust path on real data — and it **composes** with the correlation
hierarchy: freeze \(\underline{r}\) (and \(\underline{\eta}\)) from an
independent-gene fit while the \(\underline{s}_d\) hierarchy learns on top.

When the cascade source is itself a **hierarchical** (multi-dataset) fit,
the freeze can also pin a *per-donor* mean \(\mu^{(d)}\): add `"mu"` to
`informative_priors_freeze`. Each cell then takes its donor's prior mean
\(\mu^{(\sigma(c))}\) (a per-cell gather, the same mechanism as the program
scales), so the marginal and correlation hierarchies compose in one fit:

```python
results = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    correlation_hierarchy="program_scales",
    correlate_other_column=True, dataset_key="donor",
    informative_priors_from=hier_svi_source,   # a hierarchical (per-donor) SVI fit
    informative_priors_freeze=("r", "mu"),     # pool r, freeze per-donor mu^(d)
    latent_dim=16, n_steps=50_000,
)
mu_d = results.get_gene_mean_per_dataset()   # (D, G) per-donor log-rate means
mu   = results.get_mu()                      # (G,) donor-pooled mean
```

The extractor reads the source's per-donor mean (a hierarchical fit exposes
it directly), aligns the source leaves to the target leaf ordering *by
label*, and pools the dispersion \(\underline{r}\) to a shared value.
Per-donor \(\mu^{(d)}\) requires the correlation hierarchy (the per-cell
\(\underline{\underline{W}}\) Newton path).

!!! info "Relationship to the marginal hierarchy"
    The [multi-dataset hierarchy](hierarchical-priors.md#extension-to-multiple-datasets)
    links *first-order* structure — how much each gene's expression level
    \(\mu_g^{(d)}\) shifts between datasets. The correlation hierarchy links
    *second-order* structure — how much each regulatory module's activity
    \(s_{d,k}\) shifts. They are complementary and compose: a joint model
    can carry dataset-specific marginals (frozen via the per-donor
    \(\mu^{(d)}\) cascade above) *and* dataset-specific correlation over a
    shared \(\underline{\underline{W}}\).

!!! warning "Few-dataset regime"
    With a small number of datasets \(D\) (e.g. a 7-donor panel), the
    between-dataset scale \(\tau_s\) is weakly identified and the \(K\)
    activities per dataset cannot resolve fine differences in correlation
    eigenstructure. This is the expected partial-pooling regime: the
    hierarchy shares strength across datasets that share a regulatory
    architecture while letting their program usage differ — it does not
    manufacture between-donor structure the data do not support.

---

## Comparison with PLN and LNM

| Property              | LNM                                              | PLN                                | NBLN                                                     |
| --------------------- | ------------------------------------------------ | ---------------------------------- | -------------------------------------------------------- |
| Observation channel   | Multinomial composition                          | Poisson on rate                    | NB on rate (gene-specific dispersion)                    |
| Per-gene overdispersion | Implicit via composition                       | Equi-dispersed (Var = mean)        | Explicit via \(r_g\)                                     |
| Bursty transcription  | Absorbed into log-normal prior                   | Absorbed into log-normal prior     | **Native** via NB observation channel                    |
| Gauge dimension       | 0 (ALR-space W)                                  | 1 (global \(\mu\)-\(\eta\) shift)  | **\(C\)** (per-cell rigid translation)                   |
| Gauge resolution      | Parametric (ALR reference)                       | Capture prior + Gaussian prior     | Phase-2 cascade freeze on \(\eta\)                       |
| Total count           | Separate NB on \(u_T\)                           | Emergent from per-gene Poissons    | Emergent from per-gene NBs                               |
| Best for              | Direct compositional analysis                    | Heavy-tailed totals, capture clean | Bursty data, cross-gene correlation, cascade-freeze fits |

---

## Posterior predictive checks

NBLN supports three PPC conditioning regimes, plus a compositional
PPC family that specifically tests the gauge-invariant signal:

| Mode | What it tests | Conditioning |
|------|---------------|--------------|
| `ppc_level="marginal"` | Honest full generative test | Fresh \(\underline{x}\), \(\eta\), totals from the model |
| `ppc_level="library_anchored"` (default) | Compositional fit | Fresh composition, totals from data |
| `ppc_level="per_cell"` | Per-cell predictive | Per-cell MAP latents + observation noise |
| `plot_compositional_corner_ppc` | Gauge-invariant compositions | Pre-observation-noise simplex draws vs empirical + pseudobulk |

For cascade-frozen fits, PPC routes \(r_g\), \(\eta_c\) (and
optionally \(\mu_g\)) through `result.cascade_source` to preserve the
full SVI posterior structure — no moment-matching.

---

## When to use NBLN

| Use NBLN when…                                                                                      | Use PLN when…                                                  |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| You have a working NBVCP-SVI fit and want a cascade-frozen refinement                               | No NBVCP fit available; PLN as a standalone Laplace fit         |
| Low-count regime where Poisson under-fits the per-gene tail                                        | Counts are well-modeled by Poisson on the rate                  |
| Cross-gene regulatory correlation recovery is the primary goal                                      | Compositional structure is the primary downstream interest     |
| You need both per-gene dispersion *and* cross-gene covariance with structural gauge identifiability | Posterior log-concavity is more important than gauge structure |

!!! tip "Practical recommendation"
    For modern cross-gene correlation analysis on bursty scRNA-seq
    data, the recommended pipeline is:

    1. Fit `model="nbvcp"` with SVI as the cascade source.
    2. Fit `model="nbln"` with `inference_method="laplace"`,
       `informative_priors_from=svi_results`, default freeze
       `("r", "eta")`, and `priors={"loadings": {"type":
       "horseshoe_columnwise"}}` for adaptive rank selection.
    3. Validate via `plot_ppc(ppc_level="marginal")` and
       `plot_compositional_corner_ppc`.
    4. Extract correlations via `get_W_compositional()` — the
       gauge-invariant projection.

    See [Loadings Shrinkage](loadings-shrinkage.md) and the [Model
    Selection](../guide/model-selection.md) guide for the decision
    tree.

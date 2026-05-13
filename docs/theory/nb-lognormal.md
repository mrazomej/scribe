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

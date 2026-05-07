# Poisson Log-Normal Model

The Poisson Log-Normal (PLN) model connects the
[multivariate Gaussian on log-abundances](grn-biophysics.md) to observed
scRNA-seq counts via direct Poisson emission---the most biophysically direct
observation model. Each gene's count is drawn independently from a Poisson
whose rate is the exponentiated log-abundance, without any intervening
total-composition factorization.

---

## Motivation

The [Logistic-Normal Multinomial](logistic-normal-multinomial.md) model factors
the count vector into a total count (NB) and a composition (Multinomial). While
elegant, this introduces assumptions that are not mandated by the biophysics:

1. **Left-biased PPCs for totals.** The NB total-count model underestimates tail
   weight: the true total (a sum of correlated log-normal Poissons) has heavier
   tails than any NB with matched moments.

2. **Total-composition independence.** Under the biophysics, total abundance
   and composition are coupled through the shared covariance
   \(\underline{\underline{\Sigma}}\). The LNM factorization discards this
   coupling.

3. **The NB is exact only for independent genes.** For interacting genes, the
   sum of correlated log-normal variables does not follow an NB distribution.

The PLN addresses all three issues by modeling the count vector directly.

---

## Definition

The PLN distribution is a hierarchical model for a \(G\)-dimensional count
vector:

\[
\underline{x} \sim \mathcal{N}(\underline{\mu},\,
\underline{\underline{\Sigma}}), \quad
u_g \mid x_g \sim \text{Poisson}(e^{x_g}), \quad g = 1, \ldots, G,
\]

where the Poisson draws are conditionally independent given \(\underline{x}\).
The latent \(\underline{x} \in \mathbb{R}^G\) is the vector of log Poisson
rates, and \(\underline{\underline{\Sigma}}\) encodes cross-gene correlations in
log-rate space.

### Moments

The PLN has tractable closed-form moments:

| Quantity       | Expression                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Mean**       | \(\langle u_g \rangle = \exp(\mu_g + \tfrac{1}{2}\Sigma_{gg})\)                                                          |
| **Variance**   | \(\text{Var}(u_g) = \exp(2\mu_g + \Sigma_{gg})[\exp(\Sigma_{gg}) - 1] + \exp(\mu_g + \tfrac{1}{2}\Sigma_{gg})\)          |
| **Covariance** | \(\text{Cov}(u_g, u_{g'}) = \exp(\mu_g + \mu_{g'} + \tfrac{1}{2}(\Sigma_{gg} + \Sigma_{g'g'}))[\exp(\Sigma_{gg'}) - 1]\) |

These enable method-of-moments initialization of \(\underline{\mu}\) and
\(\underline{\underline{\Sigma}}\) from empirical count data.

---

## Low-rank covariance

Following the [GRN biophysics derivation](grn-biophysics.md), the covariance is
parameterized as:

\[
\underline{\underline{\Sigma}} =
\underline{\underline{W}}\,\underline{\underline{W}}^\top
+ \text{diag}(\underline{d}),
\quad W \in \mathbb{R}^{G \times k},\;
d \in \mathbb{R}_{>0}^G.
\]

The latent has dimension \(G\) (not \(G-1\) as in the LNM), so
\(W\) is \(G \times k\) and \(d\) is \(G\)-dimensional.

---

## Capture as a log-rate offset

In scRNA-seq, cell-specific capture efficiency enters naturally as an additive
offset in log-rate space. Let \(e^{-\eta^{(c)}}\) be the capture efficiency:

\[
u_g^{(c)} \mid x_g^{(c)}, \eta^{(c)} \sim
\text{Poisson}(e^{x_g^{(c)} - \eta^{(c)}}).
\]

The capture loss \(\eta^{(c)} > 0\) is shared across all genes in a cell. A
biology-informed truncated normal prior anchors the capture:

\[
\eta^{(c)} \sim \text{TruncN}(\log M_0 - \log L_c,\;
\sigma_M^2,\; \text{low} = 0),
\]

where \(L_c\) is the observed library size and \(M_0\) is the prior median
total mRNA per cell. This is structurally simpler than the NB-thinning
convolution required in the LNM framework.

---

## Log-concavity of the posterior

A key analytical advantage of the PLN is that the conditional posterior
\(\pi(\underline{x} \mid \underline{u})\) is **strictly log-concave**:

\[
\nabla^2_{\underline{x}} \log \pi =
-\text{diag}(e^{x_1}, \ldots, e^{x_G})
- \underline{\underline{\Sigma}}^{-1} \prec 0.
\]

Both terms are negative definite, giving a strictly concave log-posterior
everywhere. This guarantees:

- **Unique MAP** --- optimization-based inference (Laplace) is well-defined
- **No softmax invariance** --- unlike the LNM, which has a near-flat ridge
  along the \(\mathbf{1}\) direction from the softmax gauge symmetry
- **Accurate Gaussian approximation** --- the Laplace approximation centered
  at the MAP with covariance \((-H)^{-1}\) is well-justified

The joint \((\underline{x}, \eta)\) posterior is also strictly log-concave:
the Gaussian prior on \(\underline{x}\) and the truncated-normal on \(\eta\)
together break the one remaining near-null direction of the Poisson
likelihood.

---

## Contrast with the LNM

| Property           | LNM                                                  | PLN                                       |
| ------------------ | ---------------------------------------------------- | ----------------------------------------- |
| Latent space       | \(\mathbb{R}^{G-1}\) (ALR coordinates)               | \(\mathbb{R}^G\) (log-rates)              |
| Observation        | Multinomial\((u_T, \text{softmax}(y_\text{ALR}))\)   | \(\text{Poisson}(e^{x_g})\) independently |
| Total count        | Separate NB model for \(u_T\)                        | Emergent: \(u_T = \sum_g u_g\)            |
| Scale information  | Lost (softmax normalizes)                            | Retained in \(\underline{\mu}\)           |
| Posterior geometry | Log-concave but near-flat ridge along \(\mathbf{1}\) | Strictly log-concave, isotropic           |
| Capture model      | NB-thinning (convolution)                            | Log-rate offset (addition)                |

An important conceptual connection: the marginal compositions
\(\rho_g = u_g / \sum_j u_j\) under PLN are approximately logistic-normal,
since ratios of correlated log-normals are approximately logistic-normal. The
two models share compositional structure; they differ in whether composition is
modeled directly (LNM) or emerges as a consequence (PLN).

---

## Inference

The PLN supports two inference paths in SCRIBE:

### Laplace approximation (recommended)

```python
import scribe

results = scribe.fit(
    adata,
    model="pln",
    inference_method="laplace",
    n_steps=50_000,
)
```

The Laplace path exploits the log-concave posterior via Newton iteration on the
per-cell latents. Each Newton step costs \(O(Gk + k^3)\) using nested Woodbury
identities on the low-rank covariance. The outer loop optimizes global
parameters \((\mu, W, d)\) via Adam on the Laplace-approximated ELBO.

The Laplace path is structurally immune to **aggregate-posterior drift** (a
known failure mode of encoder-based inference) because it has no encoder: each
cell's posterior is computed locally from the data and the prior.

**Full details:** [Inference Methods > Laplace](../guide/inference.md#laplace-approximation)

### Variational Autoencoder (VAE)

```python
results = scribe.fit(
    adata,
    model="pln",
    inference_method="vae",
    vae_latent_dim=10,
    n_steps=100_000,
    batch_size=256,
)
```

The VAE path uses a neural encoder operating on factor scores
\(\underline{z} \in \mathbb{R}^k\). The encoder outputs \(2k\) numbers per cell
(mean and log-variance), making amortization tractable even at
\(G \sim 10^4\). This path is preferred when new cells must be scored at high
throughput without running Newton.

---

## Posterior predictive checks

Two PPC modes are available for the Laplace path:

| Mode                    | What it tests                                                                                | Cost                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **MAP-only**            | Does the likelihood shape match the data given point estimates?                              | One Poisson draw per (sample, cell, gene)                                               |
| **Laplace-uncertainty** | Does the posterior-predictive distribution (propagating Hessian uncertainty) match the data? | \(O(Gk + k^3)\) per sample (square-root factorization, no dense \(G \times G\) inverse) |

The Laplace-uncertainty PPC samples \(\underline{x}\) from the per-cell
Gaussian \(\mathcal{N}(\hat{\underline{x}}, (-H_{xx})^{-1})\) with
\(\hat{\eta}\) fixed, then draws Poisson counts. Holding \(\hat{\eta}\) fixed
projects out the rigid-translation null direction between \(\mu\) and \(\eta\),
avoiding spurious capture-direction spread in the histogram.

---

## When to use PLN

| Use PLN when...                                             | Use LNM when...                               |
| ----------------------------------------------------------- | --------------------------------------------- |
| Gene-level denoising with full correlation structure        | Compositional analysis is the primary goal    |
| Total-count distribution matters (heavy tails)              | Total-composition independence is acceptable  |
| Capture enters simply as a log-rate offset                  | NB totals are a good approximation            |
| You want the strongest posterior guarantees (log-concavity) | You need explicit compositional normalization |

!!! tip "Practical recommendation"
    For most analyses where gene-gene correlation recovery is the goal, start
    with `model="pln", inference_method="laplace"`. Switch to LNM/LNMVCP if
    your downstream analysis is inherently compositional (e.g., differential
    composition analysis). See [Model Selection](../guide/model-selection.md)
    for the full decision guide.

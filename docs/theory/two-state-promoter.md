# Two-state promoter (Poisson-Beta)

Despite the flexibility afforded by gene-specific parameters \(r_g\) and
\(p_g\), the negative binomial (NB) constrains every gene to the same
parametric *shape* and can never produce a genuinely bimodal histogram.
For a class of genes we call **bursty / bimodal**, the NB fails
empirically in two characteristic ways: an excess of zero counts beyond
what any NB can simultaneously fit with a heavy right tail, and
sometimes a literal bimodal histogram. The physical origin is **slow
promoter switching** — on / off rates slow relative to mRNA decay, so
cells get caught in long off stretches (near-zero counts) or long on
stretches (Poisson-like counts), and the aggregate across cells is
bimodal.

The **non-bursty two-state promoter model** is the natural likelihood
for these genes. It is the closed-form steady-state distribution of a
binary promoter switching between ON and OFF, with mRNA production
active only in the ON state. The Negative Binomial is recovered as a
limiting case (fast OFF rate relative to mRNA decay), so the two-state
model **nests inside the NB family rather than competing with it**.

---

## The Poisson-Beta compound representation

The two-state model's marginal distribution has a closed form involving
the confluent hypergeometric function \({}_1F_1\), but \({}_1F_1\) is
numerically fragile in the bursty regime and we do not evaluate it
directly. Instead, the same distribution admits an exact compound
representation:

\[
p_{gc} \;\sim\; \text{Beta}(\hat k^+_g, \hat k^-_g), \qquad
m_{gc} \mid p_{gc} \;\sim\; \text{Poisson}(\hat r_g \, p_{gc}).
\]

The latent \(p_{gc}\) is the *memory-weighted ON fraction* of the
promoter — the integral of the binary ON indicator against the
exponential mRNA-lifetime kernel — and naturally lies in \([0, 1]\)
with a Beta distribution at steady state. The non-dimensional rates
\(\hat r_g = r_g/\gamma\), \(\hat k^\pm_g = k^\pm_g/\gamma\) are
measured in units of the mRNA lifetime.

The model semantics specify \(p_{gc}\) **independent per cell-gene
pair**. Each cell's promoter is a separate stochastic system; the
shared parameters are the gene-level rates, not the latent activity.

---

## Closure under binomial thinning

Sequencing captures each mRNA molecule independently with cell-specific
probability \(\nu_c\):

\[
u_{gc} \mid m_{gc}, \nu_c \;\sim\; \text{Binomial}(m_{gc}, \nu_c).
\]

A binomial-of-Poisson is again Poisson, so

\[
u_{gc} \mid p_{gc}, \nu_c \;\sim\; \text{Poisson}(\hat r_g \, p_{gc} \,
\nu_c).
\]

The two-state distribution is **closed under binomial thinning**: the
capture probability simply rescales the Poisson rate, exactly as it
does for the NB family. The same capture priors, hierarchical
structures, and biology-informed anchors can be reused.

---

## The NB limit

When the OFF rate is fast compared to mRNA decay, \(\hat k^- \gg 1\),
the latent Beta concentrates near the origin while keeping the product
\(\hat r \, p\) at the right scale. The marginal collapses to a
Negative Binomial:

\[
\pi_{\text{ss}}(m) \;\xrightarrow{\hat k^- \gg 1}\;
\text{NegBinom}\!\Big(m \,\Big|\, \hat k^+, \;
\tfrac{1}{1 + \hat r/\hat k^-}\Big).
\]

The NB shape is \(\hat k^+\) (the ON rate) and the NB burst size is
\(\hat r/\hat k^-\) (the mean number of mRNAs per ON visit). Genes the
NB fits well are precisely those for which the fast-switching limit
applies; genes the NB fits poorly are the targets of the full two-state
machinery.

---

## Parameterizations

The four available parameterizations all describe the same family of
distributions; they differ only in which axes of the posterior
mean-field VI is asked to represent independently. Each successive
variant fixes a distinct geometric pathology of the previous.

### `two_state_natural` — physics-natural \((\mu, b, k^-)\)

Samples three positive per-gene parameters:

\[
\mu_g \;=\; \text{gene mean}, \qquad
b_g \;=\; \text{NB-limit burst size}, \qquad
k^-_g \;=\; \text{OFF rate}.
\]

Derives the natural Poisson-Beta parameters as

\[
\alpha_g = k^+_g = \mu_g/b_g, \qquad
\beta_g = k^-_g, \qquad
\hat r_g = \mu_g + b_g\, k^-_g.
\]

This map is mean-preserving by construction:
\(\langle u_{gc}\rangle = \hat r_g \cdot \alpha_g/(\alpha_g + \beta_g)
= \mu_g\) identically. Recommended for biophysical interpretation and
for NUTS, which recovers posterior correlations exactly.

### `two_state_ratio` — regime ratio \((\mu, b, s)\)

Replaces the absolute OFF rate by the dimensionless regime ratio
\(s_g = k^-_g/k^+_g\). Sampling \((\mu_g, b_g, s_g)\) gives

\[
\alpha_g = \mu_g/b_g, \qquad
\beta_g = s_g \mu_g/b_g, \qquad
\hat r_g = \mu_g\,(1 + s_g),
\]

with the mean still equal to \(\mu_g\) identically. The "NB-ness" of a
gene is now a single scalar, scale-invariant across genes — the
variational guide on \((\log\mu, \log b, \log s)\) no longer has to
discover a curved manifold along which \(b\) and \(k^-\) co-vary to
keep \(s\) sensible. Recommended for mean-field SVI on many genes with
widely varying mean expression.

### `two_state_mean_fano` — moments \((\mu, F, \kappa)\)

Samples the *first two observable moments* directly:

\[
F_g \;=\; \frac{\text{Var}[u_{gc}]}{\langle u_{gc}\rangle} - 1, \qquad
\kappa_g \;=\; \alpha_g + \beta_g.
\]

Derives

\[
\alpha_g = \frac{\kappa_g\, \mu_g}{\mu_g + F_g\,(\kappa_g + 1)},
\quad
\beta_g = \frac{\kappa_g\, F_g\,(\kappa_g + 1)}{\mu_g + F_g\,(\kappa_g + 1)},
\quad
\hat r_g = \mu_g + F_g\,(\kappa_g + 1).
\]

A direct check confirms both moments are preserved identically:
\(\langle u_{gc}\rangle = \mu_g\) and
\(\text{Var}[u_{gc}]/\langle u_{gc}\rangle - 1 = F_g\). Now
\(q(\text{excess\_fano})\) directly bounds the predictive variance per
gene by construction; \(q(\kappa)\) carries the
"departure from NB" shape only. NB limit: \(\kappa \to \infty\) gives
NB(shape = \(\mu/F\), burst = \(F\)) — the NB-default prior tilt goes
on \(\kappa\), not on \(F\). Recommended when posterior-predictive
bands under the natural or ratio parameterizations are systematically
wider than the observed gene-level variance.

### `two_state_moment_delta` — bounded shape \((\mu, F, \delta)\)

Same moment guarantees as `two_state_mean_fano`, but maps the
unbounded \(\kappa\) to a bounded \(\delta \in (0, 1)\):

\[
\delta_g \;=\; \frac{1}{\kappa_g + 1}.
\]

The forward map is

\[
\alpha_g = \frac{\mu_g\,(1-\delta_g)}{\mu_g\, \delta_g + F_g},
\quad
\beta_g = \frac{F_g\,(1-\delta_g)}{\delta_g\,(\mu_g\, \delta_g + F_g)},
\quad
\hat r_g = \frac{\mu_g\, \delta_g + F_g}{\delta_g}.
\]

The NB limit \(\kappa \to \infty\) becomes the bounded boundary
\(\delta \to 0\). \(\delta_g\) is sampled via a logit-Normal
(`SigmoidNormal`), so the variational support is naturally bounded;
mean-field \(q(\delta)\) cannot waste mass over an unbounded direction.
Recommended when `two_state_mean_fano` fits but the posterior on
\(\kappa\) visibly tracks its prior (i.e. when the data classifies a
gene as "NB-like" without identifying *how* NB-like it is).

### Choosing among the four

The four are mathematically equivalent; the distinction is operational.
NUTS recovers the same posterior up to MCMC noise regardless of choice.
Mean-field SVI should match the parameterization to the failure mode
the posterior-predictive checks expose.

| Variant | Best for |
| --- | --- |
| `two_state_natural` | Biophysical interpretation; NUTS |
| `two_state_ratio` | Many genes with widely varying \(\mu\); mean-field SVI |
| `two_state_mean_fano` | When PPC bands are systematically too wide |
| `two_state_moment_delta` | When \(\kappa\) posterior tracks its prior |

---

## Inference

The per-observation marginal log-likelihood is

\[
\log \mathcal{L}(u_{gc} \mid \mu_g, F_g, \kappa_g, \nu_c)
\;=\; \log \int_0^1
\text{Poisson}(u_{gc} \mid \hat r_g \, p \, \nu_c) \cdot
\text{Beta}(p \mid \alpha_g, \beta_g) \, dp,
\]

evaluated by **fixed Gauss-Legendre quadrature** on \([0, 1]\) with the
Beta density inside the integrand. We do not use Gauss-Jacobi
(which would absorb the Beta into the weight) because its nodes and
weights depend on \((\alpha, \beta)\) through a symmetric tridiagonal
eigendecomposition whose implicit-differentiation adjoint develops NaN
gradients near degenerate eigenvalues in the U-shaped Beta regime — the
forward log-likelihood is finite, but the backward pass blows up.
Fixed-node Gauss-Legendre has no autodiff path through `eigh`; the
Beta is evaluated explicitly at each node and its gradient flows
through standard `lgamma` operations.

The choice \(K \in \{40, \ldots, 80\}\) nodes is sufficient for UMI
counts up to several hundred. The Poisson log-PMF is computed in
log-rate form, with the \(u = 0\) branch handled separately to avoid
the IEEE-754 trap \(0 \cdot (-\infty) = \text{NaN}\) that arises from
\(u \log p_k\) when \(p_k\) underflows.

---

## Implementation correctness: ancestral sampling

A subtle but critical correctness point: when sampling from
`PoissonBetaCompound` ancestrally for posterior predictive checks, the
sampler draws \(p_{gc}\) **independently per (cell, gene)** — even
under variable capture where the rate matrix has shape \((C, G)\) and
\(\alpha, \beta\) are gene-rank \((G,)\). Sharing a single \(p_g\)
across cells per replicate would introduce a replicate-level random
effect the model does not have and would inflate every per-replicate
posterior-predictive histogram by \(\text{Std}[p] \cdot \text{rate}\)
(dominant in the U-shaped Beta regime where bursty genes live). The
log-prob path is correct independently because its gene-rank quadrature
marginalizes \(p_{gc}\) independently per \((c, g)\) by construction.

---

## Supported and unsupported features

**Fully supported:**

- **Mixture models** (`n_components=K`): the observation distribution becomes a
  `MixtureGeneral` over K `PoissonBetaCompound` components weighted by
  `mixing_weights`.  All four parameterizations work. Denoising marginalises
  over components (soft) or uses hard assignments. Log-prob decomposition
  supports `split_components`, `weights`, and `weight_type` for per-component
  analysis.
- **Biology-informed capture priors** (``priors={"capture_efficiency": (log_M0,
  sigma_M)}``): the closure under binomial thinning makes the capture factor
  enter the rate identically to its role in the NB family, so the prior math
  applies unchanged.
- **Biological PPC sampler** (``get_ppc_samples_biological`` and the MAP
  variant): by the same closure argument, the pre-capture distribution is
  exactly the gene-rank Poisson-Beta compound with ``p_capture`` dropped from
  the rate.

**Not yet supported:**

VAE inference, multi-dataset indexing, BNB overdispersion, and the Poisson-Gamma
denoiser are not wired for TwoState. Build-time validation rejects these
combinations with a clear directive.

See `paper/_two_state_promoter.qmd` for the long-form derivations.

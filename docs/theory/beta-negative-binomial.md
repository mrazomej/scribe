# Beta Negative Binomial

Despite the flexibility afforded by gene-specific parameters \(r_g\) and
\(p_g\), the negative binomial (NB) constrains every gene to the same
parametric *shape*. For many datasets this is adequate, but posterior
predictive checks (PPCs) can reveal systematic misfit: excess probability
mass at zero or heavy right tails that no NB---regardless of
parameterization---can accommodate.

The **Beta Negative Binomial (BNB)** distribution extends the NB by
allowing per-gene heavier tails while preserving the mean structure that
the entire normalization and compositional pipeline depends on. The BNB
is a strict generalization: it recovers the NB when the concentration
parameter \(\kappa_g \to \infty\), so enabling it never removes
modelling capacity.

---

## The BNB as an NB-Beta compound

### Definition

The BNB arises by placing a Beta prior on the success probability of a
negative binomial. Let

\[
X \mid p \;\sim\; \text{NB}(r, p), \qquad
p \;\sim\; \text{Beta}(\alpha, \kappa).
\]

Marginalizing over \(p\) yields the BNB distribution
\(X \;\sim\; \text{BNB}(r, \alpha, \kappa)\) with probability mass
function

\[
\pi(k \mid r, \alpha, \kappa) =
\frac{\Gamma(r + k)}{k!\;\Gamma(r)}
\frac{B(\alpha + k,\; \kappa + r)}{B(\alpha, \kappa)},
\qquad k = 0, 1, 2, \ldots
\]

where \(B(\cdot, \cdot)\) is the Beta function.

### Moments

The BNB has the following moments (when they exist):

\[
\langle X \rangle = \frac{r\,\alpha}{\kappa - 1},
\qquad \kappa > 1,
\]

\[
\text{Var}(X) =
\frac{r\,\alpha\,(r + \kappa - 1)\,(\alpha + \kappa - 1)}
{(\kappa - 1)^2\,(\kappa - 2)},
\qquad \kappa > 2.
\]

The mean requires \(\kappa > 1\) and the variance requires
\(\kappa > 2\)---mild constraints enforced by the parameterization below.

### Tail behaviour

For large \(k\), the BNB PMF decays as a **power law**:

\[
\pi(k \mid r, \alpha, \kappa)
\;\propto\; k^{-(1 + \kappa)},
\qquad k \to \infty.
\]

This is qualitatively different from the NB, whose PMF decays
geometrically (exponentially). The power-law tail means the BNB can
accommodate occasional very large counts that no NB can fit.

### Reduction to the NB

As the Beta prior concentrates around a point
\(p_0\)---parameterize \(\alpha = \kappa p_0\) and take
\(\kappa \to \infty\)---the BNB converges to \(\text{NB}(r, p_0)\).
Every NB distribution is a special case of the BNB.

---

## Mean-preserving parameterization

### Why it matters

The normalization and compositional analysis framework relies on the mean
of the count distribution factoring as (gene-specific) x (cell-specific):

\[
\langle u_{gc} \rangle =
\underbrace{\frac{r_g(1 - p_g)}{p_g}}_{\text{gene-specific}}
\;\times\;
\underbrace{\nu_c}_{\text{cell-specific}}.
\]

To preserve this structure, the BNB's mean must match the NB's mean
exactly.

### The mean-matching condition

Setting the BNB mean equal to the NB mean and solving for the first Beta
parameter:

\[
\boxed{
\alpha_{gc} =
\frac{(1 - \hat{p}_{gc})\,(\kappa_g - 1)}{\hat{p}_{gc}},
}
\]

where \(\hat{p}_{gc}\) is the effective success probability incorporating
capture (from the
[Dirichlet-Multinomial](dirichlet-multinomial.md) derivation). The only
new per-gene parameter is \(\kappa_g > 2\).

### Verification

Substituting back:

\[
\langle u_{gc} \rangle_{\text{BNB}}
= \frac{r_g\,\alpha_{gc}}{\kappa_g - 1}
= \frac{r_g(1 - p_g)}{p_g} \cdot \nu_c
= \langle u_{gc} \rangle_{\text{NB}}.
\]

The gene x cell mean factorization is intact. **All compositional
quantities---the Gamma-based composition sampling, CLR coordinates,
and differential expression---carry through unchanged.**

### Variance inflation factor

Under this parameterization, the BNB variance becomes

\[
\text{Var}(u_{gc})_{\text{BNB}} =
\underbrace{
\frac{r_g(1 - \hat{p}_{gc})}{\hat{p}_{gc}^2}
}_{\text{NB variance}}
\;\times\;
\underbrace{
\frac{r_g + \kappa_g - 1}{\kappa_g - 2}
}_{\text{inflation factor}}.
\]

The inflation factor satisfies:

| \(\kappa_g\) | Inflation factor (\(r_g = 5\)) | Regime |
|:---:|:---:|:---|
| \(\infty\) | 1 | Pure NB |
| 100 | 1.06 | Nearly NB |
| 10 | 1.75 | Moderate extra dispersion |
| 5 | 3.0 | Heavy extra dispersion |
| 3 | 7.0 | Very heavy |

As \(\kappa_g \to \infty\) the factor converges to 1 (NB variance);
as \(\kappa_g \to 2^+\) it diverges (arbitrarily large overdispersion).

---

## Biophysical interpretation

### Two routes to the NB

Two distinct biophysical models lead to the NB as a steady-state mRNA
distribution:

**Route A -- one-state bursty promoter (exact).** A constitutive
promoter initiates transcription bursts at rate \(k_i\), each producing
a geometrically distributed number of mRNAs with mean burst size \(b\).
The chemical master equation yields the NB **exactly**:

\[
r_g = \frac{k_i}{\gamma}, \qquad
p_g = \frac{1}{1 + b},
\]

where \(\gamma\) is the mRNA degradation rate.

**Route B -- two-state promoter (bursty limit).** A two-state promoter
switches between OFF and ON states. In the bursty limit
(\(k^- \gg \gamma\)), the distribution reduces to the NB with
\(r_g = k^+/\gamma\) and \(b = r_m/k^-\).

Both routes produce the same NB, which encodes the **intrinsic noise** of
gene expression: the irreducible stochastic fluctuations when all kinetic
rates are fixed across cells.

### From variable burst size to the BNB

In practice, kinetic rates vary from cell to cell (**extrinsic noise**):
fluctuating transcription-factor concentrations, varying chromatin
accessibility, cell-cycle effects. If the NB success probability \(p_g\)
varies across cells according to a Beta distribution centered on the
population value, marginalizing out this cell-to-cell variation yields
exactly the BNB.

The concentration parameter \(\kappa_g\) therefore measures the magnitude
of **extrinsic noise in burst size** for gene \(g\):

- **Large \(\kappa_g\)** (tight Beta): burst size is nearly constant
  across cells; the gene behaves as a standard NB.
- **Small \(\kappa_g\)** (diffuse Beta): substantial cell-to-cell
  heterogeneity in burst kinetics; heavier tails than any NB can
  accommodate.

### Why this is biological, not technical

Three considerations argue that \(\kappa_g\) reflects genuine biology
rather than technical artifacts:

1. **UMIs remove technical heavy tails.** PCR amplification bias---the
   most common source of technical overdispersion---is collapsed by UMI
   deduplication.
2. **Variable capture already handles technical scaling.** The
   cell-specific \(\nu_c\) absorbs variations in sequencing depth and
   droplet capture efficiency.
3. **Gene-specific hyper-variability is a biological signature.** Genes
   requiring finite \(\kappa_g\) are typically those with complex
   regulation: stress-response genes, cell-cycle-dependent genes, or
   genes sensitive to fluctuating transcription-factor gradients.

---

## Hierarchical prior on kappa_g

### Design principle

The BNB introduces one new parameter per gene. Since most genes are
adequately described by the NB, the prior must:

1. **Default to NB**: concentrate mass at \(\kappa_g = \infty\)
   (equivalently \(\omega_g = 0\)).
2. **Allow escape**: genes with strong evidence for extra overdispersion
   can have finite \(\kappa_g\).
3. **Be sparse**: only a data-driven subset of genes should deviate.

This is the same design principle used for the
[hierarchical gate](hierarchical-priors.md#hierarchical-zero-inflation-gate)
and [gene-specific p](hierarchical-p.md).

### Reparameterization via excess dispersion fraction

To apply sparsity-inducing priors, the concentration is
reparameterized as the **excess dispersion fraction**:

\[
\omega_g = \frac{r_g + 1}{\kappa_g - 2},
\qquad \kappa_g > 2.
\]

Key properties:

- \(\omega_g = 0 \Leftrightarrow \kappa_g = \infty\): no
  overdispersion (NB limit).
- \(\omega_g > 0\): positive overdispersion; the variance inflation
  factor is \(1 + \omega_g\).
- \(\omega_g \to \infty\): \(\kappa_g \to 2^+\), maximum
  overdispersion.

### Horseshoe and NEG priors

Both the **regularized horseshoe** and the **NEG** prior (see
[Hierarchical Priors](hierarchical-priors.md) for the general theory)
can be applied to \(\omega_g\). In either case, the NCP form is:

\[
z_g \;\sim\; \mathcal{N}(0, 1), \qquad
\omega_g = \text{softplus}(\mu_\omega + \sigma_g \cdot z_g),
\]

where \(\sigma_g\) is the family-specific per-gene scale
(\(\tau \tilde{\lambda}_g\) for horseshoe, \(\sqrt{\psi_g}\) for NEG)
and \(\mu_\omega\) defaults to a large negative value pushing
\(\omega_g \approx 0\).

The posterior of \(\omega_g\) serves as a **per-gene diagnostic** for
the presence of extra overdispersion beyond what the NB can accommodate.

### NB recovery

When the prior shrinks \(\omega_g \to 0\) for all genes:

\[
\kappa_g \to \infty, \qquad
\text{BNB}(r_g, \alpha_{gc}, \kappa_g) \to \text{NB}(r_g, \hat{p}_{gc}),
\]

recovering the standard NB model.

---

## Interactions with other model components

### Zero inflation

The BNB composes with the zero-inflation gate in the same way as the NB:

\[
u_{gc} \;\sim\;
\begin{cases}
0 & \text{with probability } \pi_g, \\
\text{BNB}(r_g, \alpha_{gc}, \kappa_g) & \text{with probability }
1 - \pi_g.
\end{cases}
\]

The [hierarchical gate](hierarchical-priors.md#hierarchical-zero-inflation-gate)
applies unchanged. The BNB's heavier tails may reduce the need for zero
inflation by better fitting the body and tail, potentially leading to
smaller inferred \(\pi_g\) values. If adding the BNB shifts probability
mass from \(\pi_g\) to the count distribution, the BNB is capturing
real signal that was previously mis-attributed to zero inflation.

### Variable capture probability

The VCP enters through \(\hat{p}_{gc}\), which is a function of the
gene-level \(p_g\) and the cell-level \(\nu_c\). Since \(\alpha_{gc}\)
is determined by \(\hat{p}_{gc}\) via the mean-matching condition, the
VCP framework integrates seamlessly.

### Composition sampling

The Gamma-based composition sampling relies on the mean of the count
distribution. Since the BNB mean is identical to the NB mean,
composition sampling is unchanged:

\[
\rho_g =
\frac{\gamma_g \cdot (1 - p_g) / p_g}
{\sum_{j=1}^G \gamma_j \cdot (1 - p_j) / p_j},
\qquad
\gamma_g \;\sim\; \text{Gamma}(r_g, 1).
\]

This is independent of \(\kappa_g\). The BNB does not alter the
compositional analysis or differential expression pipeline.

---

## Parameterization recommendation

The BNB can only add variance on top of an already correct mean
structure---the mean-matching condition is a **hard constraint**, not a
soft penalty. This means the BNB's effectiveness critically depends on
the underlying NB's ability to calibrate gene-level means.

### Canonical parameterization (recommended)

In the **canonical** parameterization, \(r_g\) and \(p_g\) are
independent, each with its own prior:

\[
\log r_g \;\sim\; \mathcal{N}(r_0, \sigma_r^2), \qquad
\text{logit}(p_g) \;\sim\; \mathcal{N}(p_0, \sigma_p^2).
\]

There is no coupling between the two. The logit space easily
accommodates the full dynamic range of single-cell expression
(from \(p_g \approx 0.005\) to \(p_g \approx 0.9998\)), allowing the
NB to correctly capture each gene's mean. The BNB then adds value in its
intended role: modelling genuine excess variance from extrinsic noise.

### Why mean-odds can be limiting

In the **mean-odds** parameterization, \(r_g = \mu_g \cdot \phi_g\) and
\(p_g = 1/(1 + \phi_g)\). The hierarchical prior on \(\log \phi_g\)
would need to span roughly 13 orders of magnitude in \(\phi_g\) to
cover the full expression range, providing negligible shrinkage. In
practice, the prior settles on a moderate width that prevents \(p_g\)
from reaching extreme values, producing systematic mean miscalibration
for high-expression genes. Because the mean-matching constraint is hard,
\(\kappa_g\) cannot shift the distribution's center---only its spread.

!!! warning "Parameterization matters"
    When using the BNB, the **canonical parameterization** is recommended.
    Mean-coupled parameterizations (mean-odds, mean-probability) can
    restrict the NB's dynamic range and produce mean miscalibration that
    \(\kappa_g\) cannot remedy.

---

## BNB denoising

The [Bayesian denoising](denoising.md) framework extends to the BNB,
though the clean closed-form NB result is replaced by a tractable
one-dimensional integral.

### Key insight: conditional NB denoising

Conditional on the latent Beta mixing variable \(p_g\), the BNB reduces
to an NB. All NB denoising machinery therefore applies exactly:

\[
d_g \mid u_g, p_g \;\sim\;
\text{NB}(r_g + u_g, \;\; \nu_c + p_g(1 - \nu_c)).
\]

### Marginal posterior mean

The marginal posterior mean of the denoised count is obtained by
averaging the conditional NB result over the posterior of \(p_g\):

\[
\langle m_g \mid u_g \rangle
= u_g + (r_g + u_g) \cdot
\frac{\displaystyle\int_0^1 f(p)\;\tilde{\pi}(p \mid u_g)\,dp}
{\displaystyle\int_0^1 \tilde{\pi}(p \mid u_g)\,dp},
\]

where \(f(p) = (1 - \nu_c)(1 - p) / [\nu_c + p(1 - \nu_c)]\) and

\[
\tilde{\pi}(p \mid u_g) \propto
\frac{p^{r_g + \alpha_g - 1}\;(1 - p)^{u_g + \kappa_g - 1}}
{[\nu_c + p(1 - \nu_c)]^{r_g + u_g}}
\]

is the unnormalized posterior of the latent mixing variable. Both
integrals are one-dimensional and smooth on \([0, 1]\), making them
ideally suited to **Gauss-Legendre quadrature** with 32-64 nodes.

### Sampling from the denoising posterior

For fully Bayesian denoising (preserving cross-gene correlations):

1. **Draw** \(p_g^{(s)}\) from \(\pi(p_g \mid u_g)\) via grid-based
   inverse CDF sampling on \([0, 1]\).
2. **Sample** \(d_g^{(s)}\) from the conditional NB:
   \(\text{NB}(r_g + u_g, \;\nu_c + p_g^{(s)}(1 - \nu_c))\).
3. **Compute** \(m_g^{(s)} = u_g + d_g^{(s)}\).

### NB recovery

When \(\kappa_g \to \infty\), the Beta posterior collapses to a point
mass, the integral reduces to the integrand evaluated at a single point,
and the closed-form NB denoising formula is recovered exactly.

---

## Using the BNB in SCRIBE

### Single dataset

```python
import scribe

# Enable BNB overdispersion with horseshoe prior
results = scribe.fit(
    adata,
    model="nbdm",
    parameterization="canonical",
    overdispersion="bnb",
    overdispersion_prior="horseshoe",   # or "neg"
)
```

### Multi-dataset with hierarchical prior on kappa_g

```python
# BNB with dataset-level hierarchical prior
results = scribe.fit(
    adata,
    model="nbvcp",
    parameterization="canonical",
    dataset_key="batch",
    overdispersion="bnb",
    overdispersion_prior="horseshoe",
    overdispersion_dataset_prior="neg",
    priors={"organism": "human"},
)
```

### Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overdispersion` | `"none"` | Set to `"bnb"` to enable the BNB extension |
| `overdispersion_prior` | `"horseshoe"` | Gene-level prior on \(\kappa_g\): `"horseshoe"` or `"neg"` |
| `overdispersion_dataset_prior` | `"none"` | Dataset-level: `"gaussian"`, `"horseshoe"`, `"neg"` (requires `dataset_key`) |
| `parameterization` | `"canonical"` | `"canonical"` recommended with BNB |

!!! tip "When to use the BNB"
    Enable the BNB when PPCs reveal systematic right-tail misfit or
    excess zeros for a subset of genes. The hierarchical prior ensures
    that most genes default to NB behaviour, so enabling the BNB is
    low-risk: genes that do not need it are automatically shrunk back
    to the standard NB.

For the full API details, see the [API Reference](../reference/scribe/api/).

---

!!! tip "Next steps"
    - See [Hierarchical Priors](hierarchical-priors.md) for the full theory of
      the Horseshoe and NEG prior families applied to the BNB concentration
      parameter \(\kappa_g\).
    - See [Anchoring Priors](anchoring-priors.md) for why the NB mean must be
      correctly anchored before the BNB extension is effective — the
      mean-matching constraint is a hard dependency.
    - See [Bayesian Denoising](denoising.md) for the NB denoising result on
      which BNB denoising (conditional NB plus Beta integral) is built.
    - See the [NBDM model page](../models/nbdm.md) for practical usage,
      including the `overdispersion` and `overdispersion_prior` parameters.

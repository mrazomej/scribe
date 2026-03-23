# Hierarchical Gene-Specific \(p\)

This page extends the [Dirichlet-Multinomial model](dirichlet-multinomial.md)
by relaxing the shared-\(p\) assumption. For complete derivations, see the
accompanying paper.

## Motivation

The NB–Dirichlet-Multinomial factorization from the
[previous section](dirichlet-multinomial.md#key-result-2-nbdirichlet-multinomial-factorization)
relies critically on the assumption that all genes share a single success
probability \(p\). Under that assumption, the compositional term depends
only on the gene-specific dispersion parameters \(\underline{r}\) and not on
\(\hat{p}\). This clean separation is what makes differential expression
analysis on compositions tractable.

But what if genes truly have different success probabilities? This section
shows what breaks, how to fix it, and that the fix strictly generalizes the
original model.

## Why the Factorization Breaks

When each gene has its own \(p_g\), the Poisson-Gamma representation becomes

\[
\lambda_g \sim \text{Gamma}(r_g, \theta_g), \qquad
u_g \mid \lambda_g \sim \text{Poisson}(\lambda_g),
\]

where \(\theta_g = p_g / (1 - p_g)\) now differs across genes. The product of
Gamma densities contains an exponential term:

\[
\prod_{g=1}^G e^{-\theta_g \lambda_g} =
\exp\!\left(-\sum_{g=1}^G \theta_g \lambda_g\right).
\]

Under the change of variables \(\lambda_g = \Lambda \rho_g\) (where
\(\Lambda = \sum_g \lambda_g\) is the total and \(\rho_g\) the proportions),
this becomes

\[
\exp\!\left(-\Lambda \sum_{g=1}^G \theta_g \rho_g\right).
\]

When all \(\theta_g = \theta\), this reduces to \(\exp(-\theta \Lambda)\),
which depends only on \(\Lambda\) and factors cleanly into the Gamma
distribution over totals. But when the \(\theta_g\) vary, the term
**couples \(\Lambda\) and \(\underline{\rho}\)**:

\[
\exp\!\left(-\Lambda \sum_{g=1}^G \theta_g \rho_g\right) \neq
f(\Lambda) \cdot g(\underline{\rho})
\]

for any functions \(f\) and \(g\). The Gamma-Dirichlet factorization fails,
the compositions are no longer Dirichlet, and \(\hat{p}\) can no longer be
separated from the compositional term.

## The Hierarchical Model

Rather than treating each \(p_g\) as a free parameter (adding \(G\)
unconstrained parameters), we place a **hierarchical prior** that provides
adaptive shrinkage: genes with limited data are regularized toward the
population mean, while genes with abundant data can deviate.

### Specification

Since \(p_g \in (0, 1)\), we work in the unconstrained logit space:

**Hyperpriors (population level):**

\[
\mu_p \sim \mathcal{N}(0, 1), \qquad
\sigma_p \sim \text{Softplus}\!\left(\mathcal{N}(0, 1)\right),
\]

where \(\mu_p\) is the population mean of the logit-\(p_g\) and
\(\sigma_p > 0\) is the population standard deviation (the Softplus
transformation ensures positivity).

**Gene-level parameters:**

\[
\text{logit}(p_g) \sim \mathcal{N}(\mu_p, \sigma_p),
\quad g = 1, \ldots, G,
\]

with the gene-specific success probability recovered via the sigmoid:
\(p_g = 1 / (1 + e^{-\text{logit}(p_g)})\).

**Observation model:**

\[
r_g \sim \pi(r_g), \qquad
u_g \mid r_g, p_g \sim \text{NB}(r_g, p_g),
\quad g = 1, \ldots, G.
\]

### Full posterior

For \(C\) cells, the posterior is:

\[
\pi(\underline{r}, \underline{p}, \mu_p, \sigma_p \mid
\underline{\underline{U}}) \propto
\left[\prod_{c=1}^C \prod_{g=1}^G
    \pi(u_{gc} \mid r_g, p_g)
\right]
\left[\prod_{g=1}^G \pi(p_g \mid \mu_p, \sigma_p)\,\pi(r_g)\right]
\pi(\mu_p)\,\pi(\sigma_p).
\]

### Interpreting \(\sigma_p\)

The hyperparameter \(\sigma_p\) serves as a **data-driven diagnostic** of the
shared-\(p\) assumption:

- **\(\sigma_p \approx 0\):** The data are consistent with a shared \(p\).
  All gene-specific \(p_g\) are shrunk toward the common value
  \(\sigma(\mu_p)\), and the model effectively recovers the standard NBDM.
- **\(\sigma_p \gg 0\):** There is substantial gene-to-gene variation in
  \(p_g\) that a single shared parameter cannot capture.

This means one does not need to commit a priori to either model — the
hierarchical prior lets the data decide.

## Composition Sampling with Gene-Specific \(p_g\)

Under the standard model, posterior compositions are drawn as

\[
\underline{\rho}^{(s)} \sim \text{Dir}(\underline{r}^{(s)}),
\]

using only the posterior samples of \(\underline{r}\). With gene-specific
\(p_g\), a generalized procedure is needed.

### The Gamma-based procedure

The key insight is the Poisson-Gamma representation. Draw
\(\gamma_g\) from a \(\text{Gamma}(r_g, 1)\) distribution (unit rate).

Then \(\tilde{\lambda}_g = \gamma_g \cdot (1 - p_g) / p_g\) has the
distribution \(\text{Gamma}(r_g, \theta_g)\) — the latent expected count
for gene \(g\).

The composition is then:

\[
\rho_g = \frac{
    \gamma_g \cdot \frac{1 - p_g}{p_g}
}{
    \sum_{j=1}^G \gamma_j \cdot \frac{1 - p_j}{p_j}
}.
\]

In practice, drawing a compositional sample requires three steps:

1. Draw \(\gamma_g \sim \text{Gamma}(r_g, 1)\) independently for each gene.
2. Scale: \(\tilde{\lambda}_g = \gamma_g \cdot (1 - p_g) / p_g\).
3. Normalize: \(\rho_g = \tilde{\lambda}_g \,/\, \sum_j \tilde{\lambda}_j\).

### Reduction to Dirichlet when \(p_g = p\)

When all \(p_g = p\), the scaling factor \((1-p)/p\) is a positive constant
that cancels between numerator and denominator:

\[
\rho_g = \frac{
    \gamma_g \cdot \frac{1-p}{p}
}{
    \sum_j \gamma_j \cdot \frac{1-p}{p}
}
= \frac{\gamma_g}{\sum_j \gamma_j}.
\]

By the standard construction of the Dirichlet distribution (normalizing
independent Gamma variates), this is exactly
\(\underline{\rho} \sim \text{Dir}(\underline{r})\). The Gamma-based
procedure is therefore a **strict generalization** of Dirichlet sampling —
it agrees with the standard model when the shared-\(p\) assumption holds
and correctly incorporates gene-specific probabilities when it does not.

### What changes when \(p_g\) vary

When the \(p_g\) differ, compositions are no longer governed by
\(\underline{r}\) alone. The factor \((1-p_g)/p_g\) reweights each gene's
contribution: two genes with identical \(r_g\) but different \(p_g\) will
have different expected compositional proportions, because a lower \(p_g\)
corresponds to a higher expected count in the NB parameterization. This means
differential expression can arise from changes in \(p_g\) between conditions
even when \(r_g\) remains constant — a signal the standard model would miss.

## Differential Expression

The empirical differential expression framework extends naturally. The only
change is in the composition-generating step: instead of drawing from a
Dirichlet, we use the Gamma-based procedure with both the posterior samples
of \(\underline{r}\) and \(\underline{p}\).

Given posterior samples for conditions A and B, the CLR differences are
computed as:

1. **Sample compositions** via the Gamma-based procedure for each condition.
2. **CLR transform:** \(z_{g,\text{CLR}}^{(s)} = \log \rho_g^{(s)} -
   \frac{1}{G}\sum_j \log \rho_j^{(s)}\).
3. **Paired differences:**
   \(\Delta_g^{(s)} = z_{A,g,\text{CLR}}^{(s)} - z_{B,g,\text{CLR}}^{(s)}\).
4. **Compute gene-level statistics** (lfsr, probability of effect, etc.)
   from the matrix of \(\Delta_g^{(s)}\) samples.

The output has the same structure as the standard pipeline — all downstream
analysis functions work regardless of which composition sampling method was
used.

---

!!! tip "Next steps"
    - See the [Dirichlet-Multinomial Model](dirichlet-multinomial.md) page for
      the foundational theory.
    - See the [NBDM model page](../models/nbdm.md) for practical usage and
      implementation details.

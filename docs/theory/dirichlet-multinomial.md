# Dirichlet-Multinomial Model

This page presents the key theoretical results underlying the [NBDM
model](../guide/model-selection.md#base-negative-binomial-nbdm). For complete
derivations with all algebraic steps, see the accompanying paper.

## Model Setup

Single-cell RNA-seq data is represented as a \(G \times C\) count matrix
\(\underline{\underline{U}}\), where each entry \(u_g\) records the number of
UMI counts for gene \(g\) in a given cell. The model proceeds from two
distributional assumptions about how these counts arise.

### True transcript counts

From biophysical first principles (the two-state promoter model at steady
state), the number of mRNA molecules produced by gene \(g\) follows a negative
binomial distribution:

\[
m_g \sim \text{NB}(r_g, p),
\]

where \(r_g\) is a gene-specific dispersion parameter. The key modeling
assumption is that **all genes share the same success probability** \(p\). As
discussed below, this assumption is less restrictive than it appears due to the
parameter degeneracy of the negative binomial.

### From mRNA to observed UMI counts

Each mRNA molecule is captured with probability \(\nu\), so the observed UMI
count for gene \(g\), conditioned on the true transcript count \(m_g\), follows
a binomial distribution:

\[
u_g \mid m_g, \nu \sim \text{Binomial}(m_g, \nu).
\]

Since \(m_g\) is unobserved, we marginalize over it to obtain the distribution
of the observed count:

\[
\pi(u_g \mid r_g, p, \nu) =
\sum_{m_g = u_g}^{\infty}
\underbrace{\pi(u_g \mid m_g, \nu)}_{\text{Binomial}}
\times
\underbrace{\pi(m_g \mid r_g, p)}_{\text{Negative Binomial}}.
\]

## Key Result 1: NB–Binomial Composition

The marginal distribution of UMI counts after integrating out unobserved mRNA
counts is again a negative binomial:

\[
\boxed{
u_g \mid r_g, \hat{p} \sim \text{NB}(r_g, \hat{p}),
}
\]

with a rescaled success probability

\[
\hat{p} = \frac{p}{\nu + p(1 - \nu)}.
\]

This can be shown via probability generating functions: the PGF of a negative
binomial composed with a binomial reduces to the PGF of a negative binomial
with the transformed parameter \(\hat{p}\). The shape parameter \(r_g\)
is unchanged.

!!! info "Why this matters"
    The capture process does not change the distributional family — it only
    rescales \(p\). This means we can work directly with UMI counts under the
    same negative binomial framework, absorbing the unknown capture efficiency
    into \(\hat{p}\).

## Key Result 2: NB–Dirichlet-Multinomial Factorization

Assuming genes are independent and share \(\hat{p}\), the joint distribution
of all UMI counts in a cell is

\[
\pi(\underline{u} \mid \underline{r}, \hat{p}) =
\prod_{g=1}^G \pi(u_g \mid r_g, \hat{p}).
\]

This product of independent negative binomials can be factorized as:

\[
\boxed{
\pi(\underline{u} \mid \underline{r}, \hat{p}) =
\underbrace{
    \pi(u_T \mid r_T, \hat{p})
}_{\text{Negative Binomial}}
\times
\underbrace{
    \pi(\underline{u} \mid u_T, \underline{r})
}_{\text{Dirichlet-Multinomial}},
}
\]

where \(u_T = \sum_{g=1}^G u_g\) is the total UMI count and
\(r_T = \sum_{g=1}^G r_g\).

### How the factorization works

The proof proceeds through two intermediate decompositions that exploit the
Poisson-Gamma representation of the negative binomial.

**Step 1 — Poisson-Gamma representation.** Each negative binomial can be
written as a compound distribution:

\[
\lambda_g \sim \text{Gamma}(r_g, \theta), \qquad
u_g \mid \lambda_g \sim \text{Poisson}(\lambda_g),
\]

where \(\theta = \hat{p}/(1-\hat{p})\). This is equivalent to saying: sample
a Poisson rate from a Gamma distribution, then draw the count from that
Poisson.

**Step 2 — Product of Poissons.** The joint distribution of \(G\) independent
Poisson random variables can be decomposed as:

\[
\prod_{g=1}^G \text{Poisson}(u_g \mid \lambda_g) =
\text{Poisson}(u_T \mid \lambda_T) \times
\text{Multinomial}(\underline{u} \mid u_T, \underline{\rho}),
\]

where \(\lambda_T = \sum_g \lambda_g\) and
\(\rho_g = \lambda_g / \lambda_T\) are the fractional rates.

**Step 3 — Product of Gammas.** When all Gamma distributions share the same
rate parameter \(\theta\), the joint distribution decomposes as:

\[
\prod_{g=1}^G \text{Gamma}(\lambda_g \mid r_g, \theta) =
\text{Gamma}(\lambda_T \mid r_T, \theta) \times
\text{Dirichlet}(\underline{\rho} \mid \underline{r}).
\]

This step uses a change of variables from the individual
\(\lambda_g\) to the total \(\lambda_T\) and the proportions
\(\underline{\rho}\), with a Jacobian determinant of
\(\Lambda^{G-1}\).

**Combining everything.** Substituting Steps 2 and 3 into the integral over
\(\underline{\lambda}\) and performing the integrations:

- The Poisson × Gamma integral over \(\lambda_T\) yields the negative
  binomial \(\text{NB}(u_T \mid r_T, \hat{p})\).
- The Multinomial × Dirichlet integral over \(\underline{\rho}\) yields the
  Dirichlet-Multinomial
  \(\pi(\underline{u} \mid u_T, \underline{r})\).

The Dirichlet-Multinomial probability mass function takes the explicit form:

\[
\pi(\underline{u} \mid u_T, \underline{r}) =
\frac{
    \Gamma(r_T)\,\Gamma(u_T+1)
}{
    \Gamma(u_T + r_T)
}
\prod_{g=1}^G
\frac{
    \Gamma(u_g + r_g)
}{
    \Gamma(r_g)\,\Gamma(u_g + 1)
}.
\]

## Bayesian Inference

Given the factorized likelihood, we perform Bayesian inference over the
model parameters \(\underline{r}\) and \(\hat{p}\).

### Prior distributions

The priors are chosen to respect the parameter domains:

\[
\hat{p} \sim \text{Beta}(\alpha_p, \beta_p), \qquad
r_g \sim \text{Gamma}(\alpha_r, \beta_r), \quad g = 1, \ldots, G.
\]

### Full posterior

For \(C\) independent cells, the posterior takes the form:

\[
\pi(\underline{r}, \hat{p} \mid \underline{\underline{U}}) \propto
\prod_{c=1}^C \left[
    \pi(u_{T,c} \mid r_T, \hat{p})\;
    \pi(\underline{u}_c \mid u_{T,c}, \underline{r})
\right]
\cdot \prod_{g=1}^G \pi(r_g) \cdot \pi(\hat{p}).
\]

!!! note "Separation of concerns"
    The factorized form reveals a clean separation: \(\hat{p}\) only appears in
    the negative binomial term for total counts, while the compositional term
    (Dirichlet-Multinomial) depends only on \(\underline{r}\). This means the
    **composition of the transcriptome is governed entirely by the gene-specific
    dispersion parameters** \(r_g\), independently of \(\hat{p}\).

## Interpretation

The factorization has two important implications for single-cell RNA-seq
analysis.

### Natural normalization

In the construction of the model, the \(\underline{\rho}\) parameters emerge
as the fractional abundances of each gene, lying on the unit simplex
(\(\rho_g \geq 0\), \(\sum_g \rho_g = 1\)). These are the natural
"normalized" quantities: rather than dividing raw counts by library size,
the model places a **Dirichlet distribution** over these fractions and learns
their parameters from data.

### Distributions over fractions, not point estimates

Unlike standard normalization pipelines that produce a single point estimate
of gene fractions, the NBDM model yields a full **posterior distribution** over
\(\underline{\rho}\) via the Dirichlet. This preserves the uncertainty inherent
in overdispersed count data — arguably the key motivation for performing
single-cell experiments in the first place.

When a point estimate is needed, the posterior mean of the Dirichlet provides
a natural choice. But for downstream analyses such as differential expression,
working with the full distribution enables principled uncertainty quantification.

---

!!! tip "Next steps"
    - See the [Model Selection](../guide/model-selection.md) guide for
      practical usage and choosing the right model variant.
    - See the [Hierarchical Gene-Specific \(p\)](hierarchical-p.md) page for
      what happens when the shared-\(p\) assumption is relaxed.
    - See [Bayesian Denoising](denoising.md) to learn how the NB generative
      model here enables closed-form posterior recovery of true transcript
      counts from observed UMIs.
    - See [Differential Expression](differential-expression.md) for how the
      Dirichlet compositions derived here serve as the direct input to the
      compositional DE framework.

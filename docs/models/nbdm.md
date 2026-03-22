# Negative Binomial-Dirichlet Multinomial Model (NBDM)

The Negative Binomial-Dirichlet Multinomial (NBDM) model is designed to capture
key characteristics of single-cell RNA sequencing data, particularly:

1. The overdispersed nature of molecular counts
2. The compositional aspect of sequencing data
3. Gene-specific variation in expression levels

We emphasize that the NBDM model is different from many standard scRNA-seq
pipelines, which use library-size normalization (dividing by total UMI count,
then log-transform, etc.). By contrast, the NBDM model integrates
normalization into the generative process (via the Dirichlet-multinomial). This
can provide a more principled measure of uncertainty and can help separate
technical from biological variation.

!!! abstract "Mathematical background"
    For a detailed walkthrough of the theoretical derivations behind this model,
    see the [Theory section](../theory/dirichlet-multinomial.md). To understand
    what happens when the shared-\(p\) assumption is relaxed, see
    [Hierarchical Gene-Specific \(p\)](../theory/hierarchical-p.md).

## Model Structure

The NBDM model follows a hierarchical structure where:

Each cell's total molecular count follows a
[Negative Binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution).
Given the total count, the allocation of molecules across genes follows a
[Dirichlet-Multinomial distribution](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution).

Formally, for a dataset with \(N\) cells and \(G\) genes, let \(u_{g}^{(c)}\)
be the UMI count for gene \(g\) in cell \(c\), and
\(U^{(c)} = \sum_{g=1}^G u_{g}^{(c)}\) be the total UMI count for cell
\(c\). The generative process is:

1. Draw global success probability: \(p \sim \text{Beta}(\alpha_p, \beta_p)\)
2. Draw gene-specific dispersion parameters:
   \(r_g \sim \text{Gamma}(\alpha_r, \beta_r)\) for \(g = 1,\ldots,G\)
3. For each cell \(c = 1,\ldots,N\):
    - Draw total count:
      \(U^{(c)} \sim \text{NegativeBinomial}(\sum_g r_g, p)\)
    - Draw gene proportions:
      \(u^{(c)} \sim \text{DirichletMultinomial}(r, U^{(c)})\)

## Model Derivation

To derive the NBDM model, we start with the assumption that each gene's mRNA
count in the cell follows a Negative Binomial distribution, i.e., for gene
\(g\), the mRNA count \(m_g\) is of the form

\[
m_g \sim \text{NegativeBinomial}(r_g, p),
\tag{1}
\]

where \(r_g\) is the gene-specific dispersion parameter and \(p\) is the
success probability **shared across all genes**. Assuming each mRNA count is
independent, the probability of observing an expression profile

\[
\underline{m} = (m_1, \ldots, m_G),
\tag{2}
\]

where \(G\) is the total number of genes, is given by the product of the
individual probabilities, i.e.,

\[
\pi(\underline{m} \mid \underline{r}, p) =
\prod_{g=1}^G \text{NegativeBinomial}(m_g; r_g, p).
\tag{3}
\]

Although one might question the assumption of a global success probability,
this assumption is less relevant in practice, as the negative binomial
distribution is a sloppy distribution, where changes in the success probability
can be compensated by changes in the dispersion parameter to obtain numerically
equivalent probability mass functions. In other words, the negative binomial is
a highly flexible distribution for which multiple parameter combinations result
in very similar probability mass functions.

Eq. (1) and (3) describe the probability of a cell having a given mRNA count.
However, experimentally, we do not directly observe the mRNA counts, but rather
UMI counts. To model the transformation from mRNA to UMI counts, we assume
that each mRNA molecule in the cell is captured with a probability \(\nu\).
This implies that, conditioned on the mRNA count, the UMI count follows a
binomial distribution, i.e.,

\[
u_g \sim \text{Binomial}(m_g, \nu).
\tag{4}
\]

where \(\nu\) is the capture efficiency shared across all genes and cells.
Nevertheless, since our observable only contains UMI counts, we must remove the
dependency on the mRNA count to obtain a model that is identifiable. To do so,
we can marginalize over the mRNA count, i.e.,

\[
\pi(u_g \mid r_g, p, \nu) = \sum_{m_g = u_g}^\infty \pi(m_g \mid r_g, p)
\pi(u_g \mid m_g, \nu).
\tag{5}
\]

In words, Eq. (5) states that the probability of observing a UMI count
\(u_g\) for gene \(g\) is the sum of the probabilities of observing all
possible mRNA counts \(m_g\) that result in \(u_g\) UMIs. The sum in Eq. (5)
starts at \(m_g = u_g\) because the cell cannot have more UMIs than the number
of mRNA molecules.

It can be shown that Eq. (5) results in a negative binomial distribution with
a re-scaled \(p\) parameter, i.e.,

\[
\pi(u_g \mid r_g, \hat{p}) = \text{NegativeBinomial}(u_g; r_g, \hat{p}),
\tag{6}
\]

where \(\hat{p} = \frac{p}{\nu + (1 - p){\nu}}\). Thus, the joint
distribution of the UMI counts for all genes is given by

\[
\pi(\underline{u} \mid \underline{r}, \hat{p}) =
\prod_{g=1}^G \text{NegativeBinomial}(u_g; r_g, \hat{p}).
\tag{7}
\]

Given these assumptions, we can show that the model in Eq. (7) can be expressed
in a much more compact form, where:

1\. The total number of transcripts in the cell is drawn from a negative
binomial

\[
U \sim \text{NegativeBinomial}(R, p),
\tag{8}
\]

where \(U\) is the total number of UMIs in the cell, and

\[
R = \sum_{g=1}^G r_g,
\tag{9}
\]

is the sum of the dispersion parameters across all genes. Furthermore,

2\. The total count \(U\) is then distributed across all genes via a
Dirichlet-multinomial distribution, i.e.,

\[
\underline{u} \mid U, \underline{r} \sim
\text{DirichletMultinomial}(\underline{u}; U, \underline{r}),
\tag{10}
\]

where \(\underline{r} = (r_1, \ldots, r_G)\) is the vector of dispersion
parameters across all genes.

The significance of this result is that in its derivation, we obtain a natural
normalization scheme for the UMI counts. More specifically, the
Dirichlet-multinomial from Eq. (10) is derived as

\[
\overbrace{
    \pi(\underline{u} \mid U, \underline{r})
}^{\text{Dirichlet-multinomial}} =
\int d^G\underline{\rho} \;
\overbrace{
    \pi(\underline{\rho} \mid U, \underline{r})
}^{\text{Dirichlet}} \;
\overbrace{
    \pi(\underline{u} \mid U, \underline{\rho})
}^{\text{multinomial}},
\tag{11}
\]

where \(\underline{\rho} = (\rho_1, \ldots, \rho_G)\) is the vector of
proportions across all genes that satisfies

\[
\sum_{g=1}^G \rho_g = 1, \; \rho_g \geq 0 \; \forall \; g.
\tag{12}
\]

The derivation above shows that once the total number of UMIs, \(U\), is drawn
(via the negative binomial), the allocation of those UMIs across different
genes follows a Dirichlet-multinomial distribution. Intuitively, this means we
separate how many total UMIs a cell has from how those UMIs are split among its
genes. The Dirichlet-multinomial "naturally normalizes" the data because it
lets us talk about the fraction of the total transcriptome that each gene
constitutes, rather than just raw counts.

Concretely, if you know \(U\), then you can think of a latent "proportion
vector" \(\rho=(\rho_1,\ldots,\rho_G)\), describing what fraction of \(U\)
belongs to each gene. Instead of treating \(\rho\) as fixed, we place a
Dirichlet distribution on it with parameters \(r=(r_1,\ldots,r_G)\). These
gene-specific parameters reflect how variable or overdispersed each gene's
expression tends to be. When you integrate over all possible proportion vectors
\(\rho\), you end up with a Dirichlet-multinomial distribution on the counts
\(u\). In practice, once you infer the posterior distribution of the \(r\)
parameters from data, they become the "shape" parameters of the Dirichlet,
which in turn captures your uncertainty about each gene's fraction of the total
transcriptome. This provides a principled, model-based way of normalizing
single-cell RNA-seq data by explicitly modeling both the total number of UMIs
and the gene-level fractions that compose it.

## Prior Distributions

The model uses the following prior distributions:

For the success probability \(p\):

\[
p \sim \text{Beta}(\alpha_p, \beta_p)
\tag{13}
\]

Default values: \(\alpha_p = \beta_p = 1\) (uniform prior)

For each gene's dispersion parameter \(r_j\):

\[
r_j \sim \text{Gamma}(\alpha_r, \beta_r)
\tag{14}
\]

Default values: \(\alpha_r = 2\), \(\beta_r = 0.1\)

## Variational Inference

The model uses stochastic variational inference with a mean-field variational
family as the approximate posterior. The variational distributions are:

For the success probability \(p\):

\[
q(p) = \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
\tag{15}
\]

For each gene's dispersion parameter \(r_g\):

\[
q(r_g) = \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
\tag{16}
\]

where \(\hat{\alpha}_p\), \(\hat{\beta}_p\), \(\hat{\alpha}_{r,g}\), and
\(\hat{\beta}_{r,g}\) are learnable variational parameters.

The Evidence Lower Bound (ELBO) is:

\[
\mathcal{L} = \mathbb{E}_{q}[\log \pi(u,U,r,p)]
             - \mathbb{E}_{q}[\log q(r,p)]
\tag{17}
\]

where:

- \(\pi(u,U,r,p)\) is the joint probability of the model
- \(q(r,p)\) is the variational distribution

## Learning Algorithm

The model is trained using stochastic variational inference with the following
steps:

1. Initialize variational parameters:
    - \(\hat{\alpha}_p = \alpha_p\), \(\hat{\beta}_p = \beta_p\)
    - \(\hat{\alpha}_{r,g} = \alpha_r\),
      \(\hat{\beta}_{r,g} = \beta_r\) for all genes \(g\)

2. For each iteration:
    - Sample a mini-batch of cells
    - Compute gradients of the ELBO with respect to variational parameters
    - Update parameters (using the Adam optimizer as default)

3. Continue until maximum iterations reached

## Posterior Inference

After training, we can:

1\. Sample from the approximate posterior distributions:

\[
p^{(s)} \sim \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
\tag{18}
\]

\[
r_g^{(s)} \sim \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
\tag{19}
\]

2\. Generate predictive samples:

\[
U^{(s)} \sim \text{NegativeBinomial}(\sum_g r_g^{(s)}, p^{(s)})
\tag{20}
\]

\[
u_g^{(s)} \sim \text{DirichletMultinomial}(r^{(s)}, U^{(s)})
\tag{21}
\]

## Implementation Details

The model is implemented using the NumPyro probabilistic programming framework,
which provides:

- Automatic differentiation for computing ELBO gradients
- Efficient sampling from variational distributions
- Mini-batch support for scalable inference
- GPU acceleration through JAX

## Model Assumptions

The NBDM model makes several key assumptions:

- The total count per cell follows a Negative Binomial distribution
- Given the total count, gene proportions follow a Dirichlet-Multinomial
  distribution
- Gene-specific dispersion parameters capture biological variation
- A single global success probability applies to all cells
- Genes are conditionally independent given the total count

## Usage Considerations

The model is particularly suitable when:

- The data exhibits overdispersion relative to a Poisson model
- The total count per cell varies moderately
- Gene-specific variation needs to be captured

It may be less suitable when:

- Zero-inflation is a dominant feature (consider [ZINB model](zinb.md) instead)
- Cell-specific capture efficiencies vary significantly (consider
  [NBVCP model](nbvcp.md)), reflected on a large variation in the total UMI
  count per cell
- The data contains multiple distinct cell populations (consider
  [mixture models](mixture.md))

## Recap

The NBDM model posits that each cell's total UMI count is governed by a
negative binomial, and gene-level allocations come from a
Dirichlet-multinomial. This captures both how many molecules each cell is
estimated to have and how they are allocated across genes. Together, these
assumptions yield a principled way to "normalize" the data by focusing on
per-cell fractions in a probabilistic framework.

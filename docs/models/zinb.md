# Zero-Inflated Negative Binomial Model (ZINB)

The Zero-Inflated Negative Binomial (ZINB) model extends the standard
[Negative Binomial-Dirichlet Multinomial (NBDM)](nbdm.md) model to handle
excess zeros in single-cell RNA sequencing data. This model is particularly
useful when the data exhibits more zeros than would be expected from a standard
negative binomial distribution alone.

Like the [NBDM model](nbdm.md), the ZINB model captures overdispersion in
molecular counts. However, it differs in two key aspects:

1. It explicitly models technical dropouts via a
   [zero-inflation component](https://en.wikipedia.org/wiki/Zero-inflated_model).
2. It does not use the Dirichlet-multinomial in the likelihood, instead each
   gene is fit independently to a zero-inflated negative binomial distribution.

For details on overdispersion and the basic negative binomial component, please
refer to the [NBDM model](nbdm.md).

## Model Comparison with NBDM

In the [NBDM model](nbdm.md), we focus on two key aspects:

1. How many total transcripts a cell has (drawn from a Negative Binomial)
2. How those transcripts are split among genes (captured by a
   Dirichlet-Multinomial)

When normalizing single-cell data by focusing on fractions of the
transcriptome, the Dirichlet-multinomial machinery allows us to think of
\(\rho_g\) as the fraction of transcripts going to each gene \(g\).

The Zero-Inflated Negative Binomial (ZINB) model adds an extra "dropout" or
"technical zero" component to account for unobserved transcripts. If these
extra zeros are purely technical — i.e., they do not change the true underlying
fraction of transcripts that each gene contributes but are instead due to
technical limitations when mapping mRNA molecules to UMI counts — then it's
valid to ignore the zero-inflation part, allowing us to use the same
\(\rho_g\) parameters for fraction-based normalization. The model has two key
components:

1. **Dropout Layer (Technical Zeros)**: Some fraction of transcripts is "lost"
   and recorded as zero for purely technical reasons.
2. **Underlying Gene Counts**: Conditioned on not being dropped out, the gene's
   counts still follow a Negative Binomial with parameters \(r_g\) and \(p\).

If you strip away the dropout events, the core distribution for each gene's
true expression is the same
[Negative Binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution)
as before. Consequently, you can still think of
\(\underline{r} = \{r_g\}_{g=1}^G\) as capturing the gene-specific
overdispersion for the "real" expression levels, just like in the
[NBDM model](nbdm.md).

From a normalization perspective, the key question becomes: "*Among all the
transcripts that would have been observed if there were no technical dropouts,
what fraction goes to each gene?*" If dropout is treated as a purely technical
artifact that does not alter the underlying composition, then that fraction is
governed by the same \(\underline{r}\) parameters. In other words, the dropout
layer is separate — it explains missing observations rather than redefining the
overall fraction each gene represents in the cell.

Therefore, if all zero-inflation is assumed to be technical, the dropout
component can be effectively ignored for fraction-based normalization. The
\(\{r_g\}\) parameters remain the key to describing each gene's share of the
total expression, just as in the [NBDM model](nbdm.md).

## Model Structure

The ZINB model follows a hierarchical structure where:

1. Each gene has an associated dropout probability (`gate`)
2. For genes that aren't dropped out, counts follow a negative binomial
   distribution
3. The model handles each gene independently, with shared success probability
   across genes

Formally, for a dataset with \(N\) cells and \(G\) genes, let
\(u_{g}^{(c)}\) be the UMI count for gene \(g\) in cell \(c\). The
generative process is:

1. Draw global success probability:
   \(p \sim \text{Beta}(\alpha_p, \beta_p)\)
2. For each gene \(g = 1,\ldots,G\):
    - Draw dispersion parameter:
      \(r_g \sim \text{Gamma}(\alpha_r, \beta_r)\)
    - Draw dropout probability:
      \(\pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})\)
3. For each cell \(c = 1,\ldots,N\) and gene \(g = 1,\ldots,G\):
    - Draw dropout indicator:
      \(z_g^{(c)} \sim \text{Bernoulli}(\pi_g)\)
    - If \(z_g^{(c)} = 1\): set \(u_g^{(c)} = 0\)
    - If \(z_g^{(c)} = 0\): draw
      \(u_g^{(c)} \sim \text{NegativeBinomial}(r_g, p)\)

## Model Derivation

The ZINB model combines a Bernoulli distribution for dropout events with a
negative binomial for the actual counts. For each gene \(g\) and cell \(c\),
the probability of observing a count \(u_g^{(c)}\) is:

\[
\pi(u_g^{(c)} \mid \pi_g, r_g, p) =
\pi_g \delta_{0}(u_g^{(c)}) + (1-\pi_g)
\text{NegativeBinomial}(u_g^{(c)}; r_g, p),
\tag{1}
\]

where:

- \(\pi_g\) is the dropout probability for gene \(g\)
- \(\delta_{0}(x)\) is the Dirac delta function at zero
- \(r_g\) is the gene-specific dispersion parameter
- \(p\) is the shared success probability

Unlike the [NBDM model](nbdm.md), each gene is fit to an independent
zero-inflated negative binomial. The joint probability across all genes and
cells is simply:

\[
\pi(\underline{\underline{U}} \mid \underline{\pi}, \underline{r}, p) =
\prod_{c=1}^N \prod_{g=1}^G \pi_g \delta_{0}(u_g^{(c)}) + (1-\pi_g)
\text{NegativeBinomial}(u_g^{(c)}; r_g, p)
\tag{2}
\]

where:

- \(\underline{\underline{U}}\) is the complete count matrix
- \(\underline{\pi}\) is the vector of dropout probabilities
- \(\underline{r}\) is the vector of dispersion parameters

## Prior Distributions

The model uses the following prior distributions:

For the success probability \(p\):

\[
p \sim \text{Beta}(\alpha_p, \beta_p)
\tag{3}
\]

For each gene's dispersion parameter \(r_g\):

\[
r_g \sim \text{Gamma}(\alpha_r, \beta_r)
\tag{4}
\]

For each gene's dropout probability \(\pi_g\):

\[
\pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})
\tag{5}
\]

## Variational Posterior Distribution

The model uses stochastic variational inference with a mean-field variational
family. The variational distributions are:

For the success probability \(p\):

\[
q(p) = \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
\tag{6}
\]

For each gene's dispersion parameter \(r_g\):

\[
q(r_g) = \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
\tag{7}
\]

For each gene's dropout probability \(\pi_g\):

\[
q(\pi_g) = \text{Beta}(\hat{\alpha}_{\pi,g}, \hat{\beta}_{\pi,g})
\tag{8}
\]

where hatted parameters are learnable variational parameters.

## Learning Algorithm

The training process follows the same steps as the [NBDM model](nbdm.md):

1. Initialize variational parameters
2. For each iteration:
    - Sample mini-batch of cells
    - Compute ELBO gradients
    - Update parameters (using Adam optimizer as default)
3. Continue until maximum number of iterations is reached

The key difference is that we now also track and update parameters for the
dropout probabilities.

## Implementation Details

Like the [NBDM model](nbdm.md), the ZINB model is implemented using
[NumPyro](https://num.pyro.ai/). The key additions are:

- Zero-inflated distributions using NumPyro's
  [ZeroInflatedDistribution](https://num.pyro.ai/en/stable/distributions.html#zeroinflateddistribution)
- Additional variational parameters for dropout probabilities
- Independent fitting of genes (no Dirichlet-Multinomial component)

## Model Assumptions

The ZINB model makes several key assumptions:

- Zeros can arise from two processes:
    - Technical dropouts (modeled by zero-inflation)
    - Biological absence of expression (modeled by negative binomial)
- Genes are independent
- A single global success probability applies to all cells
- Each gene has its own dropout probability and dispersion parameter

## Usage Considerations

The ZINB model is particularly suitable when:

- The data exhibits excessive zeros beyond what a negative binomial predicts
- You need to distinguish technical dropouts from biological zeros
- Genes can be reasonably modeled independently

It may be less suitable when:

- Library size variation is a major concern (consider
  [NBVCP model](nbvcp.md))
- Cell-specific capture efficiencies vary significantly (consider
  [ZINBVCP model](zinbvcp.md))
- The data contains multiple distinct cell populations (consider
  [mixture models](mixture.md))

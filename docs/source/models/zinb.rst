Zero-Inflated Negative Binomial Model
=====================================

The Zero-Inflated Negative Binomial (ZINB) model extends the standard negative
binomial model to handle excess zeros in single-cell RNA sequencing data. This
model is particularly useful when the data exhibits more zeros than would be
expected from a standard negative binomial distribution alone.

Like the NBDM model, the ZINB model captures overdispersion in molecular counts.
However, it differs in two key aspects:

1. It explicitly models technical dropouts via a `zero-inflation component
   <https://en.wikipedia.org/wiki/Zero-inflated_model>`_.
2. It does not use the Dirichlet-multinomial for normalization, instead modeling
   each gene independently

For details on overdispersion and the basic negative binomial component, please
refer to the :doc:`NBDM model <nbdm>`.

Model Comparison with :doc:`NBDM <nbdm>`
--------------------------

In the :doc:`Negative Binomial-Dirichlet Multinomial (NBDM) model <nbdm>`, we
focus on two key aspects:

1. How many total transcripts a cell has (drawn from a Negative Binomial)
2. How those transcripts are split among genes (captured by a
   Dirichlet-multinomial)

When normalizing single-cell data by focusing on fractions of the transcriptome,
the Dirichlet-multinomial machinery allows us to think of :math:`\rho_g` as the
fraction of transcripts going to each gene :math:`g`.

The Zero-Inflated Negative Binomial (ZINB) model adds an extra "dropout" or
"technical zero" component to account for unobserved transcripts. If these extra
zeros are purely technical—i.e., they do not change the true underlying fraction
of transcripts that each gene contributes—then it's valid to ignore the
zero-inflation part when performing fraction-based normalization. The model has
two key components:

1. **Dropout Layer (Technical Zeros)**: Some fraction of transcripts is "lost"
   and recorded as zero for purely technical reasons.
2. **Underlying Gene Counts**: Conditioned on not being dropped out, the gene's
   counts still follow a Negative Binomial with parameters :math:`r_g` and
   :math:`p`.

If you strip away the dropout events, the core distribution for each gene's true
expression is the same Negative Binomial as before. Consequently, you can still
think of :math:`\{r_g\}` as capturing the gene-specific overdispersion for the
"real" expression levels, just like in the NBDM model.

From a normalization perspective, the key question becomes: "Among all the
transcripts that would have been observed if there were no technical dropouts,
what fraction goes to each gene?" If dropout is treated as a purely technical
artifact that does not alter the underlying composition, then that fraction is
governed by the same :math:`\{r_g\}` parameters. In other words, the dropout
layer is separate—it explains missing observations rather than redefining the
overall fraction each gene represents in the cell.

Therefore, if all zero-inflation is assumed to be technical, the dropout
component can be effectively ignored for fraction-based normalization. The
:math:`\{r_g\}` parameters remain the key to describing each gene's share of the
total expression, just as in the :doc:`NBDM model <nbdm>`.

Model Structure
---------------

The ZINB model follows a hierarchical structure where:

1. Each gene has an associated dropout probability (gate)
2. For genes that aren't dropped out, counts follow a negative binomial
   distribution
3. The model handles each gene independently, with shared success probability
   across genes

Formally, for a dataset with :math:`N` cells and :math:`G` genes, let
:math:`u_{g}^{(c)}` be the UMI count for gene :math:`g` in cell :math:`c`. The
generative process is:

1. Draw global success probability: :math:`p \sim \text{Beta}(\alpha_p,
   \beta_p)`
2. For each gene :math:`g = 1,\ldots,G`:
   
   * Draw dispersion parameter: :math:`r_g \sim \text{Gamma}(\alpha_r, \beta_r)`
   * Draw dropout probability: :math:`\pi_g \sim \text{Beta}(\alpha_{\pi},
     \beta_{\pi})`

3. For each cell :math:`c = 1,\ldots,N` and gene :math:`g = 1,\ldots,G`:
   
   * Draw dropout indicator: :math:`z_g^{(c)} \sim \text{Bernoulli}(\pi_g)`
   * If :math:`z_g^{(c)} = 1`: set :math:`u_g^{(c)} = 0`
   * If :math:`z_g^{(c)} = 0`: draw :math:`u_g^{(c)} \sim
     \text{NegativeBinomial}(r_g, p)`

Model Derivation
---------------

The ZINB model combines a Bernoulli distribution for dropout events with a
negative binomial for the actual counts. For each gene :math:`g` and cell
:math:`c`, the probability of observing a count :math:`u_g^{(c)}` is:

.. math::
   P(u_g^{(c)} \mid \pi_g, r_g, p) = 
   \pi_g \delta_{0}(u_g^{(c)}) + (1-\pi_g)
   \text{NegativeBinomial}(u_g^{(c)}; r_g, p),
   \tag{1}

where:
- :math:`\pi_g` is the dropout probability for gene :math:`g`
- :math:`\delta_{0}(x)` is the Dirac delta function at zero
- :math:`r_g` is the gene-specific dispersion parameter
- :math:`p` is the shared success probability

The negative binomial component follows the same form as in the NBDM model:

.. math::
   \text{NegativeBinomial}(u; r, p) = \binom{u+r-1}{u}p^r(1-p)^u
   \tag{2}

Unlike the NBDM model, each gene is modeled independently, meaning the joint
probability across all genes and cells is simply:

.. math::
   P(\mathbf{U} \mid \boldsymbol{\pi}, \mathbf{r}, p) = \prod_{c=1}^N \prod_{g=1}^G P(u_g^{(c)} \mid \pi_g, r_g, p)
   \tag{3}

where :math:`\mathbf{U}` is the complete count matrix, :math:`\boldsymbol{\pi}`
is the vector of dropout probabilities, and :math:`\mathbf{r}` is the vector of
dispersion parameters.

Prior Distributions
------------------

The model uses the following prior distributions:

For the success probability :math:`p`:

.. math::
   p \sim \text{Beta}(\alpha_p, \beta_p)
   \tag{4}

Default values: :math:`\alpha_p = \beta_p = 1` (uniform prior)

For each gene's dispersion parameter :math:`r_g`:

.. math::
   r_g \sim \text{Gamma}(\alpha_r, \beta_r)
   \tag{5}

Default values: :math:`\alpha_r = 2`, :math:`\beta_r = 0.1`

For each gene's dropout probability :math:`\pi_g`:

.. math::
   \pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})
   \tag{6}

Default values: :math:`\alpha_{\pi} = \beta_{\pi} = 1` (uniform prior)

Variational Inference
--------------------

The model uses stochastic variational inference with a mean-field variational
family. The variational distributions are:

For the success probability :math:`p`:

.. math::
   q(p) = \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
   \tag{7}

For each gene's dispersion parameter :math:`r_g`:

.. math::
   q(r_g) = \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
   \tag{8}

For each gene's dropout probability :math:`\pi_g`:

.. math::
   q(\pi_g) = \text{Beta}(\hat{\alpha}_{\pi,g}, \hat{\beta}_{\pi,g})
   \tag{9}

where hatted parameters are learnable variational parameters.

The Evidence Lower Bound (ELBO) follows the same form as the NBDM model but
includes the additional zero-inflation components:

.. math::
   \mathcal{L} = 
   \mathbb{E}_{q}[\log P(\mathbf{U}, \mathbf{r}, \boldsymbol{\pi}, p)] - 
   \mathbb{E}_{q}[\log q(\mathbf{r}, \boldsymbol{\pi}, p)]
   \tag{10}

Learning Algorithm
----------------

The training process follows the same steps as the NBDM model:

1. Initialize variational parameters
2. For each iteration:
   * Sample mini-batch of cells
   * Compute ELBO gradients
   * Update parameters using Adam optimizer
3. Continue until convergence

The key difference is that we now also track and update parameters for the
dropout probabilities.

Posterior Inference
-----------------

After training, we can sample from the approximate posterior:

.. math::
   p^{(s)} \sim \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
   \tag{11}

.. math::
   r_g^{(s)} \sim \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
   \tag{12}

.. math::
   \pi_g^{(s)} \sim \text{Beta}(\hat{\alpha}_{\pi,g}, \hat{\beta}_{\pi,g})
   \tag{13}

And generate predictive samples:

.. math::
   z_g^{(s)} \sim \text{Bernoulli}(\pi_g^{(s)})
   \tag{14}

.. math::
   u_g^{(s)} \sim \begin{cases} 
   0 & \text{if } z_g^{(s)} = 1 \\
   \text{NegativeBinomial}(r_g^{(s)}, p^{(s)}) & \text{if } z_g^{(s)} = 0
   \end{cases}
   \tag{15}

Implementation Details
--------------------

Like the NBDM model, the ZINB model is implemented using NumPyro. The key
additions are:

* Zero-inflated distributions using NumPyro's `ZeroInflatedDistribution`
* Additional variational parameters for dropout probabilities
* Independent modeling of genes (no Dirichlet-multinomial component)

Model Assumptions
---------------

The ZINB model makes several key assumptions:

* Zeros can arise from two processes:
  - Technical dropouts (modeled by zero-inflation)
  - Biological absence of expression (modeled by negative binomial)
* Genes are completely independent
* A single global success probability applies to all cells
* Each gene has its own dropout probability and dispersion parameter

Usage Considerations
------------------

The ZINB model is particularly suitable when:

* The data exhibits excessive zeros beyond what a negative binomial predicts
* You need to distinguish technical dropouts from biological zeros
* Genes can be reasonably modeled independently

It may be less suitable when:

* Library size variation is a major concern (consider NBDM model)
* Cell-specific capture efficiencies vary significantly (consider ZINBVCP model)
* The data contains multiple distinct cell populations (consider mixture models)

Practical Tips
-------------

1. **Initialization**
   * For sparse datasets, consider increasing :math:`\alpha_{\pi}` relative to
   :math:`\beta_{\pi}`
   * For datasets with few zeros, do the opposite

2. **Model Selection**
   * Compare ZINB vs NBDM using WAIC or other model comparison metrics
   * Check if zero-inflation improves fit using posterior predictive checks

3. **Computational Considerations**
   * Independent modeling of genes allows for easy parallelization
   * Memory requirements scale linearly with number of genes
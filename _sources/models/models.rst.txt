SCRIBE Models for Single-Cell RNA Sequencing
============================================

``SCRIBE`` provides a suite of probabilistic models designed to capture key
characteristics of single-cell RNA sequencing data. All models share a common
foundation but address different aspects of technical and biological variation.
This document explains the mathematical formulation, assumptions, and
implementation details of each model to help you select the most appropriate one
for your analysis.

.. _nbdm-model:

Core Model: Negative Binomial-Dirichlet Multinomial (NBDM)
----------------------------------------------------------

The NBDM model forms the foundation of ``SCRIBE``'s approach, capturing three
essential aspects of scRNA-seq data:

1. The overdispersed nature of molecular counts
2. The compositional aspect of sequencing data
3. Gene-specific variation in expression levels

Unlike standard scRNA-seq pipelines that use library-size normalization
(dividing by total UMI count, then log-transform), the NBDM model integrates
normalization into the generative process via the Dirichlet-multinomial. This
provides a more principled measure of uncertainty and helps separate technical
from biological variation.

Model Structure
^^^^^^^^^^^^^^^

The NBDM model follows a hierarchical structure where:

Each cell's total molecular count follows a `Negative Binomial distribution
<https://en.wikipedia.org/wiki/Negative_binomial_distribution>`_. Given the
total count, the allocation of molecules across genes follows a
`Dirichlet-Multinomial distribution
<https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution>`_.

Formally, for a dataset with :math:`N` cells and :math:`G` genes, let
:math:`u_{g}^{(c)}` be the UMI count for gene :math:`g` in cell :math:`c`, and
:math:`U^{(c)} = \sum_{g=1}^G u_{g}^{(c)}` be the total UMI count for cell
:math:`c`. The generative process is:

1. Draw global success probability: :math:`p \sim \text{Beta}(\alpha_p, \beta_p)`
2. Draw gene-specific dispersion parameters: :math:`r_g \sim
   \text{Gamma}(\alpha_r, \beta_r)` or :math:`r_g \sim \text{LogNormal}(\mu_r,
   \sigma_r)` for :math:`g = 1,\ldots,G`
3. For each cell :math:`c = 1,\ldots,N`:

   * Draw total count: :math:`U^{(c)} \sim \text{NegativeBinomial}(\sum_g r_g, p)`
   * Draw gene proportions: :math:`u^{(c)} \sim \text{DirichletMultinomial}(r, U^{(c)})`

Model Derivation
^^^^^^^^^^^^^^^^

To derive the NBDM model, we start with the assumption that each gene's mRNA
count in the cell follows a Negative Binomial distribution, i.e., for gene
:math:`g`, the mRNA count :math:`m_g` is of the form

.. math::
   m_g \sim \text{NegativeBinomial}(r_g, p),
   \tag{1}

where :math:`r_g` is the gene-specific dispersion parameter and :math:`p` is the
success probability **shared across all genes**. Assuming each mRNA count is
independent, the probability of observing an expression profile

.. math::
   \underline{m} = (m_1, \ldots, m_G),
   \tag{2}

where :math:`G` is the total number of genes, is given by the product of the
individual probabilities, i.e.,

.. math::
   \pi(\underline{m} \mid \underline{r}, p) = 
   \prod_{g=1}^G \text{NegativeBinomial}(m_g; r_g, p).
   \tag{3}

Although one might question the assumption of a global success probability, this
assumption is less relevant in practice, as the negative binomial distribution
is a sloppy distribution, where changes in the success probability can be
compensated by changes in the dispersion parameter to obtain equivalent
numerically equivalent probability mass functions. In other words, the negative
binomial is a highly flexible distribution for which multiple parameter
combinations result in very similar probability mass functions.

Eq. (1) and (3) describe the probability of a cell having a given mRNA count.
However, experimentally, we do not directly observe the mRNA counts, but rather
UMI counts. To model the transformation from mRNA to UMI counts, we assume that
each mRNA molecule in the cell is captured with a probability :math:`\nu`. This
implies that, conditioned on the mRNA count, the UMI count follows a binomial
distribution, i.e.,

.. math::
   u_g \sim \text{Binomial}(m_g, \nu).
   \tag{4}

where :math:`\nu` is the capture efficiency shared across all genes and cells.
Nevertheless, since our observable only contains UMI counts, we must remove the
dependency on the mRNA count to obtain a model that is identifiable. To do so,
we can marginalize over the mRNA count, i.e.,

.. math::
   \pi(u_g \mid r_g, p, \nu) = \sum_{m_g = u_g}^\infty \pi(m_g \mid r_g, p) 
   \pi(u_g \mid m_g, \nu).
   \tag{5}

In words, Eq. (5) states that the probability of observing a UMI count
:math:`u_g` for gene :math:`g` is the sum of the probabilities of observing all
possible mRNA counts :math:`m_g` that result in :math:`u_g` UMIs. The sum in Eq.
(5) starts at :math:`m_g = u_g` because, the cell cannot have more UMIs than the
number of mRNA molecules.

It can be shown that Eq. (5) results in a negative binomial distribution with a
re-scaled :math:`p` parameter, i.e.,

.. math::
   \pi(u_g \mid r_g, \hat{p}) = \text{NegativeBinomial}(u_g; r_g, \hat{p}),
   \tag{6}

where :math:`\hat{p} = \frac{p \nu}{1 - p(1 - \nu)}`. Thus, the joint
distribution of the UMI counts for all genes is given by

.. math::
   \pi(\underline{u} \mid \underline{r}, \hat{p}) = 
   \prod_{g=1}^G \text{NegativeBinomial}(u_g; r_g, \hat{p}).
   \tag{7}

Given these assumptions, we can show that the model in Eq. (7) can be expressed
in a much more compact form, where:

1. The total number of transcripts in the cell is drawn from a negative binomial

.. math::
   U \sim \text{NegativeBinomial}(R, p),
   \tag{8}

where :math:`U` is the total number of UMIs in the cell, and

.. math::
   R = \sum_{g=1}^G r_g,
   \tag{9}

is the sum of the dispersion parameters across all genes. Furthermore,

2. The total count :math:`U` is then distributed across all genes via a
   Dirichlet-multinomial distribution, i.e.,

.. math::
   \underline{u} \mid U, \underline{r}, \sim 
   \text{DirichletMultinomial}(\underline{u}; U, \underline{r}),
   \tag{10}

where :math:`\underline{r} = (r_1, \ldots, r_G)` is the vector of dispersion
parameters across all genes.

The significance of this result is that in its derivation, we obtain a natural
normalization scheme for the UMI counts. More specifically, the
Dirichlet-multinomial from Eq. (10) is derived as

.. math::
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

where :math:`\underline{\rho} = (\rho_1, \ldots, \rho_G)` is the vector of
proportions across all genes that satisfies

.. math::
   \sum_{g=1}^G \rho_g = 1, \; \rho_g \geq 0 \; \forall \; g.
   \tag{12}

The derivation above shows that once the total number of UMIs, :math:`U`, is
drawn (via the negative binomial), the allocation of those UMIs across different
genes follows a Dirichlet-multinomial distribution. Intuitively, this means we
separate how many total UMIs a cell has from how those UMIs are split among its
genes. The Dirichlet-multinomial "naturally normalizes" the data because it lets
us talk about the fraction of the total transcriptome that each gene
constitutes, rather than just raw counts.

Concretely, if you know :math:`U`, then you can think of a latent "proportion
vector" :math:`\rho=(\rho_1,\ldots,\rho_G)`, describing what fraction of
:math:`U` belongs to each gene. Instead of treating :math:`\rho` as fixed, we
place a Dirichlet distribution on it with parameters :math:`r=(r_1,\ldots,r_G)`.
These gene-specific parameters reflect how variable or overdispersed each gene's
expression tends to be. When you integrate over all possible proportion vectors
:math:`\rho`, you end up with a Dirichlet-multinomial distribution on the counts
:math:`u`. In practice, once you infer the posterior distribution of the
:math:`r` parameters from data, they become the "shape" parameters of the
Dirichlet, which in turn captures your uncertainty about each gene's fraction of
the total transcriptome. This provides a principled, model-based way of
normalizing single-cell RNA-seq data by explicitly modeling both the total
number of UMIs and the gene-level fractions that compose it.

.. _zinb-model:

Zero-Inflated Negative Binomial Model (ZINB)
--------------------------------------------

The Zero-Inflated Negative Binomial (ZINB) model extends the standard NBDM model
to handle excess zeros in single-cell RNA sequencing data. This model is
particularly useful when the data exhibits more zeros than would be expected
from a standard negative binomial distribution alone.

Like the NBDM model, the ZINB model captures overdispersion in molecular counts.
However, it differs in two key aspects:

1. It explicitly models technical dropouts via a `zero-inflation component
   <https://en.wikipedia.org/wiki/Zero-inflated_model>`_.
2. It does not use the Dirichlet-multinomial in the likelihood, instead each
   gene is fit independently to a zero-inflated negative binomial distribution.

Model Comparison with NBDM
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NBDM model, we focus on two key aspects:

1. How many total transcripts a cell has (drawn from a Negative Binomial)
2. How those transcripts are split among genes (captured by a Dirichlet-Multinomial)

When normalizing single-cell data by focusing on fractions of the transcriptome,
the Dirichlet-multinomial machinery allows us to think of :math:`\rho_g` as the
fraction of transcripts going to each gene :math:`g`.

The Zero-Inflated Negative Binomial (ZINB) model adds an extra "dropout" or
"technical zero" component to account for unobserved transcripts. If these extra
zeros are purely technical—i.e., they do not change the true underlying fraction
of transcripts that each gene contributes but are instead due to technical
limitations when mapping mRNA molecules to UMI counts—then it's valid to ignore
the zero-inflation part, allowing us to use the same :math:`\rho_g` parameters
for fraction-based normalization. The model has two key components:

1. **Dropout probability (Technical Zeros)**: Some fraction of transcripts is
   "lost" and recorded as zero for purely technical reasons.
2. **Underlying Gene Counts**: Conditioned on not being dropped out, the gene's
   counts still follow a Negative Binomial with parameters :math:`r_g` and
   :math:`p`.

If you strip away the dropout events, the core distribution for each gene's true
expression is the same `Negative Binomial
<https://en.wikipedia.org/wiki/Negative_binomial_distribution>`_ as before.
Consequently, you can still think of :math:`\underline{r} = \{r_g\}_{g=1}^G` as
capturing the gene-specific overdispersion for the "real" expression levels,
just like in the NBDM model.

From a normalization perspective, the key question becomes: "*Among all the
transcripts that would have been observed if there were no technical dropouts,
what fraction goes to each gene?*" If dropout is treated as a purely technical
artifact that does not alter the underlying composition, then that fraction is
governed by the same :math:`\underline{r}` parameters. In other words, the
dropout layer is separate—it explains missing observations rather than
redefining the overall fraction each gene represents in the cell.

Model Structure
^^^^^^^^^^^^^^^

The ZINB model follows a hierarchical structure where:

1. Each gene has an associated dropout probability (`gate`)
2. For genes that aren't dropped out, counts follow a negative binomial distribution
3. The model handles each gene independently, with shared success probability across genes

Formally, for a dataset with :math:`N` cells and :math:`G` genes, let
:math:`u_{g}^{(c)}` be the UMI count for gene :math:`g` in cell :math:`c`. The
generative process is:

1. Draw global success probability: :math:`p \sim \text{Beta}(\alpha_p, \beta_p)`
2. For each gene :math:`g = 1,\ldots,G`:
   
   * Draw dispersion parameter: :math:`r_g \sim \text{Gamma}(\alpha_r, \beta_r)`
     or :math:`r_g \sim \text{LogNormal}(\mu_r, \sigma_r)`
   * Draw dropout probability: :math:`\pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})`

3. For each cell :math:`c = 1,\ldots,N` and gene :math:`g = 1,\ldots,G`:
   
   * Draw dropout indicator: :math:`z_g^{(c)} \sim \text{Bernoulli}(\pi_g)`
   * If :math:`z_g^{(c)} = 1`: set :math:`u_g^{(c)} = 0`
   * If :math:`z_g^{(c)} = 0`: draw :math:`u_g^{(c)} \sim \text{NegativeBinomial}(r_g, p)`

Model Derivation
^^^^^^^^^^^^^^^^

The ZINB model combines a Bernoulli distribution for dropout events with a
negative binomial for the actual counts. For each gene :math:`g` and cell
:math:`c`, the probability of observing a count :math:`u_g^{(c)}` is:

.. math::
   \pi(u_g^{(c)} \mid \pi_g, r_g, p) = 
   \pi_g \delta_{0}(u_g^{(c)}) + (1-\pi_g)
   \text{NegativeBinomial}(u_g^{(c)}; r_g, p),
   \tag{13}

where:

* :math:`\pi_g` is the dropout probability for gene :math:`g`
* :math:`\delta_{0}(x)` is the Dirac delta function at zero
* :math:`r_g` is the gene-specific dispersion parameter
* :math:`p` is the shared success probability

Unlike the NBDM model, each gene is fit to an independent zero-inflated negative
binomial. The joint probability across all genes and cells is simply:

.. math::
   \pi(\underline{\underline{U}} \mid \underline{\pi}, \underline{r}, p) = 
   \prod_{c=1}^N \prod_{g=1}^G \pi_g \delta_{0}(u_g^{(c)}) + (1-\pi_g)
   \text{NegativeBinomial}(u_g^{(c)}; r_g, p)
   \tag{14}

where:

* :math:`\underline{\underline{U}}` is the complete count matrix
* :math:`\underline{\pi}` is the vector of dropout probabilities
* :math:`\underline{r}` is the vector of dispersion parameters

.. _nbvcp-model:

Negative Binomial with Variable Capture Probability Model (NBVCP)
-----------------------------------------------------------------

The Negative Binomial with Variable Capture Probability (NBVCP) model extends
the NBDM model by explicitly modeling cell-specific capture efficiencies. This
model is particularly useful when cells exhibit significant variation in their
total UMI counts, which may indicate differences in mRNA capture rates rather
than true biological differences in expression.

Like the NBDM model, the NBVCP model captures overdispersion in molecular
counts. However, it differs in two key aspects:

1. It explicitly models cell-specific capture probabilities that modify the
   success probability of the negative binomial
2. It does not use the Dirichlet-multinomial in the likelihood, instead treating
   each gene independently

Model Comparison with NBDM
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the NBDM model, we assume a single success probability :math:`p` that is
shared across all cells. The NBVCP model relaxes this assumption by introducing
cell-specific capture probabilities that modify how the base success probability
manifests in each cell.

The key insight is that variations in capture efficiency can make the same
underlying mRNA abundance appear different in the UMI counts. The NBVCP model
handles this by:

1. Maintaining a base success probability :math:`p` that represents the "true"
   biological probability
2. Introducing cell-specific capture probabilities :math:`\nu^{(c)}` that modify
   this base probability
3. Computing an effective success probability for each cell :math:`c` as:

.. math::
   \hat{p}^{(c)} = \frac{p \nu^{(c)}}{1 - p (1 - \nu^{(c)})}
   \tag{15}

This adjusted probability accounts for how the capture efficiency of each cell
affects our ability to observe its true mRNA content.

Given the explicit modeling of the cell-specific capture probabilities, the
NBVCP model can remove technical variability, allowing for the same
normalization methods as the NBDM model based on the Dirichlet-Multinomial
model. In other words, since the NBVCP model fits a parameter to account for
significant technical variations in the total number of counts per cell, once
this effect is removed, the remaining variation can be modeled using the same
:math:`\underline{r}` parameters as the NBDM model. Thus, the NBVCP model
presents a more principled approach to normalization compared to other methods
in the scRNA-seq literature.

Model Structure
^^^^^^^^^^^^^^^

The NBVCP model follows a hierarchical structure where:

1. Each cell has an associated capture probability :math:`\nu^{(c)}`
2. The base success probability :math:`p` is modified by each cell's capture
   probability to give an effective success probability :math:`\hat{p}^{(c)}`
3. Gene counts follow independent negative binomial distributions with
   cell-specific effective probabilities

Formally, for a dataset with :math:`N` cells and :math:`G` genes, let
:math:`u_{g}^{(c)}` be the UMI count for gene :math:`g` in cell :math:`c`. The
generative process is:

1. Draw global success probability: :math:`p \sim \text{Beta}(\alpha_p, \beta_p)`
2. Draw gene-specific dispersion parameters: :math:`r_g \sim
   \text{Gamma}(\alpha_r, \beta_r)` or :math:`r_g \sim \text{LogNormal}(\mu_r,
   \sigma_r)` for :math:`g = 1,\ldots,G`
3. For each cell :math:`c = 1,\ldots,N`:
   
   * Draw capture probability: :math:`\nu^{(c)} \sim \text{Beta}(\alpha_{\nu}, \beta_{\nu})`
   * Compute effective probability: :math:`\hat{p}^{(c)} = \frac{p \nu^{(c)}}{1 - p (1 - \nu^{(c)})}`
   * For each gene :math:`g = 1,\ldots,G`: - Draw count: :math:`u_g^{(c)} \sim \text{NegativeBinomial}(r_g, \hat{p}^{(c)})`

Model Derivation
^^^^^^^^^^^^^^^^

The NBVCP model can be derived by considering how the mRNA capture efficiency
affects the observed UMI counts. Starting with the standard negative binomial
model for mRNA counts:

.. math::
   m_g^{(c)} \sim \text{NegativeBinomial}(r_g, p),
   \tag{16}

where :math:`m_g^{(c)}` is the unobserved mRNA count for gene :math:`g` in cell
:math:`c`, :math:`r_g` is the dispersion parameter, and :math:`p` is the base
success probability shared across all cells. We then model the capture process
as a binomial sampling where each mRNA molecule has probability
:math:`\nu^{(c)}` of being captured:

.. math::
   u_g^{(c)} \mid m_g^{(c)} \sim \text{Binomial}(m_g^{(c)}, \nu^{(c)})
   \tag{17}

Marginalizing over the unobserved mRNA counts :math:`m_g^{(c)}`, we get:

.. math::
   u_g^{(c)} \sim \text{NegativeBinomial}(r_g, \hat{p}^{(c)})
   \tag{18}

where :math:`\hat{p}^{(c)}` is the effective probability defined in Eq. (15).

.. _zinbvcp-model:

Zero-Inflated Negative Binomial with Variable Capture Probability Model (ZINBVCP)
---------------------------------------------------------------------------------

The Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP)
model combines aspects of both the ZINB and NBVCP models to handle both
technical dropouts and variable capture efficiencies in single-cell RNA
sequencing data. This model is particularly useful when the data exhibits both
excess zeros and significant variation in total UMI counts across cells.

The ZINBVCP model incorporates two key features:

1. Zero-inflation to model technical dropouts (from ZINB)
2. Cell-specific capture probabilities (from NBVCP)

Model Comparison with NBVCP and ZINB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ZINBVCP model extends both the NBVCP and ZINB models by combining their key
features. From the NBVCP model, it inherits the cell-specific capture
probabilities :math:`\nu^{(c)}` that modify the base success probability
:math:`p`. From the ZINB model, it inherits the gene-specific dropout
probabilities :math:`\pi_g` that model technical zeros.

The effective success probability for each cell :math:`c` is computed as shown
in Eq. (15). This is then combined with the dropout mechanism to give a
zero-inflated distribution where the non-zero counts use the cell-specific
effective probability.

Model Structure
^^^^^^^^^^^^^^

The ZINBVCP model follows a hierarchical structure where:

1. Each gene has an associated dropout probability :math:`\pi_g`
2. Each cell has an associated capture probability :math:`\nu^{(c)}`
3. The base success probability :math:`p` is modified by each cell's capture probability
4. For genes that aren't dropped out, counts follow a negative binomial with cell-specific effective probabilities

Formally, for a dataset with :math:`N` cells and :math:`G` genes, let
:math:`u_{g}^{(c)}` be the UMI count for gene :math:`g` in cell :math:`c`. The
generative process is:

1. Draw global success probability: :math:`p \sim \text{Beta}(\alpha_p, \beta_p)`
2. For each gene :math:`g = 1,\ldots,G`:
   
   * Draw dispersion parameter: :math:`r_g \sim \text{Gamma}(\alpha_r, \beta_r)`
     or :math:`r_g \sim \text{LogNormal}(\mu_r, \sigma_r)`
   * Draw dropout probability: :math:`\pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})`

3. For each cell :math:`c = 1,\ldots,N`:
   
   * Draw capture probability: :math:`\nu^{(c)} \sim \text{Beta}(\alpha_{\nu}, \beta_{\nu})`
   * Compute effective probability: :math:`\hat{p}^{(c)} = \frac{p \nu^{(c)}}{1 - p (1 - \nu^{(c)})}`
   * For each gene :math:`g = 1,\ldots,G`:

        - Draw dropout indicator: :math:`z_g^{(c)} \sim \text{Bernoulli}(\pi_g)`
        - If :math:`z_g^{(c)} = 1`: set :math:`u_g^{(c)} = 0`
        - If :math:`z_g^{(c)} = 0`: draw :math:`u_g^{(c)} \sim \text{NegativeBinomial}(r_g, \hat{p}^{(c)})`

Model Derivation
^^^^^^^^^^^^^^^^

The ZINBVCP model combines the derivations of the NBVCP and ZINB models.
Starting with the standard negative binomial model for mRNA counts as shown in
Eq. (16), we then model both the capture process and technical dropouts:

.. math::
   u_g^{(c)} \mid m_g^{(c)}, z_g^{(c)} \sim 
   z_g^{(c)} \delta_0 + (1-z_g^{(c)}) \text{Binomial}(m_g^{(c)}, \nu^{(c)})
   \tag{19}

where :math:`z_g^{(c)} \sim \text{Bernoulli}(\pi_g)`. Marginalizing over the
unobserved mRNA counts :math:`m_g^{(c)}` and dropout indicators
:math:`z_g^{(c)}`, we get:

.. math::
   u_g^{(c)} \sim 
   \pi_g \delta_0 + (1-\pi_g)\text{NegativeBinomial}(r_g, \hat{p}^{(c)})
   \tag{20}

where :math:`\hat{p}^{(c)}` is the effective probability defined in Eq. (15) and
:math:`\delta_0` is the Dirac delta function at zero.

Model Implementations and Prior Distributions
---------------------------------------------

All models are implemented using variational inference with the following common
prior distributions:

For the success probability :math:`p`:

.. math::
   p \sim \text{Beta}(\alpha_p, \beta_p)
   \tag{21}

Default values: :math:`\alpha_p = \beta_p = 1` (uniform prior)

For each gene's dispersion parameter :math:`r_j`:

.. math::
   r_j \sim \text{Gamma}(\alpha_r, \beta_r)
   \tag{22}

Default values: :math:`\alpha_r = 2`, :math:`\beta_r = 0.1`

Additional priors for model variants:

For each gene's dropout probability (ZINB and ZINBVCP models):

.. math::
   \pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})
   \tag{23}

Default values: :math:`\alpha_{\pi} = \beta_{\pi} = 1` (uniform prior)

For each cell's capture probability (NBVCP and ZINBVCP models):

.. math::
   \nu^{(c)} \sim \text{Beta}(\alpha_{\nu}, \beta_{\nu})
   \tag{24}

Default values: :math:`\alpha_{\nu} = \beta_{\nu} = 1` (uniform prior)

Variational Inference
---------------------

All models use stochastic variational inference with a mean-field variational
family as the approximate posterior. The variational distributions match the
form of the priors:

For the success probability :math:`p`:

.. math::
   q(p) = \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
   \tag{25}

For each gene's dispersion parameter :math:`r_g`:

.. math::
   q(r_g) = \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g}) \quad \text{if using Gamma prior}
   \tag{26.1}

.. math::
   q(r_g) = \text{LogNormal}(\hat{\mu}_{r,g}, \hat{\sigma}_{r,g}) \quad \text{if using LogNormal prior}
   \tag{26.2}

For each gene's dropout probability (ZINB and ZINBVCP models):

.. math::
   q(\pi_g) = \text{Beta}(\hat{\alpha}_{\pi,g}, \hat{\beta}_{\pi,g})
   \tag{27}

For each cell's capture probability (NBVCP and ZINBVCP models):

.. math::
   q(\nu^{(c)}) = \text{Beta}(\hat{\alpha}_{\nu}^{(c)}, \hat{\beta}_{\nu}^{(c)})
   \tag{28}

where hatted parameters are learnable variational parameters.

The Evidence Lower Bound (ELBO) is optimized for each model:

.. math::
   \mathcal{L} = \langle \log \pi(\text{data}, \text{parameters}) \rangle_q - 
   \langle \log q(\text{parameters}) \rangle_q
   \tag{29}

Learning uses the Adam optimizer with automatic differentiation through the
NumPyro framework.

Model Comparison and Selection
----------------------------

The following table summarizes the key differences between SCRIBE's models:

+-------------+--------------------+----------------------+-------------------------+
| Model       | Zero-Inflation     | Variable Capture     | Additional Parameters   |
+=============+====================+======================+=========================+
| NBDM        | No                 | No                   | None                    |
+-------------+--------------------+----------------------+-------------------------+
| ZINB        | Yes (gene-specific)| No                   | Dropout probabilities   |
+-------------+--------------------+----------------------+-------------------------+
| NBVCP       | No                 | Yes (cell-specific)  | Capture probabilities   |
+-------------+--------------------+----------------------+-------------------------+
| ZINBVCP     | Yes (gene-specific)| Yes (cell-specific)  | Both                    |
+-------------+--------------------+----------------------+-------------------------+

Which model to choose:

1. **NBDM**: Use as a baseline when data quality is good and technical variation
   is minimal.
2. **ZINB**: Choose when dropout is a dominant feature (many excess zeros) but
   total counts are relatively consistent.
3. **NBVCP**: Appropriate when cell-to-cell variation in total UMI counts is
   high, suggesting variable capture efficiency.
4. **ZINBVCP**: The most comprehensive model, handling both dropouts and capture
   variation. Start with simpler models and progress to this if needed.

You can use SCRIBE's model comparison utilities to formally compare models:

.. code-block:: python

    from scribe.model_comparison import compare_models
    
    # Fit multiple models
    nbdm_results = scribe.run_scribe(counts, zero_inflated=False, variable_capture=False)
    zinb_results = scribe.run_scribe(counts, zero_inflated=True, variable_capture=False)
    
    # Compare models using WAIC
    comparison = compare_models([nbdm_results, zinb_results], counts)

Implementation Details
----------------------

All models are implemented using:

* The `NumPyro <https://num.pyro.ai/en/stable/index.html>`_ probabilistic
  programming framework
* `JAX <https://jax.readthedocs.io/en/latest/>`_ for automatic differentiation
  and GPU acceleration
* Stochastic variational inference with mini-batching for scalability
* Mean-field variational families for approximate posteriors

Model Assumptions
-----------------

Common assumptions across all models:

* Overdispersion in molecular counts can be captured by negative binomial
  distributions
* Gene-specific dispersion parameters capture biological variation
* The base success probability represents true biological probability of success

Additional assumptions for specific models:

* **ZINB model**: Zeros arise from two distinct processes - technical dropouts
  and biological absence
* **NBVCP model**: Variation in total UMI counts partially reflects technical
  capture differences
* **ZINBVCP model**: Both zero-inflation and capture efficiency variation are
  present

Usage Considerations
-------------------

NBDM Model
^^^^^^^^^^
The model is particularly suitable when:

* The data exhibits overdispersion relative to a Poisson model
* The total count per cell varies moderately
* Gene-specific variation needs to be captured

It may be less suitable when:

* Zero-inflation is a dominant feature (consider ZINB model instead)
* Cell-specific capture efficiencies vary significantly (consider NBVCP model)
* The data contains multiple distinct cell populations (consider mixture models)

ZINB Model
^^^^^^^^^^
The ZINB model is particularly suitable when:

* The data exhibits excessive zeros beyond what a negative binomial predicts
* You need to distinguish technical dropouts from biological zeros
* Genes can be reasonably modeled independently

It may be less suitable when:

* Library size variation is a major concern (consider NBVCP model)
* Cell-specific capture efficiencies vary significantly (consider ZINBVCP model)
* The data contains multiple distinct cell populations (consider mixture models)

NBVCP Model
^^^^^^^^^^^
The NBVCP model is particularly suitable when:

* Cells show high variability in total UMI counts
* Technical variation in capture efficiency is suspected
* Library size normalization alone seems insufficient

It may be less suitable when:

* Zero-inflation is a dominant feature (consider ZINBVCP model)
* Capture efficiency variations are minimal
* The data contains multiple distinct cell populations (consider mixture models)

ZINBVCP Model
^^^^^^^^^^^^^
The ZINBVCP model is particularly suitable when:

* The data exhibits excessive zeros beyond what a negative binomial predicts
* Cells show high variability in total UMI counts
* Both technical dropouts and capture efficiency variation are suspected
* Standard library size normalization seems insufficient

It may be less suitable when:

* The data is relatively clean with few technical artifacts
* The zero-inflation or capture efficiency variation is minimal
* The data contains multiple distinct cell populations (consider mixture models)

The model provides the most comprehensive treatment of technical artifacts among
the non-mixture models in SCRIBE, accounting for both dropouts and capture
efficiency variation. However, this flexibility comes at the cost of increased
model complexity and computational demands.

Recommended Model Selection Workflow
-----------------------------------

We recommend the following workflow for model selection:

1. **Start simple**: Begin with the NBDM model as a baseline.
2. **Evaluate data characteristics**:
   - If there's a large number of zeros, try the ZINB model
   - If total UMI counts vary widely across cells, try the NBVCP model
   - If both issues are present, try the ZINBVCP model
3. **Compare models formally**:
   - Use the `compare_models` function to compute WAIC scores
   - Examine parameter estimates and their interpretation
   - Consider computational requirements
4. **Consider biological interpretability**:
   - Does the selected model align with prior knowledge?
   - Are the estimated parameters biologically plausible?
5. **Check for multiple populations**:
   - If heterogeneity persists, consider mixture variants (see next section)

Mixture Model Extensions
------------------------

Each model discussed above can be extended to include multiple components for
modeling heterogeneous cell populations. See the :doc:`models_mix` documentation
for details on mixture model variants.

The key idea is that cell populations with distinct expression profiles can be
modeled as separate components, each with their own parameters. Mixture models
have additional parameters:

* Mixing weights: :math:`w_k \sim \text{Dirichlet}(\alpha_{\text{mix}})` for
  each component :math:`k`
* Component-specific dispersion parameters: :math:`r_{g,k}` for each gene
  :math:`g` in component :math:`k`
* Component-specific dropout probabilities: :math:`\pi_{g,k}` (for models with
  zero-inflation)

The cell-specific capture probabilities :math:`\nu^{(c)}` remain shared across
components in the mixture models.

References
---------

For more details on the mathematical foundations and evaluation of these models,
please refer to our paper: [CITATION].
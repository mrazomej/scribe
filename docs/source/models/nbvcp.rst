Negative Binomial with Variable Capture Probability Model (NBVCP)
=================================================================

The Negative Binomial with Variable Capture Probability (NBVCP) model extends
the :doc:`Negative Binomial-Dirichlet Multinomial (NBDM) <nbdm>` model by
explicitly modeling cell-specific capture efficiencies. This model is
particularly useful when cells exhibit significant variation in their total UMI
counts, which may indicate differences in mRNA capture rates rather than true
biological differences in expression.

Like the :doc:`NBDM model <nbdm>`, the NBVCP model captures overdispersion in
molecular counts. However, it differs in two key aspects:

1. It explicitly models cell-specific capture probabilities that modify the
   success probability of the negative binomial
2. It does not use the Dirichlet-multinomial in the likelihood, instead treating
   each gene independently

Model Comparison with :doc:`NBDM <nbdm>`
----------------------------------------

In the :doc:`NBDM model <nbdm>`, we assume a single success probability
:math:`p` that is shared across all cells. The NBVCP model relaxes this
assumption by introducing cell-specific capture probabilities that modify how
the base success probability manifests in each cell.

The key insight is that variations in capture efficiency can make the same
underlying mRNA abundance appear different in the UMI counts. The NBVCP model
handles this by:

1. Maintaining a base success probability :math:`p` that represents the "true"
   biological probability
2. Introducing cell-specific capture probabilities :math:`\nu^{(c)}`
   that modify this base probability
3. Computing an effective success probability for each cell :math:`c` as:

.. math::
   \hat{p}^{(c)} = \frac{p \nu^{(c)}}{1 - p (1 - \nu^{(c)})}
   \tag{1}

This adjusted probability accounts for how the capture efficiency of each cell
affects our ability to observe its true mRNA content.

.. note::
   Eq. (1) differs from the one shown in the :doc:`NBDM model <nbdm>`. Although
   they are both mathematically equivalent, given the way that `Numpyro <https://num.pyro.ai/>`_
   defines the meaning of the :math:`p` parameters in the negative binomial, the
   NBVCP model uses Eq. (1) to define the effective probability.

Given the explicit modeling of the cell-specific capture probabilities, the
NBVCP model can remove technical variability, allowing for the same
normalization methods as the :doc:`NBDM model <nbdm>` based on the
Dirichlet-Multinomial model. In other words, since the NBVCP model fits a
parameter to account for significant technical variations in the total number of
counts per cell, once this effect is removed, the remaining variation can be
modeled using the same :math:`\underline{r}` parameters as the :doc:`NBDM model
<nbdm>`. Thus, the NBVCP model presents a more principled approach to
normalization compared to other methods in the scRNA-seq literature.

Model Structure
---------------

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
   \text{Gamma}(\alpha_r, \beta_r)` for :math:`g = 1,\ldots,G`
3. For each cell :math:`c = 1,\ldots,N`:
   
   * Draw capture probability: :math:`\nu^{(c)} \sim \text{Beta}(\alpha_{\nu},
     \beta_{\nu})`
   * Compute effective probability: :math:`\hat{p}^{(c)} = \frac{p \nu^{(c)}}{1
     - p (1 - \nu^{(c)})}`
   * For each gene :math:`g = 1,\ldots,G`: - Draw count: :math:`u_g^{(c)} \sim
     \text{NegativeBinomial}(r_g, \hat{p}^{(c)})`

Model Derivation
----------------

The NBVCP model can be derived by considering how the mRNA capture efficiency
affects the observed UMI counts. Starting with the standard negative binomial
model for mRNA counts:

.. math::
   m_g^{(c)} \sim \text{NegativeBinomial}(r_g, p),
   \tag{2}

where :math:`m_g^{(c)}` is the unobserved mRNA count for gene :math:`g` in cell
:math:`c`, :math:`r_g` is the dispersion parameter, and :math:`p` is the base
success probability shared across all cells. We then model the capture process
as a binomial sampling where each mRNA molecule has probability :math:`\nu^{(c)}`
of being captured:

.. math::
   u_g^{(c)} \mid m_g^{(c)} \sim \text{Binomial}(m_g^{(c)}, \nu^{(c)})
   \tag{3}

Marginalizing over the unobserved mRNA counts :math:`m_g^{(c)}`, we get:

.. math::
   u_g^{(c)} \sim \text{NegativeBinomial}(r_g, \hat{p}^{(c)})
   \tag{4}

where :math:`\hat{p}^{(c)}` is the effective probability defined in Eq. (1).

For more details, see the :ref:`nbdm-model-derivation` section in the :doc:`NBDM
model <nbdm>` documentation.

Prior Distributions
-------------------

The model uses the following prior distributions:

For the base success probability :math:`p`:

.. math::
   p \sim \text{Beta}(\alpha_p, \beta_p)
   \tag{5}

For each gene's dispersion parameter :math:`r_g`:

.. math::
   r_g \sim \text{Gamma}(\alpha_r, \beta_r)
   \tag{6}

For each cell's capture probability :math:`\nu^{(c)}`:

.. math::
   \nu^{(c)} \sim \text{Beta}(\alpha_{\nu}, \beta_{\nu})
   \tag{7}

Variational Posterior Distribution
----------------------------------

The model uses stochastic variational inference with a mean-field variational
family. The variational distributions are:

For the base success probability :math:`p`:

.. math::
   q(p) = \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
   \tag{8}

For each gene's dispersion parameter :math:`r_g`:

.. math::
   q(r_g) = \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
   \tag{9}

For each cell's capture probability :math:`\nu^{(c)}`:

.. math::
   q(\nu^{(c)}) = \text{Beta}(\hat{\alpha}_{\nu}^{(c)}, \hat{\beta}_{\nu}^{(c)})
   \tag{10}

where hatted parameters are learnable variational parameters.

Learning Algorithm
------------------

The training process follows similar steps to the :doc:`NBDM model <nbdm>`:

1. Initialize variational parameters:

   * :math:`\hat{\alpha}_p = \alpha_p`, :math:`\hat{\beta}_p = \beta_p`
   * :math:`\hat{\alpha}_{r,g} = \alpha_r`, :math:`\hat{\beta}_{r,g} = \beta_r`
     for all genes :math:`g`
   * :math:`\hat{\alpha}_{\nu}^{(c)} = \alpha_{\nu}`,
     :math:`\hat{\beta}_{\nu}^{(c)} = \beta_{\nu}` for all cells :math:`c`

2. For each iteration:

   * Sample mini-batch of cells
   * Compute ELBO gradients
   * Update parameters (using Adam optimizer as default)

3. Continue until maximum iterations reached

The key difference is the addition of cell-specific capture probability
parameters that must be learned.

Implementation Details
----------------------

Like the other models, the NBVCP model is implemented using `NumPyro
<https://num.pyro.ai/>`_. Key features include:

* Cell-specific parameter handling for capture probabilities
* Effective probability computation through deterministic transformations
* Independent fitting of genes
* Mini-batch support for scalable inference
* GPU acceleration through `JAX <https://jax.readthedocs.io/en/latest/>`_

Model Assumptions
----------------

The NBVCP model makes several key assumptions:

* Variation in total UMI counts partially reflects technical capture differences
* Each cell has its own capture efficiency that affects all genes equally
* Genes are independent given the cell-specific capture probability
* The base success probability :math:`p` represents true biological variation
* Capture probabilities modify observed counts but not underlying biology

Usage Considerations
--------------------

The NBVCP model is particularly suitable when:

* Cells show high variability in total UMI counts
* Technical variation in capture efficiency is suspected
* Library size normalization alone seems insufficient

It may be less suitable when:

* Zero-inflation is a dominant feature (consider :doc:`ZINBVCP model <zinbvcp>`)
* Capture efficiency variations are minimal
* The data contains multiple distinct cell populations (consider mixture models)

The model provides a principled way to account for technical variation in
capture efficiency while still capturing biological variation in gene
expression. This can be particularly important in situations where differences
in total UMI counts between cells might otherwise be mistaken for biological
differences.
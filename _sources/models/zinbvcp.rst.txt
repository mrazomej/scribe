Zero-Inflated Negative Binomial with Variable Capture Probability Model (ZINBVCP)
=================================================================================

The Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP)
model combines aspects of both the :doc:`ZINB <zinb>` and :doc:`NBVCP <nbvcp>`
models to handle both technical dropouts and variable capture efficiencies in
single-cell RNA sequencing data. This model is particularly useful when the data
exhibits both excess zeros and significant variation in total UMI counts across
cells.

The ZINBVCP model incorporates two key features:

1. Zero-inflation to model technical dropouts (from :doc:`ZINB <zinb>`)
2. Cell-specific capture probabilities (from :doc:`NBVCP <nbvcp>`)

Model Comparison with :doc:`NBVCP <nbvcp>` and :doc:`ZINB <zinb>`
-----------------------------------------------------------------

The ZINBVCP model extends both the :doc:`NBVCP <nbvcp>` and :doc:`ZINB <zinb>`
models by combining their key features. From the :doc:`NBVCP <nbvcp>` model, it
inherits the cell-specific capture probabilities :math:`\nu^{(c)}` that modify
the base success probability :math:`p`. From the :doc:`ZINB <zinb>` model, it
inherits the gene-specific dropout probabilities :math:`\pi_g` that model
technical zeros.

The effective success probability for each cell :math:`c` is computed as:

.. math::
   \hat{p}^{(c)} = \frac{p \nu^{(c)}}{1 - p (1 - \nu^{(c)})}
   \tag{1}

This is then combined with the dropout mechanism to give a zero-inflated
distribution where the non-zero counts use the cell-specific effective
probability.

Model Structure
---------------

The ZINBVCP model follows a hierarchical structure where:

1. Each gene has an associated dropout probability :math:`\pi_g`
2. Each cell has an associated capture probability :math:`\nu^{(c)}`
3. The base success probability :math:`p` is modified by each cell's capture
   probability
4. For genes that aren't dropped out, counts follow a negative binomial with
   cell-specific effective probabilities

Formally, for a dataset with :math:`N` cells and :math:`G` genes, let
:math:`u_{g}^{(c)}` be the UMI count for gene :math:`g` in cell :math:`c`. The
generative process is:

1. Draw global success probability: :math:`p \sim \text{Beta}(\alpha_p, \beta_p)`
2. For each gene :math:`g = 1,\ldots,G`:
   
   * Draw dispersion parameter: :math:`r_g \sim \text{Gamma}(\alpha_r, \beta_r)`
   * Draw dropout probability: :math:`\pi_g \sim \text{Beta}(\alpha_{\pi},
     \beta_{\pi})`

3. For each cell :math:`c = 1,\ldots,N`:
   
   * Draw capture probability: :math:`\nu^{(c)} \sim \text{Beta}(\alpha_{\nu},
     \beta_{\nu})`
   * Compute effective probability: :math:`\hat{p}^{(c)} = \frac{p \nu^{(c)}}{1
     - p (1 - \nu^{(c)})}`
   * For each gene :math:`g = 1,\ldots,G`:

        - Draw dropout indicator: :math:`z_g^{(c)} \sim \text{Bernoulli}(\pi_g)`
        - If :math:`z_g^{(c)} = 1`: set :math:`u_g^{(c)} = 0`
        - If :math:`z_g^{(c)} = 0`: draw :math:`u_g^{(c)} \sim
        \text{NegativeBinomial}(r_g, \hat{p}^{(c)})`

Model Derivation
----------------

The ZINBVCP model combines the derivations of the :doc:`NBVCP <nbvcp>` and
:doc:`ZINB <zinb>` models. Starting with the standard negative binomial model
for mRNA counts:

.. math::
   m_g^{(c)} \sim \text{NegativeBinomial}(r_g, p)
   \tag{2}

We then model both the capture process and technical dropouts:

.. math::
   u_g^{(c)} \mid m_g^{(c)}, z_g^{(c)} \sim 
   z_g^{(c)} \delta_0 + (1-z_g^{(c)}) \text{Binomial}(m_g^{(c)}, \nu^{(c)})
   \tag{3}

where :math:`z_g^{(c)} \sim \text{Bernoulli}(\pi_g)`. Marginalizing over the
unobserved mRNA counts :math:`m_g^{(c)}` and dropout indicators :math:`z_g^{(c)}`,
we get:

.. math::
   u_g^{(c)} \sim 
   \pi_g \delta_0 + (1-\pi_g)\text{NegativeBinomial}(r_g, \hat{p}^{(c)})
   \tag{4}

where :math:`\hat{p}^{(c)}` is the effective probability defined in Eq. (1) and
:math:`\delta_0` is the Dirac delta function at zero.

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

For each gene's dropout probability :math:`\pi_g`:

.. math::
   \pi_g \sim \text{Beta}(\alpha_{\pi}, \beta_{\pi})
   \tag{7}

For each cell's capture probability :math:`\nu^{(c)}`:

.. math::
   \nu^{(c)} \sim \text{Beta}(\alpha_{\nu}, \beta_{\nu})
   \tag{8}

Variational Posterior Distribution
----------------------------------

The model uses stochastic variational inference with a mean-field variational
family. The variational distributions are:

For the base success probability :math:`p`:

.. math::
   q(p) = \text{Beta}(\hat{\alpha}_p, \hat{\beta}_p)
   \tag{9}

For each gene's dispersion parameter :math:`r_g`:

.. math::
   q(r_g) = \text{Gamma}(\hat{\alpha}_{r,g}, \hat{\beta}_{r,g})
   \tag{10}

For each gene's dropout probability :math:`\pi_g`:

.. math::
   q(\pi_g) = \text{Beta}(\hat{\alpha}_{\pi,g}, \hat{\beta}_{\pi,g})
   \tag{11}

For each cell's capture probability :math:`\nu^{(c)}`:

.. math::
   q(\nu^{(c)}) = \text{Beta}(\hat{\alpha}_{\nu}^{(c)}, \hat{\beta}_{\nu}^{(c)})
   \tag{12}

where hatted parameters are learnable variational parameters.

Learning Algorithm
------------------

The training process follows similar steps to the :doc:`NBVCP <nbvcp>` and
:doc:`ZINB <zinb>` models:

1. Initialize variational parameters:

   * :math:`\hat{\alpha}_p = \alpha_p`, :math:`\hat{\beta}_p = \beta_p`
   * :math:`\hat{\alpha}_{r,g} = \alpha_r`, :math:`\hat{\beta}_{r,g} = \beta_r`
     for all genes :math:`g`
   * :math:`\hat{\alpha}_{\pi,g} = \alpha_{\pi}`, :math:`\hat{\beta}_{\pi,g} =
     \beta_{\pi}` for all genes :math:`g`
   * :math:`\hat{\alpha}_{\nu}^{(c)} = \alpha_{\nu}`,
     :math:`\hat{\beta}_{\nu}^{(c)} = \beta_{\nu}` for all cells :math:`c`

2. For each iteration:

   * Sample mini-batch of cells
   * Compute ELBO gradients
   * Update parameters (using Adam optimizer as default)

3. Continue until maximum iterations reached

Implementation Details
----------------------

The model is implemented using `NumPyro
<https://num.pyro.ai/en/stable/index.html>`_ with key features including:

* Cell-specific parameter handling for capture probabilities
* Gene-specific parameter handling for dropout probabilities
* Effective probability computation through deterministic transformations
* Zero-inflated distributions using NumPyro's ZeroInflatedDistribution
* Mini-batch support for scalable inference
* GPU acceleration through JAX

Model Assumptions
-----------------

The ZINBVCP model makes several key assumptions:

* Zeros can arise from two processes:

    - Technical dropouts (modeled by zero-inflation)
    - Biological absence of expression (modeled by negative binomial)

* Variation in total UMI counts partially reflects technical capture differences
* Each cell has its own capture efficiency that affects all genes equally
* Each gene has its own dropout probability
* Genes are independent given the cell-specific capture probability
* The base success probability represents true biological variation
* Capture probabilities modify observed counts but not underlying biology

Usage Considerations
--------------------

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
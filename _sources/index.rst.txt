Welcome to SCRIBE's documentation!
==================================

SCRIBE (Single-Cell RNA-Seq Inference using Bayesian Estimation) is a `Python`
package for analyzing single-cell RNA sequencing (scRNA-seq) data using
variational inference based on `Numpyro
<https://num.pyro.ai/en/stable/index.html>`_—a `Jax
<https://jax.readthedocs.io/en/latest/>`_-based probabilistic programming
library with GPU acceleration. It provides a collection of probabilistic models
and inference tools specifically designed for scRNA-seq count data.

Features
--------

- Multiple probabilistic models for scRNA-seq data analysis
- Efficient variational inference using `JAX
  <https://jax.readthedocs.io/en/latest/>`_ and `Numpyro
  <https://num.pyro.ai/en/stable/index.html>`_
- Support for both full-batch and mini-batch inference for large-scale data
- Integration with `AnnData <https://anndata.readthedocs.io/en/stable/>`_
  objects
- Comprehensive visualization tools for posterior analysis
- GPU acceleration support

Available Models
-----------------

SCRIBE includes several probabilistic models for scRNA-seq data:

1. :doc:`Negative Binomial-Dirichlet Multinomial (NBDM) <models/nbdm>`

  - Models both count magnitudes and proportions
  - Accounts for overdispersion in count data

2. :doc:`Zero-Inflated Negative Binomial (ZINB) <models/zinb>`

  - Handles excess zeros in scRNA-seq data
  - Models technical and biological dropouts
  - Includes gene-specific dropout rates

3. :doc:`Negative Binomial with Variable Capture Probability (NBVCP) <models/nbvcp>` 

  - Accounts for cell-specific mRNA capture efficiency 
  - Models technical variation in library preparation 
  - Suitable for datasets with varying sequencing depths per cell

4. :doc:`Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP) <models/zinbvcp>` 

  - Combines zero-inflation and variable capture probability 
  - Most comprehensive model for technical variation 
  - Handles both dropouts and capture efficiency

5. :doc:`Mixture Models <models/models_mix>`

  - Any of the above models can be turned into a mixture model to account for
    subpopulations.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickoverview
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Available Models
   :hidden:

   models/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   results
   api/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
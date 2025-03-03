Welcome to SCRIBE's documentation!
==================================

.. warning::
   **Project Status and Usage Restrictions**

   This project is currently in a pre-release state and is made available for
   viewing and evaluation purposes only. 
   
   At this time:
   
   - You may view and evaluate the code and documentation
   - You may not use, copy, modify, or distribute any part of this software
   - You may not incorporate this code into other projects
   
   The software will be released under the MIT License following the publication
   of the associated pre-print. 
   Until then, all rights are reserved.


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

SCRIBE includes several probabilistic models for scRNA-seq data, all documented
in detail in :doc:`models/models`:

1. :ref:`Core Model: Negative Binomial-Dirichlet Multinomial (NBDM) <nbdm-model>`

   - Models both count magnitudes and proportions
   - Accounts for overdispersion in count data
   - Forms the foundation of SCRIBE's modeling approach

2. :ref:`Zero-Inflated Negative Binomial (ZINB) <zinb-model>`

   - Handles excess zeros in scRNA-seq data
   - Models technical and biological dropouts
   - Includes gene-specific dropout rates

3. :ref:`Negative Binomial with Variable Capture Probability (NBVCP) <nbvcp-model>`

   - Accounts for cell-specific mRNA capture efficiency 
   - Models technical variation in library preparation 
   - Suitable for datasets with varying sequencing depths per cell

4. :ref:`Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP) <zinbvcp-model>`

   - Combines zero-inflation and variable capture probability 
   - Most comprehensive model for technical variation 
   - Handles both dropouts and capture efficiency

All these models can be extended to mixture variants as documented in :doc:`models/models_mix` to account for heterogeneous cell populations.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickoverview
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Models
   :hidden:

   models/models
   models/models_mix

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
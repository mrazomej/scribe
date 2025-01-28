Welcome to SCRIBE's documentation!
================================

SCRIBE (Single-Cell RNA-Seq Inference using Bayesian Estimation) is a `Python`
package for analyzing single-cell RNA sequencing (scRNA-seq) data using
variational inference based on `Numpyro`—a `Jax`-based probabilistic programming
library with GPU acceleration. It provides a collection of probabilistic models
and inference tools specifically designed for scRNA-seq count data.

Features
--------

- Multiple probabilistic models for scRNA-seq data analysis
- Efficient variational inference using JAX and Numpyro
- Support for both full-batch and mini-batch inference for large-scale data
- Integration with `AnnData` objects
- Comprehensive visualization tools for posterior analysis
- GPU acceleration support

Available Models
-----------------

SCRIBE includes several probabilistic models for scRNA-seq data:

1. **Negative Binomial-Dirichlet Multinomial (NBDM)**
   - Models both count magnitudes and proportions
   - Accounts for overdispersion in count data

2. **Zero-Inflated Negative Binomial (ZINB)**
   - Handles excess zeros in scRNA-seq data
   - Models technical and biological dropouts
   - Includes gene-specific dropout rates

3. **Negative Binomial with Variable Capture Probability (NBVCP)**
   - Accounts for cell-specific mRNA capture efficiency
   - Models technical variation in library preparation
   - Suitable for datasets with varying sequencing depths per cell

4. **Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP)**
   - Combines zero-inflation and variable capture probability
   - Most comprehensive model for technical variation
   - Handles both dropouts and capture efficiency

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
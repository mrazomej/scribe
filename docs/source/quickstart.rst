Quickstart
=========

This quickstart guide will help you get started with SCRIBE.

Basic Usage
----------

Here's a simple example of using SCRIBE:

.. code-block:: python

   import scribe
   import scanpy as sc

   # Load your data
   adata = sc.read_h5ad("your_data.h5ad")

   # Run SCRIBE
   results = scribe.run_scribe(
       adata,
       model_type="nbdm",
       n_steps=100_000
   )
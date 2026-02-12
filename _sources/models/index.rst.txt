Available Models
================

SCRIBE provides a family of probabilistic models for single-cell RNA sequencing
data, all built on the foundational **Negative Binomial-Dirichlet Multinomial
(NBDM)** framework. Rather than choosing between completely different models,
you select the variant that best matches your data characteristics using simple
boolean flags.

Quick Start
-----------

All models are accessed through the same unified interface with boolean flags:

.. code-block:: python

   import scribe
   
   # Basic NBDM model with SVI
   results = scribe.run_scribe(counts, inference_method="svi")
   
   # Zero-inflated model for data with excess zeros
   results = scribe.run_scribe(counts, inference_method="svi", zero_inflated=True)
   
   # Variable capture model for cells with different capture efficiencies  
   results = scribe.run_scribe(counts, inference_method="svi", variable_capture=True)
   
   # Combined model addressing both issues
   results = scribe.run_scribe(counts, inference_method="svi", 
                              zero_inflated=True, variable_capture=True)
   
   # Mixture model for multiple cell populations
   results = scribe.run_scribe(counts, inference_method="svi", 
                              mixture_model=True, n_components=3)
   
   # MCMC inference with any model
   results = scribe.run_scribe(counts, inference_method="mcmc", 
                              zero_inflated=True, n_samples=2000)

Model Selection Guide
--------------------

Choose your model by answering these questions:

.. raw:: html

   <div style="font-family: monospace; margin: 20px 0;">
   
   Does your data have <strong>excess zeros</strong> beyond biological variation?
   ├─ <strong>YES</strong> → Set <code>zero_inflated=True</code>
   └─ <strong>NO</strong> → Continue with basic NBDM
   
   Do <strong>cells vary significantly</strong> in total UMI counts due to technical factors?
   ├─ <strong>YES</strong> → Set <code>variable_capture=True</code>
   └─ <strong>NO</strong> → Continue with current choice
   
   Do you have <strong>multiple distinct cell populations</strong>?
   ├─ <strong>YES</strong> → Set <code>mixture_model=True, n_components=K</code>
   └─ <strong>NO</strong> → You're done!
   
   </div>

Available Model Variants
-----------------------

.. list-table:: SCRIBE Model Family
   :header-rows: 1
   :widths: 15 15 15 20 20 15

   * - **Model**
     - **Zero Inflated**
     - **Variable Capture**  
     - **Key Feature**
     - **Best For**
     - **Computational Cost**
   * - :doc:`NBDM <nbdm>`
     - ✗
     - ✗
     - Compositional normalization
     - Clean data, moderate overdispersion
     - Low
   * - :doc:`ZINB <zinb>`
     - ✓
     - ✗
     - Technical dropout modeling
     - Data with excess zeros
     - Low-Medium
   * - :doc:`NBVCP <nbvcp>`
     - ✗
     - ✓
     - Cell-specific capture rates
     - Variable library sizes
     - Medium
   * - :doc:`ZINBVCP <zinbvcp>`
     - ✓
     - ✓
     - Both dropouts and capture variation
     - Complex technical artifacts
     - High
   * - :doc:`Mixture Models <models_mix>`
     - Any
     - Any
     - Multiple cell populations
     - Heterogeneous samples
     - High

Detailed Comparison
------------------

Basic Models
~~~~~~~~~~~

**NBDM (Negative Binomial-Dirichlet Multinomial)**
  The foundational model that provides principled compositional normalization by
  modeling:
  
  - Total UMI count per cell (Negative Binomial)
  - Gene-wise allocation of UMIs (Dirichlet-Multinomial)
  
  *Use when*: Data is relatively clean with moderate overdispersion.

**ZINB (Zero-Inflated Negative Binomial)**
  Extends NBDM by adding a zero-inflation component to handle technical
  dropouts:
  
  - Gene-specific dropout probabilities
  - Independent modeling of each gene
  
  *Use when*: Excessive zeros beyond what NBDM predicts.

**NBVCP (NB with Variable Capture Probability)**
  Extends NBDM by modeling cell-specific mRNA capture efficiencies:
  
  - Cell-specific capture probabilities
  - Accounts for technical variation in library preparation
  
  *Use when*: Large variation in total UMI counts across cells.

**ZINBVCP (Zero-Inflated NB with Variable Capture Probability)**  
  Combines both zero-inflation and variable capture modeling:
  
  - Most comprehensive single-cell artifact modeling
  - Highest computational cost
  
  *Use when*: Data has both excess zeros and variable capture efficiency.

Mixture Models
~~~~~~~~~~~~~

Any of the above models can be extended to mixture variants by adding
``mixture_model=True``:

.. code-block:: python

   # ZINB mixture model for 3 cell populations
   results = scribe.fit_mcmc(
       counts, 
       zero_inflated=True, 
       mixture_model=True, 
       n_components=3
   )

*Use when*: Your sample contains multiple distinct cell types or states.

Code Examples
-------------

Typical Workflows
~~~~~~~~~~~~~~~~

**Standard Analysis with SVI**

.. code-block:: python

   import scribe
   import anndata as ad
   
   # Load data
   adata = ad.read_h5ad('data.h5ad')
   
   # Fit basic model using SVI
   results = scribe.run_scribe(
       adata, 
       inference_method="svi",
       n_steps=100_000
   )
   
   # Get posterior predictive samples
   ppc_samples = results.ppc_samples(n_samples=100)
   
   # Visualize results
   scribe.viz.plot_parameter_posteriors(results)

**MCMC Analysis**

.. code-block:: python

   # Fit model using MCMC
   mcmc_results = scribe.run_scribe(
       counts=counts,
       inference_method="mcmc",
       zero_inflated=True,
       n_samples=2000,
       n_warmup=1000
   )
   
   # Check convergence diagnostics
   print(mcmc_results.summary)

**Comparing Models**

.. code-block:: python

   # Fit different models
   basic_results = scribe.run_scribe(counts=counts, inference_method="svi")
   zinb_results = scribe.run_scribe(
    counts=counts, inference_method="svi", zero_inflated=True)
   
   # Compare model fit using WAIC
   from scribe.model_comparison import compute_waic
   basic_waic = compute_waic(basic_results, counts)
   zinb_waic = compute_waic(zinb_results, counts)
   
   print(f"Basic WAIC: {basic_waic['waic_2']:.2f}")
   print(f"ZINB WAIC: {zinb_waic['waic_2']:.2f}")

**Mixture Model Analysis**

.. code-block:: python

   # Fit mixture model
   mixture_results = scribe.run_scribe(
       counts=counts, 
       inference_method="svi",
       mixture_model=True, 
       n_components=3,
       n_steps=150_000  # More steps for mixture models
   )
   
   # Get cell type assignments
   assignments = mixture_results.cell_type_assignments(counts=counts)
   mean_probs = assignments['mean_probabilities']
   
   # Analyze component-specific parameters
   posterior_samples = mixture_results.get_posterior_samples(n_samples=1000)
   for k in range(3):
       r_k = posterior_samples[f'r_{k}']
       print(f"Component {k} mean dispersion: {r_k.mean():.3f}")

**Advanced Parameterizations**

.. code-block:: python

   # Standard parameterization (Beta/Gamma distributions)
   standard_results = scribe.run_scribe(
       counts=counts, 
       inference_method="svi",
       parameterization="standard",
       zero_inflated=True
   )
   
   # Odds-ratio parameterization (often better for optimization)
   odds_results = scribe.run_scribe(
       counts=counts,
       inference_method="svi", 
       parameterization="odds_ratio",
       zero_inflated=True,
       phi_prior=(3, 2)  # BetaPrime prior parameters
   )
   
   # Unconstrained parameterization (good for MCMC)
   unconstrained_results = scribe.run_scribe(
       counts=counts,
       inference_method="mcmc",
       parameterization="unconstrained",
       variable_capture=True
   )

Performance Considerations
-------------------------

**Computational Complexity**

- **NBDM**: O(N × G) - Linear in cells and genes
- **ZINB**: O(N × G) - Similar to NBDM  
- **NBVCP**: O(N × G) - Additional cell parameters
- **ZINBVCP**: O(N × G) - Most parameters per model
- **Mixtures**: O(K × base_model) - Scales with components

**Memory Requirements**

- Base models: ~10-50 MB for typical datasets
- Variable capture: +N parameters (cell-specific)
- Mixture models: +K×(base parameters) 

**SVI Convergence (n_steps)**

.. list-table:: Typical SVI Requirements
   :header-rows: 1

   * - **Model Type**
     - **Standard**
     - **Odds-Ratio**
     - **Unconstrained**
   * - NBDM, ZINB
     - 50,000-100,000
     - 25,000-50,000
     - 100,000-200,000
   * - NBVCP, ZINBVCP  
     - 100,000-150,000
     - 50,000-100,000
     - 150,000-300,000
   * - Mixture Models
     - 150,000-300,000
     - 100,000-200,000
     - 300,000-500,000

**MCMC Convergence (n_samples)**

.. list-table:: Typical MCMC Requirements
   :header-rows: 1

   * - **Model Type**
     - **Warmup Steps**
     - **Sample Steps**
     - **Chains**
   * - NBDM, ZINB
     - 1,000
     - 2,000
     - 2-4
   * - NBVCP, ZINBVCP  
     - 2,000
     - 3,000
     - 4
   * - Mixture Models
     - 3,000
     - 5,000
     - 4-8

**Parameterization Guide**

- **Standard**: Good default choice, uses natural parameter distributions
- **Odds-Ratio**: Often converges faster in SVI, good for optimization
- **Linked**: Alternative parameterization for specific use cases
- **Unconstrained**: Best for MCMC, allows unrestricted parameter space

Mathematical Foundation
----------------------

All SCRIBE models build on the core insight that single-cell RNA-seq data can be decomposed into:

1. **Total transcriptome size** (how many molecules per cell)
2. **Gene-wise allocation** (how molecules are distributed among genes)

This decomposition enables principled normalization and uncertainty quantification. The mathematical derivation showing how this leads to the Negative Binomial-Dirichlet Multinomial formulation is detailed in our :doc:`mathematical foundation <mathematical_foundation>`.

Model variants extend this foundation by:

- **Zero-inflation**: Adding technical dropout layers
- **Variable capture**: Cell-specific efficiency parameters  
- **Mixtures**: Multiple parameter sets for different populations

Next Steps
----------

1. **Start with the basic NBDM model** to establish baseline performance
2. **Check model diagnostics** to identify potential issues
3. **Add complexity incrementally** based on your data characteristics
4. **Compare models** using information criteria and posterior predictive checks

For detailed information about each model:

.. toctree::
   :maxdepth: 2

   nbdm
   zinb  
   nbvcp
   zinbvcp
   models_mix
   model_selection
   mathematical_foundation
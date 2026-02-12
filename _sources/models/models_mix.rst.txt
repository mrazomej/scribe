Mixture Models 
==============

This document explains SCRIBE's mixture model extensions for handling
heterogeneous cell populations. Each base model (:doc:`NBDM <nbdm>`, :doc:`ZINB
<zinb>`, :doc:`NBVCP <nbvcp>`, :doc:`ZINBVCP <zinbvcp>`) can be extended to a
mixture model by introducing multiple components and component-specific
parameters.

General Structure
-----------------

All mixture models in SCRIBE share a common hierarchical structure:

1. Global Parameters (shared across all components):
  - Base success probability :math:`p \sim \text{Beta}(\alpha_p, \beta_p)`
  - Mixing weights :math:`\pi \sim \text{Dirichlet}(\alpha_{\text{mixing}})`

2. Component-Specific Parameters:
  - Gene dispersion parameters :math:`r_{k,g} \sim \text{Gamma}(\alpha_r, \beta_r)`
  - One per gene :math:`g` per component :math:`k`
  - Additional parameters depending on base model

3. Cell-Specific Parameters (when applicable):
  - Capture probabilities :math:`\nu^{(c)}` (for :doc:`NBVCP <nbvcp>` and :doc:`ZINBVCP <zinbvcp>` variants)
  - Independent of components

4. Gene-Specific Parameters (when applicable):
  - Dropout probabilities (for :doc:`ZINB <zinb>` and :doc:`ZINBVCP <zinbvcp>` variants)
  - Component-specific versions in mixture setting

Parameter Dependencies by Model
-------------------------------

NBDM Mixture
^^^^^^^^^^^^

* Component-dependent:

  - Gene dispersion parameters :math:`r_{k,g}`

* Component-independent:

  - Base success probability :math:`p`

* No cell-specific parameters

ZINB Mixture
^^^^^^^^^^^^

* Component-dependent:

  - Gene dispersion parameters :math:`r_{k,g}`
  - Dropout probabilities :math:`\pi_{k,g}`

* Component-independent:

  - Base success probability :math:`p`

* No cell-specific parameters

NBVCP Mixture
^^^^^^^^^^^^^

* Component-dependent:

  - Gene dis0ersion parameters :math:`r_{k,g}`

* Component-independent:

  - Base success probability :math:`p`
  - Cell capture probabilities :math:`\nu^{(c)}`

* Cell-specific:

  - Capture probabilities

ZINBVCP Mixture
^^^^^^^^^^^^^^^

* Component-dependent:

  - Gene dispersion parameters :math:`r_{k,g}`
  - Dropout probabilities :math:`\pi_{k,g}`
  
* Component-independent:

  - Base success probability :math:`p`
  - Cell capture probabilities :math:`\nu^{(c)}`

* Cell-specific:

  - Capture probabilities

Learning Process
----------------

For all mixture models:

1. Component Assignment Phase:

   - Each cell's data influences the posterior over component assignments
   - Mixing weights are learned globally
   - Component-specific parameters adapt to their assigned cells

2. Parameter Updates:

   - Global parameters: Updated using data from all cells
   - Component parameters: Updated primarily using data from cells assigned to that component
   - Cell-specific parameters: Updated using that cell's data across all components

Usage Guidelines
----------------

When to use mixture models:

1. Clear biological heterogeneity (multiple cell types)
2. Multimodal expression patterns
3. Complex technical variation that varies by cell type

Model selection considerations:

- NBDM Mixture: Baseline mixture model, good for initial exploration
- ZINB Mixture: When dropout patterns vary by cell type
- NBVCP Mixture: When capture efficiency varies significantly
- ZINBVCP Mixture: Most complex, but handles both dropout and capture variation

Implementation Details
----------------------

All mixture models use:

1. Shared parameters across cells within each component
2. Soft assignments of cells to components
3. Variational inference for parameter estimation
4. Mini-batch processing for scalability

Inference and Results
---------------------

The mixture model variants return specialized results objects that provide:

1. Component-specific parameter estimates
2. Cell assignment probabilities
3. Model-specific normalizations
4. Uncertainty quantification for all parameters

Key Differences from Base Models
--------------------------------

1. Parameter Interpretation:

   - Parameters now represent component-specific patterns
   - Cell assignments provide clustering information
   - Mixing weights quantify population proportions

2. Computational Considerations:

   - Higher computational cost
   - More parameters to estimate
   - Requires more data for reliable inference

3. Biological Interpretation:

   - Captures subpopulation structure
   - Allows different technical characteristics by component
   - Provides natural clustering framework

References
----------

Base model documentation:

   - :doc:`models`
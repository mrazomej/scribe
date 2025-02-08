Results Classes
===============

The SCRIBE package provides a comprehensive set of result classes to help you
work with and analyze the output of model inference. These classes provide a
consistent interface across different model types while exposing model-specific
functionality where needed.

Base Structure
--------------

All result classes in SCRIBE inherit from ``BaseScribeResults``, which provides
core functionality for:

* Accessing variational model parameters and posterior distributions
* Indexing results by gene (single gene, ranges, boolean indexing)
* Generating posterior samples and predictive samples
* Handling metadata from ``AnnData`` objects

Basic Usage
-----------

After running inference with ``run_scribe``, you'll get a results object
specific to your model type:

.. code-block:: python

    import scribe
    
    # Run inference
    results = scribe.run_scribe(
        counts=counts,
        model_type="nbdm",  # or any other model type
        n_steps=10000
    )

The results object contains several key attributes:

* ``params``: Dictionary of learned variational parameters
* ``loss_history``: Array of ELBO values during training
* ``n_cells``, ``n_genes``: Dataset dimensions
* ``n_components``: Number of components in mixture models (``None`` for non-mixture models)
* ``cell_metadata``, ``gene_metadata``: Optional metadata if using ``AnnData``

Common Operations
-----------------

All result objects support basic operations that allow you to access the results
of the inference.

Working with Results
^^^^^^^^^^^^^^^^^^^^

All result objects support these basic operations:

1. **Accessing variational parameters and parameters posteriors**:

.. code-block:: python

    # Get raw parameters for variational posterior
    params = results.params
    
    # Get posterior distributions for parameters. 
    distributions = results.get_distributions()

.. note::

    The ``get_distributions`` method returns a dictionary of parameter names
    and their corresponding posterior distributions. The backend can be set to
    either ``scipy`` or ``numpyro`` via the ``backend`` argument.

2. **Subsetting Genes**:

.. code-block:: python

    # Get results for first gene
    gene_results = results[0]
    
    # Get results for a set of genes
    subset_results = results[0:10]  # First 10 genes

    ## Get results for a specific gene (if metadata is available)
    gene_results = results[results.gene_metadata['gene_name'] == 'Gene1']

.. warning::

    Indexing with an array of indices is supported, however, the results are
    returned not in the same order as the input indices but rather in the sorted
    order of the indices. For example, if you index with ``[3, 1, 2]``, the
    results will be returned in the order ``[1, 2, 3]``.

3. **Posterior Sampling**:

.. code-block:: python

    # Sample from posterior distributions.
    results.get_posterior_samples(n_samples=1000)
    
    # Generate predictive samples (this case using the posterior samples already
    # computed)
    results.get_predictive_samples()

    # Get posterior predictive samples
    results.get_ppc_samples(n_samples=1000)

.. note::

    Generating posterior predictive samples is computationally expensive as each
    sample simulates an entire dataset. If you don't have a massive GPU that
    can handle this, we recommend generating samples for a subset of the genes
    for diagnostic purposes.

4. **Log Likelihood function**:

.. code-block:: python

    # Get the log likelihood function
    log_likelihood = results.get_log_likelihood_fn()

    # Use the log likelihood function to compute the log likelihood of the data
    # for a set of parameters
    log_likelihood(counts, results.params)

Model-Specific Results
----------------------

The following sections describe the results objects for each model type.

.. _nbdm_results:

NBDM Results
^^^^^^^^^^^^

The ``NBDMResults`` class is for the Negative Binomial-Dirichlet Multinomial
model:

.. code-block:: python

    # Run NBDM inference
    nbdm_results = scribe.run_scribe(counts, model_type="nbdm")
    
    # Access model-specific parameters

    # Gene-specific dispersion parameters
    nbdm_results.params['alpha_r']  
    nbdm_results.params['beta_r']  

    # Global success probability
    nbdm_results.params['alpha_p']  
    nbdm_results.params['beta_p']

    # Gene specific dispersion distribution
    nbdm_results.get_distributions()['r']

    # Global success probability distribution
    nbdm_results.get_distributions()['p']

Key features:

* Gene-specific dispersion parameters
* Global success probability

ZINB Results
^^^^^^^^^^^^

The ``ZINBResults`` class has the same core parameters as the
:ref:`nbdm_results` class but adds zero-inflation handling:

.. code-block:: python

    # Run ZINB inference  
    zinb_results = scribe.run_scribe(counts, model_type="zinb")
    
    # Access dropout probabilities parameters
    zinb_results.params['alpha_gate']
    zinb_results.params['beta_gate']

    # Gene-specific dropout probabilities distribution
    zinb_results.get_distributions()['gate']

Key features:

* Same core parameters as :ref:`nbdm_results`
* Gene-specific dropout probabilities

NBVCP Results
^^^^^^^^^^^^^

The ``NBVCPResults`` class has the same core parameters as the
:ref:`nbdm_results` class but adds variable capture probabilities:

.. code-block:: python

    # Run NBVCP inference
    nbvcp_results = scribe.run_scribe(counts, model_type="nbvcp")
    
    # Access capture probabilities
    nbvcp_results.params['alpha_p_capture']
    nbvcp_results.params['beta_p_capture']

    # Capture probability distribution
    nbvcp_results.get_distributions()['p_capture']

Key features:

* Same core parameters as :ref:`nbdm_results`
* Cell-specific capture probabilities

ZINBVCP Results
^^^^^^^^^^^^^^^

The ``ZINBVCPResults`` class combines zero-inflation and variable capture:

.. code-block:: python

    # Run ZINBVCP inference
    zinbvcp_results = scribe.run_scribe(counts, model_type="zinbvcp")

    # Access dropout probabilities parameters
    zinbvcp_results.params['alpha_gate']
    zinbvcp_results.params['beta_gate']

    # Capture probability distribution
    zinbvcp_results.get_distributions()['p_capture']

    # Gene-specific dropout probabilities distribution
    zinbvcp_results.get_distributions()['gate']

Key features:

* Cell-specific capture probabilities
* Gene-specific dropout probabilities
* Most comprehensive technical artifact handling

.. _mixture_results:

Mixture Model Results
---------------------

For mixture models (e.g., ``NBDMMixtureResults``, ``ZINBMixtureResults``),
additional functionality is available:

.. code-block:: python
    
    # Run mixture model inference
    mix_results = scribe.run_scribe(
        counts=counts,
        model_type="nbdm_mix",# or any other of the base models with _mix suffix
        n_components=2
    )
    
    # Access mixing weights
    mix_results.params['alpha_mixing']

    # Mixing weights distribution
    mix_results.get_distributions()['mixing_weights']

Key features:

* Component-specific parameters
* Mixing weights
* Same core functionality as non-mixture versions

Working with Custom Models
--------------------------

The ``CustomResults`` class allows you to work with custom model implementations
while maintaining compatibility with ``SCRIBE``'s infrastructure. 

.. code-block:: python

    # Define custom model/guide functions
    custom_results = scribe.run_scribe(
        counts=counts,
        custom_model=my_model,
        custom_guide=my_guide,
        param_spec=my_param_spec,
        n_steps=10000
    )

.. note::

    We recommend checking the :doc:`./examples/custom_model` example for
    more details on how to use the ``CustomResults`` class.

Key requirements:

* Must provide ``param_spec`` dictionary indicating the parameter types
* Should implement required model methods
* Can extend with custom functionality

Best Practices
--------------

1. **Memory Management**:

  * Use ``batch_size`` for large datasets
  * Avoid generating posterior samples for all genes
  * Use subsetting for gene-specific analysis

2. **Working with Parameters**:

  * Access raw parameters through ``.params``
  * Use ``.get_distributions()`` for either sampling or parameter comparison
  * Remember parameter types (``global``/ ``gene-specific``/ ``cell-specific``)

3. **Model Selection**:

  * Use simpler models first (e.g., NBDM)
  * Add complexity (zero-inflation, capture probability) as needed
  * Consider mixture models for heterogeneous populations

4. **Error Handling**:

  * Check ``loss_history`` for convergence
  * Validate parameters are in expected ranges

See Also
--------

* :doc:`models/nbdm` - Details on the NBDM model
* :doc:`models/zinb` - Details on the ZINB model
* :doc:`models/nbvcp` - Details on the NBVCP model
* :doc:`models/zinbvcp` - Details on the ZINBVCP model
* :doc:`models/models_mix` - Details on the mixture models

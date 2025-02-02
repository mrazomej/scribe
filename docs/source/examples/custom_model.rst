Custom Models
=============

``SCRIBE`` provides a flexible framework for implementing and working with custom
models while maintaining compatibility with the package's infrastructure. This
tutorial will walk you through the process of creating and using custom models,
using a real example of modifying the :doc:`../models/nbdm` model to use a
`LogNormal <https://en.wikipedia.org/wiki/Log-normal_distribution>`_ prior.

Overview
--------

Creating a custom model in ``SCRIBE`` involves several key components:

1. Defining the model function
2. Defining the guide function
3. Specifying parameter types (either ``global``, ``gene-specific``, or ``cell-specific``)
4. Running inference using ``run_scribe``
5. Working with the results

Let's go through each step in detail. First, we begin with the needed imports:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    import numpyro
    import scribe

Defining the Model
------------------

The model function defines your probabilistic model using ``NumPyro``
primitives. For this tutorial, we will modify the :doc:`../models/nbdm` model to
use a LogNormal prior for the dispersion parameters. The function we will define
must have the following signature:

* ``n_cells``: The number of cells
* ``n_genes``: The number of genes
* ``param_prior``: The parameters used for the prior distribution of the parameters.

  - Define one entry per parameter. In our case we have two parameters, ``p``
    and ``r``. Thus, we define two entries: ``p_prior`` and ``r_prior``.

* ``counts``: The count data
* ``custom_arg``: Any additional arguments needed by the model.

  - Define one entry per argument. In our case we have one custom argument,
    ``total_counts`` as the model requires not only the individual gene counts
    but the total counts per cell as well.

* ``batch_size``: The batch size for mini-batch training

.. code-block:: python

    def nbdm_lognormal_model(
        n_cells: int,
        n_genes: int,
        p_prior: tuple = (1, 1),
        r_prior: tuple = (0, 1),  # Changed to mean, std for lognormal
        counts=None,
        total_counts=None,
        batch_size=None,
    ):
        # Define success probability prior (unchanged)
        p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

        # Define dispersion prior using LogNormal instead of Gamma
        r = numpyro.sample(
            "r", 
            dist.LogNormal(r_prior[0], r_prior[1]).expand([n_genes])
        )

        # Rest of the model definition...
        r_total = numpyro.deterministic("r_total", jnp.sum(r))

         # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts
                )

            # Define plate for cells individual counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts),
                    obs=counts
                )
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx]
                )

            # Define plate for cells individual counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(
                        r, total_count=total_counts[idx]),
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make a NegativeBinomial distribution that returns a vector of
            # length n_genes
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            counts = numpyro.sample("counts", dist_nb)


Let's dissect the function step by step. On the first part, we define the prior
for the success probability ``p`` as a Beta distribution and the dispersion
parameter ``r`` as a LogNormal distribution, feeding the parameters we set.

Since ``r`` is a ``gene-specific`` parameter (more on that later), we tell
``numpyro`` to expand it to match the number of genes via the ``expand`` method.

Then we define the total dispersion parameter ``r_total`` as the sum of the
individual dispersion parameters ``r``; telling ``numpyro`` that this is a
deterministic variable. This means that once we know the individual dispersion
parameters, we can compute the total dispersion parameter with no uncertainty
associated with it.

.. code-block:: python

    # Define success probability prior (unchanged)
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define dispersion prior using LogNormal instead of Gamma
    r = numpyro.sample(
        "r", 
        dist.LogNormal(r_prior[0], r_prior[1]).expand([n_genes])
    )

    # Rest of the model definition...
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

After defining the priors, we define the likelihood for our model.
Preferentially, we specify three cases for how to evaluate the likelihood:

1. If we have observed data but not a batch size, we condition on the entire
   dataset.
2. If we have a batch size, we use mini-batch training.
3. If we don't have any of the above, we return the predictive distribution.

With these three cases, ``SCRIBE`` can handle both training and posterior
predictive sampling, allowing our custom model to be used as any other model
in the package. Let's go through each case in detail.

1. Observed data but no batch size

.. code-block:: python

    # Define plate for cells total counts
    with numpyro.plate("cells", n_cells):
        # Likelihood for the total counts - one for each cell
        numpyro.sample(
            "total_counts",
            dist.NegativeBinomialProbs(r_total, p),
            obs=total_counts
        )

    # Define plate for cells individual counts
    with numpyro.plate("cells", n_cells):
        # Likelihood for the individual counts - one for each cell
        numpyro.sample(
            "counts",
            dist.DirichletMultinomial(r, total_count=total_counts),
            obs=counts
        )


Key requirements for the model function:

* Must accept ``n_cells`` and ``n_genes`` as first arguments
* Should handle both training (``counts is not None``) and predictive (``counts is None``) cases
* Must use NumPyro primitives for all random variables
* Should support mini-batch training through ``batch_size`` parameter

Defining the Guide
-----------------

The guide function defines the variational family used to approximate the posterior. Following our LogNormal example:

.. code-block:: python

    def nbdm_lognormal_guide(
        n_cells: int,
        n_genes: int,
        p_prior: tuple = (1, 1),
        r_prior: tuple = (0, 1),
        counts=None,
        total_counts=None,
        batch_size=None,
    ):
        # Parameters for p (using Beta)
        alpha_p = numpyro.param(
            "alpha_p",
            jnp.array(p_prior[0]),
            constraint=numpyro.distributions.constraints.positive
        )
        beta_p = numpyro.param(
            "beta_p",
            jnp.array(p_prior[1]),
            constraint=numpyro.distributions.constraints.positive
        )

        # Parameters for r (using LogNormal)
        mu_r = numpyro.param(
            "mu_r",
            jnp.ones(n_genes) * r_prior[0],
            constraint=numpyro.distributions.constraints.real
        )
        sigma_r = numpyro.param(
            "sigma_r",
            jnp.ones(n_genes) * r_prior[1],
            constraint=numpyro.distributions.constraints.positive
        )

        # Sample from variational distributions
        numpyro.sample("p", dist.Beta(alpha_p, beta_p))
        numpyro.sample("r", dist.LogNormal(mu_r, sigma_r))

Key points for the guide:

* Must match model's signature exactly
* Parameters should be registered using ``numpyro.param``
* Use appropriate constraints for parameters
* Sample from variational distributions using same names as model

Specifying Parameter Types
-------------------------

SCRIBE needs to know how to handle different parameters in your model. This is done through the ``param_spec`` dictionary:

.. code-block:: python

    param_spec = {
        "alpha_p": {"type": "global"},
        "beta_p": {"type": "global"},
        "mu_r": {"type": "gene-specific"},
        "sigma_r": {"type": "gene-specific"}
    }

Each parameter must be categorized as one of:

* ``"global"``: Single value shared across all cells/genes
* ``"gene-specific"``: One value per gene
* ``"cell-specific"``: One value per cell

For mixture models, add ``"component_specific": True`` to parameters that vary by component.

Running Inference
----------------

Use ``run_scribe`` with your custom model, guide, and parameter specification:

.. code-block:: python

    results = scribe.run_scribe(
        counts=counts,
        custom_model=nbdm_lognormal_model,
        custom_guide=nbdm_lognormal_guide,
        custom_args={
            "total_counts": jnp.sum(counts, axis=1)
        },
        param_spec=param_spec,
        n_steps=10000,
        batch_size=512,
        prior_params={
            "p_prior": (1, 1),
            "r_prior": (0, 1)
        }
    )

Key arguments:

* ``custom_model``: Your model function
* ``custom_guide``: Your guide function
* ``custom_args``: Additional arguments needed by your model/guide
* ``param_spec``: Parameter type specification
* ``prior_params``: Prior parameters for your model

Working with Results
------------------

Results from custom models are returned as ``CustomResults`` objects, which provide the same interface as built-in models:

.. code-block:: python

    # Get learned parameters
    params = results.params
    
    # Get distributions (requires implementing get_distributions_fn)
    distributions = results.get_distributions()
    
    # Generate posterior samples
    samples = results.get_posterior_samples(n_samples=1000)
    
    # Get predictive samples
    predictions = results.get_predictive_samples()

Optional Extensions
------------------

The ``CustomResults`` class supports several optional extensions:

1. Custom distribution access:

.. code-block:: python

    def get_distributions_fn(params, backend="scipy"):
        if backend == "scipy":
            return {
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.lognorm(
                    s=params['sigma_r'],
                    scale=np.exp(params['mu_r'])
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.LogNormal(params['mu_r'], params['sigma_r'])
            }

    # Pass to run_scribe
    results = scribe.run_scribe(
        ...,
        get_distributions_fn=get_distributions_fn
    )

2. Custom model arguments:

.. code-block:: python

    def get_model_args_fn(results):
        return {
            'n_cells': results.n_cells,
            'n_genes': results.n_genes,
            'my_custom_arg': results.custom_value
        }

    # Pass to run_scribe
    results = scribe.run_scribe(
        ...,
        get_model_args_fn=get_model_args_fn
    )

3. Custom log likelihood function:

.. code-block:: python

    def custom_log_likelihood_fn(counts, params):
        # Compute log likelihood
        return log_prob

    # Pass to run_scribe
    results = scribe.run_scribe(
        ...,
        custom_log_likelihood_fn=custom_log_likelihood_fn
    )

Best Practices
-------------

1. **Model Design**:
   * Start from existing models when possible
   * Keep track of dimensionality (cells vs genes)
   * Use appropriate constraints for parameters
   * Support both training and prediction modes

2. **Guide Design**:
   * Match model parameters exactly
   * Initialize variational parameters sensibly
   * Use mean-field approximation when possible
   * Consider parameter constraints carefully

3. **Parameter Specification**:
   * Be explicit about parameter types
   * Consider dimensionality requirements
   * Document parameter relationships
   * Test with small datasets first

4. **Testing**:
   * Verify model runs with small datasets
   * Check parameter ranges make sense
   * Test both training and prediction
   * Validate results against known cases

Common Issues
------------

1. **Dimension Mismatch**:
   * Check parameter shapes match expectations
   * Verify broadcast operations work correctly
   * Ensure mini-batch handling is correct

2. **Memory Issues**:
   * Use appropriate batch sizes
   * Avoid unnecessary parameter expansion
   * Monitor device memory usage

3. **Numerical Stability**:
   * Use appropriate parameter constraints
   * Consider log-space computations
   * Initialize parameters carefully

4. **Convergence Problems**:
   * Check learning rate and optimization settings
   * Monitor loss during training
   * Verify parameter updates occur

See Also
--------

* :doc:`nbdm` - Details on the base NBDM model
* :doc:`results` - Working with result objects
* NumPyro's `documentation <https://num.pyro.ai/en/stable/>`_ for distribution details
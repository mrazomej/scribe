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

Let's now define the model function. We will walk through each part of the model
function step by step after this code block.

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

        # Define the total dispersion parameter
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
parameter ``r`` as a LogNormal distribution, feeding the parameter arguments we
set.

.. code-block:: python

    # Define success probability prior (unchanged)
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define dispersion prior using LogNormal instead of Gamma
    r = numpyro.sample(
        "r", 
        dist.LogNormal(r_prior[0], r_prior[1]).expand([n_genes])
    )

Since ``r`` is a ``gene-specific`` parameter (more on that later), we tell
``numpyro`` to expand it to match the number of genes via the ``expand`` method.
This means that we assume we have `n_genes` dispersion parameters, all of which
have the same prior distribution.

Next, we define the total dispersion parameter ``r_total`` as the sum of the
individual dispersion parameters ``r``; telling ``NumPyro`` that this is a
deterministic variable. This means that once we know the individual dispersion
parameters, we can compute the total dispersion parameter with no uncertainty
associated with this computation.

.. code-block:: python

    # Define the total dispersion parameter
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

After defining the priors, we define the likelihood for our model.
Preferentially, we specify **three cases** for how to evaluate the likelihood:

1. If we have observed data but not a batch size, we condition on the entire
   dataset.

  - This allows us to use the entire dataset on each training step. However, for
    large datasets, we might run out of memory and crash.

2. If we have a batch size, we use mini-batch training.

  - One of the advantages of using ``NumPyro`` as the backend for ``SCRIBE`` is
    that we can use mini-batch training. This allows us to use a subset of the
    dataset on each training step, which is more memory efficient.

3. If we don't have any of the above, we return the predictive distribution.

  - This allows us to use the fitted model for posterior predictive sampling.

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

The key concept to understand here is the use of ``numpyro.plate``. This is how
``NumPyro`` handles having i.i.d samples. In this case, we have ``n_cells``
observations of both the total counts and the individual counts for each cell.
Thus, when we call ``numpyro.plate("cells", n_cells)``, we first tell
``NumPyro`` the name of the dimension, in this case ``cells``, and then the size
of the dimension, in this case ``n_cells``. This is equivalent to saying that
the likelihood takes the following form:

.. math::

    \pi(U_1, \ldots, U_{n_{cells}} \mid r_i, p) = 
    \prod_{i=1}^{n_{cells}} \pi(U_i \mid r_i, p)
    \tag{1}

where :math:`U_i` is the total counts for cell :math:`i` and :math:`r_i` is
the dispersion parameter for cell :math:`i`.

For this particular model, we have two plates: one for the total counts and one
for the individual counts. Their interpretation is the same: we have ``n_cells``
independent observations of the total counts and the individual counts for each
cell.

Let's now move on to the second case, where we have a batch size.

2. Observed data with batch size

.. code-block:: python

    # Define plate for cells total counts
    with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
        # Likelihood for the total counts - one for each cell
        numpyro.sample(
            "total_counts",
            dist.NegativeBinomialProbs(r_total, p),
            obs=total_counts[idx]
        )

    # Define plate for cells individual counts
    with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
        # Likelihood for the individual counts - one for each cell
        numpyro.sample(
            "counts",
            dist.DirichletMultinomial(r, total_count=total_counts[idx]),
            obs=counts[idx]
        )

The only difference in this case with the previous one is that we now have a
batch size. This means that we are using a subset of the data on each training
step to be more memory efficient. ``NumPyro`` handles this by using the ``idx``
variable to index into the ``total_counts`` and ``counts`` arrays, returning a
random subset of the data on each training step.

.. note::

    This is why it is important for our counts to be in the shape ``(n_cells,
    n_genes)`` for the indexing to work.

Let's now move on to the third case, where we don't have any observed data.

3. Predictive model

.. code-block:: python

    # Predictive model (no obs)
    with numpyro.plate("cells", n_cells):
        # Make a NegativeBinomial distribution that returns a vector of
        # length n_genes
        dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
        counts = numpyro.sample("counts", dist_nb)

For the last case—used for posterior predictive sampling—we use the same
``numpyro.plate`` structure. However, for this case, our objective is to
generate a synthetic dataset given the definition of our model. In our case, the
model likelihood can be expressed either as sampling the total number of UMIs
per cell with a Negative Binomial and then distributing to each gene via a
Dirichlet-Multinomial distribution, or as sampling the individual counts for
each gene and cell with a Negative Binomial distribution (see the
:doc:`../models/nbdm` model for more details). So, on the first step, we define
the distribution we want to sample from. In this case, we have a
``NegativeBinomialProbs`` distribution. ``NumPyro`` automatically vectorizes the
sampling to be of the corresponding size. In our case ``r`` is a vector of
length ``n_genes`` and ``p`` is a scalar, so ``NumPyro`` will sample a vector of
length ``n_genes`` from a ``NegativeBinomialProbs`` distribution. We then use
the ``to_event(1)`` method to tell ``NumPyro`` that a sample from the
``n_genes`` independent Negative Binomial distributions represents a single
cell's worth of counts. In other words, we can think of the ``.to_event(1)``
method as a way to tell ``NumPyro`` that we want to consider our ``n_genes``
negative binomial distributions as a "*multivariate* distribution" that
represents a single cell's worth of counts.


Summary of key requirements for the model function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Must accept ``n_cells`` and ``n_genes`` as first arguments
* Should handle both training (``counts is not None``) and predictive (``counts is None``) cases
* Must use ``NumPyro`` primitives for all random variables
* Should support mini-batch training through ``batch_size`` parameter

Defining the Guide
------------------

``SCRIBE`` specializes in the use of variational inference to approximate the
posterior distribution of our model. Briefly, variational inference is a method
for approximating the posterior distribution of a model by minimizing the
difference between the true posterior and an approximating distribution.
However, computing the "true" difference between the true posterior and our
approximation would require knowing the true posterior, which is what we are
trying to avoid in the first place. Instead, one can show that by minimizing a
functional known as the variational free energy, also known as the negative of
the `evidence lower bound (ELBO)
<https://en.wikipedia.org/wiki/Evidence_lower_bound>`_, we can find an
approximation to the true posterior.

The guide function defines our variational distribution, which will be used to
approximate the posterior distribution of our model. In our case, we will use
what is known as a **mean-field** approximation. This simply means that the
posterior for each of the parameters in our model defined above will be
independent of any other parameters. In other words, we will make the
simplification that each dispersion parameter is independent of the others and
of the success probability. Most likely, this is not true, as genes might have
correlations. However, a simple estimate with humans that have ~20k genes tells
us that if we wanted to fit parameters for all correlations, we would need
~20k x 20k = 400M parameters, making it not only computationally very intensive,
but the number of data we would require to uniquely determine all of these
parameters would be enormous. So, we will live with the limitations of the
mean-field approximation.

Thus, we will define a variational distribution for each of the parameters in
our model. For the success probability, we will use a Beta distribution (a
natural choice given that ``p`` is constrained to the unit interval), and for
the dispersion parameters, we will use a LogNormal distribution (a natural
choice given that ``r`` is constrained to be non-negative and our prior on ``r``
is also LogNormal).

.. note::

    We are free to choose any distribution for the variational distribution. In
    this case, the distributions we chose as priors are natural choices for the
    model, but we could have chosen any other distribution.

Very importantly, the guide function must have the **same signature** as the
model function.  Let's now define the guide function and we will walk through it
step by step after this code block.

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

Let's dissect the guide function step by step. The first thing we do is define
the parameters for the variational distributions. We do this using the
``numpyro.param`` function. This function allows us to register parameters in
our model. For example, the Beta distribution is defined by two parameters,
``alpha`` and ``beta``. For the success probability we register these two
parameters as ``alpha_p`` and ``beta_p``. However, we must indicate ``NumPyro``
the constraints on these parameters. For the Beta distribution, we know that
``alpha`` and ``beta`` must be **strictly positive**, so we use the
``constraint`` argument to tell ``NumPyro`` that our parameters are constrained
to be positive.

We can register these parameters in our model by doing the following:

.. code-block:: python

    alpha_p = numpyro.param(
        "alpha_p", 
        jnp.array(p_prior[0]), 
        constraint=numpyro.distributions.constraints.positive
    )

For the dispersion parameters, we do the equivalent parameter registration, with
the difference that the ``mu`` parameter is unconstrained in the real line, and
the ``sigma`` parameter is constrained to be positive. We also use
``jnp.ones(n_genes)`` to tell ``NumPyro`` that we want to register one parameter
per gene.

.. code-block:: python

    # mu parameter for r
    mu_r = numpyro.param(
        "mu_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=numpyro.distributions.constraints.real)

    # sigma parameter for r
    sigma_r = numpyro.param(
        "sigma_r",
        jnp.ones(n_genes) * r_prior[1],
        constraint=numpyro.distributions.constraints.positive
    )

Finally, we sample from the variational distributions using the same names as
the parameters in our model.

.. code-block:: python

    # Sample from variational distributions
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.LogNormal(mu_r, sigma_r))

Summary of key points for the guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Must match model's signature exactly
* Parameters should be registered using ``numpyro.param``
* Use appropriate constraints for parameters
* Sample from variational distributions using same names as model

Specifying Parameter Types
--------------------------

To be able to index the results object correctly, ``SCRIBE`` needs to know how
to handle different parameters in your model. This is done through the
``param_spec`` dictionary:

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

.. note::

    For mixture models, add ``"component_specific": True`` to parameters that
    vary by component.

This way, ``SCRIBE`` knows how to index the results object correctly, allowing
use to access subset of genes for general diagnostics such as plotting the
posterior predictive check samples.

Running Inference
-----------------

Once we define the ``model``, ``guide`` and ``param_spec``, we can use our model
within the ``SCRIBE`` framework. We simply pass the ``model``, ``guide``,
``param_spec``, and any other arguments to ``run_scribe``.

.. code-block:: python

    results = scribe.run_scribe(
        counts=counts,
        custom_model=nbdm_lognormal_model,
        custom_guide=nbdm_lognormal_guide,
        custom_args={
            "total_counts": jnp.sum(counts, axis=1)
        },
        param_spec=param_spec,
        n_steps=10_000,
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
--------------------

Results from custom models are returned as ``CustomResults`` objects, which
provide the same interface as built-in models:

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
-------------------

The ``CustomResults`` class supports several optional extensions:

1. Custom distribution access. Once we have our variational parameters, stored
   in ``params``, we can use them to define our variational posterior
   distributions. To do so, we define a function that takes ``params`` and
   returns a dictionary of distributions. In our case, we want to be able to
   access the distributions in both ``scipy`` and ``NumPyro`` formats, so we
   have two branches in our function.

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

.. warning::

    Sometimes the parameterization between ``scipy`` and ``NumPyro`` is
    different. Make sure to check the documentation for the distribution you are
    using to make sure you are using the correct parameterization.

2. Custom model arguments. Sometimes we need to pass additional arguments to
   our model. We can do this by defining a function that takes ``results`` and
   returns a dictionary of arguments.

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

3. Custom log likelihood function. Sometimes we need to compute the log
   likelihood of our model manually. We can do this by defining a function that
   takes ``counts`` and ``params`` and returns the log likelihood.

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
--------------

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
-------------

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

* :doc:`../models/nbdm` - Details on the base NBDM model
* :doc:`../results` - Working with result objects
* NumPyro's `documentation <https://num.pyro.ai/en/stable/>`_ for distribution
  details
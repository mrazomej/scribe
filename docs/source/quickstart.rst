Quickstart
==========

This tutorial will walk you through simulating data from a Negative
Binomial-Dirichlet Multinomial (NBDM) model, fitting it with SCRIBE, and
visualizing the results. See all the available models in the :doc:`./models/models`
documentation.

Setup
-----

First, let's import the necessary libraries and set up our directories:

.. code-block:: python

    import os
    import jax
    from jax import random
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    import scribe

    # Define directories
    OUTPUT_DIR = "path/to/output"
    FIG_DIR = "path/to/figures"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

Simulating Data
---------------

Next, we'll simulate data from the NBDM model. We'll generate data for 10,000
cells and 20,000 genes. For memory efficiency, we'll use batching when
generating samples, generating the samples in the GPU using `JAX
<https://jax.readthedocs.io/en/latest/>`_ and then moving the samples to the CPU
using `NumPy <https://numpy.org/>`_.

First, let's set up our simulation parameters:

.. code-block:: python

    # Setup random seed
    rng_key = random.PRNGKey(42)
    
    # Define dimensions
    n_cells = 10_000
    n_genes = 20_000
    batch_size = 4096
    
    # Define prior parameters
    r_prior = (2, 1)  # Shape and rate for Gamma prior on r
    p_prior = (1, 1)  # Alpha and beta for Beta prior on p

Now we'll sample the true parameters from their respective prior distributions:

.. code-block:: python

    # Split random keys
    key1, key2 = random.split(rng_key, 2)
    
    # Sample true parameters
    r_true = random.gamma(key1, r_prior[0], shape=(n_genes,)) / r_prior[1]
    p_true = random.beta(key2, p_prior[0], p_prior[1])

With our true parameters in hand, we can generate the count data. We'll do this
in batches to manage memory usage:

.. code-block:: python

    import numpyro.distributions as dist
    
    counts_true = np.zeros((n_cells, n_genes))
    
    # Sample in batches
    for i in range(0, n_cells, batch_size):
        current_batch_size = min(batch_size, n_cells - i)
        key_batch = random.fold_in(rng_key, i)
        
        # Sample from Negative Binomial distribution
        batch_samples = dist.NegativeBinomialProbs(
            r_true, p_true
        ).sample(key_batch, sample_shape=(current_batch_size,))
        
        counts_true[i:i+current_batch_size] = np.array(batch_samples)

Fitting the Model
-----------------

Now that we have our simulated data, we can fit it using SCRIBE. We'll run the
inference for 25,000 steps using the base NBDM model:

.. code-block:: python

    n_steps = 25_000
    
    # Run SCRIBE inference with the NBDM model (default settings)
    results = scribe.run_scribe(
        counts=counts_true,
        inference_method="svi",     # Use stochastic variational inference
        zero_inflated=False,        # NBDM model has no zero-inflation
        variable_capture=False,     # NBDM model has no variable capture probabilities
        mixture_model=False,        # Not using a mixture model
        r_prior=r_prior,            # Using our defined priors
        p_prior=p_prior,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=42
    )

Visualizing Results
------------------

Let's create some visualizations to assess our model fit. First, let's look at
the ELBO loss history:

.. code-block:: python

    # Plot loss history
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(results.loss_history)
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO loss")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_history.png"), dpi=300)
    plt.show()

.. figure:: _static/images/nbdm_sim/loss_history.png
   :width: 350
   :alt: ELBO loss history
   
   ELBO loss history showing convergence of the model fitting process. The spiky
   nature of the loss is due to the batching process.

We can also compare our inferred parameters to the true values. Let's look at
the posterior distribution for p:

.. code-block:: python

    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    # Get posterior distribution for p
    distributions = results.get_distributions()
    distribution = distributions['p']
    
    # Plot posterior with true value
    scribe.viz.plot_posterior(
        ax,
        distribution,
        ground_truth=p_true,
        ground_truth_color="black",
        color=scribe.viz.colors()["dark_blue"],
        fill_color=scribe.viz.colors()["light_blue"],
    )
    
    ax.set_xlabel("p")
    ax.set_ylabel("posterior density")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "p_posterior.png"), dpi=300)
    plt.show()

.. figure:: _static/images/nbdm_sim/example_p_posterior.png
   :width: 350
   :alt: Posterior distribution for p
   
   Posterior distribution for the :math:`p` parameter. The true value from
   simulation is shown in black.

Let's generate a similar plot for various examples of the inferred :math:`r`
parameter:

.. code-block:: python

    # Select a few genes to visualize
    selected_idx = np.random.choice(n_genes, 9, replace=False)
    
    # Initialize figure
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9))
    
    # Flatten axes
    axes = axes.flatten()
    
    fig.suptitle(r"$r$ parameter posterior distributions", y=1.005, fontsize=18)
    
    # Loop through selected genes
    for i, ax in enumerate(axes):
        # Get the r distribution for this gene
        r_distributions = results.get_distributions(backend="scipy")['r']
        gene_dist = r_distributions[selected_idx[i]]
        
        # Plot distribution
        scribe.viz.plot_posterior(
            ax,
            gene_dist,
            ground_truth=r_true[selected_idx[i]],
            ground_truth_color="black",
            color=scribe.viz.colors()["dark_blue"],
            fill_color=scribe.viz.colors()["light_blue"],
        )
        
        ax.set_xlabel(f"Gene {selected_idx[i]}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "r_posteriors.png"), dpi=300)
    plt.show()

.. figure:: _static/images/nbdm_sim/example_r_posterior.png
   :width: 350
   :alt: Posterior distribution for r
   
   Posterior distribution for multiple examples of the :math:`r` parameter. The
   true value from simulation is shown in black.

Finally, we can generate posterior predictive checks (PPCs) to assess model fit:

.. code-block:: python

    # Generate PPC samples
    n_samples = 500
    ppc = results.ppc_samples(n_samples=n_samples, rng_key=random.PRNGKey(43))
    
    # Select a gene to visualize
    gene_idx = 0
    
    # Plot PPCs for the selected gene
    fig, ax = plt.subplots(figsize=(3.5, 3))
    
    # Compute and plot credible regions
    credible_regions = scribe.stats.compute_histogram_credible_regions(
        ppc['predictive_samples'][:, :, gene_idx],
        credible_regions=[95, 68, 50]
    )
    
    scribe.viz.plot_histogram_credible_regions_stairs(
        ax, 
        credible_regions,
        cmap='Blues',
        alpha=0.5
    )
    
    # Plot observed counts for this gene
    bin_edges = credible_regions['bin_edges']
    hist, _ = np.histogram(counts_true[:, gene_idx], bins=bin_edges)
    hist = hist / hist.sum()  # Normalize
    ax.stairs(hist, bin_edges, color='black', alpha=0.8, label='Observed')
    
    ax.set_xlabel(f"Counts for Gene {gene_idx}")
    ax.set_ylabel("Probability")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ppc_example.png"), dpi=300)
    plt.show()

.. figure:: _static/images/nbdm_sim/example_ppc.png
   :width: 350
   :alt: Posterior predictive checks for a gene
   
   Posterior predictive checks for the data generative process. The distribution
   of counts observed in the simulated data is shown in black. The shades of
   blue show the credible regions for the distribution of counts under the
   posterior predictive distribution.

The visual assessment of the model fit reveals that the model is able to capture
the data generating process. From here, we can continue our analysis with the
inferred parameters.

Comparing Models
---------------

Let's compare the basic NBDM model with a Zero-Inflated Negative Binomial (ZINB)
model to see which fits the data better:

.. code-block:: python

    # Fit the ZINB model
    zinb_results = scribe.run_scribe(
        counts=counts_true,
        inference_method="svi",     # Use stochastic variational inference
        zero_inflated=True,         # Use zero-inflation
        variable_capture=False,
        mixture_model=False,
        r_prior=r_prior,
        p_prior=p_prior,
        gate_prior=(1, 1),          # Prior for dropout probabilities
        n_steps=n_steps,
        batch_size=batch_size,
        seed=42
    )
    
    # Compare models using WAIC
    from scribe.model_comparison import compute_waic
    
    # Compute WAIC for both models
    nbdm_waic = compute_waic(
        results, 
        counts_true, 
        n_samples=100,
        batch_size=batch_size
    )
    
    zinb_waic = compute_waic(
        zinb_results, 
        counts_true, 
        n_samples=100,
        batch_size=batch_size
    )
    
    # Display comparison results
    print(f"NBDM WAIC: {nbdm_waic['waic_2']:.2f}")
    print(f"ZINB WAIC: {zinb_waic['waic_2']:.2f}")
    print(f"Delta WAIC: {zinb_waic['waic_2'] - nbdm_waic['waic_2']:.2f}")
    
    if nbdm_waic['waic_2'] < zinb_waic['waic_2']:
        print("NBDM model fits better (lower WAIC)")
    else:
        print("ZINB model fits better (lower WAIC)")

Working with the Results
-----------------------

The results object provides several ways to access and work with the fitted
model:

1. **Accessing Parameters**: Get direct access to model parameters

.. code-block:: python

    # Access p parameters (Beta distribution)
    p_concentration1 = results.params['p_concentration1']
    p_concentration0 = results.params['p_concentration0']
    
    # Access r parameters (depends on distribution used - e.g., Gamma)
    r_concentration = results.params['r_concentration']
    r_rate = results.params['r_rate']

2. **Working with Subsets of Genes**: You can use indexing to focus on specific
   genes

.. code-block:: python

    # Get results for the first gene
    gene0_results = results[0]
    
    # Get results for genes 0, 10, and 20
    selected_genes = results[[0, 10, 20]]
    
    # Boolean indexing also works
    highly_variable = np.random.choice([True, False], size=n_genes, p=[0.1, 0.9])
    hv_results = results[highly_variable]

3. **Computing Log Likelihoods**: Use the log likelihood function to evaluate model fit

.. code-block:: python

    # Compute log likelihoods for each cell
    log_liks = results.log_likelihood(
        counts_true,
        return_by='cell',
        batch_size=batch_size
    )
    
    # Compute log likelihoods for each gene
    gene_log_liks = results.log_likelihood(
        counts_true,
        return_by='gene',
        batch_size=batch_size
    )

.. warning::

    Never trust any model fit (either from SCRIBE or any other analysis
    pipeline) without at least visualizing how the fit compares to the observed
    data. There are no silver bullets in statistics, and the best assessment of
    any fitting procedure is to visualize how the fit compares to the observed
    data.

This completes our quickstart guide! You've now learned how to:

- Simulate data from the NBDM model
- Fit the model using SCRIBE
- Compare different model types
- Visualize and assess the results
- Work with the ScribeResults object

For more detailed examples and advanced usage, check out our tutorials section.
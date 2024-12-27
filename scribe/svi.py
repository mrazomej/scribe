"""
Stochastic Variational Inference (SVI) module for single-cell RNA sequencing
data analysis.

This module implements a Dirichlet-Multinomial model for scRNA-seq count data
using Numpyro for variational inference.
"""

# Imports for inference
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, TraceMeanField_ELBO

# Imports for results
from .results import *

# Imports for data handling
from typing import Dict, Optional, Union
import scipy.sparse

# Imports for AnnData
from anndata import AnnData

# Imports for model-specific functions
from .models import get_model_and_guide, get_default_priors

# ------------------------------------------------------------------------------
# Stochastic Variational Inference with Numpyro
# ------------------------------------------------------------------------------

def create_svi_instance(
    model_type: str = "nbdm",
    custom_model = None,
    custom_guide = None,
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
):
    """
    Create an SVI instance with the defined model and guide.

    Parameters
    ----------
    model_type : str, optional
        Type of model to use. Options are:
            - "nbdm": Negative Binomial-Dirichlet Multinomial model.
            - "zinb": Zero-Inflated Negative Binomial model.
            - "nbvcp": Negative Binomial with variable capture probability.
            - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
              probability. 
        Ignored if custom_model and custom_guide are provided.
    custom_model : callable, optional
        Custom model function to use instead of built-in models
    custom_guide : callable, optional
        Custom guide function to use instead of built-in guides
    optimizer : numpyro.optim.optimizers, optional
        Optimizer to use. Default is Adam with step_size=0.001
    loss : numpyro.infer.elbo, optional
        Loss function to use. Default is TraceMeanField_ELBO

    Returns
    -------
    numpyro.infer.SVI
        Configured SVI instance ready for training

    Raises
    ------
    ValueError
        If model_type is invalid or if only one of custom_model/custom_guide is
        provided
    """
    # If custom model/guide are provided, use those
    if custom_model is not None or custom_guide is not None:
        if custom_model is None or custom_guide is None:
            raise ValueError(
                "Both custom_model and custom_guide must be provided together"
            )
        model_fn = custom_model
        guide_fn = custom_guide
    
    # Otherwise use built-in models from the registry
    else:
        model_fn, guide_fn = get_model_and_guide(model_type)

    return SVI(
        model_fn,
        guide_fn,
        optimizer,
        loss=loss
    )

# ------------------------------------------------------------------------------

def run_inference(
    svi_instance: SVI,
    rng_key: random.PRNGKey,
    counts: jnp.ndarray,
    n_steps: int = 100_000,
    batch_size: int = 512,
    model_type: str = "nbdm",
    prior_params: Optional[Dict] = None,
    cells_axis: int = 0,
    custom_args: Optional[Dict] = None,
) -> numpyro.infer.svi.SVIRunResult:
    """
    Run stochastic variational inference on the provided count data.

    Parameters
    ----------
    svi_instance : numpyro.infer.SVI
        Configured SVI instance for running inference. NOTE: Make sure that this
        was created with the same model and guide as the one being used here!
    rng_key : jax.random.PRNGKey
        Random number generator key
    counts : jax.numpy.ndarray
        Count matrix. If cells_axis=0 (default), shape is (n_cells, n_genes).
        If cells_axis=1, shape is (n_genes, n_cells).
    n_steps : int, optional
        Number of optimization steps. Default is 100,000
    batch_size : int, optional
        Mini-batch size for stochastic optimization. Default is 512
    model_type : str, optional
        Type of model being used. Built-in options are:
        - "nbdm": Negative Binomial-Dirichlet Multinomial model
        - "zinb": Zero-Inflated Negative Binomial model
    prior_params : Dict, optional
        Dictionary of prior parameters specific to the model.
        For built-in models:
        - "nbdm": {'p_prior': (1,1), 'r_prior': (2,0.1)}
        - "zinb": {'p_prior': (1,1), 'r_prior': (2,0.1), 'gate_prior': (1,1)}
        For custom models: any parameters required by the model.
    cells_axis : int, optional
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns.
    custom_args : Dict, optional
        Dictionary of additional arguments to pass to any custom model.

    Returns
    -------
    numpyro.infer.svi.SVIRunResult
        Results from the SVI run containing optimized parameters and loss
        history
    """
    # Extract dimensions and compute total counts based on cells axis
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        total_counts = counts.sum(axis=1)
    else:
        n_genes, n_cells = counts.shape
        total_counts = counts.sum(axis=0)
        counts = counts.T  # Transpose to make cells rows for model

    # Set default prior parameters for built-in models if none provided
    if prior_params is None:
        prior_params = get_default_priors(model_type)

    # Prepare base arguments that all models should receive
    model_args = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'counts': counts,
        'batch_size': batch_size,
    }

    # Add model-specific parameters
    if model_type == "nbdm":
        model_args['total_counts'] = total_counts
    
    # Add all prior parameters to model args
    model_args.update(prior_params)

    # Add any custom arguments
    if custom_args is not None:
        model_args.update(custom_args)

    # Run the inference algorithm
    return svi_instance.run(
        rng_key,
        n_steps,
        **model_args
    )

# ------------------------------------------------------------------------------
# Convert variational parameters to distributions
# ------------------------------------------------------------------------------

def params_to_dist(params: Dict, model: str) -> Dict[str, dist.Distribution]:
    """
    Convert variational parameters to their corresponding Numpyro distributions.
    
    Parameters
    ----------
    params : Dict
        Dictionary containing variational parameters with keys:
        - alpha_p, beta_p: Beta distribution parameters for p
        - alpha_r, beta_r: Gamma distribution parameters for r
    model : str
        Model type. Must be one of: 'nbdm', 'zinb'
        
    Returns
    -------
    Dict[str, dist.Distribution]
        Dictionary mapping parameter names to their distributions:
        - 'p': Beta distribution
        - 'r': Gamma distribution
    """
    if model == "nbdm":
        return {
            'p': dist.Beta(params['alpha_p'], params['beta_p']),
            'r': dist.Gamma(params['alpha_r'], params['beta_r'])
        }
    elif model == "zinb":
        return {
            'p': dist.Beta(params['alpha_p'], params['beta_p']),
            'r': dist.Gamma(params['alpha_r'], params['beta_r']),
            'gate': dist.Beta(params['alpha_gate'], params['beta_gate'])
        }
    elif model == "nbvcp":
        return {
            'p': dist.Beta(params['alpha_p'], params['beta_p']),
            'r': dist.Gamma(params['alpha_r'], params['beta_r']),
            'p_capture': dist.Beta(
                params['alpha_p_capture'], params['beta_p_capture']
            )
        }
    elif model == "zinbvcp":
        return {
            'p': dist.Beta(params['alpha_p'], params['beta_p']),
            'r': dist.Gamma(params['alpha_r'], params['beta_r']),
            'p_capture': dist.Beta(
                params['alpha_p_capture'], params['beta_p_capture']
            ),
            'gate': dist.Beta(params['alpha_gate'], params['beta_gate'])
        }
    else:
        raise ValueError(f"Invalid model type: {model}")

# ------------------------------------------------------------------------------
# SCRIBE inference pipeline
# ------------------------------------------------------------------------------

def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    model_type: str = "nbdm",
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_steps: int = 100_000,
    batch_size: int = 512,
    prior_params: Optional[Dict] = None,
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    cells_axis: int = 0,
    layer: Optional[str] = None,
) -> Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]:
    """Run the complete SCRIBE inference pipeline.
    
    Parameters
    ----------
    counts : Union[jax.numpy.ndarray, AnnData]
        Either a count matrix or an AnnData object. 
        - If ndarray and cells_axis=0 (default), shape is (n_cells, n_genes)
        - If ndarray and cells_axis=1, shape is (n_genes, n_cells)
        - If AnnData, will use .X or specified layer
    model_type : str
        Type of model to use. Options are:
        - "nbdm": Negative Binomial-Dirichlet Multinomial model
        - "zinb": Zero-Inflated Negative Binomial model
        - "nbvcp": Negative Binomial with variable capture probability
        - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
          probability
    prior_params : Dict, optional
        Dictionary of prior parameters specific to the model.
        For built-in models:
        - "nbdm": {'p_prior': (1,1), 'r_prior': (2,0.1)}
        - "zinb": {'p_prior': (1,1), 'r_prior': (2,0.1), 'gate_prior': (1,1)}
        - "nbvcp": {'p_prior': (1,1), 'r_prior': (2,0.1), 'p_capture_prior': (1,1)}
        - "zinbvcp": {'p_prior': (1,1), 'r_prior': (2,0.1), 'p_capture_prior': (1,1),
                      'gate_prior': (1,1)}
    loss : numpyro.infer.elbo, optional
        Loss function to use for the SVI. Default is TraceMeanField_ELBO
    optimizer : numpyro.optim.optimizers, optional
        Optimizer to use for SVI. Default is Adam with step_size=0.001
    cells_axis : int, optional
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns.
    layer : str, optional
        If counts is AnnData, specifies which layer to use. If None, uses .X

    Returns
    -------
    Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]
        Results container with inference results and optional metadata
    """
    # Set default prior parameters if none provided
    if prior_params is None:
        prior_params = get_default_priors(model_type)
    
    # Handle AnnData input as before
    if hasattr(counts, "obs"):
        adata = counts
        # Get counts from specified layer or .X
        count_data = adata.layers[layer] if layer else adata.X
        # Convert to dense array if sparse
        if scipy.sparse.issparse(count_data):
            count_data = count_data.toarray()
        count_data = jnp.array(count_data)
    else:
        count_data = counts
        adata = None

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = count_data.shape
    else:
        n_genes, n_cells = count_data.shape
        count_data = count_data.T  # Transpose to make cells rows

    # Create SVI instance
    svi = create_svi_instance(
        model_type=model_type,
        optimizer=optimizer,
        loss=loss
    )

    # Run inference
    svi_results = run_inference(
        svi,
        rng_key,
        count_data,
        n_steps=n_steps,
        batch_size=batch_size,
        model_type=model_type,
        prior_params=prior_params,
        cells_axis=0
    )

    # Create appropriate results class
    results_class = {
        "nbdm": NBDMResults,
        "zinb": ZINBResults,
        "nbvcp": NBVCPResults,
        "zinbvcp": ZINBVCPResults
    }.get(model_type)
    
    if results_class is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create results object
    if adata is not None:
        results = results_class.from_anndata(
            adata=adata,
            params=svi_results.params,
            loss_history=svi_results.losses,
            model_type=model_type
        )
    else:
        results = results_class(
            params=svi_results.params,
            loss_history=svi_results.losses,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type
        )

    return results

# ------------------------------------------------------------------------------
# Continue training from a previous SCRIBE results object
# ------------------------------------------------------------------------------

def rerun_scribe(
    results: Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults],
    counts: Union[jnp.ndarray, "AnnData"],
    n_steps: int = 100_000,
    batch_size: int = 512,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    prior_params: Optional[Dict] = None,
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    cells_axis: int = 0,
    layer: Optional[str] = None,
) -> Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]:
    """
    Continue training from a previous SCRIBE results object.

    This function creates a new results object starting from the parameters
    of a previous run. This is useful when:
        1. You want to run more iterations to improve convergence
        2. You're doing incremental training as new data arrives
        3. You're fine-tuning a model that was previously trained

    Parameters
    ----------
    results : Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]
        Previous results object containing trained parameters
    counts : Union[jax.numpy.ndarray, AnnData]
        Either a count matrix or an AnnData object. 
        - If ndarray and cells_axis=0 (default), shape is (n_cells, n_genes)
        - If ndarray and cells_axis=1, shape is (n_genes, n_cells)
        - If AnnData, will use .X or specified layer
    n_steps : int, optional
        Number of optimization steps (default: 100,000)
    batch_size : int, optional
        Mini-batch size for stochastic optimization (default: 512)
    rng_key : random.PRNGKey, optional
        Random key for reproducibility (default: PRNGKey(42))
    prior_params : Dict, optional
        Dictionary of prior parameters. If None, uses the same priors as the
        original training
    loss : numpyro.infer.elbo, optional
        Loss function to use (default: TraceMeanField_ELBO)
    optimizer : numpyro.optim.optimizers, optional
        Optimizer to use (default: Adam with step_size=0.001)
    cells_axis : int, optional
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns
    layer : str, optional
        If counts is AnnData, specifies which layer to use. If None, uses .X

    Returns
    -------
    Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]
        A new results object containing the continued training results
    """
    # Handle AnnData input
    if hasattr(counts, "obs"):
        adata = counts
        count_data = adata.layers[layer] if layer else adata.X
        if scipy.sparse.issparse(count_data):
            count_data = count_data.toarray()
        count_data = jnp.array(count_data)
    else:
        count_data = counts
        adata = None

    # Extract dimensions
    if cells_axis == 0:
        n_cells, n_genes = count_data.shape
    else:
        n_genes, n_cells = count_data.shape
        count_data = count_data.T  # Transpose to make cells rows

    # Verify dimensions match
    if n_genes != results.n_genes:
        raise ValueError(
            f"Number of genes in counts ({n_genes}) doesn't match model "
            f"({results.n_genes})"
        )

    # Get model and guide functions
    model, guide = get_model_and_guide(results.model_type)

    # Create SVI instance
    svi = SVI(
        model,
        guide,
        optimizer,
        loss=loss
    )

    # Use default prior parameters if none provided
    if prior_params is None:
        prior_params = get_default_priors(results.model_type)

    # Prepare model arguments - specific to each model type
    model_args = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'counts': count_data,
        'batch_size': batch_size,
    }

    # Add model-specific parameters
    if results.model_type == "nbdm":
        model_args['total_counts'] = count_data.sum(axis=1)
    
    # Add all prior parameters to model args
    model_args.update(prior_params)

    # Extract current parameters from results as init_params
    model_args['init_params'] = results.params

    # Initialize the SVI state with current parameters
    svi_state = svi.init(rng_key, **model_args)

    # Run the inference algorithm from current state
    svi_results = svi.run(
        rng_key,
        n_steps,
        init_state=svi_state,
        **model_args
    )

    # Create the appropriate results class
    results_class = type(results)
    
    # Combine the loss histories
    combined_losses = jnp.concatenate([
        results.loss_history,
        svi_results.losses
    ])

    # Create and return new results object
    if adata is not None:
        return results_class.from_anndata(
            adata=adata,
            params=svi_results.params,
            loss_history=combined_losses,
            model_type=results.model_type
        )
    else:
        return results_class(
            params=svi_results.params,
            loss_history=combined_losses,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=results.model_type
        )
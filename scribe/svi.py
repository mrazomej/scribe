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
from .model_registry import get_model_and_guide

# ------------------------------------------------------------------------------
# Stochastic Variational Inference with Numpyro
# ------------------------------------------------------------------------------

def create_svi_instance(
    model_type: Optional[str] = None,
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
):
    """
    Create an SVI instance with the defined model and guide.

    Parameters
    ----------
    model_type : str, optional
        Type of model to use. Must be one of:
            - "nbdm": Negative Binomial model
            - "zinb": Zero-Inflated Negative Binomial model
            - "nbvcp": Negative Binomial with variable capture probability
            - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
              probability
            - Mixture variants with "_mix" suffix (e.g. "nbdm_mix")
    optimizer : numpyro.optim.optimizers, optional
        Optimizer to use for stochastic optimization. Default is Adam with 
        step_size=0.001
    loss : numpyro.infer.elbo, optional
        Loss function for variational inference. Default is TraceMeanField_ELBO()

    Returns
    -------
    numpyro.infer.SVI
        Configured SVI instance ready for training

    Raises
    ------
    ValueError
        If model_type is invalid
    """
    # Get model and guide functions
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
    model_type: str = "nbdm",
    model_config: Optional[ModelConfig] = None,
    n_steps: int = 100_000,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    stable_update: bool = True,
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
        - "nbvcp": Negative Binomial with variable capture probability
        - "zinbvcp": Zero-Inflated Negative Binomial with variable capture
          probability
        - "nbdm_mix": Negative Binomial-Dirichlet Multinomial Mixture Model
    n_components : int, optional
        Number of components to fit for mixture models. Default is None.
    prior_params : Dict, optional
        Dictionary of prior parameters specific to the model.
    cells_axis : int, optional
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns.
    custom_args : Dict, optional
        Dictionary of additional arguments to pass to any custom model.
    stable_update : bool, optional
        Whether to use stable update method. Default is True. If true,  returns
        the current state if the the loss or the new state contains invalid
        values.

    Returns
    -------
    numpyro.infer.svi.SVIRunResult
        Results from the SVI run containing optimized parameters and loss
        history
    """
    # Extract dimensions and compute total counts based on cells axis
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
    else:
        n_genes, n_cells = counts.shape
        counts = counts.T  # Transpose to make cells rows for model

    # Prepare base arguments that all models should receive
    model_args = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'counts': counts,
        'batch_size': batch_size,
        'model_config': model_config,
    }

    # Add model-specific parameters
    if model_type == "nbdm":
        model_args['total_counts'] = counts.sum(axis=1)

    # Check if mixture model
    if model_type is not None and "mix" in model_type:
        # Check if n_components is provided
        if model_config.n_components is None:
            raise ValueError(
                f"n_components must be specified for mixture model {model_type}"
            )
        elif model_config.n_components == 1:
            raise ValueError(
                f"n_components must be greater than 1 for mixture model {model_type}"
            )

    # Run the inference algorithm
    return svi_instance.run(
        rng_key,
        n_steps,
        stable_update=stable_update,
        **model_args
    )

# ------------------------------------------------------------------------------
# SCRIBE inference pipeline
# ------------------------------------------------------------------------------

def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    model_type: str = "nbdm",
    n_components: Optional[int] = None,
    # Distribution choices
    r_dist: str = "gamma",
    p_dist: str = "beta",
    # Prior parameters (with intuitive defaults)
    r_prior: Optional[tuple] = None,
    p_prior: Optional[tuple] = None,
    gate_prior: Optional[tuple] = None,
    p_capture_prior: Optional[tuple] = None,
    mixing_prior: Union[float, tuple] = 1.0,
    # Training parameters
    n_steps: int = 100_000,
    batch_size: Optional[int] = None,
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    # Data handling options
    cells_axis: int = 0,
    layer: Optional[str] = None,
    # Extra options
    seed: int = 42,
    stable_update: bool = True,
) -> ScribeResults:
    """
    Run SCRIBE inference with simple configuration.
    
    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing counts
    model_type : str, default="nbdm"
        Model type to use: "nbdm", "zinb", "nbvcp", "zinbvcp", or with "_mix"
        suffix  
    n_components : Optional[int], default=None
        Number of mixture components (required for mixture models)
    r_dist : str, default="gamma"
        Distribution for dispersion parameter: "gamma" or "lognormal"
    p_dist : str, default="beta"
        Distribution for success probability parameter (currently only "beta"
        supported)
    r_prior : Optional[tuple], default=None
        Prior parameters for r distribution. If None, uses sensible defaults
        based on r_dist: - For gamma: (2, 0.1) (shape, rate) - For lognormal:
        (0, 1) (mu, sigma)
    p_prior : Optional[tuple], default=None
        Prior parameters for p distribution (alpha, beta). Default: (1, 1)
    gate_prior : Optional[tuple], default=None
        Prior parameters for dropout gate (alpha, beta). Default: (1, 1)
    p_capture_prior : Optional[tuple], default=None
        Prior parameters for capture probability (alpha, beta). Default: (1, 1)
    mixing_prior : Union[float, tuple], default=1.0
        Prior concentration for mixture weights. If float, same for all
        components.
    n_steps : int, default=100_000
        Number of optimization steps
    learning_rate : float, default=0.001
        Learning rate for optimizer
    batch_size : Optional[int], default=None
        Mini-batch size. If None, uses full dataset.
    cells_axis : int, default=0
        Axis for cells (0=rows, 1=columns)
    layer : Optional[str], default=None
        Layer in AnnData to use (if counts is AnnData)
    seed : int, default=42
        Random seed
    stable_update : bool, default=True
        Whether to use stable parameter updates
        
    Returns
    -------
    ScribeResults
        Results containing fitted model parameters and metadata
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
        count_data = count_data.T
    
    # Set default priors based on distribution choices
    if r_prior is None:
        r_prior = (2, 0.1) if r_dist == "gamma" else (1, 1)
    if p_prior is None:
        p_prior = (1, 1)
    if "zinb" in model_type and gate_prior is None:
        gate_prior = (1, 1)
    if "vcp" in model_type and p_capture_prior is None:
        p_capture_prior = (1, 1)
    
    # Create random key
    rng_key = random.PRNGKey(seed)
    
    # Create model config based on model type and specified distributions
    if r_dist == "gamma":
        r_dist_model = dist.Gamma(*r_prior)
    elif r_dist == "lognormal":
        r_dist_model = dist.LogNormal(*r_prior)
    else:
        raise ValueError(f"Unsupported r_dist: {r_dist}")
        
    if p_dist == "beta":
        p_dist_model = dist.Beta(*p_prior)
    else:
        raise ValueError(f"Unsupported p_dist: {p_dist}")
    
    # Configure gate distribution if needed
    gate_dist_model = None
    if "zinb" in model_type:
        if gate_prior is None:
            gate_prior = (1, 1)
        gate_dist_model = dist.Beta(*gate_prior)
            
    # Configure capture probability distribution if needed
    p_capture_dist_model = None
    if "vcp" in model_type:
        if p_capture_prior is None:
            p_capture_prior = (1, 1)
        p_capture_dist_model = dist.Beta(*p_capture_prior)
    
    # Create model_config with all the configurations
    model_config = ModelConfig(
        base_model=model_type,
        r_distribution_model=r_dist_model,
        r_distribution_guide=r_dist_model,
        r_param_prior=r_prior,
        r_param_guide=r_prior,
        p_distribution_model=p_dist_model,
        p_distribution_guide=p_dist_model,
        p_param_prior=p_prior,
        p_param_guide=p_prior,
        gate_distribution_model=gate_dist_model,
        gate_distribution_guide=gate_dist_model,
        gate_param_prior=gate_prior if gate_dist_model else None,
        gate_param_guide=gate_prior if gate_dist_model else None,
        p_capture_distribution_model=p_capture_dist_model,
        p_capture_distribution_guide=p_capture_dist_model,
        p_capture_param_prior=p_capture_prior if p_capture_dist_model else None,
        p_capture_param_guide=p_capture_prior if p_capture_dist_model else None,
        n_components=n_components
    )
    
    # Get model and guide functions
    model, guide = get_model_and_guide(model_type)
    
    # Set up SVI
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
    
    # Prepare model arguments
    model_args = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'counts': count_data,
        'batch_size': batch_size,
        'model_config': model_config
    }
    
    # Add total_counts for NBDM model
    if model_type == "nbdm":
        model_args['total_counts'] = count_data.sum(axis=1)
    
    # Run inference
    svi_results = svi.run(
        rng_key,
        n_steps,
        stable_update=stable_update,
        **model_args
    )
    
    # Create results object
    if adata is not None:
        results = ScribeResults.from_anndata(
            adata=adata,
            params=svi_results.params,
            loss_history=svi_results.losses,
            model_type=model_type,
            model_config=model_config,
            n_components=n_components,
            prior_params={
                'r_prior': r_prior,
                'p_prior': p_prior,
                'gate_prior': gate_prior if "zinb" in model_type else None,
                'p_capture_prior': p_capture_prior if "vcp" in model_type else None,
                'mixing_prior': mixing_prior if n_components is not None else None
            }
        )
    else:
        results = ScribeResults(
            params=svi_results.params,
            loss_history=svi_results.losses,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            n_components=n_components,
            prior_params={
                'r_prior': r_prior,
                'p_prior': p_prior,
                'gate_prior': gate_prior if "zinb" in model_type else None,
                'p_capture_prior': p_capture_prior if "vcp" in model_type else None,
                'mixing_prior': mixing_prior if n_components is not None else None
            }
        )
    
    return results

# # ------------------------------------------------------------------------------
# # Continue training from a previous SCRIBE results object
# # ------------------------------------------------------------------------------

# def rerun_scribe(
#     results: Union[
#         NBDMResults, 
#         ZINBResults, 
#         NBVCPResults, 
#         ZINBVCPResults, 
#         NBDMMixtureResults,
#         ZINBMixtureResults,
#         NBVCPMixtureResults,
#         ZINBVCPMixtureResults,
#         NBDMMixtureLogResults,
#         CustomResults
#     ],
#     counts: Union[jnp.ndarray, "AnnData"],
#     n_steps: int = 100_000,
#     batch_size: int = 512,
#     rng_key: random.PRNGKey = random.PRNGKey(42),
#     custom_args: Optional[Dict] = None,
#     loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
#     optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
#     cells_axis: int = 0,
#     layer: Optional[str] = None,
#     stable_update: bool = True,
# ) -> Union[
#     NBDMResults, 
#     ZINBResults, 
#     NBVCPResults, 
#     ZINBVCPResults, 
#     NBDMMixtureResults,
#     ZINBMixtureResults,
#     NBVCPMixtureResults,
#     ZINBVCPMixtureResults,
#     NBDMMixtureLogResults,
#     CustomResults
# ]:
#     """
#     Continue training from a previous SCRIBE results object.

#     This function creates a new results object starting from the parameters
#     of a previous run. This is useful when:
#         1. You want to run more iterations to improve convergence
#         2. You're doing incremental training as new data arrives
#         3. You're fine-tuning a model that was previously trained

#     Parameters
#     ----------
#     results : Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]
#         Previous results object containing trained parameters
#     counts : Union[jax.numpy.ndarray, AnnData]
#         Either a count matrix or an AnnData object. 
#         - If ndarray and cells_axis=0 (default), shape is (n_cells, n_genes)
#         - If ndarray and cells_axis=1, shape is (n_genes, n_cells)
#         - If AnnData, will use .X or specified layer
#     n_steps : int, optional
#         Number of optimization steps (default: 100,000)
#     batch_size : int, optional
#         Mini-batch size for stochastic optimization (default: 512)
#     rng_key : random.PRNGKey, optional
#         Random key for reproducibility (default: PRNGKey(42))
#     custom_args : Dict, optional
#         Dictionary of custom arguments for the model
#     loss : numpyro.infer.elbo, optional
#         Loss function to use (default: TraceMeanField_ELBO)
#     optimizer : numpyro.optim.optimizers, optional
#         Optimizer to use (default: Adam with step_size=0.001)
#     cells_axis : int, optional
#         Axis along which cells are arranged. 0 means cells are rows (default),
#         1 means cells are columns
#     layer : str, optional
#         If counts is AnnData, specifies which layer to use. If None, uses .X
#     stable_update : bool, optional
#         Whether to use stable update method. Default is True. If true,  returns
#         the current state if the the loss or the new state contains invalid
#         values.

#     Returns
#     -------
#     Union[NBDMResults, ZINBResults, NBVCPResults, ZINBVCPResults]
#         A new results object containing the continued training results
#     """
#     # Handle AnnData input
#     if hasattr(counts, "obs"):
#         adata = counts
#         count_data = adata.layers[layer] if layer else adata.X
#         if scipy.sparse.issparse(count_data):
#             count_data = count_data.toarray()
#         count_data = jnp.array(count_data)
#     else:
#         count_data = jnp.array(counts)
#         adata = None

#     # Extract dimensions and transpose if needed
#     if cells_axis == 0:
#         n_cells, n_genes = count_data.shape
#     else:
#         n_genes, n_cells = count_data.shape
#         count_data = count_data.T

#     # Verify dimensions match
#     if n_genes != results.n_genes:
#         raise ValueError(
#             f"Number of genes in counts ({n_genes}) doesn't match model "
#             f"({results.n_genes})"
#         )

#     # Get model and guide functions
#     model, guide = results.get_model_and_guide()

#     # Create SVI instance
#     svi = SVI(model, guide, optimizer, loss=loss)

#     # Get base model arguments from results class
#     model_args = results.get_model_args()
    
#     # Update with runtime-specific arguments
#     model_args.update({
#         'counts': count_data,
#         'batch_size': batch_size,
#     })

#     # Add prior parameters
#     model_args.update(results.prior_params)

#     # Add model-specific parameters
#     if results.model_type == "nbdm":
#         model_args['total_counts'] = count_data.sum(axis=1)
   
#     # Add prior current parameters
#     model_args['init_params'] = results.params

#     # Add custom arguments
#     if custom_args is not None:
#         model_args.update(custom_args)
    
#     # Initialize and run SVI
#     svi_state = svi.init(rng_key, **model_args)
#     svi_results = svi.run(
#         rng_key,
#         n_steps,
#         init_state=svi_state,
#         stable_update=stable_update,
#         **model_args
#     )

#     # Create new results object
#     results_class = type(results)
#     combined_losses = jnp.concatenate([results.loss_history, svi_results.losses])

#     # Build common kwargs
#     results_kwargs = {
#         "params": svi_results.params,
#         "loss_history": combined_losses,
#         "model_type": results.model_type,
#         "param_spec": results.param_spec,
#         "n_components": results.n_components,
#         "prior_params": results.prior_params
#     }

#     # Add custom model specific arguments if needed
#     if results_class.__name__ == "CustomResults":
#         results_kwargs.update({
#             "custom_model": results.custom_model,
#             "custom_guide": results.custom_guide,
#         })

#     if adata is not None:
#         return results_class.from_anndata(
#             adata=adata,
#             **results_kwargs
#         )
#     else:
#         return results_class(
#             n_cells=n_cells,
#             n_genes=n_genes,
#             **results_kwargs
#         )

# ------------------------------------------------------------------------------
# Convert variational parameters to distributions
# ------------------------------------------------------------------------------

def params_to_dist(params: Dict, model: str, backend: str = "scipy") -> Dict[str, Any]:
    """
    Convert variational parameters to their corresponding distributions.
    
    Parameters
    ----------
    params : Dict
        Dictionary containing variational parameters with keys:
            - alpha_p, beta_p: Beta distribution parameters for p
            - alpha_r, beta_r: Gamma distribution parameters for r
            - alpha_mixing: Dirichlet distribution parameters for mixing weights
            - alpha_gate, beta_gate: Beta distribution parameters for gate
    model : str
        Model type. Must be one of: 'nbdm', 'zinb', 'nbvcp', 'zinbvcp',
        'nbdm_mix', 'zinb_mix'
    backend : str, optional
        Statistical package to use for distributions. Must be one of:
            - "scipy": Returns scipy.stats distributions (default)
            - "numpyro": Returns numpyro.distributions
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to their distributions
    """
    if model == "nbdm":
        if backend == "scipy":
            return {
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r'])
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r'])
            }
    elif model == "zinb":
        if backend == "scipy":
            return {
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r']),
                'gate': stats.beta(params['alpha_gate'], params['beta_gate'])
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r']),
                'gate': dist.Beta(params['alpha_gate'], params['beta_gate'])
            }
    elif model == "nbvcp":
        if backend == "scipy":
            return {
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r']),
                'p_capture': stats.beta(params['alpha_p_capture'], params['beta_p_capture'])
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r']),
                'p_capture': dist.Beta(params['alpha_p_capture'], params['beta_p_capture'])
            }
    elif model == "zinbvcp":
        if backend == "scipy":
            return {
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r']),
                'p_capture': stats.beta(params['alpha_p_capture'], params['beta_p_capture']),
                'gate': stats.beta(params['alpha_gate'], params['beta_gate'])
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r']),
                'p_capture': dist.Beta(params['alpha_p_capture'], params['beta_p_capture']),
                'gate': dist.Beta(params['alpha_gate'], params['beta_gate'])
            }
    elif model == "nbdm_mix":
        if backend == "scipy":
            return {
                'mixing_probs': stats.dirichlet(params['alpha_mixing']),
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r'])
            }
        elif backend == "numpyro":
            return {
                'mixing_probs': dist.Dirichlet(params['alpha_mixing']),
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r'])
            }
    elif model == "zinb_mix":
        if backend == "scipy":
            return {
                'mixing_probs': stats.dirichlet(params['alpha_mixing']),
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r']),
                'gate': stats.beta(params['alpha_gate'], params['beta_gate'])
            }
        elif backend == "numpyro":
            return {
                'mixing_probs': dist.Dirichlet(params['alpha_mixing']),
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r']),
                'gate': dist.Beta(params['alpha_gate'], params['beta_gate'])
            }
    elif model == "nbvcp_mix":
        if backend == "scipy":
            return {
                'mixing_probs': stats.dirichlet(params['alpha_mixing']),
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r']),
                'p_capture': stats.beta(params['alpha_p_capture'], params['beta_p_capture'])
            }
        elif backend == "numpyro":
            return {
                'mixing_probs': dist.Dirichlet(params['alpha_mixing']),
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r']),
                'p_capture': dist.Beta(params['alpha_p_capture'], params['beta_p_capture'])
            }
    elif model == "zinbvcp_mix":
        if backend == "scipy":
            return {
                'mixing_probs': stats.dirichlet(params['alpha_mixing']),
                'p': stats.beta(params['alpha_p'], params['beta_p']),
                'r': stats.gamma(params['alpha_r'], loc=0, scale=1/params['beta_r']),
                'p_capture': stats.beta(params['alpha_p_capture'], params['beta_p_capture']),
                'gate': stats.beta(params['alpha_gate'], params['beta_gate'])
            }
        elif backend == "numpyro":
            return {
                'mixing_probs': dist.Dirichlet(params['alpha_mixing']),
                'p': dist.Beta(params['alpha_p'], params['beta_p']),
                'r': dist.Gamma(params['alpha_r'], params['beta_r']),
                'p_capture': dist.Beta(params['alpha_p_capture'], params['beta_p_capture']),
                'gate': dist.Beta(params['alpha_gate'], params['beta_gate'])
            }
    else:
        raise ValueError(f"Invalid model type: {model}")

    raise ValueError(f"Invalid backend: {backend}")
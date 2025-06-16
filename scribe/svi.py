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
from .results_svi import ScribeSVIResults

# Imports for data handling
from typing import Dict, Optional, Union
import scipy.sparse

# Imports for AnnData
from anndata import AnnData

# Imports for model-specific functions
from .model_registry import get_model_and_guide
from .model_config import ConstrainedModelConfig

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
    model_config: Optional[ConstrainedModelConfig] = None,
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
    # Model configuration
    zero_inflated: bool = False,
    variable_capture: bool = False, 
    mixture_model: bool = False,
    n_components: Optional[int] = None,
    # Distribution choices
    r_dist: str = "gamma",
    # Prior parameters
    r_prior: Optional[tuple] = None,
    p_prior: Optional[tuple] = None,
    gate_prior: Optional[tuple] = None,
    p_capture_prior: Optional[tuple] = None,
    mixing_prior: Optional[tuple] = None,
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
    r_guide: Optional[str] = None,
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
) -> ScribeSVIResults:
    """
    Run SCRIBE inference pipeline to fit a probabilistic model to single-cell
    RNA sequencing data.
    
    This function provides a high-level interface to configure and run SCRIBE
    models. It handles data preprocessing, model configuration, variational
    inference, and results packaging.
    
    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing counts. If AnnData, counts
        should be in .X or specified layer. Shape should be (cells, genes) if
        cells_axis=0, or (genes, cells) if cells_axis=1.
        
    Model Configuration:
    -------------------
    zero_inflated : bool, default=False
        Whether to use zero-inflated negative binomial (ZINB) model to account
        for dropout events
    variable_capture : bool, default=False 
        Whether to model cell-specific mRNA capture efficiencies
    mixture_model : bool, default=False
        Whether to use mixture model components for heterogeneous populations
    n_components : Optional[int], default=None
        Number of mixture components. Required if mixture_model=True.
        
    Distribution Choices:
    -------------------
    r_dist : str, default="gamma"
        Distribution family for dispersion parameter. Options:
            - "gamma": Gamma distribution (default)
            - "lognormal": Log-normal distribution
        
    Prior Parameters:
    ----------------
    r_prior : Optional[tuple], default=None
        Prior parameters for dispersion (r) distribution:
            - For gamma: (shape, rate), defaults to (2, 0.1)
            - For lognormal: (mu, sigma), defaults to (1, 1)
    p_prior : Optional[tuple], default=None
        Prior parameters (alpha, beta) for success probability Beta
        distribution. Defaults to (1, 1).
    gate_prior : Optional[tuple], default=None
        Prior parameters (alpha, beta) for dropout gate Beta distribution. Only
        used if zero_inflated=True. Defaults to (1, 1).
    p_capture_prior : Optional[tuple], default=None
        Prior parameters (alpha, beta) for capture efficiency Beta distribution.
        Only used if variable_capture=True. Defaults to (1, 1).
    mixing_prior : Optional[tuple], default=None
        Prior concentration parameter(s) for mixture weights Dirichlet
        distribution. Required if mixture_model=True.
        
    Training Parameters:
    ------------------
    n_steps : int, default=100_000
        Number of optimization steps for variational inference
    batch_size : Optional[int], default=None
        Mini-batch size for stochastic optimization. If None, uses full dataset.
    optimizer : numpyro.optim.optimizers, default=Adam(step_size=0.001)
        Optimizer for variational inference
        
    Data Handling:
    -------------
    cells_axis : int, default=0
        Axis for cells in count matrix (0=rows, 1=columns)
    layer : Optional[str], default=None
        Layer in AnnData to use for counts. If None, uses .X.
        
    Additional Options:
    -----------------
    seed : int, default=42
        Random seed for reproducibility
    stable_update : bool, default=True
        Whether to use numerically stable parameter updates during optimization
    r_guide : Optional[str], default=None
        Distribution family for guide of dispersion parameter. If None, uses
        same as r_dist. Options: "gamma" or "lognormal".
    loss : numpyro.infer.elbo, default=TraceMeanField_ELBO()
        Loss function for variational inference
        
    Returns
    -------
    ScribeSVIResults
        Results object containing:
            - Fitted model parameters
            - Loss history
            - Model configuration
            - Prior parameters
            - Dataset metadata
    """
    # Determine model type based on boolean flags
    base_model = "nbdm"
    if zero_inflated:
        base_model = "zinb"
    if variable_capture:
        if zero_inflated:
            base_model = "zinbvcp"
        else:
            base_model = "nbvcp"
    
    if n_components is not None and not mixture_model:
        raise ValueError(
            "n_components must be None when mixture_model=False"
        )
            
    # Define model type
    model_type = base_model
    # Check if mixture model
    if mixture_model:
        # Validate n_components for mixture models
        if n_components is None or n_components < 2:
            raise ValueError(
                "n_components must be specified and greater than 1 "
                "when mixture_model=True"
            )
        # Validate mixing_prior for mixture models
        if mixing_prior is None:
            raise ValueError(
                "mixing_prior must be specified when mixture_model=True"
            )
        # Add mixture suffix if needed
        model_type = f"{base_model}_mix"
    
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
    if "mix" in model_type and mixing_prior is None:
        mixing_prior = jnp.ones(n_components)

    # Create random key
    rng_key = random.PRNGKey(seed)
    
    # Create model config based on model type and specified distributions

    # Set r distribution for model
    if r_dist == "gamma":
        r_dist_model = dist.Gamma(*r_prior)
    elif r_dist == "lognormal":
        r_dist_model = dist.LogNormal(*r_prior)
    else:
        raise ValueError(f"Unsupported r_dist: {r_dist}")

    # Set r distribution for guide
    if r_guide is not None:
        r_dist_guide = dist.Gamma(*r_guide) if r_guide == "gamma" else dist.LogNormal(*r_guide)
    else:
        r_dist_guide = r_dist_model
        
    # Set p distribution for model and guide
    p_dist_model = dist.Beta(*p_prior)
    p_dist_guide = dist.Beta(*p_prior)
    
    # Set up NegativeBinomial2 parameterization distributions
    # Mean parameter: use LogNormal with parameters derived from r and p priors
    # For NB2, mean = r * (1-p) / p, so we can estimate reasonable mean priors
    mean_loc = 2.0  # ln(mean) ~ 2, so mean ~ 7.4
    mean_scale = 1.0
    mean_dist_model = dist.LogNormal(mean_loc, mean_scale)
    mean_dist_guide = dist.LogNormal(mean_loc, mean_scale)
    
    # Concentration parameter: use the same distribution as r
    concentration_dist_model = r_dist_model
    concentration_dist_guide = r_dist_guide

    # Configure gate distribution if needed
    gate_dist_model = None
    gate_dist_guide = None
    if "zinb" in model_type:
        gate_dist_model = dist.Beta(*gate_prior)
        gate_dist_guide = gate_dist_model
            
    # Configure capture probability distribution if needed
    p_capture_dist_model = None
    p_capture_dist_guide = None
    if "vcp" in model_type:
        p_capture_dist_model = dist.Beta(*p_capture_prior)
        p_capture_dist_guide = p_capture_dist_model

    # Configure mixing distribution if needed
    mixing_dist_model = None
    mixing_dist_guide = None
    if "mix" in model_type:
        mixing_dist_model = dist.Dirichlet(jnp.array(mixing_prior))
        mixing_dist_guide = mixing_dist_model

    # Create model_config with all the configurations
    model_config = ConstrainedModelConfig(
        base_model=model_type,
        r_distribution_model=r_dist_model,
        r_distribution_guide=r_dist_guide,
        r_param_prior=r_prior,
        r_param_guide=r_prior,
        p_distribution_model=p_dist_model,
        p_distribution_guide=p_dist_guide,
        p_param_prior=p_prior,
        p_param_guide=p_prior,
        # NegativeBinomial2 parameterization fields
        mean_distribution_model=mean_dist_model,
        mean_distribution_guide=mean_dist_guide,
        mean_param_prior=(mean_loc, mean_scale),
        mean_param_guide=(mean_loc, mean_scale),
        concentration_distribution_model=concentration_dist_model,
        concentration_distribution_guide=concentration_dist_guide,
        concentration_param_prior=r_prior,
        concentration_param_guide=r_prior,
        gate_distribution_model=gate_dist_model,
        gate_distribution_guide=gate_dist_guide,
        gate_param_prior=gate_prior if gate_dist_model else None,
        gate_param_guide=gate_prior if gate_dist_model else None,
        p_capture_distribution_model=p_capture_dist_model,
        p_capture_distribution_guide=p_capture_dist_guide,
        p_capture_param_prior=p_capture_prior if p_capture_dist_model else None,
        p_capture_param_guide=p_capture_prior if p_capture_dist_model else None,
        n_components=n_components,
        mixing_distribution_model=mixing_dist_model,
        mixing_distribution_guide=mixing_dist_guide,
        mixing_param_prior=mixing_prior,
        mixing_param_guide=mixing_prior
    )
    
    # Get model and guide functions
    model, guide = get_model_and_guide(model_type)
    
    # Set up SVI
    svi = SVI(model, guide, optimizer, loss=loss)
    
    # Prepare model arguments
    model_args = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'counts': count_data,
        'batch_size': batch_size,
        'model_config': model_config
    }
    
    # Run inference
    svi_results = svi.run(
        rng_key,
        n_steps,
        stable_update=stable_update,
        **model_args
    )
    
    # Create results object
    if adata is not None:
        results = ScribeSVIResults.from_anndata(
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
        results = ScribeSVIResults(
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
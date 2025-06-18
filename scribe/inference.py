"""
Unified inference interface for SCRIBE.

This module provides a single entry point for all SCRIBE inference methods,
including both SVI and MCMC, with a modular architecture.
"""

from typing import Union, Optional, Dict, Any
import jax.numpy as jnp

# Import shared components
from .core import InputProcessor, PriorConfigFactory, ModelConfigFactory

# Import inference-specific components
from .svi import SVIDistributionBuilder, SVIInferenceEngine, SVIResultsFactory
from .mcmc import MCMCInferenceEngine, MCMCResultsFactory


def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    inference_method: str = "svi",
    
    # Model configuration
    zero_inflated: bool = False,
    variable_capture: bool = False,
    mixture_model: bool = False,
    n_components: Optional[int] = None,
    
    # Data processing parameters  
    cells_axis: int = 0,
    layer: Optional[str] = None,
    
    # SVI-specific parameters
    parameterization: Optional[str] = "mean_field",
    optimizer: Optional[Any] = None,
    loss: Optional[Any] = None,
    n_steps: int = 100_000,
    batch_size: Optional[int] = None,
    stable_update: bool = True,
    
    # MCMC-specific parameters
    unconstrained_model: Optional[bool] = False,
    r_dist: Optional[str] = "gamma",
    n_samples: int = 2_000,
    n_warmup: int = 1_000,
    n_chains: int = 1,
    
    # Prior configuration
    r_prior: Optional[tuple] = None,
    p_prior: Optional[tuple] = None,
    gate_prior: Optional[tuple] = None,
    p_capture_prior: Optional[tuple] = None,
    mixing_prior: Optional[Any] = None,
    mu_prior: Optional[tuple] = None,
    phi_prior: Optional[tuple] = None,  
    phi_capture_prior: Optional[tuple] = None,
    
    # General parameters
    seed: int = 42,
) -> Any:
    """
    Unified interface for SCRIBE inference.
    
    This function provides a single entry point for both SVI and MCMC inference
    methods with a modular, maintainable architecture.
    
    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing single-cell RNA-seq counts
    inference_method : str, default="svi"
        Inference method to use ("svi" or "mcmc")
        
    Model Configuration:
    -------------------
    zero_inflated : bool, default=False
        Whether to use zero-inflated model (ZINB vs NB)
    variable_capture : bool, default=False 
        Whether to model variable capture probability
    mixture_model : bool, default=False
        Whether to use mixture model for cell heterogeneity
    n_components : Optional[int], default=None
        Number of mixture components (required if mixture_model=True)
        
    Data Processing:
    ---------------
    cells_axis : int, default=0
        Axis for cells in count matrix (0=rows, 1=columns)
    layer : Optional[str], default=None
        Layer in AnnData to use for counts. If None, uses .X
        
    SVI Parameters:
    --------------
    parameterization : Optional[str], default="mean_field"
        Parameterization type ("mean_field", "mean_variance", "beta_prime")
    optimizer : Optional[Any], default=None
        Optimizer for variational inference (defaults to Adam)
    loss : Optional[Any], default=None
        Loss function for variational inference (defaults to TraceMeanField_ELBO)
    n_steps : int, default=100_000
        Number of optimization steps
    batch_size : Optional[int], default=None
        Mini-batch size. If None, uses full dataset
    stable_update : bool, default=True
        Whether to use numerically stable parameter updates
        
    MCMC Parameters:
    ---------------
    unconstrained_model : Optional[bool], default=False
        Whether to use unconstrained parameterization
    r_dist : Optional[str], default="gamma"
        Distribution for r parameter ("gamma" or "lognormal")
    n_samples : int, default=2_000
        Number of MCMC samples
    n_warmup : int, default=1_000
        Number of warmup samples
    n_chains : int, default=1
        Number of parallel chains
        
    Prior Configuration:
    -------------------
    r_prior, p_prior, gate_prior, p_capture_prior : Optional[tuple]
        Prior parameters as (param1, param2) tuples
    mixing_prior : Optional[Any]
        Prior for mixture components (array-like or scalar)
    mu_prior, phi_prior, phi_capture_prior : Optional[tuple]
        Additional prior parameters for specific parameterizations
        
    General:
    -------
    seed : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    Union[ScribeSVIResults, ScribeMCMCResults]
        Results object containing inference results and diagnostics
        
    Raises
    ------
    ValueError
        If configuration is invalid or required parameters are missing
    """
    # Step 1: Input Processing & Validation
    InputProcessor.validate_model_configuration(
        zero_inflated, variable_capture, mixture_model, n_components
    )
    
    # Process count data
    count_data, adata, n_cells, n_genes = InputProcessor.process_counts_data(
        counts, cells_axis, layer
    )
    
    # Determine model type
    model_type = InputProcessor.determine_model_type(
        zero_inflated, variable_capture, mixture_model
    )
    
    # Step 2: Prior Configuration
    # Collect user-provided priors
    user_priors = {}
    if r_prior is not None:
        user_priors["r_prior"] = r_prior
    if p_prior is not None:
        user_priors["p_prior"] = p_prior
    if gate_prior is not None:
        user_priors["gate_prior"] = gate_prior
    if p_capture_prior is not None:
        user_priors["p_capture_prior"] = p_capture_prior
    if mixing_prior is not None:
        user_priors["mixing_prior"] = mixing_prior
    if mu_prior is not None:
        user_priors["mu_prior"] = mu_prior
    if phi_prior is not None:
        user_priors["phi_prior"] = phi_prior
    if phi_capture_prior is not None:
        user_priors["phi_capture_prior"] = phi_capture_prior
    
    # Create default priors
    default_priors = PriorConfigFactory.create_default_priors(
        model_type=model_type,
        inference_method=inference_method,
        parameterization=parameterization,
        unconstrained_model=unconstrained_model,
        r_dist=r_dist,
        n_components=n_components
    )
    
    # Merge user priors with defaults
    final_priors = {**default_priors, **user_priors}
    
    # Validate priors
    PriorConfigFactory.validate_priors(
        model_type=model_type,
        inference_method=inference_method,
        parameterization=parameterization,
        prior_dict=final_priors
    )
    
    # Step 3: Inference-Specific Execution
    if inference_method == "svi":
        return _run_svi_inference(
            model_type=model_type,
            parameterization=parameterization,
            priors=final_priors,
            count_data=count_data,
            adata=adata,
            n_cells=n_cells,
            n_genes=n_genes,
            n_components=n_components,
            optimizer=optimizer,
            loss=loss,
            n_steps=n_steps,
            batch_size=batch_size,
            stable_update=stable_update,
            seed=seed
        )
    elif inference_method == "mcmc":
        return _run_mcmc_inference(
            model_type=model_type,
            unconstrained_model=unconstrained_model,
            priors=final_priors,
            count_data=count_data,
            adata=adata,
            n_cells=n_cells,
            n_genes=n_genes,
            n_components=n_components,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_chains=n_chains,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")


def _run_svi_inference(
    model_type: str,
    parameterization: str,
    priors: Dict[str, Any],
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    n_components: Optional[int],
    optimizer: Optional[Any],
    loss: Optional[Any],
    n_steps: int,
    batch_size: Optional[int],
    stable_update: bool,
    seed: int
) -> Any:
    """Execute SVI inference."""
    # Build distributions
    distributions = SVIDistributionBuilder.build_distributions(
        model_type, parameterization, priors
    )
    
    # Create model configuration
    model_config = ModelConfigFactory.create_svi_config(
        model_type, parameterization, distributions, n_components
    )
    
    # Set defaults for optimizer and loss if not provided
    if optimizer is None:
        import numpyro.optim
        optimizer = numpyro.optim.Adam(step_size=0.001)
    if loss is None:
        from numpyro.infer import TraceMeanField_ELBO
        loss = TraceMeanField_ELBO()
    
    # Run inference
    svi_results = SVIInferenceEngine.run_inference(
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        optimizer=optimizer,
        loss=loss,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        stable_update=stable_update
    )
    
    # Package results
    return SVIResultsFactory.create_results(
        svi_results=svi_results,
        model_config=model_config,
        adata=adata,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_type,
        n_components=n_components,
        prior_params=priors
    )


def _run_mcmc_inference(
    model_type: str,
    unconstrained_model: bool,
    priors: Dict[str, Any],
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    n_components: Optional[int],
    n_samples: int,
    n_warmup: int,
    n_chains: int,
    seed: int
) -> Any:
    """Execute MCMC inference."""
    # Create model configuration
    model_config = ModelConfigFactory.create_mcmc_config(
        model_type, unconstrained_model, priors, n_components
    )
    
    # Run inference
    mcmc_results = MCMCInferenceEngine.run_inference(
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_chains=n_chains,
        seed=seed
    )
    
    # Package results
    return MCMCResultsFactory.create_results(
        mcmc_results=mcmc_results,
        model_config=model_config,
        adata=adata,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_type,
        n_components=n_components,
        prior_params=priors
    ) 
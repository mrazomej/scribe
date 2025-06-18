"""
Unified inference interface for SCRIBE with parameterization unification.

This module provides a single entry point for all SCRIBE inference methods,
treating unconstrained as just another parameterization rather than a separate
model type.
"""

from typing import Union, Optional, Dict, Any, Type
import jax.numpy as jnp
import numpyro.distributions as dist

# Import shared components
from .core import InputProcessor, PriorConfigFactory
from .core.config_factory import ModelConfigFactory

# Import inference-specific components
from .svi import SVIInferenceEngine, SVIResultsFactory
from .mcmc import MCMCInferenceEngine, MCMCResultsFactory


def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    inference_method: str = "svi",
    
    # Model configuration
    zero_inflated: bool = False,
    variable_capture: bool = False,
    mixture_model: bool = False,
    n_components: Optional[int] = None,
    
    # Parameterization (now unified!)
    # "standard", "linked", "odds_ratio", "unconstrained"
    parameterization: str = "standard",  
    
    # Data processing parameters  
    cells_axis: int = 0,
    layer: Optional[str] = None,
    
    # SVI-specific parameters
    optimizer: Optional[Any] = None,
    loss: Optional[Any] = None,
    n_steps: int = 100_000,
    batch_size: Optional[int] = None,
    stable_update: bool = True,
    
    # MCMC-specific parameters
    n_samples: int = 2_000,
    n_warmup: int = 1_000,
    n_chains: int = 1,
    mcmc_kwargs: Optional[Dict[str, Any]] = None,

    # Distribution configuration
    r_distribution: Optional[Type[dist.Distribution]] = None,
    mu_distribution: Optional[Type[dist.Distribution]] = None,
        
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
    Unified interface for SCRIBE inference with parameterization unification.
    
    This function provides a single entry point for both SVI and MCMC inference
    methods, treating unconstrained as just another parameterization. This means:
    
    - You can run MCMC with any parameterization (standard, linked,
      odds_ratio, unconstrained)
    - You can run SVI with any parameterization (though guides may not exist for
      all yet)
    - The interface is completely unified and consistent
    
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
        
    Parameterization:
    ----------------
    parameterization : str, default="standard"
        Model parameterization to use:
        - "standard": Beta/Gamma or LogNormal distributions for p/r
        - "linked": Beta/LogNormal for p/mu parameters
        - "odds_ratio": BetaPrime/LogNormal for phi/mu parameters  
        - "unconstrained": Normal distributions on transformed parameters
        
    Data Processing:
    ---------------
    cells_axis : int, default=0
        Axis for cells in count matrix (0=rows, 1=columns)
    layer : Optional[str], default=None
        Layer in AnnData to use for counts. If None, uses .X
        
    SVI Parameters:
    --------------
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
    n_samples : int, default=2_000
        Number of MCMC samples
    n_warmup : int, default=1_000
        Number of warmup samples
    n_chains : int, default=1
        Number of parallel chains
    mcmc_kwargs : Optional[Dict[str, Any]], default=None
        Keyword arguments for the MCMC kernel (e.g., target_accept_prob, max_tree_depth)
        
    Distribution Configuration:
    --------------------------
    r_distribution : Optional[Type[dist.Distribution]]
        Distribution for r parameter (when needed)
    mu_distribution : Optional[Type[dist.Distribution]]
        Distribution for mu parameter (when needed)

    Prior Configuration:
    -------------------
    r_prior, p_prior, gate_prior, p_capture_prior : Optional[tuple]
        Prior parameters as (param1, param2) tuples
            - For unconstrained: (loc, scale) for Normal distributions on
              transformed
            parameters - For constrained: Parameters for respective
            distributions (Beta, Gamma, etc.)
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
        
    Examples
    --------
    # SVI with standard parameterization
    results = run_scribe(
        counts, inference_method="svi", parameterization="standard")
    
    # MCMC with unconstrained parameterization  
    results = run_scribe(
        counts, inference_method="mcmc", parameterization="unconstrained")
    
    # SVI with odds_ratio parameterization for ZINBVCP mixture model
    results = run_scribe(
        counts, 
        inference_method="svi",
        parameterization="odds_ratio",
        zero_inflated=True,
        variable_capture=True,
        mixture_model=True,
        n_components=3
    )
    
    # MCMC with linked parameterization
    results = run_scribe(
        counts,
        inference_method="mcmc", 
        parameterization="linked",
        n_samples=1000
    )
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
    
    # Create default priors (now handles unconstrained as just another
    # parameterization)
    default_priors = PriorConfigFactory.create_default_priors(
        model_type=model_type,
        inference_method=inference_method,
        parameterization=parameterization,  # This now includes "unconstrained"
        r_distribution=r_distribution,
        mu_distribution=mu_distribution,
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
    
    # Step 3: Create Unified Model Configuration
    model_config = ModelConfigFactory.create_config(
        model_type=model_type,
        parameterization=parameterization,
        inference_method=inference_method,
        priors=final_priors,
        n_components=n_components
    )
    
    # Step 4: Run Inference
    if inference_method == "svi":
        return _run_svi_inference(
            model_config=model_config,
            count_data=count_data,
            adata=adata,
            n_cells=n_cells,
            n_genes=n_genes,
            optimizer=optimizer,
            loss=loss,
            n_steps=n_steps,
            batch_size=batch_size,
            stable_update=stable_update,
            seed=seed,
            final_priors=final_priors
        )
    elif inference_method == "mcmc":
        return _run_mcmc_inference(
            model_config=model_config,
            count_data=count_data,
            adata=adata,
            n_cells=n_cells,
            n_genes=n_genes,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_chains=n_chains,
            seed=seed,
            final_priors=final_priors,
            mcmc_kwargs=mcmc_kwargs
        )
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")


# ------------------------------------------------------------------------------
# SVI Inference
# ------------------------------------------------------------------------------

def _run_svi_inference(
    model_config: "ModelConfig",
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    optimizer: Optional[Any],
    loss: Optional[Any],
    n_steps: int,
    batch_size: Optional[int],
    stable_update: bool,
    seed: int,
    final_priors: Dict[str, Any]
) -> Any:
    """Execute SVI inference with unified configuration."""
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
        model_type=model_config.base_model,
        n_components=model_config.n_components,
        prior_params=final_priors
    )


# ------------------------------------------------------------------------------
# MCMC Inference
# ------------------------------------------------------------------------------

def _run_mcmc_inference(
    model_config: "ModelConfig",
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    n_samples: int,
    n_warmup: int,
    n_chains: int,
    seed: int,
    final_priors: Dict[str, Any],
    mcmc_kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """Execute MCMC inference with unified configuration."""
    # Run inference (now works with any parameterization!)
    mcmc_results = MCMCInferenceEngine.run_inference(
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_chains=n_chains,
        seed=seed,
        mcmc_kwargs=mcmc_kwargs
    )
    
    # Package results
    return MCMCResultsFactory.create_results(
        mcmc_results=mcmc_results,
        model_config=model_config,
        adata=adata,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_config.base_model,
        n_components=model_config.n_components,
        prior_params=final_priors
    ) 
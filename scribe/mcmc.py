"""
Markov Chain Monte Carlo (MCMC) module for single-cell RNA sequencing data
analysis.

This module implements MCMC inference for SCRIBE models using Numpyro's MCMC
samplers.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any, Type
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, init_to_value
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
from numpyro.distributions import constraints
import scipy.sparse

from .model_config import UnconstrainedModelConfig
from .model_registry import get_unconstrained_model
from .results_mcmc import ScribeMCMCResults

# ------------------------------------------------------------------------------
# MCMC Inference with Numpyro
# ------------------------------------------------------------------------------

def create_mcmc_instance(
    model_type: Optional[str] = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    chain_method: str = "parallel",
    init_strategy: Optional[Dict] = None,
    progress_bar: bool = True,
    # Kernel configuration
    kernel: Type[numpyro.infer.mcmc.MCMCKernel] = NUTS,
    kernel_kwargs: Optional[Dict] = None,
    # Reparameterization configuration
    reparam_config: Optional[Union[Dict[str, Any], Any]] = LocScaleReparam(),
):
    """
    Create an MCMC instance with the defined model and MCMC kernel.

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
    num_warmup : int, default=1000
        Number of warmup/burn-in steps
    num_samples : int, default=1000
        Number of samples to collect
    num_chains : int, default=1
        Number of independent chains to run
    chain_method : str, default="parallel"
        How to run chains: "parallel", "sequential", or "vectorized"
    init_strategy : Optional[Dict], default=None
        Strategy for initializing the chain. If None, uses default init
        strategy.
    progress_bar : bool, default=True
        Whether to show progress bar
    kernel : Type[numpyro.infer.mcmc.MCMCKernel], default=NUTS
        MCMC kernel to use. Options include:
            - NUTS: No-U-Turn Sampler (default)
            - HMC: Hamiltonian Monte Carlo
            - Other custom kernels that inherit from MCMCKernel
    kernel_kwargs : Optional[Dict], default=None
        Additional keyword arguments to pass to the kernel constructor. For
        NUTS, common options include:
            - target_accept_prob: float, default=0.8
            - step_size: float, default=None
            - adapt_step_size: bool, default=True
            - adapt_mass_matrix: bool, default=True
        For HMC, common options include:
            - step_size: float, default=None
            - num_steps: int, default=None
            - adapt_step_size: bool, default=True
            - adapt_mass_matrix: bool, default=True
    reparam_config : Optional[Union[Dict[str, Any], Any]],
    default=LocScaleReparam()
        Configuration for parameter reparameterization. Options:
            - None: No reparameterization
            - Any reparameterization instance: Apply to all parameters
            - Dict[str, Any]: Dictionary mapping parameter names to specific
              reparameterization strategies. If only one reparameterization is
              provided, it will be applied to all parameters.

    Returns
    -------
    numpyro.infer.MCMC
        Configured MCMC instance ready for sampling

    Raises
    ------
    ValueError
        If model_type is invalid or reparam_config is invalid
    """
    # Get the unconstrained model
    model_fn = get_unconstrained_model(model_type)
    
    # Apply reparameterization if configured
    if reparam_config is not None:
        # If reparam_config is a single reparameterization instance, apply it to
        # all parameters
        if not isinstance(reparam_config, dict):
            # Store the original reparameterizer instance
            original_reparam_instance = reparam_config
            
            # Initialize the reparam_config dictionary
            reparam_config = {
                "p_unconstrained": original_reparam_instance,
                "r_unconstrained": original_reparam_instance,
            }
            
            # Add more reparameterizations based on model type, using the stored instance
            if "zinb" in model_type:
                reparam_config["gate_unconstrained"] = original_reparam_instance
            if "vcp" in model_type:
                reparam_config["p_capture_unconstrained"] = original_reparam_instance
            if "_mix" in model_type:
                reparam_config["mixing_unconstrained"] = original_reparam_instance
        
        # Apply reparameterization
        model_fn = reparam(model_fn, config=reparam_config)
    
    # Set up initialization strategy if provided
    init_fn = None
    if init_strategy is not None:
        init_fn = init_to_value(values=init_strategy)
    
    # Set up kernel with provided arguments
    kernel_kwargs = kernel_kwargs or {}
    if init_fn is not None:
        kernel_kwargs["init_strategy"] = init_fn
    
    # Create kernel instance
    kernel_instance = kernel(model_fn, **kernel_kwargs)
    
    # Create MCMC object
    return MCMC(
        kernel_instance,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar
    )

# ------------------------------------------------------------------------------

def run_mcmc_sampling(
    mcmc_instance: MCMC,
    rng_key: random.PRNGKey,
    counts: jnp.ndarray,
    model_type: str,
    model_config: UnconstrainedModelConfig,
    cells_axis: int = 0,
    **kwargs
) -> Dict:
    """
    Run MCMC sampling with the provided MCMC instance.

    Parameters
    ----------
    mcmc_instance : numpyro.infer.MCMC
        Configured MCMC instance for running inference
    rng_key : random.PRNGKey
        Random key for reproducibility
    counts : jnp.ndarray
        The observed count matrix
    model_type : str
        The type of model being used
    model_config : UnconstrainedModelConfig
        Configuration object for the model
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows.
    **kwargs
        Additional arguments to pass to the model

    Returns
    -------
    Dict
        Dictionary of posterior samples
    """
    # Transpose counts if cells_axis=1
    if cells_axis == 1:
        counts = counts.T
        
    # Extract dimensions
    n_cells, n_genes = counts.shape
    
    # Set up model arguments
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
        "counts": counts,
    }
    
    # Add additional kwargs
    model_args.update(kwargs)
    
    # Run MCMC
    mcmc_instance.run(rng_key, **model_args)
    
    # Return samples
    return mcmc_instance

# ------------------------------------------------------------------------------
# SCRIBE MCMC inference pipeline
# ------------------------------------------------------------------------------

def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    # Model configuration
    zero_inflated: bool = False,
    variable_capture: bool = False, 
    mixture_model: bool = False,
    n_components: Optional[int] = None,
    # Prior parameters
    r_prior: Optional[tuple] = None,
    p_prior: Optional[tuple] = None,
    gate_prior: Optional[tuple] = None,
    p_capture_prior: Optional[tuple] = None,
    mixing_prior: Optional[tuple] = None,
    # MCMC parameters
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    chain_method: str = "parallel",
    init_strategy: Optional[Dict] = None,
    # Kernel configuration
    kernel: Type[numpyro.infer.mcmc.MCMCKernel] = NUTS,
    kernel_kwargs: Optional[Dict] = None,
    # Reparameterization configuration
    reparam_config: Optional[Union[Dict[str, Any], Any]] = LocScaleReparam(),
    # Data handling options
    cells_axis: int = 0,
    layer: Optional[str] = None,
    # Extra options
    seed: int = 42,
    progress_bar: bool = True,
) -> ScribeMCMCResults:
    """
    Run SCRIBE MCMC inference pipeline to fit a probabilistic model to
    single-cell RNA sequencing data.
    
    This function provides a high-level interface to configure and run SCRIBE
    models with MCMC inference. It handles data preprocessing, model
    configuration, MCMC sampling, and results packaging.
    
    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing counts. If AnnData, counts
        should be in .X or specified layer. Shape should be (cells, genes) if
        cells_axis=0, or (genes, cells) if cells_axis=1.
    zero_inflated : bool, default=False
        Whether to use zero-inflated negative binomial (ZINB) model
    variable_capture : bool, default=False 
        Whether to model cell-specific mRNA capture efficiencies
    mixture_model : bool, default=False
        Whether to use mixture model components
    n_components : Optional[int], default=None
        Number of mixture components. Required if mixture_model=True.
    r_prior : Optional[tuple], default=None
        Prior parameters (loc, scale) for dispersion (r) parameter. Defaults to
        (0, 1).
    p_prior : Optional[tuple], default=None
        Prior parameters (loc, scale) for success probability. Defaults to (0,
        1).
    gate_prior : Optional[tuple], default=None
        Prior parameters (loc, scale) for dropout gate. Only used if
        zero_inflated=True. Defaults to (0, 1).
    p_capture_prior : Optional[tuple], default=None
        Prior parameters (loc, scale) for capture efficiency. Only used if
        variable_capture=True. Defaults to (0, 1).
    mixing_prior : Optional[tuple], default=None
        Prior parameters for mixture weights. Required if mixture_model=True.
    num_warmup : int, default=1000
        Number of warmup/burn-in steps
    num_samples : int, default=1000
        Number of samples to collect
    num_chains : int, default=1
        Number of independent chains to run
    chain_method : str, default="parallel"
        How to run chains: "parallel", "sequential", or "vectorized"
    init_strategy : Optional[Dict], default=None
        Strategy for initializing the chain
    kernel : Type[numpyro.infer.mcmc.MCMCKernel], default=NUTS
        MCMC kernel to use (e.g. NUTS, HMC)
    kernel_kwargs : Optional[Dict], default=None
        Additional arguments to pass to MCMC kernel
    reparam_config : Optional[Union[Dict[str, Any], Any]],
    default=LocScaleReparam()
        Configuration for parameter reparameterization
    cells_axis : int, default=0
        Axis for cells in count matrix (0=rows, 1=columns)
    layer : Optional[str], default=None
        Layer in AnnData to use for counts. If None, uses .X
    seed : int, default=42
        Random seed for reproducibility
    progress_bar : bool, default=True
        Whether to show progress bar during sampling

    Returns
    -------
    ScribeMCMCResults
        Results object containing posterior samples, model configuration, prior
        parameters and dataset metadata
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
        r_prior = (0, 1)
    if p_prior is None:
        p_prior = (0, 1)
    if "zinb" in model_type and gate_prior is None:
        gate_prior = (0, 1)
    if "vcp" in model_type and p_capture_prior is None:
        p_capture_prior = (0, 1)
    if "mix" in model_type and mixing_prior is None:
        mixing_prior = jnp.ones(n_components)

    # Create random key
    rng_key = random.PRNGKey(seed)
    
    # Create model config based on model type and specified distributions
    model_config = UnconstrainedModelConfig(
        base_model=model_type,
        # Unconstrained parameterization
        p_unconstrained_loc=p_prior[0],
        p_unconstrained_scale=p_prior[1],
        r_unconstrained_loc=r_prior[0],
        r_unconstrained_scale=r_prior[1],
        gate_unconstrained_loc=gate_prior[0] if "zinb" in model_type else None,
        gate_unconstrained_scale=gate_prior[1] if "zinb" in model_type else None,
        p_capture_unconstrained_loc=p_capture_prior[0] if "vcp" in model_type else None,
        p_capture_unconstrained_scale=p_capture_prior[1] if "vcp" in model_type else None,
        mixing_unconstrained_loc=mixing_prior if "mix" in model_type else None,
        mixing_unconstrained_scale=mixing_prior if "mix" in model_type else None,
    )
    
    # Create MCMC instance
    mcmc = create_mcmc_instance(
        model_type=model_type,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        init_strategy=init_strategy,
        kernel=kernel,
        kernel_kwargs=kernel_kwargs,
        reparam_config=reparam_config,
        progress_bar=progress_bar
    )
    
    # Create prior parameters dictionary for the results
    prior_params = {
        'r_prior': r_prior,
        'p_prior': p_prior,
        'gate_prior': gate_prior if "zinb" in model_type else None,
        'p_capture_prior': p_capture_prior if "vcp" in model_type else None,
        'mixing_prior': mixing_prior if n_components is not None else None
    }
    
    # Run MCMC sampling
    mcmc = run_mcmc_sampling(
        mcmc_instance=mcmc,
        rng_key=rng_key,
        counts=count_data,
        model_type=model_type,
        model_config=model_config,
        cells_axis=cells_axis
    )
    
    # Create results object
    if adata is not None:
        results = ScribeMCMCResults.from_anndata(
            mcmc=mcmc,
            adata=adata,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params
        )
    else:
        results = ScribeMCMCResults.from_mcmc(
            mcmc=mcmc,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            n_components=n_components
        )
    
    return results
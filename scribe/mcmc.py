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

from .model_config import UnconstrainedModelConfig, ConstrainedModelConfig
from .model_registry import get_unconstrained_model
from .results_mcmc import ScribeMCMCResults

# ------------------------------------------------------------------------------
# MCMC Inference with Numpyro
# ------------------------------------------------------------------------------

def create_mcmc_instance(
    model_type: Optional[str] = None,
    unconstrained_model: bool = True,
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
    unconstrained_model : bool, default=True
        Whether to use unconstrained model parameterization. If True, uses
        unconstrained models with normal priors on transformed parameters. 
        If False, uses constrained models with natural parameter distributions.
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
    # Get the model function based on parameterization choice
    if unconstrained_model:
        model_fn = get_unconstrained_model(model_type)
        
        # Apply reparameterization if configured for unconstrained models
        if reparam_config is not None:
            # If reparam_config is a single reparameterization instance, apply
            # it to all parameters
            if not isinstance(reparam_config, dict):
                # Store the original reparameterizer instance
                original_reparam_instance = reparam_config
                
                # Initialize the reparam_config dictionary
                reparam_config = {
                    "p_unconstrained": original_reparam_instance,
                    "r_unconstrained": original_reparam_instance,
                }
                
                # Add more reparameterizations based on model type, using the
                # stored instance
                if "zinb" in model_type:
                    reparam_config["gate_unconstrained"] = original_reparam_instance
                if "vcp" in model_type:
                    reparam_config["p_capture_unconstrained"] = original_reparam_instance
                if "_mix" in model_type:
                    reparam_config["mixing_logits_unconstrained"] = original_reparam_instance
            
            # Apply reparameterization
            model_fn = reparam(model_fn, config=reparam_config)
    else:
        # Get constrained model (no guide needed for MCMC)
        from .model_registry import get_model_and_guide
        model_fn, _ = get_model_and_guide(model_type)
        
        # For constrained models, reparameterization can still be applied
        # but with different parameter names
        if reparam_config is not None:
            if not isinstance(reparam_config, dict):
                # Store the original reparameterizer instance
                original_reparam_instance = reparam_config
                
                # Initialize the reparam_config dictionary for constrained
                # parameters
                reparam_config = {
                    "p": original_reparam_instance,
                    "r": original_reparam_instance,
                }
                
                # Add more reparameterizations based on model type
                if "zinb" in model_type:
                    reparam_config["gate"] = original_reparam_instance
                if "vcp" in model_type:
                    reparam_config["p_capture"] = original_reparam_instance
                if "_mix" in model_type:
                    reparam_config["mixing_weights"] = original_reparam_instance
            
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
    model_config: Union[UnconstrainedModelConfig, ConstrainedModelConfig],
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
    model_config : Union[UnconstrainedModelConfig, ConstrainedModelConfig]
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
    # Parameterization choice
    unconstrained_model: bool = True,
    # Prior parameters
    r_prior: Optional[tuple] = None,
    p_prior: Optional[tuple] = None,
    gate_prior: Optional[tuple] = None,
    p_capture_prior: Optional[tuple] = None,
    mixing_prior: Optional[tuple] = None,
    # Prior distribution types (only for constrained models)
    r_dist: str = "gamma",
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
    unconstrained_model : bool, default=True
        Whether to use unconstrained model parameterization. If True, uses
        unconstrained models with normal priors on transformed parameters 
        (logit-normal for probabilities, log-normal for dispersion). If False,
        uses constrained models with natural parameter distributions (Beta for
        probabilities, Gamma/LogNormal for dispersion).
    r_prior : Optional[tuple], default=None
        Prior parameters for dispersion (r) parameter. For unconstrained models:
        (loc, scale) for log-normal prior. For constrained models: (shape, rate)
        for Gamma or (mu, sigma) for LogNormal. Defaults to (0, 1) for 
        unconstrained, (2, 0.1) for constrained.
    p_prior : Optional[tuple], default=None
        Prior parameters for success probability. For unconstrained models:
        (loc, scale) for logit-normal prior. For constrained models: (alpha, 
        beta) for Beta prior. Defaults to (0, 1) for unconstrained, (1, 1) for
        constrained.
    gate_prior : Optional[tuple], default=None
        Prior parameters for dropout gate. Only used if zero_inflated=True.
        For unconstrained models: (loc, scale) for logit-normal prior. For 
        constrained models: (alpha, beta) for Beta prior. Defaults to (0, 1) 
        for unconstrained, (1, 1) for constrained.
    p_capture_prior : Optional[tuple], default=None
        Prior parameters for capture efficiency. Only used if 
        variable_capture=True. For unconstrained models: (loc, scale) for 
        logit-normal prior. For constrained models: (alpha, beta) for Beta 
        prior. Defaults to (0, 1) for unconstrained, (1, 1) for constrained.
    mixing_prior : Optional[tuple], default=None
        Prior parameters for mixture weights. For unconstrained models: (loc, 
        scale) for normal prior on logits. For constrained models: concentration
        parameters for Dirichlet prior. Required if mixture_model=True. If None,
        defaults to (0.0, 1.0) for unconstrained, (1.0,) for constrained.
    r_dist : str, default="gamma"
        Prior distribution type for r parameter (only for constrained models).
        Options: "gamma" or "lognormal"
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
    
    # Set default priors based on model parameterization
    if unconstrained_model:
        # Defaults for unconstrained models (log-normal for r, logit-normal for
        # p, gate, p_capture)
        if r_prior is None:
            r_prior = (0, 1)
        if p_prior is None:
            p_prior = (0, 1)
        if "zinb" in model_type and gate_prior is None:
            gate_prior = (0, 1)
        if "vcp" in model_type and p_capture_prior is None:
            p_capture_prior = (0, 1)
        if "mix" in model_type and mixing_prior is None:
            mixing_prior = (0, 1)
    else:
        # Defaults for constrained models (Beta for p, gate, p_capture;
        # Gamma/LogNormal for r)
        if r_prior is None:
            r_prior = (2, 0.1) if r_dist == "gamma" else (1, 1)
        if p_prior is None:
            p_prior = (1, 1)
        if "zinb" in model_type and gate_prior is None:
            gate_prior = (1, 1)
        if "vcp" in model_type and p_capture_prior is None:
            p_capture_prior = (1, 1)
        if "mix" in model_type and mixing_prior is None:
            mixing_prior = (1.0,)  # Symmetric Dirichlet

    # Create random key
    rng_key = random.PRNGKey(seed)
    
    # Create model config based on parameterization choice
    if unconstrained_model:
        # Determine loc/scale for mixing logits
        mixing_logits_loc = None
        mixing_logits_scale = None
        if "mix" in model_type:
            if mixing_prior is None:
                # Let UnconstrainedModelConfig.validate() set defaults (0.0,
                # 1.0)
                pass
            elif isinstance(mixing_prior, tuple) and len(mixing_prior) == 2:
                mixing_logits_loc = mixing_prior[0]
                mixing_logits_scale = mixing_prior[1]
            else:
                # If mixing_prior is not None and not a 2-tuple, it's an error.
                raise ValueError(
                    "mixing_prior for unconstrained mixture models must be a tuple (loc, scale) or None."
                )

        # Create unconstrained model config
        model_config = UnconstrainedModelConfig(
            base_model=model_type,
            n_components=n_components,
            # Unconstrained parameterization
            p_unconstrained_loc=p_prior[0],
            p_unconstrained_scale=p_prior[1],
            r_unconstrained_loc=r_prior[0],
            r_unconstrained_scale=r_prior[1],
            gate_unconstrained_loc=gate_prior[0] if "zinb" in model_type and gate_prior else None,
            gate_unconstrained_scale=gate_prior[1] if "zinb" in model_type and gate_prior else None,
            p_capture_unconstrained_loc=p_capture_prior[0] if "vcp" in model_type and p_capture_prior else None,
            p_capture_unconstrained_scale=p_capture_prior[1] if "vcp" in model_type and p_capture_prior else None,
            mixing_logits_unconstrained_loc=mixing_logits_loc,
            mixing_logits_unconstrained_scale=mixing_logits_scale,
        )
        # This will set defaults for mixing logits if they were None
        model_config.validate()     
    else:
        # Import necessary distributions for constrained model
        import numpyro.distributions as dist
        
        # Create distribution objects for constrained model
        # Success probability distribution
        p_dist_model = dist.Beta(p_prior[0], p_prior[1])
        p_dist_guide = dist.Beta(p_prior[0], p_prior[1])
        
        # Dispersion parameter distribution
        if r_dist == "gamma":
            r_dist_model = dist.Gamma(r_prior[0], r_prior[1])
            r_dist_guide = dist.Gamma(r_prior[0], r_prior[1])
        else:  # lognormal
            r_dist_model = dist.LogNormal(r_prior[0], r_prior[1])
            r_dist_guide = dist.LogNormal(r_prior[0], r_prior[1])
        
        # Optional distributions
        gate_dist_model = gate_dist_guide = None
        if "zinb" in model_type:
            gate_dist_model = dist.Beta(gate_prior[0], gate_prior[1])
            gate_dist_guide = dist.Beta(gate_prior[0], gate_prior[1])
        
        p_capture_dist_model = p_capture_dist_guide = None
        if "vcp" in model_type:
            p_capture_dist_model = dist.Beta(p_capture_prior[0], p_capture_prior[1])
            p_capture_dist_guide = dist.Beta(p_capture_prior[0], p_capture_prior[1])
        
        mixing_dist_model = mixing_dist_guide = None
        if "mix" in model_type:
            if isinstance(mixing_prior, tuple) and len(mixing_prior) == 1:
                # Symmetric Dirichlet
                concentration = jnp.ones(n_components) * mixing_prior[0]
            elif isinstance(mixing_prior, tuple) and len(mixing_prior) == n_components:
                # Asymmetric Dirichlet
                concentration = jnp.array(mixing_prior)
            else:
                # Default to symmetric Dirichlet
                concentration = jnp.ones(n_components)
            mixing_dist_model = dist.Dirichlet(concentration)
            mixing_dist_guide = dist.Dirichlet(concentration)
        
        # Create constrained model config
        from .model_config import ConstrainedModelConfig
        model_config = ConstrainedModelConfig(
            base_model=model_type,
            n_components=n_components,
            # Distribution objects
            r_distribution_model=r_dist_model,
            r_distribution_guide=r_dist_guide,
            r_param_prior=r_prior,
            r_param_guide=r_prior,
            p_distribution_model=p_dist_model,
            p_distribution_guide=p_dist_guide,
            p_param_prior=p_prior,
            p_param_guide=p_prior,
            gate_distribution_model=gate_dist_model,
            gate_distribution_guide=gate_dist_guide,
            gate_param_prior=gate_prior if gate_dist_model else None,
            gate_param_guide=gate_prior if gate_dist_model else None,
            p_capture_distribution_model=p_capture_dist_model,
            p_capture_distribution_guide=p_capture_dist_guide,
            p_capture_param_prior=p_capture_prior if p_capture_dist_model else None,
            p_capture_param_guide=p_capture_prior if p_capture_dist_model else None,
            mixing_distribution_model=mixing_dist_model,
            mixing_distribution_guide=mixing_dist_guide,
            mixing_param_prior=mixing_prior if mixing_dist_model else None,
            mixing_param_guide=mixing_prior if mixing_dist_model else None,
        )
        model_config.validate()
    
    # Create MCMC instance For constrained models, disable reparameterization by
    # default since LocScaleReparam doesn't work with bounded distributions
    # (Beta, Dirichlet, etc.)
    if not unconstrained_model and reparam_config is not None:
        if isinstance(reparam_config, LocScaleReparam):
            reparam_config = None  # Disable reparameterization for constrained models
    
    mcmc = create_mcmc_instance(
        model_type=model_type,
        unconstrained_model=unconstrained_model,
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
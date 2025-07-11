"""
Unified inference interface for SCRIBE with parameterization unification.

This module provides a single entry point for all SCRIBE inference methods,
treating unconstrained as just another parameterization rather than a separate
model type.
"""

from typing import Union, Optional, Dict, Any, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

import jax.numpy as jnp
import numpyro.distributions as dist

# Import shared components
from .core import InputProcessor
from .models.model_config import ModelConfig

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
    component_specific_params: bool = False,
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

    # Step 3: Create ModelConfig directly
    model_config_kwargs = {
        "base_model": model_type,
        "parameterization": parameterization,
        "inference_method": inference_method,
        "n_components": n_components,
        "component_specific_params": component_specific_params,
    }

    if parameterization == "unconstrained":
        model_config_kwargs.update({
            "p_unconstrained_prior": user_priors.get("p_prior"),
            "r_unconstrained_prior": user_priors.get("r_prior"),
            "gate_unconstrained_prior": user_priors.get("gate_prior"),
            "p_capture_unconstrained_prior": user_priors.get("p_capture_prior"),
            "mixing_logits_unconstrained_prior": user_priors.get("mixing_prior"),
        })
    else:
        model_config_kwargs.update({
            "p_param_prior": user_priors.get("p_prior"),
            "r_param_prior": user_priors.get("r_prior"),
            "mu_param_prior": user_priors.get("mu_prior"),
            "phi_param_prior": user_priors.get("phi_prior"),
            "gate_param_prior": user_priors.get("gate_prior"),
            "p_capture_param_prior": user_priors.get("p_capture_prior"),
            "phi_capture_param_prior": user_priors.get("phi_capture_prior"),
            "mixing_param_prior": user_priors.get("mixing_prior"),
        })

    model_config = ModelConfig(**model_config_kwargs)
    model_config.validate()
    
    # Step 4: Run Inference
    if inference_method == "svi":
        results = _run_svi_inference(
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
        )
    elif inference_method == "mcmc":
        results = _run_mcmc_inference(
            model_config=model_config,
            count_data=count_data,
            adata=adata,
            n_cells=n_cells,
            n_genes=n_genes,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_chains=n_chains,
            seed=seed,
            mcmc_kwargs=mcmc_kwargs,
        )
    else:
        raise ValueError("Invalid inference_method. Choose 'svi' or 'mcmc'")

    return results


def _run_svi_inference(
    model_config: ModelConfig,
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
) -> Any:
    """Helper function to run SVI inference."""
    inference_kwargs = {
        "model_config": model_config,
        "count_data": count_data,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seed": seed,
        "stable_update": stable_update,
    }
    if optimizer is not None:
        inference_kwargs["optimizer"] = optimizer
    if loss is not None:
        inference_kwargs["loss"] = loss
        
    svi_results = SVIInferenceEngine.run_inference(**inference_kwargs)

    return SVIResultsFactory.create_results(
        svi_results=svi_results,
        adata=adata,
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_config.base_model,
        n_components=model_config.n_components,
        prior_params=model_config.get_active_priors(),
    )


def _run_mcmc_inference(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    n_samples: int,
    n_warmup: int,
    n_chains: int,
    seed: int,
    mcmc_kwargs: Optional[Dict[str, Any]],
) -> Any:
    """Helper function to run MCMC inference."""
    mcmc = MCMCInferenceEngine.run_inference(
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_samples=n_samples,
        n_warmup=n_warmup,
        n_chains=n_chains,
        seed=seed,
        mcmc_kwargs=mcmc_kwargs,
    )

    return MCMCResultsFactory.create_results(
        mcmc_results=mcmc,
        adata=adata,
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_config.base_model,
        n_components=model_config.n_components,
        prior_params=model_config.get_active_priors(),
    )

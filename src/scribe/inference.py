"""
Unified inference interface for SCRIBE with parameterization unification.

This module provides a single entry point for all SCRIBE inference methods,
treating unconstrained as just another parameterization rather than a separate
model type.
"""

from typing import (
    Union,
    Optional,
    Dict,
    Any,
    TYPE_CHECKING,
    List,
)

if TYPE_CHECKING:
    from anndata import AnnData

import jax.numpy as jnp

# Import shared components
from .core import InputProcessor
from .models.config import ModelConfigBuilder, ModelConfig
from .utils import ParameterCollector

# Import inference-specific components
from .svi import SVIInferenceEngine, SVIResultsFactory
from .mcmc import MCMCInferenceEngine, MCMCResultsFactory
from .vae import ScribeVAEResults


def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    inference_method: str = "svi",
    # Model configuration
    zero_inflated: bool = False,
    variable_capture: bool = False,
    mixture_model: bool = False,
    n_components: Optional[int] = None,
    component_specific_params: bool = False,
    # New guide rank parameter
    guide_rank: Optional[int] = None,
    # Parameterization (now unified!)
    # "standard", "linked", "odds_ratio"
    parameterization: str = "standard",
    # New unconstrained flag
    unconstrained: bool = False,
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
    # Prior configuration
    r_prior: Optional[tuple] = None,
    p_prior: Optional[tuple] = None,
    gate_prior: Optional[tuple] = None,
    p_capture_prior: Optional[tuple] = None,
    mixing_prior: Optional[Any] = None,
    mu_prior: Optional[tuple] = None,
    phi_prior: Optional[tuple] = None,
    phi_capture_prior: Optional[tuple] = None,
    # VAE-specific parameters
    vae_latent_dim: int = 3,
    vae_hidden_dims: Optional[list] = None,
    vae_activation: Optional[str] = None,
    # VAE VCP encoder parameters (for variable capture models)
    vae_vcp_hidden_dims: Optional[List[int]] = None,
    vae_vcp_activation: Optional[str] = None,
    # VAE prior configuration (for dpVAE)
    vae_prior_type: str = "standard",
    vae_prior_hidden_dims: Optional[List[int]] = None,
    vae_prior_num_layers: Optional[int] = None,
    vae_prior_activation: Optional[str] = None,
    vae_prior_mask_type: str = "alternating",
    # VAE data preprocessing
    vae_standardize: bool = False,
    # General parameters
    seed: int = 42,
) -> Any:
    """
    Unified interface for SCRIBE inference with parameterization unification.

    This function provides a single entry point for SVI, MCMC, and VAE inference
    methods, treating "unconstrained" as just another parameterization.

    Supported inference methods:
        - "svi": Stochastic Variational Inference
        - "mcmc": Markov Chain Monte Carlo
        - "vae": Variational Autoencoder (VAE/dpVAE)

    Supported parameterizations:
        - "standard": Beta/LogNormal for p/r
        - "linked": Beta/LogNormal for p/mu
        - "odds_ratio": BetaPrime/LogNormal for phi/mu
        - unconstrained=True: Normal distributions on transformed parameters

    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing single-cell RNA-seq counts.
    inference_method : str, default="svi"
        Inference method to use ("svi", "mcmc", or "vae").

    Model Configuration
    -------------------
    zero_inflated : bool, default=False
        Use zero-inflated model (ZINB vs NB).
    variable_capture : bool, default=False
        Model variable capture probability.
    mixture_model : bool, default=False
        Use mixture model for cell heterogeneity.
    n_components : Optional[int], default=None
        Number of mixture components (required if mixture_model=True).
    component_specific_params : bool, default=False
        Whether mixture components have their own parameters.

    Parameterization
    ----------------
    parameterization : str, default="standard"
        Model parameterization ("standard", "linked", "odds_ratio").
    unconstrained : bool, default=False
        Whether to use unconstrained parameterization variants. When True,
        parameters are sampled in unconstrained space and transformed via
        appropriate functions (e.g., sigmoid for probabilities, exp for positive
        values).
    guide_rank : Optional[int], default=None
        Rank of the low-rank approximation for the gene-specifc parameters
        (either r or mu). Only used with svi inference.

    Data Processing
    ---------------
    cells_axis : int, default=0
        Axis for cells in count matrix (0=rows, 1=columns).
    layer : Optional[str], default=None
        Layer in AnnData to use for counts. If None, uses .X.

    SVI Parameters
    --------------
    optimizer : Optional[Any], default=None
        Optimizer for variational inference (defaults to Adam).
    loss : Optional[Any], default=None
        Loss function for variational inference (defaults to
        TraceMeanField_ELBO).
    n_steps : int, default=100_000
        Number of optimization steps.
    batch_size : Optional[int], default=None
        Mini-batch size. If None, uses full dataset.
    stable_update : bool, default=True
        Use numerically stable parameter updates.

    MCMC Parameters
    ---------------
    n_samples : int, default=2_000
        Number of MCMC samples.
    n_warmup : int, default=1_000
        Number of warmup samples.
    n_chains : int, default=1
        Number of parallel chains.
    mcmc_kwargs : Optional[Dict[str, Any]], default=None
        Additional keyword arguments for the MCMC kernel.

    Prior Configuration
    -------------------
    r_prior, p_prior, gate_prior, p_capture_prior : Optional[tuple]
        Prior parameters as (param1, param2) tuples. - For unconstrained=True:
        (loc, scale) for Normal on transformed parameters - For
        unconstrained=False: Parameters for respective distributions (Beta,
        Gamma, etc.)
    mixing_prior : Optional[Any]
        Prior for mixture components (array-like or scalar).
    mu_prior, phi_prior, phi_capture_prior : Optional[tuple]
        Additional prior parameters for specific parameterizations.

    VAE Parameters
    --------------
    vae_latent_dim : int, default=3
        Dimension of the VAE latent space.
    vae_hidden_dims : Optional[list], default=None
        List of hidden layer dimensions (default: [256, 256]).
    vae_activation : Optional[str], default=None
        Activation function name for VAE (default: "gelu").
    vae_vcp_hidden_dims : Optional[List[int]], default=None
        Hidden layer dimensions for VCP encoder (default: [64, 32]). Only used
        when variable_capture=True.
    vae_vcp_activation : Optional[str], default=None
        Activation function for VCP encoder (default: "relu"). Only used when
        variable_capture=True.
    vae_prior_type : str, default="standard"
        Type of VAE prior ("standard" or "decoupled").
    vae_prior_hidden_dims : Optional[List[int]], default=None
        Hidden layer dimensions for decoupled prior coupling layers (default:
        [64, 64]).
    vae_prior_num_layers : Optional[int], default=None
        Number of coupling layers for decoupled prior (default: 2).
    vae_prior_activation : Optional[str], default=None
        Activation function for decoupled prior coupling layers (default:
        "relu").
    vae_prior_mask_type : str, default="alternating"
        Mask type for decoupled prior coupling layers ("alternating" or
        "checkerboard").
    vae_standardize : bool, default=False
        Whether to standardize the count data for VAE models. If True, applies
        z-standardization to the input data before encoding and reverses it
        after decoding. Recommended for count data with large dynamic range.

    General
    -------
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Union[ScribeSVIResults, ScribeMCMCResults, ScribeVAEResults]
        Results object containing inference results and diagnostics.

    Raises
    ------
    ValueError
        If configuration is invalid or required parameters are missing.

    Examples
    --------
    # SVI with standard parameterization results = run_scribe(counts,
    inference_method="svi", parameterization="standard")

    # MCMC with unconstrained parameterization results = run_scribe(counts,
    inference_method="mcmc", parameterization="standard", unconstrained=True)

    # SVI with odds_ratio parameterization for ZINBVCP mixture model results =
    run_scribe(
        counts, inference_method="svi", parameterization="odds_ratio",
        zero_inflated=True, variable_capture=True, mixture_model=True,
        n_components=3
    )

    # MCMC with linked parameterization results = run_scribe(
        counts, inference_method="mcmc", parameterization="linked",
        n_samples=1000
    )

    # VAE with standard parameterization results = run_scribe(
        counts, inference_method="vae", parameterization="standard",
        vae_latent_dim=5, vae_hidden_dims=[512, 256]
    )

    # VAE with linked unconstrained parameterization results = run_scribe(
        counts, inference_method="vae", parameterization="linked",
        unconstrained=True, vae_latent_dim=5, vae_hidden_dims=[512, 256]
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

    # Step 2: Build model configuration using builder pattern
    builder = (
        ModelConfigBuilder()
        .for_model(model_type)
        .with_parameterization(parameterization)
        .with_inference(inference_method)
    )

    # Add unconstrained if needed
    if unconstrained:
        builder.unconstrained()

    # Add mixture configuration
    if n_components is not None:
        builder.as_mixture(n_components, component_specific_params)

    # Add low-rank guide
    if guide_rank is not None:
        builder.with_low_rank_guide(guide_rank)

    # Add priors (simplified naming)
    priors = {}
    if r_prior is not None:
        priors["r"] = r_prior
    if p_prior is not None:
        priors["p"] = p_prior
    if gate_prior is not None:
        priors["gate"] = gate_prior
    if p_capture_prior is not None:
        priors["p_capture"] = p_capture_prior
    if mixing_prior is not None:
        priors["mixing"] = mixing_prior
    if mu_prior is not None:
        priors["mu"] = mu_prior
    if phi_prior is not None:
        priors["phi"] = phi_prior
    if phi_capture_prior is not None:
        priors["phi_capture"] = phi_capture_prior

    if priors:
        builder.with_priors(**priors)

    # Add VAE configuration
    if inference_method == "vae":
        builder.with_vae(
            latent_dim=vae_latent_dim,
            hidden_dims=vae_hidden_dims,
            activation=vae_activation,
            prior_type=vae_prior_type,
            prior_num_layers=vae_prior_num_layers,
            prior_hidden_dims=vae_prior_hidden_dims,
            prior_activation=vae_prior_activation,
            prior_mask_type=vae_prior_mask_type,
            standardize=vae_standardize,
            vcp_hidden_dims=vae_vcp_hidden_dims,
            vcp_activation=vae_vcp_activation,
        )

    # Build configuration
    model_config = builder.build()

    # Step 3: Run Inference
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
    elif inference_method == "vae":
        results = _run_vae_inference(
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
    else:
        raise ValueError(
            "Invalid inference_method. Choose 'svi', 'mcmc', or 'vae'"
        )

    return results


# ------------------------------------------------------------------------------
# SVI Inference
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# MCMC Inference
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# VAE Inference
# ------------------------------------------------------------------------------


def _run_vae_inference(
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
) -> ScribeVAEResults:
    """Helper function to run VAE inference."""
    # For now, VAE inference uses the same SVI engine but with VAE models
    # This will be enhanced later with dedicated VAE inference
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

    # Use SVI engine for VAE (VAE is essentially SVI with neural network
    # components)
    svi_results = SVIInferenceEngine.run_inference(**inference_kwargs)

    # Create base SVI results
    base_results = SVIResultsFactory.create_results(
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

    # Compute standardization statistics if requested by user
    if model_config.vae_standardize:
        from scribe.vae.architectures import compute_standardization_stats

        # Apply input transformation first (same as encoder)
        transformed_data = jnp.log1p(count_data)
        standardize_mean, standardize_std = compute_standardization_stats(
            transformed_data
        )

        # Store standardization parameters in model config
        model_config.standardize_mean = standardize_mean
        model_config.standardize_std = standardize_std
    else:
        standardize_mean = None
        standardize_std = None

    # Create VAE-specific results without passing the VAE model
    # The VAE model will be reconstructed when needed
    vae_results = ScribeVAEResults.from_svi_results(
        svi_results=base_results,
        vae_model=None,  # Don't pass VAE model to avoid pickling issues
        original_counts=count_data,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    return vae_results

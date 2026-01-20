"""
Simplified API for SCRIBE inference.

This module provides a user-friendly entry point for SCRIBE inference with
sensible defaults and flat kwargs instead of nested configuration objects.

Functions
---------
fit
    Main entry point for SCRIBE inference with simplified API.

Examples
--------
>>> import scribe
>>>
>>> # Simplest usage - just data and model name
>>> results = scribe.fit(adata, model="nbdm")
>>>
>>> # With customization via flat kwargs
>>> results = scribe.fit(
...     adata,
...     model="zinb",
...     parameterization="linked",
...     n_components=3,
...     n_steps=100000,
...     batch_size=512,
... )
>>>
>>> # Power users can still pass explicit config objects
>>> from scribe.models.config import ModelConfigBuilder, InferenceConfig, SVIConfig
>>> model_config = ModelConfigBuilder().for_model("nbdm").build()
>>> inference_config = InferenceConfig.from_svi(SVIConfig(n_steps=75000))
>>> results = scribe.fit(
...     adata,
...     model_config=model_config,
...     inference_config=inference_config,
... )
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from .models.config import (
    DataConfig,
    InferenceConfig,
    MCMCConfig,
    ModelConfig,
    SVIConfig,
)
from .models.config.enums import InferenceMethod
from .inference.utils import (
    process_counts_data,
    validate_inference_config_match,
)
from .inference.preset_builder import build_config_from_preset
from .inference.dispatcher import _run_inference

# Valid model types
VALID_MODELS = {"nbdm", "zinb", "nbvcp", "zinbvcp"}

# Valid parameterizations (including aliases)
VALID_PARAMETERIZATIONS = {
    "canonical",
    "linked",
    "odds_ratio",
    "standard",
    "mean_prob",
    "mean_odds",
}

# Valid inference methods
VALID_INFERENCE_METHODS = {"svi", "mcmc", "vae"}

# ==============================================================================
# Public API
# ==============================================================================


def fit(
    counts: Union[jnp.ndarray, "AnnData"],
    model: str = "nbdm",
    # Model options
    parameterization: str = "canonical",
    unconstrained: bool = False,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    guide_rank: Optional[int] = None,
    priors: Optional[Dict[str, Any]] = None,
    # Inference options
    inference_method: str = "svi",
    n_steps: int = 50_000,
    batch_size: Optional[int] = None,
    stable_update: bool = True,
    n_samples: int = 2_000,
    n_warmup: int = 1_000,
    n_chains: int = 1,
    # Data options
    cells_axis: int = 0,
    layer: Optional[str] = None,
    seed: int = 42,
    # Power user: explicit configs override above
    model_config: Optional[ModelConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
) -> Any:
    """
    Simplified entry point for SCRIBE inference.

    This function provides a user-friendly interface for running SCRIBE
    inference with sensible defaults. All options can be specified as flat
    keyword arguments instead of constructing nested configuration objects.

    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing single-cell RNA-seq counts.
        Shape should be (n_cells, n_genes) if cells_axis=0.

    model : str, default="nbdm"
        Model type to use:
            - "nbdm": Negative Binomial Dropout Model (simplest)
            - "zinb": Zero-Inflated Negative Binomial (handles excess zeros)
            - "nbvcp": NB with Variable Capture Probability (models technical
              dropout)
            - "zinbvcp": ZINB with Variable Capture Probability (most
              comprehensive)

    parameterization : str, default="canonical"
        Parameterization scheme:
            - "canonical" (or "standard"): Sample p ~ Beta, r ~ LogNormal
              directly
            - "linked" (or "mean_prob"): Sample p ~ Beta, mu ~ LogNormal, derive
              r
            - "odds_ratio" (or "mean_odds"): Sample phi ~ BetaPrime, mu ~
              LogNormal

    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained distributions.
        This can help with optimization in some cases.

    n_components : int, optional
        Number of mixture components for cell type discovery.
        If None (default), uses a single-component model.
        Must be >= 2 if specified.

    mixture_params : List[str], optional
        Which parameters should be component-specific in mixture models. If None
        and n_components is set, defaults to all gene-specific parameters.
        Example: ["r"] makes only r component-specific while p is shared.

    guide_rank : int, optional
        Rank for low-rank variational guide on gene-specific parameters. If None
        (default), uses mean-field guide (fully factorized). Low-rank guides can
        capture gene correlations but use more memory.

    priors : Dict[str, Any], optional
        Dictionary of prior hyperparameters keyed by parameter name. Values
        should be tuples of prior hyperparameters. Example: {"p": (1.0, 1.0),
        "r": (0.0, 1.0)}

    inference_method : str, default="svi"
        Inference method to use:
            - "svi": Stochastic Variational Inference (fast, scalable)
            - "mcmc": Markov Chain Monte Carlo (exact, slower)
            - "vae": Variational Autoencoder (for representation learning)

    n_steps : int, default=50_000
        Number of optimization steps for SVI/VAE inference.
        Increase for complex models or large datasets.

    batch_size : int, optional
        Mini-batch size for SVI/VAE. If None, uses full dataset.
        Recommended for large datasets (>10K cells).

    stable_update : bool, default=True
        Use numerically stable parameter updates in SVI.

    n_samples : int, default=2_000
        Number of MCMC samples to draw (only for inference_method="mcmc").

    n_warmup : int, default=1_000
        Number of MCMC warmup samples (only for inference_method="mcmc").

    n_chains : int, default=1
        Number of MCMC chains to run in parallel (only for
        inference_method="mcmc").

    cells_axis : int, default=0
        Axis for cells in count matrix. 0 means cells are rows (n_cells,
        n_genes).

    layer : str, optional
        Layer in AnnData to use for counts. If None, uses .X.

    seed : int, default=42
        Random seed for reproducibility.

    model_config : ModelConfig, optional
        Fully configured model configuration object.
        If provided, overrides model, parameterization, unconstrained,
        n_components, mixture_params, guide_rank, and priors.

    inference_config : InferenceConfig, optional
        Fully configured inference configuration object.
        If provided, overrides inference_method, n_steps, batch_size,
        stable_update, n_samples, n_warmup, and n_chains.

    Returns
    -------
    Union[ScribeSVIResults, ScribeMCMCResults, ScribeVAEResults]
        Results object containing:
        - Posterior samples or variational parameters
        - Loss history (for SVI/VAE)
        - Diagnostic information
        - Methods for analysis (log_likelihood, posterior_samples, etc.)

    Raises
    ------
    ValueError
        If model, parameterization, or inference_method is not recognized.
        If configuration is invalid.

    Examples
    --------
    Basic usage with NBDM model:

    >>> results = scribe.fit(adata, model="nbdm")

    Zero-inflated model with mixture components:

    >>> results = scribe.fit(
    ...     adata,
    ...     model="zinb",
    ...     n_components=3,
    ...     n_steps=100000,
    ... )

    Linked parameterization with low-rank guide:

    >>> results = scribe.fit(
    ...     adata,
    ...     model="nbdm",
    ...     parameterization="linked",
    ...     guide_rank=15,
    ... )

    MCMC inference for small datasets:

    >>> results = scribe.fit(
    ...     adata,
    ...     model="nbdm",
    ...     inference_method="mcmc",
    ...     n_samples=5000,
    ...     n_chains=4,
    ... )

    See Also
    --------
    run_scribe : Lower-level inference function with more options.
    ModelConfigBuilder : Builder for creating ModelConfig objects.
    InferenceConfig : Unified inference configuration class.
    """
    # ==========================================================================
    # Step 1: Validate inputs
    # ==========================================================================
    if model_config is None:
        # Validate model type
        model_lower = model.lower()
        if model_lower not in VALID_MODELS:
            raise ValueError(
                f"Unknown model: '{model}'. "
                f"Valid models are: {', '.join(sorted(VALID_MODELS))}"
            )

        # Validate parameterization
        param_lower = parameterization.lower()
        if param_lower not in VALID_PARAMETERIZATIONS:
            raise ValueError(
                f"Unknown parameterization: '{parameterization}'. "
                f"Valid parameterizations are: "
                f"{', '.join(sorted(VALID_PARAMETERIZATIONS))}"
            )

    if inference_config is None:
        # Validate inference method
        method_lower = inference_method.lower()
        if method_lower not in VALID_INFERENCE_METHODS:
            raise ValueError(
                f"Unknown inference_method: '{inference_method}'. "
                f"Valid methods are: "
                f"{', '.join(sorted(VALID_INFERENCE_METHODS))}"
            )

    # ==========================================================================
    # Step 2: Process data
    # ==========================================================================
    data_config = DataConfig(cells_axis=cells_axis, layer=layer)
    count_data, adata, n_cells, n_genes = process_counts_data(
        counts, data_config
    )

    # ==========================================================================
    # Step 3: Build or use ModelConfig
    # ==========================================================================
    if model_config is None:
        model_config = build_config_from_preset(
            model=model.lower(),
            parameterization=parameterization.lower(),
            inference_method=inference_method.lower(),
            unconstrained=unconstrained,
            guide_rank=guide_rank,
            n_components=n_components,
            mixture_params=mixture_params,
            priors=priors,
        )

    # ==========================================================================
    # Step 4: Build or use InferenceConfig
    # ==========================================================================
    if inference_config is None:
        # Determine inference method from model_config or parameter
        method = model_config.inference_method

        if method == InferenceMethod.SVI:
            svi_config = SVIConfig(
                n_steps=n_steps,
                batch_size=batch_size,
                stable_update=stable_update,
            )
            inference_config = InferenceConfig.from_svi(svi_config)
        elif method == InferenceMethod.MCMC:
            mcmc_config = MCMCConfig(
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_chains=n_chains,
            )
            inference_config = InferenceConfig.from_mcmc(mcmc_config)
        elif method == InferenceMethod.VAE:
            # VAE uses SVI config
            svi_config = SVIConfig(
                n_steps=n_steps,
                batch_size=batch_size,
                stable_update=stable_update,
            )
            inference_config = InferenceConfig.from_vae(svi_config)
        else:
            raise ValueError(f"Unknown inference method: {method}")
    else:
        # Validate that inference_config matches model_config
        validate_inference_config_match(model_config, inference_config)

    # ==========================================================================
    # Step 5: Run inference
    # ==========================================================================
    return _run_inference(
        inference_config.method,
        model_config=model_config,
        count_data=count_data,
        inference_config=inference_config,
        adata=adata,
        n_cells=n_cells,
        n_genes=n_genes,
        data_config=data_config,
        seed=seed,
    )

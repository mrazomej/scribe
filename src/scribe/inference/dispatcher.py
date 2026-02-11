"""
Inference routing using registry pattern.

This module provides the routing mechanism that dispatches inference execution
to the appropriate handler based on inference method. Uses a registry pattern
for extensibility.
"""

from typing import TYPE_CHECKING, Any, Optional, Callable
import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import (
    ModelConfig,
    InferenceConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
)
from ..models.config.enums import InferenceMethod

# ==============================================================================
# Registry for inference handlers
# ==============================================================================

# Type alias for inference handler function signature
_InferenceHandler = Callable[
    [
        ModelConfig,
        jnp.ndarray,
        Optional["AnnData"],
        int,
        int,
        SVIConfig | MCMCConfig,
        DataConfig,
        int,
    ],
    Any,
]

# Registry mapping inference methods to their handler functions
# Handlers are registered at module import time
_INFERENCE_HANDLERS: dict[InferenceMethod, _InferenceHandler] = {}


def _register_inference_handler(
    method: InferenceMethod, handler: _InferenceHandler
) -> None:
    """Register an inference handler function.

    This is an internal function used to populate the registry. Handlers
    are registered at module import time.

    Parameters
    ----------
    method : InferenceMethod
        The inference method this handler handles.
    handler : _InferenceHandler
        Handler function that executes inference for this method.
    """
    _INFERENCE_HANDLERS[method] = handler


# ------------------------------------------------------------------------------
# Handler implementations (registered below)
# ------------------------------------------------------------------------------


def _svi_handler(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    config: SVIConfig | MCMCConfig,
    data_config: DataConfig,
    seed: int,
    annotation_prior_logits: Optional[jnp.ndarray] = None,
) -> Any:
    """Handler for SVI inference."""
    from .svi import _run_svi_inference

    if not isinstance(config, SVIConfig):
        raise ValueError(f"Expected SVIConfig for SVI, got {type(config)}")

    return _run_svi_inference(
        model_config=model_config,
        count_data=count_data,
        adata=adata,
        n_cells=n_cells,
        n_genes=n_genes,
        svi_config=config,
        data_config=data_config,
        seed=seed,
        annotation_prior_logits=annotation_prior_logits,
    )


# ------------------------------------------------------------------------------


def _mcmc_handler(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    config: SVIConfig | MCMCConfig,
    data_config: DataConfig,
    seed: int,
    annotation_prior_logits: Optional[jnp.ndarray] = None,
) -> Any:
    """Handler for MCMC inference."""
    from .mcmc import _run_mcmc_inference

    if not isinstance(config, MCMCConfig):
        raise ValueError(f"Expected MCMCConfig for MCMC, got {type(config)}")

    return _run_mcmc_inference(
        model_config=model_config,
        count_data=count_data,
        adata=adata,
        n_cells=n_cells,
        n_genes=n_genes,
        mcmc_config=config,
        data_config=data_config,
        seed=seed,
        annotation_prior_logits=annotation_prior_logits,
    )


# ------------------------------------------------------------------------------


def _vae_handler(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    config: SVIConfig | MCMCConfig,
    data_config: DataConfig,
    seed: int,
    annotation_prior_logits: Optional[jnp.ndarray] = None,
) -> Any:
    """Handler for VAE inference."""
    from .vae import _run_vae_inference

    if not isinstance(config, SVIConfig):
        raise ValueError(f"Expected SVIConfig for VAE, got {type(config)}")

    return _run_vae_inference(
        model_config=model_config,
        count_data=count_data,
        adata=adata,
        n_cells=n_cells,
        n_genes=n_genes,
        svi_config=config,
        data_config=data_config,
        seed=seed,
    )


# ------------------------------------------------------------------------------

# Register handlers
_register_inference_handler(InferenceMethod.SVI, _svi_handler)
_register_inference_handler(InferenceMethod.MCMC, _mcmc_handler)
_register_inference_handler(InferenceMethod.VAE, _vae_handler)


# ==============================================================================
# Public API
# ==============================================================================


def _run_inference(
    inference_method: InferenceMethod,
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    inference_config: InferenceConfig,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    data_config: DataConfig,
    seed: int,
    annotation_prior_logits: Optional[jnp.ndarray] = None,
) -> Any:
    """Route inference execution to the appropriate handler.

    This function uses a registry pattern to route inference execution to the
    appropriate handler based on the inference method. The registry makes it
    easy to extend with new inference methods without modifying this function.

    Parameters
    ----------
    inference_method : InferenceMethod
        The inference method to use.
    model_config : ModelConfig
        Model configuration object.
    count_data : jnp.ndarray
        Processed count data (cells as rows).
    inference_config : InferenceConfig
        Unified inference configuration.
    adata : Optional[AnnData]
        Original AnnData object if provided.
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    data_config : DataConfig
        Data processing configuration.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Any
        Results object (ScribeSVIResults, ScribeMCMCResults, or
        ScribeVAEResults).

    Raises
    ------
    ValueError
        If the inference method is not recognized, if there's a mismatch
        between inference_method and inference_config.method, or if the
        required config type is missing.

    Notes
    -----
    The registry pattern allows easy extension - new inference methods can
    register their handlers without modifying this function.

    See Also
    --------
    _run_svi_inference : SVI inference handler.
    _run_mcmc_inference : MCMC inference handler.
    _run_vae_inference : VAE inference handler.
    """
    # Validate that inference_method matches inference_config.method
    if inference_method != inference_config.method:
        raise ValueError(
            f"Inference method mismatch: "
            f"inference_method={inference_method.value} "
            f"but inference_config.method={inference_config.method.value}"
        )

    # Extract the appropriate config based on method
    if (
        inference_method == InferenceMethod.SVI
        or inference_method == InferenceMethod.VAE
    ):
        config = inference_config.svi
        if config is None:
            raise ValueError(
                f"SVIConfig required for {inference_method.value} inference"
            )
    elif inference_method == InferenceMethod.MCMC:
        config = inference_config.mcmc
        if config is None:
            raise ValueError("MCMCConfig required for MCMC inference")
    else:
        raise ValueError(f"Unknown inference method: {inference_method}")

    # Route to appropriate handler using registry
    handler = _INFERENCE_HANDLERS.get(inference_method)
    if handler is None:
        raise ValueError(
            f"Unknown inference method: {inference_method}. "
            f"Registered methods: "
            f"{list(_INFERENCE_HANDLERS.keys())}"
        )

    return handler(
        model_config=model_config,
        count_data=count_data,
        adata=adata,
        n_cells=n_cells,
        n_genes=n_genes,
        config=config,
        data_config=data_config,
        seed=seed,
        annotation_prior_logits=annotation_prior_logits,
    )

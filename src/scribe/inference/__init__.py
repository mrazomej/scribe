"""
Unified inference interface for SCRIBE.

This module provides a single entry point for all SCRIBE inference methods
(SVI, MCMC, VAE) with support for both simple preset-based and advanced
ModelConfig-based APIs.
"""

from typing import Union, Optional, Any, TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import (
    ModelConfig,
    InferenceConfig,
    DataConfig,
)
from ..models.config.enums import InferenceMethod
from .utils import process_counts_data, validate_inference_config_match
from .preset_builder import build_config_from_preset
from .inference_config import create_default_inference_config
from .dispatcher import _run_inference

# Public API
__all__ = ["run_scribe"]


# ==============================================================================
# Public API
# ==============================================================================


def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    # Simple preset-based API
    model: Optional[str] = None,
    parameterization: str = "canonical",
    inference_method: str = "svi",
    # OR advanced API
    model_config: Optional[ModelConfig] = None,
    # Unified inference config (optional, will use defaults if not provided)
    inference_config: Optional[InferenceConfig] = None,
    # Data processing config (optional, can pass individual params)
    data_config: Optional[DataConfig] = None,
    cells_axis: int = 0,  # Used if data_config not provided
    layer: Optional[str] = None,  # Used if data_config not provided
    # Common parameters
    seed: int = 42,
) -> Any:
    """Unified inference interface with preset support.

    This function provides a single entry point for all SCRIBE inference
    methods. It supports two usage patterns:

        1. **Simple preset-based API**: Specify model type and parameters as
           strings
        2. **Advanced API**: Pass fully configured ModelConfig and
           InferenceConfig objects

    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing single-cell RNA-seq counts.
    model : Optional[str], default=None
        Model type for preset-based API: "nbdm", "zinb", "nbvcp", or "zinbvcp".
        Required if model_config is None.
    parameterization : str, default="canonical"
        Parameterization scheme: "canonical", "mean_prob", "mean_odds"
        (backward compat: "standard", "linked", "odds_ratio").
        Only used if model_config is None.
    inference_method : str, default="svi"
        Inference method: "svi", "mcmc", or "vae".
        Only used if model_config is None.
    model_config : Optional[ModelConfig], default=None
        Fully configured model configuration. If provided, overrides preset
        parameters (model, parameterization, inference_method).
    inference_config : Optional[InferenceConfig], default=None
        Unified inference configuration. If None, will be created with defaults
        based on the inference method. Can be created using:
        - InferenceConfig.from_svi(svi_config)
        - InferenceConfig.from_mcmc(mcmc_config)
        - InferenceConfig.from_vae(svi_config)
    data_config : Optional[DataConfig], default=None
        Data processing configuration. If None, uses cells_axis and layer params.
    cells_axis : int, default=0
        Axis for cells in count matrix (0=rows, 1=columns). Used if data_config=None.
    layer : Optional[str], default=None
        Layer in AnnData to use for counts. If None, uses .X. Used if data_config=None.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Union[ScribeSVIResults, ScribeMCMCResults, ScribeVAEResults]
        Results object containing inference results and diagnostics.

    Raises
    ------
    ValueError
        If configuration is invalid, required parameters are missing, or there
        are mismatches between model_config and inference_config.

    Examples
    --------
    Simple preset-based usage:

    >>> from scribe.inference import run_scribe
    >>> from scribe.models.config import InferenceConfig, SVIConfig
    >>>
    >>> # Create inference config
    >>> svi_config = SVIConfig(n_steps=50000, batch_size=256)
    >>> inference_config = InferenceConfig.from_svi(svi_config)
    >>>
    >>> # Run inference
    >>> results = run_scribe(
    ...     counts=adata,
    ...     model="nbdm",
    ...     parameterization="mean_prob",
    ...     inference_method="svi",
    ...     inference_config=inference_config,
    ... )

    Advanced usage with ModelConfig:

    >>> from scribe.models.config import ModelConfigBuilder, InferenceConfig, MCMCConfig
    >>>
    >>> # Build model config
    >>> model_config = (ModelConfigBuilder()
    ...     .for_model("zinb")
    ...     .with_parameterization("mean_odds")
    ...     .with_inference("mcmc")
    ...     .build())
    >>>
    >>> # Build inference config
    >>> mcmc_config = MCMCConfig(n_samples=5000, n_chains=4)
    >>> inference_config = InferenceConfig.from_mcmc(mcmc_config)
    >>>
    >>> # Run inference
    >>> results = run_scribe(
    ...     counts=adata,
    ...     model_config=model_config,
    ...     inference_config=inference_config,
    ... )

    Notes
    -----
    - If both preset parameters and model_config are provided, model_config
      takes precedence.
    - If inference_config is None, defaults are created based on the inference
      method (100k steps for SVI/VAE, 2000 samples for MCMC).
    - Parameterization names support both new ("canonical", "mean_prob",
      "mean_odds") and old ("standard", "linked", "odds_ratio") for backward
      compatibility.

    See Also
    --------
    ModelConfigBuilder : Builder for creating ModelConfig objects.
    InferenceConfig : Unified inference configuration class.
    build_config_from_preset : Build ModelConfig from preset parameters.
    """
    # ==========================================================================
    # Step 1: Process data
    # ==========================================================================
    if data_config is None:
        data_config = DataConfig(cells_axis=cells_axis, layer=layer)

    count_data, adata, n_cells, n_genes = process_counts_data(
        counts, data_config
    )

    # ==========================================================================
    # Step 2: Build or validate ModelConfig
    # ==========================================================================
    if model_config is None:
        if model is None:
            raise ValueError(
                "Either 'model' or 'model_config' must be provided"
            )
        model_config = build_config_from_preset(
            model=model,
            parameterization=parameterization,
            inference_method=inference_method,
        )
    else:
        # If model_config is provided, validate inference_method matches if
        # specified (only validate if inference_method was explicitly set, not
        # default)
        if inference_method != "svi":
            from .utils import validate_model_inference_match

            validate_model_inference_match(model_config, inference_method)

    # ==========================================================================
    # Step 3: Build or validate InferenceConfig
    # ==========================================================================
    if inference_config is None:
        # Create default config based on inference method
        inference_method_enum = InferenceMethod(
            inference_method
            if model_config is None
            else model_config.inference_method.value
        )
        inference_config = create_default_inference_config(
            inference_method_enum
        )
    else:
        # Validate inference config matches model config
        validate_inference_config_match(model_config, inference_config)

    # ==========================================================================
    # Step 4: Route to inference method (multiple dispatch)
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

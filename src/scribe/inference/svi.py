"""
SVI (Stochastic Variational Inference) execution.

This module handles the execution of SVI inference, including integration
with the SVI inference engine and results factory. Supports early stopping
based on loss convergence.
"""

from typing import TYPE_CHECKING, Any, Optional
import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import ModelConfig, SVIConfig, DataConfig
from ..svi import SVIInferenceEngine, SVIResultsFactory

# ==============================================================================
# SVI Inference Engine
# ==============================================================================


def _run_svi_inference(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    svi_config: SVIConfig,
    data_config: DataConfig,
    seed: int,
    annotation_prior_logits: Optional[jnp.ndarray] = None,
) -> Any:
    """Execute SVI inference with optional early stopping.

    This function runs SVI inference using the provided model and inference
    configurations. It handles the integration between the inference engine
    and results factory, with support for early stopping based on loss
    convergence.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration object specifying the model architecture.
    count_data : jnp.ndarray
        Processed count data matrix (cells as rows, genes as columns).
    adata : Optional[AnnData]
        Original AnnData object if provided. Used for creating results with
        AnnData integration.
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes in the dataset.
    svi_config : SVIConfig
        SVI-specific configuration (optimizer, loss, n_steps, early_stopping,
        etc.).
    data_config : DataConfig
        Data processing configuration (not used directly here, but passed
        for consistency).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ScribeSVIResults
        Results object containing:
        - Optimized variational parameters
        - Loss history
        - Posterior distributions
        - Diagnostic information
        - Early stopping metadata (if early stopping was used)

    Examples
    --------
    >>> from scribe.models.config import ModelConfig, SVIConfig, EarlyStoppingConfig
    >>> from scribe.inference.svi import _run_svi_inference
    >>>
    >>> # Run SVI inference with early stopping
    >>> results = _run_svi_inference(
    ...     model_config=model_config,
    ...     count_data=count_data,
    ...     adata=adata,
    ...     n_cells=1000,
    ...     n_genes=2000,
    ...     svi_config=SVIConfig(
    ...         n_steps=50000,
    ...         early_stopping=EarlyStoppingConfig(patience=500),
    ...     ),
    ...     data_config=data_config,
    ...     seed=42
    ... )

    Notes
    -----
    - The SVI inference engine handles the actual optimization loop.
    - When early stopping is enabled, training stops when the loss stops
      improving (no improvement > min_delta for patience steps).
    - Results are packaged by SVIResultsFactory which creates a comprehensive
      results object with posterior distributions and diagnostics.
    - If adata is provided, results will be integrated with the AnnData object.

    See Also
    --------
    SVIInferenceEngine : Core SVI inference execution.
    SVIResultsFactory : Results packaging and creation.
    ScribeSVIResults : Comprehensive SVI results object.
    EarlyStoppingConfig : Configuration for early stopping criteria.
    """
    # Extract parameters from SVI config
    optimizer = svi_config.optimizer
    loss = svi_config.loss
    n_steps = svi_config.n_steps
    batch_size = svi_config.batch_size
    stable_update = svi_config.stable_update
    early_stopping = svi_config.early_stopping

    # Build inference kwargs for the engine
    inference_kwargs = {
        "model_config": model_config,
        "count_data": count_data,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seed": seed,
        "stable_update": stable_update,
        "early_stopping": early_stopping,
        "annotation_prior_logits": annotation_prior_logits,
    }

    # Add optional optimizer and loss if provided
    # If None, the engine will use defaults (Adam optimizer, TraceMeanField_ELBO)
    if optimizer is not None:
        inference_kwargs["optimizer"] = optimizer
    if loss is not None:
        inference_kwargs["loss"] = loss

    # Run SVI inference
    svi_results = SVIInferenceEngine.run_inference(**inference_kwargs)

    # Use config with param_specs from run result when available (for gene
    # subsetting)
    config_for_results = (
        svi_results.model_config
        if svi_results.model_config is not None
        else model_config
    )

    # Compute model_type: add _mix suffix for mixture models
    model_type = config_for_results.base_model
    if config_for_results.n_components is not None:
        model_type = f"{config_for_results.base_model}_mix"

    # Package results using the factory
    return SVIResultsFactory.create_results(
        svi_results=svi_results,
        adata=adata,
        model_config=config_for_results,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_type,
        n_components=model_config.n_components,
        prior_params=model_config.get_active_priors(),
    )

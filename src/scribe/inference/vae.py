"""
VAE (Variational Autoencoder) execution.

This module handles the execution of VAE inference. VAE inference uses the
SVI engine but with neural network components, and creates VAE-specific results.
"""

from typing import TYPE_CHECKING, Any, Optional
import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import ModelConfig, SVIConfig, DataConfig
from ..svi import SVIInferenceEngine, SVIResultsFactory
from ..vae import ScribeVAEResults

# ==============================================================================
# VAE Inference Engine
# ==============================================================================


def _run_vae_inference(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    svi_config: SVIConfig,
    data_config: DataConfig,
    seed: int,
) -> ScribeVAEResults:
    """Execute VAE inference.

    This function runs VAE inference using the SVI engine (VAE is essentially
    SVI with neural network components). It creates VAE-specific results that
    include the VAE model and standardization statistics.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration object specifying the model architecture.
        Must have VAE configuration (model_config.vae).
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
        SVI configuration (VAE uses SVI for optimization).
    data_config : DataConfig
        Data processing configuration (not used directly here, but passed
        for consistency).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ScribeVAEResults
        VAE-specific results object containing:
        - Optimized variational parameters
        - Loss history
        - VAE model (encoder/decoder)
        - Standardization statistics (if applicable)
        - Posterior distributions

    Examples
    --------
    >>> from scribe.models.config import ModelConfig, SVIConfig
    >>> from scribe.inference.vae import _run_vae_inference
    >>>
    >>> # Run VAE inference
    >>> results = _run_vae_inference(
    ...     model_config=model_config,
    ...     count_data=count_data,
    ...     adata=adata,
    ...     n_cells=1000,
    ...     n_genes=2000,
    ...     svi_config=SVIConfig(n_steps=100000),
    ...     data_config=data_config,
    ...     seed=42
    ... )

    Notes
    -----
    - VAE inference uses the SVI engine since it's essentially SVI with
      neural network components.
    - If standardization is enabled (model_config.vae.standardize=True),
      standardization statistics are computed and stored.
    - The VAE model is not stored directly in results to avoid pickling
      issues; it will be reconstructed when needed.

    See Also
    --------
    SVIInferenceEngine : Core SVI inference execution (used by VAE).
    SVIResultsFactory : Base results creation.
    ScribeVAEResults : VAE-specific results object.
    """
    # Extract parameters from SVI config
    optimizer = svi_config.optimizer
    loss = svi_config.loss
    n_steps = svi_config.n_steps
    batch_size = svi_config.batch_size
    stable_update = svi_config.stable_update

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
    }

    # Add optional optimizer and loss if provided
    if optimizer is not None:
        inference_kwargs["optimizer"] = optimizer
    if loss is not None:
        inference_kwargs["loss"] = loss

    # Use SVI engine for VAE (VAE is essentially SVI with neural network components)
    svi_results = SVIInferenceEngine.run_inference(**inference_kwargs)

    # Create base SVI results first
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

    # Compute standardization statistics if requested
    standardize_mean = None
    standardize_std = None
    if model_config.vae is not None and model_config.vae.standardize:
        from ..vae.architectures import compute_standardization_stats

        # Apply input transformation first (same as encoder)
        transformed_data = jnp.log1p(count_data)
        standardize_mean, standardize_std = compute_standardization_stats(
            transformed_data
        )

        # Store standardization parameters in model config if mutable
        # Note: ModelConfig is frozen, so we can't modify it directly
        # The stats will be stored in the VAE results instead

    # Create VAE-specific results
    # Note: We don't pass the VAE model to avoid pickling issues
    # The VAE model will be reconstructed when needed
    vae_results = ScribeVAEResults.from_svi_results(
        svi_results=base_results,
        vae_model=None,  # Don't pass VAE model to avoid pickling issues
        original_counts=count_data,
        standardize_mean=standardize_mean,
        standardize_std=standardize_std,
    )

    return vae_results

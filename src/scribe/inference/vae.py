"""
VAE (Variational Autoencoder) execution.

This module handles the execution of VAE inference. VAE inference uses the
SVI engine but with neural network components (encoder, decoder, flow prior),
and creates VAE-specific results via the composable factory.

Uses ScribeVAEResults.from_training() from scribe.svi.vae_results — no
dependency on legacy scribe.vae modules.
"""

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import ModelConfig, SVIConfig, DataConfig
from ..models.components.guide_families import VAELatentGuide
from ..svi import SVIInferenceEngine
from ..svi.vae_results import ScribeVAEResults

# ==============================================================================
# VAE Inference
# ==============================================================================


def _extract_vae_guide_from_param_specs(
    param_specs,
) -> Optional[VAELatentGuide]:
    """
    Extract VAELatentGuide from param specs (used for results construction).
    """
    if param_specs is None:
        return None
    for spec in param_specs:
        gf = getattr(spec, "guide_family", None)
        if isinstance(gf, VAELatentGuide) and gf.decoder is not None:
            return gf
    return None


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
    """Execute VAE inference using the composable factory.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration with VAE config (model_config.vae).
    count_data : jnp.ndarray
        Processed count data matrix (cells as rows, genes as columns).
    adata : Optional[AnnData]
        Original AnnData object if provided.
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    svi_config : SVIConfig
        SVI configuration for optimization.
    data_config : DataConfig
        Data processing configuration.
    seed : int
        Random seed.

    Returns
    -------
    ScribeVAEResults
        VAE-specific results from ScribeVAEResults.from_training().
    """
    # 1. Run SVI (engine uses get_model_and_guide with n_genes for VAE path)
    svi_results = SVIInferenceEngine.run_inference(
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        n_steps=svi_config.n_steps,
        batch_size=svi_config.batch_size,
        seed=seed,
        stable_update=svi_config.stable_update,
        early_stopping=svi_config.early_stopping,
        optimizer=svi_config.optimizer,
        loss=svi_config.loss,
    )

    # 2. Extract VAELatentGuide from param_specs (in model_config_for_results)
    model_config_for_results = svi_results.model_config
    param_specs = (
        getattr(model_config_for_results, "param_specs", None)
        if model_config_for_results is not None
        else None
    )
    vae_guide_family = _extract_vae_guide_from_param_specs(param_specs)
    if vae_guide_family is None:
        raise ValueError(
            "VAE inference completed but VAELatentGuide not found in param_specs. "
            "The composable factory should have set this."
        )

    # 3. Build results with composable ScribeVAEResults.from_training
    return ScribeVAEResults.from_training(
        params=svi_results.params,
        loss_history=svi_results.losses,
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=model_config_for_results,
        vae_guide_family=vae_guide_family,
    )

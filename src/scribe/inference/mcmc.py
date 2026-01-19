"""
MCMC (Markov Chain Monte Carlo) execution.

This module handles the execution of MCMC inference, including integration
with the MCMC inference engine and results factory.
"""

from typing import TYPE_CHECKING, Any, Optional
import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import ModelConfig, MCMCConfig, DataConfig
from ..mcmc import MCMCInferenceEngine, MCMCResultsFactory


# ==============================================================================
# MCMC Inference Engine
# ==============================================================================


def _run_mcmc_inference(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    mcmc_config: MCMCConfig,
    data_config: DataConfig,
    seed: int,
) -> Any:
    """Execute MCMC inference.

    This function runs MCMC inference using the provided model and inference
    configurations. It handles the integration between the inference engine
    and results factory.

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
    mcmc_config : MCMCConfig
        MCMC-specific configuration (n_samples, n_warmup, n_chains, etc.).
    data_config : DataConfig
        Data processing configuration (not used directly here, but passed
        for consistency).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ScribeMCMCResults
        Results object containing:
        - MCMC samples
        - Posterior distributions
        - Diagnostic information (R-hat, effective sample size, etc.)

    Examples
    --------
    >>> from scribe.models.config import ModelConfig, MCMCConfig
    >>> from scribe.inference.mcmc import _run_mcmc_inference
    >>>
    >>> # Run MCMC inference
    >>> results = _run_mcmc_inference(
    ...     model_config=model_config,
    ...     count_data=count_data,
    ...     adata=adata,
    ...     n_cells=1000,
    ...     n_genes=2000,
    ...     mcmc_config=MCMCConfig(n_samples=5000, n_chains=4),
    ...     data_config=data_config,
    ...     seed=42
    ... )

    Notes
    -----
    - The MCMC inference engine uses NUTS (No-U-Turn Sampler) by default.
    - Results are packaged by MCMCResultsFactory which creates a comprehensive
      results object with posterior samples and diagnostics.
    - If adata is provided, results will be integrated with the AnnData object.

    See Also
    --------
    MCMCInferenceEngine : Core MCMC inference execution.
    MCMCResultsFactory : Results packaging and creation.
    ScribeMCMCResults : Comprehensive MCMC results object.
    """
    # Extract parameters from MCMC config
    n_samples = mcmc_config.n_samples
    n_warmup = mcmc_config.n_warmup
    n_chains = mcmc_config.n_chains
    mcmc_kwargs = mcmc_config.mcmc_kwargs

    # Run MCMC inference
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

    # Compute model_type: add _mix suffix for mixture models
    model_type = model_config.base_model
    if model_config.n_components is not None:
        model_type = f"{model_config.base_model}_mix"

    # Package results using the factory
    return MCMCResultsFactory.create_results(
        mcmc_results=mcmc,
        adata=adata,
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        model_type=model_type,
        n_components=model_config.n_components,
        prior_params=model_config.get_active_priors(),
    )

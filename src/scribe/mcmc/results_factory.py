"""
Results factory for MCMC inference.

This module handles the packaging of MCMC results into ScribeMCMCResults
objects.
"""

from typing import Optional, Dict, Any
import jax.numpy as jnp

from ..mcmc.results import ScribeMCMCResults
from ..models.config import ModelConfig


class MCMCResultsFactory:
    """Factory for creating MCMC results objects."""

    @staticmethod
    def create_results(
        mcmc_results: Any,
        model_config: ModelConfig,
        adata: Optional["AnnData"],
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        model_type: str,
        n_components: Optional[int],
        prior_params: Dict[str, Any],
    ) -> ScribeMCMCResults:
        """Package MCMC results into a ``ScribeMCMCResults`` object.

        Parameters
        ----------
        mcmc_results : numpyro.infer.MCMC
            Raw MCMC results from NumPyro.
        model_config : ModelConfig
            Model configuration object.
        adata : Optional[AnnData]
            Original AnnData object (if provided).
        count_data : jnp.ndarray
            Processed count data.
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        model_type : str
            Type of model.
        n_components : Optional[int]
            Number of mixture components.
        prior_params : Dict[str, Any]
            Dictionary of prior parameters.

        Returns
        -------
        ScribeMCMCResults
            Packaged results object.
        """
        if adata is not None:
            return ScribeMCMCResults.from_anndata(
                mcmc=mcmc_results,
                adata=adata,
                model_type=model_type,
                model_config=model_config,
                n_components=n_components,
                prior_params=prior_params,
            )

        return ScribeMCMCResults.from_mcmc(
            mcmc=mcmc_results,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            n_components=n_components,
            prior_params=prior_params,
        )

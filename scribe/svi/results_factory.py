"""
Results factory for SVI inference.

This module handles the packaging of SVI results into ScribeSVIResults objects.
"""

from typing import Optional, Dict, Any
import jax.numpy as jnp
from ..svi.results import ScribeSVIResults
from ..model_config import ModelConfig


class SVIResultsFactory:
    """Factory for creating SVI results objects."""
    
    @staticmethod
    def create_results(
        svi_results: Any,
        model_config: ModelConfig,
        adata: Optional["AnnData"],
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        model_type: str,
        n_components: Optional[int],
        prior_params: Dict[str, Any]
    ) -> ScribeSVIResults:
        """
        Package SVI results into ScribeSVIResults object.
        
        Parameters
        ----------
        svi_results : Any
            Raw SVI results from numpyro
        model_config : ModelConfig
            Model configuration object
        adata : Optional[AnnData]
            Original AnnData object (if provided)
        count_data : jnp.ndarray
            Processed count data
        n_cells : int
            Number of cells
        n_genes : int
            Number of genes
        model_type : str
            Type of model
        n_components : Optional[int]
            Number of mixture components
        prior_params : Dict[str, Any]
            Dictionary of prior parameters
            
        Returns
        -------
        ScribeSVIResults
            Packaged results object
        """
        if adata is not None:
            results = ScribeSVIResults.from_anndata(
                adata=adata,
                params=svi_results.params,
                loss_history=svi_results.losses,
                model_type=model_type,
                model_config=model_config,
                n_components=n_components,
                prior_params=prior_params,
            )
        else:
            results = ScribeSVIResults(
                params=svi_results.params,
                loss_history=svi_results.losses,
                n_cells=n_cells,
                n_genes=n_genes,
                model_type=model_type,
                model_config=model_config,
                n_components=n_components,
                prior_params=prior_params,
            )
        
        return results 
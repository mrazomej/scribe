"""
Results factory for SVI inference.

This module handles the packaging of SVI results into ScribeSVIResults objects.
"""

from typing import Optional, Dict, Any
import jax.numpy as jnp
from ..svi.results import ScribeSVIResults
from ..models.config import ModelConfig
from ..core.axis_layout import build_param_layouts, gene_axes_from_layouts


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
        prior_params: Dict[str, Any],
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
        gene_axis_by_key = None
        param_layouts = None
        specs = getattr(model_config, "param_specs", None)
        if specs:
            # Build semantic axis layouts from the full ParamSpec list
            param_layouts = build_param_layouts(
                specs, svi_results.params, has_sample_dim=False
            )
            # Forward path: gene axes from ``AxisLayout`` (SVI ``layouts`` /
            # ``reconstruct_param_layouts`` stay consistent).  Legacy pickles
            # without ``param_layouts`` still rely on stored ``_gene_axis_by_key``.
            gene_axis_by_key = (
                gene_axes_from_layouts(param_layouts) or None
            )

        if adata is not None:
            results = ScribeSVIResults.from_anndata(
                adata=adata,
                params=svi_results.params,
                loss_history=svi_results.losses,
                model_type=model_type,
                model_config=model_config,
                n_components=n_components,
                prior_params=prior_params,
                _gene_axis_by_key=gene_axis_by_key,
                param_layouts=param_layouts,
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
                _gene_axis_by_key=gene_axis_by_key,
                param_layouts=param_layouts,
            )

        return results

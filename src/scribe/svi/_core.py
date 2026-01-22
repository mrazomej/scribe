"""
Core mixin for SVI results initialization and construction.

This mixin provides basic initialization and construction methods for
ScribeSVIResults.
"""

from typing import Any, Dict
import jax.numpy as jnp
from ..models.config import ModelConfig

# ==============================================================================
# Core Results Mixin
# ==============================================================================


class CoreResultsMixin:
    """Mixin providing core initialization and construction methods."""

    def __post_init__(self):
        """Validate model configuration and parameters."""
        # Set n_components from model_config if not explicitly provided
        if (
            self.n_components is None
            and self.model_config.n_components is not None
        ):
            self.n_components = self.model_config.n_components

    # --------------------------------------------------------------------------
    # Create ScribeSVIResults from AnnData object
    # --------------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: Any,
        params: Dict,
        loss_history: jnp.ndarray,
        model_config: ModelConfig,
        **kwargs,
    ):
        """Create ScribeSVIResults from AnnData object."""
        return cls(
            params=params,
            loss_history=loss_history,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_config=model_config,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs,
        )

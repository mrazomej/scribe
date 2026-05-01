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

    @property
    def gene_coverage(self) -> float | None:
        """Coverage threshold used for pre-fit gene filtering, if any."""
        return getattr(self, "_gene_coverage", None)

    @property
    def gene_coverage_mask(self):
        """Boolean mask over original genes used for coverage filtering."""
        return getattr(self, "_gene_coverage_mask", None)

    @property
    def excluded_gene_names(self):
        """Gene names pooled into the trailing 'other' pseudo-gene."""
        return getattr(self, "_excluded_gene_names", None)

    @property
    def has_gene_coverage_filter(self) -> bool:
        """Whether this result was fit with pre-fit gene coverage filtering."""
        return getattr(self, "_gene_coverage_mask", None) is not None

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

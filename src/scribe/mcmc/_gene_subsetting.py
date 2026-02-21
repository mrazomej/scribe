"""
Gene subsetting mixin for MCMC results.

Provides ``__getitem__`` for gene-level indexing and the underlying
metadata-aware sample subsetting helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp
import numpy as np

from ..svi._gene_subsetting import build_gene_axis_by_key

if TYPE_CHECKING:
    from .results import ScribeMCMCResults


# ==============================================================================
# Gene Subsetting Mixin
# ==============================================================================


class GeneSubsettingMixin:
    """Mixin providing gene-level indexing on MCMC results."""

    # --------------------------------------------------------------------------
    # Public indexing
    # --------------------------------------------------------------------------

    def __getitem__(self, index) -> "ScribeMCMCResults":
        """Index by genes (and optionally components).

        Supports int, slice, boolean mask, integer array, and two-element
        tuple ``(gene_indexer, component_indexer)`` for joint subsetting.

        Parameters
        ----------
        index : int, slice, array-like, or tuple
            Gene selector (or ``(genes, components)`` tuple).

        Returns
        -------
        ScribeMCMCResults
            New results restricted to the selected genes.
        """
        from .results import ScribeMCMCResults

        # Two-axis indexing: (genes, components)
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(
                    "Tuple indexing must be (gene_indexer, component_indexer)."
                )
            gene_indexer, component_indexer = index
            gene_subset = self[gene_indexer]
            return gene_subset.get_component(component_indexer)

        bool_index = _to_bool_index(index, self.n_genes)

        new_var = self.var.iloc[bool_index] if self.var is not None else None
        new_samples = self._subset_posterior_samples(self.samples, bool_index)

        return ScribeMCMCResults(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=int(
                bool_index.sum()
                if hasattr(bool_index, "sum")
                else len(bool_index)
            ),
            model_type=self.model_type,
            model_config=self.model_config,
            prior_params=getattr(self, "prior_params", {}),
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            n_components=self.n_components,
        )

    # --------------------------------------------------------------------------
    # Internal: metadata-aware sample subsetting
    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """Subset a samples dictionary along the gene axis.

        When ``model_config.param_specs`` is available, uses metadata-driven
        axis detection; otherwise falls back to last-axis heuristic.
        """
        if samples is None:
            return None

        new_samples = {}
        original_n_genes = self.n_genes
        gene_axis_by_key: Optional[Dict[str, int]] = None
        if getattr(self.model_config, "param_specs", None):
            gene_axis_by_key = build_gene_axis_by_key(
                self.model_config.param_specs, samples, original_n_genes
            )

        for key, value in samples.items():
            if not hasattr(value, "ndim"):
                new_samples[key] = value
                continue
            if gene_axis_by_key is not None and key in gene_axis_by_key:
                gene_axis = gene_axis_by_key[key]
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_samples[key] = value[tuple(slicer)]
                continue
            if value.ndim > 0 and value.shape[-1] == original_n_genes:
                new_samples[key] = value[..., index]
            else:
                new_samples[key] = value
        return new_samples


# ==============================================================================
# Helpers
# ==============================================================================


def _to_bool_index(index, n_genes: int) -> jnp.ndarray:
    """Normalise an arbitrary gene selector to a boolean mask."""
    if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
        return index
    if isinstance(index, int):
        mask = jnp.zeros(n_genes, dtype=bool)
        return mask.at[index].set(True)
    if isinstance(index, slice):
        indices = jnp.arange(n_genes)[index]
        return jnp.isin(jnp.arange(n_genes), indices)
    # List / integer array
    if not isinstance(index, (bool, jnp.bool_)) and not isinstance(
        index[-1], (bool, jnp.bool_)
    ):
        indices = jnp.array(index)
        return jnp.isin(jnp.arange(n_genes), indices)
    return index

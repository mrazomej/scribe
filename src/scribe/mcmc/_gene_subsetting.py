"""
Gene subsetting mixin for MCMC results.

Provides ``__getitem__`` for gene-level indexing and the underlying
metadata-aware sample subsetting helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp
import numpy as np


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
        """Index by genes, components, and/or datasets.

        Supports int, slice, boolean mask, integer array, and tuples of
        up to three elements:

        * ``(gene_indexer, component_indexer)``
        * ``(gene_indexer, component_indexer, dataset_indexer)``

        Parameters
        ----------
        index : int, slice, array-like, or tuple
            Gene selector (or multi-axis tuple).

        Returns
        -------
        ScribeMCMCResults
            New results restricted to the selected subset.
        """
        from .results import ScribeMCMCResults

        # Multi-axis indexing: (genes, components) or (genes, components, dataset)
        if isinstance(index, tuple):
            if len(index) not in (2, 3):
                raise ValueError(
                    "Tuple indexing must be "
                    "(gene_indexer, component_indexer) or "
                    "(gene_indexer, component_indexer, dataset_indexer)."
                )
            gene_indexer = index[0]
            component_indexer = index[1]
            dataset_indexer = index[2] if len(index) == 3 else None

            gene_subset = self[gene_indexer]

            # Apply component selection (skip if slice(None))
            if isinstance(
                component_indexer, slice
            ) and component_indexer == slice(None):
                result = gene_subset
            else:
                result = gene_subset.get_component(component_indexer)

            # Apply dataset selection if requested
            if dataset_indexer is not None:
                result = result.get_dataset(dataset_indexer)

            return result

        bool_index = _to_bool_index(index, self.n_genes)

        new_var = self.var.iloc[bool_index] if self.var is not None else None
        new_samples = self._subset_posterior_samples(self.samples, bool_index)

        # Gene subsetting keeps the gene axis; layouts are unchanged.
        subset = ScribeMCMCResults(
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
            param_layouts=dict(self.layouts),
        )

        # Carry over per-dataset metadata for downstream get_dataset()
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        if per_ds is not None:
            subset._n_cells_per_dataset = per_ds
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            subset._dataset_indices = ds_idx

        return subset

    # --------------------------------------------------------------------------
    # Internal: metadata-aware sample subsetting
    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """Subset a samples dictionary along the gene axis.

        Notes
        -----
        Per-key gene axes come from :meth:`scribe.mcmc.results.ScribeMCMCResults.layouts`
        via :func:`~scribe.core.axis_layout.gene_axes_from_layouts`.  Keys with
        no annotated gene axis fall back to slicing the trailing dimension when
        its length matches ``n_genes`` (legacy compatibility).
        """
        if samples is None:
            return None

        from ..core.axis_layout import gene_axes_from_layouts

        gene_axis_by_key = gene_axes_from_layouts(self.layouts)
        original_n_genes = self.n_genes

        new_samples = {}
        for key, value in samples.items():
            # Skip non-array entries (e.g. metadata scalars).
            if not hasattr(value, "ndim"):
                new_samples[key] = value
                continue
            # Primary: slice along the layout-derived gene axis.
            if key in gene_axis_by_key:
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis_by_key[key]] = index
                new_samples[key] = value[tuple(slicer)]
            # Fallback: trailing axis matches n_genes (covers keys without
            # a layout, e.g. deterministic quantities or ad-hoc samples).
            elif value.ndim > 0 and value.shape[-1] == original_n_genes:
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

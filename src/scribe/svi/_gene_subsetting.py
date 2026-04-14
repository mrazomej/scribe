"""
Gene subsetting mixin for SVI results.

This mixin provides methods for subsetting results by gene indices, enabling
indexing operations like `results[:, genes]`.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .results import ScribeSVIResults


def _has_flow_params(params: Dict[str, Any]) -> bool:
    """Return True if the param dict contains normalizing-flow weights.

    Checks for Flax module dicts registered under the ``flow_`` or
    ``joint_flow_`` naming convention.
    """
    return any(
        k.endswith("$params")
        and (k.startswith("flow_") or k.startswith("joint_flow_"))
        for k in params
    )


# ==============================================================================
# Gene Subsetting Mixin
# ==============================================================================


class GeneSubsettingMixin:
    """Mixin providing gene-based subsetting methods."""

    # --------------------------------------------------------------------------
    # Indexing by genes
    # --------------------------------------------------------------------------

    @staticmethod
    def _subset_gene_params(params, param_prefixes, index, n_components=None):
        """
        Utility to subset all gene-specific parameters in params dict.
        param_prefixes: list of parameter name prefixes (e.g., ["r_", "mu_",
        "gate_"]) index: boolean or integer index for genes n_components: if not
        None, keep component dimension
        """
        new_params = dict(params)
        for prefix, arg_constraints in param_prefixes:
            if arg_constraints is None:
                continue
            for param_name in arg_constraints:
                key = f"{prefix}{param_name}"
                if key in params:
                    if n_components is not None:
                        new_params[key] = params[key][..., index]
                    else:
                        new_params[key] = params[key][index]
        return new_params

    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index using a dynamic,
        shape-based approach. When _gene_axis_by_key is set (from param_specs),
        subset only gene-indexed keys along the stored axis; otherwise use
        shape-based heuristic (first axis matching n_genes) as fallback.
        """
        new_params = {}
        original_n_genes = self.n_genes
        gene_axis_by_key = getattr(self, "_gene_axis_by_key", None)

        for key, value in params.items():
            # Skip nested dicts (e.g., Flax module params from flax_module)
            # These have keys like "amortizer$params" and contain nested dicts
            if not hasattr(value, "shape"):
                new_params[key] = value
                continue

            if gene_axis_by_key is not None and key in gene_axis_by_key:
                gene_axis = gene_axis_by_key[key]
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_params[key] = value[tuple(slicer)]
                continue

            # Fallback: find the first axis with size original_n_genes
            try:
                gene_axis = value.shape.index(original_n_genes)
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_params[key] = value[tuple(slicer)]
            except ValueError:
                new_params[key] = value
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        When _gene_axis_by_key is set, subset only gene-indexed keys along the
        stored axis; otherwise use last-axis heuristic as fallback.
        """
        if samples is None:
            return None

        new_samples = {}
        original_n_genes = self.n_genes
        gene_axis_by_key = getattr(self, "_gene_axis_by_key", None)

        for key, value in samples.items():
            # Skip nested dicts (e.g., Flax module params from flax_module)
            if not hasattr(value, "ndim"):
                new_samples[key] = value
                continue

            if gene_axis_by_key is not None and key in gene_axis_by_key:
                gene_axis = gene_axis_by_key[key]
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_samples[key] = value[tuple(slicer)]
                continue

            # Fallback: gene dimension is typically last
            if value.ndim > 0 and value.shape[-1] == original_n_genes:
                new_samples[key] = value[..., index]
            else:
                new_samples[key] = value
        return new_samples

    # --------------------------------------------------------------------------

    def _subset_predictive_samples(
        self, samples: jnp.ndarray, index
    ) -> jnp.ndarray:
        """Create a new predictive samples array for the given index."""
        if samples is None:
            return None

        # For predictive samples, subset the gene dimension (last dimension)
        return samples[..., index]

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Enable indexing of ``ScribeSVIResults`` by genes, components,
        and datasets.

        Parameters
        ----------
        index : int, slice, array-like, or tuple
            Gene selector, or a tuple of up to three elements:

            * ``(gene_selector, component_selector)``
            * ``(gene_selector, component_selector, dataset_selector)``

            When a tuple is provided, gene subsetting is applied first,
            then component selection, then dataset selection.  Use
            ``slice(None)`` (i.e. ``:``) to pass through an axis, e.g.
            ``results[:, :, 0]`` selects dataset 0 across all genes and
            components.

        Returns
        -------
        ScribeSVIResults
            Subset result after applying requested indexing operations.
        """
        # Support multi-axis indexing:
        #   results[genes, components]
        #   results[genes, components, dataset]
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

            if isinstance(gene_indexer, tuple):
                raise TypeError(
                    "Nested tuple indexing is not supported for gene selector."
                )
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

        # Normalize the selector to a single ``gene_index`` variable that is
        # passed to all downstream slicer helpers.  Boolean masks select genes
        # in their original list order (unchanged); integer arrays preserve the
        # caller-specified order so that, e.g., results[[2, 0, 1]] returns genes
        # in the order [gene_2, gene_0, gene_1] rather than silently sorting
        # them back to [gene_0, gene_1, gene_2].
        if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
            # Boolean mask: keep as-is; order follows original gene list.
            gene_index = index
        elif isinstance(index, int):
            # Single integer: convert to a length-1 boolean mask to keep the
            # same shape semantics as the boolean-mask path.
            gene_index = jnp.zeros(self.n_genes, dtype=bool)
            gene_index = gene_index.at[index].set(True)
        elif isinstance(index, slice):
            # Slice: extract positions from the slice in slice order (always
            # monotone), then keep as an integer array.
            gene_index = np.asarray(jnp.arange(self.n_genes)[index], dtype=int)
        elif isinstance(index, (list, np.ndarray, jnp.ndarray)):
            # Integer list/array: convert to a plain numpy integer array to
            # preserve the caller-specified ordering.  jnp.isin would produce a
            # boolean mask that silently re-sorts genes back to original-list
            # order, discarding the requested ordering (e.g. DE-significance
            # rank).  All downstream slicer methods accept integer arrays and
            # already perform order-preserving fancy indexing.
            gene_index = np.asarray(index, dtype=int)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        # Create new params dict with subset of parameters
        new_params = self._subset_params(self.params, gene_index)

        # Create new metadata if available.
        # Convert gene_index to numpy before calling iloc: pandas does not
        # recognise JAX arrays as boolean masks and would interpret a bool
        # array as integer indices (False=0, True=1), selecting the wrong rows.
        new_var = (
            self.var.iloc[np.asarray(gene_index)]
            if self.var is not None
            else None
        )

        # Create new posterior samples if available
        new_posterior_samples = (
            self._subset_posterior_samples(self.posterior_samples, gene_index)
            if self.posterior_samples is not None
            else None
        )

        # Create new predictive samples if available
        new_predictive_samples = (
            self._subset_predictive_samples(self.predictive_samples, gene_index)
            if self.predictive_samples is not None
            else None
        )

        # Create new instance with subset data
        return self._create_subset(
            index=gene_index,
            new_params=new_params,
            new_var=new_var,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples,
        )

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_var: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray],
    ) -> "ScribeSVIResults":
        """Create a new instance with a subset of genes.

        Note: When using amortized capture probability, the amortizer computes
        sufficient statistics (e.g., total UMI count) by summing across ALL genes.
        Therefore, we track the original gene count (_original_n_genes) so that
        sampling methods can validate that counts have the correct shape.
        """
        # Track the original gene count for amortizer validation.
        # If this is already a subset, preserve the original; otherwise use current.
        original_n_genes = (
            getattr(self, "_original_n_genes", None) or self.n_genes
        )

        # Compose gene indices when re-subsetting an already-subsetted result
        # so that the final index is always relative to the original gene list.
        prev_gene_idx = getattr(self, "_subset_gene_index", None)
        if prev_gene_idx is not None:
            gene_index_abs = prev_gene_idx[index]
        else:
            gene_index_abs = np.asarray(index)

        subset = type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            # Boolean masks count True entries; integer arrays count elements.
            # Both numpy and jax arrays have .sum(), so dtype must be checked
            # explicitly — summing an integer index array gives the sum of the
            # index values (e.g. 42+7+15+3=67), not the gene count (4).
            n_genes=(
                int(index.sum())
                if (
                    hasattr(index, "dtype")
                    and np.dtype(index.dtype) == np.bool_
                )
                else len(index)
            ),
            model_type=self.model_type,
            model_config=self.model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=self.n_components,
            _original_n_genes=original_n_genes,
            _gene_axis_by_key=getattr(self, "_gene_axis_by_key", None),
            _subset_gene_index=gene_index_abs,
        )

        # Carry over per-dataset metadata for downstream get_dataset()
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        if per_ds is not None:
            subset._n_cells_per_dataset = per_ds
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            subset._dataset_indices = ds_idx

        # Preserve the full-dimension params dict for flow-guided posterior
        # sampling.  When a joint flow guide mixes flow-backed parameters
        # with nondense regression (dense_params), array-valued variational
        # params (e.g. joint_flow_joint_p_loc) are sliced by the gene
        # subsetter, but the flow chain still needs them at full dimension.
        # Re-use the already-stored copy if this is a re-subset.
        orig_params = getattr(self, "_original_params", None)
        if orig_params is None and _has_flow_params(self.params):
            orig_params = self.params
        if orig_params is not None:
            subset._original_params = orig_params

        return subset

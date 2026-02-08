"""
Gene subsetting mixin for SVI results.

This mixin provides methods for subsetting results by gene indices, enabling
indexing operations like `results[:, genes]`.
"""

from typing import Dict, List, Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd

# ==============================================================================
# Gene metadata helper
# ==============================================================================


def build_gene_axis_by_key(
    param_specs: List[Any],
    params: Dict[str, Any],
    n_genes: int,
) -> Optional[Dict[str, int]]:
    """
    Build a mapping from param key to gene axis index from param_specs and
    params.

    Used for deterministic gene subsetting when param_specs are available,
    avoiding ambiguity when multiple axes have the same size (e.g. n_components
    == n_genes).

    Parameters
    ----------
    param_specs : List
        Parameter specifications (e.g. from model_config.param_specs).
        Each spec should have name, is_gene_specific, and shape_dims.
    params : Dict[str, Any]
        Flat params dict (e.g. svi_results.params).
    n_genes : int
        Number of genes (used to validate shape and as fallback for axis
        detection).

    Returns
    -------
    Optional[Dict[str, int]]
        Map from param key to gene axis index, or None if no gene-specific
        params were found. Keys not in this map should not be subset along any
        axis.
    """
    if not param_specs:
        return None

    gene_axis_by_key: Dict[str, int] = {}

    for spec in param_specs:
        is_gene_spec = getattr(spec, "is_gene_specific", False) or (
            "n_genes" in getattr(spec, "shape_dims", ())
        )
        if not is_gene_spec:
            continue

        name = getattr(spec, "name", None)
        if not name:
            continue
        shape_dims = getattr(spec, "shape_dims", ())

        for key in params:
            if "$" in key:
                continue
            if (
                key != name
                and not key.startswith(name + "_")
                and not key.startswith("log_" + name + "_")
            ):
                continue

            value = params[key]
            if not hasattr(value, "shape"):
                continue

            try:
                if (
                    shape_dims
                    and "n_genes" in shape_dims
                    and len(value.shape) == len(shape_dims)
                ):
                    gene_axis = list(shape_dims).index("n_genes")
                    if value.shape[gene_axis] == n_genes:
                        gene_axis_by_key[key] = gene_axis
                else:
                    gene_axis = value.shape.index(n_genes)
                    gene_axis_by_key[key] = gene_axis
            except (ValueError, IndexError):
                continue

    return gene_axis_by_key if gene_axis_by_key else None


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
        Enable indexing of ScribeSVIResults object.
        """
        # If index is a boolean mask, use it directly
        if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
            bool_index = index
        # Handle integer indexing
        elif isinstance(index, int):
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given index
            bool_index = bool_index.at[index].set(True)
        # Handle slice indexing
        elif isinstance(index, slice):
            # Get indices from slice
            indices = jnp.arange(self.n_genes)[index]
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given indices
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        # Handle list/array indexing (by integer indices)
        elif isinstance(index, (list, np.ndarray, jnp.ndarray)) and not (
            isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool
        ):
            indices = jnp.array(index)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

        # Create new params dict with subset of parameters
        new_params = self._subset_params(self.params, bool_index)

        # Create new metadata if available
        new_var = self.var.iloc[bool_index] if self.var is not None else None

        # Create new posterior samples if available
        new_posterior_samples = (
            self._subset_posterior_samples(self.posterior_samples, bool_index)
            if self.posterior_samples is not None
            else None
        )

        # Create new predictive samples if available
        new_predictive_samples = (
            self._subset_predictive_samples(self.predictive_samples, bool_index)
            if self.predictive_samples is not None
            else None
        )

        # Create new instance with subset data
        return self._create_subset(
            index=bool_index,
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

        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, "sum") else len(index)),
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
        )

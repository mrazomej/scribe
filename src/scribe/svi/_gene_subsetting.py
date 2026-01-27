"""
Gene subsetting mixin for SVI results.

This mixin provides methods for subsetting results by gene indices, enabling
indexing operations like `results[:, genes]`.
"""

from typing import Dict, Optional
import jax.numpy as jnp
import numpy as np
import pandas as pd

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
        shape-based approach.
        """
        new_params = {}
        original_n_genes = self.n_genes

        for key, value in params.items():
            # Skip nested dicts (e.g., Flax module params from flax_module)
            # These have keys like "amortizer$params" and contain nested dicts
            if not hasattr(value, "shape"):
                new_params[key] = value
                continue

            # Find the axis that corresponds to the number of genes.
            # This is safer than assuming the position of the gene axis.
            try:
                # Find the first occurrence of an axis with size
                # `original_n_genes`.
                gene_axis = value.shape.index(original_n_genes)
                # Build a slicer tuple to index the correct axis.
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_params[key] = value[tuple(slicer)]
            except ValueError:
                # This parameter is not gene-specific (no axis matches n_genes),
                # so we keep it as is.
                new_params[key] = value
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        """
        if samples is None:
            return None

        new_samples = {}
        # Get the original number of genes before subsetting, which is stored
        # in the instance variable self.n_genes.
        original_n_genes = self.n_genes

        for key, value in samples.items():
            # Skip nested dicts (e.g., Flax module params from flax_module)
            # These have keys like "amortizer$params" and contain nested dicts
            if not hasattr(value, "ndim"):
                new_samples[key] = value
                continue

            # The gene dimension is typically the last one in the posterior
            # sample arrays. We check if the last dimension's size matches the
            # original number of genes.
            if value.ndim > 0 and value.shape[-1] == original_n_genes:
                # This is a gene-specific parameter, so we subset it along the
                # last axis.
                new_samples[key] = value[..., index]
            else:
                # This is not a gene-specific parameter (e.g., global,
                # cell-specific), so we keep it as is.
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
        original_n_genes = getattr(self, "_original_n_genes", None) or self.n_genes

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
        )

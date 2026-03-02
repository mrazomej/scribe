"""
Dataset mixin for MCMC results.

Provides methods for extracting single-dataset views from multi-dataset
hierarchical models, analogous to how ComponentMixin handles mixture
components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp

if TYPE_CHECKING:
    from .results import ScribeMCMCResults


# ==============================================================================
# Dataset Mixin
# ==============================================================================


class DatasetMixin:
    """Mixin providing dataset subsetting for MCMC multi-dataset models."""

    def get_dataset(self, dataset_index: int) -> "ScribeMCMCResults":
        """Extract a single-dataset view from multi-dataset MCMC results.

        Slices all per-dataset MCMC samples along the dataset axis,
        returning a results object that looks like a single-dataset model.

        Parameters
        ----------
        dataset_index : int
            Which dataset to extract (0-indexed).

        Returns
        -------
        ScribeMCMCResults
            New results restricted to the selected dataset.

        Raises
        ------
        ValueError
            If the model has no dataset dimension or the index is out of
            range.
        """
        from .results import ScribeMCMCResults

        n_datasets = getattr(self.model_config, "n_datasets", None)
        if n_datasets is None:
            raise ValueError(
                "get_dataset() requires a multi-dataset model "
                "(model_config.n_datasets must be set)."
            )
        if not (0 <= dataset_index < n_datasets):
            raise ValueError(
                f"dataset_index={dataset_index} out of range for "
                f"n_datasets={n_datasets}."
            )

        new_samples = self._subset_samples_by_dataset(
            self.samples, dataset_index, n_datasets
        )

        new_model_config = self.model_config.model_copy(
            update={"n_datasets": None}
        )

        return ScribeMCMCResults(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=self.model_type,
            model_config=new_model_config,
            prior_params=getattr(self, "prior_params", {}),
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=getattr(self, "n_components", None),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subset_samples_by_dataset(
        self,
        samples: Dict[str, jnp.ndarray],
        dataset_index: int,
        n_datasets: int,
    ) -> Dict[str, jnp.ndarray]:
        """Slice MCMC samples for a single dataset.

        For per-dataset parameters the samples have shape
        ``(n_samples, n_datasets, ...)``.  This slices axis 1 at
        ``dataset_index`` and squeezes the dataset dimension.

        Parameters
        ----------
        samples : Dict[str, jnp.ndarray]
            Raw MCMC sample dictionary.
        dataset_index : int
            Dataset to extract.
        n_datasets : int
            Total number of datasets.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Samples with dataset axis removed for per-dataset keys.
        """
        if samples is None:
            return None

        specs_by_name = {}
        if getattr(self.model_config, "param_specs", None):
            specs_by_name = {s.name: s for s in self.model_config.param_specs}

        new_samples: Dict[str, jnp.ndarray] = {}
        for key, values in samples.items():
            spec = specs_by_name.get(key)
            is_ds = spec is not None and getattr(spec, "is_dataset", False)
            # Per-dataset samples: (n_samples, n_datasets, ...)
            if is_ds and hasattr(values, "ndim") and values.ndim > 1:
                # Find the dataset axis (typically axis 1 after sample axis 0)
                dataset_axis = None
                if values.shape[1] == n_datasets:
                    dataset_axis = 1
                else:
                    candidates = [
                        ax
                        for ax in range(1, values.ndim)
                        if values.shape[ax] == n_datasets
                    ]
                    dataset_axis = candidates[0] if candidates else None

                if dataset_axis is not None:
                    slicer = [slice(None)] * values.ndim
                    slicer[dataset_axis] = dataset_index
                    new_samples[key] = values[tuple(slicer)]
                    continue

            new_samples[key] = values

        return new_samples

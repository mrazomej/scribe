"""
Dataset mixin for SVI results.

Provides methods for extracting single-dataset views from multi-dataset
hierarchical models, analogous to how ComponentMixin handles mixture
components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import jax.numpy as jnp

if TYPE_CHECKING:
    from .results import ScribeSVIResults


# ==============================================================================
# Metadata helpers
# ==============================================================================


def _build_dataset_keys(
    param_specs: list, params: Dict[str, jnp.ndarray], n_datasets: int
) -> Set[str]:
    """Identify variational-parameter keys that carry a dataset axis.

    A key is dataset-specific if its ``ParamSpec`` has ``is_dataset=True``
    *or* its leading dimension equals ``n_datasets`` and its spec (matched
    by longest-name-wins) declares a dataset role.

    Parameters
    ----------
    param_specs : list
        Full parameter specifications from ``model_config.param_specs``.
    params : Dict[str, jnp.ndarray]
        Flat variational-parameter dictionary.
    n_datasets : int
        Number of datasets in the model.

    Returns
    -------
    Set[str]
        Keys in ``params`` that carry a dataset axis.
    """
    if not param_specs:
        return set()

    sorted_specs = sorted(param_specs, key=lambda s: len(s.name), reverse=True)
    dataset_keys: Set[str] = set()

    for key in params:
        if "$" in key:
            continue
        for spec in sorted_specs:
            name = spec.name
            if (
                key == name
                or key.startswith(name + "_")
                or key.startswith("log_" + name + "_")
                or key.startswith("logit_" + name + "_")
            ):
                if getattr(spec, "is_dataset", False):
                    dataset_keys.add(key)
                break

    return dataset_keys


def _infer_dataset_axis(
    value: Any, n_datasets: Optional[int]
) -> Optional[int]:
    """Infer the dataset axis for a posterior or variational tensor.

    Parameters
    ----------
    value : Any
        Tensor-like entry.
    n_datasets : Optional[int]
        Number of datasets.

    Returns
    -------
    Optional[int]
        Axis index matching ``n_datasets``, or ``None`` when no unambiguous
        dataset axis is detected.
    """
    if n_datasets is None or not hasattr(value, "ndim") or value.ndim <= 0:
        return None
    # For variational params (no sample axis): dataset axis is 0
    if value.ndim >= 1 and value.shape[0] == n_datasets:
        return 0
    return None


def _infer_dataset_axis_posterior(
    value: Any, n_datasets: Optional[int]
) -> Optional[int]:
    """Infer the dataset axis in posterior samples (axis 0 = samples).

    For posterior samples shaped ``(n_samples, n_datasets, ...)``,
    the dataset axis is 1.

    Parameters
    ----------
    value : Any
        Posterior sample tensor.
    n_datasets : Optional[int]
        Number of datasets.

    Returns
    -------
    Optional[int]
        Dataset axis index (typically 1), or ``None``.
    """
    if n_datasets is None or not hasattr(value, "ndim") or value.ndim <= 1:
        return None
    # Check axis 1 first (most common for posterior: (n_samples, n_datasets, ...))
    if value.shape[1] == n_datasets:
        return 1
    candidates = [
        ax for ax in range(1, value.ndim) if value.shape[ax] == n_datasets
    ]
    return candidates[0] if candidates else None


# ==============================================================================
# Dataset Mixin
# ==============================================================================


class DatasetMixin:
    """Mixin providing dataset subsetting for multi-dataset models."""

    def get_dataset(self, dataset_index: int) -> "ScribeSVIResults":
        """Extract a single-dataset view from multi-dataset results.

        Slices all per-dataset variational parameters and posterior samples
        along the dataset axis, returning a results object that looks like
        a single-dataset model.

        Parameters
        ----------
        dataset_index : int
            Which dataset to extract (0-indexed).

        Returns
        -------
        ScribeSVIResults
            New results object restricted to the selected dataset.

        Raises
        ------
        ValueError
            If the model has no dataset dimension or the index is out of
            range.
        """
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

        # Subset variational params
        new_params = self._subset_params_by_dataset(dataset_index, n_datasets)

        # Subset posterior samples
        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_by_dataset(
                self.posterior_samples, dataset_index, n_datasets
            )

        new_model_config = self.model_config.model_copy(
            update={"n_datasets": None}
        )

        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=self.model_type,
            model_config=new_model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            posterior_samples=new_posterior_samples,
            predictive_samples=None,
            n_components=getattr(self, "n_components", None),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subset_params_by_dataset(
        self, dataset_index: int, n_datasets: int
    ) -> Dict[str, jnp.ndarray]:
        """Slice variational params for a single dataset.

        Parameters
        ----------
        dataset_index : int
            Dataset to extract.
        n_datasets : int
            Total number of datasets.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Params with dataset axis removed for per-dataset keys.
        """
        dataset_keys = _build_dataset_keys(
            self.model_config.param_specs, self.params, n_datasets
        )
        new_params: Dict[str, jnp.ndarray] = {}
        for key, value in self.params.items():
            if key in dataset_keys and hasattr(value, "ndim"):
                ax = _infer_dataset_axis(value, n_datasets)
                if ax is not None:
                    slicer = [slice(None)] * value.ndim
                    slicer[ax] = dataset_index
                    new_params[key] = value[tuple(slicer)]
                    continue
            new_params[key] = value
        return new_params

    def _subset_posterior_by_dataset(
        self,
        samples: Dict[str, jnp.ndarray],
        dataset_index: int,
        n_datasets: int,
    ) -> Dict[str, jnp.ndarray]:
        """Slice posterior samples for a single dataset.

        Parameters
        ----------
        samples : Dict[str, jnp.ndarray]
            Posterior samples dictionary.
        dataset_index : int
            Dataset to extract.
        n_datasets : int
            Total number of datasets.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Posterior samples with dataset axis removed for per-dataset
            keys.
        """
        if samples is None:
            return None

        specs_by_name = {
            s.name: s for s in (self.model_config.param_specs or [])
        }
        new_samples: Dict[str, jnp.ndarray] = {}

        for key, value in samples.items():
            spec = specs_by_name.get(key)
            is_ds = spec is not None and getattr(spec, "is_dataset", False)
            if is_ds and hasattr(value, "ndim") and value.ndim > 1:
                ax = _infer_dataset_axis_posterior(value, n_datasets)
                if ax is not None:
                    slicer = [slice(None)] * value.ndim
                    slicer[ax] = dataset_index
                    new_samples[key] = value[tuple(slicer)]
                    continue
            new_samples[key] = value

        return new_samples

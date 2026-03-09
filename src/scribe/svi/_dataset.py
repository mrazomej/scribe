"""
Dataset mixin for SVI results.

Provides methods for extracting single-dataset views from multi-dataset
hierarchical models, analogous to how ComponentMixin handles mixture
components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import jax.numpy as jnp

if TYPE_CHECKING:
    from .results import ScribeSVIResults


# ==============================================================================
# Metadata helpers
# ==============================================================================


def _key_matches_spec(key: str, spec) -> bool:
    """Check whether a variational-parameter key belongs to a ParamSpec.

    Matches against both ``spec.name`` and any entries in
    ``spec.alias_names`` (e.g. ``"eta_capture"`` for the biology-informed
    capture spec whose canonical name is ``"phi_capture"``).

    Parameters
    ----------
    key : str
        Variational parameter name (e.g. ``"eta_capture_loc"``).
    spec : ParamSpec
        Specification to test against.

    Returns
    -------
    bool
        True if *key* matches the spec's name or any alias.
    """
    # Collect the canonical name plus any reparameterisation aliases
    names_to_check = [spec.name] + list(
        getattr(spec, "alias_names", [])
    )
    for name in names_to_check:
        if (
            key == name
            or key.startswith(name + "_")
            or key.startswith("log_" + name + "_")
            or key.startswith("logit_" + name + "_")
        ):
            return True
    return False


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
            if _key_matches_spec(key, spec):
                if getattr(spec, "is_dataset", False):
                    dataset_keys.add(key)
                break

    return dataset_keys


def _build_cell_specific_keys(
    param_specs: list, params: Dict[str, jnp.ndarray],
) -> Set[str]:
    """Identify variational-parameter keys that are cell-specific.

    A key is cell-specific if its ``ParamSpec`` has
    ``is_cell_specific=True``.  Uses the same longest-name-wins
    matching as ``_build_dataset_keys``.

    Parameters
    ----------
    param_specs : list
        Full parameter specifications from ``model_config.param_specs``.
    params : Dict[str, jnp.ndarray]
        Flat variational-parameter dictionary.

    Returns
    -------
    Set[str]
        Keys in ``params`` that are cell-specific.
    """
    if not param_specs:
        return set()

    sorted_specs = sorted(param_specs, key=lambda s: len(s.name), reverse=True)
    cell_keys: Set[str] = set()

    for key in params:
        if "$" in key:
            continue
        for spec in sorted_specs:
            if _key_matches_spec(key, spec):
                if getattr(spec, "is_cell_specific", False):
                    cell_keys.add(key)
                break

    return cell_keys


def _match_spec_for_key(key: str, param_specs: list) -> Optional[object]:
    """Find the ParamSpec that governs a variational-parameter key.

    Uses longest-name-wins matching to handle derived keys like
    ``log_r_loc``, ``logit_p_scale``, etc.

    Parameters
    ----------
    key : str
        Variational parameter name.
    param_specs : list
        Full parameter specifications.

    Returns
    -------
    Optional[ParamSpec]
        Matched spec, or ``None``.
    """
    if not param_specs:
        return None
    sorted_specs = sorted(param_specs, key=lambda s: len(s.name), reverse=True)
    for spec in sorted_specs:
        if _key_matches_spec(key, spec):
            return spec
    return None


def _infer_dataset_axis(
    value: Any,
    n_datasets: Optional[int],
    is_mixture: bool = False,
    n_components: Optional[int] = None,
) -> Optional[int]:
    """Infer the dataset axis for a variational-parameter tensor.

    For variational parameters (no sample axis):
    - Non-mixture dataset params have shape ``(D, ...)``, dataset axis 0.
    - Mixture+dataset params have shape ``(K, D, ...)``, dataset axis 1.

    When ``is_mixture`` metadata is available, the axis is determined
    deterministically.  Otherwise, falls back to shape heuristics.

    Parameters
    ----------
    value : Any
        Tensor-like entry.
    n_datasets : Optional[int]
        Number of datasets.
    is_mixture : bool
        Whether this parameter also carries a component dimension.
    n_components : Optional[int]
        Number of mixture components (used for disambiguation).

    Returns
    -------
    Optional[int]
        Axis index for the dataset dimension, or ``None``.
    """
    if n_datasets is None or not hasattr(value, "ndim") or value.ndim <= 0:
        return None

    if is_mixture and n_components is not None:
        # Shape (K, D, ...) — dataset axis is 1
        if value.ndim >= 2 and value.shape[1] == n_datasets:
            return 1
        # Fallback: search remaining axes
        candidates = [
            ax for ax in range(value.ndim) if value.shape[ax] == n_datasets
        ]
        return candidates[0] if candidates else None

    # Non-mixture: dataset axis is 0
    if value.ndim >= 1 and value.shape[0] == n_datasets:
        return 0
    return None


def _infer_dataset_axis_posterior(
    value: Any,
    n_datasets: Optional[int],
    is_mixture: bool = False,
    n_components: Optional[int] = None,
) -> Optional[int]:
    """Infer the dataset axis in posterior samples (axis 0 = samples).

    For posterior samples:
    - Non-mixture dataset params: ``(S, D, ...)``, dataset axis 1.
    - Mixture+dataset params: ``(S, K, D, ...)``, dataset axis 2.

    Parameters
    ----------
    value : Any
        Posterior sample tensor.
    n_datasets : Optional[int]
        Number of datasets.
    is_mixture : bool
        Whether this parameter also carries a component dimension.
    n_components : Optional[int]
        Number of mixture components (used for disambiguation).

    Returns
    -------
    Optional[int]
        Dataset axis index, or ``None``.
    """
    if n_datasets is None or not hasattr(value, "ndim") or value.ndim <= 1:
        return None

    if is_mixture and n_components is not None:
        # Shape (S, K, D, ...) — dataset axis is 2
        if value.ndim >= 3 and value.shape[2] == n_datasets:
            return 2
        # Fallback: search axes 1..ndim-1 (skip sample axis 0)
        candidates = [
            ax for ax in range(1, value.ndim) if value.shape[ax] == n_datasets
        ]
        return candidates[0] if candidates else None

    # Non-mixture: (S, D, ...) — dataset axis is 1
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

        # Subset per-dataset variational params (is_dataset=True)
        new_params = self._subset_params_by_dataset(dataset_index, n_datasets)

        # Subset cell-specific variational params using the cell mask
        # derived from _dataset_indices (e.g. phi_capture_loc/scale).
        ds_indices = getattr(self, "_dataset_indices", None)
        if ds_indices is not None:
            new_params = self._subset_cell_specific_params(
                new_params, ds_indices, dataset_index
            )

        # Subset posterior samples
        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_by_dataset(
                self.posterior_samples, dataset_index, n_datasets
            )

        new_model_config = self.model_config.model_copy(
            update={"n_datasets": None}
        )

        # Resolve per-dataset cell count if available
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        ds_n_cells = (
            int(per_ds[dataset_index])
            if per_ds is not None
            else self.n_cells
        )

        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=ds_n_cells,
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

        Uses ``ParamSpec`` metadata to correctly handle parameters that
        are both mixture- and dataset-specific (axis 1 instead of 0).

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
        # Keys promoted during concat (stacked at axis 0 for params).
        promoted = getattr(self, "_promoted_dataset_keys", None) or set()
        n_components = getattr(self.model_config, "n_components", None)

        new_params: Dict[str, jnp.ndarray] = {}
        for key, value in self.params.items():
            # Promoted keys always have the dataset axis at position 0.
            if key in promoted and hasattr(value, "ndim") and value.ndim >= 1:
                new_params[key] = value[dataset_index]
                continue
            if key in dataset_keys and hasattr(value, "ndim"):
                spec = _match_spec_for_key(
                    key, self.model_config.param_specs or []
                )
                is_mix = spec is not None and getattr(
                    spec, "is_mixture", False
                )
                ax = _infer_dataset_axis(
                    value, n_datasets,
                    is_mixture=is_mix, n_components=n_components,
                )
                if ax is not None:
                    slicer = [slice(None)] * value.ndim
                    slicer[ax] = dataset_index
                    new_params[key] = value[tuple(slicer)]
                    continue
            new_params[key] = value
        return new_params

    def _subset_cell_specific_params(
        self,
        params: Dict[str, jnp.ndarray],
        dataset_indices: jnp.ndarray,
        dataset_index: int,
    ) -> Dict[str, jnp.ndarray]:
        """Subset cell-specific variational params for a single dataset.

        Cell-specific params (e.g. ``phi_capture_loc``) have shape
        ``(n_cells, ...)`` where ``n_cells`` is the *total* cell count
        across all datasets.  This method applies a boolean mask derived
        from ``dataset_indices`` to keep only the cells belonging to
        ``dataset_index``.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            Variational parameter dict (already dataset-subsetted).
        dataset_indices : jnp.ndarray
            Per-cell dataset assignment, shape ``(n_cells,)``.
        dataset_index : int
            Which dataset to keep.

        Returns
        -------
        Dict[str, jnp.ndarray]
            Params with cell-specific entries sliced to only the cells
            belonging to ``dataset_index``.
        """
        cell_keys = _build_cell_specific_keys(
            self.model_config.param_specs or [], params
        )
        if not cell_keys:
            return params

        mask = dataset_indices == dataset_index
        new_params: Dict[str, jnp.ndarray] = {}
        for key, value in params.items():
            if key in cell_keys and hasattr(value, "ndim") and value.ndim >= 1:
                new_params[key] = value[mask]
            else:
                new_params[key] = value
        return new_params

    def _subset_posterior_by_dataset(
        self,
        samples: Dict[str, jnp.ndarray],
        dataset_index: int,
        n_datasets: int,
    ) -> Dict[str, jnp.ndarray]:
        """Slice posterior samples for a single dataset.

        Uses ``ParamSpec`` metadata to correctly handle parameters that
        are both mixture- and dataset-specific (axis 2 instead of 1 for
        posterior samples with shape ``(S, K, D, ...)``).

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
        # Promoted keys have the dataset axis at position 1 (after sample axis).
        promoted = getattr(self, "_promoted_dataset_keys", None) or set()
        n_components = getattr(self.model_config, "n_components", None)
        # Cell-specific posterior tensors have shape (S, n_cells, ...) and
        # must be masked on axis 1 using the global dataset assignment.
        cell_keys = _build_cell_specific_keys(
            self.model_config.param_specs or [], samples
        )
        ds_indices = getattr(self, "_dataset_indices", None)
        cell_mask = (
            (ds_indices == dataset_index) if ds_indices is not None else None
        )
        new_samples: Dict[str, jnp.ndarray] = {}

        for key, value in samples.items():
            if (
                key in cell_keys
                and cell_mask is not None
                and hasattr(value, "ndim")
                and value.ndim >= 2
            ):
                new_samples[key] = value[:, cell_mask]
                continue
            if key in promoted and hasattr(value, "ndim") and value.ndim >= 2:
                new_samples[key] = value[:, dataset_index]
                continue
            spec = specs_by_name.get(key)
            is_ds = spec is not None and getattr(spec, "is_dataset", False)
            is_mix = spec is not None and getattr(spec, "is_mixture", False)
            if is_ds and hasattr(value, "ndim") and value.ndim > 1:
                ax = _infer_dataset_axis_posterior(
                    value, n_datasets,
                    is_mixture=is_mix, n_components=n_components,
                )
                if ax is not None:
                    slicer = [slice(None)] * value.ndim
                    slicer[ax] = dataset_index
                    new_samples[key] = value[tuple(slicer)]
                    continue
            new_samples[key] = value

        return new_samples

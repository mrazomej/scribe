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
        name = spec.name
        if (
            key == name
            or key.startswith(name + "_")
            or key.startswith("log_" + name + "_")
            or key.startswith("logit_" + name + "_")
        ):
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
        n_components = getattr(self.model_config, "n_components", None)

        new_params: Dict[str, jnp.ndarray] = {}
        for key, value in self.params.items():
            if key in dataset_keys and hasattr(value, "ndim"):
                # Look up spec to determine if this param is also mixture
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
        n_components = getattr(self.model_config, "n_components", None)
        new_samples: Dict[str, jnp.ndarray] = {}

        for key, value in samples.items():
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

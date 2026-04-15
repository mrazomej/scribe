"""
Dataset mixin for SVI results.

Provides methods for extracting single-dataset views from multi-dataset
hierarchical models, analogous to how ComponentMixin handles mixture
components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import jax.numpy as jnp

from ..core.axis_layout import DATASETS, subset_layouts

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

    Also matches joint guide keys like ``"joint_joint_phi_loc"`` for a
    spec with ``name="phi"``.

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
    names_to_check = [spec.name] + list(getattr(spec, "alias_names", []))
    if spec.name == "mixing_weights":
        # Dirichlet mixture weights use specialized variational parameter names.
        names_to_check.extend(
            ["mixing_concentrations", "mixing_logits_unconstrained"]
        )
    # Also include the NCP raw name if present (horseshoe specs)
    raw_name = getattr(spec, "raw_name", None)
    if raw_name:
        names_to_check.append(raw_name)
    _JOINT_SUFFIXES = ("_loc", "_W", "_raw_diag", "_scale")
    for name in names_to_check:
        if (
            key == name
            or key.startswith(name + "_")
            or key.startswith("log_" + name + "_")
            or key.startswith("logit_" + name + "_")
        ):
            return True
        # Joint guide keys: "joint_{group}_{name}_loc", etc.
        # Use endswith to avoid false positives (e.g. "phi" matching
        # "phi_capture" via substring).
        if key.startswith("joint_"):
            for suf in _JOINT_SUFFIXES:
                if key.endswith(f"_{name}{suf}"):
                    return True
    return False


def _build_cell_specific_keys(
    param_specs: list,
    params: Dict[str, jnp.ndarray],
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

        # Snapshot parent layouts before modification so we can compute
        # adjusted gene-axis indices below.
        parent_layouts = self.layouts

        # Drop the dataset axis from every layout that has it so the
        # child results object carries correct semantic metadata.
        new_layouts = subset_layouts(parent_layouts, DATASETS)

        # Adjust _gene_axis_by_key: when a dataset axis is removed from
        # a param, any gene axis after it shifts down by one.
        parent_gene_axes = getattr(self, "_gene_axis_by_key", None)
        new_gene_axes = None
        if parent_gene_axes:
            new_gene_axes = {}
            for key, gene_ax in parent_gene_axes.items():
                if key not in new_params:
                    continue
                layout = parent_layouts.get(key)
                ds_ax = (
                    layout.dataset_axis
                    if layout is not None
                    else None
                )
                if ds_ax is not None and gene_ax > ds_ax:
                    new_gene_axes[key] = gene_ax - 1
                else:
                    new_gene_axes[key] = gene_ax

        # Resolve per-dataset cell count if available
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        ds_n_cells = (
            int(per_ds[dataset_index]) if per_ds is not None else self.n_cells
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
            param_layouts=new_layouts,
            _gene_axis_by_key=new_gene_axes,
            # Preserve fit-time annotation/component metadata so per-dataset
            # plotting and downstream component lookups stay index-consistent.
            _label_map=getattr(self, "_label_map", None),
            _component_mapping=getattr(self, "_component_mapping", None),
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
        # ``promoted`` keys from ``concat()`` use a fixed axis-0 dataset slice;
        # all other dataset-specific tensors use ``AxisLayout.dataset_axis``.
        promoted = getattr(self, "_promoted_dataset_keys", None) or set()
        param_layouts = self.layouts

        new_params: Dict[str, jnp.ndarray] = {}
        for key, value in self.params.items():
            # Promoted keys (stacked by concat) always have dataset at axis 0.
            if key in promoted and hasattr(value, "ndim") and value.ndim >= 1:
                new_params[key] = value[dataset_index]
            # Layout tells us the exact dataset axis position (handles the
            # mixture+dataset case where the axis is 1 instead of 0).
            elif (
                param_layouts.get(key) is not None
                and param_layouts[key].dataset_axis is not None
            ):
                slicer = [slice(None)] * value.ndim
                slicer[param_layouts[key].dataset_axis] = dataset_index
                new_params[key] = value[tuple(slicer)]
            # Non-dataset keys pass through unchanged.
            else:
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

        # Same promotion rule as params: axis 1 after the leading draw dimension
        # ``S`` for keys stacked during multi-result concat.
        promoted = getattr(self, "_promoted_dataset_keys", None) or set()

        # Hybrid layouts so derived posterior keys (not in specs) still get a
        # correct ``dataset_axis`` when shapes and config metadata allow it.
        # We resolve dataset_params via derive_axis_membership so that
        # derived keys (e.g. "mu" in canonical mode) inherit dataset
        # membership from their source parameters ("r", "p").
        from ..core.axis_layout import (
            build_sample_layouts,
            derive_axis_membership,
        )

        mc = self.model_config
        _mp, _dp = derive_axis_membership(
            mc,
            samples=samples,
            has_sample_dim=True,
        )
        sample_layouts = build_sample_layouts(
            list(mc.param_specs or []),
            samples,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=getattr(mc, "n_components", None),
            n_datasets=getattr(mc, "n_datasets", None),
            mixture_params=_mp,
            dataset_params=_dp,
            has_sample_dim=True,
        )

        new_samples: Dict[str, jnp.ndarray] = {}
        for key, value in samples.items():
            # If the key is in the set of promoted dataset keys, and its value
            # is at least 2-dimensional, slice at axis=1 (the dataset axis for
            # promoted keys)
            if key in promoted and hasattr(value, "ndim") and value.ndim >= 2:
                new_samples[key] = value[:, dataset_index]
            # Layout-driven slicing: handles (S, D, G) and (S, K, D, G) shapes
            # transparently since the layout already encodes the correct axis.
            elif (
                sample_layouts.get(key) is not None
                and sample_layouts[key].dataset_axis is not None
            ):
                slicer = [slice(None)] * value.ndim
                slicer[sample_layouts[key].dataset_axis] = dataset_index
                new_samples[key] = value[tuple(slicer)]
            # Non-dataset keys (population-level hyperparams, etc.) pass through.
            else:
                new_samples[key] = value

        return new_samples

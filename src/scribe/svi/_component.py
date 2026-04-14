"""
Component mixin for SVI results.

This mixin provides methods for working with mixture model components, including
single-component extraction and multi-component selection with optional
renormalization of component fractions.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import jax.numpy as jnp

from ..core.component_indexing import (
    normalize_component_indices,
    renormalize_mixing_logits,
    renormalize_mixing_weights,
)

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout
    from .results import ScribeSVIResults


# ==============================================================================
# Metadata-driven helpers
# ==============================================================================


def _build_mixture_keys(
    param_specs: List, params: Dict[str, jnp.ndarray]
) -> Set[str]:
    """
    Identify variational-parameter keys that belong to mixture-specific specs.

    Each ``ParamSpec`` with ``is_mixture=True`` owns variational parameter keys
    whose names start with ``{spec.name}_`` or ``log_{spec.name}_``. When
    multiple specs could match the same key (e.g. ``"phi"`` vs
    ``"phi_capture"``), the longest spec name wins.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Full parameter specifications from ``model_config.param_specs``.
    params : Dict[str, jnp.ndarray]
        Flat variational-parameter dictionary (``self.params``).

    Returns
    -------
    Set[str]
        Keys in ``params`` that carry a component axis.
    """
    if not param_specs:
        return set()

    sorted_specs = sorted(param_specs, key=lambda s: len(s.name), reverse=True)
    mixture_keys: Set[str] = set()

    for key in params:
        if "$" in key:
            continue
        for spec in sorted_specs:
            name = spec.name
            # Match both classic per-parameter guide keys
            # (e.g. ``phi_loc``, ``log_phi_loc``) and joint-guide keys
            # (e.g. ``joint_joint_phi_loc``).  The sorted longest-name
            # policy avoids substring ambiguity such as ``phi`` vs
            # ``phi_capture``.
            matches_spec = (
                key == name
                or key.startswith(name + "_")
                or key.startswith("log_" + name + "_")
                or f"_{name}_" in key
                or key.endswith("_" + name)
            )
            if matches_spec:
                if spec.is_mixture:
                    mixture_keys.add(key)
                break

    return mixture_keys


def _has_fallback_mixture_evidence(
    key: str, component_axis: int, explicit_component_axes: Dict[str, int]
) -> bool:
    """Check whether a canonical key should be treated as mixture-specific.

    Parameters
    ----------
    key : str
        Posterior-sample key currently being inspected.
    component_axis : int
        Candidate component axis for ``key``.
    explicit_component_axes : Dict[str, int]
        Axis map for keys explicitly marked as mixture-specific via param specs
        (plus named mixing carriers).

    Returns
    -------
    bool
        True when there is supporting evidence from a non-canonical/reference
        mixture key with a matching component axis.

    Notes
    -----
    This is a compatibility bridge for canonical/derived posterior keys that
    may be present in samples but absent from ``param_specs``.  A future
    architecture could replace this with an explicit derived-lineage contract
    (for example, persisted ``DerivedParam`` metadata) so subsetting does not
    need fallback evidence heuristics.
    """
    # Map each derived key to the spec-level keys whose component axis
    # can serve as evidence.  The tuples include both the key itself
    # (in case it has a spec) and its parameterization-level parents:
    #   - canonical: mu derived from r, p
    #   - linked/mean_prob: r derived from p, mu
    #   - mean_odds: r derived from phi, mu; p derived from phi
    reference_map = {
        "p": ("p", "phi"),
        "r": ("r", "mu", "phi"),
        "mu": ("mu", "r", "p"),
        "phi": ("phi",),
        "gate": ("gate",),
    }
    for ref_key in reference_map.get(key, ()):
        ref_axis = explicit_component_axes.get(ref_key)
        if ref_axis is not None and ref_axis == component_axis:
            return True
    return False


# ==============================================================================
# Component Mixin
# ==============================================================================


# Posterior keys that unambiguously carry mixture structure (guide params or
# their logits).
_MIXING_KEY_NAMES = frozenset({"mixing_weights", "mixing_logits_unconstrained"})
# Canonical derived tensors that may be missing from ``param_specs``; they are
# only subset when :func:`_has_fallback_mixture_evidence` agrees with pass 1.
_FALLBACK_MIXTURE_KEYS = frozenset({"p", "r", "mu", "phi", "gate"})


def _resolve_component_axes(
    mixin: "ComponentMixin",
    samples: Dict,
) -> Dict[str, int]:
    """Identify which posterior-sample keys carry a component axis.

    Returns a ``{key: axis_index}`` mapping.  Keys not in the returned
    dict should not be subset along the component dimension.

    Two-pass resolution:

    1. Collect component axes for keys explicitly marked as mixture
       (via ``ParamSpec`` or by being a named mixing-weight carrier).
    2. Accept derived/fallback keys (``p``, ``r``, ``mu``, ``phi``,
       ``gate``) only when their layout-inferred component axis is
       corroborated by the explicit axes (evidence check).
    """
    from ..core.axis_layout import build_sample_layouts, derive_axis_membership

    mc = getattr(mixin, "model_config", None)
    specs = list(getattr(mc, "param_specs", None) or []) if mc else []
    specs_by_name = {s.name: s for s in specs}

    # Resolve mixture_params and dataset_params via derive_axis_membership
    # so that derived keys (e.g. "mu" in canonical) inherit axis membership
    # from their source parameters through the DerivedParam graph.
    # When the config's mixture_params is None (= "all non-cell params are
    # mixture"), we preserve None to avoid breaking keys absent from specs
    # (e.g. p, r as derived keys in linked mode). When the config has an
    # explicit list (e.g. ['r', 'p']), we use the expanded version so
    # derived keys like 'mu' are included.
    _mp_cfg = getattr(mc, "mixture_params", None) if mc else None
    _mp_expanded, _dp = (
        derive_axis_membership(mc, samples=samples, has_sample_dim=True)
        if mc is not None
        else (None, None)
    )
    # Use expanded mixture_params only when config had an explicit list;
    # keep None to preserve "all" semantics when config was None.
    _mp_final = _mp_expanded if _mp_cfg is not None else None

    # Build semantic layouts for every key in the posterior samples.
    # Spec-matched keys get deterministic layouts; derived keys (p, r, ...)
    # fall back to shape-based inference via infer_layout.
    _n_genes = getattr(mixin, "n_genes", None)
    _n_cells = getattr(mixin, "n_cells", None)
    _n_comp = getattr(mixin, "n_components", None)
    _n_ds = getattr(mc, "n_datasets", None) if mc else None

    sample_layouts = build_sample_layouts(
        specs,
        samples,
        n_genes=_n_genes,
        n_cells=_n_cells,
        n_components=_n_comp,
        n_datasets=_n_ds,
        mixture_params=_mp_final,
        dataset_params=_dp,
        has_sample_dim=True,
    )

    # --- Pass 1: collect component axes for keys we *know* are mixture ---
    # These come from specs (is_mixture=True) or by being a mixing-weight
    # carrier (mixing_weights / mixing_logits_unconstrained).
    explicit: Dict[str, int] = {}
    for key in samples:
        spec = specs_by_name.get(key)
        if (spec is not None and spec.is_mixture) or key in _MIXING_KEY_NAMES:
            layout = sample_layouts.get(key)
            if layout is not None and layout.component_axis is not None:
                explicit[key] = layout.component_axis

    # --- Pass 2: accept derived keys only with corroborating evidence ---
    # Canonical keys (p, r, mu, phi, gate) may not appear in param_specs
    # but still carry a component axis in mixture models.  We only include
    # them when their inferred axis agrees with a known explicit key.
    result = dict(explicit)
    for key in samples:
        if key in result or key not in _FALLBACK_MIXTURE_KEYS:
            continue
        # Already handled in pass 1 if it has is_mixture.
        spec = specs_by_name.get(key)
        if spec is not None and spec.is_mixture:
            continue
        layout = sample_layouts.get(key)
        comp_ax = layout.component_axis if layout is not None else None
        if comp_ax is not None and _has_fallback_mixture_evidence(
            key, comp_ax, explicit
        ):
            result[key] = comp_ax

    return result


class ComponentMixin:
    """Mixin providing component/mixture model operations."""

    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(self, component_index: Any, renormalize: bool = True):
        """
        Create a component-restricted view of mixture results.

        This API remains backward compatible for single integer component
        selection and also supports multi-component selectors.

        Parameters
        ----------
        component_index : Any
            Component selector. Supported values:

            - ``int``
            - ``slice``
            - list/array of integer indices
            - boolean mask with length ``n_components``
        renormalize : bool, default=True
            Whether to renormalize mixture-fraction carriers for multi-component
            selections.

        Returns
        -------
        ScribeSVIResults
            New result object containing the selected components.
        """
        selected = normalize_component_indices(
            component_index, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._get_single_component(int(selected[0]))
        return self.get_components(selected, renormalize=renormalize)

    # --------------------------------------------------------------------------

    def get_components(self, component_indices: Any, renormalize: bool = True):
        """
        Select multiple mixture components while preserving mixture semantics.

        Parameters
        ----------
        component_indices : Any
            Component selector (same accepted formats as
            :meth:`get_component`).
        renormalize : bool, default=True
            Whether to renormalize ``mixing_weights`` and recenter
            ``mixing_logits_unconstrained`` after subsetting.

        Returns
        -------
        ScribeSVIResults
            New result with reduced ``n_components``.
        """
        selected = normalize_component_indices(
            component_indices, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._get_single_component(int(selected[0]))

        new_params = dict(self.params)
        self._subset_params_by_components(
            new_params, selected, renormalize=renormalize
        )

        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = (
                self._subset_posterior_samples_by_components(
                    self.posterior_samples,
                    selected,
                    renormalize=renormalize,
                )
            )

        new_predictive_samples = None
        return self._create_multi_component_subset(
            component_indices=selected,
            new_params=new_params,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples,
        )

    # --------------------------------------------------------------------------

    def _get_single_component(self, component_index: int):
        """
        Return a single-component subset using legacy non-mixture behavior.

        Parameters
        ----------
        component_index : int
            Selected component index.

        Returns
        -------
        ScribeSVIResults
            Non-mixture result corresponding to the selected component.
        """
        new_params = dict(self.params)
        self._subset_params_by_component(new_params, component_index)

        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_samples_by_component(
                self.posterior_samples, component_index
            )

        new_predictive_samples = None
        return self._create_component_subset(
            component_index=component_index,
            new_params=new_params,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples,
        )

    # --------------------------------------------------------------------------

    def _subset_params_by_component(
        self, new_params: Dict, component_index: int
    ):
        """
        Subset variational parameters to a single mixture component.

        Parameters
        ----------
        new_params : Dict
            Mutable params dictionary to update in-place.
        component_index : int
            Which mixture component to extract.
        """
        mixture_keys = _build_mixture_keys(
            self.model_config.param_specs, self.params
        )
        n_comp = self.n_components

        for key, value in self.params.items():
            if not hasattr(value, "ndim"):
                new_params[key] = value
            elif (
                key in mixture_keys
                and value.ndim > 0
                and value.shape[0] == n_comp
            ):
                new_params[key] = value[component_index]
            else:
                new_params[key] = value

    # --------------------------------------------------------------------------

    def _subset_params_by_components(
        self,
        new_params: Dict,
        component_indices: jnp.ndarray,
        renormalize: bool = True,
    ):
        """
        Subset variational parameters to multiple mixture components.

        Parameters
        ----------
        new_params : Dict
            Mutable params dictionary to update in-place.
        component_indices : jnp.ndarray
            One-dimensional integer array of selected components.
        renormalize : bool, default=True
            Whether to renormalize selected mixing fractions.
        """
        mixture_keys = _build_mixture_keys(
            self.model_config.param_specs, self.params
        )
        n_comp = self.n_components

        for key, value in self.params.items():
            if (
                hasattr(value, "ndim")
                and key in mixture_keys
                and value.ndim > 0
                and value.shape[0] == n_comp
            ):
                selected_value = value[component_indices]
                if renormalize and key == "mixing_weights":
                    selected_value = renormalize_mixing_weights(
                        selected_value, axis=-1
                    )
                elif renormalize and key == "mixing_logits_unconstrained":
                    selected_value = renormalize_mixing_logits(
                        selected_value, axis=-1
                    )
                new_params[key] = selected_value
            else:
                new_params[key] = value

    # --------------------------------------------------------------------------

    def _subset_posterior_samples_by_component(
        self, samples: Dict, component_index: int
    ) -> Dict:
        """Subset posterior samples to a single mixture component.

        Component axes come from :func:`_resolve_component_axes` (layouts +
        evidence checks), not raw shape heuristics alone.

        Parameters
        ----------
        samples : Dict
            Posterior samples dictionary.
        component_index : int
            Selected component index.

        Returns
        -------
        Dict
            Posterior samples with component axis removed for mixture keys.
        """
        if samples is None:
            return None

        # Build {key: axis} map using layouts + fallback evidence check.
        component_axes = _resolve_component_axes(self, samples)

        new_posterior_samples: Dict = {}
        for key, value in samples.items():
            # Only slice keys that _resolve_component_axes identified as
            # having a component dimension (and that are multi-dimensional).
            if (
                key in component_axes
                and hasattr(value, "ndim")
                and value.ndim > 1
            ):
                # Index into the component axis to extract the single component.
                slicer = [slice(None)] * value.ndim
                slicer[component_axes[key]] = component_index
                new_posterior_samples[key] = value[tuple(slicer)]
            else:
                # Shared (non-mixture) keys are kept as-is.
                new_posterior_samples[key] = value
        return new_posterior_samples

    # --------------------------------------------------------------------------

    def _subset_posterior_samples_by_components(
        self,
        samples: Dict,
        component_indices: jnp.ndarray,
        renormalize: bool = True,
    ) -> Dict:
        """Subset posterior samples to multiple mixture components.

        Uses the same axis map as :meth:`_subset_posterior_samples_by_component`.

        Parameters
        ----------
        samples : Dict
            Posterior samples dictionary.
        component_indices : jnp.ndarray
            One-dimensional integer array of selected components.
        renormalize : bool, default=True
            Whether to renormalize selected mixing fraction carriers.

        Returns
        -------
        Dict
            Posterior samples with restricted component axis.
        """
        if samples is None:
            return None

        # Build {key: axis} map using layouts + fallback evidence check.
        component_axes = _resolve_component_axes(self, samples)

        new_posterior_samples: Dict = {}
        for key, value in samples.items():
            # Slice along component axis for identified mixture keys.
            if (
                key in component_axes
                and hasattr(value, "ndim")
                and value.ndim > 1
            ):
                slicer = [slice(None)] * value.ndim
                slicer[component_axes[key]] = component_indices
                selected_value = value[tuple(slicer)]
                # Renormalize mixing fractions so they sum to 1 over the
                # subset of retained components.
                if renormalize and key == "mixing_weights":
                    selected_value = renormalize_mixing_weights(
                        selected_value, axis=-1
                    )
                elif renormalize and key == "mixing_logits_unconstrained":
                    selected_value = renormalize_mixing_logits(
                        selected_value, axis=-1
                    )
                new_posterior_samples[key] = selected_value
            else:
                new_posterior_samples[key] = value
        return new_posterior_samples

    # --------------------------------------------------------------------------

    def _create_component_subset(
        self,
        component_index,
        new_params: Dict,
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray],
    ) -> "ScribeSVIResults":
        """
        Create a new instance for a single selected component.

        Parameters
        ----------
        component_index : int
            Selected component index.
        new_params : Dict
            Subset variational parameters.
        new_posterior_samples : Optional[Dict]
            Subset posterior samples.
        new_predictive_samples : Optional[jnp.ndarray]
            Subset predictive samples.

        Returns
        -------
        ScribeSVIResults
            Non-mixture result for the selected component.
        """
        base_model = self.model_type.replace("_mix", "")
        new_model_config = self.model_config.model_copy(
            update={
                "base_model": base_model,
                "n_components": None,
            }
        )

        subset = type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=base_model,
            model_config=new_model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=None,
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

    def _create_multi_component_subset(
        self,
        component_indices: jnp.ndarray,
        new_params: Dict,
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray],
    ) -> "ScribeSVIResults":
        """
        Create a new instance for multiple selected mixture components.

        Parameters
        ----------
        component_indices : jnp.ndarray
            One-dimensional integer array of selected components.
        new_params : Dict
            Subset variational parameters.
        new_posterior_samples : Optional[Dict]
            Subset posterior samples.
        new_predictive_samples : Optional[jnp.ndarray]
            Subset predictive samples.

        Returns
        -------
        ScribeSVIResults
            Mixture result with reduced ``n_components``.
        """
        new_n_components = int(component_indices.shape[0])
        new_model_config = self.model_config.model_copy(
            update={"n_components": new_n_components}
        )

        subset = type(self)(
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
            predictive_samples=new_predictive_samples,
            n_components=new_n_components,
            _original_n_genes=getattr(self, "_original_n_genes", None),
            _gene_axis_by_key=getattr(self, "_gene_axis_by_key", None),
        )

        # Carry over per-dataset metadata for downstream get_dataset()
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        if per_ds is not None:
            subset._n_cells_per_dataset = per_ds
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            subset._dataset_indices = ds_idx

        return subset

"""
Component mixin for MCMC results.

Provides mixture-component selection with optional renormalization of
mixing fractions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp

from ..core.axis_layout import COMPONENTS, subset_layouts
from ..core.component_indexing import (
    normalize_component_indices,
    renormalize_mixing_logits,
    renormalize_mixing_weights,
)

if TYPE_CHECKING:
    from .results import ScribeMCMCResults


# ==============================================================================
# Metadata-driven helpers
# ==============================================================================

# Posterior keys that unambiguously carry mixture structure.
_MIXING_KEY_NAMES = frozenset({"mixing_weights", "mixing_logits_unconstrained"})

# Canonical/derived keys that may be absent from ParamSpec names. We keep an
# explicit allowlist here and require corroborating mixture evidence so that
# shared scalar keys are not accidentally treated as component-specific.
_FALLBACK_MIXTURE_KEYS = frozenset(
    {
        "p",
        "r",
        "mu",
        "phi",
        "gate",
        "p_unconstrained",
        "r_unconstrained",
        "mu_unconstrained",
        "phi_unconstrained",
        "gate_unconstrained",
    }
)


def _has_fallback_mixture_evidence(
    key: str,
    component_axis: int,
    explicit_component_axes: Dict[str, int],
) -> bool:
    """Check whether a fallback key should be treated as mixture-specific.

    Parameters
    ----------
    key : str
        Posterior-sample key being inspected.
    component_axis : int
        Candidate component axis inferred for ``key``.
    explicit_component_axes : Dict[str, int]
        Component-axis map for explicitly mixture-marked keys from pass 1.

    Returns
    -------
    bool
        True when there is corroborating evidence from an explicit
        mixture-carrying key that shares the same component axis.
    """
    # Map fallback/derived keys to explicit references whose component axis can
    # corroborate that the key should be component-sliced.
    reference_map = {
        "p": ("p", "phi", "p_unconstrained", "phi_unconstrained"),
        "p_unconstrained": (
            "p_unconstrained",
            "p",
            "phi_unconstrained",
            "phi",
        ),
        "r": ("r", "mu", "phi", "r_unconstrained", "mu_unconstrained"),
        "r_unconstrained": (
            "r_unconstrained",
            "r",
            "mu_unconstrained",
            "mu",
            "phi_unconstrained",
            "phi",
        ),
        "mu": ("mu", "r", "p", "mu_unconstrained", "r_unconstrained"),
        "mu_unconstrained": (
            "mu_unconstrained",
            "mu",
            "r_unconstrained",
            "r",
            "p_unconstrained",
            "p",
        ),
        "phi": ("phi", "phi_unconstrained"),
        "phi_unconstrained": ("phi_unconstrained", "phi"),
        "gate": ("gate", "gate_unconstrained"),
        "gate_unconstrained": ("gate_unconstrained", "gate"),
    }
    for ref_key in reference_map.get(key, ()):
        if explicit_component_axes.get(ref_key) == component_axis:
            return True
    return False


def _resolve_component_axes(
    mixin: "ComponentMixin", samples: Dict
) -> Dict[str, int]:
    """Identify posterior-sample keys that carry a component axis.

    Parameters
    ----------
    mixin : ComponentMixin
        Result mixin instance providing model metadata.
    samples : Dict
        Posterior sample dictionary to inspect.

    Returns
    -------
    Dict[str, int]
        Mapping ``{key: component_axis}`` for keys that should be sliced by
        component selection.

    Notes
    -----
    This mirrors the SVI component resolver so MCMC and SVI agree on derived
    mixture semantics. The resolver intentionally uses `DerivedParam` lineage
    through ``derive_axis_membership`` rather than introducing new key-shape
    heuristics.
    """
    from ..core.axis_layout import build_sample_layouts, derive_axis_membership

    mc = getattr(mixin, "model_config", None)
    specs = list(getattr(mc, "param_specs", None) or []) if mc else []
    specs_by_name = {s.name: s for s in specs}

    # Resolve membership using the shared derive_axis_membership cascade so
    # derived canonical keys inherit mixture axes from their sources.
    mp_cfg = getattr(mc, "mixture_params", None) if mc else None
    mp_expanded, dp = (
        derive_axis_membership(mc, samples=samples, has_sample_dim=True)
        if mc is not None
        else (None, None)
    )
    mp_final = mp_expanded if mp_cfg is not None else None

    sample_layouts = build_sample_layouts(
        specs,
        samples,
        n_genes=getattr(mixin, "n_genes", None),
        n_cells=getattr(mixin, "n_cells", None),
        n_components=getattr(mixin, "n_components", None),
        n_datasets=getattr(mc, "n_datasets", None) if mc else None,
        mixture_params=mp_final,
        dataset_params=dp,
        has_sample_dim=True,
    )
    # Prefer per-sample semantic layouts; fall back to the result object's stored
    # layouts when a key is known there but not reconstructable from current
    # sample metadata alone.
    parent_layouts = getattr(mixin, "layouts", {})

    # Pass 1: keys explicitly marked as mixture by ParamSpecs or known mixing
    # carriers (weights/logits).
    explicit: Dict[str, int] = {}
    for key in samples:
        spec = specs_by_name.get(key)
        layout = sample_layouts.get(key, parent_layouts.get(key))
        if layout is None or layout.component_axis is None:
            continue
        if (spec is not None and spec.is_mixture) or key in _MIXING_KEY_NAMES:
            explicit[key] = layout.component_axis

    # Pass 2: fallback/derived canonical keys are accepted only when explicit
    # evidence agrees on the component axis.
    result = dict(explicit)
    explicit_axes = set(explicit.values())
    for key in samples:
        if key in result or key not in _FALLBACK_MIXTURE_KEYS:
            continue
        layout = sample_layouts.get(key, parent_layouts.get(key))
        component_axis = layout.component_axis if layout is not None else None

        # Accept when canonical-reference evidence matches OR when the inferred
        # axis aligns with an explicit mixture axis observed in pass 1.
        if component_axis is not None and (
            _has_fallback_mixture_evidence(key, component_axis, explicit)
            or component_axis in explicit_axes
        ):
            result[key] = component_axis

    return result


# ==============================================================================
# Component Mixin
# ==============================================================================


class ComponentMixin:
    """Mixin providing mixture-component subsetting."""

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def get_component(
        self, component_index, renormalize: bool = True
    ) -> "ScribeMCMCResults":
        """Return a component-restricted view of the results.

        Parameters
        ----------
        component_index : int, slice, array-like, or bool mask
            Selector over mixture components.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.

        Returns
        -------
        ScribeMCMCResults
            Results restricted to the requested components.
        """
        selected = normalize_component_indices(
            component_index, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._create_single_component_subset(int(selected[0]))
        return self.get_components(selected, renormalize=renormalize)

    def get_components(
        self, component_indices, renormalize: bool = True
    ) -> "ScribeMCMCResults":
        """Select multiple components preserving mixture semantics.

        Parameters
        ----------
        component_indices : int, slice, array-like, or bool mask
            Selector over mixture components.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.

        Returns
        -------
        ScribeMCMCResults
            Mixture-aware results with reduced ``n_components``.
        """
        from .results import ScribeMCMCResults

        selected = normalize_component_indices(
            component_indices, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._create_single_component_subset(int(selected[0]))

        new_samples = self._subset_samples_by_components(
            self.samples,
            selected,
            renormalize=renormalize,
            squeeze_single=False,
        )
        new_n_components = int(selected.shape[0])
        new_model_config = self.model_config.model_copy(
            update={"n_components": new_n_components}
        )
        # Multi-component: axis kept, just fewer elements.
        subset = ScribeMCMCResults(
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
            n_components=new_n_components,
            param_layouts=dict(self.layouts),
        )
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        if per_ds is not None:
            subset._n_cells_per_dataset = per_ds
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            subset._dataset_indices = ds_idx
        return subset

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _create_single_component_subset(
        self, component_index: int
    ) -> "ScribeMCMCResults":
        """Extract a single component and drop mixture semantics."""
        from .results import ScribeMCMCResults

        new_samples = self._subset_samples_by_components(
            self.samples,
            jnp.asarray([component_index], dtype=jnp.int32),
            renormalize=True,
            squeeze_single=True,
        )
        base_model = self.model_type.replace("_mix", "")
        new_model_config = self.model_config.model_copy(
            update={"base_model": base_model, "n_components": None}
        )
        # Single-component extraction collapses the component axis.
        new_layouts = subset_layouts(self.layouts, COMPONENTS)

        subset = ScribeMCMCResults(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=base_model,
            model_config=new_model_config,
            prior_params=getattr(self, "prior_params", {}),
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=None,
            param_layouts=new_layouts,
        )
        per_ds = getattr(self, "_n_cells_per_dataset", None)
        if per_ds is not None:
            subset._n_cells_per_dataset = per_ds
        ds_idx = getattr(self, "_dataset_indices", None)
        if ds_idx is not None:
            subset._dataset_indices = ds_idx
        return subset

    def _subset_samples_by_components(
        self,
        samples: Dict,
        component_indices: jnp.ndarray,
        renormalize: bool = True,
        squeeze_single: bool = False,
    ) -> Dict:
        """Subset sample dict along the mixture-component axis.

        Parameters
        ----------
        samples : Dict
            Raw sample dictionary.
        component_indices : jnp.ndarray
            1-D selected component indices.
        renormalize : bool, default=True
            Renormalize mixing fractions after subsetting.
        squeeze_single : bool, default=False
            Remove the component axis when only one component is selected.

        Returns
        -------
        Dict
            Subset sample dictionary.
        """
        if samples is None:
            return None

        # Resolve per-key component axes using shared axis metadata and derived
        # lineage expansion so canonical keys are sliced consistently.
        component_axes = _resolve_component_axes(self, samples)
        use_single = squeeze_single and int(component_indices.shape[0]) == 1
        single_index = int(component_indices[0]) if use_single else None
        new_samples: Dict = {}

        for key, values in samples.items():
            component_axis = component_axes.get(key)
            if (
                component_axis is None
                or not hasattr(values, "ndim")
                or values.ndim <= component_axis
            ):
                new_samples[key] = values
                continue

            slicer = [slice(None)] * values.ndim
            slicer[component_axis] = (
                single_index if use_single else component_indices
            )
            selected = values[tuple(slicer)]

            if renormalize and key == "mixing_weights" and not use_single:
                selected = renormalize_mixing_weights(
                    selected, axis=component_axis
                )
            elif (
                renormalize
                and key == "mixing_logits_unconstrained"
                and not use_single
            ):
                selected = renormalize_mixing_logits(
                    selected, axis=component_axis
                )

            new_samples[key] = selected

        return new_samples

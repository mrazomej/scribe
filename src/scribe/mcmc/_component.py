"""
Component mixin for MCMC results.

Provides mixture-component selection with optional renormalization of
mixing fractions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import jax.numpy as jnp

from ..core.component_indexing import (
    normalize_component_indices,
    renormalize_mixing_logits,
    renormalize_mixing_weights,
)

if TYPE_CHECKING:
    from .results import ScribeMCMCResults


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
            n_components=new_n_components,
        )

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
        return ScribeMCMCResults(
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
        )

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

        specs_by_name = {}
        if getattr(self.model_config, "param_specs", None):
            specs_by_name = {s.name: s for s in self.model_config.param_specs}

        n_comp = self.n_components
        use_single = squeeze_single and int(component_indices.shape[0]) == 1
        single_index = int(component_indices[0]) if use_single else None
        new_samples: Dict = {}

        for key, values in samples.items():
            spec = specs_by_name.get(key)
            has_mixture_axis = hasattr(values, "ndim") and values.ndim > 1
            is_named_mixture_weight = key in {
                "mixing_weights",
                "mixing_logits_unconstrained",
            }
            is_mixture = has_mixture_axis and (
                (spec is not None and spec.is_mixture)
                or is_named_mixture_weight
            )
            if not is_mixture:
                new_samples[key] = values
                continue

            # Locate the component axis (typically axis 1)
            component_axis = 1
            if values.shape[component_axis] != n_comp:
                candidates = [
                    ax
                    for ax in range(1, values.ndim)
                    if values.shape[ax] == n_comp
                ]
                if not candidates:
                    new_samples[key] = values
                    continue
                component_axis = candidates[0]

            slicer = [slice(None)] * values.ndim
            slicer[component_axis] = (
                single_index if use_single else component_indices
            )
            selected = values[tuple(slicer)]

            if renormalize and key == "mixing_weights" and not use_single:
                selected = renormalize_mixing_weights(selected, axis=-1)
            elif (
                renormalize
                and key == "mixing_logits_unconstrained"
                and not use_single
            ):
                selected = renormalize_mixing_logits(selected, axis=-1)

            new_samples[key] = selected

        return new_samples

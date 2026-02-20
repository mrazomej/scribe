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
            if (
                key == name
                or key.startswith(name + "_")
                or key.startswith("log_" + name + "_")
            ):
                if spec.is_mixture:
                    mixture_keys.add(key)
                break

    return mixture_keys


# ==============================================================================
# Component Mixin
# ==============================================================================


class ComponentMixin:
    """Mixin providing component/mixture model operations."""

    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(
        self, component_index: Any, renormalize: bool = True
    ):
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

    def get_components(
        self, component_indices: Any, renormalize: bool = True
    ):
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
        """
        Subset posterior samples to a single mixture component.

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

        specs_by_name = {s.name: s for s in self.model_config.param_specs}
        new_posterior_samples = {}

        for key, value in samples.items():
            spec = specs_by_name.get(key)
            is_named_mixture_weight = key in {
                "mixing_weights",
                "mixing_logits_unconstrained",
            }
            if (spec is not None and spec.is_mixture) or is_named_mixture_weight:
                if not hasattr(value, "ndim") or value.ndim <= 1:
                    new_posterior_samples[key] = value
                    continue

                component_axis = 1
                if (
                    self.n_components is not None
                    and value.shape[component_axis] != self.n_components
                ):
                    candidates = [
                        ax
                        for ax in range(1, value.ndim)
                        if value.shape[ax] == self.n_components
                    ]
                    if not candidates:
                        new_posterior_samples[key] = value
                        continue
                    component_axis = candidates[0]

                slicer = [slice(None)] * value.ndim
                slicer[component_axis] = component_index
                new_posterior_samples[key] = value[tuple(slicer)]
            else:
                new_posterior_samples[key] = value

        return new_posterior_samples

    # --------------------------------------------------------------------------

    def _subset_posterior_samples_by_components(
        self,
        samples: Dict,
        component_indices: jnp.ndarray,
        renormalize: bool = True,
    ) -> Dict:
        """
        Subset posterior samples to multiple mixture components.

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

        specs_by_name = {s.name: s for s in self.model_config.param_specs}
        new_posterior_samples = {}

        for key, value in samples.items():
            spec = specs_by_name.get(key)
            is_named_mixture_weight = key in {
                "mixing_weights",
                "mixing_logits_unconstrained",
            }
            if (spec is not None and spec.is_mixture) or is_named_mixture_weight:
                if not hasattr(value, "ndim") or value.ndim <= 1:
                    selected_value = value
                else:
                    component_axis = 1
                    if (
                        self.n_components is not None
                        and value.shape[component_axis] != self.n_components
                    ):
                        candidates = [
                            ax
                            for ax in range(1, value.ndim)
                            if value.shape[ax] == self.n_components
                        ]
                        if not candidates:
                            new_posterior_samples[key] = value
                            continue
                        component_axis = candidates[0]

                    slicer = [slice(None)] * value.ndim
                    slicer[component_axis] = component_indices
                    selected_value = value[tuple(slicer)]

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

        return type(self)(
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
            predictive_samples=new_predictive_samples,
            n_components=new_n_components,
            _original_n_genes=getattr(self, "_original_n_genes", None),
            _gene_axis_by_key=getattr(self, "_gene_axis_by_key", None),
        )

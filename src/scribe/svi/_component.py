"""
Component mixin for SVI results.

This mixin provides methods for working with mixture model components, allowing
extraction and subsetting of individual component views.
"""

from typing import Dict, List, Optional, Set
import jax.numpy as jnp


# ==============================================================================
# Metadata-driven helpers
# ==============================================================================


def _build_mixture_keys(
    param_specs: List, params: Dict[str, jnp.ndarray]
) -> Set[str]:
    """
    Identify variational-parameter keys that belong to mixture-specific specs.

    Each ``ParamSpec`` with ``is_mixture=True`` owns variational parameter keys
    whose names start with ``{spec.name}_`` or ``log_{spec.name}_``.  When
    multiple specs could match the same key (e.g. ``"phi"`` vs
    ``"phi_capture"``), the **longest** spec name wins, preventing false
    positives.

    Parameters
    ----------
    param_specs : List[ParamSpec]
        Full parameter specifications from ``model_config.param_specs``.
    params : Dict[str, jnp.ndarray]
        Flat variational-parameter dictionary (``self.params``).

    Returns
    -------
    Set[str]
        Keys in *params* that should be indexed along the component axis
        when extracting a single mixture component.
    """
    if not param_specs:
        return set()

    # Sort specs longest-name-first so greedy prefix matching is correct
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
                break  # longest match found, stop searching

    return mixture_keys


# ==============================================================================
# Component Mixin
# ==============================================================================


class ComponentMixin:
    """Mixin providing component/mixture model operations."""

    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(self, component_index):
        """
        Create a view of the results selecting a specific mixture component.

        This method returns a new ScribeSVIResults object that contains
        parameter values for the specified component, allowing for further
        gene-based indexing. Only applicable to mixture models.

        Parameters
        ----------
        component_index : int
            Index of the component to select

        Returns
        -------
        ScribeSVIResults
            A new ScribeSVIResults object with parameters for the selected
            component

        Raises
        ------
        ValueError
            If the model is not a mixture model
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Component view only applies to mixture models with "
                "multiple components"
            )

        # Check if component_index is valid
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index {component_index} out of range "
                f"[0, {self.n_components-1}]"
            )

        # Create new params dict with component subset
        new_params = dict(self.params)

        # Handle all parameters based on their structure
        self._subset_params_by_component(new_params, component_index)

        # Create new posterior samples if available
        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_samples_by_component(
                self.posterior_samples, component_index
            )

        # Create new predictive samples if available - this is more complex
        # as we would need to condition on the component
        new_predictive_samples = None

        # Create new instance with component subset
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

        Uses ``ParamSpec`` metadata from ``model_config.param_specs`` to
        determine which keys carry a leading component axis.  Keys that belong
        to a spec with ``is_mixture=True`` are indexed along axis 0; all
        others are copied unchanged.

        Parameters
        ----------
        new_params : Dict
            Mutable params dict (pre-populated with a shallow copy of
            ``self.params``).  Modified in-place.
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

    def _subset_posterior_samples_by_component(
        self, samples: Dict, component_index: int
    ) -> Dict:
        """
        Subset posterior samples to a single mixture component.

        Posterior sample keys correspond directly to ``ParamSpec.name``
        values.  Samples from mixture-specific specs have a component axis
        at position 1 (axis 0 is the sample axis).

        Parameters
        ----------
        samples : Dict
            Posterior samples dictionary.
        component_index : int
            Which mixture component to extract.

        Returns
        -------
        Dict
            New dictionary with the component dimension removed for
            mixture-specific parameters.
        """
        if samples is None:
            return None

        specs_by_name = {s.name: s for s in self.model_config.param_specs}

        new_posterior_samples = {}
        for key, value in samples.items():
            spec = specs_by_name.get(key)
            if spec is not None and spec.is_mixture:
                if value.ndim > 2:
                    # (n_samples, n_components, n_genes) → (n_samples, n_genes)
                    new_posterior_samples[key] = value[:, component_index, :]
                elif value.ndim > 1:
                    # (n_samples, n_components) → (n_samples,)
                    new_posterior_samples[key] = value[:, component_index]
                else:
                    new_posterior_samples[key] = value
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
        """Create a new instance for a specific component."""
        # Create a non-mixture model type
        base_model = self.model_type.replace("_mix", "")

        # Create a modified model config with n_components=None to indicate
        # this is now a non-mixture result after component selection
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
            model_type=base_model,  # Remove _mix suffix
            model_config=new_model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=None,  # No longer a mixture model
        )

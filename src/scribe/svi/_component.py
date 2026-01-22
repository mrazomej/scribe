"""
Component mixin for SVI results.

This mixin provides methods for working with mixture model components, allowing
extraction and subsetting of individual component views.
"""

from typing import Dict, Optional
import jax.numpy as jnp

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
        Handle subsetting of all parameters based on their structure.

        This method intelligently handles parameters based on their dimensions
        and naming conventions, regardless of parameterization.
        """
        # Define parameter categories based on their structure

        # Component-gene-specific parameters (shape: [n_components, n_genes])
        # These parameters have both component and gene dimensions
        component_gene_specific = [
            # Standard parameterization
            "r_loc",
            "r_scale",  # dispersion parameters
            "gate_alpha",
            "gate_beta",  # zero-inflation parameters
            # Standard unconstrained parameterization
            "r_unconstrained_loc",
            "r_unconstrained_scale",
            "gate_unconstrained_loc",
            "gate_unconstrained_scale",
            # Linked parameterization
            "mu_loc",
            "mu_scale",  # mean parameters
            "gate_alpha",
            "gate_beta",  # zero-inflation parameters
            # Odds ratio parameterization
            "phi_alpha",
            "phi_beta",  # odds ratio parameters
            "gate_alpha",
            "gate_beta",  # zero-inflation parameters
            # Odds ratio unconstrained parameterization
            "phi_unconstrained_loc",
            "phi_unconstrained_scale",
            "gate_unconstrained_loc",
            "gate_unconstrained_scale",
            # Low-rank guide parameters (standard constrained)
            "log_r_loc",
            "log_r_W",
            "log_r_raw_diag",
            # Low-rank guide parameters (standard unconstrained)
            "r_unconstrained_W",
            "r_unconstrained_raw_diag",
            # Low-rank guide parameters (linked/odds_ratio constrained)
            "log_mu_loc",
            "log_mu_W",
            "log_mu_raw_diag",
            # Low-rank guide parameters (linked/odds_ratio unconstrained)
            "mu_unconstrained_loc",
            "mu_unconstrained_W",
            "mu_unconstrained_raw_diag",
            # Low-rank guide parameters (gate - unconstrained)
            "gate_unconstrained_W",
            "gate_unconstrained_raw_diag",
        ]

        # Component-specific parameters (shape: [n_components])
        # These parameters have only component dimension
        component_specific = [
            # Standard unconstrained parameterization
            "p_unconstrained_loc",
            "p_unconstrained_scale",
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
            # Odds ratio unconstrained parameterization
            "phi_unconstrained_loc",
            "phi_unconstrained_scale",
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
        ]

        # Cell-specific parameters (shape: [n_cells])
        # These parameters are cell-specific and not component-specific
        cell_specific = [
            # Standard parameterization
            "p_capture_alpha",
            "p_capture_beta",  # capture probability parameters
            # Standard unconstrained parameterization
            "p_capture_unconstrained_loc",
            "p_capture_unconstrained_scale",
            # Linked parameterization
            "p_capture_alpha",
            "p_capture_beta",  # capture probability parameters
            # Odds ratio parameterization
            "phi_capture_alpha",
            "phi_capture_beta",  # capture odds ratio parameters
            # Odds ratio unconstrained parameterization
            "phi_capture_unconstrained_loc",
            "phi_capture_unconstrained_scale",
        ]

        # Parameters that can be either component-specific or shared depending
        # on model config These need special handling based on
        # component_specific_params setting
        configurable_params = [
            # Standard parameterization
            "p_alpha",
            "p_beta",  # success probability parameters
            # Linked parameterization
            "p_alpha",
            "p_beta",  # success probability parameters
            # Odds ratio parameterization
            "phi_alpha",
            "phi_beta",  # odds ratio parameters
        ]

        # Shared parameters (scalar or global)
        # These parameters are shared across all components
        shared_params = [
            # Standard parameterization
            "mixing_conc",  # mixture concentrations
            # Standard unconstrained parameterization
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
            # Linked parameterization
            "mixing_conc",  # mixture concentrations
            # Odds ratio parameterization
            "mixing_conc",  # mixture concentrations
            # Odds ratio unconstrained parameterization
            "mixing_logits_unconstrained_loc",
            "mixing_logits_unconstrained_scale",
        ]

        # Additional parameters that might be present but not categorized above
        # These are typically scalar or global parameters
        additional_params = [
            # Any other parameters that don't fit the above categories
            # This list can be expanded as needed
        ]

        # Handle component-gene-specific parameters (shape: [n_components,
        # n_genes])
        for param_name in component_gene_specific:
            if param_name in self.params:
                param = self.params[param_name]
                # Check if parameter has component dimension
                if param.ndim > 1:  # Has component dimension
                    new_params[param_name] = param[component_index]
                else:  # Scalar parameter, copy as-is
                    new_params[param_name] = param

        # Handle component-specific parameters (shape: [n_components])
        for param_name in component_specific:
            if param_name in self.params:
                param = self.params[param_name]
                # Check if parameter has component dimension
                if param.ndim > 0:  # Has component dimension
                    new_params[param_name] = param[component_index]
                else:  # Scalar parameter, copy as-is
                    new_params[param_name] = param

        # Handle cell-specific parameters (copy as-is, not component-specific)
        for param_name in cell_specific:
            if param_name in self.params:
                new_params[param_name] = self.params[param_name]

        # Handle configurable parameters (can be component-specific or shared)
        for param_name in configurable_params:
            if param_name in self.params:
                param = self.params[param_name]
                # Check if parameter has component dimension
                if param.ndim > 0:  # Has component dimension
                    new_params[param_name] = param[component_index]
                else:  # Scalar parameter, copy as-is
                    new_params[param_name] = param

        # Handle shared parameters (copy as-is, used across all components)
        for param_name in shared_params:
            if param_name in self.params:
                new_params[param_name] = self.params[param_name]

        # Handle any additional parameters that might be present
        for param_name in additional_params:
            if param_name in self.params:
                new_params[param_name] = self.params[param_name]

        # Handle any remaining parameters not explicitly categorized
        # This ensures we don't miss any parameters
        for param_name in self.params:
            if param_name not in new_params:
                # For any uncategorized parameters, copy as-is
                new_params[param_name] = self.params[param_name]

    # --------------------------------------------------------------------------

    def _subset_posterior_samples_by_component(
        self, samples: Dict, component_index: int
    ) -> Dict:
        """
        Create a new posterior samples dictionary for the given component index.

        This method handles all parameter types based on their dimensions.
        """
        if samples is None:
            return None

        new_posterior_samples = {}

        # Define parameter categories for posterior samples
        component_gene_specific_samples = [
            # Standard parameterization
            "r",  # dispersion parameter
            "gate",  # zero-inflation parameter
            # Standard unconstrained parameterization
            "r_unconstrained",  # dispersion parameter
            "gate_unconstrained",  # zero-inflation parameter
            # Linked parameterization
            "mu",  # mean parameter
            "gate",  # zero-inflation parameter
            # Odds ratio parameterization
            "phi",  # odds ratio parameter
            "gate",  # zero-inflation parameter
            # Odds ratio unconstrained parameterization
            "phi_unconstrained",  # odds ratio parameter
            "gate_unconstrained",  # zero-inflation parameter
        ]

        component_specific_samples = [
            # Standard parameterization
            "p",  # success probability parameter
            # Standard unconstrained parameterization
            "p_unconstrained",  # success probability parameter
            "mixing_logits_unconstrained",  # mixing logits
            # Linked parameterization
            "p",  # success probability parameter
            # Odds ratio parameterization
            "phi",  # odds ratio parameter
            # Odds ratio unconstrained parameterization
            "phi_unconstrained",  # odds ratio parameter
            "mixing_logits_unconstrained",  # mixing logits
        ]

        cell_specific_samples = [
            # Standard parameterization
            "p_capture",  # capture probability parameter
            # Standard unconstrained parameterization
            "p_capture_unconstrained",  # capture probability parameter
            # Linked parameterization
            "p_capture",  # capture probability parameter
            # Odds ratio parameterization
            "phi_capture",  # capture odds ratio parameter
            # Odds ratio unconstrained parameterization
            "phi_capture_unconstrained",  # capture odds ratio parameter
        ]

        # Shared parameters (scalar or global)
        shared_samples = [
            # Standard parameterization
            "mixing_weights",  # mixture weights
            # Standard unconstrained parameterization
            "mixing_logits_unconstrained",  # mixing logits
            # Linked parameterization
            "mixing_weights",  # mixture weights
            # Odds ratio parameterization
            "mixing_weights",  # mixture weights
            # Odds ratio unconstrained parameterization
            "mixing_logits_unconstrained",  # mixing logits
        ]

        # Configurable parameters (can be component-specific or shared)
        configurable_samples = [
            # Standard parameterization
            "p",  # success probability parameter
            # Linked parameterization
            "p",  # success probability parameter
            # Odds ratio parameterization
            "phi",  # odds ratio parameter
        ]

        # Additional parameters that might be present in posterior samples
        # These are typically derived parameters or deterministic values
        additional_samples = [
            # Any other parameters that don't fit the above categories
            # This list can be expanded as needed
        ]

        # Handle component-gene-specific samples
        # (shape: [n_samples, n_components, n_genes])
        for param_name in component_gene_specific_samples:
            if param_name in samples:
                sample_value = samples[param_name]
                if sample_value.ndim > 2:  # Has component dimension
                    new_posterior_samples[param_name] = sample_value[
                        :, component_index, :
                    ]
                else:  # Scalar parameter, copy as-is
                    new_posterior_samples[param_name] = sample_value

        # Handle component-specific samples (shape: [n_samples, n_components])
        for param_name in component_specific_samples:
            if param_name in samples:
                sample_value = samples[param_name]
                if sample_value.ndim > 1:  # Has component dimension
                    new_posterior_samples[param_name] = sample_value[
                        :, component_index
                    ]
                else:  # Scalar parameter, copy as-is
                    new_posterior_samples[param_name] = sample_value

        # Handle cell-specific samples (copy as-is, not component-specific)
        for param_name in cell_specific_samples:
            if param_name in samples:
                new_posterior_samples[param_name] = samples[param_name]

        # Handle shared samples (copy as-is, used across all components)
        for param_name in shared_samples:
            if param_name in samples:
                new_posterior_samples[param_name] = samples[param_name]

        # Handle configurable samples (can be component-specific or shared)
        for param_name in configurable_samples:
            if param_name in samples:
                sample_value = samples[param_name]
                if sample_value.ndim > 1:  # Has component dimension
                    new_posterior_samples[param_name] = sample_value[
                        :, component_index
                    ]
                else:  # Scalar parameter, copy as-is
                    new_posterior_samples[param_name] = sample_value

        # Handle any additional samples that might be present
        for param_name in additional_samples:
            if param_name in samples:
                new_posterior_samples[param_name] = samples[param_name]

        # Handle any remaining samples not explicitly categorized
        # This ensures we don't miss any parameters
        for param_name in samples:
            if param_name not in new_posterior_samples:
                # For any uncategorized parameters, copy as-is
                new_posterior_samples[param_name] = samples[param_name]

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

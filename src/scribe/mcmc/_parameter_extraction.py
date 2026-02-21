"""
Parameter extraction mixin for MCMC results.

Provides component-specificity checks, posterior quantiles, and MAP
estimation.
"""

from typing import Dict, Optional

import jax.numpy as jnp


# ==============================================================================
# Parameter Extraction Mixin
# ==============================================================================


class ParameterExtractionMixin:
    """Mixin providing parameter extraction methods.

    No canonical conversion is needed: the new model system registers
    derived parameters as ``numpyro.deterministic`` sites and unconstrained
    specs sample via ``TransformedDistribution`` in constrained space, so
    ``MCMC.get_samples()`` already contains canonical parameters.
    """

    # --------------------------------------------------------------------------
    # Component specificity
    # --------------------------------------------------------------------------

    def _is_component_specific_param(self, param_name: str) -> bool:
        """Check whether a parameter carries a mixture-component axis.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g. ``"p"``, ``"phi"``).

        Returns
        -------
        bool
            ``True`` when the parameter has ``is_mixture=True`` in
            ``model_config.param_specs``.
        """
        if not self.model_config.param_specs:
            mixture_params = self.model_config.mixture_params or []
            return param_name in mixture_params

        for spec in self.model_config.param_specs:
            if spec.name == param_name:
                return spec.is_mixture
        return False

    # --------------------------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------------------------

    def get_posterior_quantiles(
        self,
        param: str,
        quantiles=(0.025, 0.5, 0.975),
    ):
        """Compute quantiles for a posterior parameter.

        Parameters
        ----------
        param : str
            Parameter name.
        quantiles : tuple, default=(0.025, 0.5, 0.975)
            Quantile levels.

        Returns
        -------
        dict
            Mapping from quantile level to value array.
        """
        return _get_posterior_quantiles(
            self.get_posterior_samples(), param, quantiles
        )

    # --------------------------------------------------------------------------
    # MAP estimation
    # --------------------------------------------------------------------------

    def get_map(self):
        """Maximum a posteriori (MAP) parameter estimates.

        Uses the ``potential_energy`` extra field from the MCMC run when
        available; otherwise falls back to the posterior mean.

        Returns
        -------
        dict
            Parameter name -> MAP value.
        """
        samples = self.get_posterior_samples()

        try:
            potential_energy = self.get_extra_fields()["potential_energy"]
            return _get_map_estimate(samples, potential_energy)
        except Exception:
            return _get_map_estimate(samples)


# ==============================================================================
# Module-level helpers
# ==============================================================================


def _get_posterior_quantiles(
    samples: Dict, param: str, quantiles=(0.025, 0.5, 0.975)
):
    """Compute quantiles for *param* from a samples dictionary."""
    param_samples = samples[param]
    return {q: jnp.quantile(param_samples, q) for q in quantiles}


def _get_map_estimate(
    samples: Dict, potential_energy: Optional[jnp.ndarray] = None
):
    """MAP estimate from samples, optionally guided by potential energy."""
    if potential_energy is not None:
        map_idx = int(jnp.argmin(potential_energy))
        return {k: v[map_idx] for k, v in samples.items()}

    # Fallback: posterior mean
    return {k: jnp.mean(v, axis=0) for k, v in samples.items()}

"""
Model helpers mixin for MCMC results.

Provides internal access to model functions and log-likelihood functions.
"""

from typing import Callable

from ..models.config import ModelConfig


# ==============================================================================
# Model Helpers Mixin
# ==============================================================================


class ModelHelpersMixin:
    """Mixin providing model and log-likelihood function access."""

    def _model(self) -> Callable:
        """Return the NumPyro model function for this configuration."""
        return _get_model_fn(self.model_config)

    def _log_likelihood_fn(self) -> Callable:
        """Return the log-likelihood function for this model type."""
        return _get_log_likelihood_fn(self.model_type)


# ==============================================================================
# Module-level helpers
# ==============================================================================


def _get_model_fn(model_config: ModelConfig) -> Callable:
    """Look up the model function for *model_config*.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    Callable
        The NumPyro model function.
    """
    from ..models.model_registry import get_model_and_guide

    return get_model_and_guide(model_config, guide_families=None)[0]


def _get_log_likelihood_fn(model_type: str) -> Callable:
    """Look up the log-likelihood function for *model_type*."""
    from ..models.model_registry import get_log_likelihood_fn

    return get_log_likelihood_fn(model_type)

"""
Model helpers mixin for SVI results.

This mixin provides internal helper methods for accessing model and guide
functions, parameterization information, and log-likelihood functions.
"""

from typing import Callable, Tuple, Optional
from ..models.config import ModelConfig

# ==============================================================================
# Model Helpers Mixin
# ==============================================================================


class ModelHelpersMixin:
    """Mixin providing model and guide access helpers."""

    # --------------------------------------------------------------------------
    # Get model and guide functions
    # --------------------------------------------------------------------------

    def _model_and_guide(self) -> Tuple[Callable, Optional[Callable]]:
        """Get the model and guide functions based on model type."""
        from ..models.model_registry import get_model_and_guide

        return get_model_and_guide(self.model_config)

    # --------------------------------------------------------------------------
    # Get parameterization
    # --------------------------------------------------------------------------

    def _parameterization(self) -> str:
        """Get the parameterization type."""
        return self.model_config.parameterization or ""

    # --------------------------------------------------------------------------
    # Get if unconstrained
    # --------------------------------------------------------------------------

    def _unconstrained(self) -> bool:
        """Get if the parameterization is unconstrained."""
        return self.model_config.unconstrained

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    def _log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        from ..models.model_registry import get_log_likelihood_fn

        return get_log_likelihood_fn(self.model_type)

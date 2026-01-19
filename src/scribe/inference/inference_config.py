"""
Inference configuration utilities.

This module provides helper functions for creating and validating inference
configurations. Uses a registry pattern for extensible default config creation.
"""

from typing import Callable
from ..models.config import InferenceConfig, SVIConfig, MCMCConfig
from ..models.config.enums import InferenceMethod


# ==============================================================================
# Registry for default inference config factories
# ==============================================================================

_DEFAULT_CONFIG_FACTORIES: dict[
    InferenceMethod, Callable[[], InferenceConfig]
] = {}


def _register_default_config_factory(
    method: InferenceMethod, factory: Callable[[], InferenceConfig]
) -> None:
    """Register a factory function for creating default configs.

    This is an internal function used to populate the registry. Factories
    are registered at module import time.

    Parameters
    ----------
    method : InferenceMethod
        The inference method this factory handles.
    factory : Callable[[], InferenceConfig]
        Factory function that creates a default InferenceConfig.
    """
    _DEFAULT_CONFIG_FACTORIES[method] = factory


# ------------------------------------------------------------------------------
# Default config factory implementations
# ------------------------------------------------------------------------------


def _create_default_svi_config() -> InferenceConfig:
    """Create default SVI config: 50k steps, 512 batch size, stable updates."""
    svi_config = SVIConfig(
        optimizer=None,  # Will use default Adam
        loss=None,  # Will use default TraceMeanField_ELBO
        n_steps=50_000,
        batch_size=512,
        stable_update=True,
    )
    return InferenceConfig.from_svi(svi_config)


# ------------------------------------------------------------------------------


def _create_default_mcmc_config() -> InferenceConfig:
    """Create default MCMC config: 2000 samples, 1000 warmup, 1 chain."""
    mcmc_config = MCMCConfig(
        n_samples=2_000,
        n_warmup=1_000,
        n_chains=1,
        mcmc_kwargs=None,
    )
    return InferenceConfig.from_mcmc(mcmc_config)


# ------------------------------------------------------------------------------


def _create_default_vae_config() -> InferenceConfig:
    """Create default VAE config: uses SVI defaults (100k steps, 512 batch)."""
    # VAE uses SVI config with same defaults
    svi_config = SVIConfig(
        optimizer=None,
        loss=None,
        n_steps=100_000,
        batch_size=512,
        stable_update=True,
    )
    return InferenceConfig.from_vae(svi_config)


# ------------------------------------------------------------------------------
# Register factories
# ------------------------------------------------------------------------------

# Register factories
_register_default_config_factory(
    InferenceMethod.SVI, _create_default_svi_config
)
_register_default_config_factory(
    InferenceMethod.MCMC, _create_default_mcmc_config
)
_register_default_config_factory(
    InferenceMethod.VAE, _create_default_vae_config
)


# ==============================================================================
# Public API
# ==============================================================================


def create_default_inference_config(
    inference_method: InferenceMethod,
) -> InferenceConfig:
    """Create default InferenceConfig for a given inference method.

    This function uses a registry pattern to route to method-specific default
    config factories. This makes it easy to extend with new inference methods
    without modifying this function.

    Parameters
    ----------
    inference_method : InferenceMethod
        The inference method to create a default config for.

    Returns
    -------
    InferenceConfig
        InferenceConfig with default settings for the specified method.

    Raises
    ------
    ValueError
        If the inference method is not registered in the default config factory
        registry.

    Examples
    --------
    Create default SVI config:

    >>> from scribe.inference.inference_config import create_default_inference_config
    >>> from scribe.models.config.enums import InferenceMethod
    >>>
    >>> inference_config = create_default_inference_config(InferenceMethod.SVI)

    Create default MCMC config:

    >>> inference_config = create_default_inference_config(InferenceMethod.MCMC)

    Create default VAE config:

    >>> inference_config = create_default_inference_config(InferenceMethod.VAE)

    Notes
    -----
    Default values:
    - SVI: n_steps=50000, batch_size=512, stable_update=True
    - MCMC: n_samples=2000, n_warmup=1000, n_chains=1
    - VAE: Uses SVI defaults (n_steps=100000, batch_size=512)

    The registry pattern allows easy extension - new inference methods can
    register their default config factories without modifying this function.

    See Also
    --------
    InferenceConfig : Unified inference configuration class.
    SVIConfig : SVI-specific configuration.
    MCMCConfig : MCMC-specific configuration.
    """
    factory = _DEFAULT_CONFIG_FACTORIES.get(inference_method)
    if factory is None:
        raise ValueError(
            f"Unknown inference method: {inference_method}. "
            f"Registered methods: "
            f"{list(_DEFAULT_CONFIG_FACTORIES.keys())}"
        )
    return factory()

"""Base classes and utilities for likelihood components.

This module provides the abstract base class for all likelihoods and
helper functions for capture parameter sampling.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

# Import at module level to avoid runtime import overhead
from scribe.stats.distributions import BetaPrime

if TYPE_CHECKING:
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig


# ==============================================================================
# Helper functions for capture parameter sampling
# These are defined at module level to ensure they're available for JIT tracing
# ==============================================================================


def _sample_phi_capture_constrained(
    prior_params: Tuple[float, float],
) -> jnp.ndarray:
    """Sample phi_capture from constrained BetaPrime distribution."""
    return numpyro.sample("phi_capture", BetaPrime(*prior_params))


# ------------------------------------------------------------------------------


def _sample_phi_capture_unconstrained(
    prior_params: Tuple[float, float],
    transform: dist.transforms.Transform,
    constrained_name: str,
) -> jnp.ndarray:
    """Sample phi_capture using TransformedDistribution (unconstrained)."""
    base_dist = dist.Normal(*prior_params)
    transformed_dist = dist.TransformedDistribution(base_dist, transform)
    return numpyro.sample(constrained_name, transformed_dist)


# ------------------------------------------------------------------------------


def _sample_p_capture_constrained(
    prior_params: Tuple[float, float],
) -> jnp.ndarray:
    """Sample p_capture from constrained Beta distribution."""
    return numpyro.sample("p_capture", dist.Beta(*prior_params))


# ------------------------------------------------------------------------------


def _sample_p_capture_unconstrained(
    prior_params: Tuple[float, float],
    transform: dist.transforms.Transform,
    constrained_name: str,
) -> jnp.ndarray:
    """Sample p_capture using TransformedDistribution (unconstrained)."""
    base_dist = dist.Normal(*prior_params)
    transformed_dist = dist.TransformedDistribution(base_dist, transform)
    return numpyro.sample(constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# Likelihood Base Class
# ------------------------------------------------------------------------------


class Likelihood(ABC):
    """
    Abstract base class for likelihood components.

    Subclasses implement the `sample` method which handles:

    1. Cell plate creation (with proper batching mode)
    2. Cell-specific parameter sampling
    3. Observation sampling/conditioning

    All subclasses must handle three plate modes:

    - **Prior predictive**: counts=None → sample counts from prior
    - **Full sampling**: counts provided, batch_size=None → condition on all
    - **Batch sampling**: counts provided, batch_size set → subsample cells

    Examples
    --------
    >>> class MyLikelihood(Likelihood):
    ...     def sample(self, param_values, cell_specs, counts, dims,
    ...                batch_size, model_config):
    ...         # Implementation
    ...         pass
    """

    @abstractmethod
    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
    ) -> None:
        """
        Sample observations given parameters.

        Parameters
        ----------
        param_values : Dict[str, jnp.ndarray]
            Already-sampled parameter values (global and gene-specific).
            Keys are parameter names (e.g., "p", "r", "mu").
        cell_specs : List[ParamSpec]
            Specs for cell-specific parameters to sample inside the cell plate.
            These are sampled within the plate context.
        counts : Optional[jnp.ndarray]
            Observed counts matrix of shape (n_cells, n_genes).
            If None, samples from prior (prior predictive mode).
        dims : Dict[str, int]
            Dimension sizes, e.g., {"n_cells": 10000, "n_genes": 2000}.
        batch_size : Optional[int]
            Mini-batch size for stochastic VI. If None, uses all cells.
        model_config : ModelConfig
            Model configuration with hyperparameters.
        vae_cell_fn : callable, optional
            If provided, called inside the cell plate **before** obs sampling.
            Signature: ``vae_cell_fn(batch_idx) -> Dict[str, jnp.ndarray]``.
            Returns decoder-driven parameter values to merge into
            ``param_values``.

        Notes
        -----
        This method should:
            1. Create the appropriate cell plate (with or without subsampling)
            2. If ``vae_cell_fn`` is provided, call it and update param_values
            3. Sample any cell-specific parameters from cell_specs
            4. Compute the likelihood distribution
            5. Sample or condition on counts
        """
        pass

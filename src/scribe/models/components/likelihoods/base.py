"""Base classes and utilities for likelihood components.

This module provides the abstract base class for all likelihoods and
helper functions for capture parameter sampling and cell-specific mixing.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import jax
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


# ==============================================================================
# Helper for broadcasting scalar/gene-specific p in mixture models
# ==============================================================================


def broadcast_p_for_mixture(
    p: jnp.ndarray, r: jnp.ndarray
) -> jnp.ndarray:
    """Broadcast ``p`` to match ``r``'s shape for mixture NB distributions.

    Handles all combinations of scalar, gene-specific, and mixture-specific
    shapes for the success probability parameter ``p``.  This is needed
    because hierarchical parameterizations produce gene-specific ``p``
    (shape ``(n_genes,)``), which must be expanded to ``(1, n_genes)`` for
    broadcasting with mixture ``r`` of shape ``(n_components, n_genes)``.

    After dataset indexing, ``p`` may carry a leading batch dimension
    (e.g., ``(batch, n_genes)``).  When ``r`` is 3-D
    (``(batch, n_components, n_genes)``), ``p`` is reshaped to
    ``(batch, 1, n_genes)`` so it broadcasts across components.

    Parameters
    ----------
    p : jnp.ndarray
        Success probability.  Possible shapes:

        - ``()`` — scalar (shared across components and genes)
        - ``(n_components,)`` — mixture-specific scalar
        - ``(n_genes,)`` — gene-specific (shared across components)
        - ``(n_components, n_genes)`` — both mixture- and gene-specific
        - ``(batch, n_genes)`` — per-cell gene-specific (after dataset
          indexing)
        - ``(batch, n_components, n_genes)`` — per-cell mixture+gene

    r : jnp.ndarray
        Dispersion parameter.  Shape ``(n_components, n_genes)`` or
        ``(batch, n_components, n_genes)`` in mixture models.

    Returns
    -------
    jnp.ndarray
        ``p`` reshaped for broadcasting with ``r``.
    """
    if p.ndim == 0:
        # Scalar — add two singleton dims for (K, G)
        if r.ndim == 3:
            return p[None, None, None]
        return p[None, None]
    elif p.ndim == 1:
        # Distinguish (n_genes,) from (n_components,) by comparing with r
        if r.ndim >= 2 and p.shape[0] == r.shape[-1]:
            # Gene-specific: (G,) → (1, G) or (1, 1, G) for 3-D r
            if r.ndim == 3:
                return p[None, None, :]
            return p[None, :]
        else:
            # Mixture-specific scalar: (K,) → (K, 1)
            return p[:, None]
    elif p.ndim == 2:
        if r.ndim == 3:
            # p is (batch, G) after dataset indexing, r is (batch, K, G).
            # Insert component singleton: (batch, 1, G)
            return p[:, None, :]
        # Already (K, G) or compatible 2-D shape
        return p
    # Already 3-D (batch, K, G) or compatible
    return p


# ==============================================================================
# Helper for cell-specific mixing weights (annotation priors)
# ==============================================================================


def compute_cell_specific_mixing(
    mixing_weights: jnp.ndarray,
    annotation_logits: jnp.ndarray,
) -> jnp.ndarray:
    """
    Combine global mixing weights with per-cell annotation logits.

    Computes cell-specific mixing probabilities by adding annotation logit
    offsets to the log of the global mixing weights and applying softmax.
    This implements the logit-nudging strategy for annotation priors in
    mixture models.

    Parameters
    ----------
    mixing_weights : jnp.ndarray, shape ``(K,)``
        Global mixing weight vector (simplex, sums to 1).  Typically
        sampled from a Dirichlet prior.
    annotation_logits : jnp.ndarray, shape ``(batch, K)``
        Per-cell additive logit offsets.  Zero rows leave the mixing
        weights unchanged; positive entries bias toward the corresponding
        component.

    Returns
    -------
    cell_mixing : jnp.ndarray, shape ``(batch, K)``
        Normalised per-cell mixing probabilities.  Each row sums to 1.

    Notes
    -----
    The computation is:

        πᵢₖ = softmaxₖ ( log πₖ + annotation_logitsᵢₖ )

    A small epsilon (1e-8) is added before taking the log to avoid
    −∞ when a mixing weight is numerically zero.

    This function is the single point where annotation priors interact
    with the mixture distribution.  A future auxiliary-observation
    strategy can replace this function while keeping the rest of the
    likelihood code unchanged.
    """
    log_weights = jnp.log(mixing_weights + 1e-8)  # (K,)
    cell_logits = log_weights + annotation_logits  # (batch, K)
    return jax.nn.softmax(cell_logits, axis=-1)  # (batch, K)


# ==============================================================================
# Helper: index per-dataset parameters by cell dataset assignment
# ==============================================================================


def index_dataset_params(
    param_values: Dict[str, jnp.ndarray],
    dataset_indices: jnp.ndarray,
    n_datasets: int,
    param_specs: Optional[List] = None,
) -> Dict[str, jnp.ndarray]:
    """Index per-dataset parameters using per-cell dataset assignments.

    For each parameter whose spec has ``is_dataset=True``, slice out the
    dataset axis using ``dataset_indices`` to produce per-cell values.

    When a parameter is **both** mixture and dataset (shape
    ``(K, D, ...)``) the dataset axis is 1.  After indexing, the result
    is transposed to **batch-first** layout ``(batch, K, ...)`` so that
    ``MixtureSameFamily`` sees the component dim as the rightmost batch
    dimension.

    When ``param_specs`` is ``None``, falls back to the legacy heuristic
    (leading dim equals ``n_datasets``).

    Parameters
    ----------
    param_values : Dict[str, jnp.ndarray]
        All sampled parameter values.
    dataset_indices : jnp.ndarray, shape ``(batch,)``
        Integer array mapping each cell in the current batch to a
        dataset index in ``{0, ..., n_datasets - 1}``.
    n_datasets : int
        Number of datasets.
    param_specs : List[ParamSpec], optional
        Parameter specifications.  Used to determine which parameters
        carry a dataset axis and whether they also carry a mixture axis.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Copy with per-dataset arrays replaced by per-cell arrays.
    """
    # Build spec lookup when available
    specs_by_name: Dict[str, object] = {}
    if param_specs is not None:
        for spec in param_specs:
            specs_by_name[spec.name] = spec

    indexed = {}
    for name, val in param_values.items():
        spec = specs_by_name.get(name)
        is_ds = spec is not None and getattr(spec, "is_dataset", False)
        is_mix = spec is not None and getattr(spec, "is_mixture", False)

        if is_ds:
            if is_mix and val.ndim >= 2:
                # Shape (K, D, ...) — dataset axis is 1.
                # Index axis 1 then move component axis after batch so the
                # result is (batch, K, ...) for MixtureSameFamily compat.
                result = jnp.take(val, dataset_indices, axis=1)  # (K, batch, ...)
                result = jnp.moveaxis(result, 0, 1)  # (batch, K, ...)
                indexed[name] = result
            else:
                # Shape (D, ...) — dataset axis is 0
                indexed[name] = val[dataset_indices]
        elif spec is None and val.ndim >= 1 and val.shape[0] == n_datasets:
            # Legacy fallback for params without specs
            indexed[name] = val[dataset_indices]
        else:
            indexed[name] = val
    return indexed


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
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
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
        annotation_prior_logits : jnp.ndarray, optional
            Per-cell logit offsets for mixture component assignment priors,
            shape ``(n_cells, n_components)``.  When provided for a mixture
            model, the global ``mixing_weights`` are combined with these
            logits inside the cell plate to produce cell-specific mixing
            probabilities via :func:`compute_cell_specific_mixing`.
            If ``None``, the global mixing weights are used for all cells
            (standard behaviour).

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

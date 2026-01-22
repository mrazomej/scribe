"""Standard Negative Binomial likelihood for count data.

This module provides the basic Negative Binomial likelihood without
zero-inflation or variable capture probability.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import Likelihood
from ...builders.parameter_specs import sample_prior

if TYPE_CHECKING:
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig

# ==============================================================================
# Negative Binomial Likelihood
# ==============================================================================


class NegativeBinomialLikelihood(Likelihood):
    """
    Standard Negative Binomial likelihood for UMI count data.

    Expects param_values to contain 'p' and 'r' (or derived equivalents).

    The Negative Binomial distribution is parameterized as:

        counts ~ NegativeBinomialProbs(r, p)

    where:
        - r > 0 is the dispersion parameter (number of failures)
        - p ∈ (0, 1) is the success probability

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = NegativeBinomialLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
    ) -> None:
        """Sample from Negative Binomial likelihood.

        Handles three plate modes:
        - Prior predictive: sample counts from prior
        - Full: condition on all counts
        - Batched: condition on mini-batch with subsampling
        """
        n_cells = dims["n_cells"]
        p = param_values["p"]
        r = param_values["r"]

        # ====================================================================
        # Check if this is a mixture model
        # If r has shape (n_components, n_genes), we're in mixture mode
        # ====================================================================
        is_mixture = r.ndim == 2  # (n_components, n_genes) vs (n_genes,)

        if is_mixture:
            # ================================================================
            # Mixture model: use MixtureSameFamily
            # ================================================================
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Broadcast p to match r shape if needed
            # p can be scalar (shared) or (n_components,) (component-specific)
            if p.ndim == 0:
                # Shared p: broadcast to (n_components, 1) for broadcasting
                p = p[None, None]
            elif p.ndim == 1:
                # Component-specific p: reshape to (n_components, 1)
                p = p[:, None]
            # p already (n_components, 1) or (n_components, n_genes)

            # Base distribution for each component
            base_dist_component = dist.NegativeBinomialProbs(r, p).to_event(1)
            base_dist = dist.MixtureSameFamily(mixing_dist, base_dist_component)
        else:
            # ================================================================
            # Single-component model: standard distribution
            # ================================================================
            base_dist = dist.NegativeBinomialProbs(r, p).to_event(1)

        # ====================================================================
        # MODE 1: Prior predictive (counts=None)
        # Sample synthetic counts from the prior - no conditioning
        # Used for prior predictive checks and synthetic data generation
        # ====================================================================
        if counts is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific params if any (e.g., p_capture for VCP)
                # Note: For standard NB, cell_specs is typically empty
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # Sample counts from prior
                numpyro.sample("counts", base_dist)

        # ====================================================================
        # MODE 2: Full sampling (counts provided, no batch_size)
        # Condition on all cells at once - used for MCMC or small datasets
        # ====================================================================
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample cell-specific params if any
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # Condition on observed counts
                numpyro.sample("counts", base_dist, obs=counts)

        # ====================================================================
        # MODE 3: Batch sampling (counts provided, with batch_size)
        # Subsample cells for stochastic VI on large datasets
        # The plate returns indices for the current mini-batch
        # ====================================================================
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Sample cell-specific params if any
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # Condition on subsampled counts
                numpyro.sample("counts", base_dist, obs=counts[idx])

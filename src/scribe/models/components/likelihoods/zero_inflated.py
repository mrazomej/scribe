"""Zero-Inflated Negative Binomial likelihood for count data.

This module provides the ZINB likelihood which models both biological
zeros (from the NB distribution) and structural/technical zeros
(from the zero-inflation component).
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
# Zero-Inflated Negative Binomial Likelihood
# ==============================================================================


class ZeroInflatedNBLikelihood(Likelihood):
    """
    Zero-Inflated Negative Binomial likelihood for count data.

    Expects param_values to contain 'p', 'r', and 'gate'.

    The Zero-Inflated Negative Binomial is a mixture model:

        counts ~ ZeroInflatedNegativeBinomial(gate, r, p)

    where:
        - gate ∈ (0, 1) is the zero-inflation probability per gene
        - r > 0 is the dispersion parameter
        - p ∈ (0, 1) is the success probability

    With probability `gate`, the count is zero (structural zero).
    With probability `1 - gate`, the count follows NegativeBinomialProbs(r, p).

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = ZeroInflatedNBLikelihood()
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
        """Sample from Zero-Inflated Negative Binomial likelihood."""
        n_cells = dims["n_cells"]
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        # ====================================================================
        # Check if this is a mixture model
        # ====================================================================
        is_mixture = r.ndim == 2  # (n_components, n_genes) vs (n_genes,)

        if is_mixture:
            # ================================================================
            # Mixture model: use MixtureSameFamily
            # ================================================================
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Broadcast p to match r shape if needed
            if p.ndim == 0:
                p = p[None, None]
            elif p.ndim == 1:
                p = p[:, None]

            # Broadcast gate to match r shape if needed
            if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
                # gate is (n_genes,) - broadcast to (n_components, n_genes)
                gate = gate[None, :]
            elif gate.ndim == 2:
                # gate is already (n_components, n_genes)
                pass
            else:
                # gate is scalar - broadcast
                gate = gate[None, None]

            # Base distribution for each component
            base_nb = dist.NegativeBinomialProbs(r, p)
            zinb_base = dist.ZeroInflatedDistribution(
                base_nb, gate=gate
            ).to_event(1)
            base_dist = dist.MixtureSameFamily(mixing_dist, zinb_base)
        else:
            # ================================================================
            # Single-component model: standard distribution
            # ================================================================
            base_nb = dist.NegativeBinomialProbs(r, p)
            base_dist = dist.ZeroInflatedDistribution(
                base_nb, gate=gate
            ).to_event(1)

        # ====================================================================
        # MODE 1: Prior predictive
        # ====================================================================
        if counts is None:
            with numpyro.plate("cells", n_cells):
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", base_dist)

        # ====================================================================
        # MODE 2: Full sampling
        # ====================================================================
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", base_dist, obs=counts)

        # ====================================================================
        # MODE 3: Batch sampling
        # ====================================================================
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", base_dist, obs=counts[idx])

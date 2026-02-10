"""Zero-Inflated Negative Binomial likelihood for count data.

This module provides the ZINB likelihood which models both biological
zeros (from the NB distribution) and structural/technical zeros
(from the zero-inflation component).
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

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

    def _build_dist(
        self, param_values: Dict[str, jnp.ndarray]
    ) -> dist.Distribution:
        """Build the ZINB distribution from current param_values."""
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        # ====================================================================
        # Check if this is a mixture model
        # ====================================================================
        is_mixture = "mixing_weights" in param_values

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
            return dist.MixtureSameFamily(mixing_dist, zinb_base)

        base_nb = dist.NegativeBinomialProbs(r, p)
        return dist.ZeroInflatedDistribution(base_nb, gate=gate).to_event(1)

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
        """Sample from Zero-Inflated Negative Binomial likelihood.

        Handles three plate modes:
        - Prior predictive: sample counts from prior
        - Full: condition on all counts
        - Batched: condition on mini-batch with subsampling
        """
        n_cells = dims["n_cells"]

        # ====================================================================
        # Non-VAE fast path: build distribution once outside the plate
        # ====================================================================
        # === Non-VAE fast path: All parameter values available, build
        # distribution outside plate ===
        if vae_cell_fn is None:
            # Build the entire likelihood distribution once using the provided
            # parameters
            base_dist = self._build_dist(param_values)

            if counts is None:
                # Prior predictive path: no observed counts, draw samples using
                # base_dist
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist)
            elif batch_size is None:
                # Observed counts, no batching: condition on full observed data
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist, obs=counts)
            else:
                # Observed counts, with mini-batching/subsampling
                with numpyro.plate(
                    "cells", n_cells, subsample_size=batch_size
                ) as idx:
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist, obs=counts[idx])
            return

        # === VAE path: Decoder and prior logic run inside plate, distribution
        # built per cell/batch === The vae_cell_fn callback is used to run
        # decoder/prior logic on the fly within the plate
        if counts is None:
            # Prior predictive, run decoder per cell before sampling
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", self._build_dist(param_values))
        elif batch_size is None:
            # Observed counts, no batching: decoder runs per cell, then sample
            # with observed counts
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts", self._build_dist(param_values), obs=counts
                )
        else:
            # Observed counts, with batching/subsampling: decoder runs only for
            # the subsampled indices
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                param_values.update(vae_cell_fn(idx))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts", self._build_dist(param_values), obs=counts[idx]
                )

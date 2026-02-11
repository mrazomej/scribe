"""Zero-Inflated Negative Binomial likelihood for count data.

This module provides the ZINB likelihood which models both biological
zeros (from the NB distribution) and structural/technical zeros
(from the zero-inflation component).
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import Likelihood, compute_cell_specific_mixing
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

    # --------------------------------------------------------------------------

    def _build_annotated_mixture_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        annotation_logits_batch: jnp.ndarray,
    ) -> dist.Distribution:
        """
        Build a mixture ZINB distribution with cell-specific mixing weights.

        Parameters
        ----------
        param_values : Dict[str, jnp.ndarray]
            Sampled parameter values including ``mixing_weights``, ``p``,
            ``r``, and ``gate``.
        annotation_logits_batch : jnp.ndarray, shape ``(batch, K)``
            Per-cell annotation logit offsets for the current batch.

        Returns
        -------
        dist.Distribution
            A ``MixtureSameFamily`` distribution with cell-specific
            ``Categorical`` mixing.
        """
        mixing_weights = param_values["mixing_weights"]
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        # Cell-specific mixing via logit nudging
        cell_mixing = compute_cell_specific_mixing(
            mixing_weights, annotation_logits_batch
        )
        mixing_dist = dist.Categorical(probs=cell_mixing)

        # Broadcast p to match r shape if needed
        if p.ndim == 0:
            p = p[None, None]
        elif p.ndim == 1:
            p = p[:, None]

        # Broadcast gate to match r shape if needed
        if gate.ndim == 1 and gate.shape[0] == r.shape[1]:
            gate = gate[None, :]
        elif gate.ndim == 2:
            pass
        else:
            gate = gate[None, None]

        base_nb = dist.NegativeBinomialProbs(r, p)
        zinb_base = dist.ZeroInflatedDistribution(base_nb, gate=gate).to_event(
            1
        )
        return dist.MixtureSameFamily(mixing_dist, zinb_base)

    # --------------------------------------------------------------------------

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
    ) -> None:
        """Sample from Zero-Inflated Negative Binomial likelihood.

        Handles three plate modes:
        - Prior predictive: sample counts from prior
        - Full: condition on all counts
        - Batched: condition on mini-batch with subsampling

        When ``annotation_prior_logits`` is provided and this is a mixture
        model, per-cell mixing weights are computed inside the cell plate.
        """
        n_cells = dims["n_cells"]

        # Determine whether we need cell-specific mixing (annotation path)
        is_mixture = "mixing_weights" in param_values
        use_annotation = annotation_prior_logits is not None and is_mixture

        # ====================================================================
        # Non-VAE fast path: build distribution once outside the plate
        # ====================================================================
        if vae_cell_fn is None:

            # ----------------------------------------------------------------
            # Annotation prior path: must build dist inside the cell plate
            # ----------------------------------------------------------------
            if use_annotation:
                if counts is None:
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_dist = self._build_annotated_mixture_dist(
                            param_values, annotation_prior_logits
                        )
                        numpyro.sample("counts", cell_dist)
                elif batch_size is None:
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_dist = self._build_annotated_mixture_dist(
                            param_values, annotation_prior_logits
                        )
                        numpyro.sample("counts", cell_dist, obs=counts)
                else:
                    with numpyro.plate(
                        "cells", n_cells, subsample_size=batch_size
                    ) as idx:
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_dist = self._build_annotated_mixture_dist(
                            param_values, annotation_prior_logits[idx]
                        )
                        numpyro.sample("counts", cell_dist, obs=counts[idx])
                return

            # ----------------------------------------------------------------
            # Standard (no annotation) path: build dist once outside plate
            # ----------------------------------------------------------------
            base_dist = self._build_dist(param_values)

            if counts is None:
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist)
            elif batch_size is None:
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist, obs=counts)
            else:
                with numpyro.plate(
                    "cells", n_cells, subsample_size=batch_size
                ) as idx:
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist, obs=counts[idx])
            return

        # === VAE path: Decoder and prior logic run inside plate, distribution
        # built per cell/batch ===
        if counts is None:
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample("counts", self._build_dist(param_values))
        elif batch_size is None:
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts", self._build_dist(param_values), obs=counts
                )
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                param_values.update(vae_cell_fn(idx))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts", self._build_dist(param_values), obs=counts[idx]
                )

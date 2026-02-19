"""Standard Negative Binomial likelihood for count data.

This module provides the basic Negative Binomial likelihood without
zero-inflation or variable capture probability.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import (
    Likelihood,
    broadcast_p_for_mixture,
    compute_cell_specific_mixing,
)
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
        - p in (0, 1) is the success probability

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = NegativeBinomialLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def _build_dist(
        self, param_values: Dict[str, jnp.ndarray]
    ) -> dist.Distribution:
        """Build the NB distribution from current param_values."""
        # Extract NB parameters from param_values.
        p = param_values["p"]
        r = param_values["r"]

        # Determine if this is a mixture NB model by the shape of r.
        # r shape (n_components, n_genes) → mixture; (n_genes,) → regular NB.
        is_mixture = "mixing_weights" in param_values

        if is_mixture:
            # Mixture model: expect mixing_weights giving Categorical mixture
            # probabilities.
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Broadcast p to match r shape (n_components, n_genes).
            # Handles scalar, gene-specific, and mixture-specific p.
            p = broadcast_p_for_mixture(p, r)

            base_dist_component = dist.NegativeBinomialProbs(r, p).to_event(1)
            return dist.MixtureSameFamily(mixing_dist, base_dist_component)

        # Standard (non-mixture) Negative Binomial: return as event of size 1.
        return dist.NegativeBinomialProbs(r, p).to_event(1)

    # --------------------------------------------------------------------------

    def _build_annotated_mixture_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        annotation_logits_batch: jnp.ndarray,
    ) -> dist.Distribution:
        """
        Build a mixture NB distribution with cell-specific mixing weights.

        Parameters
        ----------
        param_values : Dict[str, jnp.ndarray]
            Sampled parameter values including ``mixing_weights``, ``p``,
            and ``r``.
        annotation_logits_batch : jnp.ndarray, shape ``(batch, K)``
            Per-cell annotation logit offsets for the current batch.

        Returns
        -------
        dist.Distribution
            A ``MixtureSameFamily`` distribution whose ``Categorical``
            mixing component has per-cell probabilities.
        """
        mixing_weights = param_values["mixing_weights"]
        p = param_values["p"]
        r = param_values["r"]

        # Cell-specific mixing via logit nudging
        cell_mixing = compute_cell_specific_mixing(
            mixing_weights, annotation_logits_batch
        )  # (batch, K)
        mixing_dist = dist.Categorical(probs=cell_mixing)

        # Broadcast p to match r shape (n_components, n_genes).
        # Handles scalar, gene-specific, and mixture-specific p.
        p = broadcast_p_for_mixture(p, r)

        base_dist_component = dist.NegativeBinomialProbs(r, p).to_event(1)
        return dist.MixtureSameFamily(mixing_dist, base_dist_component)

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
        """Sample from Negative Binomial likelihood.

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
        # Non-VAE fast path: If vae_cell_fn is None, this is not a VAE model.
        # ====================================================================
        if vae_cell_fn is None:

            # ----------------------------------------------------------------
            # Annotation prior path: must build dist inside the cell plate
            # because mixing weights are now cell-specific.
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
            # for efficiency.
            # ----------------------------------------------------------------
            base_dist = self._build_dist(param_values)

            if counts is None:
                # Prior predictive: sample counts given sampled parameters,
                # plate over all cells.
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist)
            elif batch_size is None:
                # Full dataset: observe counts array for all cells.
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

        # ====================================================================
        # VAE path: For prior predictive (counts is None), run decoder inside
        # the cell plate. The decoder produces cell-specific parameters.
        # vae_cell_fn(None) returns a dict of decoder-driven parameters for the
        # entire cell plate, which param_values is updated with. Any remaining
        # cell_specs not handled by the decoder are sampled with sample_prior.
        # Only after all parameters are set do we construct and sample the NB
        # distribution.
        # ====================================================================
        # VAE path: handle prior predictive, full data, and minibatch cases.
        if counts is None:
            # Prior predictive mode: sample all counts from prior.
            with numpyro.plate("cells", n_cells):
                # 1. Get decoder-driven (VAE) cell parameters for all cells.
                param_values.update(vae_cell_fn(None))
                # 2. Sample non-decoder cell parameters, if any.
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # 3. Sample counts using all parameters.
                numpyro.sample("counts", self._build_dist(param_values))
        elif batch_size is None:
            # Full data: observe all counts.
            with numpyro.plate("cells", n_cells):
                # 1. Update param_values with decoder-driven values (for all cells).
                param_values.update(vae_cell_fn(None))
                # 2. Sample any additional (non-decoder) cell parameters.
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # 3. Observe full counts array.
                numpyro.sample(
                    "counts", self._build_dist(param_values), obs=counts
                )
        else:
            # Minibatch/subsample: only use a subset of cells at each iteration.
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # 1. Update params using the decoder for the current minibatch indices.
                param_values.update(vae_cell_fn(idx))
                # 2. Sample any remaining (non-decoder) cell parameters.
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # 3. Observe only the minibatched counts.
                numpyro.sample(
                    "counts", self._build_dist(param_values), obs=counts[idx]
                )

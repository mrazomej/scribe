"""Standard Negative Binomial likelihood for count data.

This module provides the basic Negative Binomial likelihood without
zero-inflation or variable capture probability.  BNB support is
provided by the subclass in ``beta_negative_binomial.py``.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import (
    Likelihood,
    build_mixture_general,
    compute_cell_specific_mixing,
    index_dataset_params,
)
from ._log_prob import nb_log_prob
from ....core.axis_layout import (
    AxisLayout,
    build_param_layouts,
    broadcast_param_to_layout,
    subset_layouts,
    DATASETS,
)
from ...builders.parameter_specs import sample_prior

if TYPE_CHECKING:
    from ....core.axis_layout import AxisLayout  # noqa: F811
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig

# Minimum epsilon for clamping p away from 0 and 1 to prevent log(0) NaN
# in the NB log-probability during SVI training.  Mirrors the p_floor
# default used in post-hoc log-likelihood evaluation (log_likelihood.py).
_P_EPS = 1e-6


def _drop_dataset_axis(
    param_layouts: Optional[Dict[str, "AxisLayout"]],
) -> Optional[Dict[str, "AxisLayout"]]:
    """Return a copy of *param_layouts* with the ``"datasets"`` axis removed.

    After ``index_dataset_params`` collapses the dataset dimension,
    layouts that carried a ``"datasets"`` axis need to be updated so
    that :func:`broadcast_param_to_layout` correctly treats the new
    leading dimension as a batch (cells) dim rather than a semantic axis.

    Parameters
    ----------
    param_layouts : dict or None
        Original layouts built from ``param_specs``.

    Returns
    -------
    dict or None
        Updated layouts with ``"datasets"`` removed where present.
    """
    if param_layouts is None:
        return None
    return subset_layouts(param_layouts, DATASETS)


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
        - r > 0 is the dispersion parameter (``total_count`` in NumPyro)
        - p in (0, 1) is the ``NegativeBinomialProbs`` probability parameter

    Parameters
    ----------
    None

    Examples
    --------
    >>> likelihood = NegativeBinomialLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    # ------------------------------------------------------------------
    # Hook: subclasses override to swap the base count distribution.
    # ------------------------------------------------------------------

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        """Create the base count distribution for observed UMI counts.

        Override in subclasses (e.g. ``BetaNegativeBinomialLikelihood``)
        to replace the NB with a different distribution while keeping
        all plate / batching / mixture logic unchanged.

        Parameters
        ----------
        r : jnp.ndarray
            NB dispersion parameter (>0).
        p : jnp.ndarray
            ``NegativeBinomialProbs`` probability parameter, already clamped
            to ``(eps, 1-eps)``.

        Returns
        -------
        dist.Distribution
            A distribution over non-negative integers.
        """
        return dist.NegativeBinomialProbs(r, p)

    # ------------------------------------------------------------------

    def _build_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
    ) -> dist.Distribution:
        """Build the NB distribution from current param_values.

        Parameters
        ----------
        param_values : dict
            Sampled parameter arrays (``"p"``, ``"r"``, optionally
            ``"mixing_weights"``).
        param_layouts : dict, optional
            Semantic :class:`AxisLayout` per parameter key.  When
            provided, layout-aware broadcasting replaces shape
            heuristics in mixture models.
        """
        p = param_values["p"]
        r = param_values["r"]

        # Clamp p to (eps, 1-eps) so that log(p) and log(1-p) stay finite
        p = jnp.clip(p, _P_EPS, 1.0 - _P_EPS)

        is_mixture = "mixing_weights" in param_values

        # For non-mixture paths, when r is (n_cells, n_genes) we need to
        # distinguish whether a 1-D p vector is per-cell or per-gene:
        # - per-cell p (len == n_cells) -> (n_cells, 1)
        # - per-gene p (len == n_genes) -> (1, n_genes)
        # This avoids accidental transposition-like broadcasting errors when
        # n_cells != n_genes (e.g. validation dry runs).
        if not is_mixture and p.ndim == 1 and r.ndim == 2:
            if p.shape[0] == r.shape[0]:
                p = p[:, None]
            elif p.shape[0] == r.shape[1]:
                p = p[None, :]

        if is_mixture:
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Broadcast p to match r using semantic layouts.  The
            # broadcast_param_to_layout helper handles both the
            # non-batched case (p is (K,) or (G,)) and the batched
            # case (p is (batch, G) after dataset indexing).
            r_layout = (param_layouts or {}).get("r", AxisLayout(()))
            p_layout = (param_layouts or {}).get("p", AxisLayout(()))
            p = broadcast_param_to_layout(p, p_layout, r_layout)

            return build_mixture_general(
                mixing_dist,
                lambda comp_idx: self._make_count_dist(
                    r[..., comp_idx, :], p[..., comp_idx, :]
                ).to_event(1),
            )

        # Standard (non-mixture) path
        return self._make_count_dist(r, p).to_event(1)

    # --------------------------------------------------------------------------

    def _build_annotated_mixture_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        annotation_logits_batch: jnp.ndarray,
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
    ) -> dist.Distribution:
        """Build a mixture NB distribution with cell-specific mixing weights.

        Parameters
        ----------
        param_values : Dict[str, jnp.ndarray]
            Sampled parameter values including ``mixing_weights``, ``p``,
            and ``r``.
        annotation_logits_batch : jnp.ndarray, shape ``(batch, K)``
            Per-cell annotation logit offsets for the current batch.
        param_layouts : dict, optional
            Semantic :class:`AxisLayout` per parameter key.

        Returns
        -------
        dist.Distribution
            A ``MixtureGeneral`` distribution whose ``Categorical``
            mixing component has per-cell probabilities.
        """
        mixing_weights = param_values["mixing_weights"]
        p = param_values["p"]
        r = param_values["r"]

        # Clamp p to (eps, 1-eps) so that log(p) and log(1-p) stay finite
        p = jnp.clip(p, _P_EPS, 1.0 - _P_EPS)

        # Cell-specific mixing via logit nudging
        cell_mixing = compute_cell_specific_mixing(
            mixing_weights, annotation_logits_batch
        )  # (batch, K)
        mixing_dist = dist.Categorical(probs=cell_mixing)

        # Broadcast p to match r using semantic layouts.
        r_layout = (param_layouts or {}).get("r", AxisLayout(()))
        p_layout = (param_layouts or {}).get("p", AxisLayout(()))
        p = broadcast_param_to_layout(p, p_layout, r_layout)

        return build_mixture_general(
            mixing_dist,
            lambda comp_idx: self._make_count_dist(
                r[..., comp_idx, :], p[..., comp_idx, :]
            ).to_event(1),
        )

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
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
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

        # Use externally-provided layouts (from model builder) when
        # available.  Fall back to building from model_config.param_specs
        # for legacy callers that don't pass param_layouts.
        if param_layouts is None:
            specs = getattr(model_config, "param_specs", None) or []
            if specs:
                param_layouts = build_param_layouts(specs, param_values)

        # Multi-dataset: determine n_datasets for indexing
        n_datasets = getattr(model_config, "n_datasets", None)
        use_dataset_indexing = (
            n_datasets is not None and dataset_indices is not None
        )

        # ====================================================================
        # Non-VAE fast path: If vae_cell_fn is None, this is not a VAE model.
        # ====================================================================
        if vae_cell_fn is None:

            # ----------------------------------------------------------------
            # Annotation prior path: must build dist inside the cell plate
            # because mixing weights are now cell-specific.
            # ----------------------------------------------------------------
            if use_annotation:
                if batch_size is not None:
                    with numpyro.plate(
                        "cells", n_cells, subsample_size=batch_size
                    ) as idx:
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_dist = self._build_annotated_mixture_dist(
                            param_values,
                            annotation_prior_logits[idx],
                            param_layouts=param_layouts,
                        )
                        obs = counts[idx] if counts is not None else None
                        numpyro.sample("counts", cell_dist, obs=obs)
                elif counts is None:
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_dist = self._build_annotated_mixture_dist(
                            param_values,
                            annotation_prior_logits,
                            param_layouts=param_layouts,
                        )
                        numpyro.sample("counts", cell_dist)
                else:
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_dist = self._build_annotated_mixture_dist(
                            param_values,
                            annotation_prior_logits,
                            param_layouts=param_layouts,
                        )
                        numpyro.sample("counts", cell_dist, obs=counts)
                return

            # ----------------------------------------------------------------
            # Multi-dataset path: per-dataset params must be indexed inside
            # the cell plate so that each cell uses its dataset's parameters.
            # After index_dataset_params, the dataset axis is collapsed and
            # a leading batch (cells) dim appears.  Update layouts to drop
            # the "datasets" axis so broadcast_param_to_layout correctly
            # treats the leading dim as batch.
            # ----------------------------------------------------------------
            if use_dataset_indexing:
                # Prepare post-indexing layouts: drop the dataset axis.
                ds_layouts = _drop_dataset_axis(param_layouts)

                if batch_size is not None:
                    with numpyro.plate(
                        "cells", n_cells, subsample_size=batch_size
                    ) as idx:
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_pv = index_dataset_params(
                            param_values,
                            dataset_indices[idx],
                            n_datasets,
                            param_specs=model_config.param_specs,
                        )
                        obs = counts[idx] if counts is not None else None
                        numpyro.sample(
                            "counts",
                            self._build_dist(cell_pv, param_layouts=ds_layouts),
                            obs=obs,
                        )
                elif counts is None:
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_pv = index_dataset_params(
                            param_values,
                            dataset_indices,
                            n_datasets,
                            param_specs=model_config.param_specs,
                        )
                        numpyro.sample(
                            "counts",
                            self._build_dist(cell_pv, param_layouts=ds_layouts),
                        )
                else:
                    with numpyro.plate("cells", n_cells):
                        for spec in cell_specs:
                            sample_prior(spec, dims, model_config)
                        cell_pv = index_dataset_params(
                            param_values,
                            dataset_indices,
                            n_datasets,
                            param_specs=model_config.param_specs,
                        )
                        numpyro.sample(
                            "counts",
                            self._build_dist(cell_pv, param_layouts=ds_layouts),
                            obs=counts,
                        )
                return

            # ----------------------------------------------------------------
            # Standard (no annotation) path: build dist once outside plate
            # for efficiency.
            # ----------------------------------------------------------------
            base_dist = self._build_dist(
                param_values, param_layouts=param_layouts
            )

            # batch_size takes priority so the model plate matches
            # the guide plate during batched posterior sampling.
            if batch_size is not None:
                # Batch mode: subsample cells; obs may or may not exist.
                with numpyro.plate(
                    "cells", n_cells, subsample_size=batch_size
                ) as idx:
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    obs = counts[idx] if counts is not None else None
                    numpyro.sample("counts", base_dist, obs=obs)
            elif counts is None:
                # Prior predictive: sample counts from prior, full plate.
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist)
            else:
                # Full dataset: observe all counts, full plate.
                with numpyro.plate("cells", n_cells):
                    for spec in cell_specs:
                        sample_prior(spec, dims, model_config)
                    numpyro.sample("counts", base_dist, obs=counts)
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
        # batch_size takes priority so the model plate matches the guide
        # plate during batched posterior sampling.
        if batch_size is not None:
            # Batch mode: run decoder for the subsampled cell indices;
            # obs may or may not exist (counts=None during posterior sampling).
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # 1. Update param_values with decoder-driven values for batch.
                param_values.update(vae_cell_fn(idx))
                # 2. Sample any remaining (non-decoder) cell parameters.
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                # 3. Observe minibatched counts when available.
                obs = counts[idx] if counts is not None else None
                numpyro.sample(
                    "counts",
                    self._build_dist(param_values, param_layouts=param_layouts),
                    obs=obs,
                )
        elif counts is None:
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts",
                    self._build_dist(param_values, param_layouts=param_layouts),
                )
        else:
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts",
                    self._build_dist(param_values, param_layouts=param_layouts),
                    obs=counts,
                )

    # ------------------------------------------------------------------
    # Evaluation-side contract: delegates to the shared JIT-friendly
    # full-array implementation in ``_log_prob``.  The same method is
    # inherited unchanged by :class:`BetaNegativeBinomialLikelihood`
    # because the helper dispatches on ``params["bnb_concentration"]``.
    # ------------------------------------------------------------------

    def log_prob(
        self,
        counts: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        param_layouts: Mapping[str, "AxisLayout"],
        *,
        return_by: str = "cell",
        cells_axis: int = 0,
        r_floor: float = 1e-6,
        p_floor: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
    ) -> jnp.ndarray:
        """Log-likelihood of ``counts`` under NB / NBDM (mixture or not).

        Thin wrapper around :func:`scribe.models.components.likelihoods._log_prob.nb_log_prob`.
        See :meth:`Likelihood.log_prob` for the full parameter contract.
        """
        return nb_log_prob(
            counts,
            params,
            param_layouts,
            return_by=return_by,
            cells_axis=cells_axis,
            r_floor=r_floor,
            p_floor=p_floor,
            dtype=dtype,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
        )

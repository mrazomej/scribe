"""Zero-Inflated Negative Binomial likelihood for count data.

This module provides the ZINB likelihood which models both biological
zeros (from the NB distribution) and structural/technical zeros
(from the zero-inflation component).  BNB support is provided by the
subclass in ``beta_negative_binomial.py``.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import (
    Likelihood,
    build_mixture_general,
    compute_cell_specific_mixing,
    index_dataset_params,
)
from ....core.axis_layout import (
    AxisLayout,
    build_param_layouts,
    broadcast_param_to_layout,
    DATASETS,
)
from ...builders.parameter_specs import sample_prior

if TYPE_CHECKING:
    from ....core.axis_layout import AxisLayout  # noqa: F811
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig


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
    out: Dict[str, "AxisLayout"] = {}
    for key, layout in param_layouts.items():
        if DATASETS in layout.axes:
            out[key] = layout.subset_axis(DATASETS)
        else:
            out[key] = layout
    return out


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

    # ------------------------------------------------------------------
    # Hook: subclasses override to swap the base count distribution.
    # ------------------------------------------------------------------

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        """Create the base count distribution (before zero-inflation wrap).

        Override in subclasses (e.g. ``ZeroInflatedBNBLikelihood``) to
        replace the NB with a different distribution while keeping
        all ZI / plate / mixture logic unchanged.

        Parameters
        ----------
        r : jnp.ndarray
            NB dispersion parameter (>0).
        p : jnp.ndarray
            Failure probability, already clamped to (eps, 1-eps).

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
        """Build the ZINB distribution from current param_values.

        Parameters
        ----------
        param_values : dict
            Sampled parameter arrays (``"p"``, ``"r"``, ``"gate"``, optionally
            ``"mixing_weights"``).
        param_layouts : dict, optional
            Semantic :class:`AxisLayout` per parameter key.  When
            provided, layout-aware broadcasting replaces shape
            heuristics for ``p`` and ``gate`` in mixture models relative
            to ``r``.
        """
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        # For non-mixture paths, when r is (n_cells, n_genes) we need to
        # distinguish whether a 1-D p/gate vector is per-cell or per-gene:
        # - per-cell vector (len == n_cells) -> (n_cells, 1)
        # - per-gene vector (len == n_genes) -> (1, n_genes)
        # This keeps broadcasting correct when n_cells != n_genes.
        is_mixture = "mixing_weights" in param_values
        if not is_mixture:
            if p.ndim == 1 and r.ndim == 2:
                if p.shape[0] == r.shape[0]:
                    p = p[:, None]
                elif p.shape[0] == r.shape[1]:
                    p = p[None, :]
            if gate.ndim == 1 and r.ndim == 2:
                if gate.shape[0] == r.shape[0]:
                    gate = gate[:, None]
                elif gate.shape[0] == r.shape[1]:
                    gate = gate[None, :]

        if is_mixture:
            # ================================================================
            # Mixture model: use MixtureGeneral for NumPyro>=0.20 compatibility
            # ================================================================
            mixing_weights = param_values["mixing_weights"]
            mixing_dist = dist.Categorical(probs=mixing_weights)

            # Align p and gate to r using semantic layouts (handles batched
            # shapes after dataset indexing vs. static K×G component grids).
            r_layout = (param_layouts or {}).get("r", AxisLayout(()))
            p_layout = (param_layouts or {}).get("p", AxisLayout(()))
            gate_layout = (param_layouts or {}).get("gate", AxisLayout(()))
            p = broadcast_param_to_layout(p, p_layout, r_layout)
            gate = broadcast_param_to_layout(gate, gate_layout, r_layout)

            return build_mixture_general(
                mixing_dist,
                lambda comp_idx: dist.ZeroInflatedDistribution(
                    self._make_count_dist(
                        r[..., comp_idx, :], p[..., comp_idx, :]
                    ),
                    gate=gate[..., comp_idx, :],
                ).to_event(1),
            )

        base_nb = self._make_count_dist(r, p)
        return dist.ZeroInflatedDistribution(base_nb, gate=gate).to_event(1)

    # --------------------------------------------------------------------------

    def _build_annotated_mixture_dist(
        self,
        param_values: Dict[str, jnp.ndarray],
        annotation_logits_batch: jnp.ndarray,
        param_layouts: Optional[Dict[str, "AxisLayout"]] = None,
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
        param_layouts : dict, optional
            Semantic :class:`AxisLayout` per parameter key for layout-aware
            broadcasting of ``p`` and ``gate`` to ``r``.

        Returns
        -------
        dist.Distribution
            A ``MixtureGeneral`` distribution with cell-specific
            ``Categorical`` mixing.
        """
        mixing_weights = param_values["mixing_weights"]
        p = param_values["p"]
        r = param_values["r"]
        gate = param_values["gate"]

        cell_mixing = compute_cell_specific_mixing(
            mixing_weights, annotation_logits_batch
        )
        mixing_dist = dist.Categorical(probs=cell_mixing)

        # Align p and gate to r using semantic layouts (same as non-annotated
        # mixture path; cell_mixing only affects the Categorical, not shapes).
        r_layout = (param_layouts or {}).get("r", AxisLayout(()))
        p_layout = (param_layouts or {}).get("p", AxisLayout(()))
        gate_layout = (param_layouts or {}).get("gate", AxisLayout(()))
        p = broadcast_param_to_layout(p, p_layout, r_layout)
        gate = broadcast_param_to_layout(gate, gate_layout, r_layout)

        return build_mixture_general(
            mixing_dist,
            lambda comp_idx: dist.ZeroInflatedDistribution(
                self._make_count_dist(
                    r[..., comp_idx, :], p[..., comp_idx, :]
                ),
                gate=gate[..., comp_idx, :],
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

        # Multi-dataset: determine n_datasets for indexing
        n_datasets = getattr(model_config, "n_datasets", None)
        use_dataset_indexing = (
            n_datasets is not None and dataset_indices is not None
        )

        # Use externally-provided layouts (from model builder) when
        # available.  Fall back to building from model_config.param_specs
        # for legacy callers that don't pass param_layouts.
        if param_layouts is None:
            specs = getattr(model_config, "param_specs", None) or []
            if specs:
                param_layouts = build_param_layouts(specs, param_values)

        # ====================================================================
        # Non-VAE fast path: build distribution once outside the plate
        # ====================================================================
        if vae_cell_fn is None:

            # ----------------------------------------------------------------
            # Multi-dataset path: per-dataset params indexed inside plate
            # ----------------------------------------------------------------
            if use_dataset_indexing:
                # After index_dataset_params, the dataset axis is gone and a
                # leading batch (cells) dimension appears; drop "datasets" from
                # layouts so broadcast_param_to_layout matches array ranks.
                ds_layouts = _drop_dataset_axis(param_layouts)

                # batch_size takes priority so the model plate matches
                # the guide plate during batched posterior sampling.
                if batch_size is not None:
                    # Batch mode: subsample cells and dataset_indices
                    # together; obs may or may not exist.
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
                            self._build_dist(
                                cell_pv, param_layouts=ds_layouts
                            ),
                            obs=obs,
                        )
                elif counts is None:
                    # Prior predictive: full plate, index all cells.
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
                            self._build_dist(
                                cell_pv, param_layouts=ds_layouts
                            ),
                        )
                else:
                    # Full dataset: observe all counts, full plate.
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
                            self._build_dist(
                                cell_pv, param_layouts=ds_layouts
                            ),
                            obs=counts,
                        )
                return

            # ----------------------------------------------------------------
            # Annotation prior path: must build dist inside the cell plate
            # ----------------------------------------------------------------
            if use_annotation:
                # batch_size takes priority so the model plate matches
                # the guide plate during batched posterior sampling.
                if batch_size is not None:
                    # Batch mode: subsample cells; obs may or may not exist.
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
                    # Prior predictive: sample counts from prior, full plate.
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
                    # Full dataset: observe all counts, full plate.
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
            # Standard (no annotation) path: build dist once outside plate
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

        # === VAE path: Decoder and prior logic run inside plate, distribution
        # built per cell/batch ===
        # batch_size takes priority so the model plate matches the guide
        # plate during batched posterior sampling.
        if batch_size is not None:
            # Batch mode: run decoder for subsampled cell indices;
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
                    self._build_dist(
                        param_values, param_layouts=param_layouts
                    ),
                    obs=obs,
                )
        elif counts is None:
            # Prior predictive: run decoder for all cells, sample from prior.
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts",
                    self._build_dist(
                        param_values, param_layouts=param_layouts
                    ),
                )
        else:
            # Full data: run decoder for all cells, observe all counts.
            with numpyro.plate("cells", n_cells):
                param_values.update(vae_cell_fn(None))
                for spec in cell_specs:
                    sample_prior(spec, dims, model_config)
                numpyro.sample(
                    "counts",
                    self._build_dist(
                        param_values, param_layouts=param_layouts
                    ),
                    obs=counts,
                )

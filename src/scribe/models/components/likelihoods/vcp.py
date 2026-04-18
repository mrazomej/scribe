"""Variable Capture Probability (VCP) likelihoods for single-cell data.

This module provides likelihoods that include cell-specific capture probability
parameters, modeling technical variation in capture efficiency across cells.
BNB support is provided by the subclasses in ``beta_negative_binomial.py``.

Classes
-------
NBWithVCPLikelihood
    Negative Binomial with Variable Capture Probability.
ZINBWithVCPLikelihood
    Zero-Inflated NB with Variable Capture Probability.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import (
    Likelihood,
    build_mixture_general,
    compute_cell_specific_mixing,
    index_dataset_params,
    _sample_phi_capture_constrained,
    _sample_phi_capture_unconstrained,
    _sample_p_capture_constrained,
    _sample_p_capture_unconstrained,
    _sample_capture_biology_informed,
    _sample_hierarchical_mu_eta,
)
from ._log_prob import nbvcp_log_prob, zinbvcp_log_prob
from ....core.axis_layout import (
    AxisLayout,
    build_param_layouts,
    broadcast_param_to_layout,
    subset_layouts,
    DATASETS,
)

if TYPE_CHECKING:
    from ....core.axis_layout import AxisLayout  # noqa: F811
    from ...builders.parameter_specs import ParamSpec
    from ...config import ModelConfig


# Minimum epsilon for clamping phi away from 0 (prevents log(0) in logits
# computation) and p_hat to (eps, 1-eps) (prevents NaN in NB log-prob).
# Mirrors the p_floor default in post-hoc log-likelihood evaluation.
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
# Negative Binomial with Variable Capture Probability Likelihood
# ==============================================================================


class NBWithVCPLikelihood(Likelihood):
    """Negative Binomial with Variable Capture Probability.

    Includes cell-specific p_capture parameter that modulates
    the effective capture probability per cell.

    The effective probability becomes:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))

    where p_capture is sampled per-cell inside the cell plate.

    This models technical variation in capture efficiency across cells,
    which is common in single-cell RNA sequencing.

    Parameters
    ----------
    capture_param_name : str, optional
        Name of the capture parameter ("p_capture" or "phi_capture").
        If provided, this explicitly sets which parameterization to use.
        If None (default), the likelihood will detect it from cell_specs
        for backward compatibility.
    is_unconstrained : bool, default=False
        If True, use TransformedDistribution for capture parameter sampling.
        If False, use native constrained distribution (Beta/BetaPrime).
        This is determined at construction time to avoid runtime isinstance
        checks that can slow down JIT compilation.
    transform : Transform, optional
        The transform to use for unconstrained sampling (required if
        is_unconstrained=True).
    constrained_name : str, optional
        The name for the constrained parameter when using unconstrained
        sampling (e.g., "phi_capture" when the base is "phi_capture_unconstrained").

    Examples
    --------
    >>> # Explicit capture parameter name (recommended)
    >>> likelihood = NBWithVCPLikelihood(capture_param_name="phi_capture")
    >>>
    >>> # Auto-detect from cell_specs (backward compatible)
    >>> likelihood = NBWithVCPLikelihood()
    >>> # In model building (p_capture spec should be in cell_specs):
    >>> builder.with_likelihood(likelihood)
    """

    def __init__(
        self,
        capture_param_name: Optional[str] = None,
        is_unconstrained: bool = False,
        transform: Optional[dist.transforms.Transform] = None,
        constrained_name: Optional[str] = None,
        biology_informed_spec: Optional[object] = None,
    ):
        """Initialize likelihood with optional capture parameter name.

        Parameters
        ----------
        capture_param_name : str, optional
            "p_capture" for canonical/mean_prob parameterization,
            "phi_capture" for mean_odds parameterization.
            If None, auto-detect from cell_specs.
        is_unconstrained : bool, default=False
            If True, use TransformedDistribution for capture parameter.
        transform : Transform, optional
            Transform for unconstrained sampling.
        constrained_name : str, optional
            Name for constrained parameter in unconstrained mode.
        biology_informed_spec : BiologyInformedCaptureSpec, optional
            If provided, uses the biology-informed capture prior instead
            of the standard flat prior.
        """
        self.capture_param_name = capture_param_name
        self.is_unconstrained = is_unconstrained
        self.transform = transform
        self.constrained_name = constrained_name or capture_param_name
        self.biology_informed_spec = biology_informed_spec

    # ------------------------------------------------------------------
    # Hooks: subclasses override to swap the base count distribution.
    # ------------------------------------------------------------------

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        """Create the base count distribution from (r, probs).

        Override in BNB subclasses to return BetaNegativeBinomial.

        Parameters
        ----------
        r : jnp.ndarray
            NB dispersion (>0).
        p : jnp.ndarray
            ``NegativeBinomialProbs`` probability parameter, clamped to
            ``(eps, 1-eps)``.

        Returns
        -------
        dist.Distribution
        """
        return dist.NegativeBinomialProbs(r, p)

    def _make_count_dist_logits(
        self, r: jnp.ndarray, logits: jnp.ndarray
    ) -> dist.Distribution:
        """Create the base count distribution from (r, logits).

        The NB version uses ``NegativeBinomialLogits`` for numerical
        stability.  BNB subclasses override to convert logits to probs
        and return ``BetaNegativeBinomial``.

        Parameters
        ----------
        r : jnp.ndarray
            NB dispersion (>0).
        logits : jnp.ndarray
            Log-odds of the ``NegativeBinomialLogits`` probability parameter.

        Returns
        -------
        dist.Distribution
        """
        return dist.NegativeBinomialLogits(r, logits)

    # --------------------------------------------------------------------------

    def _sample_capture_param(
        self,
        use_phi_capture: bool,
        capture_prior_params: Tuple[float, float],
    ) -> jnp.ndarray:
        """Sample the capture parameter using pre-configured sampling strategy.

        This method uses the configuration set at construction time to avoid
        runtime isinstance checks that can slow down JIT compilation.
        """
        if use_phi_capture:
            if self.is_unconstrained and self.transform is not None:
                return _sample_phi_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_phi_capture_constrained(capture_prior_params)
        else:
            if self.is_unconstrained and self.transform is not None:
                return _sample_p_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_p_capture_constrained(capture_prior_params)

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
        """Sample from NB likelihood with variable capture probability.

        Draws per-cell capture values (unless provided in ``param_values``),
        combines them with ``p``/``phi`` and ``r`` under the cell plate, and
        emits ``counts`` from the resulting NB or mixture-of-NBs.

        Mixture models align ``p`` or ``phi`` to ``r`` using
        :class:`AxisLayout` metadata from ``model_config.param_specs`` when
        available; otherwise empty layouts fall back to prior broadcasting
        behavior.

        When ``annotation_prior_logits`` is provided and this is a mixture
        model, per-cell mixing weights are computed inside the cell plate.

        Parameters
        ----------
        param_values : dict
            Global and/or cell-level parameter arrays (``"p"`` or ``"phi"``,
            ``"r"``, optional ``"mixing_weights"``, capture keys, etc.).
        cell_specs : list of ParamSpec
            Declarations for cell-plate parameters (used for capture
            auto-detection and prior sampling).
        counts : ndarray or None
            Observed counts when conditioning; ``None`` for prior predictive.
        dims : dict
            Must include ``"n_cells"``.
        batch_size : int or None
            When set, the cell plate uses ``subsample_size=batch_size``.
        model_config : ModelConfig
            Provides ``param_specs``, ``n_datasets``, and related metadata.
        vae_cell_fn : callable or None
            When set, decoder outputs are merged into ``param_values`` inside
            the cell plate.
        annotation_prior_logits : ndarray or None
            Per-cell annotation logits for mixture annotation path.
        dataset_indices : ndarray or None
            Per-cell dataset ids when ``model_config.n_datasets`` is set.

        Notes
        -----
        After ``index_dataset_params``, dataset-axis layouts are stripped via
        :func:`_drop_dataset_axis` so :func:`broadcast_param_to_layout` matches
        collapsed array ranks.
        """
        n_cells = dims["n_cells"]

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
        ds_layouts = (
            _drop_dataset_axis(param_layouts) if use_dataset_indexing else None
        )

        # When vae_cell_fn is set, r (and possibly p) come from the decoder
        # inside the plate; do not read them here.
        if vae_cell_fn is None:
            p = param_values["p"]
            r = param_values["r"]
            is_mixture = "mixing_weights" in param_values

        # Determine which capture parameter we're using
        if self.capture_param_name is not None:
            use_phi_capture = self.capture_param_name == "phi_capture"
        else:
            # Auto-detect from cell_specs (backward compatibility)
            use_phi_capture = any(
                spec.name == "phi_capture" for spec in cell_specs
            )

        # Get prior params
        default_params = (1.0, 1.0)
        capture_prior_params = default_params

        # Try to get from param_specs if available
        target_name = "phi_capture" if use_phi_capture else "p_capture"
        for pspec in model_config.param_specs:
            if pspec.name == target_name and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # Biology-informed capture: pre-compute log library sizes and sample
        # mu_eta (data-driven) before the cell plate.
        bio_spec = self.biology_informed_spec
        bio_log_lib_sizes = None
        bio_log_M0 = None
        # Track whether mu_eta is per-dataset (needs cell-level indexing)
        _hierarchical_mu_eta = False
        if bio_spec is not None:
            if counts is not None:
                bio_log_lib_sizes = jnp.log(
                    jnp.maximum(counts.sum(axis=-1), 1.0).astype(jnp.float32)
                )
            else:
                bio_log_lib_sizes = jnp.full(n_cells, bio_spec.log_M0 - 1.0)
            if bio_spec.data_driven:
                # Hierarchical per-dataset mu_eta when D >= 2
                if n_datasets is not None and n_datasets >= 2:
                    bio_log_M0 = _sample_hierarchical_mu_eta(
                        bio_spec, n_datasets
                    )
                    _hierarchical_mu_eta = True
                else:
                    # Single-dataset fallback: scalar mu_eta
                    bio_log_M0 = numpyro.sample(
                        "mu_eta",
                        dist.Normal(bio_spec.log_M0, bio_spec.sigma_mu),
                    )
            else:
                bio_log_M0 = bio_spec.log_M0

        # Create plate context based on mode.
        # batch_size takes priority: when set, always subsample so the
        # model's cell plate matches the guide's plate (which always
        # subsamples when batch_size is provided).  This prevents shape
        # mismatches during batched posterior sampling where counts=None.
        if batch_size is not None:
            plate_context = numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            )
            obs = None  # Will be set inside plate if counts available
        elif counts is None:
            plate_context = numpyro.plate("cells", n_cells)
            obs = None
        else:
            plate_context = numpyro.plate("cells", n_cells)
            obs = counts

        with plate_context as idx:
            # Subset observations for batch mode when counts are available
            if batch_size is not None and counts is not None:
                obs = counts[idx]

            # VAE path: call decoder inside plate, then read decoder-driven
            # params
            if vae_cell_fn is not None:
                param_values.update(
                    vae_cell_fn(idx if batch_size is not None else None)
                )
                p = param_values["p"]
                r = param_values["r"]
                is_mixture = "mixing_weights" in param_values

            # Multi-dataset: index per-dataset parameters to per-cell
            if use_dataset_indexing:
                ds_idx = (
                    dataset_indices[idx] if idx is not None else dataset_indices
                )
                param_values = index_dataset_params(
                    param_values,
                    ds_idx,
                    n_datasets,
                    param_specs=model_config.param_specs,
                )
                p = param_values["p"]
                r = param_values["r"]

            # Check if capture parameter is already in param_values (from
            # posterior_samples)
            # This happens when generating PPC samples with Predictive
            if target_name in param_values:
                # Use value from posterior_samples (for PPC)
                capture_value = param_values[target_name]
                # Handle batch indexing if needed
                if batch_size is not None and idx is not None:
                    capture_value = capture_value[idx]
            elif bio_log_lib_sizes is not None and bio_log_M0 is not None:
                # Biology-informed capture: sample eta_c from library-size
                # anchored Normal prior and transform to capture parameter
                log_lib_batch = (
                    bio_log_lib_sizes[idx]
                    if idx is not None
                    else bio_log_lib_sizes
                )
                # Per-dataset mu_eta: index to per-cell log_M0
                if _hierarchical_mu_eta and use_dataset_indexing:
                    ds_idx_cap = (
                        dataset_indices[idx]
                        if idx is not None
                        else dataset_indices
                    )
                    bio_log_M0_cell = bio_log_M0[ds_idx_cap]
                else:
                    bio_log_M0_cell = bio_log_M0
                capture_value = _sample_capture_biology_informed(
                    log_lib_batch,
                    bio_log_M0_cell,
                    bio_spec.sigma_M,
                    use_phi_capture,
                )
            else:
                # Sample from prior (for prior predictive checks or when not in
                # param_values)
                capture_value = self._sample_capture_param(
                    use_phi_capture, capture_prior_params
                )

            # Determine whether to use cell-specific mixing
            use_annotation = annotation_prior_logits is not None and is_mixture

            if use_phi_capture:
                # Mean-odds parameterization
                phi = param_values["phi"]

                # Scalar-per-dataset phi becomes (n_cells,) after indexing;
                # expand to (n_cells, 1) so it broadcasts with capture and r.
                # Guard with use_dataset_indexing and a cell-axis size check so
                # gene-specific phi (n_genes,) is left alone. Only per-cell
                # vectors (n_cells,) should be expanded.
                if (
                    phi.ndim == 1
                    and not is_mixture
                    and use_dataset_indexing
                    and phi.shape[0] == capture_value.shape[0]
                ):
                    phi = phi[:, None]

                # Reshape capture for broadcasting
                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast phi to match r for mixture models using semantic
                # layouts (post-index path uses layouts with "datasets" dropped).
                if is_mixture:
                    active_layouts = (
                        ds_layouts if use_dataset_indexing else param_layouts
                    )
                    r_layout = (active_layouts or {}).get("r", AxisLayout(()))
                    phi_layout = (active_layouts or {}).get(
                        "phi", AxisLayout(())
                    )
                    phi = broadcast_param_to_layout(phi, phi_layout, r_layout)

                # Clamp phi away from 0 so log(phi * ...) stays finite
                phi = jnp.maximum(phi, _P_EPS)

                # Guardrail: catch cell-axis shape desync between NB params
                # and capture (e.g. batch_size forwarded without subsampling).
                # Only compare shape[0] when ndim matches so that mixture
                # models (where phi may lack the cell axis) don't false-alarm.
                if (
                    phi.ndim >= 2
                    and phi.ndim == capture_reshaped.ndim
                    and phi.shape[0] != capture_reshaped.shape[0]
                ):
                    raise ValueError(
                        f"Cell-axis shape desync in VCP likelihood: "
                        f"phi {phi.shape} vs capture {capture_reshaped.shape}. "
                        f"batch_size may have been forwarded without "
                        f"subsampling the cell plate."
                    )

                logits = -jnp.log(phi * (1.0 + capture_reshaped))

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    if use_annotation:
                        ann_batch = (
                            annotation_prior_logits[idx]
                            if idx is not None
                            else annotation_prior_logits
                        )
                        cell_mixing = compute_cell_specific_mixing(
                            mixing_weights, ann_batch
                        )
                        mixing_dist = dist.Categorical(probs=cell_mixing)
                    else:
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                    # Build mixture from explicit component slices to avoid
                    # NumPyro>=0.20 MixtureSameFamily support restrictions
                    # when using to_event(1) for gene-level events.
                    mixture_dist = build_mixture_general(
                        mixing_dist,
                        lambda comp_idx: self._make_count_dist_logits(
                            r[..., comp_idx, :], logits[..., comp_idx, :]
                        ).to_event(1),
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    numpyro.sample(
                        "counts",
                        self._make_count_dist_logits(r, logits).to_event(1),
                        obs=obs,
                    )
            else:
                # Canonical/mean-prob parameterization

                # Scalar-per-dataset p becomes (n_cells,) after indexing;
                # expand to (n_cells, 1) so it broadcasts with capture and r.
                # Guard with use_dataset_indexing and a cell-axis size check so
                # gene-specific p (n_genes,) is left alone. Only per-cell
                # vectors (n_cells,) should be expanded.
                if (
                    p.ndim == 1
                    and not is_mixture
                    and use_dataset_indexing
                    and p.shape[0] == capture_value.shape[0]
                ):
                    p = p[:, None]

                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast p for mixture models (handles gene-specific p) using
                # semantic layouts relative to r.
                if is_mixture:
                    active_layouts = (
                        ds_layouts if use_dataset_indexing else param_layouts
                    )
                    r_layout = (active_layouts or {}).get("r", AxisLayout(()))
                    p_layout = (active_layouts or {}).get("p", AxisLayout(()))
                    p_for_hat = broadcast_param_to_layout(p, p_layout, r_layout)
                else:
                    p_for_hat = p

                # Guardrail: catch cell-axis shape desync between NB params
                # and capture (e.g. batch_size forwarded without subsampling).
                # Only compare shape[0] when ndim matches so that mixture
                # models (where p may lack the cell axis) don't false-alarm.
                if (
                    p_for_hat.ndim >= 2
                    and p_for_hat.ndim == capture_reshaped.ndim
                    and p_for_hat.shape[0] != capture_reshaped.shape[0]
                ):
                    raise ValueError(
                        f"Cell-axis shape desync in VCP likelihood: "
                        f"p {p_for_hat.shape} vs capture "
                        f"{capture_reshaped.shape}. "
                        f"batch_size may have been forwarded without "
                        f"subsampling the cell plate."
                    )

                p_hat = (
                    p_for_hat
                    * capture_reshaped
                    / (1 - p_for_hat * (1 - capture_reshaped))
                )
                # Clamp p_hat to (eps, 1-eps) so NB log-prob stays finite
                p_hat = jnp.clip(p_hat, _P_EPS, 1.0 - _P_EPS)

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    if use_annotation:
                        ann_batch = (
                            annotation_prior_logits[idx]
                            if idx is not None
                            else annotation_prior_logits
                        )
                        cell_mixing = compute_cell_specific_mixing(
                            mixing_weights, ann_batch
                        )
                        mixing_dist = dist.Categorical(probs=cell_mixing)
                    else:
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                    # Build mixture from explicit component slices to avoid
                    # NumPyro>=0.20 MixtureSameFamily support restrictions
                    # when using to_event(1) for gene-level events.
                    mixture_dist = build_mixture_general(
                        mixing_dist,
                        lambda comp_idx: self._make_count_dist(
                            r[..., comp_idx, :], p_hat[..., comp_idx, :]
                        ).to_event(1),
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    numpyro.sample(
                        "counts",
                        self._make_count_dist(r, p_hat).to_event(1),
                        obs=obs,
                    )

    # ------------------------------------------------------------------
    # Evaluation-side contract: delegates to the shared JIT-friendly
    # full-array implementation in ``_log_prob``.  The same method is
    # inherited unchanged by :class:`BNBWithVCPLikelihood`.
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
        """Log-likelihood of ``counts`` under NBVCP / NBVCP-mixture.

        Thin wrapper around
        :func:`scribe.models.components.likelihoods._log_prob.nbvcp_log_prob`.
        See :meth:`Likelihood.log_prob` for the full parameter contract.
        """
        return nbvcp_log_prob(
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


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with Variable Capture Probability Likelihood
# ------------------------------------------------------------------------------


class ZINBWithVCPLikelihood(Likelihood):
    """Zero-Inflated Negative Binomial with Variable Capture Probability.

    Combines zero-inflation with cell-specific capture probability.

    The model is:

        p_hat = p * p_capture / (1 - p * (1 - p_capture))
        counts ~ ZeroInflatedNegativeBinomial(gate, r, p_hat)

    Parameters
    ----------
    capture_param_name : str, optional
        Name of the capture parameter ("p_capture" or "phi_capture").
        If provided, this explicitly sets which parameterization to use.
        If None (default), the likelihood will detect it from cell_specs
        for backward compatibility.
    is_unconstrained : bool, default=False
        If True, use TransformedDistribution for capture parameter sampling.
        If False, use native constrained distribution (Beta/BetaPrime).
    transform : Transform, optional
        The transform to use for unconstrained sampling.
    constrained_name : str, optional
        The name for the constrained parameter when using unconstrained sampling.

    Examples
    --------
    >>> # Explicit capture parameter name (recommended)
    >>> likelihood = ZINBWithVCPLikelihood(capture_param_name="phi_capture")
    >>>
    >>> # Auto-detect from cell_specs (backward compatible)
    >>> likelihood = ZINBWithVCPLikelihood()
    >>> # In model building:
    >>> builder.with_likelihood(likelihood)
    """

    def __init__(
        self,
        capture_param_name: Optional[str] = None,
        is_unconstrained: bool = False,
        transform: Optional[dist.transforms.Transform] = None,
        constrained_name: Optional[str] = None,
        biology_informed_spec: Optional[object] = None,
    ):
        """Initialize likelihood with optional capture parameter name.

        Parameters
        ----------
        capture_param_name : str, optional
            "p_capture" for canonical/mean_prob parameterization,
            "phi_capture" for mean_odds parameterization.
            If None, auto-detect from cell_specs.
        is_unconstrained : bool, default=False
            If True, use TransformedDistribution for capture parameter.
        transform : Transform, optional
            Transform for unconstrained sampling.
        constrained_name : str, optional
            Name for constrained parameter in unconstrained mode.
        biology_informed_spec : BiologyInformedCaptureSpec, optional
            If provided, uses the biology-informed capture prior.
        """
        self.capture_param_name = capture_param_name
        self.is_unconstrained = is_unconstrained
        self.transform = transform
        self.constrained_name = constrained_name or capture_param_name
        self.biology_informed_spec = biology_informed_spec

    # ------------------------------------------------------------------
    # Hooks: subclasses override to swap the base count distribution.
    # ------------------------------------------------------------------

    def _make_count_dist(
        self, r: jnp.ndarray, p: jnp.ndarray
    ) -> dist.Distribution:
        """Create the base count distribution from (r, probs).

        Override in BNB subclasses to return BetaNegativeBinomial.
        """
        return dist.NegativeBinomialProbs(r, p)

    def _make_count_dist_logits(
        self, r: jnp.ndarray, logits: jnp.ndarray
    ) -> dist.Distribution:
        """Create the base count distribution from (r, logits).

        NB uses ``NegativeBinomialLogits``; BNB overrides to convert
        logits to probs.
        """
        return dist.NegativeBinomialLogits(r, logits)

    # --------------------------------------------------------------------------

    def _sample_capture_param(
        self,
        use_phi_capture: bool,
        capture_prior_params: Tuple[float, float],
    ) -> jnp.ndarray:
        """Sample the capture parameter using pre-configured sampling strategy."""
        if use_phi_capture:
            if self.is_unconstrained and self.transform is not None:
                return _sample_phi_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_phi_capture_constrained(capture_prior_params)
        else:
            if self.is_unconstrained and self.transform is not None:
                return _sample_p_capture_unconstrained(
                    capture_prior_params, self.transform, self.constrained_name
                )
            else:
                return _sample_p_capture_constrained(capture_prior_params)

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
        """Sample from ZINB likelihood with variable capture probability.

        Draws per-cell capture values when needed, applies the VCP link to
        obtain ``p_hat`` (canonical path) or logits (``phi`` path),         wraps the base NB in ``ZeroInflatedDistribution``, and samples
        ``counts`` under the cell plate.

        For mixture models, ``p``, ``phi``, and ``gate`` are aligned to ``r``
        via :func:`broadcast_param_to_layout` when ``param_specs`` supply
        :class:`AxisLayout` metadata; after multi-dataset indexing, the
        ``"datasets"`` axis is removed from those layouts first.

        When ``annotation_prior_logits`` is provided and this is a mixture
        model, per-cell mixing weights are computed inside the cell plate.

        Parameters
        ----------
        param_values : dict
            Arrays for ``"p"`` or ``"phi"``, ``"r"``, ``"gate"``, optional
            ``"mixing_weights"``, capture parameters, etc.
        cell_specs : list of ParamSpec
            Cell-plate parameter specs (capture auto-detection).
        counts : ndarray or None
            Observed counts or ``None`` for prior predictive.
        dims : dict
            Must include ``"n_cells"``.
        batch_size : int or None
            Subsample size for the cell plate when not ``None``.
        model_config : ModelConfig
            ``param_specs``, ``n_datasets``, and related configuration.
        vae_cell_fn : callable or None
            Decoder hook that updates ``param_values`` inside the plate.
        annotation_prior_logits : ndarray or None
            Annotation logits for the mixture annotation path.
        dataset_indices : ndarray or None
            Per-cell dataset indices for multi-dataset models.

        Notes
        -----
        Layout-aware broadcasting mirrors
        :class:`ZeroInflatedNBLikelihood` while preserving VCP-specific
        capture reshaping and clamping.
        """
        n_cells = dims["n_cells"]

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
        ds_layouts = (
            _drop_dataset_axis(param_layouts) if use_dataset_indexing else None
        )

        # When vae_cell_fn is set, r/gate (and possibly p) come from the decoder
        # inside the plate; do not read them here.
        if vae_cell_fn is None:
            p = param_values["p"]
            r = param_values["r"]
            gate = param_values["gate"]
            is_mixture = "mixing_weights" in param_values

        # Determine which capture parameter we're using
        if self.capture_param_name is not None:
            use_phi_capture = self.capture_param_name == "phi_capture"
        else:
            use_phi_capture = any(
                spec.name == "phi_capture" for spec in cell_specs
            )

        # Get prior params
        default_params = (1.0, 1.0)
        capture_prior_params = default_params

        target_name = "phi_capture" if use_phi_capture else "p_capture"
        for pspec in model_config.param_specs:
            if pspec.name == target_name and pspec.prior is not None:
                capture_prior_params = pspec.prior
                break

        # Biology-informed capture: pre-compute log library sizes and sample
        # mu_eta (data-driven) before the cell plate.
        bio_spec = self.biology_informed_spec
        bio_log_lib_sizes = None
        bio_log_M0 = None
        _hierarchical_mu_eta = False
        if bio_spec is not None:
            if counts is not None:
                bio_log_lib_sizes = jnp.log(
                    jnp.maximum(counts.sum(axis=-1), 1.0).astype(jnp.float32)
                )
            else:
                bio_log_lib_sizes = jnp.full(n_cells, bio_spec.log_M0 - 1.0)
            if bio_spec.data_driven:
                # Hierarchical per-dataset mu_eta when D >= 2
                if n_datasets is not None and n_datasets >= 2:
                    bio_log_M0 = _sample_hierarchical_mu_eta(
                        bio_spec, n_datasets
                    )
                    _hierarchical_mu_eta = True
                else:
                    # Single-dataset fallback: scalar mu_eta
                    bio_log_M0 = numpyro.sample(
                        "mu_eta",
                        dist.Normal(bio_spec.log_M0, bio_spec.sigma_mu),
                    )
            else:
                bio_log_M0 = bio_spec.log_M0

        # Create plate context based on mode.
        # batch_size takes priority: when set, always subsample so the
        # model's cell plate matches the guide's plate (which always
        # subsamples when batch_size is provided).  This prevents shape
        # mismatches during batched posterior sampling where counts=None.
        if batch_size is not None:
            plate_context = numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            )
            obs = None  # Will be set inside plate if counts available
        elif counts is None:
            plate_context = numpyro.plate("cells", n_cells)
            obs = None
        else:
            plate_context = numpyro.plate("cells", n_cells)
            obs = counts

        with plate_context as idx:
            # Subset observations for batch mode when counts are available
            if batch_size is not None and counts is not None:
                obs = counts[idx]

            # VAE path: call decoder inside plate, re-read updated params
            if vae_cell_fn is not None:
                param_values.update(
                    vae_cell_fn(idx if batch_size is not None else None)
                )
                p = param_values["p"]
                r = param_values["r"]
                gate = param_values["gate"]
                is_mixture = "mixing_weights" in param_values

            # Multi-dataset: index per-dataset parameters to per-cell
            if use_dataset_indexing:
                ds_idx = (
                    dataset_indices[idx] if idx is not None else dataset_indices
                )
                param_values = index_dataset_params(
                    param_values,
                    ds_idx,
                    n_datasets,
                    param_specs=model_config.param_specs,
                )
                p = param_values["p"]
                r = param_values["r"]
                gate = param_values["gate"]

            # Check if capture parameter is already in param_values
            if target_name in param_values:
                # Use value from posterior_samples (for PPC)
                capture_value = param_values[target_name]
                # Handle batch indexing if needed
                if batch_size is not None and idx is not None:
                    capture_value = capture_value[idx]
            elif bio_log_lib_sizes is not None and bio_log_M0 is not None:
                # Biology-informed capture sampling
                log_lib_batch = (
                    bio_log_lib_sizes[idx]
                    if idx is not None
                    else bio_log_lib_sizes
                )
                # Per-dataset mu_eta: index to per-cell log_M0
                if _hierarchical_mu_eta and use_dataset_indexing:
                    ds_idx_cap = (
                        dataset_indices[idx]
                        if idx is not None
                        else dataset_indices
                    )
                    bio_log_M0_cell = bio_log_M0[ds_idx_cap]
                else:
                    bio_log_M0_cell = bio_log_M0
                capture_value = _sample_capture_biology_informed(
                    log_lib_batch,
                    bio_log_M0_cell,
                    bio_spec.sigma_M,
                    use_phi_capture,
                )
            else:
                # Sample from prior (for prior predictive checks or when not in
                # param_values)
                capture_value = self._sample_capture_param(
                    use_phi_capture, capture_prior_params
                )

            # Determine whether to use cell-specific mixing
            use_annotation = annotation_prior_logits is not None and is_mixture

            if use_phi_capture:
                # Mean-odds parameterization
                phi = param_values["phi"]

                # Scalar-per-dataset phi becomes (n_cells,) after indexing;
                # expand to (n_cells, 1) so it broadcasts with capture and r.
                # Guard with use_dataset_indexing and a cell-axis size check so
                # gene-specific phi (n_genes,) is left alone. Only per-cell
                # vectors (n_cells,) should be expanded.
                if (
                    phi.ndim == 1
                    and not is_mixture
                    and use_dataset_indexing
                    and phi.shape[0] == capture_value.shape[0]
                ):
                    phi = phi[:, None]

                # Reshape capture for broadcasting
                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast phi and gate to r for mixture models using semantic
                # layouts (datasets axis stripped after per-cell indexing).
                if is_mixture:
                    active_layouts = (
                        ds_layouts if use_dataset_indexing else param_layouts
                    )
                    r_layout = (active_layouts or {}).get("r", AxisLayout(()))
                    phi_layout = (active_layouts or {}).get(
                        "phi", AxisLayout(())
                    )
                    gate_layout = (active_layouts or {}).get(
                        "gate", AxisLayout(())
                    )
                    phi = broadcast_param_to_layout(phi, phi_layout, r_layout)
                    gate = broadcast_param_to_layout(
                        gate, gate_layout, r_layout
                    )

                # Clamp phi away from 0 so log(phi * ...) stays finite
                phi = jnp.maximum(phi, _P_EPS)

                # Guardrail: catch cell-axis shape desync between NB params
                # and capture (e.g. batch_size forwarded without subsampling).
                # Only compare shape[0] when ndim matches so that mixture
                # models (where phi may lack the cell axis) don't false-alarm.
                if (
                    phi.ndim >= 2
                    and phi.ndim == capture_reshaped.ndim
                    and phi.shape[0] != capture_reshaped.shape[0]
                ):
                    raise ValueError(
                        f"Cell-axis shape desync in ZINB-VCP likelihood: "
                        f"phi {phi.shape} vs capture {capture_reshaped.shape}. "
                        f"batch_size may have been forwarded without "
                        f"subsampling the cell plate."
                    )

                logits = -jnp.log(phi * (1.0 + capture_reshaped))

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    if use_annotation:
                        ann_batch = (
                            annotation_prior_logits[idx]
                            if idx is not None
                            else annotation_prior_logits
                        )
                        cell_mixing = compute_cell_specific_mixing(
                            mixing_weights, ann_batch
                        )
                        mixing_dist = dist.Categorical(probs=cell_mixing)
                    else:
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                    # Build mixture from explicit component slices to avoid
                    # NumPyro>=0.20 MixtureSameFamily support restrictions
                    # when using to_event(1) for gene-level events.
                    mixture_dist = build_mixture_general(
                        mixing_dist,
                        lambda comp_idx: dist.ZeroInflatedDistribution(
                            self._make_count_dist_logits(
                                r[..., comp_idx, :], logits[..., comp_idx, :]
                            ),
                            gate=gate[..., comp_idx, :],
                        ).to_event(1),
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    base_nb = self._make_count_dist_logits(r, logits)
                    zinb_dist = dist.ZeroInflatedDistribution(
                        base_nb, gate=gate
                    )
                    numpyro.sample("counts", zinb_dist.to_event(1), obs=obs)
            else:
                # Canonical/mean-prob parameterization

                # Scalar-per-dataset p becomes (n_cells,) after indexing;
                # expand to (n_cells, 1) so it broadcasts with capture and r.
                # Guard with use_dataset_indexing and a cell-axis size check so
                # gene-specific p (n_genes,) is left alone. Only per-cell
                # vectors (n_cells,) should be expanded.
                if (
                    p.ndim == 1
                    and not is_mixture
                    and use_dataset_indexing
                    and p.shape[0] == capture_value.shape[0]
                ):
                    p = p[:, None]

                if is_mixture:
                    capture_reshaped = capture_value[:, None, None]
                else:
                    capture_reshaped = capture_value[:, None]

                # Broadcast p and gate to r for mixture models using semantic
                # layouts.
                if is_mixture:
                    active_layouts = (
                        ds_layouts if use_dataset_indexing else param_layouts
                    )
                    r_layout = (active_layouts or {}).get("r", AxisLayout(()))
                    p_layout = (active_layouts or {}).get("p", AxisLayout(()))
                    gate_layout = (active_layouts or {}).get(
                        "gate", AxisLayout(())
                    )
                    p_for_hat = broadcast_param_to_layout(p, p_layout, r_layout)
                    gate = broadcast_param_to_layout(
                        gate, gate_layout, r_layout
                    )
                else:
                    p_for_hat = p

                # Guardrail: catch cell-axis shape desync between NB params
                # and capture (e.g. batch_size forwarded without subsampling).
                # Only compare shape[0] when ndim matches so that mixture
                # models (where p may lack the cell axis) don't false-alarm.
                if (
                    p_for_hat.ndim >= 2
                    and p_for_hat.ndim == capture_reshaped.ndim
                    and p_for_hat.shape[0] != capture_reshaped.shape[0]
                ):
                    raise ValueError(
                        f"Cell-axis shape desync in ZINB-VCP likelihood: "
                        f"p {p_for_hat.shape} vs capture "
                        f"{capture_reshaped.shape}. "
                        f"batch_size may have been forwarded without "
                        f"subsampling the cell plate."
                    )

                p_hat = (
                    p_for_hat
                    * capture_reshaped
                    / (1 - p_for_hat * (1 - capture_reshaped))
                )
                # Clamp p_hat to (eps, 1-eps) so NB log-prob stays finite
                p_hat = jnp.clip(p_hat, _P_EPS, 1.0 - _P_EPS)

                if is_mixture:
                    mixing_weights = param_values["mixing_weights"]
                    if use_annotation:
                        ann_batch = (
                            annotation_prior_logits[idx]
                            if idx is not None
                            else annotation_prior_logits
                        )
                        cell_mixing = compute_cell_specific_mixing(
                            mixing_weights, ann_batch
                        )
                        mixing_dist = dist.Categorical(probs=cell_mixing)
                    else:
                        mixing_dist = dist.Categorical(probs=mixing_weights)
                    # Build mixture from explicit component slices to avoid
                    # NumPyro>=0.20 MixtureSameFamily support restrictions
                    # when using to_event(1) for gene-level events.
                    mixture_dist = build_mixture_general(
                        mixing_dist,
                        lambda comp_idx: dist.ZeroInflatedDistribution(
                            self._make_count_dist(
                                r[..., comp_idx, :], p_hat[..., comp_idx, :]
                            ),
                            gate=gate[..., comp_idx, :],
                        ).to_event(1),
                    )
                    numpyro.sample("counts", mixture_dist, obs=obs)
                else:
                    base_nb = self._make_count_dist(r, p_hat)
                    zinb_dist = dist.ZeroInflatedDistribution(
                        base_nb, gate=gate
                    )
                    numpyro.sample("counts", zinb_dist.to_event(1), obs=obs)

    # ------------------------------------------------------------------
    # Evaluation-side contract: delegates to the shared JIT-friendly
    # full-array implementation in ``_log_prob``.  The same method is
    # inherited unchanged by :class:`ZIBNBWithVCPLikelihood`.
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
        """Log-likelihood of ``counts`` under ZINBVCP / ZINBVCP-mixture.

        Thin wrapper around
        :func:`scribe.models.components.likelihoods._log_prob.zinbvcp_log_prob`.
        See :meth:`Likelihood.log_prob` for the full parameter contract.
        """
        return zinbvcp_log_prob(
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

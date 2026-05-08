"""Negative Binomial-LogNormal (NB-LogNormal) likelihood for absolute count data.

This module implements the observation model where each gene's count is an
independent Negative Binomial draw whose log-mean is offset by a correlated
log-normal latent:

    z^(c) ~ N(0, Sigma)                     [latent G-dim regulatory state]
    u_g^(c) | z_g^(c) ~ NB(mu_g exp(z_g),    [per-gene NB count]
                            r_g)

The Sigma is a low-rank-plus-diagonal global covariance shared across cells
(``Sigma = W W^T + diag(d)``).  The dispersion ``r_g`` is gene-specific and
is *independent* of ``z`` — this is the structural choice that makes the
inner Gamma layer marginalize cleanly via Gamma-Poisson conjugacy and gives
the resulting log-likelihood its concavity in ``z``.

Compared to the Poisson-LogNormal (``PoissonLogNormalLikelihood``), this
likelihood:

1. Recovers the bursty Negative Binomial steady-state at the inner layer
   (exact in the bursty limit of the chemical master equation), preserving
   the spike of probability mass at zero that PLN structurally cannot match
   for low-mean genes.

2. Approaches PLN in the limit ``r_g -> infinity`` (Gamma collapses to a
   point mass on its mean, NB collapses to Poisson).

3. Strictly generalizes the shared-``p`` NBDM model: setting
   ``Sigma -> 0`` (so ``z = 0`` for all cells) recovers
   ``u_cg ~ NB(mu_cg, r_g)`` independently across genes.

The low-rank regime (``d_mode="low_rank"``) uses ``y = mu + W z`` with no
per-coordinate residual noise.  The learned-diagonal regime
(``d_mode="learned"``) adds ``sqrt(d) * epsilon`` with
``epsilon ~ Normal(0, I)``, using a global positive vector ``d`` (per
gene, G-dimensional).

Optional capture probability is supported via a per-cell additive offset
in log-rate space: ``effective_log_rate = y_log_rate - eta``, where
``eta ~ TruncatedNormal(log M_0 - log L_c, sigma_M^2, low=0)``.  The
capture latent decouples experimental undercounting from biological
correlation, so that ``z`` retains its interpretation as a regulatory
state rather than a junk drawer for technical noise.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .base import Likelihood, _sample_capture_biology_informed
from ._log_prob import _check_return_by, _normalize_counts, _require_layout
from ....core.axis_layout import AxisLayout
from ....stats.distributions import LogMeanNegativeBinomial
from ...builders.parameter_specs import sample_prior

if TYPE_CHECKING:
    from ...builders.parameter_specs import BiologyInformedCaptureSpec, ParamSpec
    from ...config import ModelConfig

# Floor for diagonal ``d`` in ``learned`` mode when mapping to ``sqrt(d)``.
_D_EPS = 1e-8

# Floor for gene dispersion ``r_g`` to avoid degenerate NB log-prob.
_R_EPS = 1e-6

# Sanity bounds for y_log_rate.  Unlike the PLN likelihood, the NBLN
# log-prob path does *not* materialize ``exp(y_log_rate)`` -- it goes
# through ``LogMeanNegativeBinomial``, which evaluates the NB log-prob
# entirely in log-space via ``softplus(log_mean - log r)``.  These
# bounds therefore serve only to guard against truly non-finite inputs
# (NaN / +/-inf), never to bound the model.  We pick a wide window:
# exp(50) ~ 5e21 is far beyond any biological count, exp(-50) ~ 2e-22
# is well below sub-Poisson noise; under valid inference, y_log_rate
# never approaches these.
_LOG_RATE_MIN = -50.0
_LOG_RATE_MAX = 50.0


class NBLogNormalLikelihood(Likelihood):
    """Negative Binomial-LogNormal likelihood.

    Each gene's count is ``NB(mean = exp(y_log_rate_g), concentration = r_g)``
    where ``y_log_rate`` is the G-dimensional output of a linear-decoder VAE
    and ``r_g`` is a gene-specific global dispersion parameter sampled
    outside the cell plate.

    Optionally, a per-cell capture offset ``eta`` is subtracted from the
    log-rate before exponentiation.

    Parameters
    ----------
    d_mode : {"low_rank", "learned"}, default="low_rank"
        ``low_rank``: ``y`` equals the decoder output (singular Gaussian
        on ``y`` induced by low-rank ``W`` is acceptable because the ELBO
        never inverts the full ambient covariance).
        ``learned``: adds ``sqrt(d) * epsilon`` with IID standard normal
        ``epsilon``; KL from ``epsilon`` is zero when the guide matches
        the prior.
    d_param_name : str, default="d_nbln"
        Key in ``param_values`` for the learned positive diagonal ``d``
        (length G), used only when ``d_mode="learned"``.
    r_param_name : str, default="r"
        Key in ``param_values`` (and ``params`` in ``log_prob``) for the
        gene dispersion vector ``r_g`` (length G).
    capture_anchor : bool, default=False
        If True, sample a per-cell capture offset ``eta_capture`` and
        subtract it from the log-rate before exponentiation.
    biology_informed_spec : BiologyInformedCaptureSpec, optional
        Capture specification with (log_M0, sigma_M) for the
        biology-informed truncated normal prior on eta.  Required when
        ``capture_anchor=True``.
    """

    def __init__(
        self,
        d_mode: Literal["low_rank", "learned"] = "low_rank",
        d_param_name: str = "d_nbln",
        r_param_name: str = "r",
        capture_anchor: bool = False,
        biology_informed_spec: Optional["BiologyInformedCaptureSpec"] = None,
    ) -> None:
        if d_mode not in ("low_rank", "learned"):
            raise ValueError('d_mode must be "low_rank" or "learned"')
        if capture_anchor and biology_informed_spec is None:
            raise ValueError(
                "biology_informed_spec is required when capture_anchor=True"
            )
        self._d_mode: Literal["low_rank", "learned"] = d_mode
        self._d_param_name = d_param_name
        self._r_param_name = r_param_name
        self._capture_anchor = capture_anchor
        self._biology_informed_spec = biology_informed_spec

    @property
    def capture_anchor(self) -> bool:
        """Whether per-cell capture offset is enabled."""
        return self._capture_anchor

    @property
    def biology_informed_spec(
        self,
    ) -> Optional["BiologyInformedCaptureSpec"]:
        """Capture specification for biology-informed eta prior."""
        return self._biology_informed_spec

    def _clamp_log_rate(self, y: jnp.ndarray) -> jnp.ndarray:
        """Clamp log-rate to safe float32 range before exponentiation."""
        return jnp.clip(y, _LOG_RATE_MIN, _LOG_RATE_MAX)

    # ------------------------------------------------------------------
    # Internal: shared cell-body logic for both plate modes.
    # ------------------------------------------------------------------

    def _cell_body(
        self,
        idx: Optional[jnp.ndarray],
        counts: Optional[jnp.ndarray],
        n_genes: int,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        dims: Dict[str, int],
        model_config: "ModelConfig",
        vae_cell_fn: Callable[
            [Optional[jnp.ndarray]], Dict[str, jnp.ndarray]
        ],
    ) -> None:
        """Shared generative logic executed inside the cell plate.

        Parameters
        ----------
        idx : jnp.ndarray or None
            Cell indices for mini-batch mode; None for full-plate mode.
        counts : jnp.ndarray or None
            Full count matrix (subsetted by idx inside this method).
        n_genes : int
            Number of genes (G).
        param_values : dict
            Mutable dict of sampled parameter values.  Must contain the
            gene dispersion under ``self._r_param_name`` (sampled outside
            the cell plate as a global parameter).
        cell_specs : list
            Specs for cell-plate priors.
        dims : dict
            Dimension sizes.
        model_config : ModelConfig
            Model configuration.
        vae_cell_fn : callable
            Decoder function producing y_log_rate inside the plate.
        """
        # (a) Run the decoder to get y_log_rate. This samples ``z``
        # inside the VAE encoder/decoder and must come *before* the
        # capture-anchor sampling so that trace site ordering (z then
        # eta_capture) matches the guide's ordering and satisfies
        # ``TraceMeanField_ELBO``'s mean-field check.
        param_values.update(vae_cell_fn(idx))
        if "y_log_rate" not in param_values:
            raise ValueError(
                "vae_cell_fn must return a 'y_log_rate' tensor "
                "(decoder log-rate head), shape (batch, G)."
            )
        y_decoded = param_values["y_log_rate"]

        # (b) Sample any additional cell-plate priors.
        for spec in cell_specs:
            sample_prior(spec, dims, model_config)

        # (c) Optionally sample per-cell capture offset.
        # In NBLN, capture is a per-cell additive offset in log-rate
        # space (same convention as PLN):
        #
        #   effective_log_rate = y_log_rate - eta
        #
        # where ``eta = -log(p_capture)`` and
        # ``p_capture = exp(-eta) in (0, 1)``. The helper
        # ``_sample_capture_biology_informed`` (shared with the LNMVCP
        # and PLN paths) handles the actual ``eta_capture`` site
        # sampling under a biology-informed Truncated-Normal prior
        # anchored to each cell's observed library size.
        capture_offset = None
        if (
            self._capture_anchor
            and self._biology_informed_spec is not None
            and counts is not None
        ):
            # Only execute the capture path when ``counts`` is observed:
            # the biology anchor needs ``log L_c`` per cell, which is
            # meaningless in prior-predictive simulation (counts is None).
            bio_spec = self._biology_informed_spec
            counts_batch = counts[idx] if idx is not None else counts
            log_lib_batch = jnp.log(
                jnp.maximum(
                    counts_batch.sum(axis=-1).astype(jnp.float32), 1.0
                )
            )

            # ``_sample_capture_biology_informed`` samples ``eta_capture``
            # from a TruncatedNormal anchored to ``log(M_0) - log(L_c)``
            # and returns ``p_capture = exp(-eta_capture)`` (in (0, 1))
            # when ``use_phi_capture=False``. We unwind it back to
            # ``eta`` for the additive log-rate offset NBLN consumes.
            p_capture = _sample_capture_biology_informed(
                log_lib_batch,
                bio_spec.log_M0,
                bio_spec.sigma_M,
                use_phi_capture=False,
            )
            capture_offset = -jnp.log(jnp.maximum(p_capture, 1e-12))

        # (d) Optional diagonal noise in log-rate space (G-dimensional).
        if self._d_mode == "learned":
            d_vec = param_values[self._d_param_name]
            d_vec = jnp.maximum(jnp.asarray(d_vec), _D_EPS)
            sigma = jnp.sqrt(d_vec)
            with numpyro.handlers.block():
                eps = numpyro.sample(
                    "nbln_eps",
                    dist.Normal(0.0, 1.0)
                    .expand([n_genes])
                    .to_event(1),
                )
            y = y_decoded + sigma * eps
        else:
            y = y_decoded

        # (e) Apply capture offset (per-cell scalar broadcast to all genes).
        if capture_offset is not None:
            y = y - capture_offset[..., None]

        # (f) Sanity-clamp the log-mean.  No exponentiation here: the
        # ``LogMeanNegativeBinomial`` distribution evaluates log-prob
        # entirely in log-space, so this clamp is purely a guard
        # against non-finite inputs and never affects the model.
        y = self._clamp_log_rate(y)

        # (g) Look up gene dispersion ``r_g`` from globals.
        if self._r_param_name not in param_values:
            raise ValueError(
                f"NBLogNormalLikelihood expected a '{self._r_param_name}' "
                "entry in param_values (gene dispersion, length G). "
                "Make sure 'r' is registered for nbln in MODEL_EXTRA_PARAMS "
                "and sampled by the model factory before the cell plate."
            )
        r_vec = jnp.maximum(
            jnp.asarray(param_values[self._r_param_name]), _R_EPS
        )

        # (h) Negative Binomial observation model over genes.
        # ``LogMeanNegativeBinomial`` consumes ``log_mean`` directly,
        # avoiding the exp/log round-trip that ``NegativeBinomial2``
        # incurs internally.  Its success-logit is ``log r - log_mean``,
        # matching the failure-logit derivation in ``_nb_lognormal.qmd``.
        # Broadcasting: ``y`` shape ``(batch, G)`` and ``r_vec`` shape
        # ``(G,)`` broadcast as expected.
        counts_obs = None
        if counts is not None:
            counts_obs = counts[idx] if idx is not None else counts

        numpyro.sample(
            "counts",
            LogMeanNegativeBinomial(
                log_mean=y, concentration=r_vec
            ).to_event(1),
            obs=counts_obs,
        )

    def sample(
        self,
        param_values: Dict[str, jnp.ndarray],
        cell_specs: List["ParamSpec"],
        counts: Optional[jnp.ndarray],
        dims: Dict[str, int],
        batch_size: Optional[int],
        model_config: "ModelConfig",
        total_count_max: Optional[int] = None,
        vae_cell_fn: Optional[
            Callable[[Optional[jnp.ndarray]], Dict[str, jnp.ndarray]]
        ] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        param_layouts: Optional[Dict[str, AxisLayout]] = None,
    ) -> None:
        """Draw per-gene Negative Binomial counts from log-normal-modulated means.

        Requires ``vae_cell_fn`` to provide the ``y_log_rate`` decoder
        output (G-dimensional log NB means) inside the cell plate, and
        ``param_values`` to contain the gene dispersion ``r_g`` under
        ``self._r_param_name`` (sampled by the global model machinery
        before the cell plate is entered).

        Three plate modes match the other likelihoods: mini-batch SVI
        (``batch_size`` set), full training/inference with observed
        counts, and prior predictive (``counts is None``).

        Parameters
        ----------
        param_values : dict
            Sampled parameter values.  For NBLN, must contain ``r``
            (gene dispersion).  No totals parameters (``r_T``, ``p``)
            are expected — total counts emerge as the sum of per-gene
            NBs, coupled by the shared covariance ``Sigma``.
        cell_specs : list
            Specs for cell-specific parameters to sample inside the plate.
        counts : jnp.ndarray or None
            Count matrix ``(n_cells, n_genes)``.  None for prior predictive.
        dims : dict
            ``{"n_cells": ..., "n_genes": ...}``.
        batch_size : int or None
            Mini-batch size for SVI; None for full plate.
        model_config : ModelConfig
            Model configuration.
        total_count_max : int, optional
            Unused (no Multinomial in NBLN); accepted for API compatibility.
        vae_cell_fn : callable
            Decoder function returning ``{"y_log_rate": ...}``.
        annotation_prior_logits : jnp.ndarray, optional
            Unused; accepted for API compatibility.
        dataset_indices : jnp.ndarray, optional
            Unused; accepted for API compatibility.
        param_layouts : dict, optional
            Unused; accepted for API compatibility.

        Raises
        ------
        ValueError
            If ``vae_cell_fn`` is None or ``y_log_rate`` is missing.
        """
        del total_count_max, annotation_prior_logits
        del dataset_indices, param_layouts

        if vae_cell_fn is None:
            raise ValueError(
                "NBLogNormalLikelihood requires vae_cell_fn "
                "to sample the latent code and produce decoder output "
                "'y_log_rate' inside the cell plate."
            )

        n_cells = dims["n_cells"]
        n_genes = dims["n_genes"]

        # ------------------------------------------------------------------
        # Mini-batch SVI: subsampled cells plate.
        # ------------------------------------------------------------------
        if batch_size is not None:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                self._cell_body(
                    idx, counts, n_genes, param_values,
                    cell_specs, dims, model_config, vae_cell_fn,
                )
            return

        # ------------------------------------------------------------------
        # Full cell plate (prior predictive or dense observation).
        # ------------------------------------------------------------------
        with numpyro.plate("cells", n_cells):
            self._cell_body(
                None, counts, n_genes, param_values,
                cell_specs, dims, model_config, vae_cell_fn,
            )

    def log_prob(
        self,
        counts: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        param_layouts: Mapping[str, AxisLayout],
        *,
        return_by: str = "cell",
        cells_axis: int = 0,
        r_floor: float = _R_EPS,
        p_floor: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
    ) -> jnp.ndarray:
        """Evaluate ``log p(counts | y_log_rate, r)`` under NB per gene.

        Parameters
        ----------
        counts : jnp.ndarray
            Count matrix ``(n_cells, n_genes)``.
        params : dict
            Must contain:
              ``y_log_rate`` — shape ``(n_cells, G)`` per-cell log NB means
              before exponentiation, and
              ``r`` (or ``self._r_param_name``) — shape ``(G,)`` gene
              dispersions.
        param_layouts : Mapping[str, AxisLayout]
            Semantic layouts for ``y_log_rate``.
        return_by : {"cell", "gene"}, default="cell"
            ``cell``: sum log-probs across genes -> ``(n_cells,)``.
            ``gene``: sum log-probs across cells -> ``(n_genes,)``.
        cells_axis : int, default=0
            Axis indexing cells in ``counts``.
        r_floor : float, default 1e-6
            Lower bound on ``r_g`` to prevent NB log-prob blowup.
        p_floor : float
            Unused; accepted for API compatibility.
        dtype : jnp.dtype
            Working dtype.
        split_components : bool
            Not supported for NBLN.
        weights, weight_type
            Not supported for NBLN.

        Returns
        -------
        jnp.ndarray
            ``(n_cells,)`` if ``return_by="cell"``, ``(n_genes,)`` if
            ``return_by="gene"``.
        """
        del p_floor, weight_type
        _check_return_by(return_by)

        if split_components:
            raise NotImplementedError(
                "NBLogNormalLikelihood does not implement "
                "split_components=True."
            )
        if weights is not None:
            raise NotImplementedError(
                "NBLogNormalLikelihood does not implement "
                "weighted log_prob."
            )

        _require_layout(
            param_layouts, "y_log_rate", context="NBLN.log_prob"
        )

        if self._r_param_name not in params:
            raise ValueError(
                f"NBLogNormalLikelihood.log_prob expected a "
                f"'{self._r_param_name}' entry in params (gene "
                "dispersion, length G)."
            )

        counts_nm = _normalize_counts(counts, cells_axis, dtype)

        y_log_rate = jnp.asarray(params["y_log_rate"], dtype=dtype)
        y_log_rate = self._clamp_log_rate(y_log_rate)

        r_vec = jnp.maximum(
            jnp.asarray(params[self._r_param_name], dtype=dtype),
            jnp.asarray(r_floor, dtype=dtype),
        )

        # Per-gene NB log-prob with broadcasting in log-space:
        #   y_log_rate.shape == (n_cells, G)
        #   r_vec.shape      == (G,)
        # => log_prob.shape  == (n_cells, G)
        # ``LogMeanNegativeBinomial`` computes log-prob entirely in
        # log-space (no ``exp(y_log_rate)``), avoiding the exp/log
        # round-trip that ``NegativeBinomial2`` incurs internally.
        lp_per_gene = LogMeanNegativeBinomial(
            log_mean=y_log_rate, concentration=r_vec
        ).log_prob(counts_nm)

        if return_by == "cell":
            return lp_per_gene.sum(axis=-1)
        else:
            return lp_per_gene.sum(axis=0)

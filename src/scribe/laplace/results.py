"""Results container for Laplace-mode inference.

This module defines :class:`ScribeLaplaceResults`, the single public results
class returned by Laplace inference for every supported base model.  The class
is a dataclass carrying fitted state and diagnostics, while behavior is
implemented via mixins grouped by responsibility (core accessors, model
dispatch, sampling, likelihood evaluation, gene subsetting, and serialization).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from ..models.config import ModelConfig
from ._core import CoreResultsMixin
from ._dispatch import DispatchResultsMixin
from ._gene_subsetting import GeneSubsettingResultsMixin
from ._likelihood import LikelihoodResultsMixin
from ._sampling import SamplingResultsMixin
from ._serialization import SerializationResultsMixin


@dataclass(repr=False)
class ScribeLaplaceResults(
    CoreResultsMixin,
    DispatchResultsMixin,
    SamplingResultsMixin,
    LikelihoodResultsMixin,
    GeneSubsettingResultsMixin,
    SerializationResultsMixin,
):
    """Results from any Laplace-mode fit.

    Parameters
    ----------
    model_config : ModelConfig or None
        Configuration used during fitting and dispatch. The ``base_model``
        field determines runtime branch selection in dispatching methods.
    mu : jnp.ndarray
        Decoder bias vector in model-specific latent coordinates:

        - PLN: shape ``(G,)`` in log-rate coordinates.
        - LNM/LNMVCP: shape ``(G-1,)`` in ALR coordinates.
    W : jnp.ndarray
        Decoder loading matrix with shape ``(G, k)`` for PLN or ``(G-1, k)``
        for LNM-family models.
    d : jnp.ndarray
        Positive diagonal residual variance aligned with ``mu``.
    final_grad_norms : jnp.ndarray
        Final per-cell Newton gradient diagnostics.
    losses : jnp.ndarray
        Outer-loop objective history.
    n_genes : int
        Number of genes represented by this result.
    n_cells : int
        Number of cells represented by this result.
    x_loc, eta_loc, z_loc, y_alr_loc, p_capture_loc : jnp.ndarray, optional
        Model-specific per-cell MAP latent slots. Exactly one latent branch
        is typically populated:

        - PLN: ``x_loc`` (and optionally ``eta_loc`` for capture-anchor fits)
        - LNM low-rank: ``z_loc``
        - LNM learned-diagonal: ``y_alr_loc``
        - LNMVCP: one of the LNM slots plus ``p_capture_loc``
    alr_reference_idx : int, optional
        ALR reference gene index for LNM-family models.
    mu_T, r_T : jnp.ndarray, optional
        Constrained LNM-family NB-on-totals parameters.  Derived from
        unconstrained ``mu_T_loc`` / ``r_T_loc`` via ``positive_transform``.
    r : jnp.ndarray, optional
        Constrained NBLN gene-specific dispersion.  Derived from
        unconstrained ``r_loc`` via ``positive_transform``.
    r_loc, r_scale : jnp.ndarray, optional
        Unconstrained NBLN dispersion posterior: Normal(r_loc, r_scale).
        Shape ``(G,)``.  ``r_scale`` comes from the diagonal Laplace
        approximation of the profiled Hessian.
    mu_T_loc, mu_T_scale, r_T_loc, r_T_scale : jnp.ndarray, optional
        Unconstrained LNM totals marginal posterior parameters.
        Scalar arrays.
    totals_cov : jnp.ndarray, optional
        Full 2x2 covariance for the joint (mu_T, r_T) unconstrained
        posterior.  Important for deriving the distribution of the
        success probability ``p = r_T / (r_T + mu_T)``.
    var, obs : Any, optional
        AnnData-style metadata.
    n_obs, n_vars : int, optional
        Source-data shape metadata.
    early_stopped : bool, default=False
        Whether early stopping was triggered.
    best_loss : float, default=inf
        Best smoothed loss observed during optimization.
    stopped_at_step : int, default=0
        Final optimization step.
    divergence_aborted : bool, default=False
        Whether divergence detection aborted training.
    _subset_gene_index : np.ndarray, optional
        Internal bookkeeping for gene subset views.
    metadata : dict, default_factory=dict
        Free-form metadata and compatibility storage used by plotting helpers
        (for example ``predictive_samples`` and ``posterior_samples`` caches).
    frozen_params : frozenset[str], default=empty
        Set of NBLN parameter short-names ``{"r", "mu", "eta"}`` that were
        pinned at the SVI-cascade-source MAP during the Laplace M-step (see
        ``informative_priors_freeze`` on ``scribe.fit``).  Metadata only:
        indicates *provenance* of the corresponding posterior — when a key
        is present here, ``get_distributions()`` and PPC paths route through
        ``cascade_source`` rather than the NBLN profiled Hessian.
    cascade_source : ScribeSVIResults, optional
        Source SVI results object embedded by reference when a freeze cascade
        was active.  Used to recover full-fidelity posterior samples and
        moment-matched distributions for any parameter listed in
        ``frozen_params``.  ``None`` for non-cascade fits.
    cascade_source_counts : jnp.ndarray, optional
        Cached count matrix written when the embedded cascade source uses
        amortized capture **and** its own ``_original_counts`` is unset, so
        that PPC sampling can evaluate the amortized encoder after pickle.
        Set from ``ctx.count_data`` at bridge time; never mutates the source.
    w_prior_diagnostics : dict, optional
        Per-fit summary of the loadings-shrinkage prior — keys include
        ``strategy_type``, ``sigma_k``, ``column_frobenius_compositional``
        (the gauge-invariant per-factor norm — the **headline** rank
        diagnostic), ``column_frobenius_raw``, ``column_norm_effective_rank``
        / ``effective_rank``, and strategy-specific aux MAPs (e.g. ``tau``
        for horseshoe, ``gamma`` for NEG).  ``None`` when no shrinkage prior
        was active.  See ``scribe.viz.plot_w_shrinkage_spectrum``.

    Notes
    -----
    The class is intentionally model-agnostic at the type level: all Laplace
    results share one container type and dispatch internally by ``base_model``.
    This mirrors the high-level SCRIBE design of one result class per
    inference method.

    See Also
    --------
    scribe.svi.results.ScribeSVIResults
        SVI results class that uses the same mixin-composition architecture.
    """

    model_config: Optional[ModelConfig]
    mu: jnp.ndarray
    W: jnp.ndarray
    d: jnp.ndarray
    final_grad_norms: jnp.ndarray
    losses: jnp.ndarray
    n_genes: int
    n_cells: int

    x_loc: Optional[jnp.ndarray] = None
    eta_loc: Optional[jnp.ndarray] = None
    z_loc: Optional[jnp.ndarray] = None
    y_alr_loc: Optional[jnp.ndarray] = None
    p_capture_loc: Optional[jnp.ndarray] = None
    alr_reference_idx: Optional[int] = None

    # Constrained NB-on-totals globals (LNM family only).  Derived from
    # unconstrained ``*_loc`` coordinates via ``positive_transform``.
    mu_T: Optional[jnp.ndarray] = None
    r_T: Optional[jnp.ndarray] = None

    # NB-LogNormal gene-specific dispersion ``r_g`` (length G). Only
    # populated for ``base_model == "nbln"``; ``None`` for PLN /
    # LNM / LNMVCP.  Derived as ``positive_transform(r_loc)``.
    r: Optional[jnp.ndarray] = None

    # TwoState-LogNormal-Rate fields (``base_model == "twostate_ln_rate"``).
    # All constrained positive quantities; per-gene shape ``(G,)``.
    # See plan §4.A.3 and ``_obs_twostate_ln_rate.py``.
    #
    # CONVENTION.  For TSLN-Rate, the
    # interpretation of ``mu`` is the **latent log-rate prior
    # center** ``log(r_hat)`` — matching NBLN/PLN where ``self.mu``
    # is the loc of the latent log-rate distribution
    # ``LowRankMultivariateNormal(loc=self.mu, ...)``.  The
    # **positive TwoState gene-mean parameter** (the SVI source's
    # ``mu``) is stored separately in :attr:`gene_mean` so that
    # ``self.mu`` is semantically consistent across all
    # Laplace-supported base models.
    burst_size: Optional[jnp.ndarray] = None
    k_off: Optional[jnp.ndarray] = None
    # The TwoState positive gene-mean parameter (= SVI source's ``mu``).
    # Distinct from ``self.mu`` which carries the latent log-rate
    # prior center for TSLN-Rate.  ``gene_mean = pos_forward(
    # gene_mean_loc)``.
    gene_mean: Optional[jnp.ndarray] = None
    # Derived TSLN quantities (gene-level, all shape ``(G,)``).
    # ``alpha = gene_mean/burst_size``, ``beta = k_off``,
    # ``r_hat = gene_mean + burst_size*k_off`` after the mean-preserving
    # floor in ``_twostate_reparam``.
    #
    # For ``twostate_ln_logit`` (Variant B) these fields are also
    # populated but with the logit-variant derivation:
    # ``alpha = kappa · σ(eta_anchor)``,
    # ``beta = kappa · (1 − σ(eta_anchor))``.  ``r_hat`` is left
    # unpopulated for the logit variant — its analog is :attr:`rate`.
    alpha: Optional[jnp.ndarray] = None
    beta: Optional[jnp.ndarray] = None
    r_hat: Optional[jnp.ndarray] = None

    # TwoState-LogNormal-Logit fields (``base_model ==
    # "twostate_ln_logit"``).  Variant B's sampled gene-level globals.
    # See plan §4.C.3 and ``_obs_twostate_ln_logit.py``.
    #
    # ``rate`` (positive): gene-level ON-production rate.
    # **z-independent** by Rev 4 (auditor's Round-1 fix).  The Poisson
    # scale in the data-side likelihood is ``λ = rate · ν_c`` where
    # ``ν_c = exp(-eta_capture)`` for fixed-offset capture.
    # ``kappa`` (positive): Beta concentration ``α + β``.
    # ``eta_anchor`` (real): per-gene activation log-odds ``θ_g``.
    # The TSLN-Logit per-cell-per-gene shape parameters are derived
    # at the data side as ``α_cg = κ · σ(θ_g + z_g)`` and
    # ``β_cg = κ · (1 − σ(θ_g + z_g))``.  The cell-level posterior
    # over (α, β) is implicit in (x_loc, eta_anchor); the gene-level
    # ``alpha`` / ``beta`` fields above carry the values at ``z = 0``.
    rate: Optional[jnp.ndarray] = None
    kappa: Optional[jnp.ndarray] = None
    eta_anchor: Optional[jnp.ndarray] = None

    # Curvature-clamp diagnostics (TSLN-rate, plan §4.A.3 Rev 3+).  The
    # closed-form Hessian-diagonal factor ``a_g`` is defensively floored
    # at ``_A_MIN`` before the Woodbury solve because the Poisson-Beta
    # marginal is not log-concave in general (Beta U-shape regime).
    # These fields surface how often the clamp activated on the final
    # sweep so users can detect prior-dominated cells/genes.
    #
    # ``a_raw_min``: scalar minimum of ``a_raw`` across ALL
    #   ``(cell, gene)`` entries in the final-sweep tensor — a single
    #   diagnostic number for "worst case curvature on this fit".
    # ``a_raw_negative_fraction``: scalar fraction of ``(cell, gene)``
    #   entries with ``a_raw < 0`` (genuine log-concavity violations).
    # ``a_clamp_fraction``: scalar fraction where ``_A_MIN`` floor
    #   activated (≥ ``a_raw_negative_fraction``).
    # ``a_clamp_per_gene``: per-gene clamp activation rate, shape
    #   ``(G,)`` — averaged over cells so users can identify which
    #   genes most often trigger the floor.
    a_raw_min: Optional[jnp.ndarray] = None  # scalar (over (cell, gene))
    a_raw_negative_fraction: Optional[jnp.ndarray] = None  # scalar
    a_clamp_fraction: Optional[jnp.ndarray] = None  # scalar
    a_clamp_per_gene: Optional[jnp.ndarray] = None  # shape (G,)

    # --- Global posterior uncertainty (unconstrained space) --------
    #
    # All ``*_loc`` and ``*_scale`` fields live in unconstrained
    # pre-transform space.  Constrained positive parameters are
    # obtained by applying ``model_config.positive_transform`` to
    # the ``*_loc`` values.  The ``*_scale`` fields are standard
    # deviations of the approximate Normal posterior in unconstrained
    # space; they are NOT standard deviations of the constrained
    # (positive) parameter.
    #
    # NBLN fields:
    r_loc: Optional[jnp.ndarray] = None
    r_scale: Optional[jnp.ndarray] = None
    # TSLN-rate fields (unconstrained location/scale of the three
    # positive globals; constrained values are in the non-_loc
    # attributes above):
    burst_size_loc: Optional[jnp.ndarray] = None
    burst_size_scale: Optional[jnp.ndarray] = None
    k_off_loc: Optional[jnp.ndarray] = None
    k_off_scale: Optional[jnp.ndarray] = None
    # ``gene_mean_loc`` is the unconstrained pre-transform of
    # ``gene_mean`` (the TwoState positive gene-mean parameter).
    # ``gene_mean = pos_forward(gene_mean_loc)`` exactly.
    # For TSLN-Rate, ``self.mu_loc`` / ``self.mu_scale`` are NOT
    # populated — those are NBLN-specific.
    gene_mean_loc: Optional[jnp.ndarray] = None
    gene_mean_scale: Optional[jnp.ndarray] = None
    # TSLN-Logit fields (``base_model == "twostate_ln_logit"``).
    # ``rate_loc / kappa_loc`` are unconstrained pre-transforms of
    # the positive gene-level globals (``rate / kappa``).
    # ``eta_anchor_loc`` is real-valued (identity transform) and equal
    # to :attr:`eta_anchor`.  All three are populated by
    # ``compute_global_uncertainty``; the corresponding ``*_scale``
    # entries are NaN when the parameter is frozen at the cascade MAP
    # (the default L4 cascade freezes all three).
    rate_loc: Optional[jnp.ndarray] = None
    rate_scale: Optional[jnp.ndarray] = None
    kappa_loc: Optional[jnp.ndarray] = None
    kappa_scale: Optional[jnp.ndarray] = None
    eta_anchor_loc: Optional[jnp.ndarray] = None
    eta_anchor_scale: Optional[jnp.ndarray] = None
    # NBLN latent prior mean ``mu`` posterior (per gene, log-rate
    # coordinate).  Populated by ``compute_global_uncertainty`` using
    # the diagonal-Σ approximation of the profiled Hessian.  Both
    # values live in the unconstrained real-valued log-rate space
    # (NBLN ``params["mu"]`` is the latent prior mean, not a positive
    # parameter — no ``positive_transform`` is applied).
    mu_loc: Optional[jnp.ndarray] = None
    mu_scale: Optional[jnp.ndarray] = None
    #
    # LNM / LNMVCP totals fields:
    mu_T_loc: Optional[jnp.ndarray] = None
    mu_T_scale: Optional[jnp.ndarray] = None
    r_T_loc: Optional[jnp.ndarray] = None
    r_T_scale: Optional[jnp.ndarray] = None
    # Full 2x2 covariance for the joint (mu_T, r_T) unconstrained
    # posterior.  Needed to derive coherent distributions for the
    # derived quantity p = r_T / (r_T + mu_T).
    totals_cov: Optional[jnp.ndarray] = None

    var: Optional[Any] = None
    obs: Optional[Any] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # --- SVI-cascade freeze fields --------------------------------
    #
    # ``frozen_params`` is the subset of ``{"r", "mu", "eta"}`` that
    # was excluded from the optax optimizer during the M-step.  Their
    # values came from the SVI source's MAP via ``freeze_values_from_results``.
    # Downstream consumers (``get_distributions``, PPC sampling, the
    # gauge-contamination diagnostic) check this set to decide whether
    # to route through ``cascade_source``.  Empty frozenset for plain
    # Laplace fits with no cascade.
    frozen_params: frozenset = field(default_factory=frozenset)
    #
    # ``cascade_source`` holds the full ``ScribeSVIResults`` of the SVI
    # source when a cascade was active.  ``get_distributions()`` for
    # frozen parameters moment-matches samples from this guide in NBLN
    # target coordinate; PPC samples for frozen parameters route through
    # the guide directly for full-fidelity predictive sampling.
    # ``None`` for plain Laplace fits.
    cascade_source: Optional[Any] = None
    #
    # ``cascade_source_counts`` caches the count matrix needed for
    # amortized-capture SVI sources to sample (the encoder evaluates on
    # counts at sample time).  Set by the run-inference stage when the
    # cascade source is amortized and its own ``_original_counts`` field
    # is missing — populated from the target's ``ctx.count_data`` after
    # var-name identity verification.  ``None`` otherwise; PPC then
    # reads ``cascade_source._original_counts`` directly.  Storing here
    # avoids mutating the user's SVI result.
    cascade_source_counts: Optional[jnp.ndarray] = None
    #
    # ``_cascade_subset_info`` carries the panel-relationship metadata
    # from ``scribe.laplace.priors._check_gene_identity`` when the
    # Laplace target's gene panel is a STRICT subset of the SVI
    # source's.  Read by PPC (``_resolve_nbln_ppc_arrays``) and other
    # cascade-aware consumers to apply per-sample NB moment-matching
    # aggregation when slicing SVI samples onto the target gene axis.
    # ``None`` when panels match exactly (today's pass-through path).
    # The typed ``SubsetInfo`` lives in ``scribe.laplace.priors``; this
    # field is ``Optional[Any]`` to avoid a circular import.
    _cascade_subset_info: Optional[Any] = None
    #
    # ``axis_layout`` records the obs-axis / latent-covariance-axis
    # split used by the obs model under
    # ``ModelConfig.correlate_other_column``.  Under the legacy layout
    # (or when no ``_other`` column was emitted) the layout is
    # trivial and ``G_kept == G_obs``; the shapes of ``W``/``d``/etc.
    # remain identical to today.  Under the decoupled layout
    # (``axis_layout.decoupled is True``) ``W`` has shape
    # ``(G_kept, K)``, ``d`` has shape ``(G_kept,)``, and the per-cell
    # latent stores a deviation in ``(G_kept,)`` while ``mu`` and ``r``
    # stay at ``(G_obs,)``.  Downstream tools (PPC, compositional
    # sampler, ``get_W_compositional``, DE) check this field to
    # disambiguate the W/d shape from the mu/r shape; see the
    # ``G_obs`` / ``G_kept`` properties below.  ``None`` for non-PLN/
    # NBLN/TSLN base models and for legacy fits not threaded through
    # the engine's gene-name / has-pooled-other plumbing.  Typed
    # ``Optional[Any]`` to avoid a circular import with
    # ``scribe.laplace._axis_layout.AxisLayout``.
    axis_layout: Optional[Any] = None

    # ---- Per-cell Newton convergence diagnostics ------------------
    #
    # ``_pre_rescue_grad_norms`` carries the per-cell grad-norm tensor
    # from BEFORE the optional rescue pass ran.  ``None`` when the
    # rescue pass did not run (i.e. ``LaplaceConfig.rescue_diverged_cells``
    # was False, or it was True but no cell exceeded the rescue threshold).
    # When present, ``final_grad_norms`` is the post-rescue tensor;
    # users wanting "what fraction of cells were diverged on the main
    # fit" should consult ``unconverged_cell_mask`` (the property
    # defined below) which delegates to ``_pre_rescue_grad_norms`` when
    # available.
    _pre_rescue_grad_norms: Optional[jnp.ndarray] = None

    # ``_rescued_cell_mask`` is the boolean per-cell mask of cells that
    # were passed through the rescue pass (i.e. exceeded the rescue
    # threshold and got re-run with more Newton steps).  ``None`` when
    # rescue did not run.  A cell with this mask True may or may not
    # have converged on the rescue — check ``model_unfit_cell_mask`` to
    # distinguish.
    _rescued_cell_mask: Optional[np.ndarray] = None

    # ``_newton_tolerance`` is the per-cell Newton tolerance used during
    # the fit (``LaplaceConfig.newton_tolerance``).  Stored on the result
    # so the convergence-mask properties can apply the same threshold
    # the fit actually used, even when the user later runs with a
    # different default.  ``None`` for older pickles or when the bridge
    # did not record it; the mask properties fall back to ``1e-4`` in
    # that case.
    _newton_tolerance: Optional[float] = None

    # Phase-3 W-shrinkage prior diagnostics.  Populated by the obs
    # model's ``pack_result`` when a W-prior strategy is configured
    # (always populated for PLN/NBLN — even ``NoneWPrior`` produces a
    # minimal dict with ``{"strategy_type": "none", ...}``).  ``None``
    # for LNM-family results in v1 since shrinkage isn't yet supported
    # there.  See ``scribe.laplace._w_priors`` for the strategy
    # protocol and ``scribe.viz.plot_w_shrinkage_spectrum`` for the
    # companion elbow plot.
    w_prior_diagnostics: Optional[Dict[str, Any]] = None

    _gene_coverage: Optional[float] = None
    _gene_coverage_mask: Optional[np.ndarray] = None
    _excluded_gene_names: Optional[List[str]] = None
    _original_n_genes: Optional[int] = None
    _total_count_max: Optional[int] = None

    early_stopped: bool = False
    best_loss: float = float("inf")
    stopped_at_step: int = 0
    divergence_aborted: bool = False

    _subset_gene_index: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Jacobian-corrected MAP control. ``None`` lets ``get_map`` use its
    # own ``"auto"`` default; a string here pins the per-result default
    # (e.g., ``with_jacobian_map()`` sets this to ``"auto"`` to mark
    # the result's stored fields as already corrected). See
    # ``scribe.stats.jacobian_map`` for the math and the plan's §3.2.
    map_method: Optional[str] = None

    @property
    def G_obs(self) -> int:
        """Length of the observation-layer gene axis.

        Equals the number of count-data columns and the length of
        observation-only per-gene parameters (``r`` for NBLN/TSLN,
        ``eta_anchor`` for TSLN-Logit).  Always ``self.n_genes`` when
        that attribute is populated; otherwise falls back to the
        ``axis_layout.G_obs`` when an axis layout is present.

        Use this to disambiguate from :attr:`G_kept` (which excludes
        the trailing ``_other`` pooled column under the decoupled
        layout).  See ``scribe.laplace._axis_layout.AxisLayout`` for
        the full shape contract.
        """
        if self.n_genes is not None:
            return int(self.n_genes)
        if self.axis_layout is not None:
            return int(self.axis_layout.G_obs)
        # Last-resort introspection from a known per-gene array.
        if self.mu is not None and self.mu.ndim >= 1:
            return int(self.mu.shape[0])
        raise ValueError(
            "Cannot determine G_obs: n_genes, axis_layout, and mu are "
            "all unavailable on this result object."
        )

    @property
    def G_kept(self) -> int:
        """Length of the latent-covariance gene axis.

        Equals the number of rows in ``W`` and ``d`` and the per-cell
        latent's gene dimension.  Under the **decoupled** layout
        (``ModelConfig.correlate_other_column=False`` AND data has a
        trailing ``_other`` aggregate), ``G_kept == G_obs - 1`` and
        the ``_other`` row is excluded from the regulatory covariance
        Σ = W Wᵀ + diag(d).  Under the legacy layout (or when no
        ``_other`` was emitted), ``G_kept == G_obs``.

        Use this when inspecting the shape of ``self.W`` / ``self.d``
        or building compositional accessors that must operate on the
        kept-gene subspace (see ``get_W_compositional``).
        """
        if self.axis_layout is not None:
            return int(self.axis_layout.G_kept)
        # Without an axis_layout we cannot distinguish decoupled from
        # legacy; fall back to G_obs (today's behaviour).
        return self.G_obs

    @property
    def unconverged_cell_mask(self) -> Optional[np.ndarray]:
        """Boolean per-cell mask of cells that did not converge on the main fit.

        Returns ``True`` for cells whose final inner-Newton gradient norm
        exceeded ``LaplaceConfig.newton_tolerance`` BEFORE the optional
        rescue pass.  When rescue did not run, this is equivalent to
        ``final_grad_norms > newton_tolerance``.

        The threshold comes from ``self._newton_tolerance`` when it was
        recorded by the bridge (the typical path).  Older pickles missing
        the field fall back to ``1e-4`` — the configured default
        on every shipped LaplaceConfig — which gives the right answer
        unless the user fit with a custom tolerance.

        Returns
        -------
        np.ndarray or None
            Boolean array of shape ``(n_cells,)``.  ``None`` when
            ``final_grad_norms`` is unavailable on this result.
        """
        if self.final_grad_norms is None:
            return None
        # Pre-rescue norms if rescue ran, else raw norms.
        source = (
            self._pre_rescue_grad_norms
            if self._pre_rescue_grad_norms is not None
            else self.final_grad_norms
        )
        tol = (
            float(self._newton_tolerance)
            if self._newton_tolerance is not None
            else 1e-4
        )
        return np.asarray(np.asarray(source) > tol, dtype=bool)

    @property
    def model_unfit_cell_mask(self) -> Optional[np.ndarray]:
        """Boolean per-cell mask of cells the model cannot fit, post-rescue.

        Returns ``True`` for cells whose final inner-Newton gradient norm
        is STILL above ``newton_tolerance`` AFTER the rescue pass.  These
        are the cells where giving Newton more iterations and tighter
        damping was not enough — the model genuinely cannot fit them.

        Only meaningful when the rescue pass ran (i.e.
        ``_rescued_cell_mask is not None``).  When rescue did not run,
        returns ``None`` — distinguish this from "all cells fit"
        (which would be an all-``False`` mask).

        Returns
        -------
        np.ndarray or None
            Boolean array of shape ``(n_cells,)`` when rescue ran;
            ``None`` otherwise.
        """
        if (
            self._rescued_cell_mask is None
            or self.final_grad_norms is None
        ):
            return None
        tol = (
            float(self._newton_tolerance)
            if self._newton_tolerance is not None
            else 1e-4
        )
        return np.asarray(
            np.asarray(self.final_grad_norms) > tol, dtype=bool
        )

    def __repr__(self) -> str:
        """Return a concise text summary for interactive inspection.

        The dataclass default representation recursively expands all fields,
        including large arrays and metadata dictionaries. This compact repr
        keeps notebook output readable by showing only high-value summary
        fields and lightweight status flags.

        Returns
        -------
        str
            Stable summary string with model identity, dataset shape,
            optimization history length, latent-state branch, and global
            uncertainty status.
        """
        n_steps = int(len(self.losses)) if self.losses is not None else 0
        base_model = self._base_model_name()
        latent_state = self._summarize_latent_state()
        uncertainty = self._summarize_global_uncertainty()
        return (
            "ScribeLaplaceResults("
            f"model={base_model!r}, "
            f"n_cells={self.n_cells}, "
            f"n_genes={self.n_genes}, "
            f"n_steps={n_steps}, "
            f"latent={latent_state!r}, "
            f"uncertainty={uncertainty!r})"
        )

    def _repr_html_(self) -> str:
        """Return a compact HTML table for notebook frontends.

        Notebook frontends such as Jupyter and marimo use ``_repr_html_`` when
        available. This HTML mirrors the key fields from ``__repr__`` so users
        see the same summary information without large tensor dumps.

        Returns
        -------
        str
            HTML snippet containing a small summary table.
        """
        import html

        n_steps = int(len(self.losses)) if self.losses is not None else 0
        # Escape user- or config-derived strings to keep HTML rendering safe.
        base_model = html.escape(self._base_model_name())
        latent_state = html.escape(self._summarize_latent_state())
        uncertainty = html.escape(self._summarize_global_uncertainty())
        return (
            "<div>"
            "<strong>ScribeLaplaceResults</strong>"
            "<table>"
            f"<tr><td>model</td><td>{base_model}</td></tr>"
            f"<tr><td>n_cells</td><td>{self.n_cells}</td></tr>"
            f"<tr><td>n_genes</td><td>{self.n_genes}</td></tr>"
            f"<tr><td>n_steps</td><td>{n_steps}</td></tr>"
            f"<tr><td>latent</td><td>{latent_state}</td></tr>"
            f"<tr><td>uncertainty</td><td>{uncertainty}</td></tr>"
            "</table>"
            "</div>"
        )

    def _base_model_name(self) -> str:
        """Return the resolved base model name for display.

        Returns
        -------
        str
            ``model_config.base_model`` when available, otherwise
            ``"unknown"``.
        """
        if self.model_config is None:
            return "unknown"
        return str(getattr(self.model_config, "base_model", "unknown"))

    def with_jacobian_map(self) -> "ScribeLaplaceResults":
        """Return a copy where the stored positive globals are
        recomputed with the Jacobian-corrected MAP, AND all derived
        quantities are re-derived from the corrected parents.

        Semantics
        ---------
        The ``_loc`` / ``_scale`` pairs on a ``ScribeLaplaceResults``
        are the canonical posterior parameterization; the stored
        constrained fields (``self.r``, ``self.gene_mean``, etc.) are
        a *view* derived from them. By default (``map_method=None``,
        defaults to ``"auto"`` inside :meth:`get_map`), the view is
        recomputed lazily on every ``get_map`` call. This method
        materializes the corrected view into the stored fields so that:

        - ``new_result.gene_mean`` returns the corrected value.
        - Downstream code reading ``self.r`` / ``self.gene_mean`` /
          ``self.alpha`` directly (e.g., PPC samplers, cascade
          adapters) sees the corrected values.
        - ``new_result.map_method`` is set to ``"auto"`` so subsequent
          ``get_map()`` calls inherit the same default.

        Important: ``_loc`` and ``_scale`` are NOT modified, so
        ``new_result.get_map(map_method="transform")`` still derives
        from the loc/scale pair and returns the uncorrected median
        ``transform(loc)``. The chosen semantics are: ``"transform"``
        always means "raw transform-of-loc", regardless of what's in
        the stored view fields.

        Limitations
        -----------
        * ``LowRankMVN + Sigmoid``/``Softplus`` pairs cannot be
          corrected in v1; the corresponding fields fall back to
          ``transform(loc)`` with a warning.
        * ``p_capture`` requires ``eta_scale``, which is not currently
          persisted; the warning fires once (deduped by the default
          Python warning filter).

        Returns
        -------
        ScribeLaplaceResults
            A new dataclass instance with corrected constrained fields
            and ``map_method="auto"``.
        """
        from dataclasses import replace as _replace

        from ._derived import (
            lnm_p_from_parents,
            twostate_logit_derived_from_parents,
            twostate_rate_derived_from_parents,
        )
        from ._dispatch import _correct_positive
        from ._results_shared import _base_model

        bm = _base_model(self.model_config)
        updates: Dict[str, Any] = {"map_method": "auto"}

        if bm == "nbln":
            r_corrected = _correct_positive(
                self.model_config, "r",
                getattr(self, "r_loc", None),
                getattr(self, "r_scale", None),
                method="auto",
            )
            if r_corrected is not None:
                updates["r"] = r_corrected

        elif bm == "twostate_ln_rate":
            gene_mean_corrected = _correct_positive(
                self.model_config, "mu",
                getattr(self, "gene_mean_loc", None),
                getattr(self, "gene_mean_scale", None),
                method="auto",
            )
            burst_size_corrected = _correct_positive(
                self.model_config, "burst_size",
                getattr(self, "burst_size_loc", None),
                getattr(self, "burst_size_scale", None),
                method="auto",
            )
            k_off_corrected = _correct_positive(
                self.model_config, "k_off",
                getattr(self, "k_off_loc", None),
                getattr(self, "k_off_scale", None),
                method="auto",
            )
            if gene_mean_corrected is not None:
                updates["gene_mean"] = gene_mean_corrected
            if burst_size_corrected is not None:
                updates["burst_size"] = burst_size_corrected
            if k_off_corrected is not None:
                updates["k_off"] = k_off_corrected
            # Re-derive (alpha, beta, r_hat) from corrected parents,
            # iff all three are available. Otherwise leave the stored
            # derived values untouched.
            if all(
                k in updates for k in ("gene_mean", "burst_size", "k_off")
            ):
                derived = twostate_rate_derived_from_parents(
                    gene_mean=updates["gene_mean"],
                    burst_size=updates["burst_size"],
                    k_off=updates["k_off"],
                )
                updates["alpha"] = derived["alpha"]
                updates["beta"] = derived["beta"]
                updates["r_hat"] = derived["r_hat"]
                # CRITICAL: TSLN-Rate convention requires
                # ``self.mu == log(self.r_hat)`` (mu is the latent
                # log-rate prior center). PPC paths in _sampling.py
                # read self.mu directly via
                # _ppc_twostate_ln_rate_marginal — without this
                # re-sync, with_jacobian_map() would leave a stale
                # latent center and PPC samples would silently mix
                # uncorrected mu with corrected (alpha, beta).
                # Mirrors the same re-sync in _dispatch.py:get_map.
                updates["mu"] = jnp.log(
                    jnp.maximum(derived["r_hat"], 1e-30)
                )

        elif bm == "twostate_ln_logit":
            rate_corrected = _correct_positive(
                self.model_config, "rate",
                getattr(self, "rate_loc", None),
                getattr(self, "rate_scale", None),
                method="auto",
            )
            kappa_corrected = _correct_positive(
                self.model_config, "kappa",
                getattr(self, "kappa_loc", None),
                getattr(self, "kappa_scale", None),
                method="auto",
            )
            if rate_corrected is not None:
                updates["rate"] = rate_corrected
            if kappa_corrected is not None:
                updates["kappa"] = kappa_corrected
            if "kappa" in updates and self.eta_anchor is not None:
                derived = twostate_logit_derived_from_parents(
                    kappa=updates["kappa"],
                    eta_anchor=self.eta_anchor,
                )
                updates["alpha"] = derived["alpha"]
                updates["beta"] = derived["beta"]

        elif bm in ("lnm", "lnmvcp"):
            mu_T_corrected = _correct_positive(
                self.model_config, "mu_T",
                self.mu_T_loc, self.mu_T_scale, method="auto",
            )
            r_T_corrected = _correct_positive(
                self.model_config, "r_T",
                self.r_T_loc, self.r_T_scale, method="auto",
            )
            if mu_T_corrected is not None:
                updates["mu_T"] = mu_T_corrected
            if r_T_corrected is not None:
                updates["r_T"] = r_T_corrected
            # ``p`` is not a dataclass field — it's derived inside
            # get_map() from mu_T / r_T. So we do NOT add it to updates.
            # On the next get_map() call, the corrected p will be
            # computed from the updated mu_T and r_T.

        # PLN has no positive globals to correct.
        return _replace(self, **updates)

    def _summarize_latent_state(self) -> str:
        """Summarize which latent MAP slots are populated.

        Returns
        -------
        str
            Short descriptor of active latent branches.
        """
        components: List[str] = []
        # Report the dominant latent branch first for readability.
        if self.x_loc is not None:
            components.append("x")
        if self.z_loc is not None:
            components.append("z")
        if self.y_alr_loc is not None:
            components.append("y_alr")
        if self.eta_loc is not None:
            components.append("eta")
        if self.p_capture_loc is not None:
            components.append("p_capture")
        return ",".join(components) if components else "none"

    def _summarize_global_uncertainty(self) -> str:
        """Summarize availability of global uncertainty fields.

        Returns
        -------
        str
            Short descriptor of populated uncertainty blocks.
        """
        components: List[str] = []
        # NBLN uncertainty block.
        if self.r_loc is not None or self.r_scale is not None:
            components.append("r")
        if self.mu_loc is not None or self.mu_scale is not None:
            components.append("mu")
        # TSLN-Rate uncertainty blocks.
        if (
            self.gene_mean_loc is not None
            or self.gene_mean_scale is not None
        ):
            components.append("gene_mean")
        if (
            self.burst_size_loc is not None
            or self.burst_size_scale is not None
        ):
            components.append("burst_size")
        if self.k_off_loc is not None or self.k_off_scale is not None:
            components.append("k_off")
        # LNM/LNMVCP totals uncertainty block.
        if (
            self.mu_T_loc is not None
            or self.mu_T_scale is not None
            or self.r_T_loc is not None
            or self.r_T_scale is not None
            or self.totals_cov is not None
        ):
            components.append("totals")
        return ",".join(components) if components else "none"


__all__ = ["ScribeLaplaceResults"]

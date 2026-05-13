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

    # --- Phase-2 cascade-freeze fields (Round-5 R5-5 contract) ----
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

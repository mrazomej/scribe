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


@dataclass
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
        LNM-family NB-on-totals parameters.
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

    mu_T: Optional[jnp.ndarray] = None
    r_T: Optional[jnp.ndarray] = None

    var: Optional[Any] = None
    obs: Optional[Any] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

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


__all__ = ["ScribeLaplaceResults"]

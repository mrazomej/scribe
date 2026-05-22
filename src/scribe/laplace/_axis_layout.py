"""Axis-layout metadata for the latent-covariance models (PLN, NBLN, TSLN).

When the data has a trailing aggregated ``"_other"`` column (emitted by
``scribe.core.gene_coverage.aggregate_counts_by_mask`` whenever
``gene_coverage < 1.0``), the trailing column is a pooled-counts
aggregate, not a real gene.  Including it in the latent low-rank
covariance ``Σ = W Wᵀ + diag(d)`` wastes capacity on biophysically
meaningless cross-gene correlations.

The ``correlate_other_column`` flag on ``ModelConfig`` (default
``False``) excludes ``"_other"`` from Σ while keeping it in the
observation likelihood.  This requires distinguishing two gene axes
throughout the model code:

* ``G_obs`` — the observation-layer axis.  Length of the count-data
  column dimension; per-gene parameters that live in the observation
  likelihood (``r`` for NBLN/TSLN, ``eta_anchor`` for TSLN-Logit)
  retain this shape.
* ``G_kept`` — the latent-covariance axis.  Equal to ``G_obs - 1``
  when an ``"_other"`` column is present and excluded from Σ, else
  equal to ``G_obs``.  ``W`` has shape ``(G_kept, K)``, ``d`` has
  shape ``(G_kept,)``, and the per-cell latent deviation has shape
  ``(G_kept,)``.

This module provides the :class:`AxisLayout` value object that
captures both axes and the index mapping between them, plus a
factory function :func:`build_axis_layout` that constructs the
layout from the ``ModelConfig`` flag and the data shape.

See ``paper/_nb_lognormal.qmd`` §sec-nbln-decorrelate-other for the
biophysical rationale and the gauge-invariance properties of the
decoupled construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# Sentinel string for the trailing aggregated low-coverage column.
# Must match ``scribe.core.gene_coverage._OTHER_NAME`` (kept in sync
# with ``scribe.laplace.priors._OTHER_NAME``).
_OTHER_NAME = "_other"


@dataclass(frozen=True)
class AxisLayout:
    """Per-fit metadata describing the observation / latent-covariance split.

    Constructed once at obs-model init from ``model_config.correlate_other_column``
    and the data shape / gene names.  Threaded through loss, Newton,
    init, packing, PPC, and compositional-sampler code paths to drop
    the ``"_other"`` row from the latent covariance while keeping it
    in the observation likelihood.

    Attributes
    ----------
    G_obs : int
        Length of the count-data column axis.  Per-gene parameters
        that live in the observation likelihood (``r`` for NBLN/TSLN,
        ``eta_anchor`` for TSLN-Logit) have this shape.
    G_kept : int
        Length of the latent-covariance axis.  ``W`` has shape
        ``(G_kept, K)``, ``d`` has shape ``(G_kept,)``, and the
        per-cell latent deviation has shape ``(G_kept,)``.
    kept_idx : np.ndarray
        Integer array of length ``G_kept``.  Position-in-G_obs of
        each kept gene, in target-axis order.  For the trivial layout
        (``decoupled=False``) this is ``np.arange(G_obs)``.
    other_idx : int, optional
        Position-in-G_obs of the trailing ``"_other"`` column when
        present and excluded from Σ.  ``None`` for the trivial layout.
    decoupled : bool
        Convenience: ``other_idx is not None``.  True iff
        ``G_kept < G_obs``.
    """

    G_obs: int
    G_kept: int
    kept_idx: np.ndarray
    other_idx: Optional[int]

    @property
    def decoupled(self) -> bool:
        """True iff the latent-covariance axis is strictly smaller than G_obs."""
        return self.other_idx is not None

    def __post_init__(self) -> None:
        # Defensive shape / consistency checks at construction time so
        # downstream code can rely on the invariants.
        if self.G_kept > self.G_obs:
            raise ValueError(
                f"G_kept ({self.G_kept}) must be <= G_obs ({self.G_obs})."
            )
        if self.kept_idx.shape != (self.G_kept,):
            raise ValueError(
                f"kept_idx must have shape ({self.G_kept},); got "
                f"{self.kept_idx.shape}."
            )
        if self.other_idx is None and self.G_kept != self.G_obs:
            raise ValueError(
                "other_idx must be set when G_kept < G_obs."
            )
        if self.other_idx is not None and self.G_kept + 1 != self.G_obs:
            raise ValueError(
                f"Only one trailing '_other' column is supported; got "
                f"G_kept={self.G_kept}, G_obs={self.G_obs}."
            )


def build_axis_layout(
    n_genes: int,
    *,
    correlate_other_column: bool,
    gene_names: Optional[Sequence[str]] = None,
) -> AxisLayout:
    """Construct an :class:`AxisLayout` from the data shape and config flag.

    When ``correlate_other_column=True`` (legacy behaviour) or when the
    data has no trailing ``"_other"`` column, returns the trivial
    layout where ``G_kept == G_obs`` and the latent covariance spans
    the full observation axis.

    When ``correlate_other_column=False`` (the default) AND the
    trailing column is named ``"_other"``, returns a decoupled layout
    where ``G_kept == G_obs - 1`` and the ``"_other"`` row is split
    out of the latent covariance.

    Parameters
    ----------
    n_genes : int
        Observation-axis length (count-data column count).
    correlate_other_column : bool
        From ``model_config.correlate_other_column``.  ``True`` opts
        into legacy behaviour even when an ``"_other"`` column is
        present.
    gene_names : Sequence[str], optional
        Gene names in observation-axis order.  Used to detect the
        trailing ``"_other"`` sentinel.  When ``None``, the layout
        falls back to assuming no trailing aggregate column.

    Returns
    -------
    AxisLayout
    """
    n_genes = int(n_genes)
    has_other_column = False
    if gene_names is not None and len(gene_names) == n_genes and n_genes > 0:
        has_other_column = str(gene_names[-1]) == _OTHER_NAME

    if correlate_other_column or not has_other_column:
        # Legacy / no-_other path: latent covariance spans the full
        # observation axis.  All indices identity-map.
        return AxisLayout(
            G_obs=n_genes,
            G_kept=n_genes,
            kept_idx=np.arange(n_genes, dtype=np.int64),
            other_idx=None,
        )

    # Decoupled path: drop the trailing '_other' row from the latent
    # covariance.  kept_idx is contiguous positions [0, G_obs - 1].
    return AxisLayout(
        G_obs=n_genes,
        G_kept=n_genes - 1,
        kept_idx=np.arange(n_genes - 1, dtype=np.int64),
        other_idx=n_genes - 1,
    )


__all__ = [
    "AxisLayout",
    "build_axis_layout",
    "_OTHER_NAME",
]

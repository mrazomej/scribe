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
    has_pooled_other: Optional[bool] = None,
) -> AxisLayout:
    """Construct an :class:`AxisLayout` from the data shape and config flag.

    When ``correlate_other_column=True`` (legacy behaviour) or when the
    data has no trailing ``"_other"`` column, returns the trivial
    layout where ``G_kept == G_obs`` and the latent covariance spans
    the full observation axis.

    When ``correlate_other_column=False`` (the default) AND a pooled
    ``"_other"`` column is present, returns a decoupled layout where
    ``G_kept == G_obs - 1`` and the ``"_other"`` row is split out of
    the latent covariance.

    Detection priority for the trailing ``"_other"`` column (auditor
    finding rev-2):

    1. If ``has_pooled_other`` is explicitly ``True``/``False``, use it.
       This is the **primary** signal for array-input fits where
       ``gene_names`` may be unavailable (no AnnData).
    2. Else if ``gene_names[-1] == "_other"``, infer ``True``.
    3. Else infer ``False``.

    When BOTH ``has_pooled_other`` and ``gene_names`` are supplied,
    they must agree (auditor finding rev-3, Medium).  Disagreement
    indicates metadata drift in the pipeline and is raised loudly as
    a ``ValueError`` — silent disagreement here can corrupt the axis
    split downstream.

    Parameters
    ----------
    n_genes : int
        Observation-axis length (count-data column count).
    correlate_other_column : bool
        From ``model_config.correlate_other_column``.  ``True`` opts
        into legacy behaviour even when an ``"_other"`` column is
        present.
    gene_names : Sequence[str], optional
        Gene names in observation-axis order.  Used as a fallback
        signal when ``has_pooled_other`` is ``None``.
    has_pooled_other : bool, optional
        Primary signal from the gene-coverage stage
        (``ctx._has_pooled_other`` or equivalent).  When ``True``, the
        trailing column is the pooled ``"_other"`` aggregate even if
        ``gene_names`` is unavailable (array-input fits) or carries a
        different label.  When ``False``, no decoupling regardless of
        ``gene_names``.  Default ``None`` defers to the names check.

    Returns
    -------
    AxisLayout

    Raises
    ------
    ValueError
        When ``has_pooled_other`` and the ``gene_names[-1] == "_other"``
        check disagree on whether the trailing column is pooled.
    """
    n_genes = int(n_genes)

    # Names-derived detection: True iff the last gene literal label is
    # the sentinel.  Only attempt when both names exist and the length
    # matches the data axis.
    names_say_other: Optional[bool] = None
    if gene_names is not None and len(gene_names) == n_genes and n_genes > 0:
        names_say_other = (str(gene_names[-1]) == _OTHER_NAME)

    # Contradictory-signal check: raise loudly when both signals are
    # present and disagree (auditor finding rev-3 Medium).  Skip this
    # check under ``correlate_other_column=True`` — the legacy path
    # always produces a trivial layout regardless of the signals, so
    # disagreement cannot corrupt the axis split (rev-5 #2: legacy
    # must never break, even when AnnData has a manually-named
    # ``_other`` tail without the ``gene_coverage`` stage running).
    # The check is only load-bearing on the decoupled path because
    # only there does the layout actually consume the signals to
    # split rows of W / d / x.
    if (
        not correlate_other_column
        and has_pooled_other is not None
        and names_say_other is not None
    ):
        if bool(has_pooled_other) != bool(names_say_other):
            raise ValueError(
                f"Contradictory '_other' signals: has_pooled_other="
                f"{bool(has_pooled_other)} but gene_names[-1]="
                f"{gene_names[-1]!r} "
                f"({'matches' if names_say_other else 'does not match'} "
                f"the _OTHER_NAME sentinel {_OTHER_NAME!r}). This "
                "indicates metadata drift between the gene-coverage "
                "stage and the gene-names array — silent disagreement "
                "can corrupt the axis-decoupling. Fix the calling "
                "pipeline to ensure both signals agree, or pass "
                "`correlate_other_column=True` to use the legacy "
                "trivial layout (in which the signals are advisory "
                "and disagreement is harmless)."
            )

    # Resolve the effective signal in priority order:
    #   has_pooled_other (explicit primary) > names-derived > False.
    if has_pooled_other is not None:
        has_other_column = bool(has_pooled_other)
    elif names_say_other is not None:
        has_other_column = bool(names_say_other)
    else:
        has_other_column = False

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

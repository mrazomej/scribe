"""SVI-results-to-empirical-Gaussian-priors adapter for NBLN Laplace.

This module derives informative Gaussian priors on the NBLN-Laplace
globals (``r``, ``mu``, ``eta_capture``) from a previously-fit SVI
results object on the same data. The priors enter the Laplace loss as
proper log-prob terms so their **uncertainty** (not just their location)
shapes both training dynamics and post-fit global Hessian.

Why this exists
---------------
The Laplace-EM path can diverge on the NBLN per-cell Newton when the
NB curvature ``(u + r) p (1 - p)`` collapses on low-count low-``r``
cells. The root cause is that ``r_loc``, ``mu``, and ``eta_loc`` are on
effectively flat priors in the current loss. An NBVCP-SVI fit on the
same data fits robustly and produces posterior samples that anchor
exactly the parameters NBLN needs to constrain.

Coordinate handling
-------------------
The adapter is responsible for moving SVI posterior samples from their
constrained (positive / [0, 1]) space into the target NBLN-Laplace
**unconstrained coordinate** used by ``params``:

- ``r``    : positive → unconstrained via ``model_config.positive_transform``
  inverse (``log`` for ``exp``, ``inv_softplus`` for ``softplus``).
- ``mu``   : positive NB mean → real-valued log-rate via plain ``jnp.log``,
  **regardless** of ``positive_transform``. The NBLN-Laplace ``params["mu"]``
  is the prior mean of a real-valued latent log-rate, not a positive
  parameter.
- ``eta`` : already in constrained [0, ∞) space — identity transform.

Architecture
------------
Two layers:

* :func:`fit_empirical_gaussian` — pure moment-match per coordinate.
  Knows nothing about scribe; expects samples already in the target
  coordinate. Reusable for any sample source (MCMC chains, hand-crafted
  bundles).

* :func:`priors_from_results` — adapter that knows scribe's results
  conventions and the NBLN coordinate mapping. Handles gene identity,
  capture-mode detection, amortization fallback, and coordinate
  conversion before calling the pure utility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from ._global_uncertainty import _JAX_POSITIVE_FNS

logger = logging.getLogger(__name__)


# Sentinel string for the trailing aggregated low-coverage column emitted
# by ``scribe.core.gene_coverage.aggregate_counts_by_mask``.  Centralized
# here so the subset-detection logic and the aggregators agree on the
# exact label the gene-coverage stage attaches to ``var_names``.
_OTHER_NAME = "_other"


# =====================================================================
# Layer 1 — pure moment-matcher
# =====================================================================


def fit_empirical_gaussian(
    samples_in_target_coord: jnp.ndarray,
    tau: float = 1.0,
    eps_scale: float = 1e-4,
) -> Dict[str, jnp.ndarray]:
    """Per-coordinate Gaussian moment-match from posterior samples.

    Computes ``loc = mean(samples)`` and ``scale = std(samples) * tau``
    along the leading sample axis, then floors ``scale`` at ``eps_scale``
    to prevent zero-variance priors from degenerate-posterior coordinates.

    This is a pure utility: it expects samples already expressed in the
    target coordinate (e.g., already log-transformed for the
    NBLN-Laplace ``r_loc`` slot). All coordinate-conversion lives in
    :func:`priors_from_results`.

    Parameters
    ----------
    samples_in_target_coord : jnp.ndarray
        Shape ``(S, *D)`` where ``S`` is the number of posterior samples
        and ``*D`` is the parameter's natural shape (e.g., ``(G,)`` for
        per-gene, ``(N,)`` for per-cell, ``()`` for scalar).
    tau : float, default 1.0
        Prior-temperature multiplier on the moment-matched scale. Larger
        values soften the prior so the downstream NBLN data has more
        freedom to override; ``tau=1.0`` trusts the SVI posterior exactly.
    eps_scale : float, default 1e-4
        Absolute floor on the post-``tau`` scale.

    Returns
    -------
    Dict[str, jnp.ndarray]
        ``{"loc": (...,), "scale": (...,)}`` — per-coordinate Normal
        parameters in the target coordinate space.
    """
    if samples_in_target_coord.ndim < 1:
        raise ValueError(
            "samples_in_target_coord must have a leading sample axis; "
            f"got shape {samples_in_target_coord.shape}."
        )
    if samples_in_target_coord.shape[0] < 2:
        raise ValueError(
            "Need at least 2 samples to estimate variance; got "
            f"{samples_in_target_coord.shape[0]}."
        )
    loc = jnp.mean(samples_in_target_coord, axis=0)
    raw_scale = jnp.std(samples_in_target_coord, axis=0, ddof=1)
    scale = jnp.maximum(raw_scale * float(tau), float(eps_scale))
    return {"loc": loc, "scale": scale}


# =====================================================================
# Layer 2 — results-to-priors adapter
# =====================================================================


def _resolve_target_pos_inverse(name: str):
    """Look up the inverse of the target ``positive_transform``."""
    if name not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown positive_transform={name!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )
    _forward, inverse = _JAX_POSITIVE_FNS[name]
    return inverse


def _try_results_gene_names(results: Any) -> Optional[np.ndarray]:
    """Defensive attribute chain for source gene-names.

    Tries (in order) ``results.var.index``, ``results.var_names``,
    ``results.adata.var_names``. Returns ``None`` if none of these
    are populated — caller falls back to mask or count check.
    """
    var = getattr(results, "var", None)
    if var is not None:
        idx = getattr(var, "index", None)
        if idx is not None:
            return np.asarray(idx)
    var_names = getattr(results, "var_names", None)
    if var_names is not None:
        return np.asarray(var_names)
    adata = getattr(results, "adata", None)
    if adata is not None:
        adata_vn = getattr(adata, "var_names", None)
        if adata_vn is not None:
            return np.asarray(adata_vn)
    return None


def _try_results_gene_mask_bool(results: Any) -> Optional[np.ndarray]:
    """Defensive attribute chain for a boolean gene-coverage mask.

    Returns a boolean mask in the **original** (pre-filter) gene-axis
    coordinate frame, or ``None`` when the source does not expose one.
    Strictly distinguished from an integer subset index — the latter
    lives in a different coordinate system and is handled by
    :func:`_try_results_gene_subset_index`.
    """
    mask = getattr(results, "_gene_coverage_mask", None)
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.dtype != bool:
        return None
    return mask_arr


def _try_results_gene_subset_index(results: Any) -> Optional[np.ndarray]:
    """Defensive attribute chain for an integer gene-subset index.

    Returns the integer index array used by sliced result objects, or
    ``None`` when the source does not expose one.  This is NOT a boolean
    coverage mask — see :func:`_try_results_gene_mask_bool`.
    """
    idx = getattr(results, "_subset_gene_index", None)
    if idx is None:
        return None
    return np.asarray(idx)


@dataclass(frozen=True)
class SubsetInfo:
    """Metadata describing the relationship between SVI source and Laplace target panels.

    Populated by :func:`_check_gene_identity` and consumed by the
    subset-aware aggregation helpers and PPC routing.  Use ``is_equal``
    to short-circuit to today's pass-through path and ``is_subset``
    (without ``is_equal``) to trigger aggregation.

    Attributes
    ----------
    is_equal : bool
        Panels match exactly (var_names or masks element-wise equal).
        Pass-through branch.
    is_subset : bool
        Target ⊆ source (``True`` even when ``is_equal`` is ``True``).
        Aggregation branch runs only when ``is_subset and not is_equal``.
    kept_idx_in_source : np.ndarray, optional
        Source-axis positions for the target's non-``_other`` genes, in
        target-axis order.  ``None`` for the equal-panel or count-only paths.
    dropped_idx_in_source : np.ndarray, optional
        Source-axis positions for genes the SVI kept individually but the
        Laplace target drops (i.e., the aggregator's contributing set).
        ``None`` for the equal-panel or count-only paths.
    source_has_other : bool
        Whether the source's last gene is the literal ``"_other"`` sentinel.
    target_has_other : bool
        Whether the target's last gene is the literal ``"_other"`` sentinel.
    source_other_index_in_source : int, optional
        Source-axis position of ``"_other"`` (``G_src - 1`` when present).
    target_other_index_in_target : int, optional
        Target-axis position of ``"_other"`` (``G_tgt - 1`` when present).
    """

    is_equal: bool
    is_subset: bool
    kept_idx_in_source: Optional[np.ndarray] = None
    dropped_idx_in_source: Optional[np.ndarray] = None
    source_has_other: bool = False
    target_has_other: bool = False
    source_other_index_in_source: Optional[int] = None
    target_other_index_in_target: Optional[int] = None


def _check_gene_identity(
    results: Any,
    target_n_genes: int,
    target_gene_names: Optional[np.ndarray],
    target_gene_mask: Optional[np.ndarray],
) -> Tuple[bool, str, SubsetInfo]:
    """Verify source and target gene panels match, or detect subset.

    Priority: var-names > boolean coverage mask > count-only-with-warning.

    Equal-panel detection short-circuits and returns
    ``is_equal=True, is_subset=True``.  Proper-subset detection populates
    ``kept_idx_in_source`` and ``dropped_idx_in_source`` so the aggregator
    can pool the SVI-only-kept genes into the Laplace target's ``"_other"``
    column.  Non-subset relationships raise ``ValueError``.

    Returns
    -------
    strict_var_name_verified : bool
        ``True`` when var-names matched element-wise.  Used by the
        amortized-capture safeguard to allow ``source_counts`` to feed
        the SVI encoder safely.
    identity_method : str
        One of ``"var_names"``, ``"mask"``, ``"count_only"``.
    subset_info : SubsetInfo
        Panel-relationship metadata.  ``is_equal=True`` for the
        pass-through path; ``is_subset=True, is_equal=False`` activates
        aggregation; non-subset relationships raise before returning.

    Raises
    ------
    ValueError
        On a verifiable non-subset relationship (target panel contains
        genes the source does not have, or target lacks an ``"_other"``
        column to receive the pooled signal).
    """
    source_n_genes = getattr(results, "n_genes", None)

    source_gene_names = _try_results_gene_names(results)
    if source_gene_names is not None and target_gene_names is not None:
        src_names = np.asarray(source_gene_names)
        tgt_names = np.asarray(target_gene_names)

        # Equal-panel short-circuit.  ``np.array_equal`` requires shape
        # equality, so this also handles the same-length case correctly.
        if np.array_equal(src_names, tgt_names):
            tgt_has_other = (
                tgt_names.size > 0 and str(tgt_names[-1]) == _OTHER_NAME
            )
            return True, "var_names", SubsetInfo(
                is_equal=True,
                is_subset=True,
                source_has_other=tgt_has_other,
                target_has_other=tgt_has_other,
                source_other_index_in_source=(
                    int(src_names.size - 1) if tgt_has_other else None
                ),
                target_other_index_in_target=(
                    int(tgt_names.size - 1) if tgt_has_other else None
                ),
            )

        # Proper-subset detection.  Distinguish the literal ``"_other"``
        # sentinel from real gene names on both sides — it labels a
        # pooled aggregate, not a real gene, and the two panels' ``"_other"``
        # columns mean DIFFERENT aggregates when the kept sets differ.
        src_has_other = (
            src_names.size > 0 and str(src_names[-1]) == _OTHER_NAME
        )
        tgt_has_other = (
            tgt_names.size > 0 and str(tgt_names[-1]) == _OTHER_NAME
        )
        src_real_names = src_names[:-1] if src_has_other else src_names
        tgt_real_names = tgt_names[:-1] if tgt_has_other else tgt_names

        src_set = set(map(str, src_real_names.tolist()))
        tgt_set = set(map(str, tgt_real_names.tolist()))

        missing = tgt_set - src_set
        if missing:
            preview = sorted(missing)[:10]
            raise ValueError(
                f"Laplace target gene panel is NOT a subset of the SVI "
                f"source's panel. Found {len(missing)} target genes "
                f"missing from source: {preview}. The cascade requires "
                "SVI_kept ⊇ Laplace_kept; re-fit SVI with a broader "
                "`gene_coverage` (`SVI gene_coverage >= Laplace "
                "gene_coverage`) or align the panels."
            )

        dropped = src_set - tgt_set
        if dropped and not tgt_has_other:
            preview = sorted(dropped)[:10]
            raise ValueError(
                f"SVI source has {len(dropped)} genes the Laplace target "
                f"drops (preview: {preview}), but the target has no "
                "trailing '_other' column to receive the pooled signal. "
                "Set `gene_coverage < 1.0` on the Laplace target so the "
                "coverage stage emits an '_other' aggregate column."
            )
        # Even with no extra dropped genes, the SVI's own '_other'
        # column needs a target slot to receive the aggregate.  When
        # the SVI has '_other' but the target does not, the aggregator
        # would silently produce wrong-shape arrays downstream.
        if src_has_other and not tgt_has_other:
            raise ValueError(
                "SVI source has an '_other' aggregate column but the "
                "Laplace target does not. The source's pooled aggregate "
                "has nowhere to go on the target gene axis. Set "
                "`gene_coverage < 1.0` on the Laplace target so the "
                "coverage stage emits an '_other' column."
            )

        # Build target-aligned kept/dropped source indices.
        src_pos_by_name = {
            str(n): int(i) for i, n in enumerate(src_real_names)
        }
        kept_idx = np.array(
            [src_pos_by_name[str(n)] for n in tgt_real_names],
            dtype=np.int64,
        )
        # ``dropped`` is unordered (set); take source-axis order for a
        # deterministic, reproducible aggregation.
        dropped_idx = np.array(
            [
                int(i)
                for i, n in enumerate(src_real_names)
                if str(n) not in tgt_set
            ],
            dtype=np.int64,
        )

        return True, "var_names", SubsetInfo(
            is_equal=False,
            is_subset=True,
            kept_idx_in_source=kept_idx,
            dropped_idx_in_source=dropped_idx,
            source_has_other=src_has_other,
            target_has_other=tgt_has_other,
            source_other_index_in_source=(
                int(src_names.size - 1) if src_has_other else None
            ),
            target_other_index_in_target=(
                int(tgt_names.size - 1) if tgt_has_other else None
            ),
        )

    # Boolean-mask path — requires BOTH sides to expose a boolean mask
    # in the SAME original-axis coordinate frame.  Integer
    # ``_subset_gene_index`` arrays are excluded here because they
    # cannot be interpreted as coverage masks without their reference
    # axis.
    source_mask = _try_results_gene_mask_bool(results)
    if source_mask is not None and target_gene_mask is not None:
        tgt_mask_arr = np.asarray(target_gene_mask)
        if tgt_mask_arr.dtype != bool or tgt_mask_arr.shape != source_mask.shape:
            # Cannot reliably compare — fall back to count check.
            pass
        elif np.array_equal(source_mask, tgt_mask_arr):
            return False, "mask", SubsetInfo(
                is_equal=True,
                is_subset=True,
                # Mask path does not know whether the trailing column
                # is ``"_other"``; the equal-panel branch never triggers
                # aggregation anyway, so leave the ``_other`` slots None.
            )
        elif np.all(tgt_mask_arr <= source_mask):
            # target_mask is True only where source_mask is True ⇒ subset.
            # The aggregator does not run on the mask-only path because
            # the kept/dropped source-axis indices depend on var-names
            # to disambiguate gene identity across the two panels.
            raise ValueError(
                "Subset-aware cascade via boolean coverage masks alone "
                "is not supported. Pass `target_gene_names` and ensure "
                "the SVI source exposes `var_names` so the aggregator "
                "can identify which SVI-kept genes are dropped by the "
                "Laplace target."
            )
        else:
            raise ValueError(
                "Source SVI fit and Laplace target disagree on the gene "
                "coverage mask, and the target is NOT a subset of the "
                "source. Re-fit SVI with a broader `gene_coverage` or "
                "align the panels."
            )

    # Count-only fallback path: refuse subset semantics outright; require
    # strict count equality (today's behavior preserved).
    if source_n_genes is None:
        logger.warning(
            "Could not verify gene identity beyond count — gene names "
            "and masks are unavailable on the source SVI results. "
            "Proceeding assuming the same gene panel was used in both fits."
        )
        return False, "count_only", SubsetInfo(
            is_equal=True, is_subset=True
        )
    if int(source_n_genes) != int(target_n_genes):
        raise ValueError(
            f"Source SVI fit and Laplace target disagree on genes "
            f"(n_src={int(source_n_genes)}, n_tgt={int(target_n_genes)}). "
            f"Did `gene_coverage` change between fits? Subset-aware "
            "cascading requires var-names on both sides; the count-only "
            "path can only verify equality."
        )
    logger.warning(
        "Could not verify gene identity beyond count — gene names and "
        "masks are unavailable on the source SVI results. Proceeding "
        "assuming the same gene panel was used in both fits."
    )
    return False, "count_only", SubsetInfo(
        is_equal=True, is_subset=True
    )


# =====================================================================
# Subset-aware aggregation helpers
# =====================================================================
#
# When the Laplace target's gene panel is a STRICT subset of the SVI
# source's panel (``SubsetInfo.is_subset and not is_equal``), the
# cascade must reconstruct a prior for the target's ``"_other"`` column
# by pooling the SVI's per-gene posteriors on the dropped genes (plus
# the SVI's own ``"_other"`` posterior if present).
#
# The math (per posterior sample s):
#
#     μ_other⁽ˢ⁾ = μ_svi_other⁽ˢ⁾ + Σ_{g ∈ dropped} μ_g⁽ˢ⁾
#
#     r_other⁽ˢ⁾ = (μ_other⁽ˢ⁾)² /
#                  [ (μ_svi_other⁽ˢ⁾)² / r_svi_other⁽ˢ⁾
#                    + Σ_{g ∈ dropped} (μ_g⁽ˢ⁾)² / r_g⁽ˢ⁾ ]
#
# This is moment matching: exact in the first two moments of the sum
# of independent NB variables (under the upstream's conditional
# independence assumption), but NOT distributionally exact — a sum of
# NBs with gene-specific p_g is not itself NB except in the shared-p
# (NBDM) limit.  Sufficient for anchor-prior purposes; documented as
# approximate in ``paper/_nb_lognormal.qmd`` §sec-nbln-cascade-aggregation.


def _assemble_per_gene_subset_samples(
    samples_source: jnp.ndarray,
    kept_idx_in_source: np.ndarray,
) -> jnp.ndarray:
    """Index source samples down to the target's kept (non-``_other``) axis.

    Parameters
    ----------
    samples_source : jnp.ndarray
        SVI posterior samples in constrained space, shape ``(S, G_src)``.
    kept_idx_in_source : np.ndarray
        Source-axis positions for the target's non-``_other`` genes, in
        target-axis order.  From :class:`SubsetInfo`.

    Returns
    -------
    jnp.ndarray
        Reordered subset samples, shape ``(S, len(kept_idx_in_source))``.
    """
    return samples_source[:, jnp.asarray(kept_idx_in_source)]


def _aggregate_other_nb(
    r_samples: jnp.ndarray,
    mu_samples: jnp.ndarray,
    dropped_idx_in_source: np.ndarray,
    source_other_index_in_source: Optional[int],
    *,
    eps_r: float = 1e-8,
    eps_mu: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-sample NB moment-match the pooled ``_other`` (r, μ) for NBLN.

    Returns ``(r_other_samples, mu_other_samples)``, each shape ``(S,)``,
    in **constrained** (positive) space.  The caller applies the
    coordinate conversion to NBLN's unconstrained target coordinate.

    This is MOMENT MATCHING: exact in the first two moments of the sum
    of independent NB variables under the upstream model's
    conditional-independence assumption.  It is NOT distributionally
    exact — a sum of independent NBs with gene-specific ``p_g`` is not
    itself NB except in the shared-``p`` NBDM limit.  Sufficient for
    anchor-prior purposes; see ``paper/_nb_lognormal.qmd``
    §sec-nbln-cascade-aggregation for the derivation and caveats.

    Parameters
    ----------
    r_samples : jnp.ndarray
        Source posterior samples of NB shape ``r``, constrained
        (positive), shape ``(S, G_src)``.
    mu_samples : jnp.ndarray
        Source posterior samples of NB mean ``μ``, constrained
        (positive), shape ``(S, G_src)``.
    dropped_idx_in_source : np.ndarray
        Source positions of SVI-kept genes the Laplace target drops.
        These contribute to the pooled aggregate.  Length 0 is allowed
        only when ``source_other_index_in_source`` is not ``None``.
    source_other_index_in_source : int, optional
        Position of the SVI's ``"_other"`` column on the source axis,
        or ``None`` when the SVI fit used ``gene_coverage == 1.0`` (no
        upstream pooling).
    eps_r, eps_mu : float
        Numerical floors before division/log; avoid degenerate samples
        from blowing up the aggregator.

    Returns
    -------
    r_other_samples, mu_other_samples : Tuple[jnp.ndarray, jnp.ndarray]
        Each shape ``(S,)``, both positive (constrained NB coordinate).
    """
    dropped_idx = jnp.asarray(dropped_idx_in_source, dtype=jnp.int32)
    has_drops = int(dropped_idx.size) > 0
    has_svi_other = source_other_index_in_source is not None

    if not has_drops and not has_svi_other:
        raise ValueError(
            "_aggregate_other_nb invoked with no contributing terms "
            "(neither dropped genes nor a SVI '_other' column). This is "
            "a programming error; the caller should have routed through "
            "the equal-panel pass-through path."
        )

    # Floor both ⟨r⟩ and ⟨μ⟩ so log/divide are safe even for degenerate
    # samples.  The floor is the same value used in the existing
    # per-gene coordinate conversion paths.
    r_safe = jnp.maximum(r_samples, eps_r)
    mu_safe = jnp.maximum(mu_samples, eps_mu)

    # Initialize per-sample accumulators with zeros, then add the SVI
    # '_other' contribution (if present) and the dropped-gene
    # contributions.  Working in constrained (positive) space.
    n_samples = int(r_safe.shape[0])
    mu_other = jnp.zeros((n_samples,), dtype=mu_safe.dtype)
    var_extra = jnp.zeros((n_samples,), dtype=mu_safe.dtype)

    if has_svi_other:
        # SVI's own '_other' column contributes one NB term per sample.
        # ⟨μ_other⟩ += ⟨μ_svi_other⟩ ;  Σ μ²/r += μ_svi_other² / r_svi_other.
        i_other = int(source_other_index_in_source)
        mu_other_svi = mu_safe[:, i_other]
        r_other_svi = r_safe[:, i_other]
        mu_other = mu_other + mu_other_svi
        var_extra = var_extra + (mu_other_svi ** 2) / r_other_svi

    if has_drops:
        # Per-sample sums over dropped genes, all kept in JAX-vectorized
        # form so this is a single matmul-like reduction per posterior
        # sample regardless of how many genes are pooled.
        mu_dropped = mu_safe[:, dropped_idx]  # (S, G_drop)
        r_dropped = r_safe[:, dropped_idx]    # (S, G_drop)
        mu_other = mu_other + jnp.sum(mu_dropped, axis=1)
        var_extra = var_extra + jnp.sum(
            (mu_dropped ** 2) / r_dropped, axis=1
        )

    # Recover r_other from the moment-matched variance excess.  Floor
    # ⟨μ_other⟩ to avoid 0/0 when every contributing μ is degenerate.
    mu_other_safe = jnp.maximum(mu_other, eps_mu)
    var_extra_safe = jnp.maximum(var_extra, eps_mu)
    r_other = (mu_other_safe ** 2) / var_extra_safe

    return r_other, mu_other


def _aggregate_other_tsln_rate(
    mu_samples: jnp.ndarray,
    burst_size_samples: jnp.ndarray,
    k_off_samples: jnp.ndarray,
    dropped_idx_in_source: np.ndarray,
    source_other_index_in_source: Optional[int],
    *,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-sample pooled ``_other`` for TSLN-rate.  Pragmatic, approximate.

    The two-state telegraph distribution does NOT close under
    summation: the sum of independent Poisson-Beta marginals is not
    Poisson-Beta.  So we cannot derive ``(burst_size_other, k_off_other)``
    from first principles.

    The compromise:
    • ``μ_other⁽ˢ⁾`` is the additive sum over the pooled set (exact).
    • ``burst_size_other⁽ˢ⁾`` and ``k_off_other⁽ˢ⁾`` inherit the SVI's
      ``"_other"`` posterior when one is present; otherwise we fall
      back to the per-sample median across the dropped genes — a vague
      default that does not pretend to be the closed-form aggregate.

    Documented in ``paper/_nb_lognormal.qmd`` as a known approximation.

    Returns
    -------
    (mu_other_samples, burst_size_other_samples, k_off_other_samples)
        Each shape ``(S,)``, all positive (constrained).
    """
    dropped_idx = jnp.asarray(dropped_idx_in_source, dtype=jnp.int32)
    has_drops = int(dropped_idx.size) > 0
    has_svi_other = source_other_index_in_source is not None

    if not has_drops and not has_svi_other:
        raise ValueError(
            "_aggregate_other_tsln_rate invoked with no contributing "
            "terms. The caller should have routed through the "
            "equal-panel pass-through path."
        )

    mu_safe = jnp.maximum(mu_samples, eps)
    bs_safe = jnp.maximum(burst_size_samples, eps)
    ko_safe = jnp.maximum(k_off_samples, eps)

    n_samples = int(mu_safe.shape[0])
    mu_other = jnp.zeros((n_samples,), dtype=mu_safe.dtype)

    if has_svi_other:
        i_other = int(source_other_index_in_source)
        mu_other = mu_other + mu_safe[:, i_other]
        bs_other = bs_safe[:, i_other]
        ko_other = ko_safe[:, i_other]
        if has_drops:
            mu_other = mu_other + jnp.sum(
                mu_safe[:, dropped_idx], axis=1
            )
    else:
        # No SVI '_other' to inherit from — use per-sample median across
        # the dropped genes as a vague default for the shape parameters,
        # and additive sum for the mean.
        mu_other = mu_other + jnp.sum(mu_safe[:, dropped_idx], axis=1)
        bs_other = jnp.median(bs_safe[:, dropped_idx], axis=1)
        ko_other = jnp.median(ko_safe[:, dropped_idx], axis=1)

    return mu_other, bs_other, ko_other


def _detect_capture_mode(samples: Dict[str, jnp.ndarray]) -> str:
    """Detect the source's capture parameterization from sample keys.

    Returns one of ``"eta"``, ``"phi_only"``, ``"none"``.
    """
    if "eta_capture" in samples:
        return "eta"
    if "phi_capture" in samples or "p_capture" in samples:
        return "phi_only"
    return "none"


def _is_amortized_capture(results: Any) -> bool:
    """Detect amortized-capture SVI sources."""
    if hasattr(results, "_uses_amortized_capture"):
        try:
            return bool(results._uses_amortized_capture())
        except Exception:
            return False
    return False


def _draw_samples(
    results: Any,
    n_samples: int,
    source_counts: Optional[jnp.ndarray],
    strict_var_name_verified: bool,
    rng_seed: int,
) -> Dict[str, jnp.ndarray]:
    """Draw posterior samples from SVI results with amortization safeguards.

    Handles amortized-capture sources with the round-6 audit fallback:
    prefer ``results._original_counts``; else accept ``source_counts``
    only when strict var-name identity has been verified; else refuse
    with an explicit error message listing remediation options.
    """
    from jax import random

    rng_key = random.PRNGKey(int(rng_seed))

    if _is_amortized_capture(results):
        # Defensive fallback hierarchy — see plan Round-6 Finding 1.
        svi_source_counts = getattr(results, "_original_counts", None)
        if svi_source_counts is None:
            svi_source_counts = getattr(results, "counts", None)
        if svi_source_counts is not None:
            counts_for_encoder = svi_source_counts
        elif strict_var_name_verified and source_counts is not None:
            counts_for_encoder = source_counts
        else:
            raise ValueError(
                "Source SVI results use amortized capture, but the "
                "encoder's training counts could not be reconstructed "
                "safely. Either:\n"
                "  (a) re-fit SVI with a non-amortized capture "
                "parameterization, OR\n"
                "  (b) ensure the SVI results object stores its training "
                "counts (e.g., via `_original_counts`), OR\n"
                "  (c) pass `source_counts=` explicitly AND ensure both "
                "fits used identical gene panels (var_names match) — "
                "only then is the amortizer's input shape guaranteed "
                "correct."
            )
        return results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=int(n_samples),
            counts=counts_for_encoder,
            store_samples=False,
        )

    return results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=int(n_samples),
        store_samples=False,
    )


def priors_from_results(
    results: Any,
    target_positive_transform: str,
    target_n_genes: int,
    target_n_cells: int,
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    n_samples: int = 1000,
    tau: float = 1.0,
    rng_seed: int = 0,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, jnp.ndarray]], str]:
    """Adapter: SVI results object → empirical Gaussian prior bundle.

    Builds Gaussian priors on the NBLN-Laplace globals ``r``, ``mu``,
    and ``eta`` (when available) from posterior samples drawn from an
    SVI results object. The priors live in the target NBLN-Laplace's
    unconstrained coordinate space and are intended to enter the loss
    as ``dist.Normal(loc, scale).log_prob(params[...])`` terms.

    See the module docstring for the coordinate-mapping table.

    Parameters
    ----------
    results
        Scribe SVI results object with ``get_posterior_samples()`` and,
        ideally, ``var_names`` / ``_gene_coverage_mask`` / ``n_genes``
        attributes for identity verification.
    target_positive_transform : {"exp", "softplus"}
        Resolved from the target ``model_config.positive_transform``.
        Used **only** for the ``r`` coordinate (positive parameter).
        Not used for ``mu`` (which is unconstrained log-rate) or ``eta``
        (already in the target's constrained space).
    target_n_genes, target_n_cells : int
        Target dataset shape — checked against the source.
    target_gene_names : Optional[np.ndarray]
        Target var-names array, if available. Enables strict gene
        identity verification.
    target_gene_mask : Optional[np.ndarray]
        Target ``_gene_coverage_mask``, used as a fallback identity
        signal when var-names are unavailable on either side.
    source_counts : Optional[jnp.ndarray]
        Target's count matrix, passed opportunistically for amortized-
        capture SVI sources. Honored only when strict var-name identity
        was verified.
    n_samples : int, default 1000
        Number of posterior samples to draw for moment matching.
    tau : float, default 1.0
        Prior-temperature multiplier on every prior scale.
    rng_seed : int, default 0
        JAX PRNG seed for sample-drawing reproducibility.
    verbose : bool, default True
        Print user-facing progress messages to stdout at each stage
        (gene-identity verification, SVI sampling, capture-mode
        detection, per-parameter moment-matching, bundle summary).
        Set ``False`` to silence when wrapping in larger pipelines.

    Returns
    -------
    prior_bundle : Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc": ..., "scale": ...}`` dicts. Keys are a
        subset of ``{"r", "mu", "eta"}`` depending on what the source
        provides.
    capture_mode : str
        One of ``"eta"``, ``"phi_only"``, ``"none"``. Used by the
        upstream caller to decide whether to override the target's
        scalar ``capture_anchor``.

    Raises
    ------
    ValueError
        On gene-identity mismatch, on amortized-capture source without
        reconstructible training counts, or when the source provides
        per-cell ``eta_capture`` of the wrong shape.
    """
    if target_positive_transform not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown target_positive_transform={target_positive_transform!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Building informative priors from SVI source "
        f"(target G={int(target_n_genes)}, N={int(target_n_cells)}, "
        f"tau={float(tau):.2f}, n_samples={int(n_samples)})"
    )

    # --- Gene-identity safeguard --------------------------------------
    _say("Verifying gene identity against target...")
    strict_var_name_verified, identity_method, subset_info = (
        _check_gene_identity(
            results=results,
            target_n_genes=int(target_n_genes),
            target_gene_names=target_gene_names,
            target_gene_mask=target_gene_mask,
        )
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    # Subset-aware path detection.  When the target's panel is a STRICT
    # subset of the source's, the cascade reconstructs the target's
    # ``"_other"`` prior by per-sample NB moment-matching over the
    # SVI's dropped per-gene posteriors (and the SVI's own ``"_other"``
    # if present).  See paper section ``sec-nbln-cascade-aggregation``.
    _subset_active = (
        subset_info.is_subset and not subset_info.is_equal
    )
    if _subset_active and _is_amortized_capture(results):
        raise NotImplementedError(
            "Subset-aware cascade (SVI panel ⊃ Laplace panel) is not "
            "supported with amortized-capture SVI sources. The encoder "
            "needs source-shape counts and cannot be safely evaluated "
            "on the target's smaller panel. Workarounds: refit SVI with "
            "non-amortized capture; or align the gene panels."
        )
    if _subset_active:
        _dropped_n = int(
            0 if subset_info.dropped_idx_in_source is None else subset_info.dropped_idx_in_source.size
        )
        _say(
            f"Subset-aware cascade active: NBLN '_other' prior "
            f"moment-matched across {_dropped_n} SVI per-gene "
            f"posteriors ({'+' if subset_info.source_has_other else 'no'} "
            "SVI '_other')."
        )

    # --- Draw samples (with amortized-capture handling) ---------------
    if _is_amortized_capture(results):
        _say(
            "Sampling SVI posterior distribution "
            "(amortized capture detected; consulting encoder)..."
        )
    else:
        _say("Sampling SVI posterior distribution...")
    samples = _draw_samples(
        results=results,
        n_samples=int(n_samples),
        source_counts=source_counts,
        strict_var_name_verified=strict_var_name_verified,
        rng_seed=int(rng_seed),
    )
    _say(
        f"  Drew {int(n_samples)} samples; available keys: "
        f"{sorted(samples.keys())}"
    )

    # --- Capture-mode detection ---------------------------------------
    capture_mode = _detect_capture_mode(samples)
    _say(f"Detected capture mode: {capture_mode!r}")
    if capture_mode == "none":
        logger.warning(
            "SVI source has no capture latent (no eta_capture, phi_capture, "
            "or p_capture key). r and mu priors will be applied; the "
            "target's capture configuration is left intact."
        )

    pos_inverse = _resolve_target_pos_inverse(target_positive_transform)
    prior_bundle: Dict[str, Dict[str, jnp.ndarray]] = {}

    _say("Fitting empirical Gaussian priors to posterior samples...")

    # When the subset path is active we need BOTH ``r`` and ``mu``
    # samples to evaluate the moment-matching formula
    # ``r_other = μ_other² / Σ μ_g² / r_g``.  The aggregator cannot run
    # without both, so we require both keys here even though the
    # equal-panel path treats them independently.
    if _subset_active:
        if "r" not in samples or "mu" not in samples:
            raise ValueError(
                "Subset-aware cascade requires the SVI source to expose "
                "both 'r' and 'mu' per-sample. The moment-matching "
                "formula for the pooled '_other' column uses ⟨μ²/r⟩ "
                "per posterior sample."
            )

    # --- r prior: positive → target unconstrained ---------------------
    if "r" in samples:
        r_samples = jnp.asarray(samples["r"])
        # SVI may return shape (S, G) or (S, 1) for scalar-r models.
        if r_samples.ndim < 2:
            raise ValueError(
                f"Expected r samples to have shape (S, G); got "
                f"{r_samples.shape}."
            )
        if _subset_active:
            # Source has G_src columns; the aggregator pools the
            # SVI-only-kept genes into a single trailing column aligned
            # to the target's ``"_other"`` slot.
            mu_samples_for_agg = jnp.asarray(samples["mu"])
            r_kept = _assemble_per_gene_subset_samples(
                r_samples, subset_info.kept_idx_in_source
            )
            r_other_s, _mu_other_s = _aggregate_other_nb(
                r_samples,
                mu_samples_for_agg,
                subset_info.dropped_idx_in_source,
                subset_info.source_other_index_in_source,
            )
            r_full = jnp.concatenate(
                [r_kept, r_other_s[:, None]], axis=1
            )
            if r_full.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"Subset aggregation produced r samples of shape "
                    f"{r_full.shape}; target expects "
                    f"(S, {int(target_n_genes)})."
                )
            r_samples_for_fit = r_full
        else:
            if r_samples.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"Source r samples have {r_samples.shape[1]} genes; "
                    f"target expects {int(target_n_genes)}."
                )
            r_samples_for_fit = r_samples
        # Floor at small positive value to avoid log(0) / inv_softplus(0).
        r_pos = jnp.maximum(r_samples_for_fit, 1e-8)
        r_unconstrained = pos_inverse(r_pos)
        prior_bundle["r"] = fit_empirical_gaussian(
            r_unconstrained, tau=float(tau)
        )
        _say(
            f"  Fitted r prior (G={int(r_samples_for_fit.shape[1])}, "
            f"transform={target_positive_transform!r} inverse"
            f"{', subset-aware aggregation' if _subset_active else ''})."
        )

    # --- mu prior: positive NB mean → log-rate (real-valued) ----------
    if "mu" in samples:
        mu_samples = jnp.asarray(samples["mu"])
        if mu_samples.ndim < 2:
            raise ValueError(
                f"Expected mu samples to have shape (S, G); got "
                f"{mu_samples.shape}."
            )
        if _subset_active:
            r_samples_for_agg = jnp.asarray(samples["r"])
            mu_kept = _assemble_per_gene_subset_samples(
                mu_samples, subset_info.kept_idx_in_source
            )
            _r_other_s, mu_other_s = _aggregate_other_nb(
                r_samples_for_agg,
                mu_samples,
                subset_info.dropped_idx_in_source,
                subset_info.source_other_index_in_source,
            )
            mu_full = jnp.concatenate(
                [mu_kept, mu_other_s[:, None]], axis=1
            )
            if mu_full.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"Subset aggregation produced mu samples of shape "
                    f"{mu_full.shape}; target expects "
                    f"(S, {int(target_n_genes)})."
                )
            mu_samples_for_fit = mu_full
        else:
            if mu_samples.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"Source mu samples have {mu_samples.shape[1]} genes; "
                    f"target expects {int(target_n_genes)}."
                )
            mu_samples_for_fit = mu_samples
        # IMPORTANT: NBLN-Laplace `params["mu"]` is the prior mean of an
        # unconstrained real-valued log-rate latent — not a positive
        # parameter. So the coordinate conversion is plain log(mu), NOT
        # pos_inverse(mu), regardless of model_config.positive_transform.
        mu_pos = jnp.maximum(mu_samples_for_fit, 1e-8)
        mu_log = jnp.log(mu_pos)
        prior_bundle["mu"] = fit_empirical_gaussian(mu_log, tau=float(tau))
        _say(
            f"  Fitted mu prior (G={int(mu_samples_for_fit.shape[1])}, "
            f"transform='log' — NBLN mu is real-valued log-rate"
            f"{', subset-aware aggregation' if _subset_active else ''})."
        )

    # --- eta prior: identity (already in target's [0, ∞) space) -------
    if capture_mode == "eta":
        eta_samples = jnp.asarray(samples["eta_capture"])
        if eta_samples.ndim < 2:
            raise ValueError(
                f"Expected eta_capture samples to have shape (S, N); got "
                f"{eta_samples.shape}."
            )
        if eta_samples.shape[1] != int(target_n_cells):
            raise ValueError(
                f"Source eta_capture samples have {eta_samples.shape[1]} "
                f"cells; target expects {int(target_n_cells)}."
            )
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        _say(
            f"  Fitted eta prior (N={int(eta_samples.shape[1])}, "
            "transform='identity' — constrained [0, ∞) matches target)."
        )
    elif capture_mode == "phi_only":
        # Convert odds-ratio capture (p/phi) to NBLN eta_capture using:
        #   eta = -log(p_capture), where p_capture = phi / (1 + phi).
        if "p_capture" in samples:
            p_cap_samples = jnp.asarray(samples["p_capture"])
            p_cap_samples = jnp.clip(p_cap_samples, 1e-8, 1.0 - 1e-8)
        elif "phi_capture" in samples:
            phi_samples = jnp.maximum(jnp.asarray(samples["phi_capture"]), 1e-8)
            p_cap_samples = phi_samples / (1.0 + phi_samples)
            p_cap_samples = jnp.clip(p_cap_samples, 1e-8, 1.0 - 1e-8)
        else:
            raise ValueError(
                "Detected capture_mode='phi_only' but neither 'p_capture' "
                "nor 'phi_capture' samples are available."
            )
        if p_cap_samples.ndim < 2:
            raise ValueError(
                f"Expected p_capture/phi_capture samples to have shape "
                f"(S, N); got {p_cap_samples.shape}."
            )
        if p_cap_samples.shape[1] != int(target_n_cells):
            raise ValueError(
                f"Source p_capture/phi_capture samples have "
                f"{p_cap_samples.shape[1]} cells; target expects "
                f"{int(target_n_cells)}."
            )
        eta_samples = -jnp.log(p_cap_samples)
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        # Promote to eta so downstream uses per-cell cascade eta on target.
        capture_mode = "eta"
        _say(
            f"  Mapped p_capture/phi_capture → eta_capture per cell "
            f"(eta = −log p, N={int(eta_samples.shape[1])}). "
            "Promoted capture_mode 'phi_only' → 'eta'."
        )

    _say(
        f"Built informative prior bundle: keys={sorted(prior_bundle.keys())}, "
        f"capture_mode={capture_mode!r}, identity_method={identity_method!r}, "
        f"n_samples={int(n_samples)}, tau={float(tau):.2f}."
    )
    return prior_bundle, capture_mode


__all__ = [
    "fit_empirical_gaussian",
    "priors_from_results",
    "freeze_values_from_results",
    "priors_from_twostate_results",
    "freeze_values_from_twostate_results",
    "SubsetInfo",
    "_check_gene_identity",
    "_aggregate_other_nb",
    "_aggregate_other_tsln_rate",
    "_assemble_per_gene_subset_samples",
]


# =====================================================================
# Layer 3 — freeze-value extractor (point estimates, no moment-matching)
# =====================================================================


def freeze_values_from_results(
    results: Any,
    target_positive_transform: str,
    target_n_genes: int,
    target_n_cells: int,
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    freeze_params: Tuple[str, ...] = ("r", "eta"),
    map_method: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Extract point-estimate freeze values from an SVI results object.

    Unlike :func:`priors_from_results` (which moment-matches posterior
    samples into Gaussian priors for the soft-cascade loss term), this
    function extracts a single point per coordinate from the SVI
    variational MAP and converts to the NBLN-Laplace target coordinate.

    The freeze values are the **fixed values** used during NBLN's M-step
    when the corresponding parameter is in ``freeze_params``.  No
    moment-matching, no MC error from sampling — the point estimate is
    directly what SVI converged to.

    For the **reported posterior** on frozen parameters, downstream code
    consults the full SVI guide via the embedded ``cascade_source`` field
    on the Laplace result (see :class:`ScribeLaplaceResults`).  This
    function only extracts the M-step point estimate.

    Parameters
    ----------
    results
        Scribe SVI results object with ``get_map()`` and gene-identity
        metadata.
    target_positive_transform : {"exp", "softplus"}
        Resolved from the target ``model_config.positive_transform``.
        Used only for the ``r`` coordinate.
    target_n_genes, target_n_cells : int
        Target dataset shape — checked against the source.
    target_gene_names : Optional[np.ndarray]
        Target var-names array for strict gene identity verification.
    target_gene_mask : Optional[np.ndarray]
        Target gene coverage mask for fallback identity verification.
    source_counts : Optional[jnp.ndarray]
        Target count matrix, passed for amortized-capture SVI sources.
        Same three-tier defensive hierarchy as
        :func:`priors_from_results`.
    freeze_params : Tuple[str, ...], default ("r", "eta")
        Which parameters to extract freeze values for.  Valid keys are
        a subset of ``{"r", "mu", "eta"}``.
    verbose : bool
        Whether to print user-facing progress messages.

    Returns
    -------
    Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc": ...}`` dicts (no ``scale`` — point
        estimates only).  Keys are the requested subset of
        ``{"r", "mu", "eta"}`` that the SVI source can supply.

    Raises
    ------
    ValueError
        On gene-identity mismatch, on amortized-capture source without
        reconstructible training counts, or when a requested freeze
        parameter is absent from the SVI source.
    """
    if target_positive_transform not in _JAX_POSITIVE_FNS:
        raise ValueError(
            f"Unknown target_positive_transform={target_positive_transform!r}; "
            f"expected one of {set(_JAX_POSITIVE_FNS)}."
        )
    valid = {"r", "mu", "eta"}
    invalid = set(freeze_params) - valid
    if invalid:
        raise ValueError(
            f"freeze_params has invalid keys {invalid}; valid = {valid}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Extracting freeze values from SVI source "
        f"(freeze_params={list(freeze_params)})"
    )

    # --- Gene-identity safeguard (reuses the priors_from_results helper) ---
    strict_var_name_verified, identity_method, subset_info = (
        _check_gene_identity(
            results=results,
            target_n_genes=int(target_n_genes),
            target_gene_names=target_gene_names,
            target_gene_mask=target_gene_mask,
        )
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    _subset_active = (
        subset_info.is_subset and not subset_info.is_equal
    )
    if _subset_active and _is_amortized_capture(results):
        raise NotImplementedError(
            "Subset-aware freeze cascade (SVI panel ⊃ Laplace panel) "
            "is not supported with amortized-capture SVI sources."
        )
    if _subset_active:
        _dropped_n = int(
            0 if subset_info.dropped_idx_in_source is None else subset_info.dropped_idx_in_source.size
        )
        _say(
            f"Subset-aware freeze cascade active: aggregating "
            f"{_dropped_n} SVI per-gene MAPs into the target '_other' "
            f"slot ({'+' if subset_info.source_has_other else 'no'} "
            "SVI '_other')."
        )

    # --- Amortized-capture-aware get_map() call ---
    # Same defensive hierarchy as priors_from_results._draw_samples:
    # prefer results._original_counts; else accept source_counts only
    # when strict var-name identity verified; else refuse.
    if _is_amortized_capture(results):
        _say(
            "  SVI source uses amortized capture; resolving counts for "
            "encoder evaluation..."
        )
        svi_source_counts = getattr(results, "_original_counts", None)
        if svi_source_counts is None:
            svi_source_counts = getattr(results, "counts", None)
        if svi_source_counts is not None:
            counts_for_encoder = svi_source_counts
        elif strict_var_name_verified and source_counts is not None:
            counts_for_encoder = source_counts
        else:
            raise ValueError(
                "Source SVI results use amortized capture, but the "
                "encoder's training counts could not be reconstructed "
                "safely for get_map(). Same remediation options as "
                "priors_from_results: refit SVI with non-amortized "
                "capture, store training counts on the SVI result, or "
                "pass source_counts with strict var-name identity."
            )
        # CASCADE PROPAGATION: see analogous block in
        # freeze_values_from_twostate_results for the rationale.
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(
            counts=counts_for_encoder, verbose=False, **_map_kwargs
        )
    else:
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(verbose=False, **_map_kwargs)

    _say(f"  SVI MAP keys: {sorted(map_dict.keys())}")

    pos_inverse = _resolve_target_pos_inverse(target_positive_transform)
    freeze_values: Dict[str, Dict[str, jnp.ndarray]] = {}

    # Subset-aware freeze needs both ``r`` and ``mu`` MAPs together
    # (the moment-match formula couples them).  Require both keys
    # explicitly when the subset path is active.
    if _subset_active and (
        ("r" in freeze_params and "r" not in map_dict)
        or ("mu" in freeze_params and "mu" not in map_dict)
    ):
        raise ValueError(
            "Subset-aware freeze cascade requires both 'r' and 'mu' "
            "MAPs from the SVI source; the moment-matching formula "
            "couples them. Available MAP keys: "
            f"{sorted(map_dict.keys())}."
        )

    def _aggregate_other_nb_map(
        r_map: jnp.ndarray, mu_map: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """MAP-only NB aggregation (single sample wrapper).

        Wraps the per-sample helper by injecting a singleton sample
        axis, calling :func:`_aggregate_other_nb`, and squeezing the
        sample axis back out.  Returns scalar arrays for ⟨r_other⟩
        and ⟨μ_other⟩ in constrained (positive) space.
        """
        r_other_s, mu_other_s = _aggregate_other_nb(
            r_map[None, :],
            mu_map[None, :],
            subset_info.dropped_idx_in_source,
            subset_info.source_other_index_in_source,
        )
        return r_other_s[0], mu_other_s[0]

    # --- r: positive → NBLN unconstrained (via pos_inverse) ---
    if "r" in freeze_params:
        if "r" not in map_dict:
            raise ValueError(
                "freeze_params requests 'r' but SVI source's get_map() "
                "does not include an 'r' key. "
                f"Available keys: {sorted(map_dict.keys())}."
            )
        r_pos = jnp.asarray(map_dict["r"])
        if _subset_active:
            # Per-MAP aggregation; subset and append moment-matched
            # ``_other`` entry at the target's trailing slot.
            mu_map_for_agg = jnp.asarray(map_dict["mu"])
            if r_pos.ndim != 1 or r_pos.shape[0] == 0:
                raise ValueError(
                    f"SVI 'r' MAP has shape {r_pos.shape}; expected "
                    "1-D source-axis array for subset-aware aggregation."
                )
            r_kept = r_pos[jnp.asarray(subset_info.kept_idx_in_source)]
            r_other, _mu_other = _aggregate_other_nb_map(
                r_pos, mu_map_for_agg
            )
            r_full = jnp.concatenate(
                [r_kept, r_other[None]], axis=0
            )
            if r_full.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"Subset aggregation produced r MAP of shape "
                    f"{r_full.shape}; target expects "
                    f"({int(target_n_genes)},)."
                )
            r_for_freeze = r_full
        else:
            if r_pos.ndim != 1 or r_pos.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"SVI 'r' MAP has shape {r_pos.shape}; expected "
                    f"({int(target_n_genes)},)."
                )
            r_for_freeze = r_pos
        r_uncon = pos_inverse(jnp.maximum(r_for_freeze, 1e-8))
        freeze_values["r"] = {"loc": r_uncon}
        _say(
            f"  Extracted r freeze value (G={target_n_genes}, "
            f"transform={target_positive_transform!r} inverse"
            f"{', subset-aware aggregation' if _subset_active else ''})."
        )

    # --- mu: positive NB mean → NBLN log-rate (via jnp.log) ---
    if "mu" in freeze_params:
        if "mu" not in map_dict:
            raise ValueError(
                "freeze_params requests 'mu' but SVI source's get_map() "
                "does not include a 'mu' key. "
                f"Available keys: {sorted(map_dict.keys())}."
            )
        mu_pos = jnp.asarray(map_dict["mu"])
        if _subset_active:
            r_map_for_agg = jnp.asarray(map_dict["r"])
            if mu_pos.ndim != 1 or mu_pos.shape[0] == 0:
                raise ValueError(
                    f"SVI 'mu' MAP has shape {mu_pos.shape}; expected "
                    "1-D source-axis array for subset-aware aggregation."
                )
            mu_kept = mu_pos[jnp.asarray(subset_info.kept_idx_in_source)]
            _r_other, mu_other = _aggregate_other_nb_map(
                r_map_for_agg, mu_pos
            )
            mu_full = jnp.concatenate(
                [mu_kept, mu_other[None]], axis=0
            )
            if mu_full.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"Subset aggregation produced mu MAP of shape "
                    f"{mu_full.shape}; target expects "
                    f"({int(target_n_genes)},)."
                )
            mu_for_freeze = mu_full
        else:
            if mu_pos.ndim != 1 or mu_pos.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"SVI 'mu' MAP has shape {mu_pos.shape}; expected "
                    f"({int(target_n_genes)},)."
                )
            mu_for_freeze = mu_pos
        mu_log = jnp.log(jnp.maximum(mu_for_freeze, 1e-8))
        freeze_values["mu"] = {"loc": mu_log}
        _say(
            f"  Extracted mu freeze value (G={target_n_genes}, "
            f"transform='log' — NBLN mu is real-valued log-rate"
            f"{', subset-aware aggregation' if _subset_active else ''})."
        )

    # --- eta: constrained [0, ∞) → identity (NBLN's coord is the same) ---
    if "eta" in freeze_params:
        if "eta_capture" in map_dict:
            eta = jnp.asarray(map_dict["eta_capture"])
        elif "p_capture" in map_dict:
            p_cap = jnp.asarray(map_dict["p_capture"])
            p_cap = jnp.clip(p_cap, 1e-8, 1.0 - 1e-8)
            eta = -jnp.log(p_cap)
            _say(
                "  Mapped p_capture MAP → eta_capture freeze value "
                "via eta = −log p."
            )
        elif "phi_capture" in map_dict:
            phi = jnp.maximum(jnp.asarray(map_dict["phi_capture"]), 1e-8)
            p_cap = phi / (1.0 + phi)
            p_cap = jnp.clip(p_cap, 1e-8, 1.0 - 1e-8)
            eta = -jnp.log(p_cap)
            _say(
                "  Mapped phi_capture MAP → eta_capture freeze value "
                "via p = phi/(1+phi), eta = −log p."
            )
        else:
            raise ValueError(
                "freeze_params requests 'eta' but SVI source's get_map() "
                "does not include any capture key among "
                "('eta_capture', 'p_capture', 'phi_capture'). "
                f"Available keys: {sorted(map_dict.keys())}."
            )
        if eta.ndim != 1 or eta.shape[0] != int(target_n_cells):
            raise ValueError(
                f"SVI capture MAP has shape {eta.shape}; expected "
                f"({int(target_n_cells)},)."
            )
        freeze_values["eta"] = {"loc": eta}
        _say(
            f"  Extracted eta freeze value (N={target_n_cells}, "
            "transform='identity' — constrained [0, ∞) matches target)."
        )

    _say(
        f"Built freeze-values bundle: keys={sorted(freeze_values.keys())}, "
        f"requested={list(freeze_params)!r}, identity_method={identity_method!r}."
    )
    return freeze_values


# =====================================================================
# TwoState-LogNormal cascade (TSLN-Rate / TSLN-Logit) adapter
# =====================================================================
#
# Mirrors ``priors_from_results`` / ``freeze_values_from_results`` for
# the TSLN family, with the TwoState SVI source's parameter set:
#
#   SVI source emits (per gene): mu, burst_size, k_off
#                  + deterministics: alpha (= k_on = mu/burst_size),
#                                    beta (= k_off),
#                                    r_hat (= mu + burst_size * k_off),
#                                    eta_act (= log(alpha/beta)) [PR-2 only]
#
# Coordinate maps:
#
#   TSLN-Rate target (PR-1):
#     mu_loc          : pos_inverse(mu_pos)
#     burst_size_loc  : pos_inverse(burst_size_pos)
#     k_off_loc       : pos_inverse(k_off_pos)
#     eta             : identity (per-cell capture)
#
#   TSLN-Logit target (PR-2, deferred):
#     rate_loc        : pos_inverse(r_hat) [prefer deterministic]
#                       or pos_inverse(mu + burst_size * k_off)
#     kappa_loc       : pos_inverse(alpha + beta)
#                       or pos_inverse(mu/burst_size + k_off)
#     eta_anchor_loc  : eta_act (real-valued)
#                       or log(mu/burst_size) - log(k_off)
#     eta             : identity
#
# Both ``target_variant="rate"`` (PR-1) and ``"logit"`` (PR-2) are
# implemented.  The logit branch prefers the SVI source's effective
# ``alpha``/``beta``/``r_hat``/``eta_act`` deterministics over raw
# (mu, burst_size, k_off) derivation — see the inline documentation
# in the function body for the Rev 4 motivation.


def priors_from_twostate_results(
    results: Any,
    target_positive_transform,
    target_n_genes: int,
    target_n_cells: int,
    target_variant: str = "rate",
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    n_samples: int = 1000,
    tau: float = 1.0,
    rng_seed: int = 0,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, jnp.ndarray]], str]:
    """Adapter: TwoState SVI results → TSLN empirical Gaussian prior bundle.

    Builds Gaussian priors on the TSLN-Laplace globals from posterior
    samples drawn from an upstream TwoState SVI fit. The priors live in
    the TSLN target's unconstrained coordinate space and are intended
    to enter the Laplace loss as ``Normal(loc, scale).log_prob(...)``
    terms.

    Parameters
    ----------
    results
        Scribe SVI results from a ``model="twostate"`` (or
        ``"twostatevcp"``) fit. Must expose ``get_posterior_samples()``
        and ideally var-names / gene-mask metadata.
    target_positive_transform : str or Dict[str, str]
        Either a single transform name (e.g. ``"softplus"`` /
        ``"exp"``) applied uniformly to every positive parameter, OR
        a mapping from internal parameter name to transform name.
        Recognized keys: ``{"mu", "burst_size", "k_off"}`` for the
        rate variant; ``{"rate", "kappa"}`` for the logit variant
        (``eta_anchor`` is real-valued — identity transform).  Missing
        keys fall back to ``"softplus"``.  Pass a dict whenever the
        target's ``model_config.positive_transform`` is itself the
        dict form (e.g.
        ``positive_transform={"rate": "exp", "kappa": "softplus"}``)
        — using a single string in that case would silently apply the
        wrong inverse and corrupt the cascade priors.
    target_n_genes, target_n_cells : int
        Target dataset shape.
    target_variant : {"rate", "logit"}, default ``"rate"``
        Which TSLN target variant to build priors for.  Both are
        implemented; ``"logit"`` (PR-2) prefers the SVI source's
        effective deterministics (``alpha`` / ``beta`` / ``r_hat`` /
        ``eta_act``) over raw ``(mu, burst_size, k_off)`` derivation
        per Rev 4.
    n_samples : int, default 1000
    tau : float, default 1.0
    rng_seed : int, default 0
    verbose : bool, default True

    Returns
    -------
    prior_bundle : Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc", "scale"}`` dicts. Keys depend on
        ``target_variant``:
        - rate: subset of ``{"mu", "burst_size", "k_off", "eta"}``.
        - logit: subset of ``{"rate", "kappa", "eta_anchor", "eta"}``.
    capture_mode : str
        ``"eta"`` (per-cell biology-informed) / ``"phi_only"``
        (odds-ratio capture) / ``"none"``.
    """
    if target_variant not in ("rate", "logit"):
        raise ValueError(
            f"Unknown target_variant={target_variant!r}; expected one of "
            "{'rate', 'logit'}."
        )
    # Validate the transform argument — accept either a string or a
    # dict mapping parameter name → transform name.
    if isinstance(target_positive_transform, str):
        if target_positive_transform not in _JAX_POSITIVE_FNS:
            raise ValueError(
                f"Unknown target_positive_transform="
                f"{target_positive_transform!r}; expected one of "
                f"{set(_JAX_POSITIVE_FNS)}."
            )
    elif isinstance(target_positive_transform, dict):
        for k, v in target_positive_transform.items():
            if v not in _JAX_POSITIVE_FNS:
                raise ValueError(
                    f"target_positive_transform[{k!r}]={v!r} is not a "
                    f"recognised transform; expected one of "
                    f"{set(_JAX_POSITIVE_FNS)}."
                )
    else:
        raise TypeError(
            "target_positive_transform must be a string or a dict "
            f"mapping parameter name → transform name; got "
            f"{type(target_positive_transform).__name__}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Building TSLN-{target_variant} priors from TwoState SVI source "
        f"(G={int(target_n_genes)}, N={int(target_n_cells)}, "
        f"tau={float(tau):.2f}, n_samples={int(n_samples)})"
    )

    # Gene identity (reuse NBLN-side helper — generic across SVI sources)
    strict_var_name_verified, identity_method, subset_info = (
        _check_gene_identity(
            results=results,
            target_n_genes=int(target_n_genes),
            target_gene_names=target_gene_names,
            target_gene_mask=target_gene_mask,
        )
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    _subset_active = (
        subset_info.is_subset and not subset_info.is_equal
    )
    # TSLN-logit subset cascade is intentionally unsupported: the
    # (rate, kappa, eta_anchor) reparameterization is even less
    # tractable under summation than the rate variant.  See
    # ``paper/_nb_lognormal.qmd`` §sec-nbln-cascade-aggregation, TSLN
    # caveat.  TODO(follow-up): design an aggregator for the
    # effective-deterministic representation if a use case appears.
    if _subset_active and target_variant == "logit":
        raise NotImplementedError(
            "Subset-aware cascade (SVI panel ⊃ Laplace panel) is not "
            "supported for the TSLN-Logit target variant in this "
            "release. The (rate, kappa, eta_anchor) reparameterization "
            "does not admit a clean per-sample aggregator. Workarounds: "
            "use TSLN-Rate as the target variant; or align the gene "
            "panels."
        )
    if _subset_active and _is_amortized_capture(results):
        raise NotImplementedError(
            "Subset-aware cascade is not supported with amortized-"
            "capture TSLN SVI sources."
        )
    if _subset_active:
        _dropped_n = int(
            0 if subset_info.dropped_idx_in_source is None else subset_info.dropped_idx_in_source.size
        )
        _say(
            f"Subset-aware cascade active (TSLN-rate): "
            f"aggregating {_dropped_n} SVI per-gene posteriors into the "
            f"target '_other' slot "
            f"({'+' if subset_info.source_has_other else 'no'} "
            "SVI '_other')."
        )

    # Sample SVI posterior (with amortized-capture safeguards).
    if _is_amortized_capture(results):
        _say(
            "Sampling SVI posterior (amortized capture detected)..."
        )
    else:
        _say("Sampling SVI posterior...")
    samples = _draw_samples(
        results=results,
        n_samples=int(n_samples),
        source_counts=source_counts,
        strict_var_name_verified=strict_var_name_verified,
        rng_seed=int(rng_seed),
    )
    _say(
        f"  Drew {int(n_samples)} samples; available keys: "
        f"{sorted(samples.keys())}"
    )

    capture_mode = _detect_capture_mode(samples)
    _say(f"Detected capture mode: {capture_mode!r}")

    # Per-parameter ``pos_inverse`` resolver — handles both the
    # string-form (``"softplus"``) and the dict-form
    # (``{"rate": "exp", "kappa": "softplus"}``) of the target's
    # ``positive_transform`` (Step 5 auditor fix).  Missing keys in the
    # dict fall back to ``"softplus"``.  Real-valued parameters
    # (``eta_anchor``) bypass this and use identity.
    def _pos_inv_for(param_name: str):
        if isinstance(target_positive_transform, dict):
            xform = target_positive_transform.get(param_name, "softplus")
        else:
            xform = target_positive_transform
        return _resolve_target_pos_inverse(xform)

    prior_bundle: Dict[str, Dict[str, jnp.ndarray]] = {}

    if target_variant == "logit":
        # --- TSLN-Logit coordinate map (plan §4.C.4 Rev 4) -----------
        # PRIMARY PATH: use the effective parameters emitted by the
        # TwoState SVI as ``numpyro.deterministic`` sites
        # (``alpha``, ``beta``, ``r_hat``, ``eta_act``).  These are
        # the POST-FLOOR effective values consistent with the
        # likelihood the upstream fit actually generated — if the
        # SVI's ``_twostate_reparam`` activated mean-preserving
        # floors (``_ALPHA_MIN``, ``_K_OFF_MIN``), the raw
        # ``(mu, burst_size, k_off)`` derivation would produce
        # cascade priors out of phase with the likelihood.
        #
        # FALLBACK PATH: raw (mu, burst_size, k_off) derivation —
        # only consistent when no floors activated upstream.  Emits
        # a UserWarning when the fallback fires.
        has_effective = all(
            k in samples for k in ("alpha", "beta", "r_hat")
        )
        if has_effective:
            alpha_s = jnp.maximum(jnp.asarray(samples["alpha"]), 1e-8)
            beta_s = jnp.maximum(jnp.asarray(samples["beta"]), 1e-8)
            rate_s = jnp.maximum(jnp.asarray(samples["r_hat"]), 1e-8)
            kappa_s = alpha_s + beta_s
            if "eta_act" in samples:
                eta_anchor_s = jnp.asarray(samples["eta_act"])
            else:
                eta_anchor_s = jnp.log(alpha_s) - jnp.log(beta_s)
            source_path = "effective deterministics (alpha, beta, r_hat, eta_act)"
        else:
            # Raw fallback: derive sample-wise from (mu, burst_size, k_off).
            for src in ("mu", "burst_size", "k_off"):
                if src not in samples:
                    raise ValueError(
                        f"SVI source missing required key {src!r} (and "
                        "no effective alpha/beta/r_hat deterministics "
                        f"either). Available keys: {sorted(samples.keys())}."
                    )
            mu_s = jnp.maximum(jnp.asarray(samples["mu"]), 1e-8)
            bs_s = jnp.maximum(jnp.asarray(samples["burst_size"]), 1e-8)
            ko_s = jnp.maximum(jnp.asarray(samples["k_off"]), 1e-8)
            # rate = r_hat = mu + burst_size · k_off  (TwoState reparam).
            rate_s = mu_s + bs_s * ko_s
            # kappa = alpha + beta = mu/burst_size + k_off.
            kappa_s = mu_s / bs_s + ko_s
            # eta_anchor = log(alpha/beta) = log(mu / (burst_size · k_off)).
            eta_anchor_s = jnp.log(mu_s) - jnp.log(bs_s) - jnp.log(ko_s)
            import warnings
            warnings.warn(
                "TSLN-Logit cascade fell back to raw "
                "(mu, burst_size, k_off) derivation because the SVI "
                "source did not expose effective alpha/beta/r_hat "
                "deterministics. If the SVI fit activated "
                "mean-preserving floors in _twostate_reparam, the "
                "cascade priors may be inconsistent with the upstream "
                "likelihood. Re-fit the SVI source with the latest "
                "scribe to get the effective deterministics.",
                UserWarning,
                stacklevel=3,
            )
            source_path = "raw (mu/burst_size/k_off) — fallback"

        # Validate shapes.
        for arr_name, arr in (
            ("rate", rate_s), ("kappa", kappa_s), ("eta_anchor", eta_anchor_s),
        ):
            if arr.ndim < 2 or arr.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"Derived {arr_name!r} samples have shape "
                    f"{arr.shape}; expected (S, {int(target_n_genes)})."
                )

        # Each positive global uses its OWN configured transform so
        # that ``positive_transform={"rate": "exp", "kappa":
        # "softplus"}`` is honored.  ``eta_anchor`` is real-valued
        # so it skips ``pos_inverse`` regardless.
        rate_pos_inv = _pos_inv_for("rate")
        kappa_pos_inv = _pos_inv_for("kappa")
        prior_bundle["rate"] = fit_empirical_gaussian(
            rate_pos_inv(rate_s), tau=float(tau)
        )
        prior_bundle["kappa"] = fit_empirical_gaussian(
            kappa_pos_inv(kappa_s), tau=float(tau)
        )
        prior_bundle["eta_anchor"] = fit_empirical_gaussian(
            eta_anchor_s, tau=float(tau)  # real-valued — identity transform
        )
        _say(
            f"  Built TSLN-Logit priors from {source_path}; "
            f"per-parameter positive_transform={target_positive_transform!r}; "
            "identity for eta_anchor."
        )

    else:
        # --- TSLN-Rate coordinate map --------------------------------
        # All three positive globals pass through their own configured
        # pos_inverse — so ``positive_transform={"mu": "exp",
        # "burst_size": "softplus", ...}`` is honored per-parameter.
        # Pre-check all three keys are present so the subset-aware
        # aggregator (which couples them) can run safely below.
        for src_key in ("mu", "burst_size", "k_off"):
            if src_key not in samples:
                raise ValueError(
                    f"SVI source missing required key {src_key!r}. "
                    f"Available keys: {sorted(samples.keys())}. The "
                    "TSLN-Rate cascade requires the natural TwoState "
                    "parameterization (mu, burst_size, k_off) on the "
                    "source."
                )

        if _subset_active:
            # Pre-aggregate all three positive parameters in one pass:
            # μ is additive, (burst_size, k_off) inherit SVI '_other'
            # or fall back to per-sample medians.  See
            # _aggregate_other_tsln_rate for the math and caveats.
            mu_src = jnp.asarray(samples["mu"])
            bs_src = jnp.asarray(samples["burst_size"])
            ko_src = jnp.asarray(samples["k_off"])
            mu_other_s, bs_other_s, ko_other_s = (
                _aggregate_other_tsln_rate(
                    mu_src,
                    bs_src,
                    ko_src,
                    subset_info.dropped_idx_in_source,
                    subset_info.source_other_index_in_source,
                )
            )
            kept_idx = subset_info.kept_idx_in_source
            mu_kept = _assemble_per_gene_subset_samples(mu_src, kept_idx)
            bs_kept = _assemble_per_gene_subset_samples(bs_src, kept_idx)
            ko_kept = _assemble_per_gene_subset_samples(ko_src, kept_idx)
            subset_assembled = {
                "mu": jnp.concatenate([mu_kept, mu_other_s[:, None]], axis=1),
                "burst_size": jnp.concatenate(
                    [bs_kept, bs_other_s[:, None]], axis=1
                ),
                "k_off": jnp.concatenate(
                    [ko_kept, ko_other_s[:, None]], axis=1
                ),
            }
        else:
            subset_assembled = None

        for src_key, tgt_key in (
            ("mu", "mu"),
            ("burst_size", "burst_size"),
            ("k_off", "k_off"),
        ):
            if subset_assembled is not None:
                s = subset_assembled[src_key]
            else:
                s = jnp.asarray(samples[src_key])
            if s.ndim < 2 or s.shape[1] != int(target_n_genes):
                raise ValueError(
                    f"SVI {src_key!r} samples have shape {s.shape}; "
                    f"expected (S, {int(target_n_genes)})."
                )
            s_pos = jnp.maximum(s, 1e-8)
            tgt_pos_inv = _pos_inv_for(tgt_key)
            s_uncon = tgt_pos_inv(s_pos)
            prior_bundle[tgt_key] = fit_empirical_gaussian(
                s_uncon, tau=float(tau)
            )
            _say(
                f"  Fitted {tgt_key!r} prior (G={int(s.shape[1])}"
                f"{', subset-aware aggregation' if _subset_active else ''})."
            )

    # --- eta (per-cell capture) -----------------------------------------
    if capture_mode == "eta":
        eta_samples = jnp.asarray(samples["eta_capture"])
        if eta_samples.ndim < 2 or eta_samples.shape[1] != int(target_n_cells):
            raise ValueError(
                f"SVI eta_capture samples have shape {eta_samples.shape}; "
                f"expected (S, {int(target_n_cells)})."
            )
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        _say(
            f"  Fitted eta prior (N={int(eta_samples.shape[1])}, "
            "transform='identity')."
        )
    elif capture_mode == "phi_only":
        # Convert the SVI source's p_capture (or phi_capture) into the
        # TSLN-Rate / NBLN-Laplace ``eta_capture`` coordinate system.
        # The mapping is mathematically clean:
        #
        #   NBVCP / twostatevcp:  log λ_cg = log r_g + log p_capture_c
        #   biology-anchored:     log λ_cg = log r_g − eta_capture_c
        #   ⇒  eta_capture_c = − log p_capture_c.
        #
        # A cell with ``p_capture = 0.1`` maps to ``eta_capture ≈ 2.30``
        # (positive — larger ``eta`` ↔ smaller capture).  Both quantities
        # live in the same per-cell semantic space; only the
        # coordinate / sign differs.
        #
        # ``phi_capture`` (mean-odds form) is first mapped back to
        # ``p_capture`` via ``p = phi / (1 + phi)``, then through the
        # same log transform.
        if "p_capture" in samples:
            p_cap_samples = jnp.maximum(
                jnp.asarray(samples["p_capture"]), 1e-8
            )
        else:
            phi_samples = jnp.maximum(
                jnp.asarray(samples["phi_capture"]), 1e-8
            )
            p_cap_samples = phi_samples / (1.0 + phi_samples)
            p_cap_samples = jnp.maximum(
                jnp.clip(p_cap_samples, 1e-8, 1.0 - 1e-8), 1e-8
            )

        if p_cap_samples.ndim < 2 or p_cap_samples.shape[1] != int(
            target_n_cells
        ):
            raise ValueError(
                f"SVI p_capture / phi_capture samples have shape "
                f"{p_cap_samples.shape}; expected "
                f"(S, {int(target_n_cells)})."
            )
        # Sample-wise log transform.  ``eta`` is real-valued (no
        # positive transform needed); the Laplace target stores it
        # directly in this coordinate.
        eta_samples = -jnp.log(p_cap_samples)
        prior_bundle["eta"] = fit_empirical_gaussian(
            eta_samples, tau=float(tau)
        )
        # Promote the detected mode to "eta" so downstream code
        # treats the cascade as having biology-anchored capture (the
        # bridge then sets up the ``x_only_offset`` Newton path when
        # ``eta`` is hard-frozen, or the joint Newton with the
        # per-cell soft cascade scale otherwise).
        capture_mode = "eta"
        _say(
            f"  Mapped p_capture/phi_capture → eta_capture per cell "
            f"(eta = −log p, N={int(eta_samples.shape[1])}).  "
            "Promoted capture_mode 'phi_only' → 'eta' so the cascade "
            "passes per-cell capture information through to the "
            "Laplace fit."
        )

    _say(
        f"Built TSLN-{target_variant} prior bundle: "
        f"keys={sorted(prior_bundle.keys())}, capture_mode={capture_mode!r}, "
        f"identity_method={identity_method!r}, n_samples={int(n_samples)}, "
        f"tau={float(tau):.2f}."
    )
    return prior_bundle, capture_mode


def freeze_values_from_twostate_results(
    results: Any,
    target_positive_transform,
    target_n_genes: int,
    target_n_cells: int,
    target_variant: str = "rate",
    target_gene_names: Optional[np.ndarray] = None,
    target_gene_mask: Optional[np.ndarray] = None,
    source_counts: Optional[jnp.ndarray] = None,
    freeze_params: Tuple[str, ...] = ("mu", "burst_size", "k_off"),
    map_method: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Extract point-estimate freeze values from a TwoState SVI fit.

    Hard-cascade analog of :func:`priors_from_twostate_results`. For
    each parameter in ``freeze_params``, returns the SVI MAP value
    transformed to the TSLN target's unconstrained coordinate.

    Parameters
    ----------
    results
        TwoState SVI results object with ``get_map()``.
    target_positive_transform : {"exp", "softplus"}
    target_n_genes, target_n_cells : int
    target_variant : {"rate", "logit"}, default ``"rate"``
        Both variants implemented.  For ``"logit"`` the valid keys
        are ``{"rate", "kappa", "eta_anchor", "eta"}``.
    freeze_params : Tuple[str, ...], default ``("mu", "burst_size", "k_off")``
        For TSLN-Rate, valid keys are
        ``{"mu", "burst_size", "k_off", "eta"}``.  For TSLN-Logit,
        valid keys are ``{"rate", "kappa", "eta_anchor", "eta"}``.
    map_method : str, optional
        Controls which ``map_method`` is passed to the SVI source's
        :meth:`~scribe.svi.results.ScribeSVIResults.get_map`. ``None``
        (default) lets the SVI source use its own default (currently
        ``"auto"``, i.e., Jacobian-corrected MAP). Pass ``"transform"``
        to pin the cascade to legacy ``transform(loc)`` semantics —
        useful when reproducing pre-correction cascade fits.

        IMPORTANT: when the SVI default flips to ``"auto"``, this
        cascade silently picks up the corrected (Jacobian-shifted)
        MAP values, which propagate into the downstream Laplace fit.
        For reproducibility of old scripts, pin to ``"transform"``.

    Returns
    -------
    Dict[str, Dict[str, jnp.ndarray]]
        Per-parameter ``{"loc": ...}`` dicts (no ``scale``).
    """
    if target_variant not in ("rate", "logit"):
        raise ValueError(
            f"Unknown target_variant={target_variant!r}; expected one of "
            "{'rate', 'logit'}."
        )
    # Accept str or per-parameter dict, same as priors_from_twostate_results.
    if isinstance(target_positive_transform, str):
        if target_positive_transform not in _JAX_POSITIVE_FNS:
            raise ValueError(
                f"Unknown target_positive_transform="
                f"{target_positive_transform!r}."
            )
    elif isinstance(target_positive_transform, dict):
        for k, v in target_positive_transform.items():
            if v not in _JAX_POSITIVE_FNS:
                raise ValueError(
                    f"target_positive_transform[{k!r}]={v!r} is not a "
                    f"recognised transform; expected one of "
                    f"{set(_JAX_POSITIVE_FNS)}."
                )
    else:
        raise TypeError(
            "target_positive_transform must be a string or a dict; got "
            f"{type(target_positive_transform).__name__}."
        )
    valid_by_variant = {
        "rate": {"mu", "burst_size", "k_off", "eta"},
        "logit": {"rate", "kappa", "eta_anchor", "eta"},
    }
    valid = valid_by_variant[target_variant]
    invalid = set(freeze_params) - valid
    if invalid:
        raise ValueError(
            f"freeze_params has invalid keys {invalid}; valid = {valid} "
            f"for target_variant={target_variant!r}."
        )

    def _say(msg: str) -> None:
        """Emit a user-facing progress line when ``verbose`` is on."""
        if verbose:
            logger.info(msg)

    _say(
        f"Extracting TSLN-{target_variant} freeze values "
        f"(freeze_params={list(freeze_params)})"
    )

    strict_var_name_verified, identity_method, subset_info = (
        _check_gene_identity(
            results=results,
            target_n_genes=int(target_n_genes),
            target_gene_names=target_gene_names,
            target_gene_mask=target_gene_mask,
        )
    )
    _say(f"  Gene identity verified via {identity_method!r}.")

    _subset_active = (
        subset_info.is_subset and not subset_info.is_equal
    )
    if _subset_active and target_variant == "logit":
        raise NotImplementedError(
            "Subset-aware freeze cascade is not supported for the "
            "TSLN-Logit target variant. See "
            "priors_from_twostate_results for the same caveat."
        )
    if _subset_active and _is_amortized_capture(results):
        raise NotImplementedError(
            "Subset-aware freeze cascade is not supported with "
            "amortized-capture TSLN SVI sources."
        )

    # Amortized-capture-aware get_map()
    if _is_amortized_capture(results):
        _say("  Source uses amortized capture; resolving encoder counts...")
        svi_source_counts = getattr(results, "_original_counts", None)
        if svi_source_counts is None:
            svi_source_counts = getattr(results, "counts", None)
        if svi_source_counts is not None:
            counts_for_encoder = svi_source_counts
        elif strict_var_name_verified and source_counts is not None:
            counts_for_encoder = source_counts
        else:
            raise ValueError(
                "Source uses amortized capture but counts can't be "
                "resolved. Same remediation options as the NBLN cascade."
            )
        # CASCADE PROPAGATION: pass map_method through to the SVI
        # source's get_map. ``None`` lets the source use its own
        # default; an explicit value pins the cascade to that method
        # (e.g., ``"transform"`` reproduces legacy uncorrected behavior).
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(
            counts=counts_for_encoder, verbose=False, **_map_kwargs
        )
    else:
        _map_kwargs = (
            {} if map_method is None else {"map_method": map_method}
        )
        map_dict = results.get_map(verbose=False, **_map_kwargs)

    _say(f"  SVI MAP keys: {sorted(map_dict.keys())}")

    # Resolve (burst_size, k_off) MAP from whichever TwoState
    # parameterization the SVI source used.  ``get_map()`` only
    # returns sampled sites (not deterministics), so for non-natural
    # parameterizations we derive (burst_size, k_off) by running the
    # appropriate reparam helper on the MAP of the sampled params.
    #
    # Skip this whole block for TSLN-Logit when the effective
    # deterministics are present — the logit branch below consumes
    # ``(alpha, beta, r_hat, eta_act)`` directly and does NOT need
    # ``(burst_size, k_off)``.
    # ``get_map()`` returns only the *sampled* sites, so deterministics
    # (``alpha`` / ``beta`` / ``r_hat`` / ``eta_act``) are NOT
    # available from the MAP unless we derive them locally.  Run the
    # appropriate reparam helper on the MAP of the sampled params and
    # populate the effective deterministics — this gives the logit
    # variant the same Rev 4 "effective-parameter primary path" the
    # samples-based cascade uses.
    _logit_effective_available = (
        target_variant == "logit"
        and all(k in map_dict for k in ("alpha", "beta", "r_hat"))
    )
    _rate_needs_derivation = (
        target_variant == "rate"
        and ("burst_size" not in map_dict or "k_off" not in map_dict)
    )
    _logit_needs_derivation = (
        target_variant == "logit"
        and not _logit_effective_available
    )
    if _rate_needs_derivation or _logit_needs_derivation:
        from ..models.components.likelihoods.two_state import (
            _twostate_reparam,
            _twostate_ratio_reparam,
            _twostate_moments_reparam,
            _twostate_moment_delta_reparam,
        )

        mu_map = jnp.asarray(map_dict["mu"])
        if "burst_size" in map_dict and "k_off" in map_dict:
            # Natural parameterization — burst_size and k_off already
            # present.  Run _twostate_reparam to get the effective
            # alpha / beta / r_hat (with the floors applied).
            _alpha, _beta, _rate, _eff_b = _twostate_reparam(
                mu_map,
                jnp.asarray(map_dict["burst_size"]),
                jnp.asarray(map_dict["k_off"]),
            )
            bs_derived = jnp.asarray(map_dict["burst_size"])
            ko_derived = _beta
        elif "switching_ratio" in map_dict:
            # ratio parameterization
            _alpha, _beta, _rate, _eff_b = _twostate_ratio_reparam(
                mu_map,
                jnp.asarray(map_dict["burst_size"]),
                jnp.asarray(map_dict["switching_ratio"]),
            )
            bs_derived = jnp.asarray(map_dict["burst_size"])
            ko_derived = _beta
        elif "inv_concentration" in map_dict:
            # moment_delta parameterization
            _alpha, _beta, _rate, _eff_b = _twostate_moment_delta_reparam(
                mu_map,
                jnp.asarray(map_dict["excess_fano"]),
                jnp.asarray(map_dict["inv_concentration"]),
            )
            bs_derived = _eff_b
            ko_derived = _beta
        elif "excess_fano" in map_dict and "concentration" in map_dict:
            # mean_fano parameterization
            _alpha, _beta, _rate, _eff_b = _twostate_moments_reparam(
                mu_map,
                jnp.asarray(map_dict["excess_fano"]),
                jnp.asarray(map_dict["concentration"]),
            )
            bs_derived = _eff_b
            ko_derived = _beta
        else:
            raise ValueError(
                "Cascade source's get_map() does not provide "
                "'burst_size'/'k_off' directly, and the parameterization "
                "is unrecognised (expected one of: natural / ratio / "
                "mean_fano / moment_delta).  Available keys: "
                f"{sorted(map_dict.keys())}."
            )
        # Augment the MAP dict with the derived natural-form values AND
        # the effective deterministics so the per-key loops below find
        # them.  ``eta_act = log(alpha) - log(beta)`` matches the SVI
        # deterministic site emitted by ``_emit_deterministics``.
        map_dict = dict(map_dict)
        map_dict.setdefault("burst_size", bs_derived)
        map_dict.setdefault("k_off", ko_derived)
        map_dict.setdefault("alpha", _alpha)
        map_dict.setdefault("beta", _beta)
        map_dict.setdefault("r_hat", _rate)
        map_dict.setdefault(
            "eta_act",
            jnp.log(jnp.maximum(_alpha, 1e-30))
            - jnp.log(jnp.maximum(_beta, 1e-30)),
        )
        _say(
            "  Derived (burst_size, k_off, alpha, beta, r_hat, eta_act) "
            "from the SVI MAP via _twostate_*_reparam."
        )

    # Per-parameter resolver — same shape as in priors_from_twostate_results.
    def _pos_inv_for(param_name: str):
        if isinstance(target_positive_transform, dict):
            xform = target_positive_transform.get(param_name, "softplus")
        else:
            xform = target_positive_transform
        return _resolve_target_pos_inverse(xform)

    freeze_values: Dict[str, Dict[str, jnp.ndarray]] = {}

    if target_variant == "logit":
        # ---- TSLN-Logit gene-level extraction ------------------------
        # Same primary / fallback split as the priors path: prefer the
        # SVI's effective deterministic MAPs (alpha, beta, r_hat,
        # eta_act) when available; fall back to raw derivation from
        # (mu, burst_size, k_off) with a warning.
        has_effective_map = all(
            k in map_dict for k in ("alpha", "beta", "r_hat")
        )
        if has_effective_map:
            alpha_m = jnp.maximum(jnp.asarray(map_dict["alpha"]), 1e-8)
            beta_m = jnp.maximum(jnp.asarray(map_dict["beta"]), 1e-8)
            rate_m = jnp.maximum(jnp.asarray(map_dict["r_hat"]), 1e-8)
            kappa_m = alpha_m + beta_m
            if "eta_act" in map_dict:
                eta_anchor_m = jnp.asarray(map_dict["eta_act"])
            else:
                eta_anchor_m = jnp.log(alpha_m) - jnp.log(beta_m)
            source_path = "effective deterministics (alpha, beta, r_hat, eta_act)"
        else:
            mu_m = jnp.maximum(jnp.asarray(map_dict["mu"]), 1e-8)
            bs_m = jnp.maximum(jnp.asarray(map_dict["burst_size"]), 1e-8)
            ko_m = jnp.maximum(jnp.asarray(map_dict["k_off"]), 1e-8)
            rate_m = mu_m + bs_m * ko_m
            kappa_m = mu_m / bs_m + ko_m
            eta_anchor_m = jnp.log(mu_m) - jnp.log(bs_m) - jnp.log(ko_m)
            import warnings
            warnings.warn(
                "TSLN-Logit freeze_values fell back to raw "
                "(mu, burst_size, k_off) derivation. See the "
                "priors_from_twostate_results docstring for details.",
                UserWarning,
                stacklevel=3,
            )
            source_path = "raw (mu/burst_size/k_off) — fallback"

        for tgt_key, val, is_positive in (
            ("rate", rate_m, True),
            ("kappa", kappa_m, True),
            ("eta_anchor", eta_anchor_m, False),
        ):
            if tgt_key not in freeze_params:
                continue
            if val.ndim != 1 or val.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"Derived {tgt_key!r} MAP has shape {val.shape}; "
                    f"expected ({int(target_n_genes)},)."
                )
            # Per-parameter positive transform so the dict-form
            # ``positive_transform={"rate": "exp", "kappa": "softplus"}``
            # is honored.
            loc = _pos_inv_for(tgt_key)(val) if is_positive else val
            freeze_values[tgt_key] = {"loc": loc}
            _say(
                f"  Extracted {tgt_key!r} freeze value from "
                f"{source_path} (G={target_n_genes})."
            )
    else:
        # ---- TSLN-Rate gene-level extraction -------------------------
        # Under subset cascade, pre-aggregate (mu, burst_size, k_off)
        # so the per-key loop below picks up target-shaped MAPs.  We
        # wrap the per-sample helper with a singleton sample axis to
        # reuse the same math at MAP-only granularity.
        if _subset_active:
            for src_key in ("mu", "burst_size", "k_off"):
                if src_key not in map_dict:
                    raise ValueError(
                        f"Subset-aware TSLN-rate freeze cascade requires "
                        f"the source MAP to expose {src_key!r}; available "
                        f"keys: {sorted(map_dict.keys())}."
                    )
            mu_src = jnp.asarray(map_dict["mu"])
            bs_src = jnp.asarray(map_dict["burst_size"])
            ko_src = jnp.asarray(map_dict["k_off"])
            for arr, name in (
                (mu_src, "mu"), (bs_src, "burst_size"), (ko_src, "k_off"),
            ):
                if arr.ndim != 1 or arr.shape[0] == 0:
                    raise ValueError(
                        f"SVI {name!r} MAP has shape {arr.shape}; "
                        "expected 1-D source-axis array for "
                        "subset-aware aggregation."
                    )
            mu_other_s, bs_other_s, ko_other_s = (
                _aggregate_other_tsln_rate(
                    mu_src[None, :],
                    bs_src[None, :],
                    ko_src[None, :],
                    subset_info.dropped_idx_in_source,
                    subset_info.source_other_index_in_source,
                )
            )
            kept_idx_jnp = jnp.asarray(subset_info.kept_idx_in_source)
            mu_kept = mu_src[kept_idx_jnp]
            bs_kept = bs_src[kept_idx_jnp]
            ko_kept = ko_src[kept_idx_jnp]
            subset_assembled_map = {
                "mu": jnp.concatenate([mu_kept, mu_other_s[0:1]], axis=0),
                "burst_size": jnp.concatenate(
                    [bs_kept, bs_other_s[0:1]], axis=0
                ),
                "k_off": jnp.concatenate(
                    [ko_kept, ko_other_s[0:1]], axis=0
                ),
            }
        else:
            subset_assembled_map = None

        for src_key, tgt_key in (
            ("mu", "mu"),
            ("burst_size", "burst_size"),
            ("k_off", "k_off"),
        ):
            if tgt_key not in freeze_params:
                continue
            if subset_assembled_map is not None:
                s = subset_assembled_map[src_key]
            else:
                if src_key not in map_dict:
                    raise ValueError(
                        f"freeze_params requests {tgt_key!r} but SVI MAP "
                        f"has no {src_key!r} key. Available: "
                        f"{sorted(map_dict.keys())}."
                    )
                s = jnp.asarray(map_dict[src_key])
            if s.ndim != 1 or s.shape[0] != int(target_n_genes):
                raise ValueError(
                    f"SVI {src_key!r} MAP has shape {s.shape}; expected "
                    f"({int(target_n_genes)},)."
                )
            # Per-parameter positive transform (dict-form support).
            s_uncon = _pos_inv_for(tgt_key)(jnp.maximum(s, 1e-8))
            freeze_values[tgt_key] = {"loc": s_uncon}
            _say(
                f"  Extracted {tgt_key!r} freeze value (G={target_n_genes}"
                f"{', subset-aware aggregation' if _subset_active else ''})."
            )

    if "eta" in freeze_params:
        # Same p_capture / phi_capture → eta mapping as
        # ``priors_from_twostate_results``: ``eta = −log p_capture``.
        # If the SVI source emits ``eta_capture`` directly (biology-
        # anchored), use it unchanged.
        if "eta_capture" in map_dict:
            eta = jnp.asarray(map_dict["eta_capture"])
        elif "p_capture" in map_dict:
            p_cap = jnp.maximum(jnp.asarray(map_dict["p_capture"]), 1e-8)
            eta = -jnp.log(p_cap)
            _say(
                "  Mapped p_capture MAP → eta_capture freeze value "
                "via eta = −log p."
            )
        elif "phi_capture" in map_dict:
            phi = jnp.maximum(jnp.asarray(map_dict["phi_capture"]), 1e-8)
            p_cap = phi / (1.0 + phi)
            eta = -jnp.log(jnp.maximum(p_cap, 1e-8))
            _say(
                "  Mapped phi_capture MAP → eta_capture freeze value "
                "via p = phi/(1+phi), eta = −log p."
            )
        else:
            raise ValueError(
                "freeze_params requests 'eta' but SVI MAP has no "
                "'eta_capture', 'p_capture', or 'phi_capture' key. The "
                "source does not appear to model per-cell capture. "
                f"Available: {sorted(map_dict.keys())}."
            )
        if eta.ndim != 1 or eta.shape[0] != int(target_n_cells):
            raise ValueError(
                f"SVI capture MAP has shape {eta.shape} after mapping; "
                f"expected ({int(target_n_cells)},)."
            )
        freeze_values["eta"] = {"loc": eta}
        _say(
            f"  Extracted eta freeze value (N={target_n_cells})."
        )

    _say(
        f"Built TSLN-{target_variant} freeze-values bundle: "
        f"keys={sorted(freeze_values.keys())}, requested={list(freeze_params)!r}, "
        f"identity_method={identity_method!r}."
    )
    return freeze_values

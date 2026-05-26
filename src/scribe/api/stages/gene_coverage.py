"""
Stage 2b: Gene coverage filtering and ALR reference resolution.

Applies optional pre-fit gene coverage filtering, aggregates excluded
genes into a trailing "other" pseudo-gene column, and resolves the ALR
reference index for LNM models.

FitContext reads : count_data, n_genes, adata, dataset_indices, model,
                   _original_n_genes, kwargs[gene_coverage, alr_reference_idx]
FitContext writes: count_data, n_genes, alr_reference_idx,
                   _gene_coverage_mask, _gene_coverage_rank,
                   _excluded_gene_names, _filtered_gene_names,
                   _adata_for_inference
"""

import logging

from ..context import FitContext

_log = logging.getLogger(__name__)


def apply_gene_coverage_and_alr(ctx: FitContext) -> None:
    """
    Filter low-coverage genes and resolve the ALR reference index.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.

    Raises
    ------
    ValueError
        If ``alr_reference_idx`` is out of bounds or points to a gene
        excluded by coverage filtering.
    """
    kw = ctx.kwargs
    gene_coverage = kw.get("gene_coverage")
    alr_reference_idx = kw.get("alr_reference_idx")

    # -- Gene coverage filtering ----------------------------------------------
    if gene_coverage is not None:
        from ...core.gene_coverage import (
            aggregate_counts_by_mask,
            build_filtered_gene_names,
            compute_empirical_gene_coverage_mask,
            compute_gene_coverage_rank,
        )
        import numpy as np

        count_data = ctx.count_data
        dataset_indices = ctx.dataset_indices
        adata = ctx.adata
        _original_n_genes = ctx._original_n_genes

        # Compute keep mask (pooled or union-of-per-dataset).
        _gene_coverage_mask = compute_empirical_gene_coverage_mask(
            count_data,
            coverage=gene_coverage,
            dataset_indices=dataset_indices,
        )
        _gene_coverage_rank = compute_gene_coverage_rank(count_data)
        _count_data_precoverage = np.asarray(count_data)

        # Annotate AnnData gene metadata in the original gene space.
        _filtered_gene_names = None
        _excluded_gene_names = None
        if adata is not None:
            adata.var["scribe_gene_coverage_included"] = _gene_coverage_mask
            adata.var["scribe_gene_coverage_rank"] = _gene_coverage_rank
            full_gene_names = [
                str(name) for name in adata.var_names.tolist()
            ]
            _filtered_gene_names, _excluded_gene_names = (
                build_filtered_gene_names(
                    gene_names=full_gene_names,
                    mask=_gene_coverage_mask,
                )
            )
            # If genes were excluded, we cannot pass adata to factories
            # because n_genes would mismatch.
            if int(np.asarray(~_gene_coverage_mask, dtype=int).sum()) > 0:
                ctx._adata_for_inference = None

        # Aggregate excluded genes into a trailing "other" column.
        count_data = aggregate_counts_by_mask(
            count_data, mask=_gene_coverage_mask
        )
        n_genes = int(count_data.shape[1])
        n_kept = int(np.asarray(_gene_coverage_mask, dtype=bool).sum())

        # Log summary and per-dataset breakdown.
        if dataset_indices is not None:
            _ds = np.asarray(dataset_indices).ravel()
            _per_dataset = []
            for _dataset_id in np.unique(_ds):
                _ds_mask = _ds == _dataset_id
                _orig_ds_counts = _count_data_precoverage[_ds_mask, :]
                _ds_keep_mask = compute_empirical_gene_coverage_mask(
                    _orig_ds_counts,
                    coverage=gene_coverage,
                    dataset_indices=None,
                )
                _per_dataset.append(
                    f"{_dataset_id}:{int(_ds_keep_mask.sum())}"
                    f"/{_original_n_genes}"
                )
            _log.info(
                "Applied gene_coverage pre-filtering with union across "
                f"datasets. Kept {n_kept}/{_original_n_genes} genes; "
                f"per-dataset keep counts: {', '.join(_per_dataset)}."
            )
        else:
            _log.info(
                "Applied gene_coverage pre-filtering. "
                f"Kept {n_kept}/{_original_n_genes} genes and pooled "
                f"{_original_n_genes - n_kept} genes into 'other'."
            )

        ctx.count_data = count_data
        ctx.n_genes = n_genes
        ctx._gene_coverage_mask = _gene_coverage_mask
        ctx._gene_coverage_rank = _gene_coverage_rank
        ctx._excluded_gene_names = _excluded_gene_names
        ctx._filtered_gene_names = _filtered_gene_names
        # Persist the pooled-`_other` signal on the ctx so the
        # downstream Laplace engine (any base model) can construct the
        # correct AxisLayout for `correlate_other_column` decoupling.
        # Array-input fits (no AnnData → no `_filtered_gene_names`)
        # rely on this primary signal since they cannot detect `_other`
        # from gene names.  See `scribe.laplace._axis_layout`.
        ctx._has_pooled_other = bool(
            int(np.asarray(~_gene_coverage_mask, dtype=int).sum()) > 0
        )
    else:
        # No gene_coverage filtering was applied.  Use ``None`` (not
        # ``False``) so downstream signals (e.g. AnnData var_names
        # fallback in ``run_inference.py``) can take primary
        # responsibility for detecting a manually-named ``_other``
        # tail.  ``False`` here would be an *explicit* "no _other"
        # claim, which conflicts with the var_names fallback when
        # the user manually pre-filtered their AnnData and named
        # the trailing aggregate ``_other`` (auditor finding rev-5
        # #2 cascade: ``has_pooled_other=False`` + names-say-other
        # would otherwise raise the contradictory-signal ValueError
        # under ``correlate_other_column=False``).
        if not hasattr(ctx, "_has_pooled_other"):
            ctx._has_pooled_other = None

    # -- ALR reference resolution (LNM models only) ---------------------------
    if ctx.model.lower() in ("lnm", "lnmvcp"):
        from ...models.components.likelihoods.lnm import select_alr_reference
        import logging
        import numpy as np

        n_genes = ctx.n_genes
        _gene_coverage_mask = ctx._gene_coverage_mask
        _original_n_genes = ctx._original_n_genes

        # Mask-derived signal (load-bearing for the original→filtered
        # index mapping below — only meaningful when the gene_coverage
        # stage actually ran).
        _has_pooled_other_from_mask = bool(
            _gene_coverage_mask is not None
            and np.any(~np.asarray(_gene_coverage_mask, dtype=bool))
        )

        # Names-derived fallback (auditor rev-9 Medium): detects a
        # manually-pre-filtered AnnData where the user named the
        # trailing column ``_other`` WITHOUT running the gene_coverage
        # stage.  Matches the detection priority used by
        # ``scribe.laplace._axis_layout.build_axis_layout`` so the LNM
        # auto-pin behaviour is consistent with the four count-
        # likelihood models' shared layout path.  Only consulted when
        # ``gene_coverage`` is None — when the stage ran, the mask is
        # the authoritative signal (even if it says no pooling
        # happened: a coincidental ``var_names[-1] == "_other"`` would
        # be a real retained gene, not a pooled aggregate, and must
        # not trigger the fallback).  Array-input fits (no AnnData)
        # have no names and cannot trigger this path by design.
        _has_pooled_other_from_names = False
        if gene_coverage is None and ctx.adata is not None:
            try:
                _var_names = ctx.adata.var_names.tolist()
                _has_pooled_other_from_names = (
                    len(_var_names) == n_genes
                    and str(_var_names[-1]) == "_other"
                )
            except Exception:
                _has_pooled_other_from_names = False

        _has_pooled_other = (
            _has_pooled_other_from_mask or _has_pooled_other_from_names
        )

        # Harmonic-hare Commit 6: when ``correlate_other_column=False``
        # AND a pooled ``_other`` exists, LNM realises the decoupling
        # by pinning the ALR reference to ``_other``'s position (the
        # ALR reference gene is excluded from the latent covariance by
        # construction).  Under legacy (``True``), preserve today's
        # contract verbatim — including raising if the reference
        # resolves to ``_other`` — so the bit-equal contract holds.
        # See ``paper/_nb_lognormal.qmd`` §sec-nbln-decorrelate-other
        # and ``paper/_logistic_normal_multinomial.qmd`` cross-ref for
        # the biophysical rationale.
        _correlate_other_column = bool(
            kw.get("correlate_other_column", False)
        )
        _other_pos_in_filtered = n_genes - 1  # last position post-filter

        if alr_reference_idx is None:
            if (not _correlate_other_column) and _has_pooled_other:
                # Decoupled + pooled ``_other`` exists: pin the ALR
                # reference to ``_other``'s position so the latent
                # covariance excludes ``_other`` by construction.
                # This is the only way LNM realises the same decoupling
                # the other four models implement via the deviation
                # reparameterisation.  No min-variance fallback runs
                # here.
                alr_reference_idx = int(_other_pos_in_filtered)
                logging.getLogger(__name__).info(
                    "LNM (correlate_other_column=False): pinned ALR "
                    "reference to the pooled '_other' pseudo-gene at "
                    "filtered position %d.  '_other' is excluded from "
                    "the latent low-rank covariance by ALR construction; "
                    "see paper/_nb_lognormal.qmd §sec-nbln-decorrelate-"
                    "other for the biophysical rationale.",
                    alr_reference_idx,
                )
            else:
                # Legacy or no-_other path: auto-selection by minimum
                # variance of log-proportion (today's behaviour).
                alr_reference_idx = int(
                    select_alr_reference(ctx.count_data)
                )
                _is_other = (
                    _has_pooled_other
                    and alr_reference_idx == ctx.count_data.shape[1] - 1
                )
                logging.getLogger(__name__).info(
                    "LNM: auto-selected gene %d as ALR reference "
                    "(%s; minimum-variance criterion).",
                    alr_reference_idx,
                    (
                        "pooled '_other' pseudo-gene won the variance "
                        "competition"
                        if _is_other
                        else "individual gene"
                    ),
                )
        else:
            _ref_input = int(alr_reference_idx)
            if gene_coverage is None:
                if not (0 <= _ref_input < n_genes):
                    raise ValueError(
                        f"alr_reference_idx must be in "
                        f"[0, {n_genes - 1}], got {_ref_input}."
                    )
                alr_reference_idx = _ref_input
            else:
                if not (0 <= _ref_input < _original_n_genes):
                    raise ValueError(
                        "With gene_coverage enabled, alr_reference_idx "
                        "is interpreted in the original gene space and "
                        f"must be in [0, {_original_n_genes - 1}], "
                        f"got {_ref_input}."
                    )
                if _gene_coverage_mask is not None and not bool(
                    _gene_coverage_mask[_ref_input]
                ):
                    raise ValueError(
                        "alr_reference_idx points to a gene excluded "
                        "by gene_coverage filtering (pooled into "
                        "'other'). Choose a retained gene index."
                    )
                # Map original-gene index to filtered-gene index.
                if _gene_coverage_mask is not None and _has_pooled_other:
                    alr_reference_idx = int(
                        np.asarray(
                            _gene_coverage_mask[:_ref_input]
                        ).sum()
                    )
                else:
                    alr_reference_idx = _ref_input

            # Harmonic-hare Commit 6: bifurcate the post-resolution check
            # on ``correlate_other_column``.  Under decoupling, an
            # explicit non-``_other`` reference contradicts the user's
            # intent to exclude ``_other`` from Σ — raise with a clear
            # message pointing to both flags.  Under legacy, retain
            # today's contract (reject any resolution to ``_other``).
            if _has_pooled_other:
                if not _correlate_other_column:
                    if alr_reference_idx != _other_pos_in_filtered:
                        raise ValueError(
                            "Inconsistent configuration: "
                            "correlate_other_column=False requests that "
                            "the pooled '_other' aggregate be excluded "
                            "from the LNM latent covariance, but the "
                            "explicit alr_reference_idx resolves to a "
                            f"retained gene (filtered position "
                            f"{alr_reference_idx}, not the '_other' "
                            f"position {_other_pos_in_filtered}).  "
                            "Under correlate_other_column=False, the "
                            "ALR reference must be pinned to '_other' "
                            "to realise the decoupling.  Either drop "
                            "the explicit alr_reference_idx (it will "
                            "auto-pin to '_other') or pass "
                            "correlate_other_column=True to keep "
                            "today's behaviour with a real-gene "
                            "reference."
                        )
                else:
                    # Legacy: explicit references that resolve to a
                    # real gene are valid; explicit references that
                    # resolve to ``_other`` are still rejected
                    # because today's LNM code assumes the reference
                    # is a real gene.
                    if alr_reference_idx == _other_pos_in_filtered:
                        raise ValueError(
                            "alr_reference_idx resolved to the pooled "
                            "'other' gene. Please select a retained "
                            "original gene, or pass "
                            "correlate_other_column=False to pin the "
                            "ALR reference to '_other' automatically."
                        )

        ctx.alr_reference_idx = alr_reference_idx

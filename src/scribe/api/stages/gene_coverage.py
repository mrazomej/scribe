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

    # -- ALR reference resolution (LNM models only) ---------------------------
    if ctx.model.lower() in ("lnm", "lnmvcp"):
        from ...models.components.likelihoods.lnm import select_alr_reference
        import logging
        import numpy as np

        n_genes = ctx.n_genes
        _gene_coverage_mask = ctx._gene_coverage_mask
        _original_n_genes = ctx._original_n_genes

        _has_pooled_other = bool(
            _gene_coverage_mask is not None
            and np.any(~np.asarray(_gene_coverage_mask, dtype=bool))
        )

        if alr_reference_idx is None:
            # Auto-selection by minimum variance of log-proportion.
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

            if _has_pooled_other and alr_reference_idx == (n_genes - 1):
                raise ValueError(
                    "alr_reference_idx resolved to the pooled 'other' "
                    "gene. Please select a retained original gene."
                )

        ctx.alr_reference_idx = alr_reference_idx

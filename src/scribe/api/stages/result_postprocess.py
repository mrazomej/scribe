"""
Stage 6: Attach post-inference metadata to the results object.

Persists gene-coverage, annotation, and multinomial ceiling metadata
on the frozen results object via ``object.__setattr__``.

FitContext reads : results, _total_count_max, _gene_coverage_mask,
                   _gene_coverage_rank, _excluded_gene_names,
                   _filtered_gene_names, _original_n_genes,
                   _adata_for_inference, adata, n_genes,
                   _label_map, _component_mapping,
                   kwargs[gene_coverage]
FitContext writes: results (metadata attrs attached in place)
"""

from ..context import FitContext


def postprocess_results(ctx: FitContext) -> None:
    """
    Attach pipeline metadata to the inference results object.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.results`` is mutated in place
        via ``object.__setattr__`` because results are frozen Pydantic
        models.
    """
    results = ctx.results
    kw = ctx.kwargs

    # Multinomial total-count ceiling for predictive sampling.
    object.__setattr__(
        results, "_total_count_max", int(ctx._total_count_max)
    )

    # -- Gene-coverage metadata -----------------------------------------------
    if ctx._gene_coverage_mask is not None:
        import pandas as pd
        import numpy as np

        object.__setattr__(
            results, "_gene_coverage", float(kw["gene_coverage"])
        )
        object.__setattr__(
            results,
            "_gene_coverage_mask",
            np.asarray(ctx._gene_coverage_mask, dtype=bool),
        )
        object.__setattr__(
            results, "_original_n_genes", int(ctx._original_n_genes)
        )
        object.__setattr__(
            results,
            "_excluded_gene_names",
            (
                list(ctx._excluded_gene_names)
                if ctx._excluded_gene_names is not None
                else None
            ),
        )

        if ctx.adata is not None and ctx._adata_for_inference is None:
            _names = ctx._filtered_gene_names
            if _names is None:
                _names = [f"gene_{i}" for i in range(ctx.n_genes)]
            _var = pd.DataFrame(index=pd.Index(_names))
            object.__setattr__(results, "var", _var)
            object.__setattr__(results, "n_vars", int(_var.shape[0]))
            object.__setattr__(results, "obs", ctx.adata.obs.copy())
            object.__setattr__(results, "uns", ctx.adata.uns.copy())
            object.__setattr__(results, "n_obs", int(ctx.adata.n_obs))

    # -- Annotation metadata --------------------------------------------------
    if ctx._label_map is not None and hasattr(results, "_label_map"):
        object.__setattr__(results, "_label_map", ctx._label_map)
    if ctx._component_mapping is not None and hasattr(
        results, "_component_mapping"
    ):
        object.__setattr__(
            results, "_component_mapping", ctx._component_mapping
        )

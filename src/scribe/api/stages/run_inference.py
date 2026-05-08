"""
Stage 5: Dispatch inference to the appropriate backend.

Delegates to :func:`scribe.inference.dispatcher._run_inference` after
assembling the call arguments from FitContext.

The dispatcher is resolved at call time through the ``scribe.api``
module object rather than bound at import time.  This preserves
monkeypatch compatibility: tests that ``patch("scribe.api._run_inference")``
replace the attribute on the ``scribe.api`` module, and this stage
honours that replacement by looking it up dynamically.

FitContext reads : model_config, count_data, inference_config,
                   _adata_for_inference, n_cells, n_genes, data_config,
                   annotation_prior_logits, dataset_indices,
                   effective_x64, kwargs[seed]
FitContext writes: results
"""

import sys

from ..context import FitContext


def dispatch_inference(ctx: FitContext) -> None:
    """
    Call the inference dispatcher and store the results.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.results`` is set here with the
        returned inference results object.
    """
    # Look up _run_inference at call time so unittest.mock.patch
    # on "scribe.api._run_inference" is honoured.
    _api = sys.modules["scribe.api"]
    _run_inference = _api._run_inference

    results = _run_inference(
        ctx.inference_config.method,
        model_config=ctx.model_config,
        count_data=ctx.count_data,
        inference_config=ctx.inference_config,
        adata=ctx._adata_for_inference,
        n_cells=ctx.n_cells,
        n_genes=ctx.n_genes,
        data_config=ctx.data_config,
        seed=ctx.kwargs.get("seed", 42),
        annotation_prior_logits=ctx.annotation_prior_logits,
        dataset_indices=ctx.dataset_indices,
        enable_x64=ctx.effective_x64,
    )
    ctx.results = results

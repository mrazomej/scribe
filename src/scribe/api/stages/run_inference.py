"""
Stage 5: Dispatch inference to the appropriate backend.

Delegates to :func:`scribe.inference.dispatcher._run_inference` after
assembling the call arguments from FitContext.

The dispatcher is resolved at call time through the ``scribe.api``
module object rather than bound at import time.  This preserves
monkeypatch compatibility: tests that ``patch("scribe.api._run_inference")``
replace the attribute on the ``scribe.api`` module, and this stage
honours that replacement by looking it up dynamically.

This stage also handles the **SVI-informative-prior cascade** for
NBLN-Laplace fits: when the user passes ``informative_priors_from``,
the stage builds an empirical-Gaussian prior bundle from the SVI
results object here (where ``ctx`` has all gene metadata in scope)
and forwards it as two opaque arguments through the dispatcher:
``informative_priors`` (the bundle) and ``capture_mode_override``
(``"eta"`` / ``"phi_only"`` / ``"none"`` from capture-mode detection).

FitContext reads : model_config, count_data, inference_config,
                   _adata_for_inference, n_cells, n_genes, data_config,
                   annotation_prior_logits, dataset_indices,
                   effective_x64, kwargs[seed],
                   kwargs[informative_priors_from],
                   kwargs[informative_priors_tau],
                   kwargs[informative_priors_n_samples],
                   _filtered_gene_names, _gene_coverage_mask
FitContext writes: results
"""

import sys

from ..context import FitContext


def dispatch_inference(ctx: FitContext) -> None:
    """
    Build any SVI-derived prior bundle and call the inference dispatcher.

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

    informative_priors_from = ctx.kwargs.get("informative_priors_from")
    informative_priors = None
    capture_mode_override = None

    if informative_priors_from is not None:
        # --- Scope validation (Round-4 Finding 1) ----------------------
        # Run AFTER ctx.model_config and ctx.inference_config are built,
        # so we check the resolved base_model / method even when the
        # user passes prebuilt config objects rather than raw kwargs.
        base_model = ctx.model_config.base_model
        method = ctx.inference_config.method.value
        if base_model != "nbln" or method != "laplace":
            raise ValueError(
                "informative_priors_from is only supported for "
                "model='nbln' and inference_method='laplace'; got "
                f"base_model={base_model!r}, inference_method={method!r}."
            )

        # --- Target gene-identity inputs (Round-4 Finding 3) -----------
        # Defensive fallback chain: prefer var_names from the inference-
        # time AnnData; fall back to _filtered_gene_names when
        # _adata_for_inference is None.
        target_gene_names = None
        if ctx._adata_for_inference is not None:
            var_names = getattr(
                ctx._adata_for_inference, "var_names", None
            )
            if var_names is not None:
                target_gene_names = var_names.values
        if target_gene_names is None:
            filtered = getattr(ctx, "_filtered_gene_names", None)
            if filtered is not None:
                target_gene_names = filtered

        # --- Build prior bundle via Layer 2 ----------------------------
        from ...laplace.priors import priors_from_results

        informative_priors, capture_mode_override = priors_from_results(
            informative_priors_from,
            target_positive_transform=ctx.model_config.positive_transform,
            target_n_genes=ctx.n_genes,
            target_n_cells=ctx.n_cells,
            target_gene_names=target_gene_names,
            target_gene_mask=getattr(ctx, "_gene_coverage_mask", None),
            # Pass target counts so amortized-capture SVI sources can
            # draw samples; the adapter decides whether it's safe to
            # use them based on the identity-verification state.
            source_counts=ctx.count_data,
            n_samples=int(ctx.kwargs.get("informative_priors_n_samples", 1000)),
            tau=float(ctx.kwargs.get("informative_priors_tau", 1.0)),
            verbose=bool(ctx.kwargs.get("informative_priors_verbose", True)),
        )

    # --- Phase 2 freeze: extract point estimates + resolve cascade source ---
    # The freeze API is activated only when a cascade is supplied.  When
    # `informative_priors_from is None`, normalize freeze to empty so a
    # plain Laplace fit (no cascade) is unaffected by the default-on
    # `informative_priors_freeze=("r", "eta")` kwarg (Round-4 R5-2 fix).
    freeze_values = None
    freeze_params: tuple = ()
    cascade_source = None
    cascade_source_counts = None
    if informative_priors_from is not None:
        # Read and normalize the freeze tuple.  Accept any iterable
        # of {"r", "mu", "eta"}; coerce to a tuple of strings.
        raw_freeze = ctx.kwargs.get("informative_priors_freeze", ("r", "eta"))
        if raw_freeze is None:
            freeze_params = ()
        else:
            freeze_params = tuple(raw_freeze)
        # Build the freeze-values bundle from SVI's get_map().
        if freeze_params:
            from ...laplace.priors import freeze_values_from_results
            freeze_values = freeze_values_from_results(
                informative_priors_from,
                target_positive_transform=ctx.model_config.positive_transform,
                target_n_genes=ctx.n_genes,
                target_n_cells=ctx.n_cells,
                target_gene_names=target_gene_names,
                target_gene_mask=getattr(ctx, "_gene_coverage_mask", None),
                source_counts=ctx.count_data,
                freeze_params=freeze_params,
                verbose=bool(
                    ctx.kwargs.get("informative_priors_verbose", True)
                ),
            )
        # Embed the SVI results object so PPC and get_distributions can
        # consult its guide directly for frozen parameters.  Per Round-5
        # R5-6: store counts on the Laplace result (cascade_source_counts)
        # rather than mutating cascade_source._original_counts.
        cascade_source = informative_priors_from
        if hasattr(cascade_source, "_uses_amortized_capture") and \
                cascade_source._uses_amortized_capture():
            existing_counts = getattr(cascade_source, "_original_counts", None)
            if existing_counts is None:
                # Source is amortized but didn't cache its training counts;
                # cache the target counts (var-name identity verified by
                # priors_from_results above) so pickle-then-PPC works.
                cascade_source_counts = ctx.count_data

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
        informative_priors=informative_priors,
        capture_mode_override=capture_mode_override,
        freeze_values=freeze_values,
        freeze_params=freeze_params,
        cascade_source=cascade_source,
        cascade_source_counts=cascade_source_counts,
    )
    ctx.results = results

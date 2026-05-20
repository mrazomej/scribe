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
        # --- Scope validation (Round-4 Finding 1; round-7 extended) -----
        # Run AFTER ctx.model_config and ctx.inference_config are built,
        # so we check the resolved base_model / method even when the
        # user passes prebuilt config objects rather than raw kwargs.
        # Single source of truth:
        # ``api.constants.CASCADE_FROM_SVI_SUPPORTED_BASE_MODELS``.
        from ..constants import CASCADE_FROM_SVI_SUPPORTED_BASE_MODELS

        base_model = ctx.model_config.base_model
        method = ctx.inference_config.method.value
        if (
            base_model not in CASCADE_FROM_SVI_SUPPORTED_BASE_MODELS
            or method != "laplace"
        ):
            raise ValueError(
                "informative_priors_from is only supported for "
                f"model in {sorted(CASCADE_FROM_SVI_SUPPORTED_BASE_MODELS)} "
                f"with inference_method='laplace'; got "
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

        # --- positive_transform string for the cascade adapter ---------
        # Cascade adapters accept either a single transform name OR a
        # per-parameter dict mapping internal parameter name → transform.
        # Honour the dict-form ``positive_transform`` on the target
        # ``model_config`` so the cascade adapter applies the right
        # ``pos_inverse`` to each positive parameter — getting this
        # wrong would silently corrupt the cascade priors / freeze
        # values for users who set per-parameter transforms (the
        # auditor's Step 3-5 catch).
        #
        # For NBLN's older single-string API we resolve the canonical
        # ``r`` transform; for the TwoState variants we build a dict
        # covering all positive globals that the adapter touches:
        #   - TSLN-Rate  → keys ``mu`` / ``burst_size`` / ``k_off``
        #   - TSLN-Logit → keys ``rate`` / ``kappa``
        #     (``eta_anchor`` is real-valued — no transform).
        if hasattr(ctx.model_config, "resolve_positive_transform"):
            if base_model == "twostate_ln_rate":
                _pos_xform_str = {
                    "mu": ctx.model_config.resolve_positive_transform("mu"),
                    "burst_size": ctx.model_config.resolve_positive_transform(
                        "burst_size"
                    ),
                    "k_off": ctx.model_config.resolve_positive_transform(
                        "k_off"
                    ),
                }
            elif base_model == "twostate_ln_logit":
                _pos_xform_str = {
                    "rate": ctx.model_config.resolve_positive_transform(
                        "rate"
                    ),
                    "kappa": ctx.model_config.resolve_positive_transform(
                        "kappa"
                    ),
                }
            else:
                _pos_xform_str = ctx.model_config.resolve_positive_transform(
                    "r"
                )
        else:
            _pos_xform_str = ctx.model_config.positive_transform
            if not isinstance(_pos_xform_str, (str, dict)):
                _pos_xform_str = "softplus"

        # --- Build prior bundle (dispatch on base_model) ---------------
        if base_model in ("twostate_ln_rate", "twostate_ln_logit"):
            from ...laplace.priors import priors_from_twostate_results

            _target_variant = (
                "rate" if base_model == "twostate_ln_rate" else "logit"
            )
            informative_priors, capture_mode_override = (
                priors_from_twostate_results(
                    informative_priors_from,
                    target_positive_transform=_pos_xform_str,
                    target_n_genes=ctx.n_genes,
                    target_n_cells=ctx.n_cells,
                    target_variant=_target_variant,
                    target_gene_names=target_gene_names,
                    target_gene_mask=getattr(
                        ctx, "_gene_coverage_mask", None
                    ),
                    source_counts=ctx.count_data,
                    n_samples=int(
                        ctx.kwargs.get(
                            "informative_priors_n_samples", 1000
                        )
                    ),
                    tau=float(
                        ctx.kwargs.get("informative_priors_tau", 1.0)
                    ),
                    verbose=bool(
                        ctx.kwargs.get(
                            "informative_priors_verbose", True
                        )
                    ),
                )
            )
            # PR-2 capture restriction (Rev 4): TSLN-Logit cannot
            # accept a soft-cascade eta because the joint Newton path
            # is deferred to phase 3.  When the SVI source supplies
            # per-cell eta (capture_mode_override == "eta"), the
            # downstream obs-model constructor would reject it via
            # ``informative_priors['eta']``.  Strip the eta entry from
            # the prior bundle here and force the cascade to be a
            # *frozen-offset* one (the eta freeze value is built below
            # from the SVI MAP).  ``capture_mode_override`` stays
            # "eta" so the bridge knows capture is on; the obs model
            # sees no soft eta prior.
            if (
                base_model == "twostate_ln_logit"
                and informative_priors is not None
                and "eta" in informative_priors
            ):
                del informative_priors["eta"]
        else:
            from ...laplace.priors import priors_from_results

            informative_priors, capture_mode_override = priors_from_results(
                informative_priors_from,
                target_positive_transform=_pos_xform_str,
                target_n_genes=ctx.n_genes,
                target_n_cells=ctx.n_cells,
                target_gene_names=target_gene_names,
                target_gene_mask=getattr(ctx, "_gene_coverage_mask", None),
                # Pass target counts so amortized-capture SVI sources can
                # draw samples; the adapter decides whether it's safe to
                # use them based on the identity-verification state.
                source_counts=ctx.count_data,
                n_samples=int(
                    ctx.kwargs.get("informative_priors_n_samples", 1000)
                ),
                tau=float(ctx.kwargs.get("informative_priors_tau", 1.0)),
                verbose=bool(
                    ctx.kwargs.get("informative_priors_verbose", True)
                ),
            )

    # --- Freeze API: extract point estimates + resolve cascade source ---
    # The freeze API is activated only when a cascade is supplied.  When
    # `informative_priors_from is None`, normalize freeze to empty so a
    # plain Laplace fit (no cascade) is unaffected by the default-on
    # `informative_priors_freeze=("r", "eta")` kwarg.
    freeze_values = None
    freeze_params: tuple = ()
    cascade_source = None
    cascade_source_counts = None
    if informative_priors_from is not None:
        # Read and normalize the freeze tuple.  Accept any iterable
        # of {"r", "mu", "eta"} OR their descriptive aliases
        # ({"dispersion", "expression"/"mean_expression",
        # "capture_efficiency"}) — see FREEZE_KEY_ALIASES in
        # parameter_mapping.py.  ``normalize_freeze_keys`` resolves
        # aliases to internal short names and raises on ambiguity
        # (e.g. ("r", "dispersion")).
        from ...models.config.parameter_mapping import normalize_freeze_keys
        # Per-base-model defaults — the plan §5.4 "Level 4" cascade
        # for each variant:
        #   NBLN          → (r, eta)
        #   TSLN-Rate     → (mu, burst_size, k_off)
        #   TSLN-Logit    → (rate, kappa, eta_anchor)   [Rev 3]
        if base_model == "twostate_ln_rate":
            _default_freeze = ("mu", "burst_size", "k_off")
        elif base_model == "twostate_ln_logit":
            _default_freeze = ("rate", "kappa", "eta_anchor")
        else:
            _default_freeze = ("r", "eta")
        raw_freeze = ctx.kwargs.get(
            "informative_priors_freeze", _default_freeze
        )
        if raw_freeze is None:
            freeze_params = ()
        else:
            freeze_params = normalize_freeze_keys(raw_freeze)

        # PR-2 capture restriction (Rev 4): for TSLN-Logit, when the
        # SVI source has per-cell capture (``capture_mode_override ==
        # "eta"``), we MUST route capture through the fixed-offset
        # path because soft-cascade eta would require the joint
        # Newton (deferred to phase 3).  Auto-include ``"eta"`` in
        # ``freeze_params`` when the user didn't drop it explicitly,
        # so the cascade adapter populates ``freeze_values["eta"]``
        # from the SVI MAP and the obs model routes to the
        # ``x_only_offset`` Newton.
        if (
            base_model == "twostate_ln_logit"
            and capture_mode_override == "eta"
            and "eta" not in freeze_params
        ):
            import logging
            logging.getLogger("scribe").info(
                "TSLN-Logit: auto-adding 'eta' to informative_priors_freeze "
                "because the SVI source has per-cell capture and Rev 4 "
                "permits only fixed-offset capture in PR-2.  To opt out "
                "of capture entirely, refit the SVI source without "
                "capture or pass informative_priors_freeze without 'eta' "
                "explicitly after this point."
            )
            freeze_params = freeze_params + ("eta",)
        # Build the freeze-values bundle from SVI's get_map().
        if freeze_params:
            if base_model in ("twostate_ln_rate", "twostate_ln_logit"):
                from ...laplace.priors import (
                    freeze_values_from_twostate_results,
                )

                _target_variant = (
                    "rate" if base_model == "twostate_ln_rate" else "logit"
                )
                freeze_values = freeze_values_from_twostate_results(
                    informative_priors_from,
                    target_positive_transform=_pos_xform_str,
                    target_n_genes=ctx.n_genes,
                    target_n_cells=ctx.n_cells,
                    target_variant=_target_variant,
                    target_gene_names=target_gene_names,
                    target_gene_mask=getattr(
                        ctx, "_gene_coverage_mask", None
                    ),
                    source_counts=ctx.count_data,
                    freeze_params=freeze_params,
                    verbose=bool(
                        ctx.kwargs.get(
                            "informative_priors_verbose", True
                        )
                    ),
                )
            else:
                from ...laplace.priors import freeze_values_from_results

                freeze_values = freeze_values_from_results(
                    informative_priors_from,
                    target_positive_transform=_pos_xform_str,
                    target_n_genes=ctx.n_genes,
                    target_n_cells=ctx.n_cells,
                    target_gene_names=target_gene_names,
                    target_gene_mask=getattr(
                        ctx, "_gene_coverage_mask", None
                    ),
                    source_counts=ctx.count_data,
                    freeze_params=freeze_params,
                    verbose=bool(
                        ctx.kwargs.get(
                            "informative_priors_verbose", True
                        )
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

    # Phase-3: W-shrinkage prior plumbing.  Normalize the explicit no-op
    # config {"type": "none"} to None *before* scope validation so the
    # no-op config is universally accepted (Round-1 fix 7).  Validate
    # against base_model ∈ {nbln, pln} and method == laplace.
    w_prior = ctx.kwargs.get("w_prior")
    if isinstance(w_prior, dict) and w_prior.get("type") == "none":
        w_prior = None
    if w_prior is not None:
        bm = getattr(ctx.model_config, "base_model", None)
        method = getattr(ctx.inference_config, "method", None)
        method_value = getattr(method, "value", method)
        if (
            bm not in ("nbln", "pln")
            or str(method_value).lower() != "laplace"
        ):
            raise ValueError(
                "w_prior is supported only for model='nbln' or 'pln' "
                "with inference_method='laplace'; got "
                f"base_model={bm!r}, method={method_value!r}."
            )

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
        w_prior=w_prior,
    )
    ctx.results = results

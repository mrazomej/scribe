"""
Stage 2c-annotation: Build annotation prior logits for mixture models.

Resolves ``n_components`` from annotations when not explicitly set,
auto-downgrades to non-mixture mode when too few labels survive
filtering, and builds per-cell logit matrices.

FitContext reads : adata, n_cells, n_components, kwargs[annotation_key,
                   annotation_confidence, annotation_component_order,
                   annotation_min_cells, dataset_key, model_config,
                   expression_prior]
FitContext writes: annotation_prior_logits, _label_map,
                   _component_mapping, n_components,
                   effective_mixture_params, and may mutate
                   kwargs[expression_prior]
"""

import logging

from ...core.annotation_prior import (
    build_annotation_prior_logits,
    build_component_mapping,
    validate_annotation_prior_logits,
)
from ...models.config.enums import HierarchicalPriorType
from ..helpers import _count_unique_labels, _normalize_prior_type_name
from ..context import FitContext

_log = logging.getLogger(__name__)


def build_annotation_priors(ctx: FitContext) -> None:
    """
    Build annotation prior logits for mixture-model fits.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.

    Raises
    ------
    ValueError
        If ``annotation_key`` is given but counts is not AnnData.
    """
    kw = ctx.kwargs
    annotation_key = kw.get("annotation_key")

    # -- Pre-built annotation logits supplied via ``priors={...}`` ------------
    # Power-user path: inject a per-cell ``(n_cells, n_components)`` logit
    # matrix directly through the unified ``priors`` dict instead of deriving
    # it from a label column.  Pop it here (this stage runs before
    # ``build_model_config`` normalizes the remaining model-parameter priors).
    priors = ctx.priors if isinstance(ctx.priors, dict) else None
    if priors is not None and "annotation_logits" in priors:
        import jax.numpy as jnp

        if annotation_key is not None:
            raise ValueError(
                "Provide annotation component priors EITHER via "
                "annotation_key (derive from a label column) OR via "
                "priors={'annotation_logits': <(n_cells, n_components) "
                "array>}, not both."
            )
        logits = jnp.asarray(priors.pop("annotation_logits"))
        _n_comp = ctx.n_components
        if _n_comp is None:
            _mc = kw.get("model_config")
            _n_comp = (
                _mc.n_components
                if _mc is not None and _mc.n_components is not None
                else int(logits.shape[-1])
            )
        validate_annotation_prior_logits(logits, ctx.n_cells, _n_comp)
        ctx.n_components = _n_comp
        ctx.effective_mixture_params = kw.get("mixture_params", "all")
        ctx.annotation_prior_logits = logits
        return

    if annotation_key is None:
        ctx.effective_mixture_params = kw.get("mixture_params", "all")
        return

    adata = ctx.adata
    if adata is None:
        raise ValueError(
            "annotation_key requires counts to be an AnnData object "
            "(not a raw array), so that adata.obs can be read."
        )

    model_config = kw.get("model_config")
    _n_comp = ctx.n_components
    _n_comp_inferred = False
    if _n_comp is None and model_config is not None:
        _n_comp = model_config.n_components
    _min_cells = kw.get("annotation_min_cells") or 0
    if _n_comp is None:
        _n_comp = _count_unique_labels(
            adata, annotation_key, min_cells=_min_cells
        )
        _n_comp_inferred = True

    annotation_confidence = kw.get("annotation_confidence", 3.0)
    annotation_component_order = kw.get("annotation_component_order")
    dataset_key = kw.get("dataset_key")
    expression_prior = kw.get("expression_prior", "none")
    mixture_params = kw.get("mixture_params", "all")

    # -- Auto-downgrade when <=1 surviving class ------------------------------
    if _n_comp_inferred and _n_comp <= 1:
        downgraded_msgs = []
        ctx.n_components = None
        ctx.effective_mixture_params = None

        if (
            _normalize_prior_type_name(expression_prior)
            != HierarchicalPriorType.NONE.value
        ):
            _old = _normalize_prior_type_name(expression_prior)
            kw["expression_prior"] = HierarchicalPriorType.NONE.value
            downgraded_msgs.append(
                f"expression_prior='{_old}' -> 'none'"
            )

        _suffix = (
            f"; {'; '.join(downgraded_msgs)}" if downgraded_msgs else ""
        )
        _log.warning(
            "annotation_key/annotation_min_cells left <=1 surviving "
            "annotation class after filtering. "
            "Auto-downgrading to non-mixture mode "
            f"(n_components=None, mixture_params ignored{_suffix})."
        )
        return

    # -- Build annotation logits ----------------------------------------------
    ctx.n_components = _n_comp
    ctx.effective_mixture_params = mixture_params

    _component_mapping = None
    _effective_order = annotation_component_order
    _shared_override = None
    if model_config is not None:
        _shared_override = getattr(model_config, "shared_components", None)

    if dataset_key is not None:
        _component_mapping = build_component_mapping(
            adata=adata,
            annotation_key=annotation_key,
            dataset_key=dataset_key,
            min_cells=_min_cells,
            shared_components=_shared_override,
        )
        if _effective_order is None:
            _effective_order = _component_mapping.component_order
        if _n_comp_inferred:
            ctx.n_components = _component_mapping.n_components
            _n_comp = ctx.n_components

    logits, _label_map = build_annotation_prior_logits(
        adata=adata,
        obs_key=annotation_key,
        n_components=_n_comp,
        confidence=annotation_confidence,
        component_order=_effective_order,
        min_cells=_min_cells,
    )
    validate_annotation_prior_logits(logits, ctx.n_cells, _n_comp)

    ctx.annotation_prior_logits = logits
    ctx._label_map = _label_map
    ctx._component_mapping = _component_mapping

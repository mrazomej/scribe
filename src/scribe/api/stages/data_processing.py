"""
Stage 2: Process count data, build dataset indices, and apply
single-dataset hierarchy downgrade.

FitContext reads : counts, kwargs[cells_axis, layer, dataset_key,
                   n_datasets, auto_downgrade_..., *_dataset_prior, ...]
FitContext writes: count_data, adata, n_cells, n_genes, data_config,
                   _total_count_max, dataset_indices, n_datasets,
                   _adata_for_inference, _original_n_genes,
                   and mutates kwargs for downgraded priors
"""

import logging
from typing import List

import jax.numpy as jnp

from ...inference.utils import process_counts_data
from ...models.config import DataConfig
from ..context import FitContext

_log = logging.getLogger(__name__)


def process_data_and_datasets(ctx: FitContext) -> None:
    """
    Process raw counts into a JAX array, derive per-cell dataset
    indices, and apply the single-dataset hierarchy downgrade when
    applicable.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  Multiple fields are populated here
        for consumption by downstream stages.

    Raises
    ------
    ValueError
        If ``dataset_key`` is given but counts is not AnnData, if the
        column is missing, if ``n_datasets`` conflicts, or if
        dataset-level priors are used without ``dataset_key``.
    """
    kw = ctx.kwargs
    import numpy as np

    # -- Process raw counts into a JAX array ----------------------------------
    data_config = DataConfig(
        cells_axis=kw.get("cells_axis", 0),
        layer=kw.get("layer"),
    )
    count_data, adata, n_cells, n_genes = process_counts_data(
        ctx.counts, data_config
    )

    # Upper bound on per-cell total counts for predictive sampling.
    _total_count_max = int(
        1.5 * float(np.asarray(count_data).sum(axis=1).max())
    )

    ctx.count_data = count_data
    ctx.adata = adata
    ctx.n_cells = n_cells
    ctx.n_genes = n_genes
    ctx.data_config = data_config
    ctx._total_count_max = _total_count_max
    ctx._original_n_genes = n_genes
    ctx._adata_for_inference = adata

    # -- Build grouping / dataset indices -------------------------------------
    # Two paths: the single-`str` legacy path (preserved bit-for-bit) and the
    # multi-factor path (list of factors / structured hierarchy / interactions,
    # and/or dict-valued dataset priors). In both cases the leaf axis is the
    # existing dataset axis: ``dataset_indices`` is the per-cell leaf index.
    from ...models.config.grouping import (
        GroupingSpec,
        Factor,
        normalize_grouping,
        _reduce_leaf_axis_family,
    )

    dataset_key = kw.get("dataset_key")
    hierarchy = kw.get("hierarchy")
    interactions = kw.get("interactions")
    n_datasets = kw.get("n_datasets")
    dataset_indices = None
    grouping_spec = None

    auto_downgrade = kw.get("auto_downgrade_single_dataset_hierarchy", True)
    expression_dataset_prior = kw.get("expression_dataset_prior", "none")
    prob_dataset_prior = kw.get("prob_dataset_prior", "none")
    prob_dataset_mode = kw.get("prob_dataset_mode", "gene_specific")
    zi_dataset_prior = kw.get("zero_inflation_dataset_prior", "none")
    od_dataset_prior = kw.get("overdispersion_dataset_prior", "none")
    regime_dataset_prior = kw.get("regime_dataset_prior", "none")
    dataset_mixing = kw.get("dataset_mixing")
    prob_prior = kw.get("prob_prior", "none")
    zi_prior = kw.get("zero_inflation_prior", "none")

    # The multi-factor path is taken for a list/hierarchy/interactions request
    # or when any dataset-prior is given as a per-factor dict.
    _priors_have_dict = any(
        isinstance(v, dict)
        for v in (
            expression_dataset_prior,
            prob_dataset_prior,
            zi_dataset_prior,
            od_dataset_prior,
            regime_dataset_prior,
        )
    )
    _multifactor = (
        hierarchy is not None
        or interactions is not None
        or isinstance(dataset_key, (list, tuple))
        or _priors_have_dict
    )
    _grouping_requested = dataset_key is not None or hierarchy is not None

    if _grouping_requested and adata is None:
        raise ValueError(
            "dataset_key / hierarchy require counts to be an AnnData object "
            "(not a raw array), so that adata.obs can be read."
        )

    if _multifactor:
        result = normalize_grouping(
            dataset_key=dataset_key,
            hierarchy=hierarchy,
            interactions=interactions,
            obs=adata.obs if adata is not None else None,
            dataset_priors={
                "expression": expression_dataset_prior,
                "prob": prob_dataset_prior,
                "zero_inflation": zi_dataset_prior,
                "overdispersion": od_dataset_prior,
                "regime": regime_dataset_prior,
            },
        )
        if result is not None:
            grouping_spec, leaf_index = result
            if (
                n_datasets is not None
                and n_datasets != grouping_spec.n_leaves
            ):
                raise ValueError(
                    f"n_datasets={n_datasets} but the grouping has "
                    f"{grouping_spec.n_leaves} present leaf combinations."
                )
            n_datasets = grouping_spec.n_leaves
            dataset_indices = jnp.asarray(
                np.asarray(leaf_index, dtype=np.int32)
            )
            # Reduce per-factor families to the single leaf-axis family that
            # the existing single-axis hierarchy consumes in Milestone 1.
            expression_dataset_prior = _reduce_leaf_axis_family(
                grouping_spec, "expression"
            )
            prob_dataset_prior = _reduce_leaf_axis_family(grouping_spec, "prob")
            zi_dataset_prior = _reduce_leaf_axis_family(
                grouping_spec, "zero_inflation"
            )
            od_dataset_prior = _reduce_leaf_axis_family(
                grouping_spec, "overdispersion"
            )
            regime_dataset_prior = _reduce_leaf_axis_family(
                grouping_spec, "regime"
            )
    elif dataset_key is not None:
        # Legacy single-string path: preserved verbatim so the model is
        # bit-identical to the pre-multi-factor implementation.
        if dataset_key not in adata.obs.columns:
            raise ValueError(
                f"dataset_key '{dataset_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        ds_cat = adata.obs[dataset_key].astype("category")
        ds_codes = ds_cat.cat.codes.values
        _inferred = len(ds_cat.cat.categories)
        if n_datasets is not None and n_datasets != _inferred:
            raise ValueError(
                f"n_datasets={n_datasets} but dataset_key "
                f"'{dataset_key}' has {_inferred} unique "
                f"values: {list(ds_cat.cat.categories)}"
            )
        n_datasets = _inferred
        dataset_indices = jnp.asarray(np.asarray(ds_codes, dtype=np.int32))
        # A one-factor grouping spec carries labels (donor/condition names) for
        # downstream comparisons. Built directly from the categorical levels so
        # leaf indices align with ``cat.codes`` (incl. any unused categories).
        _levels = tuple(str(c) for c in ds_cat.cat.categories)
        grouping_spec = GroupingSpec(
            factors=(
                Factor(
                    name=dataset_key,
                    kind="base",
                    levels=_levels,
                    leaf_to_level=tuple(range(len(_levels))),
                    priors={},
                ),
            ),
            leaf_labels=_levels,
            n_leaves=len(_levels),
        )

    # -- Single-dataset hierarchy downgrade -----------------------------------

    if auto_downgrade and n_datasets == 1 and dataset_indices is not None:
        msgs: List[str] = []

        if expression_dataset_prior != "none":
            expression_dataset_prior = "none"
            msgs.append("expression_dataset_prior -> 'none'")

        if prob_dataset_prior != "none":
            if prob_dataset_mode == "scalar":
                prob_dataset_prior = "none"
                msgs.append("prob_dataset_prior='scalar' mode -> 'none'")
            else:
                prob_prior = prob_dataset_prior
                prob_dataset_prior = "none"
                msgs.append(f"prob_dataset_prior -> prob_prior='{prob_prior}'")

        if zi_dataset_prior != "none":
            zi_prior = zi_dataset_prior
            zi_dataset_prior = "none"
            msgs.append(
                f"zero_inflation_dataset_prior -> "
                f"zero_inflation_prior='{zi_prior}'"
            )

        if od_dataset_prior != "none":
            od_dataset_prior = "none"
            msgs.append("overdispersion_dataset_prior -> 'none'")

        # The two-state regime hierarchy collapses to the gene-level prior
        # when only one dataset is present (there is nothing to pool across).
        if regime_dataset_prior != "none":
            regime_dataset_prior = "none"
            msgs.append("regime_dataset_prior -> 'none'")

        if msgs:
            n_datasets = None
            grouping_spec = None
            if dataset_mixing is None:
                dataset_mixing = False
            _log.warning(
                f"Detected a single dataset from dataset_key "
                f"'{dataset_key}'. Applied automatic hierarchy downgrade: "
                + "; ".join(msgs)
            )

    # Enforce dataset-level priors require dataset_key mapping.
    _uses_ds = (
        expression_dataset_prior != "none"
        or prob_dataset_prior != "none"
        or zi_dataset_prior != "none"
        or od_dataset_prior != "none"
        or regime_dataset_prior != "none"
    )
    if _uses_ds and dataset_indices is None:
        raise ValueError(
            "Dataset-level hierarchical priors "
            "(expression_dataset_prior, prob_dataset_prior, "
            "zero_inflation_dataset_prior, overdispersion_dataset_prior, "
            "regime_dataset_prior) "
            "require dataset_key so cells can "
            "be mapped to datasets. Provide dataset_key as an adata.obs "
            "column when using dataset-level hierarchical priors."
        )

    # -- Write back resolved values -------------------------------------------
    ctx.dataset_indices = dataset_indices
    ctx.n_datasets = n_datasets
    ctx.grouping_spec = grouping_spec
    kw["n_datasets"] = n_datasets
    kw["expression_dataset_prior"] = expression_dataset_prior
    kw["prob_dataset_prior"] = prob_dataset_prior
    kw["prob_dataset_mode"] = prob_dataset_mode
    kw["zero_inflation_dataset_prior"] = zi_dataset_prior
    kw["overdispersion_dataset_prior"] = od_dataset_prior
    kw["regime_dataset_prior"] = regime_dataset_prior
    kw["dataset_mixing"] = dataset_mixing
    kw["prob_prior"] = prob_prior
    kw["zero_inflation_prior"] = zi_prior

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

    # -- Build dataset indices ------------------------------------------------
    dataset_key = kw.get("dataset_key")
    n_datasets = kw.get("n_datasets")
    dataset_indices = None

    if dataset_key is not None:
        if adata is None:
            raise ValueError(
                "dataset_key requires counts to be an AnnData object "
                "(not a raw array), so that adata.obs can be read."
            )
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

    # -- Single-dataset hierarchy downgrade -----------------------------------
    auto_downgrade = kw.get("auto_downgrade_single_dataset_hierarchy", True)
    expression_dataset_prior = kw.get("expression_dataset_prior", "none")
    prob_dataset_prior = kw.get("prob_dataset_prior", "none")
    prob_dataset_mode = kw.get("prob_dataset_mode", "gene_specific")
    zi_dataset_prior = kw.get("zero_inflation_dataset_prior", "none")
    od_dataset_prior = kw.get("overdispersion_dataset_prior", "none")
    dataset_mixing = kw.get("dataset_mixing")
    prob_prior = kw.get("prob_prior", "none")
    zi_prior = kw.get("zero_inflation_prior", "none")

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

        if msgs:
            n_datasets = None
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
    )
    if _uses_ds and dataset_indices is None:
        raise ValueError(
            "Dataset-level hierarchical priors "
            "(expression_dataset_prior, prob_dataset_prior, "
            "zero_inflation_dataset_prior, overdispersion_dataset_prior) "
            "require dataset_key so cells can "
            "be mapped to datasets. Provide dataset_key as an adata.obs "
            "column when using dataset-level hierarchical priors."
        )

    # -- Write back resolved values -------------------------------------------
    ctx.dataset_indices = dataset_indices
    ctx.n_datasets = n_datasets
    kw["n_datasets"] = n_datasets
    kw["expression_dataset_prior"] = expression_dataset_prior
    kw["prob_dataset_prior"] = prob_dataset_prior
    kw["prob_dataset_mode"] = prob_dataset_mode
    kw["zero_inflation_dataset_prior"] = zi_dataset_prior
    kw["overdispersion_dataset_prior"] = od_dataset_prior
    kw["dataset_mixing"] = dataset_mixing
    kw["prob_prior"] = prob_prior
    kw["zero_inflation_prior"] = zi_prior

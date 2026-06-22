"""Factor-level (population) differential expression for multi-factor models.

The multi-factor hierarchy (see ``paper/_hierarchical_datasets.qmd``) fits a
shared population effect for a *contrast* factor (e.g. treatment) across the
levels of a *pairing* factor (e.g. donor). :func:`compare_groups` summarises
that contrast at the population level.

Default estimand — ``"paired_main_effect"`` (recommended for paired designs):
for each pairing level present at *both* contrast levels, form the within-pair
CLR difference between the two leaf compositions (exactly the leaf-vs-leaf
quantity of :func:`compare_datasets`), then average those per-draw differences
over the pairing levels with normalized weights. Averaging is done on the
CLR-space deltas (a linear operation), never on raw ``r``/``mu``/logit values.

With a donor x condition interaction present, this estimates the *observed-donor
average* treatment effect (incl. the average interaction deviation), which is
not identical to the pure main-effect parameter; the distinction is documented
in the paper.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Pure helpers (no inference) — unit-testable
# ------------------------------------------------------------------------------


def _grouping_spec(results):
    return getattr(getattr(results, "model_config", None), "grouping_spec", None)


def _resolve_pairs(
    grouping_spec,
    factor_name: str,
    level_A: str,
    level_B: str,
    pairing_factor: Optional[str],
) -> Tuple[str, List[Tuple[str, int, int]], List[str]]:
    """Resolve the pairing factor and the present (pairing-level, leaf_A, leaf_B).

    Returns ``(pairing_factor, present, dropped)`` where ``present`` is a list of
    ``(pairing_level_label, leaf_A_index, leaf_B_index)`` for pairing levels
    realised at *both* contrast levels, and ``dropped`` lists the pairing levels
    missing one side.

    Raises on an unknown contrast factor, an ambiguous/invalid pairing factor,
    or extra (non-contrast, non-pairing) grouping factors (not yet supported).
    """
    base = [f for f in grouping_spec.factors if f.kind == "base"]
    base_names = [f.name for f in base]
    if factor_name not in base_names:
        raise ValueError(
            f"contrast factor {factor_name!r} is not a base grouping factor. "
            f"Base factors: {base_names}."
        )

    candidates = [n for n in base_names if n != factor_name]
    if pairing_factor is None:
        if len(candidates) == 1:
            pairing_factor = candidates[0]
        else:
            raise ValueError(
                f"compare_groups(): more than one non-contrast factor "
                f"({candidates}); pass pairing_factor= to choose which factor "
                f"is held across the contrast."
            )
    elif pairing_factor not in candidates:
        raise ValueError(
            f"pairing_factor {pairing_factor!r} must be a base factor other "
            f"than the contrast {factor_name!r}. Candidates: {candidates}."
        )

    others = [n for n in candidates if n != pairing_factor]
    if others:
        raise NotImplementedError(
            "compare_groups() currently supports designs whose only grouping "
            f"factors are the contrast and pairing factors; extra factors "
            f"{others} would need to be held fixed or marginalized (not yet "
            "implemented). Use compare_datasets() on specific leaves instead."
        )

    coords = grouping_spec.leaf_coords()

    def _leaf_where(contrast_level, pairing_level):
        hits = [
            i
            for i, c in enumerate(coords)
            if c.get(factor_name) == contrast_level
            and c.get(pairing_factor) == pairing_level
        ]
        return hits[0] if len(hits) == 1 else None

    pairing_levels = next(
        f.levels for f in base if f.name == pairing_factor
    )
    present: List[Tuple[str, int, int]] = []
    dropped: List[str] = []
    for pv in pairing_levels:
        la = _leaf_where(level_A, pv)
        lb = _leaf_where(level_B, pv)
        if la is not None and lb is not None:
            present.append((pv, la, lb))
        else:
            dropped.append(pv)
    return pairing_factor, present, dropped


def _pair_weights(
    pairs: List[Tuple[str, int, int]],
    n_cells_per_leaf,
    weighting: str,
) -> np.ndarray:
    """Per-pair weights (normalized to sum 1), derived from each pair's leaves.

    ``weighting`` ∈ {"uniform", "min_cells", "harmonic", "total_cells"}. The
    cell-count weightings fall back to uniform when leaf cell counts are
    unavailable.
    """
    npairs = len(pairs)
    if weighting == "uniform" or n_cells_per_leaf is None:
        w = np.ones(npairs, dtype=float)
    else:
        nc = np.asarray(n_cells_per_leaf, dtype=float)
        w = np.empty(npairs, dtype=float)
        for i, (_, la, lb) in enumerate(pairs):
            na, nb = float(nc[la]), float(nc[lb])
            if weighting == "min_cells":
                w[i] = min(na, nb)
            elif weighting == "harmonic":
                w[i] = 0.0 if (na <= 0 or nb <= 0) else 2.0 / (1.0 / na + 1.0 / nb)
            elif weighting == "total_cells":
                w[i] = na + nb
            else:
                raise ValueError(
                    f"unknown pair_weighting {weighting!r}; choose from "
                    "uniform / min_cells / harmonic / total_cells."
                )
    total = float(w.sum())
    if total <= 0:
        w = np.ones(npairs, dtype=float)
        total = float(w.sum())
    return w / total


# ------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------


def compare_groups(
    results,
    factor_name: str,
    level_A: str,
    level_B: str,
    *,
    estimand: str = "paired_main_effect",
    pairing_factor: Optional[str] = None,
    pair_weighting: str = "uniform",
    incomplete_pairs: str = "drop",
    min_complete_pairs: int = 2,
    method: str = "empirical",
    component: Optional[int] = None,
    n_samples: Optional[int] = None,
    **kwargs,
):
    """Population-level differential expression for one grouping factor.

    Parameters
    ----------
    results : object
        Multi-dataset results object with a ``grouping_spec`` (built by
        ``scribe.fit`` with ``hierarchy=``/``dataset_key=[...]``).
    factor_name : str
        The contrast (base) factor, e.g. ``"perturbation"``.
    level_A, level_B : str
        The two levels to contrast (B vs A), e.g. ``"control"`` /
        ``"panobinostat"``.
    estimand : {"paired_main_effect"}, default
        Currently only the paired main-effect estimand is implemented.
    pairing_factor : str, optional
        The grouping factor held across the contrast (e.g. ``"sample"``).
        Inferred when there is exactly one non-contrast factor; required
        otherwise.
    pair_weighting : {"uniform","min_cells","harmonic","total_cells"}
        How to weight pairing levels when averaging within-pair deltas.
    incomplete_pairs : {"drop","error"}
        Whether to drop (with a warning) or error on pairing levels missing
        one contrast level.
    min_complete_pairs : int, default 2
        Error if fewer than this many complete pairs remain.
    method : {"empirical"}, default
        DE method for the averaged deltas. (Shrinkage on factor-level deltas is
        a future addition.)
    component : int, optional
        Mixture component to select before leaf slicing.
    n_samples : int, optional
        Number of posterior draws to sample on the full results before slicing
        leaves. This sets ``N`` in the returned ``delta_samples`` (``N =
        n_samples x n_samples_dirichlet``). Defaults to the existing posterior
        samples if present, else 100. Pass a larger value (e.g. ``5000``) for
        smoother lfsr/PEFP estimates. Ignored for MCMC results (fixed at fit
        time). Note: more draws cost proportionally more memory and time.
    **kwargs
        Forwarded to :func:`compare` for each within-pair comparison
        (e.g. ``gene_mask``, ``n_samples_dirichlet``, ``rng_key``).

    Returns
    -------
    ScribeEmpiricalDEResults
        DE result on the averaged per-draw CLR deltas; compatible with the
        standard gene-level / lfsr / PEFP analysis.  The per-arm summaries
        (``mu_map_A``/``mu_map_B`` and the biological sample tensors) are the
        donor-weighted-average population value for each contrast level, so the
        object also drives the mean-expression DE plots and ``biological_level``
        like a leaf-vs-leaf ``compare()`` result.  ``simplex_A``/``simplex_B``
        are not aggregated (see Notes), so a post-hoc mask recompute is
        unavailable — pass ``gene_mask`` here to re-mask within each pair.

    Notes
    -----
    The delta sign convention matches the rest of SCRIBE's DE:
    ``delta = CLR(level_A) - CLR(level_B)``. So a gene up-regulated in
    ``level_B`` relative to ``level_A`` has a *negative* delta.
    """
    from .results import compare, ScribeEmpiricalDEResults

    if estimand != "paired_main_effect":
        raise NotImplementedError(
            f"estimand={estimand!r} is not yet implemented; only "
            "'paired_main_effect' is available."
        )
    if method != "empirical":
        raise NotImplementedError(
            f"method={method!r} is not yet implemented for compare_groups; "
            "use 'empirical' (shrinkage on factor-level deltas is planned)."
        )

    gs = _grouping_spec(results)
    if gs is None:
        raise ValueError(
            "compare_groups() requires a multi-factor fit (results."
            "model_config.grouping_spec must be set)."
        )

    pairing_factor, present, dropped = _resolve_pairs(
        gs, factor_name, level_A, level_B, pairing_factor
    )

    if dropped:
        if incomplete_pairs == "error":
            raise ValueError(
                f"pairing levels missing one contrast level: {dropped}. "
                "Pass incomplete_pairs='drop' to skip them."
            )
        warnings.warn(
            f"compare_groups(): dropping {len(dropped)} pairing level(s) "
            f"missing {factor_name!r}={level_A!r} or {level_B!r}: {dropped}.",
            stacklevel=2,
        )
    if len(present) < min_complete_pairs:
        raise ValueError(
            f"only {len(present)} complete pair(s) for {factor_name!r} "
            f"({level_A!r} vs {level_B!r}); need >= {min_complete_pairs}."
        )

    weights = _pair_weights(
        present, getattr(results, "_n_cells_per_dataset", None), pair_weighting
    )

    # Draw posterior samples ONCE on the full results so every leaf slice below
    # shares the same posterior draws — essential for the paired (within-donor)
    # CLR differences to be coherent across draws. The number of posterior draws
    # sets N in the returned delta_samples (N = n_samples x n_samples_dirichlet);
    # the default get_posterior_samples count is only 100, so expose n_samples to
    # raise it.
    if hasattr(results, "get_posterior_samples"):
        if n_samples is not None:
            # (Re)draw the requested number so the count is honoured even when a
            # smaller set is already cached. MCMC results have a fixed sample
            # count from the fit and ignore n_samples.
            try:
                results.get_posterior_samples(n_samples=int(n_samples))
            except TypeError:
                warnings.warn(
                    "n_samples is ignored for this results type (e.g. MCMC, "
                    "whose posterior sample count is fixed at fit time).",
                    stacklevel=2,
                )
                if getattr(results, "posterior_samples", None) is None:
                    results.get_posterior_samples()
        elif getattr(results, "posterior_samples", None) is None:
            results.get_posterior_samples()

    working = results
    if component is not None:
        working = working.get_component(component)

    # Biological summaries are needed for the per-arm population fields
    # (mu_map_A/B drive the mean-expression DE plots; the per-arm sample
    # tensors drive biological_level()). Default them on; allow opt-out.
    compute_biological = kwargs.pop("compute_biological", True)

    # Per-pair within-pair quantities, then a weighted average over pairs:
    #   - ``delta`` is the paired main effect (avg of within-pair CLR deltas);
    #   - every per-arm field is the population (donor-weighted-average) value
    #     for that contrast level, so the resulting object summarises and plots
    #     exactly like a leaf-vs-leaf ``compare()`` result.
    # ``simplex_A/B`` are intentionally NOT aggregated: the paired estimand
    # averages within-pair CLR deltas, and CLR(avg simplex) != avg CLR delta,
    # so storing averaged simplices would let a post-hoc mask recompute return a
    # different estimand. Re-mask by passing ``gene_mask`` through to the pairs.
    _ARM_FIELDS = (
        "mu_map_A", "mu_map_B",
        "r_samples_A", "r_samples_B",
        "p_samples_A", "p_samples_B",
        "mu_samples_A", "mu_samples_B",
        "phi_samples_A", "phi_samples_B",
    )
    acc = {"delta_samples": None}
    acc.update({k: None for k in _ARM_FIELDS})
    _missing = set()  # fields absent from at least one pair -> drop entirely
    gene_names = None

    def _accumulate(key, w, arr):
        if arr is None:
            _missing.add(key)
            return
        contrib = w * np.asarray(arr, dtype=float)
        acc[key] = contrib if acc[key] is None else acc[key] + contrib

    last_pair = None
    for w, (_pv, leaf_A, leaf_B) in zip(weights, present):
        de_pair = compare(
            model_A=working.get_dataset(leaf_A),
            model_B=working.get_dataset(leaf_B),
            method="empirical",
            paired=True,
            compute_biological=compute_biological,
            **kwargs,
        )
        last_pair = de_pair
        if gene_names is None:
            gene_names = de_pair.gene_names
        _accumulate("delta_samples", w, de_pair.delta_samples)
        for key in _ARM_FIELDS:
            _accumulate(key, w, getattr(de_pair, key, None))

    def _final(key):
        return None if key in _missing else acc[key]

    label_A = f"{factor_name}={level_A}"
    label_B = f"{factor_name}={level_B}"
    result = ScribeEmpiricalDEResults(
        delta_samples=_final("delta_samples"),
        gene_names=gene_names,
        label_A=label_A,
        label_B=label_B,
        method="empirical",
        **{k: _final(k) for k in _ARM_FIELDS},
    )

    # The aggregated mu_map vectors are full-gene length (like compare()'s),
    # while delta_samples / gene_names are kept-gene length when a gene_mask
    # pools the rest into "other". Propagate the pairs' (shared) mask bookkeeping
    # so to_dataframe() masks the per-arm vectors down to the kept genes instead
    # of raising a length mismatch on the dropped "other" pseudo-gene.
    if last_pair is not None:
        result._gene_mask = getattr(last_pair, "_gene_mask", None)
        result._all_gene_names = getattr(last_pair, "_all_gene_names", None)

    return result

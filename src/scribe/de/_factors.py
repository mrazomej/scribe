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

from ._extract import has_compositional_marginal

_log = logging.getLogger(__name__)


# NB-family base-model roots whose empirical DE consumes the {r, p, mu, phi}
# posterior contract. Context-aware narrowing is an ALLOWLIST: any family NOT
# listed here (including future ones) bypasses narrowing and keeps today's
# full-cache draw. See _supports_de_narrowing.
_NB_FAMILY_ROOTS = frozenset({"nbdm", "zinb", "nbvcp", "zinbvcp", "bnb"})

# Explicit deny-set (defense-in-depth) for families that lack a compositional
# marginal yet are not NB {r,p} models (two-state), plus the marginal families
# (already caught by has_compositional_marginal — listed for clarity).
_DE_NARROWING_DENY = frozenset({
    "twostate", "twostatevcp", "twostate_ln_rate", "twostate_ln_logit",
    "nbln", "pln", "lnm", "lnmvcp",
})


# ------------------------------------------------------------------------------
# Pure helpers (no inference) — unit-testable
# ------------------------------------------------------------------------------


def _grouping_spec(results):
    return getattr(getattr(results, "model_config", None), "grouping_spec", None)


def _supports_de_narrowing(results) -> bool:
    """Whether context-aware posterior narrowing applies to ``results``.

    Narrowing the posterior draw to the DE keep-set ``{r, p, mu, phi}``
    (``purpose="de_paired"``) or to a factor's effect site
    (``purpose="de_effect"``) is an **NB-family** optimisation: those keep-sets
    are the empirical-DE input contract only for Negative-Binomial-family
    models. Other families either drive DE from a fitted compositional marginal
    (LNM / PLN / NBLN / TSLN) or use a different concentration contract
    (two-state: ``alpha`` / ``beta`` / ``r_hat``), so applying the NB keep-set
    would drop sites their DE needs. Those families therefore **bypass**
    narrowing and keep the full-cache draw (identical to today's behaviour).

    The predicate is an explicit ALLOWLIST: only recognised NB-family roots
    return ``True``, so any unrecognised / future family is conservatively not
    narrowed until deliberately added (and tested). ``has_compositional_marginal``
    and an explicit deny-set provide defense-in-depth, evaluated first.

    Parameters
    ----------
    results
        A fitted results object exposing ``model_config``.

    Returns
    -------
    bool
        ``True`` iff ``results`` is an NB-family fit eligible for DE narrowing.
    """
    # Defense-in-depth, first: marginal-driven families (LNM/PLN/NBLN/TSLN)
    # never read the NB posterior {r,p} path.
    if has_compositional_marginal(results):
        return False
    cfg = getattr(results, "model_config", None)
    if cfg is None:
        return False
    bm = getattr(cfg, "base_model", None)
    bm_str = str(getattr(bm, "value", bm) or "").lower()
    if bm_str in _DE_NARROWING_DENY:
        return False
    # Final decision: explicit NB-family allowlist on the base-model root.
    return bm_str.split("_")[0] in _NB_FAMILY_ROOTS


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


def _ensure_posterior_draw(
    results,
    n_samples,
    batch_size,
    convert_to_numpy,
    *,
    purpose=None,
    return_sites=None,
):
    """Make ``results.posterior_samples`` available, honouring ``n_samples``.

    Reuses cached draws when their count already matches the request **and** the
    cached site-set covers what this call needs (so a second ``compare_groups``
    call with the same ``n_samples`` does not re-sample a large posterior — the
    dominant cost). Otherwise draws, offloading to host RAM by default for large
    ``n_samples`` and chunking with ``batch_size`` to bound memory. Shared by
    both estimands; a no-op for results without sampling.

    ``purpose`` / ``return_sites`` request a context-aware narrowed draw (e.g.
    ``purpose="de_paired"`` keeps only ``{r, p, mu, phi}``). They are forwarded
    to ``get_posterior_samples`` and used to validate cache reuse: a cache is
    reusable only when it is *full* or its requested keep-set is a superset of
    this call's keep-set (otherwise a narrower prior draw — e.g. a ``de_effect``
    draw — would be wrongly reused for a ``de_paired`` call that needs ``r``).
    """
    if not hasattr(results, "get_posterior_samples"):
        return

    # Local import to avoid a de -> svi import cycle at module load.
    from ..svi._posterior_policy import resolve_keep_set

    # Requested keep-set (intent) for this draw; None => full draw.
    requested = None
    if purpose is not None or return_sites is not None:
        requested = resolve_keep_set(
            getattr(results, "model_config", None),
            purpose=purpose,
            return_sites=return_sites,
        )

    _cached = getattr(results, "posterior_samples", None)
    _cached_n = None
    if _cached:
        try:
            _cached_n = int(next(iter(_cached.values())).shape[0])
        except (StopIteration, AttributeError, IndexError, TypeError):
            _cached_n = None
    # Cache is site-sufficient if it is full (covers any request) or its stored
    # requested keep-set already contains everything this call needs.
    _cached_full = getattr(results, "_posterior_is_full", True)
    _cached_sites = getattr(results, "_posterior_sites", None)
    if requested is None:
        _sites_ok = bool(_cached_full)
    else:
        _sites_ok = bool(_cached_full) or (
            _cached_sites is not None and set(requested) <= set(_cached_sites)
        )
    _count_ok = not (n_samples is not None and _cached_n != int(n_samples))
    if _cached is not None and _count_ok and _sites_ok:
        return

    _to_numpy = (
        bool(convert_to_numpy)
        if convert_to_numpy is not None
        else bool(n_samples is not None and int(n_samples) > 500)
    )
    kw = {"convert_to_numpy": _to_numpy}
    if n_samples is not None:
        kw["n_samples"] = int(n_samples)
    if batch_size is not None:
        kw["batch_size"] = int(batch_size)
    if purpose is not None:
        kw["purpose"] = purpose
    if return_sites is not None:
        kw["return_sites"] = return_sites
    try:
        results.get_posterior_samples(**kw)
    except TypeError:
        warnings.warn(
            "n_samples / batch_size / convert_to_numpy / purpose are not "
            "supported by this results type (e.g. MCMC, whose posterior sample "
            "count is fixed at fit time); using its default samples.",
            stacklevel=2,
        )
        if getattr(results, "posterior_samples", None) is None:
            results.get_posterior_samples()


def _compare_groups_effect(
    results,
    factor_name,
    level_A,
    level_B,
    *,
    gene_mask=None,
    n_samples=None,
    batch_size=None,
    convert_to_numpy=None,
):
    """``estimand="effect"``: DE on the fitted additive factor effect.

    Reads the per-draw effect contrast ``alpha[level_A] - alpha[level_B]``
    straight from the fitted hierarchy (via ``get_factor_effect``) — the
    log-mean treatment effect, with the donor random effect already partitioned
    out by the model. No composition sampling, no per-pair comparison, no CLR:
    cheap and memory-light. The returned ``delta_samples`` is that log-mean
    contrast (NOT a CLR composition contrast), in SCRIBE's ``A - B`` sign
    convention, so ``gene_level`` / ``to_dataframe`` give lfsr / PEFP on the
    treatment effect directly.
    """
    from .results import ScribeEmpiricalDEResults

    # Context-aware narrowing for the "effect" estimand: keep only the factor's
    # effect (+ scale) site(s). NB-family only — other families bypass narrowing
    # (full cache, current behaviour). The keep-set is pre-resolved here because
    # it depends on ``factor_name``.
    _eff_return_sites = None
    if _supports_de_narrowing(results):
        from ..svi._posterior_policy import resolve_keep_set

        _eff_return_sites = resolve_keep_set(
            getattr(results, "model_config", None),
            purpose="de_effect",
            factor_name=factor_name,
        )
    _ensure_posterior_draw(
        results,
        n_samples,
        batch_size,
        convert_to_numpy,
        return_sites=_eff_return_sites,
    )

    fx = results.get_factor_effect(factor_name)
    if level_A not in fx.levels or level_B not in fx.levels:
        raise ValueError(
            f"levels {level_A!r}/{level_B!r} are not both present for factor "
            f"{factor_name!r}; available levels: {fx.levels}."
        )
    # delta = alpha[A] - alpha[B]: matches CLR(A) - CLR(B), so a gene up in
    # level_B has a negative delta — but here in log-mean (not CLR) space.
    delta = np.asarray(fx.contrast(level_A, level_B))  # (N, [K,] G_model)
    if delta.ndim > 2:
        raise NotImplementedError(
            "estimand='effect' does not yet support mixture (K>1) models."
        )

    # Gene names over the model categories (includes the gene_coverage '_other').
    var = getattr(results, "var", None)
    names = (
        [str(n) for n in var.index]
        if var is not None
        else [f"gene_{i}" for i in range(delta.shape[1])]
    )
    names = np.asarray(names)

    keep = names != "_other"  # never report the gene_coverage anchor pseudo-gene
    if gene_mask is not None:
        gm = np.asarray(gene_mask, dtype=bool)
        if gm.shape[0] != keep.shape[0]:
            raise ValueError(
                f"gene_mask has length {gm.shape[0]} but the effect spans "
                f"{keep.shape[0]} genes (including '_other'); pass a full-length "
                "boolean mask (e.g. from composition_coverage_mask)."
            )
        keep = keep & gm
    delta = delta[:, keep]

    return ScribeEmpiricalDEResults(
        delta_samples=delta,
        gene_names=list(names[keep]),
        label_A=f"{factor_name}={level_A}",
        label_B=f"{factor_name}={level_B}",
        method="empirical",
    )


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
    batch_size: Optional[int] = None,
    convert_to_numpy: Optional[bool] = None,
    reference: Union[str, "np.ndarray", List[str]] = "clr",
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
    estimand : {"paired_main_effect", "effect"}, default "paired_main_effect"
        ``"paired_main_effect"`` — the donor-averaged within-pair **CLR**
        (compositional) contrast (the default; uses composition sampling).
        ``"effect"`` — the fitted additive **log-mean** factor effect
        ``alpha[level_A] - alpha[level_B]`` read directly from the hierarchy (no
        composition sampling, no pairing; cheap and memory-light). For a fixed
        contrast factor this is the model's treatment effect with the donor
        random effect already partitioned out. The returned ``delta_samples`` is
        then a log-mean effect, not a CLR contrast (so ``clr_*`` columns hold the
        log-mean effect); with an interaction present it is the pure main-effect
        ``alpha``, not the observed-donor average. Only ``gene_mask``, ``n_samples``,
        ``batch_size`` and ``convert_to_numpy`` apply to the ``"effect"`` path.
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
    batch_size : int, optional
        Chunk size for the posterior draw and for each pair's composition
        sampling. Lower it (e.g. ``500``) to cap peak memory for large
        ``n_samples`` over many leaves/genes.
    convert_to_numpy : bool, optional
        Offload the posterior draw to host (NumPy) memory so the device stays
        free for composition sampling — the recommended path when a large
        ``n_samples`` would not fit in GPU memory. Defaults to ``True`` when
        ``n_samples > 500`` and ``False`` otherwise; pass explicitly to force
        either path.
    reference : {"clr", "iqlr"} | list of str | boolean array, default="clr"
        Log-ratio reference frame for the ``"paired_main_effect"`` estimand,
        forwarded to each within-pair :func:`compare`. With ``"iqlr"`` (or an
        explicit reference set) **each pair resolves its own reference**, so
        the result records only the reference *mode* and keeps no aggregate
        reference mask. Must be ``"clr"`` for ``estimand="effect"`` (that path
        has no reference frame).
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

    if estimand not in ("paired_main_effect", "effect"):
        raise NotImplementedError(
            f"estimand={estimand!r} is not implemented; choose "
            "'paired_main_effect' (donor-averaged compositional contrast) or "
            "'effect' (the fitted log-mean factor effect)."
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

    if estimand == "effect":
        if not (isinstance(reference, str) and reference == "clr"):
            raise ValueError(
                "reference applies only to estimand='paired_main_effect' "
                "(the compositional CLR contrast). estimand='effect' reads "
                "the fitted log-mean effect directly and has no reference "
                "frame."
            )
        return _compare_groups_effect(
            results,
            factor_name,
            level_A,
            level_B,
            gene_mask=kwargs.get("gene_mask"),
            n_samples=n_samples,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
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
    #
    # Context-aware narrowing: for NB-family fits, draw only the DE keep-set
    # ({r, p, mu, phi}, + mixing_weights for mixtures) — dropping the per-cell
    # capture and per-factor effect tensors DE never reads, which is what
    # otherwise blows up memory at large n_samples. Non-NB families bypass
    # (full draw, current behaviour). The mixture-weight site is included
    # automatically by the policy when the model is a mixture.
    _purpose = "de_paired" if _supports_de_narrowing(results) else None
    _ensure_posterior_draw(
        results,
        n_samples,
        batch_size,
        convert_to_numpy,
        purpose=_purpose,
    )

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

    # Chunk each pair's composition sampling at the same batch_size (compare()
    # defaults to 2048 otherwise).
    _pair_kwargs = dict(kwargs)
    if batch_size is not None and "batch_size" not in _pair_kwargs:
        _pair_kwargs["batch_size"] = int(batch_size)
    # Each within-pair compare() resolves and applies its own reference set
    # (per-pair IQLR / explicit), so the reference is per-pair by design.
    _pair_kwargs["reference"] = reference

    last_pair = None
    for w, (_pv, leaf_A, leaf_B) in zip(weights, present):
        de_pair = compare(
            model_A=working.get_dataset(leaf_A),
            model_B=working.get_dataset(leaf_B),
            method="empirical",
            paired=True,
            compute_biological=compute_biological,
            **_pair_kwargs,
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
        # Carry the pairs' post-slice p/phi layouts so biological_level() reads
        # the gene axis semantically instead of falling back to an ndim heuristic
        # (which emits a deprecation warning). Same axes for the aggregate (N, G)
        # as for each pair.
        p_post_layout=getattr(last_pair, "p_post_layout", None),
        phi_post_layout=getattr(last_pair, "phi_post_layout", None),
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
    # Record the reference *mode* only. Each pair resolved its own IQLR /
    # explicit reference set, so there is no single aggregate mask; storing
    # one would be misleading. set_reference()/pathway tests read this and
    # behave correctly (no stored simplex -> set_reference rejects; a
    # non-"clr" mode -> pathway tests reject).
    result._reference = reference

    return result

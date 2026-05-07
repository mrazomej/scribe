"""Dispatch helpers for SVI vs MCMC result types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from multipledispatch import dispatch
import scribe

from .gene_selection import (
    _coerce_and_align_counts_to_results,
    _coerce_counts,
)

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout


def _coerce_counts_for_sampling(results, counts, *, context):
    """Coerce plot counts for inference-time sampling calls.

    Parameters
    ----------
    results : object
        Fitted results object.
    counts : array-like
        Counts provided by the caller.
    context : str
        Error-context label for alignment helpers.

    Returns
    -------
    numpy.ndarray
        Count matrix suited for downstream sampling methods.

    Notes
    -----
    Gene-subset results with amortized capture require full original-gene
    counts when available. In that specific case this helper preserves the
    full matrix and skips model-space realignment.
    """
    counts_arr = _coerce_counts(counts)
    uses_amortized = bool(
        hasattr(results, "_uses_amortized_capture")
        and results._uses_amortized_capture()
    )
    original_n_genes = getattr(results, "_original_n_genes", None)
    if (
        uses_amortized
        and original_n_genes is not None
        and int(original_n_genes) > int(getattr(results, "n_genes", 0))
        and counts_arr.ndim == 2
        and int(counts_arr.shape[1]) == int(original_n_genes)
    ):
        return counts_arr
    return _coerce_and_align_counts_to_results(
        counts_arr, results, context=context
    )


@dispatch(scribe.ScribeVariationalResults, object)
def _get_inference_metadata_for_filenames(results, cfg):
    """Get filename metadata for SVI runs."""
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "n_steps"):
        n_steps = cfg.inference.n_steps
    else:
        n_steps = cfg.get("n_steps", 50000)
    return {
        "run_size_value": int(n_steps),
        "run_size_label": "steps",
        "run_size_token": f"{int(n_steps)}steps",
        "n_steps": int(n_steps),
    }


@dispatch(scribe.ScribeMCMCResults, object)
def _get_inference_metadata_for_filenames(results, cfg):
    """Get filename metadata for MCMC runs."""
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "n_samples"):
        n_samples = int(cfg.inference.n_samples)
        n_warmup = int(getattr(cfg.inference, "n_warmup", 0))
    else:
        n_samples = int(cfg.get("n_samples", 1000))
        n_warmup = int(cfg.get("n_warmup", 0))
    return {
        "run_size_value": n_samples,
        "run_size_label": "samples",
        "run_size_token": f"{n_samples}samples_{n_warmup}warmup",
        "n_steps": n_samples,
    }


@dispatch(type(None), object)
def _get_inference_metadata_for_filenames(results, cfg):
    """Get filename metadata when no typed results object is available."""
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "method"):
        method = str(cfg.inference.method).lower()
    else:
        method = str(cfg.get("method", "svi")).lower()

    if method == "mcmc":
        if hasattr(cfg, "inference"):
            n_samples = int(getattr(cfg.inference, "n_samples", 1000))
            n_warmup = int(getattr(cfg.inference, "n_warmup", 0))
        else:
            n_samples = int(cfg.get("n_samples", 1000))
            n_warmup = int(cfg.get("n_warmup", 0))
        return {
            "run_size_value": n_samples,
            "run_size_label": "samples",
            "run_size_token": f"{n_samples}samples_{n_warmup}warmup",
            "n_steps": n_samples,
        }

    if hasattr(cfg, "inference"):
        n_steps = int(getattr(cfg.inference, "n_steps", 50000))
    else:
        n_steps = int(cfg.get("n_steps", 50000))
    return {
        "run_size_value": n_steps,
        "run_size_label": "steps",
        "run_size_token": f"{n_steps}steps",
        "n_steps": n_steps,
    }


@dispatch(scribe.ScribeVariationalResults)
def _get_predictive_samples_for_plot(
    results,
    *,
    rng_key,
    n_samples,
    counts,
    batch_size=None,
    store_samples=True,
):
    """Get PPC samples for plotting from variational results."""
    _ = batch_size
    counts = _coerce_counts_for_sampling(
        results, counts, context="_get_predictive_samples_for_plot"
    )
    # Generate posterior draws explicitly for this plotting call so we can run
    # one PPC batch at a time without relying on persistent cached samples.
    posterior_samples = results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples,
        # Use full-cell posterior draws to keep local latent parameter shapes
        # consistent with downstream predictive generation across all cells.
        batch_size=None,
        store_samples=False,
        counts=counts,
    )
    previous_posterior_samples = results.posterior_samples
    previous_predictive_samples = results.predictive_samples
    predictive_samples = None
    try:
        results.posterior_samples = posterior_samples
        # Forward ``counts`` so LNM-family results trigger
        # conditional PPC (per-cell ``u_T`` fixed at observed
        # library size). For non-LNM models this kwarg is ignored
        # in get_predictive_samples, so the call is safe across
        # all result types.
        predictive_samples = results.get_predictive_samples(
            rng_key=rng_key,
            store_samples=False,
            counts=counts,
        )
    finally:
        if store_samples:
            results.posterior_samples = posterior_samples
            if predictive_samples is not None:
                results.predictive_samples = predictive_samples
        else:
            results.posterior_samples = previous_posterior_samples
            results.predictive_samples = previous_predictive_samples

    return np.array(predictive_samples)


@dispatch(scribe.ScribeLaplaceResults)
def _get_predictive_samples_for_plot(
    results,
    *,
    rng_key,
    n_samples,
    counts,
    batch_size=None,
    store_samples=True,
):
    """Get PPC samples for plotting from Laplace results.

    Laplace results have no encoder; per-cell Laplace posteriors
    are reconstructed from the converged Newton MAP via the
    Woodbury / Sherman-Morrison machinery in
    :mod:`scribe.laplace._newton_pln` /
    :mod:`scribe.laplace._newton_lnm`. When ``counts`` is provided
    we route through ``get_per_cell_predictive_samples``, the
    *conditional* level-2 PPC: each predictive draw samples the
    per-cell composition latent from its Laplace posterior
    ``N(MAP, (-H_c)^{-1})`` *before* drawing the observation, so
    the histograms reflect both per-cell latent uncertainty and
    observation noise. When ``counts`` is None we fall back to the
    population PPC (samples cells from the prior + decoder).

    Users who specifically want the cheap MAP-only path (no
    per-cell Hessian solves) should call
    :meth:`ScribeLaplaceResults.get_map_ppc_samples` directly.
    """
    _ = batch_size
    if counts is not None:
        # Per-cell PPC at the stored MAP, with totals fixed at
        # observed library sizes (conditional PPC).
        predictive_samples = results.get_per_cell_predictive_samples(
            rng_key=rng_key,
            n_samples=int(n_samples) if n_samples is not None else 100,
            counts=counts,
        )
    else:
        predictive_samples = results.get_ppc_samples(
            rng_key=rng_key,
            n_samples=int(n_samples) if n_samples is not None else 100,
            per_cell=False,
        )
    predictive_np = np.array(predictive_samples)
    if store_samples:
        results.predictive_samples = predictive_np
    return predictive_np


@dispatch(scribe.ScribeMCMCResults)
def _get_predictive_samples_for_plot(
    results,
    *,
    rng_key,
    n_samples,
    counts,
    batch_size=None,
    store_samples=True,
):
    """Get PPC samples for plotting from MCMC results."""
    _ = counts
    _ = batch_size
    predictive_samples = results.get_ppc_samples(
        rng_key=rng_key,
        store_samples=store_samples,
    )
    predictive_np = np.array(predictive_samples)
    if n_samples is not None and predictive_np.shape[0] > int(n_samples):
        key_parts = np.asarray(rng_key, dtype=np.uint64).ravel()
        seed = int(key_parts[0] * np.uint64(2**32) + key_parts[1])
        draw_rng = np.random.default_rng(seed)
        selected_idx = draw_rng.choice(
            predictive_np.shape[0], size=int(n_samples), replace=False
        )
        predictive_np = predictive_np[selected_idx]
        if store_samples:
            results.predictive_samples = predictive_np
    return predictive_np


@dispatch(scribe.ScribeVariationalResults)
def _get_map_like_predictive_samples_for_plot(
    results,
    *,
    rng_key,
    n_samples,
    cell_batch_size,
    use_mean=True,
    store_samples=False,
    verbose=True,
    counts=None,
):
    """Generate MAP-based predictive samples for variational plotting."""
    if counts is not None:
        counts = _coerce_counts_for_sampling(
            results,
            counts,
            context="_get_map_like_predictive_samples_for_plot",
        )
    return np.array(
        results.get_map_ppc_samples(
            rng_key=rng_key,
            n_samples=n_samples,
            cell_batch_size=cell_batch_size,
            use_mean=use_mean,
            store_samples=store_samples,
            verbose=verbose,
            counts=counts,
        )
    )


@dispatch(scribe.ScribeMCMCResults)
def _get_map_like_predictive_samples_for_plot(
    results,
    *,
    rng_key,
    n_samples,
    cell_batch_size=None,
    use_mean=True,
    store_samples=False,
    verbose=True,
    counts=None,
):
    """Generate MAP-like predictive samples for MCMC plotting."""
    return np.array(
        results.get_map_ppc_samples(
            rng_key=rng_key,
            n_samples=n_samples,
            use_mean=use_mean,
            store_samples=store_samples,
            verbose=verbose,
            counts=counts,
        )
    )


@dispatch(scribe.ScribeVariationalResults)
def _get_map_estimates_for_plot(
    results, *, counts=None, use_mean=True, targets=None
):
    """Get plot-ready MAP estimates from variational results."""
    if counts is not None:
        counts = _coerce_counts_for_sampling(
            results, counts, context="_get_map_estimates_for_plot"
        )
    return results.get_map(
        targets=targets,
        use_mean=use_mean,
        canonical=True,
        verbose=False,
        counts=counts,
    )


@dispatch(scribe.ScribeMCMCResults)
def _get_map_estimates_for_plot(
    results, *, counts=None, use_mean=True, targets=None
):
    """Get plot-ready MAP estimates from MCMC results."""
    _ = counts
    _ = use_mean
    _ = targets
    return results.get_map()


# ---------------------------------------------------------------------------
# Layout metadata (AxisLayout) for plot helpers
# ---------------------------------------------------------------------------


@dispatch(scribe.ScribeVariationalResults)
def _get_layouts_for_plot(results) -> dict[str, "AxisLayout"]:
    """Get canonical MAP-level AxisLayout metadata from SVI results.

    Builds layouts keyed by canonical parameter names (``r``, ``p``,
    ``gate``, ``mixing_weights``, …) with ``has_sample_dim=False``
    so callers can query axis semantics without shape heuristics.
    """
    from ..sampling import _build_canonical_layouts

    # SVI variational params use internal names (e.g. p_alpha, p_beta);
    # the viz layer needs layouts keyed by canonical names.  Build them
    # from the canonical MAP dict, falling back to the raw variational
    # layouts when MAP extraction is unavailable (e.g. in unit tests
    # with minimal results objects).
    try:
        map_est = results.get_map(canonical=True, verbose=False)
    except (KeyError, ValueError, AttributeError):
        raw = results.layouts
        return {k: v.without_sample_dim() for k, v in raw.items()}
    return _build_canonical_layouts(
        map_est,
        results.model_config,
        n_genes=results.n_genes,
        n_cells=results.n_cells,
        n_components=getattr(results.model_config, "n_components", None),
        has_sample_dim=False,
    )


@dispatch(scribe.ScribeMCMCResults)
def _get_layouts_for_plot(results) -> dict[str, "AxisLayout"]:
    """Get MAP-level AxisLayout metadata from MCMC results.

    MCMC samples carry a leading draw axis; the MAP (posterior mean)
    does not, so we strip it for the viz layer.
    """
    raw = results.layouts
    return {k: v.without_sample_dim() for k, v in raw.items()}


@dispatch(scribe.ScribeVariationalResults)
def _get_cell_assignment_probabilities_for_plot(
    results, *, counts, batch_size=None, use_mean=False
):
    """Get MAP component-assignment probabilities from SVI mixture results."""
    counts = _coerce_counts_for_sampling(
        results,
        counts,
        context="_get_cell_assignment_probabilities_for_plot",
    )
    # Use optional batching to avoid OOM on large cell counts.
    assignment_info = results.cell_type_probabilities_map(
        counts=counts,
        batch_size=batch_size,
        use_mean=use_mean,
        verbose=False,
    )
    return np.array(assignment_info["probabilities"])


@dispatch(scribe.ScribeMCMCResults)
def _get_cell_assignment_probabilities_for_plot(
    results, *, counts, batch_size=None, use_mean=False
):
    """Get component-assignment probabilities from MCMC results."""
    _ = use_mean
    counts = _coerce_counts_for_sampling(
        results,
        counts,
        context="_get_cell_assignment_probabilities_for_plot",
    )
    # Use optional batching to avoid OOM on large cell counts.
    assignment_info = results.cell_type_probabilities(
        counts=counts, batch_size=batch_size, verbose=False
    )
    if "mean_probabilities" in assignment_info:
        return np.array(assignment_info["mean_probabilities"])
    sample_probs = np.array(assignment_info["sample_probabilities"])
    return sample_probs.mean(axis=0)


# ---------------------------------------------------------------------------
# Biological PPC samples (NB(r, p) only, no technical noise)
# ---------------------------------------------------------------------------


@dispatch(scribe.ScribeVariationalResults)
def _get_biological_ppc_samples_for_plot(
    results, *, rng_key, n_samples, counts, batch_size=None, store_samples=True
):
    """Get biological PPC samples from SVI results.

    Samples from NB(r, p) only, stripping capture probability and
    zero-inflation gate.  Follows the same save/restore pattern as
    ``_get_predictive_samples_for_plot``.
    """
    counts = _coerce_counts_for_sampling(
        results, counts, context="_get_biological_ppc_samples_for_plot"
    )
    bio_result = results.get_ppc_samples_biological(
        rng_key=rng_key,
        n_samples=n_samples,
        batch_size=batch_size,
        store_samples=store_samples,
        counts=counts,
    )
    bio_samples = np.array(bio_result["predictive_samples"])
    if store_samples:
        results.predictive_samples = bio_samples
    return bio_samples


@dispatch(scribe.ScribeMCMCResults)
def _get_biological_ppc_samples_for_plot(
    results, *, rng_key, n_samples, counts, batch_size=None, store_samples=True
):
    """Get biological PPC samples from MCMC results.

    Samples from NB(r, p) only.  Optionally subsamples to ``n_samples``
    posterior draws.
    """
    _ = counts
    bio_samples = results.get_ppc_samples_biological(
        rng_key=rng_key,
        batch_size=batch_size,
        store_samples=store_samples,
    )
    bio_np = np.array(bio_samples)
    if n_samples is not None and bio_np.shape[0] > int(n_samples):
        key_parts = np.asarray(rng_key, dtype=np.uint64).ravel()
        seed = int(key_parts[0] * np.uint64(2**32) + key_parts[1])
        draw_rng = np.random.default_rng(seed)
        idx = draw_rng.choice(
            bio_np.shape[0], size=int(n_samples), replace=False
        )
        bio_np = bio_np[idx]
    if store_samples:
        results.predictive_samples = bio_np
    return bio_np


# ---------------------------------------------------------------------------
# Denoised observed counts (single point-estimate matrix)
# ---------------------------------------------------------------------------


@dispatch(scribe.ScribeVariationalResults)
def _get_denoised_counts_for_plot(
    results, *, counts, rng_key, method=("mean", "sample"), cell_batch_size=None
):
    """MAP-denoise observed counts for SVI results.

    Returns a 2-D ``(n_cells, n_genes)`` numpy array.
    """
    counts = _coerce_counts_for_sampling(
        results, counts, context="_get_denoised_counts_for_plot"
    )
    denoised = results.denoise_counts_map(
        counts=counts,
        method=method,
        rng_key=rng_key,
        cell_batch_size=cell_batch_size,
        store_result=False,
        verbose=False,
    )
    return np.array(denoised)


@dispatch(scribe.ScribeMCMCResults)
def _get_denoised_counts_for_plot(
    results, *, counts, rng_key, method=("mean", "sample"), cell_batch_size=None
):
    """Denoise observed counts for MCMC results.

    Averages over posterior samples to produce a single 2-D
    ``(n_cells, n_genes)`` numpy array.
    """
    counts = _coerce_counts_for_sampling(
        results, counts, context="_get_denoised_counts_for_plot"
    )
    denoised = results.denoise_counts(
        counts=counts,
        method=method,
        rng_key=rng_key,
        cell_batch_size=cell_batch_size,
        store_result=False,
        verbose=False,
    )
    return np.array(denoised).mean(axis=0)


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------


@dispatch(scribe.ScribeVariationalResults)
def _get_training_diagnostic_payload(results):
    """Build training diagnostics payload for SVI loss plots."""
    return {
        "plot_kind": "loss",
        "loss_history": np.array(results.loss_history),
    }


@dispatch(scribe.ScribeMCMCResults)
def _get_training_diagnostic_payload(results):
    """Build training diagnostics payload for MCMC diagnostics plots."""
    extra_fields = results.get_extra_fields()
    potential_energy = extra_fields.get("potential_energy")
    diverging = extra_fields.get("diverging")
    payload = {
        "plot_kind": "mcmc_diagnostics",
        "potential_energy": (
            np.array(potential_energy) if potential_energy is not None else None
        ),
        "diverging": (
            np.array(diverging).astype(int) if diverging is not None else None
        ),
        "trace_by_chain": None,
        "trace_param_name": None,
    }
    try:
        grouped_samples = results.get_samples(group_by_chain=True)
        if grouped_samples:
            param_name = sorted(grouped_samples.keys())[0]
            param_samples = np.array(grouped_samples[param_name])
            if param_samples.ndim >= 2:
                if param_samples.ndim > 2:
                    reduce_axes = tuple(range(2, param_samples.ndim))
                    param_samples = param_samples.mean(axis=reduce_axes)
                payload["trace_by_chain"] = param_samples
                payload["trace_param_name"] = param_name
    except Exception:
        payload["trace_by_chain"] = None
        payload["trace_param_name"] = None
    return payload


# =====================================================================
# Laplace-result registrations
# =====================================================================
#
# The dispatch table is keyed by *result class*, so adding a new
# inference path (Laplace) requires explicit registrations of every
# helper the viz layer uses. Without these, callers like
# ``scribe.viz.plot_loss(laplace_result)`` raise
# ``NotImplementedError``. The registrations below mirror the
# Variational/MCMC versions semantically, adapted to the
# Laplace-specific shape of state on ``ScribeLaplaceResults``.


@dispatch(scribe.ScribeLaplaceResults, object)
def _get_inference_metadata_for_filenames(results, cfg):
    """Filename metadata for Laplace runs.

    Mirrors the SVI implementation but uses ``laplace_config`` /
    inference-config plumbing the user passed to ``scribe.fit``.
    Falls back to ``cfg.get("n_steps", 50000)`` when the higher-
    level config object isn't structured.
    """
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "n_steps"):
        n_steps = cfg.inference.n_steps
    else:
        n_steps = cfg.get("n_steps", 50000)
    return {
        "run_size_value": int(n_steps),
        "run_size_label": "steps",
        "run_size_token": f"{int(n_steps)}steps_laplace",
        "n_steps": int(n_steps),
    }


@dispatch(scribe.ScribeLaplaceResults)
def _get_training_diagnostic_payload(results):
    """Build training diagnostics payload for Laplace loss plots.

    Laplace produces a per-step negative-ELBO loss curve identical
    in shape to SVI's, so the payload format matches SVI's. The
    plot_loss helper consumes ``loss_history`` as a numpy array.
    """
    return {
        "plot_kind": "loss",
        "loss_history": np.array(results.losses),
    }


@dispatch(scribe.ScribeLaplaceResults)
def _get_map_estimates_for_plot(
    results, *, counts=None, use_mean=True, targets=None
):
    """Get plot-ready MAP estimates from Laplace results.

    Laplace results store a literal MAP (no posterior to take a
    mean over), so ``use_mean`` is ignored. ``targets`` and
    ``counts`` are accepted for signature parity with the
    variational dispatch but currently ignored — the Laplace
    ``get_map`` returns a fixed dict of point estimates that
    downstream callers (mean-calibration, MAP plots) handle
    directly.
    """
    _ = counts
    _ = use_mean
    _ = targets
    return results.get_map()


@dispatch(scribe.ScribeLaplaceResults)
def _get_layouts_for_plot(results) -> dict[str, "AxisLayout"]:
    """Layout metadata for Laplace results.

    Laplace results don't carry a per-cell ``layouts`` dict the way
    SVI/MCMC do — the per-cell state is typed dataclass fields
    rather than NumPyro-named samples. For viz callers that ask
    for layouts, return an empty dict; the plotting helpers we've
    wired (loss + PPC) don't actually consult layouts.

    If a future plot helper does need layouts, it should be
    extended to read shapes directly from the typed
    ``ScribeLaplaceResults`` slots.
    """
    return {}


@dispatch(scribe.ScribeLaplaceResults)
def _get_map_like_predictive_samples_for_plot(
    results,
    *,
    rng_key,
    n_samples,
    cell_batch_size=None,
    use_mean=True,
    store_samples=False,
    verbose=True,
    counts=None,
):
    """MAP-based predictive samples for Laplace plotting.

    Laplace already stores per-cell MAP estimates, so "MAP-based"
    PPC is identical to the standard per-cell PPC: sample from
    ``Multinomial(L_c, softmax(mu + W z_c*))`` (LNM) or
    ``Poisson(exp(x_c* - eta_c*))`` (PLN) using the stored MAP.
    We forward to ``get_per_cell_predictive_samples`` which
    handles both model branches via ``model_config.base_model``
    dispatch.
    """
    _ = cell_batch_size
    _ = use_mean
    _ = store_samples
    _ = verbose
    return np.array(
        results.get_per_cell_predictive_samples(
            rng_key=rng_key,
            n_samples=int(n_samples) if n_samples is not None else 100,
            counts=counts,
        )
    )

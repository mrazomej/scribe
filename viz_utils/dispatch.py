"""Dispatch helpers for SVI vs MCMC result types."""

import numpy as np
from multipledispatch import dispatch
import scribe


@dispatch(scribe.ScribeSVIResults, object)
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


@dispatch(scribe.ScribeSVIResults)
def _get_predictive_samples_for_plot(
    results, *, rng_key, n_samples, counts, batch_size=None, store_samples=True
):
    """Get PPC samples for plotting from SVI results."""
    results.get_ppc_samples(
        rng_key=rng_key,
        n_samples=n_samples,
        batch_size=batch_size,
        store_samples=store_samples,
        counts=counts,
    )
    return np.array(results.predictive_samples)


@dispatch(scribe.ScribeMCMCResults)
def _get_predictive_samples_for_plot(
    results, *, rng_key, n_samples, counts, batch_size=None, store_samples=True
):
    """Get PPC samples for plotting from MCMC results."""
    _ = counts
    predictive_samples = results.get_ppc_samples(
        rng_key=rng_key,
        batch_size=batch_size,
        store_samples=store_samples,
    )
    predictive_np = np.array(predictive_samples)
    if n_samples is not None and predictive_np.shape[0] > int(n_samples):
        predictive_np = predictive_np[: int(n_samples)]
        if store_samples:
            results.predictive_samples = predictive_np
    return predictive_np


@dispatch(scribe.ScribeSVIResults)
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
    """Generate MAP-based predictive samples for SVI plotting."""
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
    cell_batch_size,
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
            cell_batch_size=cell_batch_size,
            use_mean=use_mean,
            store_samples=store_samples,
            verbose=verbose,
            counts=counts,
        )
    )


@dispatch(scribe.ScribeSVIResults)
def _get_map_estimates_for_plot(results, *, counts=None):
    """Get plot-ready MAP estimates from SVI results."""
    return results.get_map(
        use_mean=True, canonical=True, verbose=False, counts=counts
    )


@dispatch(scribe.ScribeMCMCResults)
def _get_map_estimates_for_plot(results, *, counts=None):
    """Get plot-ready MAP estimates from MCMC results."""
    _ = counts
    return results.get_map()


@dispatch(scribe.ScribeSVIResults)
def _get_cell_assignment_probabilities_for_plot(results, *, counts):
    """Get MAP component-assignment probabilities from SVI results."""
    assignment_info = results.cell_type_probabilities_map(
        counts=counts, verbose=False
    )
    return np.array(assignment_info["probabilities"])


@dispatch(scribe.ScribeMCMCResults)
def _get_cell_assignment_probabilities_for_plot(results, *, counts):
    """Get component-assignment probabilities from MCMC results."""
    assignment_info = results.cell_type_probabilities(
        counts=counts, verbose=False
    )
    if "mean_probabilities" in assignment_info:
        return np.array(assignment_info["mean_probabilities"])
    sample_probs = np.array(assignment_info["sample_probabilities"])
    return sample_probs.mean(axis=0)


@dispatch(scribe.ScribeSVIResults)
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
        "diverging": np.array(diverging).astype(int)
        if diverging is not None
        else None,
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

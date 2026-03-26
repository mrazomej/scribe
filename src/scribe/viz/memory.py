"""Memory utilities for long-running visualization pipelines.

This module centralizes best-effort memory cleanup logic that is useful when
rendering multiple heavy plots in sequence (for example PPC, biological PPC,
UMAP, and heatmap).  The helper intentionally avoids raising cleanup errors so
plot generation failures are not masked by cleanup issues.
"""

from __future__ import annotations

import gc
from typing import Any

import matplotlib.pyplot as plt


def cleanup_plot_memory(
    results: Any | None = None, *, reset_result_caches: bool = True
) -> None:
    """Release host/GPU memory after a plotting step.

    Parameters
    ----------
    results : object or None, optional
        Results object used for plotting. When provided and
        ``reset_result_caches=True``, large cached sample attributes are reset
        to ``None`` if present.
    reset_result_caches : bool, optional
        Whether to clear large cache-like attributes on ``results``.

    Notes
    -----
    This function is intentionally best-effort:

    - it closes all Matplotlib figures,
    - clears common sample caches on the results object,
    - runs Python garbage collection, and
    - clears JAX compilation/runtime caches when JAX is available.

    The function suppresses cleanup-specific exceptions to avoid interrupting
    visualization execution.
    """
    # Ensure figure objects and their buffers are released promptly.
    plt.close("all")

    if results is not None and reset_result_caches:
        # Clear large arrays that are often populated by PPC helpers.
        cache_attrs = (
            "predictive_samples",
            "predictive_samples_biological",
            "posterior_samples",
            "denoised_counts",
        )
        for cache_attr in cache_attrs:
            if hasattr(results, cache_attr):
                try:
                    setattr(results, cache_attr, None)
                except Exception:
                    # Some attributes can be read-only depending on result type.
                    pass

    # Trigger Python GC after dropping references.
    gc.collect()

    # Ask JAX to clear any compilations/cache state if available.
    try:
        import jax

        jax.clear_caches()
    except Exception:
        # JAX may be unavailable in minimal environments.
        pass

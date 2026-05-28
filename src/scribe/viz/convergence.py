"""Per-cell Newton convergence diagnostics for Laplace fits.

Produces a two-panel figure that surfaces which cells did NOT converge
in the inner Newton loop during a Laplace inference run.  Reads
``results.final_grad_norms`` (per-cell L-infinity gradient norm at the
final sweep) and ``results._newton_tolerance`` (the threshold used by
the fit).  Cells with ``final_grad_norm > newton_tolerance`` are
flagged as unconverged; the diagnostic helps decide whether the
problem is a small tail of pathological cells (likely doublets / dead
cells / unusual count profiles) versus a broad mis-fit suggesting the
fit needs more Newton iterations.

When the optional rescue pass ran (i.e.
``results._rescued_cell_mask is not None``), points are color-coded
by their rescue outcome:

- ``converged`` — never diverged (passed Newton on the main fit).
- ``rescued`` — diverged on main fit, then converged after the rescue
  pass.
- ``model_unfit`` — diverged on main fit AND still diverged after
  rescue (the model genuinely cannot fit these cells).

Without rescue, points are simply ``converged`` vs ``unconverged``.

Panels
------
1. Histogram of ``log10(final_grad_norm + ε)`` with vertical lines at
   ``newton_tolerance`` and (when set) the rescue threshold.  Counts
   in each bucket are annotated.
2. Scatter of per-cell ``log10(library_size)`` against ``log10(final_grad_norm)``,
   colored by convergence bucket.  This surfaces whether unconverged
   cells are unusual in their count statistics — the most common
   cause in production single-cell data.
"""

import matplotlib.pyplot as plt
import numpy as np

from ._common import console
from ._interactive import (
    _create_or_validate_grid_axes,
    plot_function,
)
from .gene_selection import _coerce_counts


# Default tolerance used when ``results._newton_tolerance`` is missing
# (older pickles).  Matches ``LaplaceConfig.newton_tolerance``'s default.
_DEFAULT_NEWTON_TOLERANCE = 1e-4

# Default rescue threshold multiplier (matches
# ``LaplaceConfig.rescue_threshold_multiplier`` default).  Used only
# for the secondary vertical reference line on panel 1 when rescue
# did not run; when rescue ran the actual threshold is implicit in
# ``_rescued_cell_mask``.
_DEFAULT_RESCUE_THRESHOLD_MULTIPLIER = 10.0


def _prepare_convergence_data(results, counts, viz_cfg=None):
    """Extract the per-cell tensors needed for the convergence diagnostic.

    Returns ``None`` when ``final_grad_norms`` is unavailable (non-Laplace
    fit, or an older pickle that pre-dates the diagnostic).
    """
    grad_norms = getattr(results, "final_grad_norms", None)
    if grad_norms is None:
        console.print(
            "[yellow]Skipping convergence diagnostic: "
            "``final_grad_norms`` unavailable on this result.[/yellow]"
        )
        return None

    grad_arr = np.asarray(grad_norms, dtype=float).reshape(-1)
    if grad_arr.size == 0:
        return None

    # Threshold used by the fit itself (or the conservative default
    # when the pickle predates ``_newton_tolerance``).
    tol = getattr(results, "_newton_tolerance", None)
    if tol is None:
        tol = _DEFAULT_NEWTON_TOLERANCE
    tol = float(tol)

    # Rescue metadata.  ``rescued_mask`` is True for cells the rescue
    # pass touched; ``pre_rescue_grad_norms`` (when present) is the
    # tensor BEFORE rescue ran.  When rescue did not run, both are
    # ``None`` and the diagnostic shows a single "unconverged" bucket.
    rescued_mask = getattr(results, "_rescued_cell_mask", None)
    pre_rescue = getattr(results, "_pre_rescue_grad_norms", None)
    if rescued_mask is not None:
        rescued_mask = np.asarray(rescued_mask, dtype=bool).reshape(-1)
    if pre_rescue is not None:
        pre_rescue = np.asarray(pre_rescue, dtype=float).reshape(-1)

    # Library size per cell from the count matrix.  Coerce in case
    # counts arrives as an AnnData / sparse matrix.
    counts_arr = _coerce_counts(counts)
    if counts_arr.ndim != 2:
        lib = None
    else:
        lib = counts_arr.sum(axis=1)
        # Defensive: trim to length(grad_arr) if the counts were filtered
        # against a different cell axis (shouldn't happen for Laplace,
        # but guards against future axis-confusion bugs).
        if int(lib.shape[0]) != int(grad_arr.shape[0]):
            lib = None

    # Convergence buckets: 0=converged, 1=rescued, 2=model_unfit, 3=unconverged_no_rescue
    bucket = np.zeros(grad_arr.shape[0], dtype=np.int8)
    if rescued_mask is not None:
        # Rescue ran.  Cells that started above threshold AND ended
        # above threshold are "model_unfit"; cells that started above
        # threshold and ended below are "rescued"; everything else
        # is "converged".
        post_diverged = grad_arr > tol
        bucket = np.where(
            rescued_mask & post_diverged, 2, np.where(rescued_mask, 1, 0)
        ).astype(np.int8)
    else:
        # No rescue: simple converged/unconverged split.
        bucket = np.where(grad_arr > tol, 3, 0).astype(np.int8)

    return {
        "grad_arr": grad_arr,
        "tol": tol,
        "rescued_mask": rescued_mask,
        "pre_rescue": pre_rescue,
        "lib": lib,
        "bucket": bucket,
    }


def _bucket_colors_and_labels(buckets, rescued_present):
    """Return color/label arrays for the 4 convergence buckets."""
    # Use a tab10-inspired palette: green for OK, orange for rescued,
    # red for model-unfit, gray for unconverged-no-rescue.
    palette = {
        0: ("#2ca02c", "converged"),
        1: ("#ff7f0e", "rescued"),
        2: ("#d62728", "model_unfit"),
        3: ("#7f7f7f", "unconverged"),
    }
    if rescued_present:
        # Show all three meaningful buckets even if some are empty.
        bucket_keys = [0, 1, 2]
    else:
        bucket_keys = [0, 3]
    return palette, bucket_keys


@plot_function(
    suffix="convergence",
    save_label="convergence plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_convergence(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Per-cell inner-Newton convergence diagnostic for Laplace fits.

    Produces a two-panel figure:

    1. Histogram of per-cell ``log10(final_grad_norm + ε)`` with vertical
       reference lines at ``newton_tolerance`` and the rescue threshold.
    2. Scatter of ``log10(library_size)`` vs ``log10(final_grad_norm)``
       colored by convergence bucket.  Surfaces whether unconverged
       cells correlate with library-size extremes.

    Returns ``None`` when ``final_grad_norms`` is unavailable (e.g.
    non-Laplace fits or older pickles).

    Parameters
    ----------
    results : ScribeLaplaceResults
        Fitted Laplace result.  Must expose ``final_grad_norms``.
    counts : array-like
        Observed UMI count matrix ``(n_cells, n_genes)``.  Used to
        compute per-cell library size for panel 2.
    """
    console.print(
        "[dim]Plotting per-cell Newton convergence diagnostic...[/dim]"
    )
    if ax is not None:
        raise ValueError(
            "Convergence diagnostic uses 2 panels; provide `fig` or "
            "2 `axes`, not `ax`."
        )

    data = _prepare_convergence_data(results, counts, viz_cfg)
    if data is None:
        return None

    grad_arr = data["grad_arr"]
    tol = data["tol"]
    rescued_mask = data["rescued_mask"]
    lib = data["lib"]
    bucket = data["bucket"]

    fig, _, flat_axes = _create_or_validate_grid_axes(
        n_rows=1,
        n_cols=2,
        fig=fig,
        axes=axes,
        figsize=figsize or (12.0, 4.5),
    )
    ax1, ax2 = flat_axes

    # ---- Panel 1: histogram of log10(grad_norm) -----------------------------
    # Small floor avoids log10(0) for cells that converged perfectly.
    eps = 1e-30
    log_grad = np.log10(np.maximum(grad_arr, eps))

    palette, bucket_keys = _bucket_colors_and_labels(
        bucket, rescued_mask is not None
    )

    n_bins = 60
    bin_edges = np.linspace(log_grad.min(), log_grad.max(), n_bins + 1)
    # Stack per-bucket histograms.
    for k in bucket_keys:
        mask_k = bucket == k
        if mask_k.sum() == 0:
            continue
        color, label = palette[k]
        n_k = int(mask_k.sum())
        ax1.hist(
            log_grad[mask_k],
            bins=bin_edges,
            color=color,
            alpha=0.7,
            label=f"{label} ({n_k})",
            edgecolor="none",
        )

    ax1.axvline(
        np.log10(tol),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"newton_tolerance = {tol:.0e}",
    )
    # Show the rescue threshold even when rescue didn't run (it's
    # what `rescue_diverged_cells=True` *would* use).
    rescue_thr = tol * _DEFAULT_RESCUE_THRESHOLD_MULTIPLIER
    ax1.axvline(
        np.log10(rescue_thr),
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"rescue threshold = {rescue_thr:.0e}",
    )
    ax1.set_xlabel(
        r"$\log_{10}(\|\nabla_{\mathrm{inner}}\|_{\infty} + \varepsilon)$"
    )
    ax1.set_ylabel("cell count")
    ax1.set_title("Per-cell final Newton grad norm")
    ax1.legend(fontsize=8, loc="best")

    # ---- Panel 2: scatter (library size, grad norm) -------------------------
    if lib is None:
        ax2.text(
            0.5,
            0.5,
            "library size unavailable\n(counts shape mismatch)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_axis_off()
    else:
        log_lib = np.log10(np.maximum(lib, 1.0))
        # Plot converged cells first so unconverged points sit on top.
        for k in bucket_keys:
            mask_k = bucket == k
            if mask_k.sum() == 0:
                continue
            color, label = palette[k]
            ax2.scatter(
                log_lib[mask_k],
                log_grad[mask_k],
                s=10,
                color=color,
                alpha=0.5 if k == 0 else 0.85,
                edgecolor="none",
                label=label,
            )
        ax2.axhline(
            np.log10(tol),
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax2.set_xlabel(r"$\log_{10}$ library size")
        ax2.set_ylabel(
            r"$\log_{10}\|\nabla_{\mathrm{inner}}\|_{\infty}$"
        )
        ax2.set_title("Convergence vs library size")
        ax2.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig, flat_axes, 2

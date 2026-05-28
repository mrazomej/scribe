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

    # Library size + n_genes_expressed per cell from the count matrix.
    # Coerce in case counts arrives as an AnnData / sparse matrix.
    counts_arr = _coerce_counts(counts)
    lib = None
    n_genes_expr = None
    if counts_arr.ndim == 2 and int(counts_arr.shape[0]) == int(
        grad_arr.shape[0]
    ):
        lib = counts_arr.sum(axis=1)
        # n_genes_expressed is the per-cell count of genes with at
        # least one UMI — the standard scRNA-seq QC metric.
        n_genes_expr = (counts_arr > 0).sum(axis=1)

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
        "n_genes_expr": n_genes_expr,
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

    Produces a 2x2 panel grid:

    1. **Histogram** of per-cell ``log10(final_grad_norm + ε)`` with
       vertical reference lines at ``newton_tolerance`` and the rescue
       threshold.
    2. **Scatter (library size)**: ``log10(library_size)`` vs
       ``log10(final_grad_norm)`` colored by convergence bucket —
       surfaces whether unconverged cells correlate with extreme
       library sizes.
    3. **Scatter (n genes measured)**: ``log10(n genes measured)`` vs
       ``log10(final_grad_norm)``.  Same role as panel 2 but with the
       gene-detection axis.  Often more diagnostic for doublets and
       stressed cells, which show unusual gene-detection counts
       independent of total UMI.
    4. **QC scatter**: ``log10(library_size)`` vs
       ``log10(n genes measured)`` colored by convergence bucket.
       The classic scRNA-seq QC overlay — places unconverged cells in
       the standard QC space so you can see whether they're outliers
       in the count-vs-detection plane (the typical home of doublets
       and dying cells).

    Returns ``None`` when ``final_grad_norms`` is unavailable (e.g.
    non-Laplace fits or older pickles).

    Parameters
    ----------
    results : ScribeLaplaceResults
        Fitted Laplace result.  Must expose ``final_grad_norms``.
    counts : array-like
        Observed UMI count matrix ``(n_cells, n_genes)``.  Used to
        compute per-cell library size + ``n genes measured`` for
        panels 2-4.
    """
    console.print(
        "[dim]Plotting per-cell Newton convergence diagnostic...[/dim]"
    )
    if ax is not None:
        raise ValueError(
            "Convergence diagnostic uses 4 panels; provide `fig` or "
            "4 `axes`, not `ax`."
        )

    data = _prepare_convergence_data(results, counts, viz_cfg)
    if data is None:
        return None

    grad_arr = data["grad_arr"]
    tol = data["tol"]
    rescued_mask = data["rescued_mask"]
    lib = data["lib"]
    n_genes_expr = data["n_genes_expr"]
    bucket = data["bucket"]

    fig, _, flat_axes = _create_or_validate_grid_axes(
        n_rows=2,
        n_cols=2,
        fig=fig,
        axes=axes,
        figsize=figsize or (12.0, 9.0),
    )
    ax1, ax2, ax3, ax4 = flat_axes

    # Common: log10 grad norms with a small floor so cells that
    # converged perfectly don't trip log10(0).
    eps = 1e-30
    log_grad = np.log10(np.maximum(grad_arr, eps))

    palette, bucket_keys = _bucket_colors_and_labels(
        bucket, rescued_mask is not None
    )
    rescue_thr = tol * _DEFAULT_RESCUE_THRESHOLD_MULTIPLIER

    def _scatter_by_bucket(ax_, x, y):
        """Helper: scatter points coloured by convergence bucket."""
        for k in bucket_keys:
            mask_k = bucket == k
            if mask_k.sum() == 0:
                continue
            color, label = palette[k]
            ax_.scatter(
                x[mask_k],
                y[mask_k],
                s=10,
                color=color,
                alpha=0.5 if k == 0 else 0.85,
                edgecolor="none",
                label=label,
            )

    # ---- Panel 1: histogram of log10(grad_norm) ---------------------------
    n_bins = 60
    bin_edges = np.linspace(log_grad.min(), log_grad.max(), n_bins + 1)
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

    # ---- Panel 2: scatter (library size, grad norm) -----------------------
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
        _scatter_by_bucket(ax2, log_lib, log_grad)
        ax2.axhline(
            np.log10(tol),
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax2.set_xlabel(r"$\log_{10}$ library size")
        ax2.set_ylabel(r"$\log_{10}\|\nabla_{\mathrm{inner}}\|_{\infty}$")
        ax2.set_title("Convergence vs library size")
        ax2.legend(fontsize=8, loc="best")

    # ---- Panel 3: scatter (n_genes_expressed, grad norm) ------------------
    if n_genes_expr is None:
        ax3.text(
            0.5,
            0.5,
            "n genes measured unavailable\n(counts shape mismatch)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_axis_off()
    else:
        log_ng = np.log10(np.maximum(n_genes_expr, 1.0))
        _scatter_by_bucket(ax3, log_ng, log_grad)
        ax3.axhline(
            np.log10(tol),
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax3.set_xlabel(r"$\log_{10}$ n genes measured")
        ax3.set_ylabel(r"$\log_{10}\|\nabla_{\mathrm{inner}}\|_{\infty}$")
        ax3.set_title("Convergence vs n genes measured")
        ax3.legend(fontsize=8, loc="best")

    # ---- Panel 4: QC scatter (library size, n_genes_expressed) ------------
    if lib is None or n_genes_expr is None:
        ax4.text(
            0.5,
            0.5,
            "QC scatter unavailable\n(counts shape mismatch)",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_axis_off()
    else:
        log_lib = np.log10(np.maximum(lib, 1.0))
        log_ng = np.log10(np.maximum(n_genes_expr, 1.0))
        _scatter_by_bucket(ax4, log_lib, log_ng)
        ax4.set_xlabel(r"$\log_{10}$ library size")
        ax4.set_ylabel(r"$\log_{10}$ n genes measured")
        ax4.set_title("QC scatter (cells colored by convergence)")
        ax4.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig, flat_axes, 4

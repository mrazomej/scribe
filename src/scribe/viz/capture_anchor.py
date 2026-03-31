r"""Capture-anchor diagnostic plotting utilities.

This module implements the eta-based capture anchor diagnostic used to
validate biology-informed capture priors. The diagnostic checks whether
the learned per-cell capture latent follows the expected relationship:

.. math::

    \eta_c + \log(L_c) \approx \log(M_0),

where :math:`L_c` is the observed library size and :math:`M_0` is the
expected total mRNA molecules from the prior.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from scribe.models.config.organism_priors import resolve_organism_priors

from ._common import console
from ._interactive import (
    _create_or_validate_grid_axes,
    plot_function,
)
from .dispatch import (
    _get_map_estimates_for_plot,
    _get_cell_assignment_probabilities_for_plot,
)


def _resolve_expected_log_m0(cfg):
    """Resolve expected ``log(M_0)`` from Hydra configuration.

    Parameters
    ----------
    cfg : OmegaConf
        Run configuration loaded from ``.hydra/config.yaml``.

    Returns
    -------
    float or None
        Expected ``log(M_0)``. Returns ``None`` when the configuration does
        not provide sufficient information to derive the anchor.
    """
    # Prefer explicit eta prior values when available in the run config.
    priors_cfg = cfg.get("priors") if hasattr(cfg, "get") else None
    if priors_cfg is not None and hasattr(priors_cfg, "get"):
        eta_capture = priors_cfg.get("eta_capture")
        if eta_capture is not None and len(eta_capture) >= 1:
            return float(eta_capture[0])

        # Fall back to organism-resolved defaults when explicit eta prior is
        # absent but organism shortcut was used.
        organism = priors_cfg.get("organism")
        if organism is not None:
            try:
                organism_prior = resolve_organism_priors(str(organism))
                return float(np.log(organism_prior["total_mrna_mean"]))
            except Exception as exc:
                console.print(
                    "[yellow]Could not resolve organism prior for capture-anchor "
                    f"diagnostic:[/yellow] {exc}"
                )
                return None

    return None


def _prepare_capture_anchor_data(results, counts, cfg, viz_cfg):
    """Prepare data for capture-anchor diagnostic.

    Resolves the expected anchor, extracts MAP eta values, and computes
    derived quantities for the diagnostic panels.

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : array-like
        Observed UMI count matrix ``(n_cells, n_genes)``.
    cfg : object
        Run configuration.
    viz_cfg : object
        Visualization configuration.

    Returns
    -------
    dict or None
        Dictionary with keys ``expected_log_m0``, ``eta_capture``,
        ``log_library_size``, ``eta_plus_log_lib``, ``n_bins``,
        ``scatter_size``, ``scatter_alpha``.  Returns ``None`` when
        the anchor or eta values are unavailable.
    """
    expected_log_m0 = _resolve_expected_log_m0(cfg)
    if expected_log_m0 is None:
        console.print(
            "[yellow]Skipping capture-anchor plot: could not infer "
            r"$\log(M_0)$ from priors.eta_capture or priors.organism.[/yellow]"
        )
        return None

    map_estimates = _get_map_estimates_for_plot(
        results, counts=counts, targets=["eta_capture"]
    )
    eta_capture = map_estimates.get("eta_capture")
    if eta_capture is None:
        console.print(
            "[yellow]Skipping capture-anchor plot: eta_capture is unavailable "
            "in MAP estimates.[/yellow]"
        )
        return None
    eta_capture = np.asarray(eta_capture, dtype=float).reshape(-1)

    library_size = np.asarray(counts.sum(axis=1), dtype=float).reshape(-1)
    log_library_size = np.log(np.maximum(library_size, 1.0))
    eta_plus_log_lib = eta_capture + log_library_size

    opts = viz_cfg.get("capture_anchor_opts", {})
    n_bins = int(opts.get("n_bins", 50))
    scatter_size = float(opts.get("scatter_size", 6.0))
    scatter_alpha = float(opts.get("scatter_alpha", 0.35))

    return {
        "expected_log_m0": expected_log_m0,
        "eta_capture": eta_capture,
        "log_library_size": log_library_size,
        "eta_plus_log_lib": eta_plus_log_lib,
        "n_bins": n_bins,
        "scatter_size": scatter_size,
        "scatter_alpha": scatter_alpha,
    }


@plot_function(
    suffix="capture_anchor",
    save_label="capture-anchor plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_capture_anchor(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    fig=None,
    axes=None,
    ax=None,
):
    r"""Plot and save eta capture-anchor diagnostics.

    The figure contains two panels:

    1. Scatter of :math:`\eta_c` versus :math:`\log(L_c)` with expected
       anchor line :math:`\eta_c = \log(M_0) - \log(L_c)`.
    2. Distribution of :math:`\eta_c + \log(L_c)` with reference at
       :math:`\log(M_0)`.

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Fitted model results object for the current dataset view.
    counts : array-like
        Observed UMI count matrix with shape ``(n_cells, n_genes)``.
    figs_dir : str
        Output directory where the figure will be written.
    cfg : OmegaConf
        Hydra configuration loaded from the run directory.
    viz_cfg : OmegaConf
        Visualization configuration with optional ``capture_anchor_opts``:
        ``n_bins`` (int), ``scatter_size`` (float), and ``scatter_alpha``
        (float).

    Returns
    -------
    PlotResult or None
        Wrapped result containing the figure, axes, and metadata, or
        ``None`` when required inputs for the diagnostic are unavailable.
    """
    console.print("[dim]Plotting capture-anchor diagnostic...[/dim]")
    if ax is not None:
        raise ValueError(
            "Capture-anchor uses 2 panels; provide `fig` or 2 `axes`."
        )
    data = _prepare_capture_anchor_data(results, counts, ctx.cfg, viz_cfg)
    if data is None:
        return None

    expected_log_m0 = data["expected_log_m0"]
    eta_capture = data["eta_capture"]
    log_library_size = data["log_library_size"]
    eta_plus_log_lib = data["eta_plus_log_lib"]
    n_bins = data["n_bins"]
    scatter_size = data["scatter_size"]
    scatter_alpha = data["scatter_alpha"]

    # Create side-by-side diagnostic figure.
    fig, _, flat_axes = _create_or_validate_grid_axes(
        n_rows=1,
        n_cols=2,
        fig=fig,
        axes=axes,
        figsize=(12.0, 4.5),
    )
    ax1, ax2 = flat_axes

    ax1.scatter(
        log_library_size,
        eta_capture,
        s=scatter_size,
        alpha=scatter_alpha,
    )
    sorted_log_lib = np.sort(log_library_size)
    ax1.plot(
        sorted_log_lib,
        expected_log_m0 - sorted_log_lib,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=r"$\eta_c = \log(M_0) - \log(L_c)$",
    )
    ax1.set_xlabel(r"$\log(L_c)$")
    ax1.set_ylabel(r"$\hat{\eta}_c^{\mathrm{MAP}}$")
    ax1.set_title(r"Capture-anchor: $\eta_c$ vs $\log(L_c)$")
    ax1.legend(fontsize=8)

    ax2.hist(eta_plus_log_lib, bins=n_bins, density=True, color="steelblue")
    ax2.axvline(expected_log_m0, color="black", linestyle="--", linewidth=1.5)
    ax2.set_xlabel(r"$\eta_c + \log(L_c)$")
    ax2.set_title(r"Distribution of $\eta_c + \log(L_c)$")

    fig.tight_layout()

    return fig, flat_axes, 2


def _compute_binned_trend(x, y, n_bins=30, min_cells_per_bin=5):
    """Compute robust binned trend statistics for plotting.

    Parameters
    ----------
    x : ndarray
        Independent variable values (library size).
    y : ndarray
        Dependent variable values (capture probability).
    n_bins : int, optional
        Number of quantile bins used to aggregate points.
    min_cells_per_bin : int, optional
        Minimum number of cells required for a bin to contribute to the
        returned trend.

    Returns
    -------
    tuple of ndarray
        Pair ``(x_center, y_center)`` with median bin centers and median
        response values. Empty arrays are returned when no stable bins can be
        formed.
    """
    # Ensure finite pairs only so trend statistics are numerically stable.
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_valid = np.asarray(x[finite_mask], dtype=float)
    y_valid = np.asarray(y[finite_mask], dtype=float)
    if x_valid.size == 0:
        return np.array([]), np.array([])

    # Quantile bins are robust to long-tail library-size distributions.
    q = np.linspace(0.0, 1.0, int(max(n_bins, 2)) + 1)
    bin_edges = np.unique(np.quantile(x_valid, q))
    if bin_edges.size < 2:
        return np.array([]), np.array([])

    x_centers = []
    y_centers = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right <= left:
            continue
        in_bin = (x_valid >= left) & (x_valid < right)
        if np.count_nonzero(in_bin) < int(min_cells_per_bin):
            continue
        x_centers.append(float(np.median(x_valid[in_bin])))
        y_centers.append(float(np.median(y_valid[in_bin])))

    return np.asarray(x_centers), np.asarray(y_centers)


def _plot_trend_line(ax, x, y, label, color, n_bins, min_cells_per_bin):
    """Plot one binned trend curve for ``p_capture`` scaling diagnostics."""
    # Draw a light point cloud first to preserve raw-data context.
    ax.scatter(
        x,
        y,
        s=4.0,
        alpha=0.08,
        color=color,
        linewidths=0.0,
    )

    # Overlay the robust trend summary used for comparison across groups.
    trend_x, trend_y = _compute_binned_trend(
        x,
        y,
        n_bins=n_bins,
        min_cells_per_bin=min_cells_per_bin,
    )
    if trend_x.size > 1:
        ax.plot(trend_x, trend_y, linewidth=2.0, color=color, label=label)


def _set_dynamic_y_limits(ax, y_values, pad_fraction=0.15):
    """Set data-driven y-axis limits for capture-probability panels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis whose limits will be updated.
    y_values : ndarray
        Capture-probability values used to infer display range.
    pad_fraction : float, optional
        Relative padding added around the data range.
    """
    # Keep only finite values to avoid propagating NaN/inf into axis bounds.
    y_valid = np.asarray(y_values, dtype=float)
    y_valid = y_valid[np.isfinite(y_valid)]
    if y_valid.size == 0:
        return

    y_min = float(np.min(y_valid))
    y_max = float(np.max(y_valid))
    y_range = y_max - y_min

    # Expand narrow ranges so small p_capture variation remains visible.
    if y_range <= 1e-6:
        pad = 0.02
    else:
        pad = max(1e-3, y_range * float(pad_fraction))

    lower = max(0.0, y_min - pad)
    upper = min(1.0, y_max + pad)

    # If clipping to [0,1] collapses the interval, widen slightly.
    if upper - lower <= 1e-5:
        lower = max(0.0, y_min - 0.02)
        upper = min(1.0, y_max + 0.02)

    ax.set_ylim(lower, upper)


def _prepare_p_capture_data(
    results,
    counts,
    viz_cfg,
    *,
    is_mixture=False,
    is_multi_dataset=False,
    dataset_codes=None,
    dataset_names=None,
):
    """Prepare data for p-capture scaling diagnostic.

    Extracts MAP capture probabilities, computes library sizes,
    and resolves component/dataset splits.

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : array-like
        Observed count matrix.
    viz_cfg : object
        Visualization configuration.
    is_mixture : bool
        Whether the model uses more than one component.
    is_multi_dataset : bool
        Whether the run includes multiple datasets.
    dataset_codes : ndarray or None
        Integer dataset code per cell.
    dataset_names : sequence of str or None
        Category names for dataset codes.

    Returns
    -------
    dict or None
        Dictionary with keys ``p_capture``, ``library_size``,
        ``n_bins``, ``min_cells_per_bin``, ``panel_specs``,
        and optionally ``component_ids``, ``component_probs``.
        Returns ``None`` when p_capture is unavailable.
    """
    map_estimates = _get_map_estimates_for_plot(
        results, counts=counts, targets=["p_capture"]
    )
    p_capture = map_estimates.get("p_capture")
    if p_capture is None:
        console.print(
            "[yellow]Skipping p-capture scaling: p_capture is unavailable "
            "in MAP estimates.[/yellow]"
        )
        return None
    p_capture = np.asarray(p_capture, dtype=float).reshape(-1)

    library_size = np.asarray(counts.sum(axis=1), dtype=float).reshape(-1)

    opts = viz_cfg.get("p_capture_scaling_opts", {})
    n_bins = int(opts.get("n_bins", 30))
    min_cells_per_bin = int(opts.get("min_cells_per_bin", 5))
    assignment_batch_size = int(opts.get("assignment_batch_size", 512))

    panel_specs = [("global", None)]
    if is_mixture:
        panel_specs.append(("component", None))
    if is_multi_dataset:
        panel_specs.append(("dataset", None))

    out = {
        "p_capture": p_capture,
        "library_size": library_size,
        "n_bins": n_bins,
        "min_cells_per_bin": min_cells_per_bin,
        "panel_specs": panel_specs,
    }

    if is_mixture:
        if (
            is_multi_dataset
            and dataset_codes is not None
            and hasattr(results, "get_dataset")
        ):
            _ds_codes = np.asarray(dataset_codes, dtype=int).reshape(-1)
            _unique_ds = np.unique(_ds_codes)
            _per_ds_probs = []
            for _d in _unique_ds:
                _mask = _ds_codes == _d
                _ds_results = results.get_dataset(int(_d))
                _ds_probs = _get_cell_assignment_probabilities_for_plot(
                    _ds_results,
                    counts=counts[_mask],
                    batch_size=assignment_batch_size,
                    use_mean=False,
                )
                _per_ds_probs.append(np.asarray(_ds_probs))
            component_probs = np.concatenate(_per_ds_probs, axis=0)
        else:
            component_probs = _get_cell_assignment_probabilities_for_plot(
                results,
                counts=counts,
                batch_size=assignment_batch_size,
                use_mean=False,
            )
        component_ids = np.argmax(np.asarray(component_probs), axis=1)
        out["component_ids"] = component_ids
        out["component_probs"] = component_probs

    return out


@plot_function(
    suffix="p_capture_scaling",
    save_label="p-capture scaling plot",
    save_kwargs={"bbox_inches": "tight"},
)
def plot_p_capture_scaling(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    is_mixture=False,
    is_multi_dataset=False,
    dataset_codes=None,
    dataset_names=None,
    fig=None,
    axes=None,
    ax=None,
):
    r"""Plot ``p_capture`` versus library-size scaling diagnostics.

    This diagnostic is intended for VCP models and is independent of
    eta-parameterization. It includes:

    1. Global trend over all cells.
    2. Optional split by mixture component (MAP assignment).
    3. Optional split by dataset when multiple datasets are present.

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Fitted model results object.
    counts : array-like
        Observed UMI count matrix ``(n_cells, n_genes)``.
    figs_dir : str
        Directory where output figure is saved.
    cfg : OmegaConf
        Hydra run configuration loaded from ``.hydra/config.yaml``.
    viz_cfg : OmegaConf
        Visualization config. Expected to include ``p_capture_scaling_opts``.
    is_mixture : bool, optional
        Whether the model uses more than one component.
    is_multi_dataset : bool, optional
        Whether the run includes multiple datasets.
    dataset_codes : ndarray, optional
        Integer dataset code per cell (same ordering as ``counts`` rows).
    dataset_names : sequence of str, optional
        Category names corresponding to ``dataset_codes``.

    Returns
    -------
    PlotResult or None
        Wrapped result on success, or ``None`` when ``p_capture`` is
        unavailable.
    """
    console.print("[dim]Plotting p-capture scaling diagnostic...[/dim]")
    if ax is not None:
        raise ValueError(
            "p-capture scaling may use multiple panels; provide `fig` or `axes`."
        )
    data = _prepare_p_capture_data(
        results,
        counts,
        viz_cfg,
        is_mixture=is_mixture,
        is_multi_dataset=is_multi_dataset,
        dataset_codes=dataset_codes,
        dataset_names=dataset_names,
    )
    if data is None:
        return None

    p_capture = data["p_capture"]
    library_size = data["library_size"]
    n_bins = data["n_bins"]
    min_cells_per_bin = data["min_cells_per_bin"]
    panel_specs = data["panel_specs"]

    fig, _, axes_flat = _create_or_validate_grid_axes(
        n_rows=1,
        n_cols=len(panel_specs),
        fig=fig,
        axes=axes,
        figsize=(6.0 * len(panel_specs), 5.0),
    )

    # Global panel: one trend using all cells.
    ax_global = axes_flat[0]
    _plot_trend_line(
        ax_global,
        library_size,
        p_capture,
        label=r"all cells",
        color="black",
        n_bins=n_bins,
        min_cells_per_bin=min_cells_per_bin,
    )
    ax_global.set_xlabel(r"$L_c$")
    ax_global.set_ylabel(r"$\hat{p}_{\mathrm{capture},c}^{\mathrm{MAP}}$")
    ax_global.set_title(r"Global $p_{\mathrm{capture}}$ scaling")
    _set_dynamic_y_limits(ax_global, p_capture)
    ax_global.legend(fontsize=8)

    panel_idx = 1

    # Component panel: derive hard MAP assignments from assignment probabilities.
    if is_mixture:
        component_ids = data["component_ids"]
        unique_components = np.unique(component_ids)
        colors = plt.cm.tab10(np.linspace(0, 1, unique_components.size))
        # Track values that were actually plotted to derive faithful y-limits.
        component_values = []

        ax_comp = axes_flat[panel_idx]
        for comp_color, comp_id in zip(colors, unique_components):
            in_comp = component_ids == comp_id
            if np.count_nonzero(in_comp) < max(min_cells_per_bin, 10):
                continue
            _plot_trend_line(
                ax_comp,
                library_size[in_comp],
                p_capture[in_comp],
                label=rf"$k={int(comp_id)}$",
                color=comp_color,
                n_bins=n_bins,
                min_cells_per_bin=min_cells_per_bin,
            )
            component_values.append(p_capture[in_comp])
        ax_comp.set_xlabel(r"$L_c$")
        ax_comp.set_ylabel(r"$\hat{p}_{\mathrm{capture},c}^{\mathrm{MAP}}$")
        ax_comp.set_title(r"Split by component")
        if component_values:
            _set_dynamic_y_limits(ax_comp, np.concatenate(component_values))
        else:
            _set_dynamic_y_limits(ax_comp, p_capture)
        handles, labels = ax_comp.get_legend_handles_labels()
        if handles:
            ax_comp.legend(fontsize=8)
        panel_idx += 1

    # Dataset panel: use dataset key coding prepared in packaged viz pipeline.
    if (
        is_multi_dataset
        and dataset_codes is not None
        and dataset_names is not None
    ):
        dataset_codes = np.asarray(dataset_codes, dtype=int).reshape(-1)
        unique_datasets = np.unique(dataset_codes)
        colors = plt.cm.Set2(np.linspace(0, 1, unique_datasets.size))
        # Track values that were actually plotted to derive faithful y-limits.
        dataset_values = []

        ax_ds = axes_flat[panel_idx]
        for ds_color, ds_code in zip(colors, unique_datasets):
            in_ds = dataset_codes == ds_code
            if np.count_nonzero(in_ds) < max(min_cells_per_bin, 10):
                continue
            if int(ds_code) < len(dataset_names):
                ds_label = str(dataset_names[int(ds_code)])
            else:
                ds_label = f"dataset_{int(ds_code)}"
            _plot_trend_line(
                ax_ds,
                library_size[in_ds],
                p_capture[in_ds],
                label=ds_label,
                color=ds_color,
                n_bins=n_bins,
                min_cells_per_bin=min_cells_per_bin,
            )
            dataset_values.append(p_capture[in_ds])
        ax_ds.set_xlabel(r"$L_c$")
        ax_ds.set_ylabel(r"$\hat{p}_{\mathrm{capture},c}^{\mathrm{MAP}}$")
        ax_ds.set_title(r"Split by dataset")
        if dataset_values:
            _set_dynamic_y_limits(ax_ds, np.concatenate(dataset_values))
        else:
            _set_dynamic_y_limits(ax_ds, p_capture)
        handles, labels = ax_ds.get_legend_handles_labels()
        if handles:
            ax_ds.legend(fontsize=8)

    fig.tight_layout()

    return fig, axes_flat, len(panel_specs)

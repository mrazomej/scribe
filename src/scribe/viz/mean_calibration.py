"""Mean calibration diagnostic: observed vs predicted per-gene means.

Produces a log-log scatter comparing the empirical gene mean from the
count matrix against the predicted gene mean from the model's MAP
parameters.  This is the single most informative diagnostic for mean
calibration.

For mixture models the predicted mean is the marginal (weighted)
average across components, which is directly comparable to the
sample-wide observed mean regardless of whether cell labels are
available.

For VCP models the biological NB mean is scaled by the average
capture probability to yield the predicted *observed* mean.

The BNB uses a mean-preserving parameterization, so the predicted
mean formula ``r * p / (1 - p)`` is valid whether BNB is active or
not.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from ._common import console
from ._interactive import (
    _create_or_validate_grid_axes,
    _create_or_validate_single_axis,
    plot_function,
)
from .dispatch import _get_layouts_for_plot, _get_map_estimates_for_plot
from .gene_selection import _coerce_counts


# =========================================================================
# Helpers
# =========================================================================


def _compute_predicted_mean(
    r,
    p,
    mixing_weights=None,
    p_capture=None,
    *,
    layouts,
):
    """Compute predicted per-gene observed mean from MAP parameters.

    Parameters
    ----------
    r : ndarray
        NB dispersion in canonical form.
        Standard model: ``(G,)``.
        Mixture: ``(K, G)`` or ``(K, ...dataset..., G)``.
    p : ndarray
        NB success probability (numpyro convention).
        Standard model: scalar or ``(G,)``.
        Mixture: scalar, ``(K,)``, or ``(K, G)``.
    mixing_weights : ndarray or None
        Component weights ``(K,)`` for mixture models.
    p_capture : ndarray or None
        Per-cell capture probability ``(C,)``.  ``None`` for
        non-VCP models (treated as :math:`\\nu = 1`).
    layouts : dict of str to AxisLayout
        **Required.** Canonical MAP-level layouts from
        ``_get_layouts_for_plot``.  Must contain entries for ``"r"``
        and ``"p"`` (and ``"mixing_weights"`` when present).

    Returns
    -------
    pred : ndarray, shape ``(G,)``
        Predicted per-gene observed mean.
    """
    r = np.asarray(r, dtype=float)
    p = np.asarray(p, dtype=float)

    # Align p with r for broadcasting.  When p has a component axis
    # but NOT a gene axis, add trailing dimensions so it broadcasts
    # along the component axis of r.  When p already matches r's shape
    # (e.g. both (K, G) for gene-specific p), leave it unchanged.
    if (
        layouts["p"].component_axis is not None
        and layouts["p"].gene_axis is None
        and p.shape != r.shape
    ):
        p = p.reshape((-1,) + (1,) * (r.ndim - 1))

    p_safe = np.clip(p, 1e-8, 1.0 - 1e-8)
    mu_bio = r * p_safe / (1.0 - p_safe)

    # Use layout metadata to determine whether r is per-component.
    if layouts["r"].component_axis is not None and mixing_weights is not None:
        w = np.asarray(mixing_weights, dtype=float)
        w = w.reshape((-1,) + (1,) * (mu_bio.ndim - 1))
        mu_bio = np.sum(w * mu_bio, axis=0)

    # Collapse remaining non-gene dimensions (e.g. dataset) by averaging.
    while mu_bio.ndim > 1:
        mu_bio = np.mean(mu_bio, axis=0)

    if p_capture is not None:
        mean_nu = float(np.mean(np.asarray(p_capture, dtype=float)))
    else:
        mean_nu = 1.0

    return mu_bio * mean_nu


def _compute_per_dataset_means(
    counts,
    r,
    p,
    dataset_codes,
    dataset_names,
    mixing_weights=None,
    p_capture=None,
    n_datasets=None,
    *,
    layouts,
):
    """Compute observed and predicted means per dataset.

    Parameters
    ----------
    counts : ndarray, shape ``(C, G)``
    r, p : ndarray
        Canonical MAP parameters.
    dataset_codes : ndarray, shape ``(C,)``
        Integer dataset index per cell.
    dataset_names : sequence of str
    mixing_weights : ndarray or None
    p_capture : ndarray or None, shape ``(C,)``
    n_datasets : int or None
    layouts : dict of str to AxisLayout
        **Required.** Canonical MAP-level layouts from
        ``_get_layouts_for_plot``.  Must contain entries for every
        parameter key (``"r"``, ``"p"``, and optionally
        ``"mixing_weights"``) so that dataset-axis slicing uses
        ``layout.dataset_axis``.

    Returns
    -------
    list of dict
        Each dict has keys ``name``, ``obs_mean``, ``pred_mean``.
    """
    counts = _coerce_counts(counts)
    ds_codes = np.asarray(dataset_codes, dtype=int).ravel()
    unique_ds = np.unique(ds_codes)
    _n_ds = n_datasets if n_datasets is not None else len(unique_ds)

    results = []
    for d in unique_ds:
        mask = ds_codes == d

        obs_mean = np.mean(np.asarray(counts[mask], dtype=float), axis=0)

        def _slice(param, ds_idx, key):
            """Slice a parameter along its dataset axis via layout metadata."""
            if param is None:
                return None
            param = np.asarray(param, dtype=float)
            ds_ax = layouts[key].dataset_axis
            if ds_ax is not None:
                return np.take(param, ds_idx, axis=ds_ax)
            return param

        r_d = _slice(r, int(d), key="r")
        p_d = _slice(p, int(d), key="p")

        def _slice_mixing(weights, ds_idx):
            """Slice mixing weights along the dataset axis via layout metadata."""
            if weights is None:
                return None
            weights = np.asarray(weights, dtype=float)
            ds_ax = layouts["mixing_weights"].dataset_axis
            if ds_ax is not None:
                return np.take(weights, ds_idx, axis=ds_ax)
            return weights

        mixing_d = _slice_mixing(mixing_weights, int(d))
        pc_d = (
            np.asarray(p_capture, dtype=float)[mask]
            if p_capture is not None
            else None
        )

        pred_mean = _compute_predicted_mean(
            r_d,
            p_d,
            mixing_d,
            pc_d,
            layouts=layouts,
        )

        name = (
            str(dataset_names[int(d)])
            if int(d) < len(dataset_names)
            else f"dataset_{int(d)}"
        )
        results.append(
            {
                "name": name,
                "obs_mean": obs_mean,
                "pred_mean": pred_mean,
            }
        )

    return results


def _scatter_panel(
    ax,
    obs,
    pred,
    *,
    label=None,
    color=None,
    alpha=0.35,
    s=6,
):
    """Draw a single log-log scatter on *ax* with identity line and stats.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    obs, pred : ndarray, shape ``(G,)``
    label : str or None
    color : str or array-like or None
    alpha, s : float
        Scatter aesthetics.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    # Pseudocount to avoid log(0)
    log_obs = np.log10(obs + 1.0)
    log_pred = np.log10(pred + 1.0)

    ax.scatter(
        log_obs,
        log_pred,
        s=s,
        alpha=alpha,
        color=color,
        label=label,
        edgecolors="none",
        rasterized=True,
    )

    # Identity line spanning the data range
    lo = min(log_obs.min(), log_pred.min())
    hi = max(log_obs.max(), log_pred.max())
    margin = (hi - lo) * 0.05
    lims = [lo - margin, hi + margin]
    ax.plot(lims, lims, ls="--", lw=1.0, color="0.3", zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    # Correlation and median ratio (only for finite values)
    valid = np.isfinite(log_obs) & np.isfinite(log_pred) & (obs > 0)
    if np.sum(valid) > 10:
        r_pearson = float(np.corrcoef(log_obs[valid], log_pred[valid])[0, 1])
        r_spearman = float(
            sp_stats.spearmanr(obs[valid], pred[valid]).statistic
        )
        ratio = np.median(pred[valid] / obs[valid])
        stat_text = (
            f"Pearson $r = {r_pearson:.3f}$\n"
            f"Spearman $\\rho = {r_spearman:.3f}$\n"
            f"median ratio $= {ratio:.2f}$"
        )
        ax.text(
            0.05,
            0.95,
            stat_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85
            ),
        )


# =========================================================================
# Public API
# =========================================================================


def _prepare_calibration_data(
    results,
    counts,
    *,
    is_mixture=False,
    is_multi_dataset=False,
    dataset_codes=None,
    dataset_names=None,
):
    """Prepare observed and predicted means for calibration plotting.

    Extracts MAP parameters, computes predicted gene means, and
    optionally splits by dataset.

    Parameters
    ----------
    results : object
        Fitted model results.
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
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
        Dictionary with keys:
        - ``mode``: ``"multi_dataset"`` or ``"single"``
        - ``ds_results``: list of dicts (multi-dataset) or None
        - ``obs_mean``, ``pred_mean``: arrays (single-dataset)
        - ``is_mixture``: bool
        - ``annotations``: list of annotation strings
        Returns ``None`` when r/p are unavailable.
    """
    counts = _coerce_counts(counts)
    # Request only required keys up front so this diagnostic works for
    # non-mixture and non-BNB models without forcing optional parameters.
    required_targets = ["r", "p"]
    if bool(getattr(results.model_config, "uses_variable_capture", False)):
        required_targets.append("p_capture")
    map_estimates = _get_map_estimates_for_plot(
        results, counts=counts, targets=required_targets
    )
    r = map_estimates.get("r")
    p = map_estimates.get("p")
    if r is None or p is None:
        console.print(
            "[yellow]Skipping mean calibration: r and/or p unavailable "
            "in MAP estimates.[/yellow]"
        )
        return None

    # Fetch AxisLayout metadata for layout-driven axis lookups.
    layouts = _get_layouts_for_plot(results)

    mixing_weights = None
    if is_mixture:
        try:
            mix_map = _get_map_estimates_for_plot(
                results, counts=counts, targets=["mixing_weights"]
            )
            mixing_weights = mix_map.get("mixing_weights")
        except ValueError:
            mixing_weights = None
    p_capture = map_estimates.get("p_capture")

    _annotations = []
    bnb_kappa = None
    bnb_concentration = None
    if bool(getattr(results.model_config, "is_bnb", False)):
        # BNB-specific annotations are optional and should never block plotting.
        try:
            bnb_map = _get_map_estimates_for_plot(
                results, counts=counts, targets=["bnb_kappa"]
            )
            bnb_kappa = bnb_map.get("bnb_kappa")
        except ValueError:
            bnb_kappa = None
        if bnb_kappa is None:
            try:
                bnb_map = _get_map_estimates_for_plot(
                    results, counts=counts, targets=["bnb_concentration"]
                )
                bnb_concentration = bnb_map.get("bnb_concentration")
            except ValueError:
                bnb_concentration = None
    if bnb_kappa is not None:
        median_kappa = float(np.median(np.asarray(bnb_kappa)))
        _annotations.append(
            f"BNB active (median $\\kappa = {median_kappa:.1f}$)"
        )
    elif bnb_concentration is not None:
        _annotations.append("BNB active")

    if p_capture is not None:
        mean_nu = float(np.mean(np.asarray(p_capture, dtype=float)))
        _annotations.append(f"$\\bar{{\\nu}} = {mean_nu:.4f}$")

    if (
        is_multi_dataset
        and dataset_codes is not None
        and dataset_names is not None
    ):
        n_ds_cfg = getattr(
            getattr(results, "model_config", None), "n_datasets", None
        )
        ds_results = _compute_per_dataset_means(
            counts,
            r,
            p,
            dataset_codes,
            dataset_names,
            mixing_weights=mixing_weights,
            p_capture=p_capture,
            n_datasets=n_ds_cfg,
            layouts=layouts,
        )
        return {
            "mode": "multi_dataset",
            "ds_results": ds_results,
            "obs_mean": None,
            "pred_mean": None,
            "is_mixture": is_mixture,
            "annotations": _annotations,
        }

    obs_mean = np.mean(np.asarray(counts, dtype=float), axis=0)
    pred_mean = _compute_predicted_mean(
        r,
        p,
        mixing_weights,
        p_capture,
        layouts=layouts,
    )
    return {
        "mode": "single",
        "ds_results": None,
        "obs_mean": obs_mean,
        "pred_mean": pred_mean,
        "is_mixture": is_mixture,
        "annotations": _annotations,
    }


@plot_function(
    suffix="mean_calibration",
    save_label="mean-calibration plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_mean_calibration(
    results,
    counts,
    *,
    ctx,
    viz_cfg=None,
    is_mixture=False,
    is_multi_dataset=False,
    dataset_codes=None,
    dataset_names=None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    r"""Log-log scatter of observed vs predicted per-gene mean counts.

    The predicted mean is derived from the MAP estimates of the
    canonical parameters :math:`r_g` and :math:`p_g`:

    .. math::

        \langle u_g \rangle_{\text{pred}}
        = \bar{\nu} \sum_k w_k \frac{r_{kg}\,p_{kg}}{1 - p_{kg}}

    where the mixture weights :math:`w_k` and capture probability
    :math:`\bar{\nu}` default to 1 for non-mixture / non-VCP models.

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
        Visualization config.
    is_mixture : bool, optional
        Whether the model uses more than one component.
    is_multi_dataset : bool, optional
        Whether the run includes multiple datasets.
    dataset_codes : ndarray, optional
        Integer dataset code per cell.
    dataset_names : sequence of str, optional
        Category names for each dataset code.

    Returns
    -------
    PlotResult or None
        Wrapped result containing the figure, axes, and metadata.
    """
    console.print("[dim]Plotting mean-calibration diagnostic...[/dim]")
    if ax is not None and is_multi_dataset:
        raise ValueError(
            "Multi-dataset mean calibration requires `fig` or `axes`, not `ax`."
        )
    prep = _prepare_calibration_data(
        results,
        counts,
        is_mixture=is_mixture,
        is_multi_dataset=is_multi_dataset,
        dataset_codes=dataset_codes,
        dataset_names=dataset_names,
    )
    if prep is None:
        return None

    _annotations = prep["annotations"]

    # ---- Multi-dataset: per-dataset panels ----------------------------------
    if prep["mode"] == "multi_dataset":
        ds_results = prep["ds_results"]
        n_panels = len(ds_results)
        fig, _, axes_flat = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=n_panels,
            fig=fig,
            axes=axes,
            figsize=figsize or (5.5 * n_panels, 5.0),
        )
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_panels, 2)))
        for i, ds in enumerate(ds_results):
            _scatter_panel(
                axes_flat[i],
                ds["obs_mean"],
                ds["pred_mean"],
                color=colors[i],
            )
            axes_flat[i].set_xlabel(
                r"$\log_{10}(\bar{u}_g^{\mathrm{obs}} + 1)$"
            )
            axes_flat[i].set_ylabel(
                r"$\log_{10}(\bar{u}_g^{\mathrm{pred}} + 1)$"
            )
            axes_flat[i].set_title(ds["name"], fontsize=10)

    # ---- Single-dataset (with or without mixture) ---------------------------
    else:
        obs_mean = prep["obs_mean"]
        pred_mean = prep["pred_mean"]

        fig, ax = _create_or_validate_single_axis(
            fig=fig,
            ax=ax,
            axes=axes,
            figsize=figsize or (5.5, 5.0),
        )
        _scatter_panel(ax, obs_mean, pred_mean, color="steelblue")
        ax.set_xlabel(r"$\log_{10}(\bar{u}_g^{\mathrm{obs}} + 1)$")
        ax.set_ylabel(r"$\log_{10}(\bar{u}_g^{\mathrm{pred}} + 1)$")
        axes_flat = [ax]

    # Suptitle with model annotations
    title = "Mean Calibration"
    if prep["is_mixture"]:
        n_comp = getattr(results, "n_components", None)
        if n_comp:
            title += f" ({n_comp}-component mixture, weighted)"
    if _annotations:
        title += "\n" + ", ".join(_annotations)
    fig.suptitle(title, fontsize=11, y=1.02)

    fig.tight_layout()

    return fig, axes_flat, len(axes_flat)

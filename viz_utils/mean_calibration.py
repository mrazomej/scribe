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
from .config import _get_config_values
from .dispatch import _get_map_estimates_for_plot


# =========================================================================
# Helpers
# =========================================================================


def _compute_predicted_mean(
    r,
    p,
    mixing_weights=None,
    p_capture=None,
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

    Returns
    -------
    pred : ndarray, shape ``(G,)``
        Predicted per-gene observed mean.
    """
    r = np.asarray(r, dtype=float)
    p = np.asarray(p, dtype=float)

    # Broadcast p to match r when p is (K,) and r is (K, G) or deeper
    if r.ndim >= 2 and p.ndim == 1 and p.shape[0] == r.shape[0]:
        p = p.reshape((-1,) + (1,) * (r.ndim - 1))

    # Biological NB mean per gene (per component if mixture)
    p_safe = np.clip(p, 1e-8, 1.0 - 1e-8)
    mu_bio = r * p_safe / (1.0 - p_safe)

    # For mixture models: weighted average across component axis (axis 0)
    is_mixture = mixing_weights is not None and r.ndim >= 2
    if is_mixture:
        w = np.asarray(mixing_weights, dtype=float)
        # Reshape weights to (K, 1, ...) for broadcasting
        w = w.reshape((-1,) + (1,) * (mu_bio.ndim - 1))
        mu_bio = np.sum(w * mu_bio, axis=0)

    # Collapse any remaining non-gene dimensions (e.g. dataset) by averaging
    while mu_bio.ndim > 1:
        mu_bio = np.mean(mu_bio, axis=0)

    # Scale by average capture probability
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

    Returns
    -------
    list of dict
        Each dict has keys ``name``, ``obs_mean``, ``pred_mean``.
    """
    ds_codes = np.asarray(dataset_codes, dtype=int).ravel()
    unique_ds = np.unique(ds_codes)
    _n_ds = n_datasets if n_datasets is not None else len(unique_ds)

    results = []
    for d in unique_ds:
        mask = ds_codes == d

        # Observed mean for this dataset's cells
        obs_mean = np.mean(np.asarray(counts[mask], dtype=float), axis=0)

        # Slice per-dataset parameters if the leading dim matches n_datasets
        def _slice(param, ds_idx):
            if param is None:
                return None
            param = np.asarray(param, dtype=float)
            if param.ndim >= 1 and param.shape[0] == _n_ds:
                return param[ds_idx]
            # Mixture with dataset dim: (K, D, G) -> (K, G)
            if param.ndim >= 2 and param.shape[1] == _n_ds:
                return param[:, ds_idx]
            return param

        r_d = _slice(r, int(d))
        p_d = _slice(p, int(d))

        # Mixing weights need dedicated slicing logic because a 1-D global
        # weight vector has shape (K,). When K == n_datasets, generic shape
        # heuristics would incorrectly treat it as per-dataset and slice a
        # scalar, breaking mixture broadcasting.
        def _slice_mixing(weights, ds_idx):
            if weights is None:
                return None
            weights = np.asarray(weights, dtype=float)
            if weights.ndim == 1:
                # Global mixture weights (K,) shared by all datasets.
                return weights
            if weights.ndim >= 2 and weights.shape[0] == _n_ds:
                # Dataset-major layout: (D, K)
                return weights[ds_idx]
            if weights.ndim >= 2 and weights.shape[1] == _n_ds:
                # Component-major layout: (K, D)
                return weights[:, ds_idx]
            return weights

        mixing_d = _slice_mixing(mixing_weights, int(d))
        pc_d = np.asarray(p_capture, dtype=float)[mask] if p_capture is not None else None

        pred_mean = _compute_predicted_mean(r_d, p_d, mixing_d, pc_d)

        name = (
            str(dataset_names[int(d)])
            if int(d) < len(dataset_names)
            else f"dataset_{int(d)}"
        )
        results.append({
            "name": name,
            "obs_mean": obs_mean,
            "pred_mean": pred_mean,
        })

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

    ax.scatter(log_obs, log_pred, s=s, alpha=alpha, color=color, label=label,
               edgecolors="none", rasterized=True)

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
        r_spearman = float(sp_stats.spearmanr(obs[valid], pred[valid]).statistic)
        ratio = np.median(pred[valid] / obs[valid])
        stat_text = (
            f"Pearson $r = {r_pearson:.3f}$\n"
            f"Spearman $\\rho = {r_spearman:.3f}$\n"
            f"median ratio $= {ratio:.2f}$"
        )
        ax.text(
            0.05, 0.95, stat_text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85),
        )


# =========================================================================
# Public API
# =========================================================================


def plot_mean_calibration(
    results,
    counts,
    figs_dir,
    cfg,
    viz_cfg,
    *,
    is_mixture=False,
    is_multi_dataset=False,
    dataset_codes=None,
    dataset_names=None,
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
    str
        Output file path.
    """
    console.print("[dim]Plotting mean-calibration diagnostic...[/dim]")

    map_estimates = _get_map_estimates_for_plot(
        results,
        counts=counts,
        targets=[
            "r",
            "p",
            "mixing_weights",
            "p_capture",
            "bnb_kappa",
            "bnb_concentration",
        ],
    )
    r = map_estimates.get("r")
    p = map_estimates.get("p")
    if r is None or p is None:
        console.print(
            "[yellow]Skipping mean calibration: r and/or p unavailable "
            "in MAP estimates.[/yellow]"
        )
        return None

    mixing_weights = map_estimates.get("mixing_weights") if is_mixture else None
    p_capture = map_estimates.get("p_capture")

    # Build annotation string for model details
    _annotations = []
    bnb_kappa = map_estimates.get("bnb_kappa")
    if bnb_kappa is not None:
        median_kappa = float(np.median(np.asarray(bnb_kappa)))
        _annotations.append(f"BNB active (median $\\kappa = {median_kappa:.1f}$)")
    elif map_estimates.get("bnb_concentration") is not None:
        _annotations.append("BNB active")

    if p_capture is not None:
        mean_nu = float(np.mean(np.asarray(p_capture, dtype=float)))
        _annotations.append(f"$\\bar{{\\nu}} = {mean_nu:.4f}$")

    # ---- Multi-dataset: per-dataset panels ----------------------------------
    if (
        is_multi_dataset
        and dataset_codes is not None
        and dataset_names is not None
    ):
        n_ds_cfg = getattr(
            getattr(results, "model_config", None), "n_datasets", None
        )
        ds_results = _compute_per_dataset_means(
            counts, r, p, dataset_codes, dataset_names,
            mixing_weights=mixing_weights,
            p_capture=p_capture,
            n_datasets=n_ds_cfg,
        )
        n_panels = len(ds_results)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5.5 * n_panels, 5.0),
            squeeze=False,
        )
        axes = axes.flatten()
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_panels, 2)))
        for i, ds in enumerate(ds_results):
            _scatter_panel(
                axes[i], ds["obs_mean"], ds["pred_mean"],
                color=colors[i],
            )
            axes[i].set_xlabel(r"$\log_{10}(\bar{u}_g^{\mathrm{obs}} + 1)$")
            axes[i].set_ylabel(r"$\log_{10}(\bar{u}_g^{\mathrm{pred}} + 1)$")
            axes[i].set_title(ds["name"], fontsize=10)

    # ---- Single-dataset (with or without mixture) ---------------------------
    else:
        obs_mean = np.mean(np.asarray(counts, dtype=float), axis=0)
        pred_mean = _compute_predicted_mean(r, p, mixing_weights, p_capture)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
        _scatter_panel(ax, obs_mean, pred_mean, color="steelblue")
        ax.set_xlabel(r"$\log_{10}(\bar{u}_g^{\mathrm{obs}} + 1)$")
        ax.set_ylabel(r"$\log_{10}(\bar{u}_g^{\mathrm{pred}} + 1)$")

    # Suptitle with model annotations
    title = "Mean Calibration"
    if is_mixture:
        n_comp = getattr(results, "n_components", None)
        if n_comp:
            title += f" ({n_comp}-component mixture, weighted)"
    if _annotations:
        title += "\n" + ", ".join(_annotations)
    fig.suptitle(title, fontsize=11, y=1.02)

    plt.tight_layout()

    # Save
    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_mean_calibration.{output_format}"
    )
    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    console.print(
        "[green]\u2713[/green] [dim]Saved mean-calibration plot to[/dim] "
        f"[cyan]{output_path}[/cyan]"
    )
    plt.close(fig)
    return output_path

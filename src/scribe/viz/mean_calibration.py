"""Mean calibration diagnostic: observed vs predicted per-gene means.

Produces a log-log scatter comparing the empirical gene mean from the
count matrix against the predicted gene mean from the model's MAP
parameters.  This is the single most informative diagnostic for mean
calibration.

For NB-family models (DM, NBVCP, ZINB, etc.) the predicted mean is
``r * p / (1 - p)`` scaled by VCP capture probability when applicable.

For LNM (Logistic Normal Multinomial) models the predicted mean is
``rho_g * mean(u_T_obs)`` where ``rho`` is the compositional simplex
from the inverse-ALR of the MAP ``y_alr``, and ``mean(u_T_obs)`` is
the observed mean total UMI per cell.  We use the observed total
rather than the NB model's ``r_T * p / (1 - p)`` because the
composition and total-count are separate sub-models in LNM and the
NB MAP point estimates can be unreliable for this purpose; the
diagnostic therefore isolates the compositional fit quality.

For PLN (Poisson-LogNormal) models the predicted mean is
``exp(y_log_rate)`` where ``y_log_rate`` is the MAP decoder output in
log-rate space.

For mixture models the predicted mean is the marginal (weighted)
average across components, which is directly comparable to the
sample-wide observed mean regardless of whether cell labels are
available.

The BNB uses a mean-preserving parameterization, so the predicted
mean formula ``r * p / (1 - p)`` is valid whether BNB is active or
not.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from ._common import _is_pln_model, console
from ._interactive import (
    _create_or_validate_grid_axes,
    _create_or_validate_single_axis,
    plot_function,
)
from .dispatch import _get_layouts_for_plot, _get_map_estimates_for_plot
from .gene_selection import _coerce_and_align_counts_to_results, _coerce_counts


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
    """Compute predicted per-gene observed mean from NB MAP parameters.

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


def _compute_predicted_mean_lnm(
    y_alr,
    counts,
    *,
    alr_reference_idx=-1,
):
    """Compute predicted per-gene observed mean for LNM models.

    The LNM predicted mean per gene is ``rho_g * mean(u_T)`` where
    ``rho`` is the compositional simplex from the inverse-ALR of the
    MAP ``y_alr``, and ``mean(u_T)`` is the observed mean total UMI
    count per cell.

    We use the observed mean total rather than the NB model's
    ``r_T * p / (1 - p)`` because the composition (``rho``) and
    total-count (``NB(r_T, p)``) are separate sub-models in LNM.
    The NB MAP point estimates can lie in extreme regions of the
    parameter space (small ``r_T``, ``p`` near 1) where the formula
    produces unreliable values, even though posterior predictive
    samples are fine.  Using the observed total isolates this
    diagnostic to test the compositional model only.

    Parameters
    ----------
    y_alr : ndarray, shape ``(G-1,)``
        MAP ALR coordinates from the VAE decoder.
    counts : ndarray, shape ``(C, G)``
        Observed UMI count matrix (already aligned to model gene space).
    alr_reference_idx : int
        Index of the ALR reference gene (default ``-1`` = last gene).

    Returns
    -------
    pred : ndarray, shape ``(G,)``
        Predicted per-gene observed mean.
    """
    from ..core.normalization_logistic import _inverse_alr
    import jax.numpy as jnp

    # Step 1: recover the simplex composition ρ from ALR
    # coordinates. ``_inverse_alr`` augments y_alr with a zero
    # at the reference position and applies softmax — so the
    # output ``rho`` is in proper composition space (each row
    # sums to 1).
    y_alr_arr = jnp.asarray(y_alr, dtype=jnp.float32)
    counts_arr = np.asarray(counts, dtype=float)
    n_per_cell = np.sum(counts_arr, axis=1)  # observed totals per cell

    # Two input shapes are supported:
    #   * 1-D ``(G-1,)`` — a single population-level y_alr (e.g.
    #     the global decoder output for VAE-style results
    #     without an explicit per-cell ALR map). Population
    #     prediction is ``ρ_global × mean_c[N_c]``.
    #   * 2-D ``(n_cells, G-1)`` — per-cell y_alr from Laplace
    #     results (one ALR vector per training cell). The
    #     honest population prediction multiplies each cell's
    #     composition by its OWN observed total (preserving
    #     correlation between library size and composition),
    #     then averages across cells:
    #         pred[g] = mean_c[ρ_c[g] · N_c]
    #     rather than
    #         pred[g] = mean_c[ρ_c[g]] · mean_c[N_c]
    #     which would assume independence between N and ρ.
    if y_alr_arr.ndim == 1:
        y_alr_arr = y_alr_arr[None, :]
        rho_global = np.asarray(
            _inverse_alr(y_alr_arr, reference_index=alr_reference_idx)
        ).squeeze(axis=0)
        return rho_global * float(np.mean(n_per_cell))
    if y_alr_arr.ndim == 2:
        rho_per_cell = np.asarray(
            _inverse_alr(y_alr_arr, reference_index=alr_reference_idx)
        )  # shape (n_cells, G)
        # Per-cell predicted counts then population-mean.
        pred_per_cell = rho_per_cell * n_per_cell[:, None]
        return pred_per_cell.mean(axis=0)
    raise ValueError(
        f"y_alr must be 1-D (G-1,) or 2-D (n_cells, G-1); "
        f"got shape {tuple(y_alr_arr.shape)}."
    )


# Map-level log-rate predictions for PLN, accounting for any per-cell
# capture offset and learned diagonal residual variance.
def _compute_predicted_mean_pln(
    y_log_rate, eta_capture=None, d_pln=None
):
    """Compute predicted per-gene observed mean for PLN models.

    The PLN observation model is

    .. math::
        u_g^{(c)} \\mid x_g^{(c)} \\sim \\text{Poisson}(\\exp(x_g^{(c)})),
        \\qquad x_g^{(c)} = y_{\\text{log-rate},\\,g,\\,c} - \\eta_c,

    where :math:`\\eta_c` is the per-cell capture offset (zero when no
    capture anchor is in use) and ``y_log_rate`` is the decoder output
    in *biological* log-rate space (before capture). The per-gene
    expected observed count is therefore

    .. math::
        \\mathbb{E}[u_g] \\;=\\; \\frac{1}{C}\\sum_c
        \\exp\\!\\left(y_{\\text{log-rate},\\,g,\\,c} - \\eta_c\\right).

    When ``d_mode = "learned"`` the decoder also adds a diagonal
    residual ``sqrt(d_g) * eps`` in log-rate space, contributing a
    multiplicative ``exp(d_g/2)`` to each gene's mean by the
    log-normal moment formula.

    Earlier revisions of this helper computed ``exp(y_log_rate)``
    *without* subtracting ``eta_capture``, which returned the
    *biological* rate rather than the *observed* rate -- on a fitted
    PLNVCP model the diagnostic was systematically inflated by
    ``1/p_capture``. The corrected formula brings the diagnostic into
    the same observed-counts space as the empirical mean it is
    plotted against.

    Parameters
    ----------
    y_log_rate : ndarray, shape ``(G,)``, ``(1, G)``, or ``(n_cells, G)``
        MAP log-rate from the PLN decoder. When per-cell, the
        diagnostic averages ``exp(...)`` across cells.
    eta_capture : ndarray or None, shape ``(n_cells,)`` or scalar
        Per-cell capture offset. When ``None``, no capture correction is
        applied (e.g. fits without ``priors={"capture_efficiency": ...}``).
    d_pln : ndarray or None, shape ``(G,)``
        Per-gene learned residual variance. When ``None`` the
        log-normal moment correction is skipped.

    Returns
    -------
    pred : ndarray, shape ``(G,)``
        Per-gene predicted observed mean.
    """
    y_arr = np.asarray(y_log_rate, dtype=float)

    # Apply per-cell capture offset, if any. The offset enters as a
    # *subtraction* in log-rate space, broadcast across genes.
    if eta_capture is not None:
        eta_arr = np.asarray(eta_capture, dtype=float)
        if y_arr.ndim == 1:
            # Scalar / per-cell-collapsed y_log_rate: subtract the
            # mean offset; the per-cell distribution is lost so we use
            # the cell-averaged shift as a best-effort summary.
            y_arr = y_arr - float(eta_arr.mean())
        else:
            # Per-cell y_log_rate: align eta_capture to (n_cells,) and
            # subtract per cell, broadcasting across genes.
            eta_arr = eta_arr.reshape(-1)
            if eta_arr.shape[0] != y_arr.shape[0]:
                # Shape mismatch: fall back to the mean offset and
                # warn at most via the calibration caller.
                y_arr = y_arr - float(eta_arr.mean())
            else:
                y_arr = y_arr - eta_arr[:, None]

    # Average ``exp`` across cells for per-cell input; otherwise leave
    # the (G,) shape alone.
    rate = np.exp(y_arr)
    if rate.ndim > 1:
        rate = rate.mean(axis=0)
    rate = np.squeeze(rate)

    # Log-normal moment correction for learned diagonal residual.
    if d_pln is not None:
        d_arr = np.asarray(d_pln, dtype=float).reshape(-1)
        rate = rate * np.exp(d_arr / 2.0)

    return rate


def _is_lnm_model(results) -> bool:
    """Return True if the results use a logistic-normal parameterization."""
    param = getattr(
        getattr(results, "model_config", None), "parameterization", None
    )
    param_value = getattr(param, "value", param)
    param_name = getattr(param, "name", None)
    # Match any of the LNM-family variants (canonical / mean_prob /
    # mean_odds). All three share the compositional path for which the
    # mean-calibration plot is meaningful.
    return (
        isinstance(param_value, str)
        and param_value.startswith("logistic_normal")
    ) or (
        isinstance(param_name, str)
        and param_name.startswith("LOGISTIC_NORMAL")
    )


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
        Returns ``None`` when required MAP parameters are unavailable.
    """
    counts = _coerce_and_align_counts_to_results(
        counts, results, context="_prepare_calibration_data"
    )

    _use_lnm = _is_lnm_model(results)
    _use_pln = _is_pln_model(results)

    # ---- LNM path: predicted mean = rho * obs_mean_total ---------------------
    if _use_lnm:
        lnm_targets = ["y_alr"]
        if bool(getattr(results.model_config, "uses_variable_capture", False)):
            lnm_targets.append("p_capture")
        map_estimates = _get_map_estimates_for_plot(
            results, counts=counts, targets=lnm_targets
        )
        y_alr = map_estimates.get("y_alr")
        if y_alr is None:
            console.print(
                "[yellow]Skipping mean calibration: y_alr unavailable "
                "in MAP estimates for LNM model.[/yellow]"
            )
            return None

        alr_ref = getattr(results.model_config, "alr_reference_idx", -1)
        p_capture = map_estimates.get("p_capture")

        _annotations = []
        if p_capture is not None:
            mean_nu = float(np.mean(np.asarray(p_capture, dtype=float)))
            _annotations.append(f"$\\bar{{\\nu}} = {mean_nu:.4f}$")
        _annotations.append("LNM (composition)")

        obs_mean = np.mean(np.asarray(counts, dtype=float), axis=0)
        pred_mean = _compute_predicted_mean_lnm(
            y_alr, counts,
            alr_reference_idx=alr_ref,
        )
        return {
            "mode": "single",
            "ds_results": None,
            "obs_mean": obs_mean,
            "pred_mean": pred_mean,
            "is_mixture": False,
            "annotations": _annotations,
        }

    # ---- PLN path: predicted observed mean -----------------------------------
    # We have two routes for the per-gene predicted mean:
    #
    # 1. **Per-cell MAP path (preferred when available)**. Laplace
    #    results expose the per-cell MAP log-rate ``x_loc`` directly,
    #    and the honest predicted mean for the data-informative
    #    regime is then
    #
    #        pred_g  =  (1/C) sum_c  exp(x_g^(c)  -  eta_c).
    #
    #    No log-normal moment correction is added: ``x_loc`` is the
    #    fitted *posterior MAP*, not a prior sample, and for
    #    informative likelihood the per-cell posterior is sharply
    #    concentrated around the MAP (so ``E[exp(x-eta)] ≈
    #    exp(MAP-eta)`` to leading order). Adding ``exp(0.5*sigma_g^2)``
    #    here would over-correct by treating the data as
    #    non-informative. For PPC-consistency in the non-informative
    #    limit a per-cell Hessian-based variance would be needed; the
    #    PPC plot remains the gold standard for that regime.
    #
    # 2. **Population-level fallback (VAE / encoder-only results)**.
    #    When ``x_loc`` is not available, we approximate
    #
    #        pred_g  ≈  exp(mu_g  +  0.5*(||W_g||²  +  d_g))
    #                   *  (1/C) sum_c  exp(-eta_c).
    #
    #    The first factor is ``LowRankPoissonLogNormal.mean``
    #    (prior-predictive moment of the log-rate) and the second is
    #    the empirical capture factor. The factorization is exact
    #    only when the aggregate posterior over ``z`` matches the
    #    ``N(0, I)`` prior. Aggregate-posterior drift -- e.g. the
    #    ``q(z)``-expanded regime that arises in PLN Laplace with a
    #    weak capture prior -- biases this path by a factor that
    #    depends on the drift magnitude (under-prediction when
    #    ``var(z) > 1`` in some directions). The per-cell MAP path
    #    (route 1) sidesteps that issue.
    #
    # Subtle Jensen point on the capture factor: the empirical mean
    # capture is ``mean_c[exp(-eta_c)]``, *not* ``exp(-mean(eta_c))``.
    # The two differ by ``exp(tau^2/2)`` where ``tau`` is the std of
    # ``eta_c``; using the wrong form would under-predict by ~22%
    # for typical ``tau ~ 0.7``. Both routes above use the unbiased
    # form.
    if _use_pln:
        # Pull eta_capture and d_pln when available.
        pln_targets = []
        if bool(
            getattr(
                results.model_config,
                "uses_biology_informed_capture",
                False,
            )
        ):
            pln_targets.append("eta_capture")
        if (
            getattr(results.model_config, "d_mode", "low_rank") == "learned"
        ):
            pln_targets.append("d_pln")
        map_estimates = (
            _get_map_estimates_for_plot(
                results, counts=counts, targets=pln_targets
            )
            if pln_targets
            else {}
        )
        eta_capture = map_estimates.get("eta_capture")
        d_pln = map_estimates.get("d_pln")

        # Pull eta as numpy for both the formula and the label.
        if eta_capture is not None:
            eta_arr = np.asarray(eta_capture, dtype=float).reshape(-1)
            mean_eta = float(np.mean(eta_arr))
        else:
            eta_arr = None
            mean_eta = None

        # Try the per-cell MAP path first. Laplace results store
        # the per-cell log-rate MAP under ``x_loc`` (shape
        # ``(n_cells, G)``). VAE results don't expose this directly
        # at the results-object level so we fall through.
        x_loc = getattr(results, "x_loc", None)
        x_arr: np.ndarray | None = None
        if x_loc is not None:
            x_loc_np = np.asarray(x_loc, dtype=float)
            if x_loc_np.ndim == 2 and x_loc_np.shape[0] > 1:
                x_arr = x_loc_np

        if x_arr is not None:
            # Per-cell MAP path: directly evaluate
            #    pred_g = mean_c[ exp(x_g^(c) - eta_c) ].
            # No LogN moment correction: ``x_loc`` is a fitted MAP,
            # not a prior sample, so the per-cell exponential is
            # already at the right scale for informative likelihood.
            # See the design comment above.
            if eta_arr is not None:
                if eta_arr.shape[0] == x_arr.shape[0]:
                    x_eff = x_arr - eta_arr[:, None]
                else:
                    # Defensive: shape mismatch ⇒ apply the average
                    # capture factor instead of per-cell. Falls back
                    # to the route-2 form for the eta term.
                    log_capture = float(np.log(np.mean(np.exp(-eta_arr))))
                    x_eff = x_arr + log_capture
            else:
                x_eff = x_arr
            pred_mean = np.exp(x_eff).mean(axis=0)
        else:
            # Population-level fallback (VAE).
            try:
                distributions = results.get_distributions(
                    backend="numpyro", split=False
                )
            except (TypeError, AttributeError):
                distributions = results.get_distributions()
            lambda_dist = distributions.get("lambda_rate")
            if lambda_dist is None:
                console.print(
                    "[yellow]Skipping mean calibration: ``lambda_rate`` "
                    "(LowRankPoissonLogNormal) unavailable for this PLN "
                    "result.[/yellow]"
                )
                return None
            pop_mean_bio = np.asarray(lambda_dist.mean, dtype=float)
            if eta_arr is not None:
                mean_capture = float(np.mean(np.exp(-eta_arr)))
            else:
                mean_capture = 1.0
            pred_mean = pop_mean_bio * mean_capture

        _annotations = []
        if mean_eta is not None:
            _annotations.append(f"$\\bar{{\\eta}} = {mean_eta:.4f}$")
        _annotations.append("PLN (log-rate)")

        obs_mean = np.mean(np.asarray(counts, dtype=float), axis=0)
        return {
            "mode": "single",
            "ds_results": None,
            "obs_mean": obs_mean,
            "pred_mean": pred_mean,
            "is_mixture": False,
            "annotations": _annotations,
        }

    # ---- TwoState path: predicted mean = mu (mean-preserving) ---------------
    # All four TwoState parameterizations (natural / ratio / mean_fano /
    # moment_delta) sample ``mu`` as the per-gene mean expression and are
    # mean-preserving by construction: E[u_gc] = mu_g (no capture) or
    # mu_g · ν^(c) (VCP).  We can therefore read mu directly from the
    # MAP and skip the NB-family (r, p) extraction below.
    _base_model = getattr(results.model_config, "base_model", None)
    if _base_model in ("twostate", "twostatevcp"):
        _ts_targets = ["mu"]
        if _base_model == "twostatevcp":
            _ts_targets.append("p_capture")
        _ts_map = _get_map_estimates_for_plot(
            results, counts=counts, targets=_ts_targets
        )
        mu = _ts_map.get("mu")
        if mu is None:
            console.print(
                "[yellow]Skipping mean calibration: mu unavailable in "
                "TwoState MAP estimates.[/yellow]"
            )
            return None

        mu_arr = np.asarray(mu, dtype=float).reshape(-1)
        p_capture_ts = _ts_map.get("p_capture")
        _annotations = []
        if p_capture_ts is not None:
            mean_nu = float(np.mean(np.asarray(p_capture_ts, dtype=float)))
            _annotations.append(f"$\\bar{{\\nu}} = {mean_nu:.4f}$")
            pred_mean = mu_arr * mean_nu
        else:
            pred_mean = mu_arr
        _annotations.append("TwoState (mean-preserving)")

        obs_mean = np.mean(np.asarray(counts, dtype=float), axis=0)
        return {
            "mode": "single",
            "ds_results": None,
            "obs_mean": obs_mean,
            "pred_mean": pred_mean,
            "is_mixture": False,
            "annotations": _annotations,
        }

    # ---- NB-family path: predicted mean = r * p / (1 - p) -------------------
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

"""Differential-expression diagnostic plotting with CLR/BIO mode selection.

This module exposes four DE plotting entrypoints that all accept a
``ScribeDEResults``-style object and a shared ``mode`` selector:

- ``"clr"``: compositional CLR metrics
- ``"bio"``: biological LFC metrics
- ``"all"``: combined multi-panel figure with CLR and biological panels

The plotting API is dual-mode via :func:`scribe.viz._interactive.plot_function`,
so these functions work consistently in both notebook and CLI contexts.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scribe

from ._interactive import (
    _create_or_validate_grid_axes,
    _create_or_validate_single_axis,
    plot_function,
)

# Keep plotting numerically stable when taking -log10 on tiny probabilities.
_LOG10_EPS = 1e-30

# Shared type alias for mode validation and editor autocompletion.
DEPlotMode = Literal["clr", "bio", "all"]


def _validate_mode(mode: str) -> DEPlotMode:
    """Validate and normalize the DE plotting mode.

    Parameters
    ----------
    mode : str
        Requested mode value. Valid options are ``"clr"``, ``"bio"``,
        and ``"all"``.

    Returns
    -------
    {"clr", "bio", "all"}
        Normalized mode string.

    Raises
    ------
    ValueError
        Raised when ``mode`` is not one of the supported values.
    """
    normalized = str(mode).strip().lower()
    if normalized not in {"clr", "bio", "all"}:
        raise ValueError(
            "Invalid mode. Expected one of {'clr', 'bio', 'all'}, "
            f"got {mode!r}."
        )
    return normalized  # type: ignore[return-value]


def _compute_lfsr_threshold(
    score_values: np.ndarray,
    target_pefp: float | None,
) -> float | None:
    """Compute an lfsr threshold for PEFP control when requested.

    Parameters
    ----------
    score_values : numpy.ndarray
        Per-gene error-score vector where smaller values indicate stronger
        DE evidence.
    target_pefp : float or None
        Desired posterior expected false positive proportion. When ``None``,
        no threshold is computed.

    Returns
    -------
    float or None
        Threshold selected by :func:`scribe.de.find_lfsr_threshold`, or
        ``None`` when ``target_pefp`` is not provided.
    """
    if target_pefp is None:
        return None
    return float(
        scribe.de.find_lfsr_threshold(
            score_values, target_pefp=float(target_pefp)
        )
    )


def _resolve_is_de_mask(
    df,
    *,
    is_de_column: str,
    score_column: str,
    target_pefp: float | None,
) -> tuple[np.ndarray, float | None]:
    """Resolve DE calls and optional score threshold from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DE dataframe holding score columns and optional boolean call columns.
    is_de_column : str
        Column name used for boolean DE calls.
    score_column : str
        Column used for thresholding when PEFP control is requested.
    target_pefp : float or None
        Optional PEFP target for deriving a threshold and call mask.

    Returns
    -------
    tuple of (numpy.ndarray, float or None)
        Boolean DE mask and threshold value.
    """
    score_values = np.asarray(df[score_column], dtype=float)
    threshold = _compute_lfsr_threshold(score_values, target_pefp=target_pefp)

    # Prefer explicit call columns when present; otherwise derive from scores.
    if is_de_column in df.columns:
        is_de = np.asarray(df[is_de_column], dtype=bool)
    elif threshold is not None:
        is_de = np.asarray(score_values < threshold, dtype=bool)
    else:
        is_de = np.zeros(score_values.shape[0], dtype=bool)
    return is_de, threshold


def _extract_clr_df(
    de_results,
    *,
    tau: float,
    target_pefp: float | None,
    use_lfsr_tau: bool,
):
    """Extract CLR-family DE metrics from a results object.

    Parameters
    ----------
    de_results : object
        Differential-expression results object exposing ``to_dataframe``.
    tau : float
        Practical significance threshold for CLR summaries.
    target_pefp : float or None
        Optional PEFP target used to request a CLR call column.
    use_lfsr_tau : bool
        Whether thresholding should use ``lfsr_tau``.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing CLR columns in prefixed naming mode.
    """
    return de_results.to_dataframe(
        tau=float(tau),
        target_pefp=target_pefp,
        use_lfsr_tau=bool(use_lfsr_tau),
        metrics="clr",
        column_naming="prefixed",
    )


def _extract_bio_df(
    de_results,
    *,
    tau_lfc: float,
    target_pefp: float | None,
    use_lfsr_tau: bool,
):
    """Extract biological LFC and auxiliary metrics from a results object.

    Parameters
    ----------
    de_results : object
        Differential-expression results object exposing ``to_dataframe``.
    tau_lfc : float
        Practical significance threshold for biological LFC summaries.
    target_pefp : float or None
        Optional PEFP target used to request a biological-LFC call column.
    use_lfsr_tau : bool
        Whether biological call thresholding should use ``lfsr_tau``.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing biological LFC and auxiliary columns in prefixed
        naming mode.
    """
    return de_results.to_dataframe(
        metrics=("bio_lfc", "bio_aux"),
        tau_lfc=float(tau_lfc),
        target_pefp_lfc=target_pefp,
        use_lfsr_tau_lfc=bool(use_lfsr_tau),
        column_naming="prefixed",
    )


def _resolve_clr_mean_expression(
    df, de_results
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve CLR mean-expression vectors for A/B conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        CLR dataframe potentially containing explicit mean-expression columns.
    de_results : object
        DE results object that may carry ``mu_map_A`` and ``mu_map_B``.

    Returns
    -------
    tuple of numpy.ndarray
        Mean-expression vectors ``(mu_A, mu_B)`` aligned to the active gene set.

    Raises
    ------
    ValueError
        Raised when CLR mean expression cannot be resolved from either the
        dataframe or the DE results object.
    """
    if {"clr_mean_expression_A", "clr_mean_expression_B"}.issubset(df.columns):
        return (
            np.asarray(df["clr_mean_expression_A"], dtype=float),
            np.asarray(df["clr_mean_expression_B"], dtype=float),
        )

    # Fallback for callers that pass DE objects without pre-exported CLR means.
    mu_A = getattr(de_results, "mu_map_A", None)
    mu_B = getattr(de_results, "mu_map_B", None)
    if mu_A is None or mu_B is None:
        raise ValueError(
            "CLR mean-expression columns are unavailable. Provide a DE object "
            "with `mu_map_A`/`mu_map_B` or precompute those columns."
        )

    mu_A = np.asarray(mu_A, dtype=float).ravel()
    mu_B = np.asarray(mu_B, dtype=float).ravel()

    # Respect active masks so vectors align to the exported dataframe rows.
    mask = getattr(de_results, "_gene_mask", None)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).ravel()
        if mask_arr.shape[0] == mu_A.shape[0]:
            mu_A = mu_A[mask_arr]
            mu_B = mu_B[mask_arr]

    if mu_A.shape[0] != len(df) or mu_B.shape[0] != len(df):
        raise ValueError(
            "Resolved CLR mean-expression vectors do not match DE dataframe "
            f"length ({mu_A.shape[0]}, {mu_B.shape[0]} vs {len(df)})."
        )
    return mu_A, mu_B


def _scatter_de_split(
    ax, x_values, y_values, is_de, *, x_label, y_label, title
):
    """Render DE-aware scatter points with consistent styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    x_values : array-like
        X-axis values.
    y_values : array-like
        Y-axis values.
    is_de : array-like of bool
        Boolean DE-call mask.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    title : str
        Panel title.
    """
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    de_mask = np.asarray(is_de, dtype=bool)

    ax.scatter(
        x_arr[~de_mask], y_arr[~de_mask], s=6, alpha=0.3, color="lightgray"
    )
    ax.scatter(x_arr[de_mask], y_arr[de_mask], s=8, alpha=0.7, color="darkred")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


@plot_function(
    suffix="de_mean_expression",
    save_label="DE mean-expression plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_de_mean_expression(
    de_results,
    *,
    ctx,
    viz_cfg=None,
    mode: DEPlotMode = "clr",
    tau: float = 0.0,
    tau_lfc: float = 0.0,
    target_pefp: float | None = 0.05,
    use_lfsr_tau: bool = True,
    title_suffix: str | None = None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot DE mean expression for CLR, biological LFC, or both.

    Parameters
    ----------
    de_results : object
        Differential-expression results object with ``to_dataframe`` support.
    viz_cfg : OmegaConf or None, optional
        Visualization config passed for API consistency.
    mode : {"clr", "bio", "all"}, default="clr"
        Metric family to plot. ``"all"`` draws side-by-side CLR and biological
        panels.
    tau : float, default=0.0
        Practical threshold forwarded to CLR dataframe export.
    tau_lfc : float, default=0.0
        Practical threshold forwarded to biological LFC dataframe export.
    target_pefp : float or None, default=0.05
        Optional PEFP target used to derive/resolve DE call masks.
    use_lfsr_tau : bool, default=True
        Whether call thresholding uses ``lfsr_tau`` instead of ``lfsr``.
    title_suffix : str or None, optional
        Optional suffix appended to panel titles.
    figsize : tuple, optional
        Figure size override.
    fig : matplotlib.figure.Figure, optional
        Existing figure host.
    axes : array-like of Axes, optional
        Existing axes container (1 axis for single mode, 2 axes for ``all``).
    ax : matplotlib.axes.Axes, optional
        Single-axis handle for non-``all`` modes.

    Returns
    -------
    PlotResult
        Wrapped plotting result with figure, axes, and output metadata.
    """
    del viz_cfg  # Kept for decorator-compatible public signature.
    resolved_mode = _validate_mode(mode)
    title_tail = f" ({title_suffix})" if title_suffix else ""

    if resolved_mode == "all":
        if ax is not None:
            raise ValueError(
                "mode='all' requires `fig`/`axes`, not a single `ax`."
            )
        fig, _, axes_flat = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=2,
            fig=fig,
            axes=axes,
            figsize=figsize or (11.5, 4.6),
        )
        panels = [("clr", axes_flat[0]), ("bio", axes_flat[1])]
    else:
        fig, panel_ax = _create_or_validate_single_axis(
            fig=fig,
            ax=ax,
            axes=axes,
            figsize=figsize or (6.0, 5.5),
        )
        axes_flat = [panel_ax]
        panels = [(resolved_mode, panel_ax)]

    for panel_mode, panel_ax in panels:
        if panel_mode == "clr":
            clr_df = _extract_clr_df(
                de_results,
                tau=tau,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            clr_is_de, _ = _resolve_is_de_mask(
                clr_df,
                is_de_column="clr_is_de",
                score_column="clr_lfsr_tau" if use_lfsr_tau else "clr_lfsr",
                target_pefp=target_pefp,
            )
            mu_A, mu_B = _resolve_clr_mean_expression(clr_df, de_results)
            _scatter_de_split(
                panel_ax,
                mu_A,
                mu_B,
                clr_is_de,
                x_label="CLR mean expression (A)",
                y_label="CLR mean expression (B)",
                title=f"Mean expression (CLR){title_tail}",
            )
        else:
            bio_df = _extract_bio_df(
                de_results,
                tau_lfc=tau_lfc,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            bio_is_de, _ = _resolve_is_de_mask(
                bio_df,
                is_de_column="bio_lfc_is_de",
                score_column=(
                    "bio_lfc_lfsr_tau" if use_lfsr_tau else "bio_lfc_lfsr"
                ),
                target_pefp=target_pefp,
            )
            _scatter_de_split(
                panel_ax,
                bio_df["bio_mu_A_mean"],
                bio_df["bio_mu_B_mean"],
                bio_is_de,
                x_label="Biological mean expression (A)",
                y_label="Biological mean expression (B)",
                title=f"Mean expression (BIO){title_tail}",
            )

        # Identity line helps compare agreement between condition means.
        xlim = panel_ax.get_xlim()
        ylim = panel_ax.get_ylim()
        line_min = min(xlim[0], ylim[0])
        line_max = max(xlim[1], ylim[1])
        panel_ax.plot(
            [line_min, line_max], [line_min, line_max], "k--", linewidth=1.0
        )
        panel_ax.set_xscale("log")
        panel_ax.set_yscale("log")

    return (
        fig,
        axes_flat,
        len(axes_flat),
        {"suffix": f"de_mean_expression_{resolved_mode}"},
    )


@plot_function(
    suffix="de_volcano",
    save_label="DE volcano plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_de_volcano(
    de_results,
    *,
    ctx,
    viz_cfg=None,
    mode: DEPlotMode = "clr",
    tau: float = 0.0,
    tau_lfc: float = 0.0,
    target_pefp: float | None = 0.05,
    use_lfsr_tau: bool = True,
    title_suffix: str | None = None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot DE volcano diagnostics for CLR, biological LFC, or both.

    Parameters
    ----------
    de_results : object
        Differential-expression results object with ``to_dataframe`` support.
    viz_cfg : OmegaConf or None, optional
        Visualization config passed for API consistency.
    mode : {"clr", "bio", "all"}, default="clr"
        Metric family to plot. ``"all"`` draws side-by-side CLR and biological
        volcano panels.
    tau : float, default=0.0
        Practical threshold forwarded to CLR dataframe export.
    tau_lfc : float, default=0.0
        Practical threshold forwarded to biological LFC dataframe export.
    target_pefp : float or None, default=0.05
        Optional PEFP target used to derive threshold call lines.
    use_lfsr_tau : bool, default=True
        Whether call thresholding uses ``lfsr_tau`` instead of ``lfsr``.
    title_suffix : str or None, optional
        Optional suffix appended to panel titles.
    figsize : tuple, optional
        Figure size override.
    fig : matplotlib.figure.Figure, optional
        Existing figure host.
    axes : array-like of Axes, optional
        Existing axes container (1 axis for single mode, 2 axes for ``all``).
    ax : matplotlib.axes.Axes, optional
        Single-axis handle for non-``all`` modes.

    Returns
    -------
    PlotResult
        Wrapped plotting result with figure, axes, and output metadata.
    """
    del viz_cfg
    resolved_mode = _validate_mode(mode)
    title_tail = f" ({title_suffix})" if title_suffix else ""

    if resolved_mode == "all":
        if ax is not None:
            raise ValueError(
                "mode='all' requires `fig`/`axes`, not a single `ax`."
            )
        fig, _, axes_flat = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=2,
            fig=fig,
            axes=axes,
            figsize=figsize or (12.0, 5.2),
        )
        panels = [("clr", axes_flat[0]), ("bio", axes_flat[1])]
    else:
        fig, panel_ax = _create_or_validate_single_axis(
            fig=fig,
            ax=ax,
            axes=axes,
            figsize=figsize or (7.0, 5.2),
        )
        axes_flat = [panel_ax]
        panels = [(resolved_mode, panel_ax)]

    for panel_mode, panel_ax in panels:
        if panel_mode == "clr":
            clr_df = _extract_clr_df(
                de_results,
                tau=tau,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            score_col = "clr_lfsr_tau" if use_lfsr_tau else "clr_lfsr"
            is_de, threshold = _resolve_is_de_mask(
                clr_df,
                is_de_column="clr_is_de",
                score_column=score_col,
                target_pefp=target_pefp,
            )
            x_values = np.asarray(clr_df["clr_delta_mean"], dtype=float)
            y_values = -np.log10(
                np.clip(
                    np.asarray(clr_df[score_col], dtype=float), _LOG10_EPS, 1.0
                )
            )
            panel_ax.set_xlabel("delta mean (CLR)")
            panel_ax.set_title(f"Volcano (CLR){title_tail}")
        else:
            bio_df = _extract_bio_df(
                de_results,
                tau_lfc=tau_lfc,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            score_col = "bio_lfc_lfsr_tau" if use_lfsr_tau else "bio_lfc_lfsr"
            is_de, threshold = _resolve_is_de_mask(
                bio_df,
                is_de_column="bio_lfc_is_de",
                score_column=score_col,
                target_pefp=target_pefp,
            )
            x_values = np.asarray(bio_df["bio_lfc_mean"], dtype=float)
            y_values = -np.log10(
                np.clip(
                    np.asarray(bio_df[score_col], dtype=float), _LOG10_EPS, 1.0
                )
            )
            panel_ax.set_xlabel("biological LFC mean")
            panel_ax.set_title(f"Volcano (BIO){title_tail}")

        # Use directional coloring so sign and call status are both visible.
        panel_ax.scatter(
            x_values[~is_de],
            y_values[~is_de],
            s=5,
            alpha=0.25,
            color="lightgray",
        )
        panel_ax.scatter(
            x_values[is_de & (x_values > 0)],
            y_values[is_de & (x_values > 0)],
            s=8,
            alpha=0.65,
            color="#C62828",
        )
        panel_ax.scatter(
            x_values[is_de & (x_values <= 0)],
            y_values[is_de & (x_values <= 0)],
            s=8,
            alpha=0.65,
            color="#1565C0",
        )
        if threshold is not None:
            panel_ax.axhline(
                -np.log10(max(threshold, _LOG10_EPS)),
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        panel_ax.set_ylabel(r"-log10(lfsr)")

    return (
        fig,
        axes_flat,
        len(axes_flat),
        {"suffix": f"de_volcano_{resolved_mode}"},
    )


@plot_function(
    suffix="de_evidence",
    save_label="DE evidence plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_de_evidence(
    de_results,
    *,
    ctx,
    viz_cfg=None,
    mode: DEPlotMode = "clr",
    tau: float = 0.0,
    tau_lfc: float = 0.0,
    target_pefp: float | None = 0.05,
    use_lfsr_tau: bool = True,
    title_suffix: str | None = None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot DE evidence distributions (lfsr and effect probability).

    Parameters
    ----------
    de_results : object
        Differential-expression results object with ``to_dataframe`` support.
    viz_cfg : OmegaConf or None, optional
        Visualization config passed for API consistency.
    mode : {"clr", "bio", "all"}, default="clr"
        Metric family to plot. ``"all"`` renders a 2x2 panel grid:
        CLR row on top and biological row on bottom.
    tau : float, default=0.0
        Practical threshold forwarded to CLR dataframe export.
    tau_lfc : float, default=0.0
        Practical threshold forwarded to biological LFC dataframe export.
    target_pefp : float or None, default=0.05
        Optional PEFP target used to derive threshold markers.
    use_lfsr_tau : bool, default=True
        Whether call thresholding uses ``lfsr_tau`` instead of ``lfsr``.
    title_suffix : str or None, optional
        Optional suffix appended to panel titles.
    figsize : tuple, optional
        Figure size override.
    fig : matplotlib.figure.Figure, optional
        Existing figure host.
    axes : array-like of Axes, optional
        Existing axes container (2 axes for single mode, 4 for ``all``).
    ax : matplotlib.axes.Axes, optional
        Not supported because evidence plots are always multi-panel.

    Returns
    -------
    PlotResult
        Wrapped plotting result with figure, axes, and output metadata.
    """
    del viz_cfg
    resolved_mode = _validate_mode(mode)
    if ax is not None:
        raise ValueError(
            "Evidence plots are multi-panel; provide `fig` or `axes`."
        )

    title_tail = f" ({title_suffix})" if title_suffix else ""
    if resolved_mode == "all":
        fig, axes_grid, axes_flat = _create_or_validate_grid_axes(
            n_rows=2,
            n_cols=2,
            fig=fig,
            axes=axes,
            figsize=figsize or (11.5, 8.2),
        )
        panel_specs = [
            ("clr", axes_grid[0, 0], axes_grid[0, 1]),
            ("bio", axes_grid[1, 0], axes_grid[1, 1]),
        ]
    else:
        fig, axes_grid, axes_flat = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=2,
            fig=fig,
            axes=axes,
            figsize=figsize or (11.5, 4.2),
        )
        panel_specs = [(resolved_mode, axes_grid[0, 0], axes_grid[0, 1])]

    for panel_mode, ax_lfsr, ax_peff in panel_specs:
        if panel_mode == "clr":
            clr_df = _extract_clr_df(
                de_results,
                tau=tau,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            score_col = "clr_lfsr_tau" if use_lfsr_tau else "clr_lfsr"
            score_vals = np.clip(
                np.asarray(clr_df[score_col], dtype=float), 0.0, 1.0
            )
            peff_vals = np.clip(
                np.asarray(clr_df["clr_prob_effect"], dtype=float), 0.0, 1.0
            )
            threshold = _compute_lfsr_threshold(
                score_vals, target_pefp=target_pefp
            )
            label_prefix = "CLR"
        else:
            bio_df = _extract_bio_df(
                de_results,
                tau_lfc=tau_lfc,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            score_col = "bio_lfc_lfsr_tau" if use_lfsr_tau else "bio_lfc_lfsr"
            score_vals = np.clip(
                np.asarray(bio_df[score_col], dtype=float), 0.0, 1.0
            )
            peff_vals = np.clip(
                np.asarray(bio_df["bio_lfc_prob_effect"], dtype=float), 0.0, 1.0
            )
            threshold = _compute_lfsr_threshold(
                score_vals, target_pefp=target_pefp
            )
            label_prefix = "BIO"

        ax_lfsr.hist(
            score_vals,
            bins=np.linspace(0.0, 1.0, 51),
            color="#5C6BC0",
            alpha=0.85,
        )
        if threshold is not None:
            ax_lfsr.axvline(
                threshold, color="crimson", linestyle="--", linewidth=1.0
            )
        ax_lfsr.set_title(f"{label_prefix} lfsr distribution{title_tail}")
        ax_lfsr.set_xlabel("lfsr")
        ax_lfsr.set_ylabel("count")

        ax_peff.hist(
            peff_vals,
            bins=np.linspace(0.0, 1.0, 51),
            color="#26A69A",
            alpha=0.85,
        )
        if threshold is not None:
            ax_peff.axvline(
                1.0 - threshold, color="crimson", linestyle="--", linewidth=1.0
            )
        ax_peff.set_title(f"{label_prefix} effect probability{title_tail}")
        ax_peff.set_xlabel("P(|effect| > tau)")
        ax_peff.set_ylabel("count")

    return (
        fig,
        axes_flat,
        len(axes_flat),
        {"suffix": f"de_evidence_{resolved_mode}"},
    )


@plot_function(
    suffix="de_ma",
    save_label="DE MA plot",
    save_kwargs={"bbox_inches": "tight", "dpi": 150},
)
def plot_de_ma(
    de_results,
    *,
    ctx,
    viz_cfg=None,
    mode: DEPlotMode = "clr",
    tau: float = 0.0,
    tau_lfc: float = 0.0,
    target_pefp: float | None = 0.05,
    use_lfsr_tau: bool = True,
    title_suffix: str | None = None,
    figsize=None,
    fig=None,
    axes=None,
    ax=None,
):
    """Plot MA diagnostics for CLR, biological LFC, or both.

    Parameters
    ----------
    de_results : object
        Differential-expression results object with ``to_dataframe`` support.
    viz_cfg : OmegaConf or None, optional
        Visualization config passed for API consistency.
    mode : {"clr", "bio", "all"}, default="clr"
        Metric family to plot. ``"all"`` draws side-by-side CLR and biological
        MA panels.
    tau : float, default=0.0
        Practical threshold forwarded to CLR dataframe export.
    tau_lfc : float, default=0.0
        Practical threshold forwarded to biological LFC dataframe export.
    target_pefp : float or None, default=0.05
        Optional PEFP target used to derive/resolve DE call masks.
    use_lfsr_tau : bool, default=True
        Whether call thresholding uses ``lfsr_tau`` instead of ``lfsr``.
    title_suffix : str or None, optional
        Optional suffix appended to panel titles.
    figsize : tuple, optional
        Figure size override.
    fig : matplotlib.figure.Figure, optional
        Existing figure host.
    axes : array-like of Axes, optional
        Existing axes container (1 axis for single mode, 2 axes for ``all``).
    ax : matplotlib.axes.Axes, optional
        Single-axis handle for non-``all`` modes.

    Returns
    -------
    PlotResult
        Wrapped plotting result with figure, axes, and output metadata.
    """
    del viz_cfg
    resolved_mode = _validate_mode(mode)
    title_tail = f" ({title_suffix})" if title_suffix else ""

    if resolved_mode == "all":
        if ax is not None:
            raise ValueError(
                "mode='all' requires `fig`/`axes`, not a single `ax`."
            )
        fig, _, axes_flat = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=2,
            fig=fig,
            axes=axes,
            figsize=figsize or (12.0, 5.0),
        )
        panels = [("clr", axes_flat[0]), ("bio", axes_flat[1])]
    else:
        fig, panel_ax = _create_or_validate_single_axis(
            fig=fig,
            ax=ax,
            axes=axes,
            figsize=figsize or (7.0, 5.0),
        )
        axes_flat = [panel_ax]
        panels = [(resolved_mode, panel_ax)]

    for panel_mode, panel_ax in panels:
        if panel_mode == "clr":
            clr_df = _extract_clr_df(
                de_results,
                tau=tau,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            clr_is_de, _ = _resolve_is_de_mask(
                clr_df,
                is_de_column="clr_is_de",
                score_column="clr_lfsr_tau" if use_lfsr_tau else "clr_lfsr",
                target_pefp=target_pefp,
            )
            mu_A, mu_B = _resolve_clr_mean_expression(clr_df, de_results)
            x_values = np.log1p(0.5 * (mu_A + mu_B))
            y_values = np.asarray(clr_df["clr_delta_mean"], dtype=float)
            x_label = r"log1p(mean $\mu$)"
            y_label = "delta mean CLR"
            title = f"MA plot (CLR){title_tail}"
        else:
            bio_df = _extract_bio_df(
                de_results,
                tau_lfc=tau_lfc,
                target_pefp=target_pefp,
                use_lfsr_tau=use_lfsr_tau,
            )
            bio_is_de, _ = _resolve_is_de_mask(
                bio_df,
                is_de_column="bio_lfc_is_de",
                score_column=(
                    "bio_lfc_lfsr_tau" if use_lfsr_tau else "bio_lfc_lfsr"
                ),
                target_pefp=target_pefp,
            )
            x_values = np.log1p(
                0.5
                * (
                    np.asarray(bio_df["bio_mu_A_mean"], dtype=float)
                    + np.asarray(bio_df["bio_mu_B_mean"], dtype=float)
                )
            )
            y_values = np.asarray(bio_df["bio_lfc_mean"], dtype=float)
            clr_is_de = bio_is_de
            x_label = r"log1p(mean biological $\mu$)"
            y_label = "biological LFC mean"
            title = f"MA plot (BIO){title_tail}"

        # Keep the same sign-aware coloring used in volcano plots.
        panel_ax.scatter(
            x_values[~clr_is_de],
            y_values[~clr_is_de],
            s=5,
            alpha=0.25,
            color="lightgray",
        )
        panel_ax.scatter(
            x_values[clr_is_de & (y_values > 0)],
            y_values[clr_is_de & (y_values > 0)],
            s=8,
            alpha=0.6,
            color="#C62828",
        )
        panel_ax.scatter(
            x_values[clr_is_de & (y_values <= 0)],
            y_values[clr_is_de & (y_values <= 0)],
            s=8,
            alpha=0.6,
            color="#1565C0",
        )
        panel_ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        panel_ax.set_xlabel(x_label)
        panel_ax.set_ylabel(y_label)
        panel_ax.set_title(title)

    return fig, axes_flat, len(axes_flat), {"suffix": f"de_ma_{resolved_mode}"}


__all__ = [
    "plot_de_mean_expression",
    "plot_de_volcano",
    "plot_de_evidence",
    "plot_de_ma",
]

"""Loss and MCMC diagnostics plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt

from ._common import console
from ._interactive import (
    _create_or_validate_grid_axes,
    _finalize_figure,
    _resolve_render_flags,
)
from .config import _get_config_values
from .dispatch import _get_training_diagnostic_payload


def plot_loss(
    results,
    figs_dir=None,
    cfg=None,
    viz_cfg=None,
    *,
    fig=None,
    ax=None,
    axes=None,
    save=None,
    show=None,
    close=None,
):
    """Plot optimization loss or MCMC diagnostics.

    Parameters
    ----------
    results : object
        Fitted result object exposing training diagnostics.
    figs_dir : str, optional
        Output directory used when ``save`` resolves to ``True``.
    cfg : OmegaConf, optional
        Run configuration used to build default filenames.
    viz_cfg : OmegaConf, optional
        Visualization configuration with optional ``format`` key.
    fig : matplotlib.figure.Figure, optional
        Existing figure used to host multi-panel layouts.
    ax : matplotlib.axes.Axes, optional
        Single-axis input supported only for loss-history mode (2-panel mode
        requires ``fig`` or ``axes``).
    axes : array-like of matplotlib.axes.Axes, optional
        Optional explicit axes for multi-panel layouts.
    save : bool, optional
        Save policy. Defaults to ``True`` when ``figs_dir`` is provided.
    show : bool, optional
        Whether to display the figure interactively.
    close : bool, optional
        Whether to close the figure after rendering.

    Returns
    -------
    PlotResult
        Wrapped result containing the figure, axes, and metadata.
    """
    _fig_owned = fig is None and axes is None
    payload = _get_training_diagnostic_payload(results)
    save, show, close = _resolve_render_flags(
        figs_dir=figs_dir,
        save=save,
        show=show,
        close=close,
    )

    if payload["plot_kind"] == "loss":
        console.print("[dim]Plotting loss history...[/dim]")
        if ax is not None:
            raise ValueError(
                "Loss history plot needs 2 panels; provide `fig` or 2 `axes`."
            )
        fig, _, flat_axes = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=2,
            fig=fig,
            axes=axes,
            figsize=(7.0, 3.0),
        )
        ax_log, ax_linear = flat_axes
        ax_log.plot(payload["loss_history"])
        ax_linear.plot(payload["loss_history"])
        ax_log.set_xlabel("step")
        ax_log.set_ylabel("ELBO loss")
        ax_linear.set_xlabel("step")
        ax_linear.set_ylabel("ELBO loss")
        ax_log.set_yscale("log")
        plot_suffix = "loss"
        save_label = "loss plot"
        n_panels = 2
        used_axes = [ax_log, ax_linear]
    else:
        console.print("[dim]Plotting MCMC diagnostics...[/dim]")
        if ax is not None:
            raise ValueError(
                "MCMC diagnostics plot needs 3 panels; provide `fig` or 3 `axes`."
            )
        fig, _, flat_axes = _create_or_validate_grid_axes(
            n_rows=1,
            n_cols=3,
            fig=fig,
            axes=axes,
            figsize=(11.0, 3.2),
        )
        ax_energy, ax_div, ax_trace = flat_axes

        potential_energy = payload.get("potential_energy")
        if potential_energy is not None and potential_energy.size > 0:
            ax_energy.plot(potential_energy, lw=0.8)
            ax_energy.set_title("Potential Energy")
            ax_energy.set_xlabel("draw")
            ax_energy.set_ylabel("energy")
        else:
            ax_energy.text(
                0.5,
                0.5,
                "No potential_energy",
                ha="center",
                va="center",
            )
            ax_energy.set_axis_off()

        diverging = payload.get("diverging")
        if diverging is not None and diverging.size > 0:
            cumulative_div = np.cumsum(diverging)
            ax_div.plot(cumulative_div, color="tab:red", lw=1.0)
            ax_div.set_title("Cumulative Divergences")
            ax_div.set_xlabel("draw")
            ax_div.set_ylabel("count")
        else:
            ax_div.text(
                0.5,
                0.5,
                "No divergence field",
                ha="center",
                va="center",
            )
            ax_div.set_axis_off()

        trace_by_chain = payload.get("trace_by_chain")
        trace_param_name = payload.get("trace_param_name")
        if trace_by_chain is not None and trace_by_chain.size > 0:
            for chain_idx in range(trace_by_chain.shape[0]):
                ax_trace.plot(
                    trace_by_chain[chain_idx],
                    lw=0.8,
                    alpha=0.9,
                    label=f"chain {chain_idx}",
                )
            ax_trace.set_title(
                f"Trace: {trace_param_name}" if trace_param_name else "Trace"
            )
            ax_trace.set_xlabel("draw")
            ax_trace.set_ylabel("value")
            if trace_by_chain.shape[0] <= 4:
                ax_trace.legend(fontsize=7, frameon=False)
        else:
            ax_trace.text(
                0.5,
                0.5,
                "No chain trace",
                ha="center",
                va="center",
            )
            ax_trace.set_axis_off()

        plot_suffix = "diagnostics"
        save_label = "diagnostics plot"
        n_panels = 3
        used_axes = [ax_energy, ax_div, ax_trace]

    if save:
        output_format = viz_cfg.get("format", "png")
        config_vals = _get_config_values(cfg, results=results)
        fname = (
            f"{config_vals['method']}_"
            f"{config_vals['parameterization'].replace('-', '_')}_"
            f"{config_vals['model_type'].replace('_', '-')}_"
            f"{config_vals['n_components']:02d}components_"
            f"{config_vals['run_size_token']}_{plot_suffix}.{output_format}"
        )
    else:
        fname = None
    fig.tight_layout()
    return _finalize_figure(
        fig=fig,
        axes=used_axes,
        n_panels=n_panels,
        save=save,
        show=show,
        close=close,
        figs_dir=figs_dir,
        filename=fname,
        save_kwargs={"bbox_inches": "tight"},
        save_label=save_label,
        _fig_owned=_fig_owned,
    )

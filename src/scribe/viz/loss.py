"""Loss and MCMC diagnostics plotting."""

import os
import numpy as np
import matplotlib.pyplot as plt

from ._common import console
from .config import _get_config_values
from .dispatch import _get_training_diagnostic_payload


def plot_loss(results, figs_dir, cfg, viz_cfg):
    """Plot and save optimization loss or MCMC diagnostics."""
    payload = _get_training_diagnostic_payload(results)
    if payload["plot_kind"] == "loss":
        console.print("[dim]Plotting loss history...[/dim]")
        fig, (ax_log, ax_linear) = plt.subplots(1, 2, figsize=(7.0, 3))
        ax_log.plot(payload["loss_history"])
        ax_linear.plot(payload["loss_history"])
        ax_log.set_xlabel("step")
        ax_log.set_ylabel("ELBO loss")
        ax_linear.set_xlabel("step")
        ax_linear.set_ylabel("ELBO loss")
        ax_log.set_yscale("log")
        plot_suffix = "loss"
        save_label = "loss plot"
    else:
        console.print("[dim]Plotting MCMC diagnostics...[/dim]")
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
        ax_energy, ax_div, ax_trace = axes

        potential_energy = payload.get("potential_energy")
        if potential_energy is not None and potential_energy.size > 0:
            ax_energy.plot(potential_energy, lw=0.8)
            ax_energy.set_title("Potential Energy")
            ax_energy.set_xlabel("draw")
            ax_energy.set_ylabel("energy")
        else:
            ax_energy.text(
                0.5, 0.5, "No potential_energy",
                ha="center", va="center",
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
                0.5, 0.5, "No divergence field",
                ha="center", va="center",
            )
            ax_div.set_axis_off()

        trace_by_chain = payload.get("trace_by_chain")
        trace_param_name = payload.get("trace_param_name")
        if trace_by_chain is not None and trace_by_chain.size > 0:
            for chain_idx in range(trace_by_chain.shape[0]):
                ax_trace.plot(
                    trace_by_chain[chain_idx],
                    lw=0.8, alpha=0.9,
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
                0.5, 0.5, "No chain trace",
                ha="center", va="center",
            )
            ax_trace.set_axis_off()

        plot_suffix = "diagnostics"
        save_label = "diagnostics plot"

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_"
        f"{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_{plot_suffix}.{output_format}"
    )

    plt.tight_layout()
    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        f"[green]✓[/green] [dim]Saved {save_label} to[/dim] [cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

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
from .config import _get_config_values
from .dispatch import _get_map_estimates_for_plot


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


def plot_capture_anchor(results, counts, figs_dir, cfg, viz_cfg):
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
    str or None
        Saved output path when successful; ``None`` when required inputs for
        the diagnostic are unavailable.
    """
    console.print("[dim]Plotting capture-anchor diagnostic...[/dim]")

    # Resolve expected anchor from configuration and bail out cleanly if absent.
    expected_log_m0 = _resolve_expected_log_m0(cfg)
    if expected_log_m0 is None:
        console.print(
            "[yellow]Skipping capture-anchor plot: could not infer "
            r"$\log(M_0)$ from priors.eta_capture or priors.organism.[/yellow]"
        )
        return None

    # Pull MAP-like parameter estimates and ensure eta parameter is present.
    map_estimates = _get_map_estimates_for_plot(results, counts=counts)
    eta_capture = map_estimates.get("eta_capture")
    if eta_capture is None:
        console.print(
            "[yellow]Skipping capture-anchor plot: eta_capture is unavailable "
            "in MAP estimates.[/yellow]"
        )
        return None
    eta_capture = np.asarray(eta_capture, dtype=float).reshape(-1)

    # Compute per-cell library size and the diagnostic transformed quantity.
    library_size = np.asarray(counts.sum(axis=1), dtype=float).reshape(-1)
    log_library_size = np.log(np.maximum(library_size, 1.0))
    eta_plus_log_lib = eta_capture + log_library_size

    # Read plot options with conservative defaults when absent.
    opts = viz_cfg.get("capture_anchor_opts", {})
    n_bins = int(opts.get("n_bins", 50))
    scatter_size = float(opts.get("scatter_size", 6.0))
    scatter_alpha = float(opts.get("scatter_alpha", 0.35))

    # Create side-by-side diagnostic figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

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

    plt.tight_layout()

    output_format = viz_cfg.get("format", "png")
    config_vals = _get_config_values(cfg, results=results)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['run_size_token']}_capture_anchor.{output_format}"
    )
    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    console.print(
        "[green]✓[/green] [dim]Saved capture-anchor plot to[/dim] "
        f"[cyan]{output_path}[/cyan]"
    )
    plt.close(fig)

    return output_path

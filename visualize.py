"""
visualize.py

Visualize SCRIBE inference results from a model output directory.

This script takes a model output directory as input and generates diagnostic
plots including loss curves, ECDF plots, and posterior predictive checks.
All necessary configuration (data path, model settings, etc.) is automatically
loaded from the stored .hydra/config.yaml file in the output directory.

Typical usage:
    # Basic usage - generates all default plots
    $ python visualize.py outputs/5050mix/svi/variable_capture=true

    # Disable specific plots
    $ python visualize.py outputs/myrun --no-ecdf

    # Enable optional plots
    $ python visualize.py outputs/myrun --ppc --umap --heatmap

    # Generate all plots
    $ python visualize.py outputs/myrun --all

    # Custom PPC settings (5x5 grid with 2000 samples)
    $ python visualize.py outputs/myrun --ppc --ppc-rows 5 --ppc-cols 5 --ppc-samples 2000

    # Recursively process all model directories under a root
    $ python visualize.py outputs/ --recursive --all
"""

import argparse
from omegaconf import OmegaConf
import scribe
import pickle
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from viz_utils import (
    plot_loss,
    plot_ecdf,
    plot_ppc,
    plot_umap,
    plot_correlation_heatmap,
    plot_mixture_ppc,
)

console = Console()

# Suppress scanpy/anndata deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

# ------------------------------------------------------------------------------


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize SCRIBE inference results from a model output directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python visualize.py outputs/5050mix/svi/variable_capture=true

    # Disable specific plots
    python visualize.py outputs/myrun --no-ecdf

    # Enable optional plots
    python visualize.py outputs/myrun --ppc --umap --heatmap

    # Generate all plots
    python visualize.py outputs/myrun --all

    # Custom PPC settings (5x5 grid)
    python visualize.py outputs/myrun --ppc --ppc-rows 5 --ppc-cols 5 --ppc-samples 2000

    # Recursively process all model directories
    python visualize.py outputs/ --recursive --all
        """,
    )

    # Required argument
    parser.add_argument(
        "model_dir",
        help="Path to a model output directory containing scribe_results.pkl "
        "and .hydra/config.yaml. When used with --recursive, this is the "
        "root directory to search.",
    )

    # Plot toggles (default plots)
    parser.add_argument(
        "--no-loss",
        action="store_true",
        help="Disable loss history plot",
    )
    parser.add_argument(
        "--no-ecdf",
        action="store_true",
        help="Disable ECDF plot",
    )

    # Optional plots (disabled by default)
    parser.add_argument(
        "--ppc",
        action="store_true",
        help="Enable posterior predictive check plots",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        help="Enable UMAP projection plot",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Enable correlation heatmap",
    )
    parser.add_argument(
        "--mixture-ppc",
        action="store_true",
        help="Enable mixture model PPC (for mixture models only)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_plots",
        help="Enable all plots (loss, ECDF, PPC, UMAP, heatmap, mixture PPC)",
    )

    # Recursive mode
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search the given directory for model output "
        "directories (containing scribe_results.pkl) and generate "
        "plots for each one found",
    )

    # Plot options
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg", "eps"],
        default="png",
        help="Output format for figures (default: png)",
    )
    parser.add_argument(
        "--ecdf-genes",
        type=int,
        default=25,
        help="Number of genes to show in ECDF plot (default: 25)",
    )
    parser.add_argument(
        "--ppc-rows",
        type=int,
        default=5,
        help="Number of rows in PPC grid (default: 5)",
    )
    parser.add_argument(
        "--ppc-cols",
        type=int,
        default=5,
        help="Number of columns in PPC grid (default: 5)",
    )
    parser.add_argument(
        "--ppc-samples",
        type=int,
        default=1500,
        help="Number of posterior samples for PPC (default: 1500)",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------


def _find_model_dirs(root_dir):
    """
    Recursively search for directories containing ``scribe_results.pkl``.

    Parameters
    ----------
    root_dir : str
        Root directory to search from.

    Returns
    -------
    list of str
        Sorted list of absolute paths to directories that contain a
        ``scribe_results.pkl`` file.
    """
    model_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        if "scribe_results.pkl" in filenames:
            model_dirs.append(os.path.abspath(dirpath))
    return sorted(model_dirs)


# ------------------------------------------------------------------------------


def _process_single_model_dir(model_dir, viz_cfg):
    """
    Run the full visualization pipeline on a single model output directory.

    Loads the stored Hydra configuration, inference results, and original
    dataset from *model_dir*, then generates all diagnostic plots requested
    by *viz_cfg*.

    Parameters
    ----------
    model_dir : str
        Absolute path to a model output directory that must contain
        ``scribe_results.pkl`` and ``.hydra/config.yaml``.
    viz_cfg : DictConfig
        Visualization configuration built from the command-line arguments.

    Returns
    -------
    bool
        ``True`` if the directory was processed successfully, ``False`` if
        it was skipped due to missing files or other errors.
    """
    import numpy as np

    console.print(f"[dim]Model directory:[/dim] [cyan]{model_dir}[/cyan]")

    # ======================================================================
    # Validate Input Directory
    # ======================================================================
    results_file = os.path.join(model_dir, "scribe_results.pkl")
    config_file = os.path.join(model_dir, ".hydra", "config.yaml")

    if not os.path.exists(results_file):
        console.print("[bold red]ERROR: Results file not found![/bold red]")
        console.print(f"[red]   Missing:[/red] [cyan]{results_file}[/cyan]")
        console.print(
            "[yellow]Make sure you've run inference first and the directory "
            "contains scribe_results.pkl[/yellow]"
        )
        return False

    if not os.path.exists(config_file):
        console.print("[bold red]ERROR: Config file not found![/bold red]")
        console.print(f"[red]   Missing:[/red] [cyan]{config_file}[/cyan]")
        console.print(
            "[yellow]The .hydra/config.yaml file is required to load the "
            "original data and settings.[/yellow]"
        )
        return False

    console.print("[green]Model directory validated![/green]")

    # ======================================================================
    # Load Original Configuration
    # ======================================================================
    console.print(
        f"[dim]Loading config from:[/dim] [cyan]{config_file}[/cyan]"
    )
    orig_cfg = OmegaConf.load(config_file)
    console.print("[green]Configuration loaded![/green]")

    # Display key config info
    console.print(
        f"[dim]  Data:[/dim] {orig_cfg.data.get('name', 'unknown')}"
    )
    console.print(
        f"[dim]  Model:[/dim] {orig_cfg.get('model', 'derived from flags')}"
    )
    console.print(
        f"[dim]  Parameterization:[/dim] "
        f"{orig_cfg.get('parameterization', 'unknown')}"
    )
    console.print(
        f"[dim]  Inference:[/dim] "
        f"{orig_cfg.inference.get('method', 'unknown')}"
    )

    # ======================================================================
    # Load Results
    # ======================================================================
    figs_dir = os.path.join(model_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    console.print(f"[dim]Figures directory:[/dim] [cyan]{figs_dir}[/cyan]")

    console.print(
        f"[dim]Loading results from:[/dim] [cyan]{results_file}[/cyan]"
    )
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    console.print("[green]Results loaded![/green]")

    # ======================================================================
    # Load Data
    # ======================================================================
    # Get data path from original config
    data_path = orig_cfg.data.path
    # Handle relative paths (relative to original working directory)
    if not os.path.isabs(data_path):
        # Try relative to current directory first
        if not os.path.exists(data_path):
            # Try relative to model directory's parent
            alt_path = os.path.join(os.path.dirname(model_dir), data_path)
            if os.path.exists(alt_path):
                data_path = alt_path

    console.print(f"[dim]Loading data from:[/dim] [cyan]{data_path}[/cyan]")

    if not os.path.exists(data_path):
        console.print("[bold red]ERROR: Data file not found![/bold red]")
        console.print(f"[red]   Missing:[/red] [cyan]{data_path}[/cyan]")
        console.print(
            "[yellow]The data file from the original config could not be "
            "found. Make sure the data is accessible at the same "
            "path.[/yellow]"
        )
        return False

    # Determine which AnnData layer holds the raw counts.
    # Priority: data-level layer > top-level layer > None (uses .X)
    layer = orig_cfg.data.get("layer", orig_cfg.get("layer", None))

    adata = scribe.data_loader.load_and_preprocess_anndata(
        data_path, orig_cfg.data.get("preprocessing"), return_jax=False
    )

    # Extract the correct count matrix (from layer or .X)
    if layer is not None:
        if layer not in adata.layers:
            console.print(
                f"[bold red]ERROR: Layer '{layer}' not found in "
                f"AnnData![/bold red]"
            )
            console.print(
                f"[yellow]Available layers: "
                f"{list(adata.layers.keys())}[/yellow]"
            )
            return False
        raw = adata.layers[layer]
        console.print(f"[dim]Using layer:[/dim] [cyan]{layer}[/cyan]")
    else:
        raw = adata.X
        console.print("[dim]Using layer:[/dim] [cyan].X[/cyan]")

    counts = np.asarray(raw.toarray() if hasattr(raw, "toarray") else raw)
    console.print(
        f"[green]Data loaded![/green] [dim]Shape:[/dim] {counts.shape}"
    )

    # ======================================================================
    # Generate Plots
    # ======================================================================
    plots_generated = []

    if viz_cfg.loss:
        console.print("[dim]Generating loss history plot...[/dim]")
        try:
            plot_loss(results, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("loss")
            console.print("[green]  Loss plot saved[/green]")
        except Exception as e:
            console.print(
                f"[red]  Failed to generate loss plot: {e}[/red]"
            )

    if viz_cfg.ecdf:
        console.print("[dim]Generating ECDF plot...[/dim]")
        try:
            plot_ecdf(counts, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("ECDF")
            console.print("[green]  ECDF plot saved[/green]")
        except Exception as e:
            console.print(
                f"[red]  Failed to generate ECDF plot: {e}[/red]"
            )

    if viz_cfg.ppc:
        console.print(
            "[dim]Generating posterior predictive check plots...[/dim]"
        )
        try:
            plot_ppc(results, counts, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("PPC")
            console.print("[green]  PPC plots saved[/green]")
        except Exception as e:
            console.print(
                f"[red]  Failed to generate PPC plots: {e}[/red]"
            )

    if viz_cfg.umap:
        console.print("[dim]Generating UMAP projection plot...[/dim]")
        try:
            plot_umap(results, counts, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("UMAP")
            console.print("[green]  UMAP plot saved[/green]")
        except Exception as e:
            console.print(
                f"[red]  Failed to generate UMAP plot: {e}[/red]"
            )

    if viz_cfg.heatmap:
        console.print("[dim]Generating correlation heatmap...[/dim]")
        try:
            plot_correlation_heatmap(
                results, counts, figs_dir, orig_cfg, viz_cfg
            )
            plots_generated.append("heatmap")
            console.print("[green]  Heatmap saved[/green]")
        except Exception as e:
            console.print(
                f"[red]  Failed to generate heatmap: {e}[/red]"
            )

    # Check for mixture model before generating mixture PPC.
    # Prefer results.n_components (handles auto-inferred from annotations)
    # over the config value (which may be null when inferred at runtime).
    _res_nc = getattr(results, "n_components", None)
    _cfg_nc = orig_cfg.get("n_components")
    _nc = _res_nc if _res_nc is not None else _cfg_nc
    is_mixture = _nc is not None and _nc > 1
    if viz_cfg.mixture_ppc:
        if is_mixture:
            console.print("[dim]Generating mixture model PPC...[/dim]")
            try:
                plot_mixture_ppc(
                    results, counts, figs_dir, orig_cfg, viz_cfg
                )
                plots_generated.append("mixture PPC")
                console.print("[green]  Mixture PPC saved[/green]")
            except Exception as e:
                console.print(
                    f"[red]  Failed to generate mixture PPC: {e}[/red]"
                )
        else:
            console.print(
                "[yellow]  Skipping mixture PPC "
                "(not a mixture model)[/yellow]"
            )

    # ======================================================================
    # Completion Summary
    # ======================================================================
    plots_str = ", ".join(plots_generated) if plots_generated else "None"
    console.print(
        f"[dim]Plots generated:[/dim] [bold]{plots_str}[/bold]"
    )
    console.print(f"[dim]Figures directory:[/dim] [cyan]{figs_dir}[/cyan]")
    return True


# ------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    console.print()
    console.print(
        Panel.fit(
            "[bold bright_magenta]SCRIBE VISUALIZATION PIPELINE[/bold bright_magenta]",
            border_style="bright_magenta",
        )
    )

    # ======================================================================
    # Setup Visualization Config
    # ======================================================================
    console.print("[dim]Setting up matplotlib style...[/dim]")
    scribe.viz.matplotlib_style()

    # Build viz config from command line arguments
    if args.all_plots:
        viz_cfg = OmegaConf.create(
            {
                "loss": True,
                "ecdf": True,
                "ppc": True,
                "umap": True,
                "heatmap": True,
                "mixture_ppc": True,
                "format": args.format,
                "ecdf_opts": {"n_genes": args.ecdf_genes},
                "ppc_opts": {
                    "n_rows": args.ppc_rows,
                    "n_cols": args.ppc_cols,
                    "n_samples": args.ppc_samples,
                },
            }
        )
    else:
        viz_cfg = OmegaConf.create(
            {
                "loss": not args.no_loss,
                "ecdf": not args.no_ecdf,
                "ppc": args.ppc,
                "umap": args.umap,
                "heatmap": args.heatmap,
                "mixture_ppc": args.mixture_ppc,
                "format": args.format,
                "ecdf_opts": {"n_genes": args.ecdf_genes},
                "ppc_opts": {
                    "n_rows": args.ppc_rows,
                    "n_cols": args.ppc_cols,
                    "n_samples": args.ppc_samples,
                },
            }
        )

    # Display what will be generated
    enabled_plots = []
    if viz_cfg.loss:
        enabled_plots.append("loss")
    if viz_cfg.ecdf:
        enabled_plots.append("ECDF")
    if viz_cfg.ppc:
        enabled_plots.append("PPC")
    if viz_cfg.umap:
        enabled_plots.append("UMAP")
    if viz_cfg.heatmap:
        enabled_plots.append("heatmap")
    if viz_cfg.mixture_ppc:
        enabled_plots.append("mixture PPC")

    console.print(
        f"[dim]Plots to generate:[/dim] {', '.join(enabled_plots)}"
    )
    console.print(f"[dim]Output format:[/dim] {viz_cfg.format}")

    # ======================================================================
    # Dispatch: recursive or single directory
    # ======================================================================
    if args.recursive:
        root_dir = os.path.abspath(args.model_dir)
        if not os.path.exists(root_dir):
            console.print(
                "[bold red]ERROR: Directory does not exist![/bold red]"
            )
            console.print(
                f"[red]   Missing:[/red] [cyan]{root_dir}[/cyan]"
            )
            return

        console.print()
        console.print(
            Panel.fit(
                "[bold bright_cyan]RECURSIVE SEARCH[/bold bright_cyan]",
                border_style="bright_cyan",
            )
        )
        console.print(
            f"[dim]Searching recursively in:[/dim] "
            f"[cyan]{root_dir}[/cyan]"
        )
        model_dirs = _find_model_dirs(root_dir)

        if not model_dirs:
            console.print(
                "[bold yellow]No model directories found containing "
                "scribe_results.pkl[/bold yellow]"
            )
            return

        n_dirs = len(model_dirs)
        console.print(
            f"[green]Found {n_dirs} model "
            f"director{'ies' if n_dirs != 1 else 'y'}[/green]"
        )
        for d in model_dirs:
            console.print(f"[dim]  \u2022 {d}[/dim]")

        succeeded = 0
        failed = 0
        for i, model_dir in enumerate(model_dirs, 1):
            console.print()
            console.print(
                Panel.fit(
                    f"[bold bright_yellow]DIRECTORY "
                    f"{i}/{n_dirs}[/bold bright_yellow]",
                    border_style="bright_yellow",
                )
            )
            try:
                if _process_single_model_dir(model_dir, viz_cfg):
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                console.print(
                    f"[bold red]UNEXPECTED ERROR:[/bold red] "
                    f"[red]{e}[/red]"
                )
                failed += 1

        # Final summary
        console.print()
        console.print(
            Panel.fit(
                "[bold bright_green]RECURSIVE VISUALIZATION "
                "COMPLETE[/bold bright_green]",
                border_style="bright_green",
            )
        )
        console.print(f"[dim]Directories processed:[/dim] {n_dirs}")
        console.print(
            f"[dim]Succeeded:[/dim] [green]{succeeded}[/green]"
        )
        if failed > 0:
            console.print(
                f"[dim]Failed:[/dim] [red]{failed}[/red]"
            )
    else:
        model_dir = os.path.abspath(args.model_dir)
        if not os.path.exists(model_dir):
            console.print(
                "[bold red]ERROR: Model directory does not "
                "exist![/bold red]"
            )
            console.print(
                f"[red]   Missing:[/red] [cyan]{model_dir}[/cyan]"
            )
            return

        _process_single_model_dir(model_dir, viz_cfg)

    console.print()


if __name__ == "__main__":
    main()

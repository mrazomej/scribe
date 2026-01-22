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
    $ python visualize.py outputs/myrun --no-ecdf --no-ppc

    # Enable optional plots
    $ python visualize.py outputs/myrun --umap --heatmap

    # Custom PPC settings (5x5 grid with 2000 samples)
    $ python visualize.py outputs/myrun --ppc-rows 5 --ppc-cols 5 --ppc-samples 2000
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
    python visualize.py outputs/myrun --no-ecdf --no-ppc

    # Enable optional plots
    python visualize.py outputs/myrun --umap --heatmap

    # Custom PPC settings (5x5 grid)
    python visualize.py outputs/myrun --ppc-rows 5 --ppc-cols 5 --ppc-samples 2000
        """,
    )

    # Required argument
    parser.add_argument(
        "model_dir",
        help="Path to the model output directory containing scribe_results.pkl and .hydra/config.yaml",
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
    parser.add_argument(
        "--no-ppc",
        action="store_true",
        help="Disable posterior predictive check plots",
    )

    # Optional plots (disabled by default)
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


def main() -> None:
    args = parse_args()

    console.print()
    console.print(
        Panel.fit(
            "[bold bright_magenta]SCRIBE VISUALIZATION PIPELINE[/bold bright_magenta]",
            border_style="bright_magenta",
        )
    )

    # ==========================================================================
    # Validate Input Directory
    # ==========================================================================
    model_dir = os.path.abspath(args.model_dir)
    console.print(f"[dim]Model directory:[/dim] [cyan]{model_dir}[/cyan]")

    if not os.path.exists(model_dir):
        console.print(
            "[bold red]ERROR: Model directory does not exist![/bold red]"
        )
        console.print(f"[red]   Missing:[/red] [cyan]{model_dir}[/cyan]")
        return

    # Check for required files
    results_file = os.path.join(model_dir, "scribe_results.pkl")
    config_file = os.path.join(model_dir, ".hydra", "config.yaml")

    if not os.path.exists(results_file):
        console.print(
            "[bold red]ERROR: Results file not found![/bold red]"
        )
        console.print(f"[red]   Missing:[/red] [cyan]{results_file}[/cyan]")
        console.print(
            "[yellow]Make sure you've run inference first and the directory "
            "contains scribe_results.pkl[/yellow]"
        )
        return

    if not os.path.exists(config_file):
        console.print(
            "[bold red]ERROR: Config file not found![/bold red]"
        )
        console.print(f"[red]   Missing:[/red] [cyan]{config_file}[/cyan]")
        console.print(
            "[yellow]The .hydra/config.yaml file is required to load the "
            "original data and settings.[/yellow]"
        )
        return

    console.print("[green]Model directory validated![/green]")

    # ==========================================================================
    # Load Original Configuration
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_cyan]LOADING CONFIGURATION[/bold bright_cyan]",
            border_style="bright_cyan",
        )
    )

    console.print(
        f"[dim]Loading config from:[/dim] [cyan]{config_file}[/cyan]"
    )
    orig_cfg = OmegaConf.load(config_file)
    console.print("[green]Configuration loaded![/green]")

    # Display key config info
    console.print(f"[dim]  Data:[/dim] {orig_cfg.data.get('name', 'unknown')}")
    console.print(f"[dim]  Model:[/dim] {orig_cfg.get('model', 'derived from flags')}")
    console.print(f"[dim]  Parameterization:[/dim] {orig_cfg.get('parameterization', 'unknown')}")
    console.print(f"[dim]  Inference:[/dim] {orig_cfg.inference.get('method', 'unknown')}")

    # ==========================================================================
    # Load Results
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_blue]LOADING INFERENCE RESULTS[/bold bright_blue]",
            border_style="bright_blue",
        )
    )

    # Create figs directory
    figs_dir = os.path.join(model_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    console.print(f"[dim]Figures directory:[/dim] [cyan]{figs_dir}[/cyan]")

    # Load scribe results
    console.print(
        f"[dim]Loading results from:[/dim] [cyan]{results_file}[/cyan]"
    )
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    console.print("[green]Results loaded![/green]")

    # ==========================================================================
    # Load Data
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_cyan]LOADING ORIGINAL DATA[/bold bright_cyan]",
            border_style="bright_cyan",
        )
    )

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
        console.print(
            "[bold red]ERROR: Data file not found![/bold red]"
        )
        console.print(f"[red]   Missing:[/red] [cyan]{data_path}[/cyan]")
        console.print(
            "[yellow]The data file from the original config could not be found. "
            "Make sure the data is accessible at the same path.[/yellow]"
        )
        return

    counts = scribe.data_loader.load_and_preprocess_anndata(
        data_path, orig_cfg.data.get("preprocessing")
    )
    console.print(
        f"[green]Data loaded![/green] [dim]Shape:[/dim] {counts.shape}"
    )

    # ==========================================================================
    # Setup Visualization Config
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_magenta]VISUALIZATION SETUP[/bold bright_magenta]",
            border_style="bright_magenta",
        )
    )

    # Set plotting style
    console.print("[dim]Setting up matplotlib style...[/dim]")
    scribe.viz.matplotlib_style()

    # Build viz config from command line arguments
    viz_cfg = OmegaConf.create(
        {
            "loss": not args.no_loss,
            "ecdf": not args.no_ecdf,
            "ppc": not args.no_ppc,
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

    console.print(f"[dim]Plots to generate:[/dim] {', '.join(enabled_plots)}")
    console.print(f"[dim]Output format:[/dim] {viz_cfg.format}")

    # ==========================================================================
    # Generate Plots
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_yellow]GENERATING PLOTS[/bold bright_yellow]",
            border_style="bright_yellow",
        )
    )

    plots_generated = []

    if viz_cfg.loss:
        console.print("[dim]Generating loss history plot...[/dim]")
        try:
            plot_loss(results, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("loss")
            console.print("[green]  Loss plot saved[/green]")
        except Exception as e:
            console.print(f"[red]  Failed to generate loss plot: {e}[/red]")

    if viz_cfg.ecdf:
        console.print("[dim]Generating ECDF plot...[/dim]")
        try:
            plot_ecdf(counts, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("ECDF")
            console.print("[green]  ECDF plot saved[/green]")
        except Exception as e:
            console.print(f"[red]  Failed to generate ECDF plot: {e}[/red]")

    if viz_cfg.ppc:
        console.print("[dim]Generating posterior predictive check plots...[/dim]")
        try:
            plot_ppc(results, counts, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("PPC")
            console.print("[green]  PPC plots saved[/green]")
        except Exception as e:
            console.print(f"[red]  Failed to generate PPC plots: {e}[/red]")

    if viz_cfg.umap:
        console.print("[dim]Generating UMAP projection plot...[/dim]")
        try:
            plot_umap(results, counts, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("UMAP")
            console.print("[green]  UMAP plot saved[/green]")
        except Exception as e:
            console.print(f"[red]  Failed to generate UMAP plot: {e}[/red]")

    if viz_cfg.heatmap:
        console.print("[dim]Generating correlation heatmap...[/dim]")
        try:
            plot_correlation_heatmap(results, figs_dir, orig_cfg, viz_cfg)
            plots_generated.append("heatmap")
            console.print("[green]  Heatmap saved[/green]")
        except Exception as e:
            console.print(f"[red]  Failed to generate heatmap: {e}[/red]")

    # Check for mixture model before generating mixture PPC
    is_mixture = orig_cfg.get("n_components") is not None and orig_cfg.get("n_components", 1) > 1
    if viz_cfg.mixture_ppc:
        if is_mixture:
            console.print("[dim]Generating mixture model PPC...[/dim]")
            try:
                plot_mixture_ppc(results, counts, figs_dir, orig_cfg, viz_cfg)
                plots_generated.append("mixture PPC")
                console.print("[green]  Mixture PPC saved[/green]")
            except Exception as e:
                console.print(f"[red]  Failed to generate mixture PPC: {e}[/red]")
        else:
            console.print(
                "[yellow]  Skipping mixture PPC (not a mixture model)[/yellow]"
            )

    # ==========================================================================
    # Completion Summary
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_green]VISUALIZATION COMPLETE[/bold bright_green]",
            border_style="bright_green",
        )
    )
    console.print(f"[dim]Model directory:[/dim] [cyan]{model_dir}[/cyan]")
    console.print(f"[dim]Figures directory:[/dim] [cyan]{figs_dir}[/cyan]")
    plots_str = ", ".join(plots_generated) if plots_generated else "None"
    console.print(f"[dim]Plots generated:[/dim] [bold]{plots_str}[/bold]")
    console.print()


if __name__ == "__main__":
    main()

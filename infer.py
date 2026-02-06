"""
infer.py

This script serves as the primary entry point for performing probabilistic model
inference within the SCRIBE framework. Designed for execution via Hydra—a
powerful configuration management system—this script automates the loading of
user-specified configurations, manages output organization, and launches the
inference pipeline. It is intended for users who wish to fit probabilistic
models on single-cell RNA-seq data (or similar), with all run settings encoded
in easily reproducible configuration files.


Detailed script explanation:

1. **Hydra-Based Orchestration**: The script is decorated with `@hydra.main`,
   which parses the provided YAML configuration files, initializes the config
   object (`cfg`), and sets up an output directory structure according to
   Hydra's conventions. This ensures all runs are reproducible and output
   files are well-organized by experiment and configuration.

2. **Configuration-Driven Execution**: Users specify data sources, model
   types, parameterizations, inference options, and other run-specific details
   in configuration files (typically under the `conf/` directory). The script
   receives these settings as a nested config structure (`cfg`), which it adapts
   for internal use.

3. **Data Loading**: Using the configured data path and any associated
   preprocessing steps, the script loads the input data with SCRIBE's
   `load_and_preprocess_anndata` routine. This function ensures data is in the
   standardized, preprocessed format required for model fitting.

4. **Argument Preparation for Model Fitting**: The configuration (`cfg`) is
   converted to a plain dictionary for easier manipulation. The new unified
   `model` parameter is used instead of the old boolean flags.

5. **Model Inference Launch**: The core line—calling `scribe.fit()`—initiates
   probabilistic inference using the chosen model, parameterization, inference
   technique (SVI, MCMC, or VAE), and hyperparameters, all defined by the user
   configuration.

6. **Result Serialization**: The inference results are serialized (pickled)
   and saved to the Hydra-created output directory. This ensures users can
   later find results in a structured location (e.g.,
   `outputs/<data>/<model>/<method>/<overrides>/`), supporting downstream
   analysis or visualization.

7. **Reproducibility & Standardization**: By relying on Hydra for
   configuration and output management, the script guarantees reproducible and
   standardized experiment handling. Every run is documented via its config,
   and outputs do not conflict, even when multiple experiments are queued or run
   concurrently.

Typical usage:

    # Basic model (default: mean_odds parameterization)
    $ python infer.py data=<your_data_config>

    # Enable model features with intuitive flags
    $ python infer.py data=singer variable_capture=true
    $ python infer.py data=singer zero_inflation=true
    $ python infer.py data=singer zero_inflation=true variable_capture=true

    # Custom output directory
    $ python infer.py data=singer variable_capture=true output_dir=my_experiment

    # Power users can still use model names directly
    $ python infer.py data=singer model=zinbvcp

    # Override inference parameters
    $ python infer.py data=singer inference.n_steps=100000 inference.batch_size=512

This command will launch SCRIBE's probabilistic inference engine using your
chosen configuration, automatically manage outputs, and save a binary results
file for further diagnostic or biological interpretation.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import scribe
import pickle
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from scribe.data_loader import load_and_preprocess_anndata

console = Console()

# Suppress scanpy/anndata deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

# ==============================================================================
# Custom Hydra Resolvers
# ==============================================================================

# Register custom Hydra resolver to sanitize directory names
# This removes brackets [] which cause issues with Orbax checkpoint library
OmegaConf.register_new_resolver(
    "sanitize_dirname",
    lambda x: x.replace("[", "").replace("]", "") if isinstance(x, str) else x,
    replace=True,
)

# ------------------------------------------------------------------------------


def _resolve_model_type(cfg: DictConfig) -> str:
    """Resolve the model type from config.

    The model type is determined by one of two methods:

    1. **Feature flags (recommended)**: Use `zero_inflation` and
       `variable_capture` boolean flags to automatically derive the model type:
       - zero_inflation=false, variable_capture=false → nbdm
       - zero_inflation=true,  variable_capture=false → zinb
       - zero_inflation=false, variable_capture=true  → nbvcp
       - zero_inflation=true,  variable_capture=true  → zinbvcp

    2. **Direct model specification (power users)**: Set
       `model=nbdm|zinb|nbvcp|zinbvcp` directly. This takes precedence over the
       feature flags.

    Returns the appropriate model type string.
    """
    # Check if model is explicitly set (power user override)
    # Since defaults use `optional model: null`, a non-null model means explicit override
    explicit_model = cfg.get("model", None)
    if explicit_model is not None:
        return explicit_model

    # Derive model from feature flags
    zero_inflation = cfg.get("zero_inflation", False)
    variable_capture = cfg.get("variable_capture", False)

    if zero_inflation and variable_capture:
        return "zinbvcp"
    elif zero_inflation:
        return "zinb"
    elif variable_capture:
        return "nbvcp"
    else:
        return "nbdm"


# ------------------------------------------------------------------------------


def _build_priors_dict(priors_cfg):
    """Convert OmegaConf priors config to dict, filtering out None values."""
    if priors_cfg is None:
        return None
    priors = OmegaConf.to_container(priors_cfg, resolve=True)
    # Filter out None values and convert lists to tuples
    return {
        k: tuple(v) if v is not None else None
        for k, v in priors.items()
        if v is not None
    } or None


# ==============================================================================
# Main Function
# ==============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_blue]SCRIBE PROBABILISTIC INFERENCE PIPELINE[/bold bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print(f"[dim]Working directory:[/dim] [cyan]{os.getcwd()}[/cyan]")
    console.print("\n[bold]Configuration:[/bold]")
    config_yaml = OmegaConf.to_yaml(cfg)
    syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)

    # ==========================================================================
    # Data Loading Section
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_cyan]DATA LOADING[/bold bright_cyan]",
            border_style="bright_cyan",
        )
    )

    data_path = hydra.utils.to_absolute_path(cfg.data.path)
    console.print(f"[dim]Loading data from:[/dim] [cyan]{data_path}[/cyan]")
    counts = load_and_preprocess_anndata(
        data_path, cfg.data.get("preprocessing")
    )
    console.print(
        f"[green]✓[/green] [bold green]Data loaded successfully![/bold green] [dim]Shape:[/dim] {counts.shape}"
    )

    # ==========================================================================
    # Configuration Preparation Section
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_magenta]CONFIGURATION PREPARATION[/bold bright_magenta]",
            border_style="bright_magenta",
        )
    )

    # Extract inference config
    inference_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
    inference_method = inference_cfg.pop("method")

    # Set checkpoint directory for early stopping (automatically within Hydra
    # output)
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    # Note: output_dir is already sanitized by the sanitize_dirname resolver
    # in config.yaml, so brackets are removed before Hydra creates the directory

    if "early_stopping" in inference_cfg and inference_cfg["early_stopping"]:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        inference_cfg["early_stopping"]["checkpoint_dir"] = checkpoint_dir
        console.print(
            f"[dim]Checkpoint directory:[/dim] [cyan]{checkpoint_dir}[/cyan]"
        )

    # Build priors dict (filtering None values)
    priors = _build_priors_dict(cfg.get("priors"))

    # Handle deprecated flags (zero_inflated, variable_capture, mixture_model)
    model_type = _resolve_model_type(cfg)

    # Handle deprecated n_components from mixture_model flag
    n_components = cfg.get("n_components")
    if cfg.get("mixture_model", False) and n_components is None:
        # If mixture_model=true but n_components not set, default to 2
        n_components = 2
        warnings.warn(
            "mixture_model=true without n_components specified. "
            "Defaulting to n_components=2. "
            "Please use n_components=N directly instead of mixture_model flag.",
            DeprecationWarning,
        )

    # Build amortization kwargs if configured
    amortize_capture = False
    capture_hidden_dims = None
    capture_activation = "leaky_relu"
    capture_output_transform = "softplus"
    capture_clamp_min = 0.1
    capture_clamp_max = 50.0

    amort_cfg = cfg.get("amortization", {})
    if amort_cfg:
        capture_cfg = amort_cfg.get("capture", {})
        if capture_cfg and capture_cfg.get("enabled", False):
            amortize_capture = True
            capture_hidden_dims = capture_cfg.get("hidden_dims")
            capture_activation = capture_cfg.get("activation", "leaky_relu")
            capture_output_transform = capture_cfg.get(
                "output_transform", "softplus"
            )
            capture_clamp_min = capture_cfg.get("output_clamp_min", 0.1)
            capture_clamp_max = capture_cfg.get("output_clamp_max", 50.0)

    # Build kwargs for scribe.fit()
    kwargs = {
        # Model configuration
        "model": model_type,
        "parameterization": cfg.parameterization,
        "unconstrained": cfg.unconstrained,
        "n_components": n_components,
        "mixture_params": cfg.get("mixture_params"),
        "guide_rank": cfg.guide_rank,
        "priors": priors,
        # Amortization configuration
        "amortize_capture": amortize_capture,
        "capture_hidden_dims": capture_hidden_dims,
        "capture_activation": capture_activation,
        "capture_output_transform": capture_output_transform,
        "capture_clamp_min": capture_clamp_min,
        "capture_clamp_max": capture_clamp_max,
        # Inference configuration
        "inference_method": inference_method,
        # Data configuration
        "cells_axis": cfg.cells_axis,
        "layer": cfg.layer,
        "seed": cfg.seed,
    }

    # Add inference-specific parameters
    kwargs.update(inference_cfg)

    console.print(f"[dim]Model:[/dim] [bold]{kwargs['model']}[/bold]")
    console.print(
        f"[dim]Parameterization:[/dim] [bold]{kwargs['parameterization']}[/bold]"
    )
    console.print(
        f"[dim]Inference method:[/dim] [bold]{kwargs['inference_method']}[/bold]"
    )
    if kwargs.get("n_components"):
        console.print(
            f"[dim]Mixture components:[/dim] [bold]{kwargs['n_components']}[/bold]"
        )
    if kwargs.get("guide_rank"):
        console.print(
            f"[dim]Guide rank:[/dim] [bold]{kwargs['guide_rank']}[/bold]"
        )
    if kwargs.get("amortize_capture"):
        console.print(
            f"[dim]Amortized capture:[/dim] [bold]{kwargs['capture_hidden_dims']}[/bold] "
            f"[dim]({kwargs['capture_activation']})[/dim]"
        )

    # ==========================================================================
    # Model Inference Section
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_yellow]MODEL INFERENCE[/bold bright_yellow]",
            border_style="bright_yellow",
        )
    )
    console.print("[dim]Starting probabilistic inference...[/dim]")

    # Run the inference using the simplified API
    results = scribe.fit(counts=counts, **kwargs)

    console.print(
        "[green]✓[/green] [bold green]Inference completed successfully![/bold green]"
    )

    # ==========================================================================
    # Results Saving Section
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_green]SAVING RESULTS[/bold bright_green]",
            border_style="bright_green",
        )
    )

    # output_dir was already set earlier when configuring checkpoints
    output_file = os.path.join(output_dir, "scribe_results.pkl")
    console.print(f"[dim]Output directory:[/dim] [cyan]{output_dir}[/cyan]")
    console.print(f"[dim]Saving results to:[/dim] [cyan]{output_file}[/cyan]")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    console.print(
        "[green]✓[/green] [bold green]Results saved successfully![/bold green]"
    )

    # ==========================================================================
    # Visualization Section
    # ==========================================================================
    viz_cfg = cfg.get("viz")
    if viz_cfg:
        console.print()
        console.print(
            Panel.fit(
                "[bold bright_cyan]VISUALIZATION[/bold bright_cyan]",
                border_style="bright_cyan",
            )
        )

        from viz_utils import plot_loss, plot_ecdf, plot_ppc

        # Check if viz=true (enables all) or viz.all=true
        enable_all = (viz_cfg is True) or viz_cfg.get("all", False)

        # Determine which plots to generate
        should_plot_loss = enable_all or viz_cfg.get("loss", False)
        should_plot_ecdf = enable_all or viz_cfg.get("ecdf", False)
        should_plot_ppc = enable_all or viz_cfg.get("ppc", False)

        if should_plot_loss or should_plot_ecdf or should_plot_ppc:
            figs_dir = os.path.join(output_dir, "figs")
            os.makedirs(figs_dir, exist_ok=True)
            console.print(
                f"[dim]Creating figures directory:[/dim] [cyan]{figs_dir}[/cyan]"
            )

            scribe.viz.matplotlib_style()
            console.print("[dim]Setting up matplotlib style...[/dim]")

            if should_plot_loss:
                console.print("[dim]Generating loss history plot...[/dim]")
                plot_loss(results, figs_dir, cfg, viz_cfg)
            if should_plot_ecdf:
                console.print("[dim]Generating ECDF plot...[/dim]")
                plot_ecdf(counts, figs_dir, cfg, viz_cfg)
            if should_plot_ppc:
                console.print(
                    "[dim]Generating posterior predictive check plots...[/dim]"
                )
                plot_ppc(results, counts, figs_dir, cfg, viz_cfg)

            console.print(
                "[green]✓[/green] [bold green]All visualizations completed![/bold green]"
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] [yellow]No plots requested (all visualization options disabled)[/yellow]"
            )
    else:
        console.print()
        console.print(
            Panel.fit(
                "[bold dim]VISUALIZATION SKIPPED[/bold dim]", border_style="dim"
            )
        )
        console.print("[dim]No visualization configuration provided[/dim]")

    # ==========================================================================
    # Completion Summary
    # ==========================================================================
    console.print()
    console.print(
        Panel.fit(
            "[bold bright_green]✓ PIPELINE COMPLETED SUCCESSFULLY![/bold bright_green]",
            border_style="bright_green",
        )
    )
    console.print(f"[dim]Results saved to:[/dim] [cyan]{output_dir}[/cyan]")
    if viz_cfg and (should_plot_loss or should_plot_ecdf or should_plot_ppc):
        console.print(
            f"[dim]Figures saved to:[/dim] [cyan]{os.path.join(output_dir, 'figs')}[/cyan]"
        )
    console.print()


if __name__ == "__main__":
    main()

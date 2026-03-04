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
import glob
import os
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from viz_utils import (
    plot_loss,
    plot_ecdf,
    plot_ppc,
    plot_bio_ppc,
    plot_umap,
    plot_correlation_heatmap,
    plot_mixture_ppc,
    plot_mixture_composition,
    plot_annotation_ppc,
    plot_capture_anchor,
    plot_p_capture_scaling,
)
from viz_utils.memory import cleanup_plot_memory

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

    # Process multiple runs via wildcard pattern
    python visualize.py "outputs/*/zinb*/*" --umap

    # Unquoted wildcard expansion (shell expands to many paths)
    python visualize.py outputs/bleo_study0*/zinbvcp/* --recursive --umap
        """,
    )

    # Required argument
    parser.add_argument(
        "model_dir",
        nargs="+",
        help="Path to a model output directory containing scribe_results.pkl "
        "and .hydra/config.yaml. Supports shell-style wildcards (e.g. "
        "'outputs/*/zinb*/*'). Multiple paths are accepted. When used with "
        "--recursive, matching paths are treated as roots to search.",
    )

    # Plot toggles (default plots)
    parser.add_argument(
        "--no-loss",
        action="store_true",
        default=None,
        help="Disable loss history plot",
    )
    parser.add_argument(
        "--no-ecdf",
        action="store_true",
        default=None,
        help="Disable ECDF plot",
    )

    # Optional plots (disabled by default)
    parser.add_argument(
        "--ppc",
        action="store_true",
        default=None,
        help="Enable posterior predictive check plots",
    )
    parser.add_argument(
        "--bio-ppc",
        action="store_true",
        default=None,
        dest="bio_ppc",
        help="Enable biological PPC plot (NB(r,p) bands + denoised data)",
    )
    parser.add_argument(
        "--umap",
        action="store_true",
        default=None,
        help="Enable UMAP projection plot",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        default=None,
        help="Enable correlation heatmap",
    )
    parser.add_argument(
        "--mixture-ppc",
        action="store_true",
        default=None,
        help="Enable mixture model PPC (for mixture models only)",
    )
    parser.add_argument(
        "--mixture-composition",
        action="store_true",
        default=None,
        help="Enable mixture component composition barplot "
        "(for mixture models only)",
    )
    parser.add_argument(
        "--annotation-ppc",
        action="store_true",
        default=None,
        help="Enable per-annotation PPC (for mixture models with "
        "annotation_key only). Plots each annotation's observed data "
        "against its corresponding component's posterior predictive.",
    )
    parser.add_argument(
        "--capture-anchor",
        action="store_true",
        default=None,
        dest="capture_anchor",
        help="Enable eta capture-anchor diagnostic "
        "(for biology-informed capture prior runs).",
    )
    parser.add_argument(
        "--p-capture-scaling",
        action="store_true",
        default=None,
        dest="p_capture_scaling",
        help="Enable p_capture versus library-size scaling diagnostic "
        "(for VCP models).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_plots",
        help="Enable all plots (loss, ECDF, PPC, bio-PPC, UMAP, heatmap, "
        "mixture PPC, mixture composition, annotation PPC, capture-anchor, "
        "p-capture-scaling)",
    )

    # Recursive mode
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search the given directory for model output "
        "directories (containing scribe_results.pkl) and generate "
        "plots for each one found",
    )

    # Overwrite control
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate plots even if the output files already exist. "
        "Without this flag, existing plots are skipped.",
    )

    # Plot options
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg", "eps"],
        default=None,
        help="Output format for figures (default: png)",
    )
    parser.add_argument(
        "--ecdf-genes",
        type=int,
        default=None,
        help="Number of genes to show in ECDF plot (default: 25)",
    )
    parser.add_argument(
        "--ppc-rows",
        type=int,
        default=None,
        help="Number of rows in PPC grid (default: 5)",
    )
    parser.add_argument(
        "--ppc-cols",
        type=int,
        default=None,
        help="Number of columns in PPC grid (default: 5)",
    )
    parser.add_argument(
        "--ppc-samples",
        type=int,
        default=None,
        help="Number of posterior samples for PPC (default: 1500)",
    )
    parser.add_argument(
        "--umap-ppc-samples",
        type=int,
        default=None,
        help="Number of PPC samples for UMAP overlay (default: 50)",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------


def _load_default_viz_config():
    """Load visualization defaults from ``conf/viz/default.yaml``.

    Returns
    -------
    DictConfig
        Visualization config from the project defaults. If the defaults file is
        unavailable, returns a minimal fallback config with safe defaults.
    """
    defaults_path = os.path.join(
        os.path.dirname(__file__), "conf", "viz", "default.yaml"
    )

    if not os.path.exists(defaults_path):
        console.print(
            "[yellow]⚠[/yellow] [yellow]Could not find conf/viz/default.yaml; "
            "using built-in defaults.[/yellow]"
        )
        return OmegaConf.create(
            {
                "loss": True,
                "ecdf": True,
                "ppc": False,
                "bio_ppc": False,
                "umap": False,
                "heatmap": False,
                "mixture_ppc": False,
                "mixture_composition": False,
                "annotation_ppc": False,
                "capture_anchor": False,
                "p_capture_scaling": False,
                "format": "png",
                "ecdf_opts": {"n_genes": 25},
                "ppc_opts": {"n_rows": 6, "n_cols": 6, "n_samples": 1500},
                "umap_opts": {
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "n_components": 2,
                    "random_state": 42,
                    "batch_size": 1000,
                    "data_color": "dark_blue",
                    "synthetic_color": "dark_red",
                    "cache_umap": True,
                    "target_sum": 1e4,
                    "gene_filter_min_cells": 3,
                    "use_hvg": True,
                    "hvg_n_top_genes": 2000,
                    "hvg_flavor": "seurat",
                    "use_pca": True,
                    "pca_n_comps": 50,
                },
                "heatmap_opts": {
                    "n_genes": 1500,
                    "n_samples": 512,
                    "figsize": 12,
                    "cmap": "RdBu_r",
                },
                "mixture_ppc_opts": {
                    "n_rows": 6,
                    "n_cols": 6,
                    "n_samples": 1500,
                },
                "mixture_composition_opts": {
                    "assignment_batch_size": 512,
                },
                "annotation_ppc_opts": {
                    "n_rows": 5,
                    "n_cols": 5,
                    "n_samples": 1500,
                },
                "capture_anchor_opts": {
                    "n_bins": 50,
                    "scatter_size": 6,
                    "scatter_alpha": 0.35,
                },
                "p_capture_scaling_opts": {
                    "n_bins": 30,
                    "min_cells_per_bin": 5,
                    "assignment_batch_size": 512,
                },
            }
        )

    defaults_cfg = OmegaConf.load(defaults_path)
    # The defaults file is packaged as:
    #   viz:
    #     ...
    return defaults_cfg.get("viz", defaults_cfg)


# ------------------------------------------------------------------------------


def _build_viz_config(args):
    """Build visualization config from defaults + explicit CLI overrides."""
    viz_cfg = _load_default_viz_config()

    # --all forces all plot toggles on while preserving nested option defaults.
    if args.all_plots:
        viz_cfg.loss = True
        viz_cfg.ecdf = True
        viz_cfg.ppc = True
        viz_cfg.bio_ppc = True
        viz_cfg.umap = True
        viz_cfg.heatmap = True
        viz_cfg.mixture_ppc = True
        viz_cfg.mixture_composition = True
        viz_cfg.annotation_ppc = True
        viz_cfg.capture_anchor = True
        viz_cfg.p_capture_scaling = True

    # Apply boolean overrides only when flags are explicitly provided.
    if args.no_loss:
        viz_cfg.loss = False
    if args.no_ecdf:
        viz_cfg.ecdf = False
    if args.ppc:
        viz_cfg.ppc = True
    if args.bio_ppc:
        viz_cfg.bio_ppc = True
    if args.umap:
        viz_cfg.umap = True
    if args.heatmap:
        viz_cfg.heatmap = True
    if args.mixture_ppc:
        viz_cfg.mixture_ppc = True
    if args.mixture_composition:
        viz_cfg.mixture_composition = True
    if args.annotation_ppc:
        viz_cfg.annotation_ppc = True
    if args.capture_anchor:
        viz_cfg.capture_anchor = True
    if args.p_capture_scaling:
        viz_cfg.p_capture_scaling = True

    # Scalar / numeric overrides (only when provided).
    if args.format is not None:
        viz_cfg.format = args.format
    if args.ecdf_genes is not None:
        viz_cfg.ecdf_opts.n_genes = args.ecdf_genes
    if args.ppc_rows is not None:
        viz_cfg.ppc_opts.n_rows = args.ppc_rows
    if args.ppc_cols is not None:
        viz_cfg.ppc_opts.n_cols = args.ppc_cols
    if args.ppc_samples is not None:
        viz_cfg.ppc_opts.n_samples = args.ppc_samples
    if args.umap_ppc_samples is not None:
        viz_cfg.umap_opts.n_ppc_samples = args.umap_ppc_samples

    return viz_cfg


def _has_biology_informed_capture_prior(cfg):
    """Return whether biology-informed eta capture priors are active.

    Parameters
    ----------
    cfg : OmegaConf
        Run configuration loaded from ``.hydra/config.yaml``.

    Returns
    -------
    bool
        ``True`` when any of ``priors.organism``, ``priors.eta_capture``,
        or ``priors.mu_eta`` is set in the run configuration.
    """
    priors_cfg = cfg.get("priors") if hasattr(cfg, "get") else None
    if priors_cfg is None or not hasattr(priors_cfg, "get"):
        return False
    return any(
        priors_cfg.get(key) is not None
        for key in ("organism", "eta_capture", "mu_eta")
    )


def _is_vcp_model(cfg, results):
    """Return whether the current run corresponds to a VCP model.

    Parameters
    ----------
    cfg : OmegaConf
        Run configuration loaded from ``.hydra/config.yaml``.
    results : object
        Loaded results object that may expose ``model_config`` metadata.

    Returns
    -------
    bool
        ``True`` when model metadata indicates variable-capture behavior.
    """
    # Use explicit config flags first, which are cheap and unambiguous.
    model_name = (
        str(cfg.get("model", "")).lower() if hasattr(cfg, "get") else ""
    )
    if "vcp" in model_name:
        return True
    if hasattr(cfg, "get") and cfg.get("variable_capture", False):
        return True

    # Fall back to serialized model config metadata when available.
    model_cfg = getattr(results, "model_config", None)
    if model_cfg is not None:
        if bool(getattr(model_cfg, "uses_variable_capture", False)):
            return True
        base_model = str(getattr(model_cfg, "base_model", "")).lower()
        if "vcp" in base_model:
            return True

    return False


# ------------------------------------------------------------------------------


def _plot_exists(figs_dir, suffix, fmt):
    """
    Check whether output files for a given plot type already exist.

    Parameters
    ----------
    figs_dir : str
        Path to the figures directory.
    suffix : str
        Filename suffix or glob fragment that identifies the plot type
        (e.g. ``"_loss"``, ``"steps_ppc"``, ``"_correlation_heatmap*"``).
        Matched as ``*{suffix}.{fmt}`` via glob.
    fmt : str
        File extension / output format (e.g. ``"png"``).

    Returns
    -------
    bool
        ``True`` if at least one matching file exists.
    """
    pattern = os.path.join(figs_dir, f"*{suffix}.{fmt}")
    return len(glob.glob(pattern)) > 0


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


def _contains_glob_pattern(path_expr):
    """
    Return whether a path expression contains glob wildcards.

    Parameters
    ----------
    path_expr : str
        User-supplied path or path pattern.

    Returns
    -------
    bool
        ``True`` when ``path_expr`` contains at least one glob token
        (``*``, ``?``, or character class ``[...]``).
    """
    return any(token in path_expr for token in ("*", "?", "["))


# ------------------------------------------------------------------------------


def _resolve_model_dirs(path_expr, recursive=False):
    """
    Resolve user path input to concrete model output directories.

    Parameters
    ----------
    path_expr : str
        Path to a model directory or a glob pattern.
    recursive : bool, optional
        When ``True``, treat resolved paths as roots and recursively discover
        all subdirectories containing ``scribe_results.pkl``. When ``False``,
        only directories that themselves contain ``scribe_results.pkl`` are
        returned.

    Returns
    -------
    list of str
        Sorted absolute model-directory paths.
    """
    if _contains_glob_pattern(path_expr):
        candidates = glob.glob(path_expr, recursive=True)
    else:
        candidates = [path_expr]

    resolved = []
    for candidate in candidates:
        abs_candidate = os.path.abspath(candidate)

        # Accept file paths by promoting to their containing directory.
        if os.path.isfile(abs_candidate):
            abs_candidate = os.path.dirname(abs_candidate)

        if not os.path.isdir(abs_candidate):
            continue

        if recursive:
            resolved.extend(_find_model_dirs(abs_candidate))
        else:
            results_file = os.path.join(abs_candidate, "scribe_results.pkl")
            if os.path.exists(results_file):
                resolved.append(abs_candidate)

    return sorted(set(resolved))


# ------------------------------------------------------------------------------


def _process_single_model_dir(model_dir, viz_cfg, overwrite=False):
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
    overwrite : bool, optional
        When ``False`` (the default), plots whose output files already
        exist in the ``figs/`` subdirectory are skipped.  Set to ``True``
        to regenerate all requested plots unconditionally.

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
    console.print(f"[dim]Loading config from:[/dim] [cyan]{config_file}[/cyan]")
    orig_cfg = OmegaConf.load(config_file)
    console.print("[green]Configuration loaded![/green]")

    # Display key config info
    console.print(f"[dim]  Data:[/dim] {orig_cfg.data.get('name', 'unknown')}")
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
        data_path,
        orig_cfg.data.get("preprocessing"),
        return_jax=False,
        subset_column=orig_cfg.data.get("subset_column"),
        subset_value=orig_cfg.data.get("subset_value"),
        filter_obs=orig_cfg.data.get("filter_obs"),
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

    # Extract annotation labels (needed for annotation PPC)
    annotation_key = orig_cfg.get("annotation_key", None)
    cell_labels = None
    if annotation_key is not None:
        if isinstance(annotation_key, str):
            cell_labels = np.array(adata.obs[annotation_key].astype(str))
        else:
            # List of columns → composite labels joined with "__"
            annotation_key = list(annotation_key)
            parts = [adata.obs[k].astype(str) for k in annotation_key]
            combined = parts[0]
            for part in parts[1:]:
                combined = combined + "__" + part
            cell_labels = np.array(combined)
        console.print(
            f"[dim]Annotation key:[/dim] [cyan]{annotation_key}[/cyan] "
            f"({len(np.unique(cell_labels))} unique labels)"
        )

    # ======================================================================
    # Multi-dataset detection
    # ======================================================================
    dataset_key = orig_cfg.data.get("dataset_key") or orig_cfg.get(
        "dataset_key"
    )
    n_datasets = getattr(results.model_config, "n_datasets", None)
    is_multi_dataset = dataset_key is not None and n_datasets is not None

    if is_multi_dataset:
        ds_col = adata.obs[dataset_key].astype("category")
        dataset_names = list(ds_col.cat.categories)
        dataset_codes = np.array(ds_col.cat.codes.values)
        console.print(
            f"[dim]Multi-dataset model detected "
            f"({n_datasets} datasets via '{dataset_key}'):[/dim] "
            f"[cyan]{dataset_names}[/cyan]"
        )
    else:
        # Single-pass: one iteration with no subsetting
        dataset_names = [None]
        dataset_codes = None

    # ======================================================================
    # Generate Plots
    # ======================================================================
    fmt = viz_cfg.format
    uses_bio_capture_prior = _has_biology_informed_capture_prior(orig_cfg)
    is_vcp_model = _is_vcp_model(orig_cfg, results)

    # Global mixture detection is used by p_capture scaling before per-dataset
    # subsetting occurs.
    global_nc = getattr(results, "n_components", None)
    if global_nc is None:
        global_nc = orig_cfg.get("n_components")
    is_global_mixture = global_nc is not None and global_nc > 1

    # Ensure we release plot-related host/GPU memory between heavy stages.
    def _cleanup_after_plot():
        cleanup_plot_memory(results=results, reset_result_caches=True)

    plots_generated = []
    plots_skipped = []

    # ------------------------------------------------------------------
    # Loss plot is global (joint optimisation) -- never per-dataset
    # ------------------------------------------------------------------
    if viz_cfg.loss:
        if not overwrite and _plot_exists(figs_dir, "_loss", fmt):
            plots_skipped.append("loss")
            console.print("[yellow]  Skipping loss (already exists)[/yellow]")
        else:
            # For MCMC runs this branch now renders diagnostics instead of ELBO.
            if hasattr(results, "loss_history"):
                console.print("[dim]Generating loss history plot...[/dim]")
            else:
                console.print("[dim]Generating MCMC diagnostics plot...[/dim]")
            try:
                plot_loss(results, figs_dir, orig_cfg, viz_cfg)
                plots_generated.append("loss")
                console.print("[green]  Loss plot saved[/green]")
            except Exception as e:
                console.print(f"[red]  Failed to generate loss plot: {e}[/red]")
            finally:
                _cleanup_after_plot()

    # ------------------------------------------------------------------
    # p_capture scaling is a global diagnostic over all cells.
    # This runs independently of eta-parameterization and only requires VCP.
    # ------------------------------------------------------------------
    if viz_cfg.p_capture_scaling:
        if not is_vcp_model:
            console.print(
                "[yellow]  Skipping p-capture scaling "
                "(model is not VCP)[/yellow]"
            )
        elif not overwrite and _plot_exists(
            figs_dir, "_p_capture_scaling", fmt
        ):
            plots_skipped.append("p-capture scaling")
            console.print(
                "[yellow]  Skipping p-capture scaling "
                "(already exists)[/yellow]"
            )
        else:
            console.print(
                "[dim]Generating p-capture scaling diagnostic...[/dim]"
            )
            try:
                output_path = plot_p_capture_scaling(
                    results=results,
                    counts=counts,
                    figs_dir=figs_dir,
                    cfg=orig_cfg,
                    viz_cfg=viz_cfg,
                    is_mixture=is_global_mixture,
                    is_multi_dataset=is_multi_dataset,
                    dataset_codes=dataset_codes if is_multi_dataset else None,
                    dataset_names=dataset_names if is_multi_dataset else None,
                )
                if output_path is not None:
                    plots_generated.append("p-capture scaling")
                    console.print(
                        "[green]  p-capture scaling plot saved[/green]"
                    )
                else:
                    plots_skipped.append("p-capture scaling")
            except Exception as e:
                console.print(
                    f"[red]  Failed to generate p-capture scaling plot: "
                    f"{e}[/red]"
                )
            finally:
                _cleanup_after_plot()

    # ------------------------------------------------------------------
    # Per-dataset loop for all remaining plots.
    # For single-dataset models this iterates once with the full data.
    # ------------------------------------------------------------------
    for ds_idx, ds_name in enumerate(dataset_names):
        if ds_name is not None:
            ds_results = results.get_dataset(ds_idx)
            ds_counts = counts[dataset_codes == ds_idx]
            ds_cell_labels = (
                cell_labels[dataset_codes == ds_idx]
                if cell_labels is not None
                else None
            )
            ds_figs_dir = os.path.join(figs_dir, f"dataset_{ds_name}")
            os.makedirs(ds_figs_dir, exist_ok=True)
            ds_label = f" [dataset={ds_name}]"
            console.print(
                f"\n[bold]--- Dataset {ds_idx}: "
                f"{ds_name} ({ds_counts.shape[0]} cells) ---[/bold]"
            )
        else:
            ds_results = results
            ds_counts = counts
            ds_cell_labels = cell_labels
            ds_figs_dir = figs_dir
            ds_label = ""

        if viz_cfg.ecdf:
            if not overwrite and _plot_exists(ds_figs_dir, "_ecdf", fmt):
                plots_skipped.append(f"ECDF{ds_label}")
                console.print(
                    "[yellow]  Skipping ECDF (already exists)[/yellow]"
                )
            else:
                console.print("[dim]Generating ECDF plot...[/dim]")
                try:
                    plot_ecdf(ds_counts, ds_figs_dir, orig_cfg, viz_cfg)
                    plots_generated.append(f"ECDF{ds_label}")
                    console.print("[green]  ECDF plot saved[/green]")
                except Exception as e:
                    console.print(
                        f"[red]  Failed to generate ECDF plot: {e}[/red]"
                    )
                finally:
                    _cleanup_after_plot()

        if viz_cfg.ppc:
            if not overwrite and _plot_exists(ds_figs_dir, "steps_ppc", fmt):
                plots_skipped.append(f"PPC{ds_label}")
                console.print(
                    "[yellow]  Skipping PPC (already exists)[/yellow]"
                )
            else:
                console.print(
                    "[dim]Generating posterior predictive check "
                    "plots...[/dim]"
                )
                try:
                    plot_ppc(
                        ds_results,
                        ds_counts,
                        ds_figs_dir,
                        orig_cfg,
                        viz_cfg,
                    )
                    plots_generated.append(f"PPC{ds_label}")
                    console.print("[green]  PPC plots saved[/green]")
                except Exception as e:
                    console.print(
                        f"[red]  Failed to generate PPC plots: {e}[/red]"
                    )
                finally:
                    _cleanup_after_plot()

        if viz_cfg.bio_ppc:
            if not overwrite and _plot_exists(ds_figs_dir, "_bio_ppc", fmt):
                plots_skipped.append(f"bio-PPC{ds_label}")
                console.print(
                    "[yellow]  Skipping bio-PPC (already exists)[/yellow]"
                )
            else:
                console.print(
                    "[dim]Generating biological PPC (denoised) "
                    "plots...[/dim]"
                )
                try:
                    plot_bio_ppc(
                        ds_results,
                        ds_counts,
                        ds_figs_dir,
                        orig_cfg,
                        viz_cfg,
                    )
                    plots_generated.append(f"bio-PPC{ds_label}")
                    console.print("[green]  Bio-PPC plots saved[/green]")
                except Exception as e:
                    console.print(
                        f"[red]  Failed to generate bio-PPC plots: "
                        f"{e}[/red]"
                    )
                finally:
                    _cleanup_after_plot()

        if viz_cfg.umap:
            if not overwrite and _plot_exists(ds_figs_dir, "_umap", fmt):
                plots_skipped.append(f"UMAP{ds_label}")
                console.print(
                    "[yellow]  Skipping UMAP (already exists)[/yellow]"
                )
            else:
                console.print("[dim]Generating UMAP projection plot...[/dim]")
                try:
                    plot_umap(
                        ds_results,
                        ds_counts,
                        ds_figs_dir,
                        orig_cfg,
                        viz_cfg,
                        force_refit=overwrite,
                    )
                    plots_generated.append(f"UMAP{ds_label}")
                    console.print("[green]  UMAP plot saved[/green]")
                except Exception as e:
                    console.print(
                        f"[red]  Failed to generate UMAP plot: {e}[/red]"
                    )
                finally:
                    _cleanup_after_plot()

        if viz_cfg.heatmap:
            if not overwrite and _plot_exists(
                ds_figs_dir, "_correlation_heatmap*", fmt
            ):
                plots_skipped.append(f"heatmap{ds_label}")
                console.print(
                    "[yellow]  Skipping heatmap (already exists)[/yellow]"
                )
            else:
                console.print("[dim]Generating correlation heatmap...[/dim]")
                try:
                    plot_correlation_heatmap(
                        ds_results,
                        ds_counts,
                        ds_figs_dir,
                        orig_cfg,
                        viz_cfg,
                    )
                    plots_generated.append(f"heatmap{ds_label}")
                    console.print("[green]  Heatmap saved[/green]")
                except Exception as e:
                    console.print(
                        f"[red]  Failed to generate heatmap: {e}[/red]"
                    )
                finally:
                    _cleanup_after_plot()

        # Mixture model detection (use per-dataset view's n_components)
        _res_nc = getattr(ds_results, "n_components", None)
        _cfg_nc = orig_cfg.get("n_components")
        _nc = _res_nc if _res_nc is not None else _cfg_nc
        is_mixture = _nc is not None and _nc > 1

        if viz_cfg.mixture_ppc:
            if is_mixture:
                if not overwrite and _plot_exists(
                    ds_figs_dir, "_mixture_ppc", fmt
                ):
                    plots_skipped.append(f"mixture PPC{ds_label}")
                    console.print(
                        "[yellow]  Skipping mixture PPC "
                        "(already exists)[/yellow]"
                    )
                else:
                    console.print("[dim]Generating mixture model PPC...[/dim]")
                    try:
                        plot_mixture_ppc(
                            ds_results,
                            ds_counts,
                            ds_figs_dir,
                            orig_cfg,
                            viz_cfg,
                        )
                        plots_generated.append(f"mixture PPC{ds_label}")
                        console.print("[green]  Mixture PPC saved[/green]")
                    except Exception as e:
                        console.print(
                            f"[red]  Failed to generate mixture "
                            f"PPC: {e}[/red]"
                        )
                    finally:
                        _cleanup_after_plot()
            else:
                console.print(
                    "[yellow]  Skipping mixture PPC "
                    "(not a mixture model)[/yellow]"
                )

        if viz_cfg.mixture_composition:
            if is_mixture:
                if not overwrite and _plot_exists(
                    ds_figs_dir, "_mixture_composition", fmt
                ):
                    plots_skipped.append(f"mixture composition{ds_label}")
                    console.print(
                        "[yellow]  Skipping mixture composition "
                        "(already exists)[/yellow]"
                    )
                else:
                    console.print(
                        "[dim]Generating mixture composition " "plot...[/dim]"
                    )
                    try:
                        plot_mixture_composition(
                            ds_results,
                            ds_counts,
                            ds_figs_dir,
                            orig_cfg,
                            viz_cfg,
                            cell_labels=ds_cell_labels,
                        )
                        plots_generated.append(f"mixture composition{ds_label}")
                        console.print(
                            "[green]  Mixture composition saved[/green]"
                        )
                    except Exception as e:
                        console.print(
                            f"[red]  Failed to generate mixture "
                            f"composition: {e}[/red]"
                        )
                    finally:
                        _cleanup_after_plot()
            else:
                console.print(
                    "[yellow]  Skipping mixture composition "
                    "(not a mixture model)[/yellow]"
                )

        if viz_cfg.annotation_ppc:
            if is_mixture and ds_cell_labels is not None:
                if not overwrite and _plot_exists(
                    ds_figs_dir, "_annotation_ppc_", fmt
                ):
                    plots_skipped.append(f"annotation PPC{ds_label}")
                    console.print(
                        "[yellow]  Skipping annotation PPC "
                        "(already exists)[/yellow]"
                    )
                else:
                    console.print("[dim]Generating annotation PPC...[/dim]")
                    try:
                        plot_annotation_ppc(
                            ds_results,
                            ds_counts,
                            ds_cell_labels,
                            ds_figs_dir,
                            orig_cfg,
                            viz_cfg,
                        )
                        plots_generated.append(f"annotation PPC{ds_label}")
                        console.print("[green]  Annotation PPC saved[/green]")
                    except Exception as e:
                        console.print(
                            f"[red]  Failed to generate annotation "
                            f"PPC: {e}[/red]"
                        )
                    finally:
                        _cleanup_after_plot()
            elif not is_mixture:
                console.print(
                    "[yellow]  Skipping annotation PPC "
                    "(not a mixture model)[/yellow]"
                )
            else:
                console.print(
                    "[yellow]  Skipping annotation PPC "
                    "(no annotation_key in config)[/yellow]"
                )

        if viz_cfg.capture_anchor:
            if not uses_bio_capture_prior:
                console.print(
                    "[yellow]  Skipping capture-anchor "
                    "(biology-informed capture prior not active)[/yellow]"
                )
            elif not overwrite and _plot_exists(
                ds_figs_dir, "_capture_anchor", fmt
            ):
                plots_skipped.append(f"capture-anchor{ds_label}")
                console.print(
                    "[yellow]  Skipping capture-anchor "
                    "(already exists)[/yellow]"
                )
            else:
                console.print(
                    "[dim]Generating capture-anchor diagnostic...[/dim]"
                )
                try:
                    output_path = plot_capture_anchor(
                        ds_results, ds_counts, ds_figs_dir, orig_cfg, viz_cfg
                    )
                    if output_path is not None:
                        plots_generated.append(f"capture-anchor{ds_label}")
                        console.print(
                            "[green]  Capture-anchor plot saved[/green]"
                        )
                    else:
                        plots_skipped.append(f"capture-anchor{ds_label}")
                except Exception as e:
                    console.print(
                        f"[red]  Failed to generate capture-anchor plot: "
                        f"{e}[/red]"
                    )
                finally:
                    _cleanup_after_plot()

    # ======================================================================
    # Completion Summary
    # ======================================================================
    plots_str = ", ".join(plots_generated) if plots_generated else "None"
    console.print(f"[dim]Plots generated:[/dim] [bold]{plots_str}[/bold]")
    if plots_skipped:
        skipped_str = ", ".join(plots_skipped)
        console.print(
            f"[dim]Plots skipped (already exist):[/dim] "
            f"[yellow]{skipped_str}[/yellow]"
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

    # Build viz config from project defaults + explicit CLI overrides
    viz_cfg = _build_viz_config(args)

    # Display what will be generated
    enabled_plots = []
    if viz_cfg.loss:
        enabled_plots.append("loss")
    if viz_cfg.ecdf:
        enabled_plots.append("ECDF")
    if viz_cfg.ppc:
        enabled_plots.append("PPC")
    if viz_cfg.bio_ppc:
        enabled_plots.append("bio-PPC")
    if viz_cfg.umap:
        enabled_plots.append("UMAP")
    if viz_cfg.heatmap:
        enabled_plots.append("heatmap")
    if viz_cfg.mixture_ppc:
        enabled_plots.append("mixture PPC")
    if viz_cfg.mixture_composition:
        enabled_plots.append("mixture composition")
    if viz_cfg.annotation_ppc:
        enabled_plots.append("annotation PPC")

    console.print(f"[dim]Plots to generate:[/dim] {', '.join(enabled_plots)}")
    console.print(f"[dim]Output format:[/dim] {viz_cfg.format}")

    # ======================================================================
    # Dispatch: single path, wildcard pattern, or recursive search
    # ======================================================================
    input_exprs = args.model_dir
    uses_glob = any(_contains_glob_pattern(expr) for expr in input_exprs)

    if args.recursive:
        console.print()
        console.print(
            Panel.fit(
                "[bold bright_cyan]RECURSIVE SEARCH[/bold bright_cyan]",
                border_style="bright_cyan",
            )
        )
        if uses_glob:
            console.print(
                f"[dim]Searching recursively for roots matching:[/dim] "
                f"[cyan]{', '.join(input_exprs)}[/cyan]"
            )
        else:
            console.print(
                f"[dim]Searching recursively in:[/dim] "
                f"[cyan]{', '.join(os.path.abspath(x) for x in input_exprs)}[/cyan]"
            )
    elif uses_glob:
        console.print(
            f"[dim]Resolving wildcard pattern(s):[/dim] "
            f"[cyan]{', '.join(input_exprs)}[/cyan]"
        )

    model_dirs = []
    for expr in input_exprs:
        model_dirs.extend(_resolve_model_dirs(expr, recursive=args.recursive))
    model_dirs = sorted(set(model_dirs))

    if not model_dirs:
        if uses_glob:
            console.print(
                "[bold yellow]No model directories matched the pattern "
                "with scribe_results.pkl[/bold yellow]"
            )
        elif args.recursive:
            console.print(
                "[bold yellow]No model directories found containing "
                "scribe_results.pkl[/bold yellow]"
            )
        else:
            missing = ", ".join(os.path.abspath(x) for x in input_exprs)
            console.print(
                "[bold red]ERROR: Model directory does not exist or "
                "does not contain scribe_results.pkl![/bold red]"
            )
            console.print(f"[red]   Input:[/red] [cyan]{missing}[/cyan]")
        return

    n_dirs = len(model_dirs)
    if n_dirs > 1 or args.recursive or uses_glob:
        console.print(
            f"[green]Found {n_dirs} model "
            f"director{'ies' if n_dirs != 1 else 'y'}[/green]"
        )
        for d in model_dirs:
            console.print(f"[dim]  \u2022 {d}[/dim]")

    succeeded = 0
    failed = 0
    for i, model_dir in enumerate(model_dirs, 1):
        if n_dirs > 1:
            console.print()
            console.print(
                Panel.fit(
                    f"[bold bright_yellow]DIRECTORY "
                    f"{i}/{n_dirs}[/bold bright_yellow]",
                    border_style="bright_yellow",
                )
            )
        try:
            if _process_single_model_dir(
                model_dir, viz_cfg, overwrite=args.overwrite
            ):
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            console.print(
                f"[bold red]UNEXPECTED ERROR:[/bold red] [red]{e}[/red]"
            )
            failed += 1

    if n_dirs > 1 or args.recursive or uses_glob:
        console.print()
        console.print(
            Panel.fit(
                "[bold bright_green]VISUALIZATION COMPLETE[/bold bright_green]",
                border_style="bright_green",
            )
        )
        console.print(f"[dim]Directories processed:[/dim] {n_dirs}")
        console.print(f"[dim]Succeeded:[/dim] [green]{succeeded}[/green]")
        if failed > 0:
            console.print(f"[dim]Failed:[/dim] [red]{failed}[/red]")

    console.print()


if __name__ == "__main__":
    main()

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

    $ python infer.py data=<your_data_config> model=nbdm inference=svi

    # With model options
    $ python infer.py data=singer model=zinb parameterization=linked n_components=3

    # Override inference parameters
    $ python infer.py data=singer model=nbdm inference.n_steps=100000 inference.batch_size=512

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
from scribe.data_loader import load_and_preprocess_anndata

# Suppress scanpy/anndata deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")


def _handle_deprecated_flags(cfg: DictConfig) -> str:
    """Handle deprecated boolean flags and convert to model type.

    Deprecated flags: zero_inflated, variable_capture, mixture_model

    Returns the appropriate model type string.
    """
    # Check for deprecated flags
    has_deprecated = any(
        [
            "zero_inflated" in cfg,
            "variable_capture" in cfg,
            "mixture_model" in cfg,
        ]
    )

    if has_deprecated:
        warnings.warn(
            "The 'zero_inflated', 'variable_capture', and 'mixture_model' flags are deprecated. "
            "Please use 'model' parameter instead:\n"
            "  - model=nbdm (basic)\n"
            "  - model=zinb (zero-inflated)\n"
            "  - model=nbvcp (variable capture)\n"
            "  - model=zinbvcp (zero-inflated + variable capture)\n"
            "  - n_components=N for mixture models\n"
            "These flags will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert deprecated flags to model type
        zero_inflated = cfg.get("zero_inflated", False)
        variable_capture = cfg.get("variable_capture", False)

        if zero_inflated and variable_capture:
            return "zinbvcp"
        elif zero_inflated:
            return "zinb"
        elif variable_capture:
            return "nbvcp"
        else:
            return "nbdm"

    # Return the model from config, or default to nbdm
    return cfg.get("model", "nbdm")


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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("SCRIBE PROBABILISTIC INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Working directory: {os.getcwd()}")
    print("\nConfiguration:")
    print("-" * 40)
    print(OmegaConf.to_yaml(cfg))

    # ==========================================================================
    # Data Loading Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("DATA LOADING")
    print("=" * 80)

    data_path = hydra.utils.to_absolute_path(cfg.data.path)
    print(f"Loading data from: {data_path}")
    counts = load_and_preprocess_anndata(
        data_path, cfg.data.get("preprocessing")
    )
    print(f"Data loaded successfully! Shape: {counts.shape}")

    # ==========================================================================
    # Configuration Preparation Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CONFIGURATION PREPARATION")
    print("=" * 80)

    # Extract inference config
    inference_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
    inference_method = inference_cfg.pop("method")

    # Build priors dict (filtering None values)
    priors = _build_priors_dict(cfg.get("priors"))

    # Handle deprecated flags (zero_inflated, variable_capture, mixture_model)
    model_type = _handle_deprecated_flags(cfg)

    # Handle deprecated n_components from mixture_model flag
    n_components = cfg.get("n_components")
    if cfg.get("mixture_model", False) and n_components is None:
        # If mixture_model=true but n_components not set, default to 2
        n_components = 2
        warnings.warn(
            "mixture_model=true without n_components specified. Defaulting to n_components=2. "
            "Please use n_components=N directly instead of mixture_model flag.",
            DeprecationWarning,
        )

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
        # Inference configuration
        "inference_method": inference_method,
        # Data configuration
        "cells_axis": cfg.cells_axis,
        "layer": cfg.layer,
        "seed": cfg.seed,
    }

    # Add inference-specific parameters
    kwargs.update(inference_cfg)

    print(f"Model: {kwargs['model']}")
    print(f"Parameterization: {kwargs['parameterization']}")
    print(f"Inference method: {kwargs['inference_method']}")
    if kwargs.get("n_components"):
        print(f"Mixture components: {kwargs['n_components']}")
    if kwargs.get("guide_rank"):
        print(f"Guide rank: {kwargs['guide_rank']}")

    # ==========================================================================
    # Model Inference Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("MODEL INFERENCE")
    print("=" * 80)
    print("Starting probabilistic inference...")

    # Run the inference using the simplified API
    results = scribe.fit(counts=counts, **kwargs)

    print("Inference completed successfully!")

    # ==========================================================================
    # Results Saving Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    output_file = os.path.join(output_dir, "scribe_results.pkl")
    print(f"Output directory: {output_dir}")
    print(f"Saving results to: {output_file}")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print("Results saved successfully!")

    # ==========================================================================
    # Visualization Section
    # ==========================================================================
    viz_cfg = cfg.get("viz")
    if viz_cfg:
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)

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
            print(f"Creating figures directory: {figs_dir}")

            scribe.viz.matplotlib_style()
            print("Setting up matplotlib style...")

            if should_plot_loss:
                print("Generating loss history plot...")
                plot_loss(results, figs_dir, cfg, viz_cfg)
            if should_plot_ecdf:
                print("Generating ECDF plot...")
                plot_ecdf(counts, figs_dir, cfg, viz_cfg)
            if should_plot_ppc:
                print("Generating posterior predictive check plots...")
                plot_ppc(results, counts, figs_dir, cfg, viz_cfg)

            print("All visualizations completed!")
        else:
            print("No plots requested (all visualization options disabled)")
    else:
        print("\n" + "=" * 80)
        print("VISUALIZATION SKIPPED")
        print("=" * 80)
        print("No visualization configuration provided")

    # ==========================================================================
    # Completion Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    if viz_cfg and (should_plot_loss or should_plot_ecdf or should_plot_ppc):
        print(f"Figures saved to: {os.path.join(output_dir, 'figs')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

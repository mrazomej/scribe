"""
run_hydra.py

This script serves as the primary entry point for performing probabilistic model
inference within the SCRIBE framework. Designed for execution via Hydra—a
powerful configuration management system—this script automates the loading of
user-specified configurations, manages output organization, and launches the
inference pipeline. It is intended for users who wish to fit probabilistic
models on single-cell RNA-seq data (or similar), with all run settings encoded
in easily reproducible configuration files.


Detailed script explanation:

1. **Hydra-Based Orchestration**: - The script is decorated with `@hydra.main`,
   which parses the provided YAML configuration files, initializes the config
   object (`cfg`), and sets up an output directory structure according to
   Hydra's conventions. - This ensures all runs are reproducible and output
   files are well-organized by experiment and configuration.

2. **Configuration-Driven Execution**: - Users specify data sources, model
   types, parameterizations, inference options, and other run-specific details
   in configuration files (typically under the `conf/` directory). - The script
   receives these settings as a nested config structure (`cfg`), which it adapts
   for internal use.

3. **Data Loading**: - Using the configured data path and any associated
   preprocessing steps, the script loads the input data with SCRIBE's
   `load_and_preprocess_anndata` routine. - This function ensures data is in the
   standardized, preprocessed format required for model fitting.

4. **Argument Preparation for Model Fitting**: - The configuration (`cfg`) is
   converted to a plain dictionary for easier manipulation. - Inference-specific
   parameters (such as the inference `method`, number of steps, etc.) are
   extracted and moved into the appropriate places expected by SCRIBE's core
   function, `run_scribe`. - Keys unrelated to inference (e.g., data paths,
   Hydra internals, and visualization options) are removed at this stage to
   avoid passing unexpected arguments to the model runner.

5. **Model Inference Launch**: - The core line—calling
   `scribe.run_scribe`—initiates probabilistic inference using the chosen model,
   parameterization, inference technique (SVI, MCMC, or VAE), and
   hyperparameters, all defined by the user configuration. - All arguments are
   passed explicitly; this enables flexible, highly configurable invocations.

6. **Result Serialization**: - The inference results are serialized (pickled)
   and saved to the Hydra-created output directory. - This ensures users can
   later find results in a structured location (e.g.,
   `outputs/<data>/<method>/<overrides>/`), supporting downstream analysis or
   visualization.

7. **Reproducibility & Standardization**: - By relying on Hydra for
   configuration and output management, the script guarantees reproducible and
   standardized experiment handling. - Every run is documented via its config,
   and outputs do not conflict, even when multiple experiments are queued or run
   concurrently.

Typical usage:

    $ python run_hydra.py data=<your_data_config> model=<your_model_config>
    inference=<your_inference_config> ...

This command will launch SCRIBE's probabilistic inference engine using your
chosen configuration, automatically manage outputs, and save a binary results
file for further diagnostic or biological interpretation.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import scribe
import pandas as pd
import jax.numpy as jnp
import pickle
import os
from scribe.data_loader import load_and_preprocess_anndata


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("🚀 SCRIBE PROBABILISTIC INFERENCE PIPELINE")
    print("=" * 80)
    print(f"📁 Working directory: {os.getcwd()}")
    print("\n📋 Configuration:")
    print("-" * 40)
    print(OmegaConf.to_yaml(cfg))

    # ==========================================================================
    # Data Loading Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📊 DATA LOADING")
    print("=" * 80)

    data_path = hydra.utils.to_absolute_path(cfg.data.path)
    print(f"📂 Loading data from: {data_path}")
    counts = load_and_preprocess_anndata(
        data_path, cfg.data.get("preprocessing")
    )
    print(f"✅ Data loaded successfully! Shape: {counts.shape}")

    # ==========================================================================
    # Configuration Preparation Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURATION PREPARATION")
    print("=" * 80)

    # Prepare arguments for run_scribe from the config
    kwargs = OmegaConf.to_container(cfg, resolve=True)

    # Move inference-specific args to top level and rename 'method' to
    # 'inference_method'
    inference_kwargs = kwargs.pop("inference")
    kwargs["inference_method"] = inference_kwargs.pop("method")
    kwargs.update(inference_kwargs)

    # Remove keys that are not arguments to run_scribe
    del kwargs["data"]
    if "hydra" in kwargs:
        del kwargs["hydra"]
    if "viz" in kwargs:
        del kwargs[
            "viz"
        ]  # Remove visualization config - not needed for inference

    print("🔧 Configuration prepared for SCRIBE inference")
    print(f"🎯 Inference method: {kwargs.get('inference_method', 'unknown')}")
    print(f"📈 Number of steps: {kwargs.get('n_steps', 'default')}")

    # ==========================================================================
    # Model Inference Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🧠 MODEL INFERENCE")
    print("=" * 80)
    print("🔄 Starting probabilistic inference...")

    # Run the inference
    results = scribe.run_scribe(counts=counts, **kwargs)

    print("✅ Inference completed successfully!")

    # ==========================================================================
    # Results Saving Section
    # ==========================================================================
    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)

    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    output_file = os.path.join(output_dir, "scribe_results.pkl")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Saving results to: {output_file}")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print("✅ Results saved successfully!")

    # ==========================================================================
    # Visualization Section
    # ==========================================================================
    viz_cfg = cfg.get("viz")
    if viz_cfg:
        print("\n" + "=" * 80)
        print("🎨 VISUALIZATION")
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
            print(f"📁 Creating figures directory: {figs_dir}")

            scribe.viz.matplotlib_style()
            print("🎨 Setting up matplotlib style...")

            if should_plot_loss:
                print("📈 Generating loss history plot...")
                plot_loss(results, figs_dir, cfg, viz_cfg)
            if should_plot_ecdf:
                print("📊 Generating ECDF plot...")
                plot_ecdf(counts, figs_dir, cfg, viz_cfg)
            if should_plot_ppc:
                print("🔍 Generating posterior predictive check plots...")
                plot_ppc(results, counts, figs_dir, cfg, viz_cfg)

            print("✅ All visualizations completed!")
        else:
            print("ℹ️  No plots requested (all visualization options disabled)")
    else:
        print("\n" + "=" * 80)
        print("ℹ️  VISUALIZATION SKIPPED")
        print("=" * 80)
        print("No visualization configuration provided")

    # ==========================================================================
    # Completion Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📁 Results saved to: {output_dir}")
    if viz_cfg and (should_plot_loss or should_plot_ecdf or should_plot_ppc):
        print(f"🎨 Figures saved to: {os.path.join(output_dir, 'figs')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

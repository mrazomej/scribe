import hydra
from omegaconf import DictConfig, OmegaConf
import scribe
import pandas as pd
import jax.numpy as jnp
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import numpy as np

# ------------------------------------------------------------------------------


def _get_config_values(cfg):
    """
    Extracts relevant configuration values from a Hydra/OmegaConf config object.

    This function is designed to handle both legacy and current config
    structures. It retrieves the following values:
        - parameterization: The parameterization type for the model.
        - n_components: Number of mixture components (default 1 if not
          specified).
        - n_steps: Number of inference steps (default 50000 if not specified).
        - method: Inference method (e.g., 'svi', 'mcmc').
        - model_type: The type of model (default 'default' if not specified).

    The function first checks for the presence of the 'inference' attribute to
    determine if the config uses the old structure. If not, it falls back to the
    new structure where keys are at the top level.

    Parameters
    ----------
    cfg : OmegaConf.DictConfig or dict-like
        The configuration object from which to extract values.

    Returns
    -------
    dict
        A dictionary containing the extracted configuration values.
    """
    # Check if the config uses the old structure (has 'inference' attribute and
    # 'parameterization' under it)
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "parameterization"):
        # Old structure: extract values from cfg.inference

        # Model parameterization type
        parameterization = cfg.inference.parameterization
        # Number of mixture components, default to 1 if None/0
        n_components = cfg.inference.n_components or 1
        # Number of inference steps
        n_steps = cfg.inference.n_steps
        # Inference method (e.g., 'svi', 'mcmc')
        method = cfg.inference.method
    else:
        # New structure: extract values from top-level keys

        # Default to 'standard' if not present
        parameterization = cfg.get("parameterization", "standard")
        # Default to 1 if not present or falsy
        n_components = cfg.get("n_components") or 1
        # Default to 50000 if not present
        n_steps = cfg.get("n_steps", 50000)
        # Try to get method from cfg.inference if it exists, otherwise default
        # to 'svi'
        method = cfg.inference.method if hasattr(cfg, "inference") else "svi"

    # Attempt to extract model type from cfg.model.type if present, otherwise
    # default to 'default' getattr returns the value of 'type' in cfg.model if
    # it exists, else 'default'
    model_type = getattr(cfg, "model", {}).get("type", "default")

    # Return all extracted values in a dictionary for easy access
    return {
        "parameterization": parameterization,
        "n_components": n_components,
        "n_steps": n_steps,
        "method": method,
        "model_type": model_type,
    }


# ------------------------------------------------------------------------------


def plot_loss(results, figs_dir, cfg, viz_cfg):
    """Plot and save the ELBO loss history."""
    print("Plotting loss history...")

    # Initialize figure
    fig, ax = plt.subplots(figsize=(3.5, 3))
    # Plot loss history
    ax.plot(results.loss_history)

    # Set labels
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO loss")

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename from original config
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_loss.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved loss plot to {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------------------


def _select_genes(counts, n_genes):
    """Select a subset of genes for plotting based on expression."""
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    spaced_indices = np.linspace(0, len(sorted_idx) - 1, num=n_genes, dtype=int)
    selected_idx = sorted_idx[spaced_indices]
    return selected_idx, mean_counts


# ------------------------------------------------------------------------------


def plot_ecdf(counts, figs_dir, cfg, viz_cfg):
    """Plot and save the ECDF of selected genes."""
    print("Plotting ECDF...")

    # Gene selection
    n_genes = viz_cfg.ecdf_opts.n_genes
    selected_idx, _ = _select_genes(counts, n_genes)

    # Sort selected indices for consistency
    selected_idx = np.sort(selected_idx)

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    for i, idx in enumerate(selected_idx):
        sns.ecdfplot(
            data=counts[:, idx],
            ax=ax,
            color=sns.color_palette("Blues", n_colors=n_genes)[i],
            lw=1.5,
            label=None,
        )
    ax.set_xlabel("UMI count")
    ax.set_xscale("log")
    ax.set_ylabel("ECDF")
    plt.tight_layout()

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_example_ecdf.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved ECDF plot to {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------------------


def plot_ppc(results, counts, figs_dir, cfg, viz_cfg):
    """Plot and save the posterior predictive checks."""
    print("Plotting PPC...")

    # Gene selection
    n_genes = viz_cfg.ppc_opts.n_genes
    selected_idx, mean_counts = _select_genes(counts, n_genes)

    # Sort selected indices - this is crucial for proper indexing of results
    selected_idx = np.sort(selected_idx)

    # Index results for selected genes
    results_subset = results[selected_idx]

    # Generate posterior predictive samples
    n_samples = viz_cfg.ppc_opts.n_samples
    print(f"Generating {n_samples} posterior predictive samples...")
    results_subset.get_ppc_samples(n_samples=n_samples)

    # Plotting
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= n_genes:
            ax.axis("off")
            continue

        print(f"Plotting gene {i} PPC...")

        true_counts = counts[:, selected_idx[i]]

        credible_regions = scribe.stats.compute_histogram_credible_regions(
            results_subset.predictive_samples[:, :, i],
            credible_regions=[95, 68, 50],
        )

        hist_results = np.histogram(
            true_counts, bins=credible_regions["bin_edges"], density=True
        )

        cumsum_indices = np.where(np.cumsum(hist_results[0]) <= 0.99)[0]
        max_bin = np.max(
            [cumsum_indices[-1] if len(cumsum_indices) > 0 else 0, 10]
        )

        scribe.viz.plot_histogram_credible_regions_stairs(
            ax, credible_regions, cmap="Blues", alpha=0.5, max_bin=max_bin
        )

        max_bin_hist = (
            max_bin if len(hist_results[0]) > max_bin else len(hist_results[0])
        )
        ax.step(
            hist_results[1][:max_bin_hist],
            hist_results[0][:max_bin_hist],
            where="post",
            label="data",
            color="black",
        )

        ax.set_xlabel("counts")
        ax.set_ylabel("frequency")
        ax.set_title(
            f"$\\langle U \\rangle = {np.round(mean_counts[selected_idx[i]], 0).astype(int)}$",
            fontsize=8,
        )

    plt.tight_layout()
    fig.suptitle("Example PPC", y=1.02)

    # Get output format
    output_format = viz_cfg.get("format", "png")

    # Construct filename
    config_vals = _get_config_values(cfg)
    fname = (
        f"{config_vals['method']}_{config_vals['parameterization'].replace('-', '_')}_"
        f"{config_vals['model_type'].replace('_', '-')}_"
        f"{config_vals['n_components']:02d}components_"
        f"{config_vals['n_steps']}steps_ppc.{output_format}"
    )

    output_path = os.path.join(figs_dir, fname)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved PPC plot to {output_path}")
    plt.close(fig)


# ------------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Running visualization with config:\n", OmegaConf.to_yaml(cfg))

    # Auto-detect the run directory based on the same Hydra path structure
    # that would have been used for the original inference run
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()

    # Construct the expected output directory path using the same logic as the original config
    # This mirrors the hydra.run.dir pattern: outputs/${data.name}/${inference.method}/${hydra:job.override_dirname}
    base_output_dir = "outputs"
    data_name = cfg.data.name
    inference_method = cfg.inference.method

    # Get the override dirname from current hydra config
    override_dirname = hydra_cfg.job.override_dirname

    # Construct the expected run directory
    run_dir = os.path.join(
        base_output_dir, data_name, inference_method, override_dirname
    )
    run_dir = hydra.utils.to_absolute_path(run_dir)

    print(f"Auto-detected run directory: {run_dir}")

    # Check if the directory exists
    if not os.path.exists(run_dir):
        print(f"Error: Run directory does not exist: {run_dir}")
        print(
            "Make sure you've run the inference first with the same configuration."
        )
        return

    # Create figs directory
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    print(f"Saving figures to: {figs_dir}")

    # Load scribe results
    results_file = os.path.join(run_dir, "scribe_results.pkl")
    print(f"Loading scribe results from {results_file}")
    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # Load original config from the run directory
    orig_cfg_file = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(orig_cfg_file):
        print(f"Error: Original config file not found: {orig_cfg_file}")
        return

    orig_cfg = OmegaConf.load(orig_cfg_file)
    print("Loaded original config from inference run")

    # Load data using the original config's data settings
    data_path = hydra.utils.to_absolute_path(orig_cfg.data.path)
    counts = scribe.data_loader.load_and_preprocess_anndata(
        data_path, orig_cfg.data.get("preprocessing")
    )

    # Set plotting style
    scribe.viz.matplotlib_style()

    # Get visualization settings from current config or use defaults
    viz_cfg = getattr(cfg, "viz", None)
    if viz_cfg is None:
        # Use default visualization settings
        viz_cfg = OmegaConf.create(
            {
                "loss": True,
                "ecdf": True,
                "ppc": True,
                "format": "png",
                "ecdf_opts": {"n_genes": 25},
                "ppc_opts": {"n_genes": 25, "n_samples": 1500},
            }
        )
        print("Using default visualization settings")

    # --- Plotting functions will be called here based on viz config ---
    if viz_cfg.get("loss", True):
        plot_loss(results, figs_dir, orig_cfg, viz_cfg)

    if viz_cfg.get("ecdf", True):
        plot_ecdf(counts, figs_dir, orig_cfg, viz_cfg)

    if viz_cfg.get("ppc", True):
        plot_ppc(results, counts, figs_dir, orig_cfg, viz_cfg)

    print("Visualization script finished.")


if __name__ == "__main__":
    main()

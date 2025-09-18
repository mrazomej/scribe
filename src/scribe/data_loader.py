import pandas as pd
import jax.numpy as jnp
import scanpy as sc
from anndata import AnnData
from omegaconf import DictConfig
from typing import Optional
import os
from omegaconf import OmegaConf

# ==============================================================================
# Data Loader
# ==============================================================================


def load_and_preprocess_anndata(
    path: str, prep_config: Optional[DictConfig] = None
) -> jnp.ndarray:
    """
    Load count data from a CSV or h5ad file and optionally apply scanpy
    preprocessing steps.

    Parameters
    ----------
    path : str
        Path to the data file (CSV or h5ad). For CSV, the file should have
        cells as rows and genes as columns.
    prep_config : Optional[DictConfig]
        A configuration object (e.g., OmegaConf DictConfig or dict) specifying
        preprocessing steps. Supported keys:
            - "filter_cells": dict of arguments for scanpy.pp.filter_cells
            - "filter_genes": dict of arguments for scanpy.pp.filter_genes

    Returns
    -------
    jnp.ndarray
        The processed count data as a JAX numpy array (cells x genes).
    """
    # Print the path from which data will be loaded
    print(f"Loading data from {path}...")

    # Check file type and load accordingly
    _, extension = os.path.splitext(path)
    if extension == ".h5ad":
        adata = sc.read_h5ad(path)
    elif extension == ".csv":
        # Read the CSV file into a pandas DataFrame, ignoring lines starting
        # with '#'
        counts_df = pd.read_csv(path, comment="#")
        # Create an AnnData object from the DataFrame values (cells x genes)
        adata = AnnData(counts_df.values)
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. Please use .csv or .h5ad"
        )

    # Print the original shape of the data
    print(f"Original data shape: {adata.shape}")

    # If a preprocessing configuration is provided, apply the specified steps
    if prep_config:
        print("Applying preprocessing steps...")
        # If cell filtering is specified, apply scanpy's filter_cells
        if "filter_cells" in prep_config:
            print(f"Filtering cells with {prep_config.filter_cells}")
            sc.pp.filter_cells(adata, **prep_config.filter_cells)
            print(f"Shape after filtering cells: {adata.shape}")

        # If gene filtering is specified, apply scanpy's filter_genes
        if "filter_genes" in prep_config:
            print(f"Filtering genes with {prep_config.filter_genes}")
            sc.pp.filter_genes(adata, **prep_config.filter_genes)
            print(f"Shape after filtering genes: {adata.shape}")

        # If highly variable gene selection is specified, apply the steps
        if "highly_variable_genes" in prep_config:
            print("Selecting highly variable genes...")

            # Make a copy to avoid modifying the original data during HVG selection
            adata_hvg = adata.copy()
            hvg_config = prep_config.highly_variable_genes

            # OmegaConf DictConfig needs to be converted to a regular dict for modification
            hvg_config_dict = (
                OmegaConf.to_container(hvg_config, resolve=True)
                if isinstance(hvg_config, DictConfig)
                else dict(hvg_config)
            )

            flavor = hvg_config_dict.get("flavor", "seurat")

            # Flavors like 'seurat' expect log-normalized data.
            # Flavors like 'seurat_v3' expect raw counts.
            if flavor not in ["seurat_v3", "seurat_v3_paper"]:
                print(
                    f"Flavor '{flavor}' expects log-normalized data. Preprocessing a copy..."
                )
                if "normalize_total" in prep_config:
                    print(
                        f"Normalizing total with {prep_config.normalize_total}"
                    )
                    sc.pp.normalize_total(
                        adata_hvg, **prep_config.normalize_total
                    )

                if "log1p" in prep_config:
                    print("Applying log1p transformation")
                    sc.pp.log1p(adata_hvg)
            else:
                print(
                    f"Flavor '{flavor}' expects raw counts. Skipping normalization."
                )

            # We handle subsetting manually to ensure raw counts are preserved.
            # So we force subset=False and inplace=True for the annotation step.
            hvg_config_dict["subset"] = False
            hvg_config_dict["inplace"] = True

            print(
                f"Finding highly variable genes with config: {hvg_config_dict}"
            )
            sc.pp.highly_variable_genes(adata_hvg, **hvg_config_dict)

            # Filter the original adata object using the mask from the processed one
            if "highly_variable" in adata_hvg.var.columns:
                adata = adata[:, adata_hvg.var.highly_variable].copy()
                print(
                    f"Shape after selecting highly variable genes: {adata.shape}"
                )
            else:
                print(
                    "Warning: 'highly_variable' mask not found. No genes were filtered."
                )

    # Convert the processed AnnData matrix to a JAX numpy array and return
    return jnp.array(adata.X.toarray())

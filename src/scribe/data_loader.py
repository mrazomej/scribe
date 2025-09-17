import pandas as pd
import jax.numpy as jnp
import scanpy as sc
from anndata import AnnData
from omegaconf import DictConfig
from typing import Optional
import os

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

    # Return the processed count data as an AnnData object
    return adata

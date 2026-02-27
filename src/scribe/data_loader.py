import pandas as pd
import jax.numpy as jnp
import scanpy as sc
from anndata import AnnData
from omegaconf import DictConfig
from typing import Optional, Union
import os
from omegaconf import OmegaConf, ListConfig

# ==============================================================================
# Data Loader
# ==============================================================================


def load_and_preprocess_anndata(
    path: str,
    prep_config: Optional[DictConfig] = None,
    return_jax: bool = True,
    subset_column: Optional[Union[str, list[str]]] = None,
    subset_value: Optional[Union[str, list[str]]] = None,
) -> Union[jnp.ndarray, AnnData]:
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
    return_jax : bool, default True
        If True, return the data as a JAX numpy array. If False, return the
        original AnnData object with all preprocessing applied.
    subset_column : Optional[str or list[str]], default None
        Column name(s) in ``adata.obs`` used to subset the data before
        preprocessing.  When a single string is given, only observations where
        ``adata.obs[subset_column] == subset_value`` are retained.  When a list
        is given, each column is paired with the corresponding entry in
        ``subset_value`` and the masks are ANDed together (i.e. all conditions
        must hold simultaneously).  This is used by ``infer_split.py`` to fit
        separate models per covariate combination without writing temporary
        h5ad files.
    subset_value : Optional[str or list[str]], default None
        The value(s) within ``subset_column`` to keep.  Must be the same length
        as ``subset_column`` when both are lists.  Ignored when
        ``subset_column`` is ``None``.

    Returns
    -------
    jnp.ndarray or AnnData
        If return_jax=True: The processed count data as a JAX numpy array (cells x genes).
        If return_jax=False: The processed AnnData object with all metadata preserved.
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

    # Subset by covariate column(s) if requested (used by infer_split.py).
    # Both single-column (str) and multi-column (list[str]) are supported;
    # multi-column subsets AND all per-column masks together.
    if subset_column is not None and subset_value is not None:
        # Normalise OmegaConf ListConfig → plain list so isinstance checks work
        if isinstance(subset_column, ListConfig):
            subset_column = list(subset_column)
        if isinstance(subset_value, ListConfig):
            subset_value = list(subset_value)

        if isinstance(subset_column, list):
            # Multi-column path: use the DataFrame to AND masks per column
            columns = subset_column
            values = list(subset_value)
            missing = [c for c in columns if c not in adata.obs.columns]
            if missing:
                raise ValueError(
                    f"subset_column(s) {missing} not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
            mask = pd.Series(True, index=adata.obs.index)
            for col, val in zip(columns, values):
                mask = mask & (adata.obs[col].astype(str) == str(val))
            if mask.sum() == 0:
                combos = (
                    adata.obs[columns]
                    .drop_duplicates()
                    .to_dict(orient="records")
                )
                raise ValueError(
                    f"No observations found for combination "
                    f"{dict(zip(columns, values))}. "
                    f"Available combinations: {combos}"
                )
            adata = adata[mask].copy()
            label = ", ".join(f"{c}='{v}'" for c, v in zip(columns, values))
            print(f"Subset to {label}: {adata.shape[0]} cells retained")
        else:
            # Single-column path: existing behaviour
            if subset_column not in adata.obs.columns:
                raise ValueError(
                    f"subset_column '{subset_column}' not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
            mask = adata.obs[subset_column] == subset_value
            if mask.sum() == 0:
                raise ValueError(
                    f"No observations found where "
                    f"adata.obs['{subset_column}'] == '{subset_value}'. "
                    f"Available values: "
                    f"{sorted(adata.obs[subset_column].unique())}"
                )
            adata = adata[mask].copy()
            print(
                f"Subset to {subset_column}='{subset_value}': "
                f"{adata.shape[0]} cells retained"
            )

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

    # Return either JAX array or AnnData object based on return_jax parameter
    if return_jax:
        # Convert the processed AnnData matrix to a JAX numpy array
        # Handle both sparse and dense matrices
        if hasattr(adata.X, "toarray"):
            return jnp.array(adata.X.toarray())
        else:
            return jnp.array(adata.X)
    else:
        # Return the processed AnnData object with all metadata preserved
        return adata

import numpy as np
import pandas as pd
import scipy.sparse as sp
import jax.numpy as jnp
import scanpy as sc
from anndata import AnnData
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union
import os
from pathlib import Path

# ==============================================================================
# Data Loader
# ==============================================================================


def _is_non_string_sequence(value: object) -> bool:
    """Return whether ``value`` is a sequence that is not string-like.

    Parameters
    ----------
    value : object
        Candidate object to inspect.

    Returns
    -------
    bool
        ``True`` when ``value`` is list-like (including optional config
        containers) and should be treated as an iterable of items.
    """
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _to_plain_python(value: object) -> Any:
    """Recursively normalize config-like containers to builtin Python types.

    Parameters
    ----------
    value : object
        Arbitrary value that may contain mapping/sequence containers from
        optional configuration libraries.

    Returns
    -------
    Any
        The same structure represented with builtin ``dict``/``list`` types.
    """
    # Convert mapping-like containers (including DictConfig-style objects) into
    # plain dicts so downstream logic can stay dependency-free.
    if isinstance(value, Mapping):
        return {str(k): _to_plain_python(v) for k, v in value.items()}

    # Convert non-string sequences (including ListConfig-style objects) into
    # lists while preserving scalar string semantics.
    if _is_non_string_sequence(value):
        return [_to_plain_python(v) for v in value]

    return value


def _load_10x_mex_directory(path: str) -> AnnData:
    """Load a 10x Matrix Exchange (MEX) dataset from a directory.

    Parameters
    ----------
    path : str
        Path to a directory containing a 10x MEX bundle. The directory must
        include:

        - ``matrix.mtx`` or ``matrix.mtx.gz``
        - ``barcodes.tsv`` or ``barcodes.tsv.gz``
        - ``features.tsv`` / ``features.tsv.gz`` **or**
          ``genes.tsv`` / ``genes.tsv.gz``

    Returns
    -------
    AnnData
        AnnData object with cells as rows and genes as columns.

    Raises
    ------
    ValueError
        If the directory is missing one or more required MEX files.
    """
    mex_dir = Path(path)

    # Validate expected 10x MEX files explicitly so users get actionable
    # errors instead of a generic parser failure from scanpy internals.
    matrix_candidates = ("matrix.mtx", "matrix.mtx.gz")
    barcode_candidates = ("barcodes.tsv", "barcodes.tsv.gz")
    feature_candidates = (
        "features.tsv",
        "features.tsv.gz",
        "genes.tsv",
        "genes.tsv.gz",
    )
    has_matrix = any((mex_dir / name).exists() for name in matrix_candidates)
    has_barcodes = any((mex_dir / name).exists() for name in barcode_candidates)
    has_features = any((mex_dir / name).exists() for name in feature_candidates)
    if not (has_matrix and has_barcodes and has_features):
        raise ValueError(
            "10x MEX directory is missing required files. Expected "
            "matrix.mtx(.gz), barcodes.tsv(.gz), and features.tsv(.gz) or "
            f"genes.tsv(.gz). Got directory: {path}"
        )

    return sc.read_10x_mtx(str(mex_dir))


def _load_counts_as_anndata(path: str) -> AnnData:
    """Load supported count matrix inputs into ``AnnData``.

    Parameters
    ----------
    path : str
        Path to one of the supported input formats:

        - ``.h5ad``
        - ``.csv`` (cells as rows, genes as columns)
        - 10x MEX directory (contains ``matrix.mtx`` and companion files)
        - ``matrix.mtx`` / ``matrix.mtx.gz`` file within a 10x MEX directory

    Returns
    -------
    AnnData
        Loaded count matrix as an ``AnnData`` object.

    Raises
    ------
    ValueError
        If ``path`` does not point to a supported format.
    """
    path_obj = Path(path)

    # Handle 10x directory-style inputs first to support configs where
    # data.path points directly at a MEX folder rather than a file.
    if path_obj.is_dir():
        return _load_10x_mex_directory(path)

    _, extension = os.path.splitext(path)
    if extension == ".h5ad":
        return sc.read_h5ad(path)
    elif extension == ".csv":
        # Read the CSV file into a pandas DataFrame, ignoring lines starting
        # with '#'
        counts_df = pd.read_csv(path, comment="#")
        # Create an AnnData object from the DataFrame values (cells x genes)
        return AnnData(counts_df.values)
    elif path_obj.name in {"matrix.mtx", "matrix.mtx.gz"}:
        # Accept file-level MEX entrypoints by resolving to their parent dir.
        return _load_10x_mex_directory(str(path_obj.parent))
    else:
        raise ValueError(
            "Unsupported file format. Please use .csv, .h5ad, "
            "a 10x MEX directory, or matrix.mtx(.gz)."
        )


def _apply_filter_cells_steps(
    adata: AnnData, filter_cells_cfg: Mapping[str, Any] | dict[str, Any]
) -> None:
    """Apply ``scanpy.pp.filter_cells`` criteria sequentially.

    Scanpy allows only one of ``min_counts``, ``min_genes``, ``max_counts``, or
    ``max_genes`` per call. This helper accepts a config map that can contain
    multiple criteria and applies them one-by-one in a deterministic order.

    Parameters
    ----------
    adata : AnnData
        AnnData object to filter in place.
    filter_cells_cfg : Mapping[str, Any] or dict[str, Any]
        ``preprocessing.filter_cells`` configuration.

    Raises
    ------
    ValueError
        Raised when unsupported keys are provided under ``filter_cells``.
    """
    # Normalize config-like containers into plain Python mappings so we can
    # validate keys and iterate in a deterministic order.
    filter_cells_cfg = _to_plain_python(filter_cells_cfg)

    if not isinstance(filter_cells_cfg, dict):
        raise ValueError(
            "preprocessing.filter_cells must be a mapping of threshold keys "
            "to values."
        )

    allowed_keys = ("min_counts", "min_genes", "max_counts", "max_genes")
    unknown_keys = sorted(k for k in filter_cells_cfg if k not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            "Unsupported preprocessing.filter_cells key(s): "
            f"{unknown_keys}. Supported keys are: {allowed_keys}"
        )

    for key in allowed_keys:
        if key not in filter_cells_cfg:
            continue
        value = filter_cells_cfg[key]
        if value is None:
            continue
        n_before = adata.shape[0]
        sc.pp.filter_cells(adata, **{key: value})
        print(
            f"Filtering cells with {key}={value}: "
            f"{adata.shape[0]}/{n_before} cells retained"
        )


def load_and_preprocess_anndata(
    path: str,
    prep_config: Optional[Mapping[str, Any]] = None,
    return_jax: bool = True,
    subset_column: Optional[Union[str, Sequence[str]]] = None,
    subset_value: Optional[Union[str, Sequence[str]]] = None,
    filter_obs: Optional[Mapping[str, Sequence[str]]] = None,
) -> Union[jnp.ndarray, AnnData]:
    """
    Load count data and optionally apply scanpy preprocessing steps.

    Parameters
    ----------
    path : str
        Path to the data source. Supported inputs are:

        - ``.h5ad`` files
        - ``.csv`` files (cells as rows and genes as columns)
        - 10x MEX directories
        - ``matrix.mtx``/``matrix.mtx.gz`` files within a 10x MEX directory
    prep_config : Optional[Mapping[str, Any]]
        A configuration object (for example, a plain ``dict``) specifying
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
    filter_obs : Optional[dict[str, list[str]]], default None
        Declarative observation-level pre-filter applied **before**
        ``subset_column``/``subset_value`` and preprocessing.  Keys are column
        names in ``adata.obs``; values are lists of allowed values for that
        column.  For each entry, only observations whose column value is in the
        allowed list are retained (an ``isin`` filter).  When multiple columns
        are specified, the per-column masks are ANDed (all conditions must hold).
        This is useful for excluding unwanted categories (e.g. specific siRNA
        treatments) prior to covariate-split inference via ``infer_split.py``.

    Returns
    -------
    jnp.ndarray or AnnData
        If return_jax=True: The processed count data as a JAX numpy array (cells x genes).
        If return_jax=False: The processed AnnData object with all metadata preserved.
    """
    # Print the path from which data will be loaded
    print(f"Loading data from {path}...")

    # Centralize input-format handling so all call sites share the same set
    # of supported dataset formats and validation behavior.
    adata = _load_counts_as_anndata(path)

    # Newer anndata versions (≥0.10) may return adata.X as a lazy _CSRDataset
    # rather than a proper scipy sparse matrix, even without explicit backing
    # mode. Materialise it immediately so all downstream operations (scanpy
    # preprocessing, subsetting, and JAX conversion) work with standard arrays.
    if not (sp.issparse(adata.X) or isinstance(adata.X, np.ndarray)):
        adata.X = adata.X[:]

    # Print the original shape of the data
    print(f"Original data shape: {adata.shape}")

    # Apply declarative obs-level pre-filter before any split-based subsetting.
    # Each key in filter_obs is a column name and each value is a list of
    # allowed values; rows not matching ANY column are dropped.
    if filter_obs is not None:
        # Normalize config-style containers to plain Python types before
        # iterating so optional config libraries remain truly optional.
        filter_obs = _to_plain_python(filter_obs)
        if not isinstance(filter_obs, dict):
            raise ValueError(
                "filter_obs must be a mapping of obs columns to allowed values."
            )

        for col, allowed in filter_obs.items():
            if col not in adata.obs.columns:
                raise ValueError(
                    f"filter_obs column '{col}' not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )
            # Normalize allowed values to a list while accepting either scalar
            # values or sequence-like config containers.
            if _is_non_string_sequence(allowed):
                allowed = list(allowed)
            else:
                allowed = [allowed]
            allowed_str = [str(v) for v in allowed]
            mask = adata.obs[col].astype(str).isin(allowed_str)
            n_before = adata.shape[0]
            adata = adata[mask].copy()
            print(
                f"filter_obs {col} in {allowed}: "
                f"{adata.shape[0]}/{n_before} cells retained"
            )

        if adata.shape[0] == 0:
            raise ValueError(
                "No observations remain after applying filter_obs: "
                f"{filter_obs}"
            )

    # Subset by covariate column(s) if requested (used by infer_split.py).
    # Both single-column (str) and multi-column (list[str]) are supported;
    # multi-column subsets AND all per-column masks together.
    if subset_column is not None and subset_value is not None:
        # Normalize config-like sequence containers once so list/scalar checks
        # and error messages are deterministic.
        subset_column = _to_plain_python(subset_column)
        subset_value = _to_plain_python(subset_value)

        if _is_non_string_sequence(subset_column):
            # Multi-column path: use the DataFrame to AND masks per column
            columns = list(subset_column)
            if not _is_non_string_sequence(subset_value):
                raise ValueError(
                    "subset_value must be a sequence when subset_column is a "
                    "sequence."
                )
            values = list(subset_value)
            if len(columns) != len(values):
                raise ValueError(
                    "subset_column and subset_value must have the same length "
                    "for multi-column subsetting."
                )
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
        # Normalize config-like containers once, then use dict-style access to
        # avoid optional config dependencies in core library code.
        prep_config = _to_plain_python(prep_config)
        if not isinstance(prep_config, dict):
            raise ValueError(
                "prep_config must be a mapping of preprocessing settings."
            )
        print("Applying preprocessing steps...")
        # If cell filtering is specified, apply scanpy's filter_cells
        if "filter_cells" in prep_config:
            # Apply one threshold per Scanpy call so combined criteria such as
            # min_counts + min_genes can be configured safely.
            _apply_filter_cells_steps(adata, prep_config["filter_cells"])
            print(f"Shape after filtering cells: {adata.shape}")

        # If gene filtering is specified, apply scanpy's filter_genes
        if "filter_genes" in prep_config:
            filter_genes_cfg = _to_plain_python(prep_config["filter_genes"])
            if not isinstance(filter_genes_cfg, dict):
                raise ValueError(
                    "preprocessing.filter_genes must be a mapping of keyword "
                    "arguments."
                )
            print(f"Filtering genes with {filter_genes_cfg}")
            sc.pp.filter_genes(adata, **filter_genes_cfg)
            print(f"Shape after filtering genes: {adata.shape}")

        # If highly variable gene selection is specified, apply the steps
        if "highly_variable_genes" in prep_config:
            print("Selecting highly variable genes...")

            # Make a copy to avoid modifying the original data during HVG selection
            adata_hvg = adata.copy()
            hvg_config = _to_plain_python(prep_config["highly_variable_genes"])
            if not isinstance(hvg_config, dict):
                raise ValueError(
                    "preprocessing.highly_variable_genes must be a mapping of "
                    "keyword arguments."
                )
            hvg_config_dict = dict(hvg_config)

            flavor = hvg_config_dict.get("flavor", "seurat")

            # Flavors like 'seurat' expect log-normalized data.
            # Flavors like 'seurat_v3' expect raw counts.
            if flavor not in ["seurat_v3", "seurat_v3_paper"]:
                print(
                    f"Flavor '{flavor}' expects log-normalized data. Preprocessing a copy..."
                )
                if "normalize_total" in prep_config:
                    normalize_total_cfg = _to_plain_python(
                        prep_config["normalize_total"]
                    )
                    if normalize_total_cfg is None:
                        normalize_total_cfg = {}
                    if not isinstance(normalize_total_cfg, dict):
                        raise ValueError(
                            "preprocessing.normalize_total must be a mapping "
                            "when provided."
                        )
                    print(
                        f"Normalizing total with {normalize_total_cfg}"
                    )
                    sc.pp.normalize_total(
                        adata_hvg, **normalize_total_cfg
                    )

                if "log1p" in prep_config:
                    log1p_cfg = prep_config["log1p"]
                    print("Applying log1p transformation")
                    if isinstance(log1p_cfg, Mapping):
                        sc.pp.log1p(
                            adata_hvg, **_to_plain_python(log1p_cfg)
                        )
                    else:
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

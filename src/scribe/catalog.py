"""
Experiment catalog for managing and querying scribe experiment results.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import yaml
from dataclasses import dataclass
import hydra

# ==============================================================================
# ExperimentRun class
# ==============================================================================


@dataclass
class ExperimentRun:
    """
    Represents a single experiment run with metadata and file paths.
    """

    path: str
    metadata: Dict[str, Any]

    # --------------------------------------------------------------------------

    def load_results(self):
        """Load the scribe results pickle file."""
        results_path = os.path.join(self.path, "scribe_results.pkl")
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"No scribe_results.pkl found in {self.path}"
            )

        with open(results_path, "rb") as f:
            return pickle.load(f)

    # --------------------------------------------------------------------------

    def load_config(self):
        """Load the experiment configuration."""
        config_path = os.path.join(self.path, ".hydra", "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config.yaml found in {self.path}/.hydra/"
            )

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------------------------

    def load_overrides(self):
        """Load the hydra overrides."""
        overrides_path = os.path.join(self.path, ".hydra", "overrides.yaml")
        if not os.path.exists(overrides_path):
            return []

        with open(overrides_path, "r") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------------------------

    def list_files(self):
        """List all files in the experiment directory."""
        files = []
        for root, dirs, filenames in os.walk(self.path):
            for filename in filenames:
                rel_path = os.path.relpath(
                    os.path.join(root, filename), self.path
                )
                files.append(rel_path)
        return sorted(files)

    # --------------------------------------------------------------------------

    def load_data(self, return_jax: bool = False):
        """
        Load and preprocess the original data used in this experiment.

        Uses the data path and preprocessing configuration from the experiment's
        Hydra config to recreate the exact data that was used during inference.

        Parameters
        ----------
        return_jax : bool, default True
            If True, return the data as a JAX numpy array. If False, return the
            original AnnData object with all preprocessing applied.

        Returns
        -------
        jnp.ndarray or AnnData
            - If return_jax=True: The preprocessed count data as a JAX numpy
              array (cells x genes).
            - If return_jax=False: The processed AnnData object with all
              metadata preserved.
        """
        from .data_loader import load_and_preprocess_anndata

        # Load the experiment config
        config = self.load_config()

        # Extract data configuration
        if "data" not in config:
            raise ValueError(f"No 'data' configuration found in {self.path}")

        data_config = config["data"]

        # Get the data path - need to resolve it relative to the experiment path
        if "path" not in data_config:
            raise ValueError(
                f"No 'path' found in data configuration for {self.path}"
            )

        data_path = data_config["path"]

        # Convert to absolute path using hydra's utility
        # The path in config might be relative to the original working directory
        data_path = hydra.utils.to_absolute_path(data_path)

        # Get preprocessing configuration if it exists
        prep_config = data_config.get("preprocessing", None)

        print(f"Loading data for experiment: {self.path}")
        print(f"Data path: {data_path}")
        if prep_config:
            print(f"Preprocessing config: {prep_config}")

        # Load and preprocess the data
        return load_and_preprocess_anndata(
            data_path, prep_config, return_jax=return_jax
        )

    # --------------------------------------------------------------------------

    def __repr__(self):
        return f"ExperimentRun({self.path}, metadata={self.metadata})"


# ==============================================================================
# ExperimentCatalog class
# ==============================================================================


class ExperimentCatalog:
    """
    Catalog for managing experiment results based on metadata.

    Usage:
    ------
    # Single directory
    catalog = ExperimentCatalog("outputs/")

    # Multiple directories
    catalog = ExperimentCatalog(["outputs/", "results/", "archived/"])

    # List all experiments
    catalog.list()

    # Query experiments
    results = catalog.find(
        data="jurkat_cells", inference="svi", parameterization="odds_ratio"
    )

    # Load specific files
    scribe_results = results[0].load_results()
    config = results[0].load_config()

    # Convenience method to load results directly
    results = catalog.load_results(
        data="jurkat_cells", parameterization="standard"
    )
    """

    def __init__(self, base_dir: Union[str, List[str]]):
        """
        Initialize the experiment catalog.

        Parameters
        ----------
        base_dir : str or List[str]
            Base directory (or list of directories) containing experiment
            outputs (e.g., "outputs/" or ["outputs/", "results/"])
        """
        # Convert single directory to list for uniform handling
        if isinstance(base_dir, str):
            base_dir = [base_dir]

        # Resolve and validate all directories
        self.base_dirs = []
        for dir_path in base_dir:
            resolved_path = Path(dir_path).resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"Base directory does not exist: {resolved_path}"
                )
            self.base_dirs.append(resolved_path)

        # For backward compatibility, keep base_dir as single path if only one
        # provided
        self.base_dir = self.base_dirs[0] if len(self.base_dirs) == 1 else None

        self.experiments = self._scan_experiments()
        dir_summary = ", ".join(str(d) for d in self.base_dirs)
        print(
            f"Found {len(self.experiments)} experiments across "
            f"{len(self.base_dirs)} "
            f"director{'y' if len(self.base_dirs) == 1 else 'ies'}: "
            f"{dir_summary}"
        )

    # --------------------------------------------------------------------------

    def _parse_override_dirname(self, dirname: str) -> Dict[str, Any]:
        """
        Parse Hydra override dirname into metadata dictionary.

        Handles formats like:
        - 'data=jurkat_cells,inference.batch_size=512,variable_capture=true'
        - 'parameterization=odds_ratio'

        Parameters
        ----------
        dirname : str
            The override dirname from Hydra

        Returns
        -------
        Dict[str, Any]
            Parsed metadata dictionary
        """
        metadata = {}

        if not dirname or dirname == "":
            return metadata

        # Split only on commas that start a new key=value segment.
        # This preserves comma-delimited values such as
        # "mixture_params=phi,mu,gate" as one logical entry.
        pairs = []
        current_pair = ""
        for part in dirname.split(","):
            part = part.strip()
            if not part:
                continue

            if "=" in part:
                if current_pair:
                    pairs.append(current_pair)
                current_pair = part
            elif current_pair:
                current_pair = f"{current_pair},{part}"

        if current_pair:
            pairs.append(current_pair)

        # Parse each recovered key=value pair.
        for pair in pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)  # Split only on first =

                # Convert string values to appropriate types
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() == "null" or value.lower() == "none":
                    value = None
                else:
                    # Try to convert to number
                    try:
                        if "." in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # Keep as string
                        pass

                # Handle nested keys (e.g., inference.batch_size)
                if "." in key:
                    parts = key.split(".")
                    current = metadata
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    metadata[key] = value

        return metadata

    # --------------------------------------------------------------------------

    @staticmethod
    def _normalize_comma_delimited_value(value: Any) -> Any:
        """Normalize comma-delimited list-like filter values.

        Parameters
        ----------
        value : Any
            Raw filter value provided by the caller, or stored metadata value.

        Returns
        -------
        Any
            If ``value`` encodes a comma-delimited list (for example
            ``"phi,mu,gate"`` or ``["phi,mu,gate"]``), returns a normalized
            ``list[str]``. Otherwise, returns ``value`` unchanged.
        """
        # Normalize singleton list values like ["a,b,c"] into ["a", "b", "c"].
        if isinstance(value, list) and len(value) == 1:
            singleton = value[0]
            if isinstance(singleton, str) and "," in singleton:
                return [
                    piece.strip()
                    for piece in singleton.split(",")
                    if piece.strip()
                ]

        # Normalize string values like "a,b,c" into ["a", "b", "c"].
        if isinstance(value, str) and "," in value:
            return [piece.strip() for piece in value.split(",") if piece.strip()]

        return value

    # --------------------------------------------------------------------------

    def _scan_experiments(self) -> List[ExperimentRun]:
        """
        Scan the base directories for experiment runs.

        Looks for directories containing both *.log files and *.pkl files (saved
        model files) and extracts metadata from the Hydra config.yaml files.
        Filters out parameters where all experiments have null values and
        excludes any 'viz' related parameters.

        Returns
        -------
        List[ExperimentRun]
            List of found experiment runs
        """
        temp_experiments = []

        # Walk through each base directory
        for base_dir in self.base_dirs:
            # Walk through the directory structure
            for root, dirs, files in os.walk(base_dir):
                root_path = Path(root)

                # Check if this directory contains experiment indicators
                has_log_file = any(f.endswith(".log") for f in files)
                has_pkl_file = any(f.endswith(".pkl") for f in files)
                has_hydra_dir = ".hydra" in dirs

                # This is an experiment directory if it has both log and pkl files
                if has_log_file and has_pkl_file and has_hydra_dir:
                    try:
                        # Load metadata from config.yaml
                        config_path = root_path / ".hydra" / "config.yaml"
                        if not config_path.exists():
                            print(
                                f"Warning: No config.yaml found in "
                                f"{root_path}/.hydra/"
                            )
                            continue

                        with open(config_path, "r") as f:
                            config = yaml.safe_load(f)

                        # Extract metadata from config, excluding 'viz' related
                        # parameters
                        metadata = self._extract_config_metadata(config)

                        print(
                            f"Found experiment: {root_path} with metadata keys: "
                            f"{list(metadata.keys())}"
                        )

                        temp_experiments.append(
                            ExperimentRun(
                                path=str(root_path), metadata=metadata
                            )
                        )

                    except Exception as e:
                        print(
                            f"Warning: Could not process experiment directory "
                            f"{root_path}: {e}"
                        )
                        continue

        # Filter out parameters where all experiments have null/None values
        filtered_experiments = self._filter_null_parameters(temp_experiments)

        return filtered_experiments

    # --------------------------------------------------------------------------

    def _extract_config_metadata(
        self, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from a Hydra config dictionary.

        Recursively flattens the config into a flat dictionary with dot notation
        for nested keys, excluding any parameters related to 'viz'.

        Parameters
        ----------
        config : Dict[str, Any]
            The loaded Hydra config dictionary

        Returns
        -------
        Dict[str, Any]
            Flattened metadata dictionary
        """
        metadata = {}

        def flatten_config(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Skip anything related to 'viz'
                    if "viz" in key.lower():
                        continue

                    new_key = f"{prefix}{key}" if prefix else key

                    if isinstance(value, dict):
                        flatten_config(value, f"{new_key}.")
                    else:
                        metadata[new_key] = value
            else:
                # This shouldn't happen at the top level, but handle gracefully
                if prefix and "viz" not in prefix.lower():
                    metadata[prefix.rstrip(".")] = obj

        flatten_config(config)
        return metadata

    # --------------------------------------------------------------------------

    def _filter_null_parameters(
        self, experiments: List[ExperimentRun]
    ) -> List[ExperimentRun]:
        """
        Filter out parameters where all experiments have null/None values.

        Parameters
        ----------
        experiments : List[ExperimentRun]
            List of experiments to filter

        Returns
        -------
        List[ExperimentRun]
            Experiments with filtered metadata
        """
        if not experiments:
            return experiments

        # Collect all parameter keys across all experiments
        all_params = set()
        for exp in experiments:
            all_params.update(exp.metadata.keys())

        # Find parameters where all experiments have null/None values
        params_to_remove = set()
        for param in all_params:
            all_null = True
            for exp in experiments:
                value = exp.metadata.get(param)
                if value is not None and value != "null":
                    all_null = False
                    break
            if all_null:
                params_to_remove.add(param)

        # Remove null parameters from all experiments
        filtered_experiments = []
        for exp in experiments:
            filtered_metadata = {
                k: v
                for k, v in exp.metadata.items()
                if k not in params_to_remove
            }
            filtered_experiments.append(
                ExperimentRun(path=exp.path, metadata=filtered_metadata)
            )

        if params_to_remove:
            print(
                f"Filtered out {len(params_to_remove)} parameters with all "
                f"null values: {sorted(params_to_remove)}"
            )

        return filtered_experiments

    # --------------------------------------------------------------------------

    def list(self) -> pd.DataFrame:
        """
        List all experiments as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with experiment metadata and paths
        """
        if not self.experiments:
            return pd.DataFrame()

        # Convert metadata to flat dictionary for DataFrame
        rows = []
        for exp in self.experiments:
            row = {"path": exp.path}

            # Flatten nested metadata
            def flatten_dict(d, prefix=""):
                for key, value in d.items():
                    if isinstance(value, dict):
                        flatten_dict(value, f"{prefix}{key}.")
                    else:
                        row[f"{prefix}{key}"] = value

            flatten_dict(exp.metadata)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder columns to put common ones first
        common_cols = [
            "data",
            "inference",
            "parameterization",
            "variable_capture",
            "path",
        ]
        other_cols = [col for col in df.columns if col not in common_cols]
        ordered_cols = [
            col for col in common_cols if col in df.columns
        ] + sorted(other_cols)

        return df[ordered_cols]

    # --------------------------------------------------------------------------

    def find(self, **filters) -> List[ExperimentRun]:
        """
        Find experiments matching the given filters.

        Parameters
        ----------
        **filters
            Key-value pairs to filter experiments by. Supports nested keys with
            dot notation (e.g., inference.batch_size=512)

        Returns
        -------
        List[ExperimentRun]
            List of matching experiment runs
        """
        matching_experiments = []

        for exp in self.experiments:
            matches = True

            for key, value in filters.items():
                normalized_filter_value = self._normalize_comma_delimited_value(
                    value
                )

                # Handle nested keys
                if "." in key:
                    parts = key.split(".")
                    current = exp.metadata
                    try:
                        for part in parts:
                            current = current[part]
                        normalized_current = self._normalize_comma_delimited_value(
                            current
                        )
                        if normalized_current != normalized_filter_value:
                            matches = False
                            break
                    except (KeyError, TypeError):
                        matches = False
                        break
                else:
                    current = exp.metadata.get(key)
                    normalized_current = self._normalize_comma_delimited_value(
                        current
                    )
                    if normalized_current != normalized_filter_value:
                        matches = False
                        break

            if matches:
                matching_experiments.append(exp)

        return matching_experiments

    # --------------------------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[ExperimentRun], bool],
        experiments: Optional[List[ExperimentRun]] = None,
    ) -> List[ExperimentRun]:
        """Filter experiments using a user-provided callable predicate.

        Parameters
        ----------
        predicate : Callable[[ExperimentRun], bool]
            Callable that receives an ``ExperimentRun`` and returns ``True`` for
            runs that should be kept. This supports flexible lambda-based logic,
            including path-name checks, metadata combinations, and arbitrary
            custom predicates.
        experiments : Optional[List[ExperimentRun]], default None
            Optional source list to filter. If ``None``, filters the full
            catalog (``self.experiments``).

        Returns
        -------
        List[ExperimentRun]
            Experiments that satisfy the predicate.

        Raises
        ------
        TypeError
            If ``predicate`` is not callable.
        """
        # Validate callable input early so usage errors are explicit.
        if not callable(predicate):
            raise TypeError(
                "predicate must be callable and accept an ExperimentRun."
            )

        # Support filtering either the full catalog or an intermediate subset.
        source_experiments = (
            self.experiments if experiments is None else experiments
        )
        return [exp for exp in source_experiments if predicate(exp)]

    # --------------------------------------------------------------------------

    def load_results(self, **filters):
        """
        Convenience method to load scribe results directly.

        Parameters
        ----------
        **filters
            Key-value pairs to filter experiments by

        Returns
        -------
        Any
            Loaded scribe results. If multiple experiments match, returns the
            first one.

        Raises
        ------
        ValueError
            If no experiments match the filters or multiple experiments match
        """
        experiments = self.find(**filters)

        if len(experiments) == 0:
            raise ValueError(
                f"No experiments found matching filters: {filters}"
            )
        elif len(experiments) > 1:
            print(
                f"Warning: {len(experiments)} experiments match filters. "
                "Using the first one."
            )
            print("Matching experiments:")
            for exp in experiments:
                print(f"  - {exp.path}")

        return experiments[0].load_results()

    # --------------------------------------------------------------------------

    def load_data(self, return_jax: bool = True, **filters):
        """
        Convenience method to load the original data directly.

        Parameters
        ----------
        return_jax : bool, default True
            If True, return the data as a JAX numpy array. If False, return the
            original AnnData object with all preprocessing applied.
        **filters
            Key-value pairs to filter experiments by

        Returns
        -------
        jnp.ndarray or AnnData
            If return_jax=True: The preprocessed count data as a JAX numpy array.
            If return_jax=False: The processed AnnData object.
            If multiple experiments match, returns data from the first one.

        Raises
        ------
        ValueError
            If no experiments match the filters
        """
        experiments = self.find(**filters)

        if len(experiments) == 0:
            raise ValueError(
                f"No experiments found matching filters: {filters}"
            )
        elif len(experiments) > 1:
            print(
                f"Warning: {len(experiments)} experiments match filters. "
                "Using the first one."
            )
            print("Matching experiments:")
            for exp in experiments:
                print(f"  - {exp.path}")

        return experiments[0].load_data(return_jax=return_jax)

    # --------------------------------------------------------------------------

    def refresh(self):
        """Refresh the catalog by re-scanning the base directory."""
        self.experiments = self._scan_experiments()
        print(f"Refreshed catalog. Found {len(self.experiments)} experiments.")

    # --------------------------------------------------------------------------
    # Make the catalog indexable and iterable
    # --------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of experiments in the catalog."""
        return len(self.experiments)

    # --------------------------------------------------------------------------

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ExperimentRun, List[ExperimentRun]]:
        """
        Get experiment(s) by index.

        Parameters
        ----------
        index : int or slice
            Index or slice to select experiments

        Returns
        -------
        ExperimentRun or List[ExperimentRun]
            The selected experiment(s)
        """
        return self.experiments[index]

    def __iter__(self):
        """Make the catalog iterable over experiments."""
        return iter(self.experiments)

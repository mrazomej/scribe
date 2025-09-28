"""
Experiment catalog for managing and querying scribe experiment results.
"""

import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
from dataclasses import dataclass
import hydra


@dataclass
class ExperimentRun:
    """
    Represents a single experiment run with metadata and file paths.
    """

    path: str
    metadata: Dict[str, Any]

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

    def load_data(self):
        """
        Load and preprocess the original data used in this experiment.

        Uses the data path and preprocessing configuration from the experiment's
        Hydra config to recreate the exact data that was used during inference.

        Returns
        -------
        jnp.ndarray
            The preprocessed count data as a JAX numpy array (cells x genes)
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
        return load_and_preprocess_anndata(data_path, prep_config)

    # --------------------------------------------------------------------------

    def __repr__(self):
        return f"ExperimentRun({self.path}, metadata={self.metadata})"


class ExperimentCatalog:
    """
    Catalog for managing experiment results based on metadata.

    Usage:
    ------
    catalog = ExperimentCatalog("outputs/")

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

    def __init__(self, base_dir: str):
        """
        Initialize the experiment catalog.

        Parameters
        ----------
        base_dir : str
            Base directory containing experiment outputs (e.g., "outputs/")
        """
        self.base_dir = Path(base_dir).resolve()
        if not self.base_dir.exists():
            raise FileNotFoundError(
                f"Base directory does not exist: {self.base_dir}"
            )

        self.experiments = self._scan_experiments()
        print(f"Found {len(self.experiments)} experiments in {self.base_dir}")

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

        # Split by comma and parse each key=value pair
        for pair in dirname.split(","):
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

    def _scan_experiments(self) -> List[ExperimentRun]:
        """
        Scan the base directory for experiment runs.

        Looks for directories containing experiment indicators (*.log files,
        scribe_results.pkl, or .hydra directories) and extracts metadata from
        the directory structure.

        Returns
        -------
        List[ExperimentRun]
            List of found experiment runs
        """
        experiments = []

        # Walk through the directory structure
        for root, dirs, files in os.walk(self.base_dir):
            root_path = Path(root)

            # Check if this directory contains experiment indicators
            has_log_file = any(f.endswith(".log") for f in files)
            has_scribe_results = "scribe_results.pkl" in files
            has_hydra_dir = ".hydra" in dirs

            # This is an experiment directory if it has any of these indicators
            if has_log_file or (has_scribe_results and has_hydra_dir):
                try:
                    relative_path = root_path.relative_to(self.base_dir)
                    path_parts = relative_path.parts

                    # Initialize metadata with what we can extract from path
                    metadata = {}

                    # Try to extract data name and inference method from path
                    # structure
                    if len(path_parts) >= 1:
                        # First part is likely the data name
                        metadata["data"] = path_parts[0]

                    if len(path_parts) >= 2:
                        # Second part is likely the inference method
                        metadata["inference"] = path_parts[1]

                    # The last part (or parts) might contain override
                    # information
                    if len(path_parts) >= 3:
                        # Try to parse the last part as override dirname
                        override_dirname = path_parts[-1]
                        override_metadata = self._parse_override_dirname(
                            override_dirname
                        )
                        metadata.update(override_metadata)

                    # If we have a .hydra directory, try to get more accurate
                    # metadata from config
                    if has_hydra_dir:
                        try:
                            config_path = root_path / ".hydra" / "config.yaml"
                            if config_path.exists():
                                with open(config_path, "r") as f:
                                    config = yaml.safe_load(f)

                                # Extract data name from config if available
                                if (
                                    "data" in config
                                    and "name" in config["data"]
                                ):
                                    metadata["data"] = config["data"]["name"]

                                # Extract inference method from config if
                                # available
                                if (
                                    "inference" in config
                                    and "method" in config["inference"]
                                ):
                                    metadata["inference"] = config["inference"][
                                        "method"
                                    ]

                                # Extract other config parameters
                                for key in [
                                    "parameterization",
                                    "variable_capture",
                                    "zero_inflated",
                                    "mixture_model",
                                ]:
                                    if key in config:
                                        metadata[key] = config[key]

                                # Extract nested inference parameters
                                if "inference" in config:
                                    for key, value in config[
                                        "inference"
                                    ].items():
                                        if (
                                            key != "method"
                                        ):  # Skip method as we already have it
                                            metadata[f"inference.{key}"] = value

                        except Exception as e:
                            print(
                                f"Warning: Could not parse config from "
                                f"{config_path}: {e}"
                            )

                    print(
                        f"Found experiment: {root_path} with metadata: "
                        f"{metadata}"
                    )

                    experiments.append(
                        ExperimentRun(path=str(root_path), metadata=metadata)
                    )

                except ValueError as e:
                    print(
                        f"Warning: Could not process experiment directory "
                        f"{root_path}: {e}"
                    )
                    continue

        return experiments

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
                # Handle nested keys
                if "." in key:
                    parts = key.split(".")
                    current = exp.metadata
                    try:
                        for part in parts:
                            current = current[part]
                        if current != value:
                            matches = False
                            break
                    except (KeyError, TypeError):
                        matches = False
                        break
                else:
                    if exp.metadata.get(key) != value:
                        matches = False
                        break

            if matches:
                matching_experiments.append(exp)

        return matching_experiments

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

    def load_data(self, **filters):
        """
        Convenience method to load the original data directly.

        Parameters
        ----------
        **filters
            Key-value pairs to filter experiments by

        Returns
        -------
        jnp.ndarray
            The preprocessed count data. If multiple experiments match,
            returns data from the first one.

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

        return experiments[0].load_data()

    # --------------------------------------------------------------------------

    def refresh(self):
        """Refresh the catalog by re-scanning the base directory."""
        self.experiments = self._scan_experiments()
        print(f"Refreshed catalog. Found {len(self.experiments)} experiments.")

"""Dispatch helpers for ``scribe-infer``.

This module contains small pure functions that parse CLI override arguments
and decide whether split orchestration should be used.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


_DATA_OVERRIDE_PATTERN = re.compile(r"^data=([^\s]+)$")


def extract_data_keys(overrides: Iterable[str]) -> list[str]:
    """Extract one or more ``data=...`` override values.

    Parameters
    ----------
    overrides : Iterable[str]
        CLI tokens after ``scribe-infer`` options parsing.

    Returns
    -------
    list[str]
        Dataset config keys from ``data=...`` entries. Comma-separated values
        are expanded into a flat list and empty tokens are removed.
    """
    data_keys: list[str] = []
    for token in overrides:
        match = _DATA_OVERRIDE_PATTERN.match(token)
        if match is None:
            continue
        raw_value = match.group(1)
        for part in raw_value.split(","):
            key = part.strip()
            if key:
                data_keys.append(key)
    return data_keys


def _load_omegaconf():
    """Import OmegaConf lazily so error messaging can be user-friendly."""
    from omegaconf import OmegaConf  # type: ignore

    return OmegaConf


def has_split_by(config_path: str | Path, data_key: str) -> bool:
    """Return whether a dataset config enables split orchestration.

    Parameters
    ----------
    config_path : str or Path
        Directory that contains the Hydra configuration tree.
    data_key : str
        Dataset config key from a ``data=...`` override.

    Returns
    -------
    bool
        ``True`` when ``<config_path>/data/<data_key>.yaml`` contains a
        non-null ``split_by`` field, ``False`` otherwise.
    """
    omega_conf = _load_omegaconf()
    data_yaml = Path(config_path) / "data" / f"{data_key}.yaml"
    if not data_yaml.exists():
        return False
    data_cfg = omega_conf.to_container(omega_conf.load(data_yaml), resolve=True)
    if not isinstance(data_cfg, dict):
        return False
    return data_cfg.get("split_by") is not None


def should_use_split_mode(config_path: str | Path, overrides: Iterable[str]) -> bool:
    """Determine whether ``scribe-infer`` should dispatch to split mode.

    Parameters
    ----------
    config_path : str or Path
        Directory that contains Hydra configs.
    overrides : Iterable[str]
        Hydra-style CLI override tokens.

    Returns
    -------
    bool
        ``True`` when at least one selected data config has ``split_by``.
    """
    for data_key in extract_data_keys(overrides):
        if has_split_by(config_path=config_path, data_key=data_key):
            return True
    return False


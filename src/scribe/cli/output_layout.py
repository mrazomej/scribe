"""Output directory layout helpers for SCRIBE inference runs.

These utilities derive nested output prefixes from Hydra ``data=...`` config
keys so run artifacts mirror the ``conf/data/`` folder structure.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict


def derive_output_prefix(data_key: str) -> str:
    """Derive a nested output prefix from a Hydra data config key.

    Parameters
    ----------
    data_key : str
        Data config key from ``data=...`` (e.g.
        ``panfibrosis/cell_type_genecorr/CKD/GSE140023__sham__...``).

    Returns
    -------
    str
        Parent path portion of the key (e.g.
        ``panfibrosis/cell_type_genecorr/CKD``), or an empty string when no
        parent directories are present.
    """
    # Build parent path from slash-separated Hydra key, independent of OS
    # separator semantics.
    parent_parts = [part for part in data_key.split("/") if part][:-1]
    return "/".join(parent_parts)


def format_nested_output_prefix(prefix: str) -> str:
    """Format a prefix segment for Hydra output path templates.

    Parameters
    ----------
    prefix : str
        Raw output prefix without trailing slash.

    Returns
    -------
    str
        ``"{prefix}/"`` when ``prefix`` is non-empty, otherwise ``""``.
    """
    return f"{prefix}/" if prefix else ""


def _explicit_output_prefix(data_cfg: Any) -> str | None:
    """Return a non-empty explicit ``output_prefix`` from a data config node.

    Parameters
    ----------
    data_cfg : Any
        Hydra ``data`` config node, mapping, or ``None``.

    Returns
    -------
    str or None
        Explicit prefix string when set and non-empty; otherwise ``None``.
    """
    if data_cfg is None:
        return None
    if OmegaConf.is_config(data_cfg):
        if "output_prefix" not in data_cfg:
            return None
        value = data_cfg.get("output_prefix")
    elif isinstance(data_cfg, dict):
        if "output_prefix" not in data_cfg:
            return None
        value = data_cfg.get("output_prefix")
    else:
        return None

    if value is None or value == "":
        return None
    return str(value)


def resolve_nested_output_prefix(
    *,
    data_cfg: Any = None,
    data_choice: str | None = None,
) -> str:
    """Resolve the nested output prefix segment used in Hydra path templates.

    Priority order:

    1. Non-empty ``data.output_prefix`` from the composed config.
    2. Parent path derived from ``data_choice`` (Hydra ``runtime.choices.data``).
    3. Empty string for flat configs or unrecognized temp split keys.

    Parameters
    ----------
    data_cfg : Any, optional
        Composed Hydra ``data`` config node.
    data_choice : str or None, optional
        Selected data config key from Hydra runtime choices.

    Returns
    -------
    str
        Prefix segment with a trailing slash when non-empty, else ``""``.
    """
    explicit = _explicit_output_prefix(data_cfg)
    if explicit is not None:
        return format_nested_output_prefix(explicit)

    if not data_choice:
        return ""

    # Split orchestrator temp configs live under ``_tmp_split_*``; never derive
    # a prefix from those keys because the parent path is not meaningful.
    if data_choice.startswith("_tmp_split_"):
        return ""

    return format_nested_output_prefix(derive_output_prefix(data_choice))


def apply_output_prefix_to_config(cfg: DictConfig) -> None:
    """Materialize ``data.output_prefix`` on a composed Hydra config when unset.

    Called from ``OutputPrefixCallback.on_run_start`` so single-run configs
    persist the derived prefix in ``.hydra/config.yaml``. Multirun jobs rely
    on the ``nested_output_prefix`` resolver at path interpolation time.

    Parameters
    ----------
    cfg : DictConfig
        Fully composed Hydra configuration for the current job.
    """
    if "data" not in cfg:
        return

    data_cfg = cfg.data
    if _explicit_output_prefix(data_cfg) is not None:
        return

    data_choice: str | None = None
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        choice = OmegaConf.select(hydra_cfg, "runtime.choices.data")
        if choice is not None:
            data_choice = str(choice)
    except Exception:
        return

    if data_choice is None or data_choice.startswith("_tmp_split_"):
        return

    prefix = derive_output_prefix(data_choice)
    with open_dict(data_cfg):
        data_cfg.output_prefix = prefix

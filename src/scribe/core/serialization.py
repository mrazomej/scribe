"""Utilities for making SCRIBE objects pickle-safe.

This module centralizes small helpers used by multiple results classes to
sanitize runtime-only fields prior to pickling.
"""

from __future__ import annotations

from typing import Any
import pickle


def _is_picklable(value: Any) -> bool:
    """Return ``True`` when *value* can be serialized by ``pickle``."""
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


def make_model_config_pickle_safe(model_config: Any) -> Any:
    """Return a pickle-safe ``ModelConfig``-like object.

    Parameters
    ----------
    model_config : Any
        Model configuration object, typically ``ModelConfig``.

    Returns
    -------
    Any
        Original object when already picklable, otherwise a sanitized copy
        with callable-bearing fields removed when possible.
    """
    if model_config is None or _is_picklable(model_config):
        return model_config

    # Try progressively safer copies for pydantic-based configs.
    if hasattr(model_config, "model_copy"):
        for update in (
            {"param_specs": []},
            {"param_specs": [], "guide_families": None},
        ):
            try:
                candidate = model_config.model_copy(update=update)
                if _is_picklable(candidate):
                    return candidate
            except Exception:
                continue

    if hasattr(model_config, "model_dump"):
        try:
            payload = model_config.model_dump(mode="python")
            payload["param_specs"] = []
            payload["guide_families"] = None
            candidate = model_config.__class__(**payload)
            if _is_picklable(candidate):
                return candidate
        except Exception:
            pass

    # Last resort: return original object and let caller surface any error.
    return model_config

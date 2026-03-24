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
    # The ordering matters: preserve as much config as possible.
    #  1. Clear only param_specs (preserves guide_families fully).
    #  2. Clear param_specs + strip capture_amortization from
    #     guide_families (preserves flow guide markers that are needed
    #     for posterior reconstruction while dropping the potentially
    #     unpicklable AmortizationConfig).
    #  3. Clear both param_specs and guide_families (last resort).
    if hasattr(model_config, "model_copy"):
        updates_to_try = [{"param_specs": []}]

        # Build an intermediate update that sanitizes only the
        # amortization part of guide_families, keeping flow markers.
        gf = getattr(model_config, "guide_families", None)
        if gf is not None and hasattr(gf, "model_copy"):
            try:
                sanitized_gf = gf.model_copy(
                    update={"capture_amortization": None}
                )
                updates_to_try.append(
                    {"param_specs": [], "guide_families": sanitized_gf}
                )
            except Exception:
                pass

        updates_to_try.append({"param_specs": [], "guide_families": None})

        for update in updates_to_try:
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

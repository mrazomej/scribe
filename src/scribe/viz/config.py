"""Configuration extraction from Hydra/OmegaConf."""

import scribe

from .dispatch import _get_inference_metadata_for_filenames


def _get_config_values(cfg, results=None):
    """
    Extracts relevant configuration values from a Hydra/OmegaConf config object.

    Handles both legacy and current config structures. When results is provided
    and has n_components, that value takes precedence over the config.
    """
    if hasattr(cfg, "inference") and hasattr(cfg.inference, "parameterization"):
        parameterization = cfg.inference.parameterization
        n_components = cfg.inference.n_components or 1
        n_steps = getattr(
            cfg.inference, "n_steps", getattr(cfg.inference, "n_samples", 50000)
        )
        method = cfg.inference.method
    else:
        parameterization = cfg.get("parameterization", "standard")
        n_components = cfg.get("n_components") or 1
        n_steps = cfg.get("n_steps", 50000)
        method = cfg.inference.method if hasattr(cfg, "inference") else "svi"

    model_attr = getattr(cfg, "model", None)
    if model_attr is None:
        zero_inflation = (
            cfg.get("zero_inflation", False) if hasattr(cfg, "get") else False
        )
        variable_capture = (
            cfg.get("variable_capture", False) if hasattr(cfg, "get") else False
        )
        if zero_inflation and variable_capture:
            model_type = "zinbvcp"
        elif zero_inflation:
            model_type = "zinb"
        elif variable_capture:
            model_type = "nbvcp"
        else:
            model_type = "nbdm"
    elif isinstance(model_attr, str):
        model_type = model_attr
    elif hasattr(model_attr, "get"):
        model_type = model_attr.get("type", "default")
    else:
        model_type = "default"

    if results is not None:
        res_nc = getattr(results, "n_components", None)
        if res_nc is not None:
            n_components = res_nc

    typed_results = (
        results
        if isinstance(results, (scribe.ScribeSVIResults, scribe.ScribeMCMCResults))
        else None
    )
    inference_meta = _get_inference_metadata_for_filenames(typed_results, cfg)

    return {
        "parameterization": parameterization,
        "n_components": n_components,
        "n_steps": inference_meta["n_steps"],
        "run_size_label": inference_meta["run_size_label"],
        "run_size_value": inference_meta["run_size_value"],
        "run_size_token": inference_meta["run_size_token"],
        "method": method,
        "model_type": model_type,
    }

"""Utilities for constructing NumPyro optimizers from config.

This module converts a serialized optimizer specification from
``SVIConfig.optimizer_config`` into an instantiated NumPyro optimizer
object. It also enforces precedence between:

1. Explicit ``SVIConfig.optimizer`` (power-user object override).
2. Structured ``SVIConfig.optimizer_config`` (API/Hydra-friendly path).
3. Inference-engine defaults when neither is provided.
"""

from __future__ import annotations

from inspect import signature, Parameter
from typing import Any, Dict, Optional

import numpyro

from ..models.config import SVIConfig

# Supported optimizer registry. Values are attribute names on numpyro.optim.
_OPTIMIZER_REGISTRY: Dict[str, str] = {
    "adam": "Adam",
    "clipped_adam": "ClippedAdam",
    "adagrad": "Adagrad",
    "rmsprop": "RMSProp",
    "sgd": "SGD",
    "momentum": "Momentum",
}


def _constructor_kwargs(
    optimizer_cls: type, raw_kwargs: Dict[str, Any], optimizer_name: str
) -> Dict[str, Any]:
    """Validate kwargs against an optimizer constructor signature.

    Parameters
    ----------
    optimizer_cls : type
        NumPyro optimizer constructor type (e.g., ``numpyro.optim.Adam``).
    raw_kwargs : dict
        Candidate kwargs assembled from ``optimizer_config``.
    optimizer_name : str
        Optimizer name for error messages.

    Returns
    -------
    dict
        Filtered kwargs accepted by ``optimizer_cls``.

    Raises
    ------
    ValueError
        If unsupported kwargs are supplied for an optimizer that does not
        accept ``**kwargs``.
    """
    init_sig = signature(optimizer_cls.__init__)
    accepts_var_kwargs = any(
        param.kind == Parameter.VAR_KEYWORD
        for param in init_sig.parameters.values()
    )
    if accepts_var_kwargs:
        return raw_kwargs

    accepted = set(init_sig.parameters.keys()) - {"self"}
    unsupported = sorted(set(raw_kwargs.keys()) - accepted)
    if unsupported:
        raise ValueError(
            f"Unsupported kwargs for optimizer {optimizer_name!r}: {unsupported}. "
            f"Accepted kwargs: {sorted(accepted)}."
        )
    return {key: value for key, value in raw_kwargs.items() if key in accepted}


def build_optimizer_from_config(
    optimizer_config: SVIConfig.OptimizerConfig,
):
    """Build a NumPyro optimizer from a serialized optimizer config.

    Parameters
    ----------
    optimizer_config : SVIConfig.OptimizerConfig
        Structured optimizer specification.

    Returns
    -------
    Any
        Instantiated NumPyro optimizer object.

    Raises
    ------
    ValueError
        If the optimizer name is unsupported or incompatible kwargs are
        supplied.
    """
    optimizer_name = optimizer_config.name
    class_name = _OPTIMIZER_REGISTRY[optimizer_name]
    optimizer_cls = getattr(numpyro.optim, class_name, None)
    if optimizer_cls is None:
        raise ValueError(
            f"Optimizer {optimizer_name!r} is not available in numpyro.optim."
        )

    # Gather common typed fields first.
    kwargs: Dict[str, Any] = {}
    if optimizer_config.step_size is not None:
        kwargs["step_size"] = optimizer_config.step_size
    if optimizer_config.b1 is not None:
        kwargs["b1"] = optimizer_config.b1
    if optimizer_config.b2 is not None:
        kwargs["b2"] = optimizer_config.b2
    if optimizer_config.eps is not None:
        kwargs["eps"] = optimizer_config.eps
    if optimizer_config.weight_decay is not None:
        kwargs["weight_decay"] = optimizer_config.weight_decay

    # Map shared momentum field to the expected constructor key.
    if optimizer_config.momentum is not None:
        if optimizer_name == "momentum":
            kwargs["mass"] = optimizer_config.momentum
        else:
            kwargs["momentum"] = optimizer_config.momentum

    # Gradient clipping is only guaranteed for clipped_adam.
    if optimizer_config.grad_clip_norm is not None:
        if optimizer_name != "clipped_adam":
            raise ValueError(
                "grad_clip_norm is currently supported only with "
                "optimizer name 'clipped_adam'."
            )
        kwargs["clip_norm"] = optimizer_config.grad_clip_norm

    # Allow optimizer-specific passthrough kwargs from extra config fields.
    kwargs.update(optimizer_config.extra_kwargs())
    kwargs = _constructor_kwargs(optimizer_cls, kwargs, optimizer_name)
    return optimizer_cls(**kwargs)


def resolve_svi_optimizer(svi_config: SVIConfig) -> Optional[Any]:
    """Resolve the optimizer object to pass into the inference engine.

    Parameters
    ----------
    svi_config : SVIConfig
        SVI configuration object.

    Returns
    -------
    Optional[Any]
        Optimizer object when explicitly resolved from config; ``None`` when
        caller should rely on engine defaults.
    """
    # Explicit optimizer object is the highest-priority override.
    if svi_config.optimizer is not None:
        return svi_config.optimizer
    if svi_config.optimizer_config is not None:
        return build_optimizer_from_config(svi_config.optimizer_config)
    return None

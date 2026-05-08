"""
Stage 4: Build or validate InferenceConfig and resolve float64 mode.

Branches on inference method (SVI, MCMC, VAE, Laplace) to assemble the
appropriate config.  Handles SVI-to-MCMC initialization, KL annealing
defaults, and float64 precision resolution.

FitContext reads : model_config, n_cells, kwargs[inference_config,
                   inference_method, n_steps, batch_size, ...,
                   svi_init, enable_x64, guide_flow_log_det_f64, ...]
FitContext writes: inference_config, effective_x64
"""

import gc
import warnings
from typing import Any, Dict, Optional

import jax.numpy as jnp

from ...models.config import (
    EarlyStoppingConfig,
    InferenceConfig,
    KLAnnealingConfig,
    MCMCConfig,
    SVIConfig,
)
from ...models.config.enums import InferenceMethod
from ...inference.utils import validate_inference_config_match
from ..helpers import _coerce_batch_size_for_dataset
from ..context import FitContext


def build_inference_config(ctx: FitContext) -> None:
    """
    Build or validate the ``InferenceConfig`` for the fit and resolve
    whether float64 precision should be enabled.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.inference_config`` and
        ``ctx.effective_x64`` are set here.

    Raises
    ------
    ValueError
        If ``early_stopping`` has an unsupported type, or if
        ``kl_annealing`` has an unsupported type.
    TypeError
        If ``laplace_config`` has an unsupported type.
    """
    kw = ctx.kwargs
    inference_config = kw.get("inference_config")
    model_config = ctx.model_config

    if inference_config is not None:
        # User-supplied config: just validate compatibility.
        validate_inference_config_match(model_config, inference_config)
        ctx.inference_config = inference_config
        _resolve_x64(ctx)
        return

    method = model_config.inference_method
    effective_batch_size = _coerce_batch_size_for_dataset(
        batch_size=kw.get("batch_size"), n_cells=ctx.n_cells
    )

    # -- Process early_stopping -----------------------------------------------
    early_stopping = kw.get("early_stopping")
    early_stop_config = None
    if early_stopping is not None:
        if isinstance(early_stopping, EarlyStoppingConfig):
            early_stop_config = early_stopping
        elif isinstance(early_stopping, dict):
            early_stop_config = EarlyStoppingConfig(**early_stopping)
        else:
            raise ValueError(
                f"early_stopping must be EarlyStoppingConfig or dict, "
                f"got {type(early_stopping)}"
            )

    # -- Resolve KL annealing -------------------------------------------------
    kl_annealing = kw.get("kl_annealing")
    kl_annealing_warmup = kw.get("kl_annealing_warmup")
    kl_annealing_config: Optional[KLAnnealingConfig] = None

    if kl_annealing is not None:
        if isinstance(kl_annealing, KLAnnealingConfig):
            kl_annealing_config = kl_annealing
        elif isinstance(kl_annealing, bool):
            kl_annealing_config = (
                KLAnnealingConfig() if kl_annealing else None
            )
        elif isinstance(kl_annealing, dict):
            kl_annealing_config = KLAnnealingConfig(**kl_annealing)
        else:
            raise ValueError(
                "kl_annealing must be KLAnnealingConfig, dict, bool, "
                f"or None; got {type(kl_annealing).__name__}"
            )
    elif kl_annealing_warmup is not None:
        kl_annealing_config = KLAnnealingConfig(
            enabled=True, warmup=int(kl_annealing_warmup)
        )
    elif method == InferenceMethod.VAE:
        kl_annealing_config = KLAnnealingConfig()

    # -- Common SVI kwargs ----------------------------------------------------
    n_steps = kw.get("n_steps", 50_000)
    optimizer_config = kw.get("optimizer_config")
    stable_update = kw.get("stable_update", True)
    log_progress_lines = kw.get("log_progress_lines", False)
    restore_best = kw.get("restore_best", False)

    # -- Branch on inference method -------------------------------------------
    if method == InferenceMethod.SVI:
        if kl_annealing_config is not None:
            import logging as _kl_log
            _kl_log.getLogger(__name__).info(
                "KL annealing requested for plain SVI inference "
                "(method=%s, warmup=%d). Annealing is normally a "
                "VAE-mode optimisation; proceeding because the "
                "user passed an explicit kl_annealing argument.",
                method.value,
                kl_annealing_config.warmup,
            )
        svi_config = SVIConfig(
            n_steps=n_steps,
            batch_size=effective_batch_size,
            optimizer_config=optimizer_config,
            stable_update=stable_update,
            log_progress_lines=log_progress_lines,
            early_stopping=early_stop_config,
            restore_best=restore_best,
            kl_annealing=kl_annealing_config,
        )
        inference_config = InferenceConfig.from_svi(svi_config)

    elif method == InferenceMethod.MCMC:
        if early_stopping is not None:
            warnings.warn(
                "early_stopping is only supported for SVI and VAE "
                "inference methods. Ignoring for MCMC.",
                UserWarning,
            )
        if optimizer_config is not None:
            warnings.warn(
                "optimizer_config is only supported for SVI and VAE "
                "inference methods. Ignoring for MCMC.",
                UserWarning,
            )

        # Optionally inject SVI init strategy.
        svi_init = kw.get("svi_init")
        enable_x64 = kw.get("enable_x64")
        svi_init_kwargs: Dict[str, Any] = {}
        if svi_init is not None:
            from numpyro.infer.initialization import init_to_value
            from ...mcmc._init_from_svi import (
                clamp_init_values,
                compute_init_values,
            )

            if (
                svi_init.model_config.base_model
                != model_config.base_model
            ):
                warnings.warn(
                    f"SVI base model "
                    f"'{svi_init.model_config.base_model}' differs "
                    f"from MCMC target '{model_config.base_model}'.",
                    UserWarning,
                )
            same_param = (
                svi_init.model_config.parameterization
                == model_config.parameterization
            )
            svi_map = svi_init.get_map(
                use_mean=True, canonical=not same_param
            )
            init_values = (
                svi_map
                if same_param
                else compute_init_values(svi_map, model_config)
            )
            init_values = clamp_init_values(init_values)

            # Promote to float64 when MCMC will run under x64.
            if enable_x64 is not False:
                import jax
                with jax.enable_x64(True):
                    init_values = {
                        k: jnp.asarray(v, dtype=jnp.float64)
                        for k, v in init_values.items()
                    }

            svi_init_kwargs["init_strategy"] = init_to_value(
                values=init_values
            )
            del svi_map, init_values, svi_init
            gc.collect()

        mcmc_config = MCMCConfig(
            n_samples=kw.get("n_samples", 2_000),
            n_warmup=kw.get("n_warmup", 1_000),
            n_chains=kw.get("n_chains", 1),
            mcmc_kwargs=svi_init_kwargs or None,
        )
        inference_config = InferenceConfig.from_mcmc(mcmc_config)

    elif method == InferenceMethod.VAE:
        svi_config = SVIConfig(
            n_steps=n_steps,
            batch_size=effective_batch_size,
            optimizer_config=optimizer_config,
            stable_update=stable_update,
            log_progress_lines=log_progress_lines,
            early_stopping=early_stop_config,
            restore_best=restore_best,
            kl_annealing=kl_annealing_config,
        )
        inference_config = InferenceConfig.from_vae(svi_config)

    elif method == InferenceMethod.LAPLACE:
        from ...models.config import LaplaceConfig

        if kl_annealing_config is not None:
            warnings.warn(
                "kl_annealing has no effect under "
                "inference_method='laplace' (no encoder, no KL "
                "term to anneal). Ignoring.",
                UserWarning,
            )

        _base = dict(
            n_steps=n_steps,
            batch_size=effective_batch_size,
            optimizer_config=optimizer_config,
            early_stopping=early_stop_config,
            restore_best=restore_best,
            log_progress_lines=log_progress_lines,
        )
        laplace_config = kw.get("laplace_config")
        if laplace_config is None:
            _resolved = LaplaceConfig(**_base)
        elif isinstance(laplace_config, LaplaceConfig):
            _resolved = laplace_config
        elif isinstance(laplace_config, dict):
            _resolved = LaplaceConfig(**{**_base, **laplace_config})
        else:
            raise TypeError(
                "laplace_config must be a dict of overrides, a "
                "LaplaceConfig instance, or None; got "
                f"{type(laplace_config).__name__}."
            )
        inference_config = InferenceConfig.from_laplace(_resolved)

    else:
        raise ValueError(f"Unknown inference method: {method}")

    ctx.inference_config = inference_config
    _resolve_x64(ctx)


def _resolve_x64(ctx: FitContext) -> None:
    """Resolve the effective float64 precision flag."""
    kw = ctx.kwargs
    enable_x64 = kw.get("enable_x64")

    if enable_x64 is None:
        effective_x64 = (
            ctx.inference_config.method == InferenceMethod.MCMC
        )
    else:
        effective_x64 = enable_x64

    # Float64 log-det accumulation in flows requires x64 support.
    if kw.get("guide_flow_log_det_f64", False):
        effective_x64 = True

    ctx.effective_x64 = effective_x64

"""
Stage 3: Build or validate ModelConfig.

Assembles the model configuration from flat kwargs via
``build_config_from_preset``, then injects data-derived initializers
(expression anchor, LNM ALR bias, PLN PCA loadings).

FitContext reads : model, priors, n_components, n_datasets,
                   alr_reference_idx, effective_mixture_params,
                   count_data, _component_mapping,
                   kwargs[parameterization, unconstrained, ...,
                   model_config, inference_method, gene_coverage, ...]
FitContext writes: model_config
"""

import logging

from ...models.config import AmortizationConfig
from ...inference.preset_builder import build_config_from_preset
from ..context import FitContext

_log = logging.getLogger(__name__)


def build_model_config(ctx: FitContext) -> None:
    """
    Build or validate the ``ModelConfig`` for the fit.

    Parameters
    ----------
    ctx : FitContext
        Shared pipeline state.  ``ctx.model_config`` is set here.
    """
    kw = ctx.kwargs

    d_mode = kw.get("d_mode")
    if d_mode is None:
        d_mode = "learned"

    model_config = kw.get("model_config")

    if model_config is None:
        model_config = _build_from_kwargs(ctx, kw, d_mode)

    model_config = _post_process(ctx, kw, model_config)
    ctx.model_config = model_config


def _build_from_kwargs(ctx, kw, d_mode):
    """Build ModelConfig from flat kwargs via the preset builder."""
    capture_amortization = kw.get("capture_amortization")
    amortize_capture = kw.get("amortize_capture", False)
    effective_capture_amort = None
    if capture_amortization is not None:
        effective_capture_amort = (
            AmortizationConfig(**capture_amortization)
            if isinstance(capture_amortization, dict)
            else capture_amortization
        )
    elif amortize_capture:
        effective_capture_amort = AmortizationConfig(
            enabled=True,
            hidden_dims=kw.get("capture_hidden_dims") or [64, 32],
            activation=kw.get("capture_activation", "leaky_relu"),
            output_transform=kw.get(
                "capture_output_transform", "softplus"
            ),
            output_clamp_min=kw.get("capture_clamp_min", 0.1),
            output_clamp_max=kw.get("capture_clamp_max", 50.0),
        )

    return build_config_from_preset(
        model=ctx.model.lower(),
        parameterization=kw.get("parameterization", "canonical").lower(),
        inference_method=kw.get("inference_method", "svi").lower(),
        unconstrained=kw.get("unconstrained", False),
        expression_prior=kw.get("expression_prior", "none"),
        prob_prior=kw.get("prob_prior", "none"),
        zero_inflation_prior=kw.get("zero_inflation_prior", "none"),
        n_datasets=ctx.n_datasets,
        dataset_params=kw.get("dataset_params"),
        dataset_mixing=kw.get("dataset_mixing"),
        expression_dataset_prior=kw.get(
            "expression_dataset_prior", "none"
        ),
        prob_dataset_prior=kw.get("prob_dataset_prior", "none"),
        prob_dataset_mode=kw.get("prob_dataset_mode", "gene_specific"),
        zero_inflation_dataset_prior=kw.get(
            "zero_inflation_dataset_prior", "none"
        ),
        overdispersion_dataset_prior=kw.get(
            "overdispersion_dataset_prior", "none"
        ),
        horseshoe_tau0=kw.get("horseshoe_tau0", 1.0),
        horseshoe_slab_df=kw.get("horseshoe_slab_df", 4),
        horseshoe_slab_scale=kw.get("horseshoe_slab_scale", 2.0),
        neg_u=kw.get("neg_u", 1.0),
        neg_a=kw.get("neg_a", 1.0),
        neg_tau=kw.get("neg_tau", 1.0),
        capture_scaling_prior=kw.get("capture_scaling_prior", "none"),
        expression_anchor=kw.get("expression_anchor", False),
        expression_anchor_sigma=kw.get("expression_anchor_sigma", 0.3),
        overdispersion=kw.get("overdispersion", "none"),
        overdispersion_prior=kw.get("overdispersion_prior", "horseshoe"),
        d_mode=d_mode,
        alr_reference_idx=(
            ctx.alr_reference_idx
            if ctx.alr_reference_idx is not None
            else -1
        ),
        guide_rank=kw.get("guide_rank"),
        joint_params=kw.get("joint_params"),
        dense_params=kw.get("dense_params"),
        guide_flow=kw.get("guide_flow"),
        guide_flow_num_layers=kw.get("guide_flow_num_layers", 4),
        guide_flow_hidden_dims=kw.get("guide_flow_hidden_dims"),
        guide_flow_activation=kw.get("guide_flow_activation", "relu"),
        guide_flow_n_bins=kw.get("guide_flow_n_bins", 8),
        guide_flow_mixture_strategy=kw.get(
            "guide_flow_mixture_strategy", "independent"
        ),
        guide_flow_zero_init=kw.get("guide_flow_zero_init", True),
        guide_flow_layer_norm=kw.get("guide_flow_layer_norm", True),
        guide_flow_residual=kw.get("guide_flow_residual", True),
        guide_flow_soft_clamp=kw.get("guide_flow_soft_clamp", True),
        guide_flow_loft=kw.get("guide_flow_loft", True),
        guide_flow_log_det_f64=kw.get("guide_flow_log_det_f64", False),
        n_components=ctx.n_components,
        mixture_params=ctx.effective_mixture_params,
        priors=ctx.priors,
        vae_latent_dim=kw.get("vae_latent_dim", 10),
        vae_encoder_hidden_dims=kw.get("vae_encoder_hidden_dims"),
        vae_decoder_hidden_dims=kw.get("vae_decoder_hidden_dims"),
        vae_activation=kw.get("vae_activation"),
        vae_input_transform=kw.get("vae_input_transform", "log1p"),
        vae_standardize=kw.get("vae_standardize"),
        vae_decoder_transforms=kw.get("vae_decoder_transforms"),
        vae_flow_type=kw.get("vae_flow_type", "none"),
        vae_flow_num_layers=kw.get("vae_flow_num_layers", 4),
        vae_flow_hidden_dims=kw.get("vae_flow_hidden_dims"),
        amortize_capture=kw.get("amortize_capture", False),
        capture_hidden_dims=kw.get("capture_hidden_dims"),
        capture_activation=kw.get("capture_activation", "leaky_relu"),
        capture_output_transform=kw.get(
            "capture_output_transform", "softplus"
        ),
        capture_clamp_min=kw.get("capture_clamp_min", 0.1),
        capture_clamp_max=kw.get("capture_clamp_max", 50.0),
        capture_amortization=effective_capture_amort,
    )


def _post_process(ctx, kw, model_config):
    """Inject gene coverage, shared components, anchors, and VAE inits."""
    # Persist gene coverage threshold for reproducibility.
    gene_coverage = kw.get("gene_coverage")
    if gene_coverage is not None:
        model_config = model_config.model_copy(
            update={"gene_coverage": gene_coverage}
        )

    # Inject shared_component_indices from ComponentMapping.
    if ctx._component_mapping is not None:
        model_config = model_config.model_copy(
            update={
                "shared_component_indices": (
                    ctx._component_mapping.shared_indices
                ),
            }
        )

    # Compute data-informed mean anchor.
    if model_config.expression_anchor:
        model_config = _inject_expression_anchor(ctx, model_config)

    # LNM: inject data-derived VAE initializers.
    if (
        model_config.inference_method.value == "vae"
        and model_config.parameterization.value.startswith(
            "logistic_normal"
        )
    ):
        model_config = _inject_lnm_vae_init(ctx, model_config)

    # PLN: inject data-derived VAE initializers.
    if (
        model_config.inference_method.value == "vae"
        and model_config.parameterization.value == "poisson_lognormal"
    ):
        model_config = _inject_pln_vae_init(ctx, model_config)

    return model_config


def _inject_expression_anchor(ctx, model_config):
    """Compute and attach data-informed mu anchors."""
    from ...models.model_utils import compute_mu_anchor
    import numpy as _np

    _counts_np = _np.asarray(ctx.count_data)
    _lib_sizes = _counts_np.sum(axis=1)

    _extra = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    _eta_capture = _extra.get("eta_capture")
    _total_mrna = None
    if _eta_capture is not None:
        import math as _math
        _total_mrna = _math.exp(_eta_capture[0])

    _log_anchors = compute_mu_anchor(
        counts=_counts_np,
        library_sizes=_lib_sizes,
        total_mrna_mean=_total_mrna,
        epsilon=1e-3,
    )

    _updated_priors = dict(_extra)
    _updated_priors["mu_anchor_centers"] = _log_anchors
    from ...models.config.groups import PriorOverrides
    return model_config.model_copy(
        update={"priors": PriorOverrides(**_updated_priors)}
    )


def _inject_lnm_vae_init(ctx, model_config):
    """Inject empirical ALR bias and encoder stats for LNM VAE."""
    from ...core.lnm_data_init import inject_lnm_vae_data_init

    _ref = (
        ctx.alr_reference_idx
        if ctx.alr_reference_idx is not None
        else -1
    )
    model_config = inject_lnm_vae_data_init(
        model_config, ctx.count_data, alr_reference_idx=_ref
    )
    _log.info(
        "LNM: injected empirical ALR bias init (length %d, "
        "ref idx %d) and per-feature encoder standardization "
        "stats into VAEConfig.",
        int(model_config.vae.empirical_alr_bias_init.shape[0]),
        int(_ref),
    )
    return model_config


def _inject_pln_vae_init(ctx, model_config):
    """Inject empirical log-mean bias and PCA loadings for PLN VAE."""
    from ...core.pln_data_init import inject_pln_vae_data_init

    _latent_dim = model_config.vae.latent_dim
    model_config = inject_pln_vae_data_init(
        model_config, ctx.count_data, latent_dim=_latent_dim
    )
    _log.info(
        "PLN: injected empirical log-mean bias init (length %d), "
        "PCA loadings init %s, and encoder standardization stats "
        "into VAEConfig.",
        int(model_config.vae.empirical_log_mean_bias_init.shape[0]),
        (
            model_config.vae.pca_loadings_init.shape
            if model_config.vae.pca_loadings_init is not None
            else "None"
        ),
    )
    return model_config

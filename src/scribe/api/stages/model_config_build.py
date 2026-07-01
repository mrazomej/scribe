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
        # ``None`` lets ``build_config_from_preset`` apply the model-
        # aware default (``{"mu": "exp"}`` for TwoState, ``"softplus"``
        # for everyone else).  If the user explicitly passed
        # ``positive_transform=...`` to ``fit()``, that value is in
        # ``kw`` and overrides the default.
        positive_transform=kw.get("positive_transform"),
        expression_prior=kw.get("expression_prior", "none"),
        prob_prior=kw.get("prob_prior", "none"),
        zero_inflation_prior=kw.get("zero_inflation_prior", "none"),
        n_datasets=ctx.n_datasets,
        grouping_spec=ctx.grouping_spec,
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
        regime_dataset_prior=kw.get("regime_dataset_prior", "none"),
        regime_dataset_target=kw.get("regime_dataset_target"),
        overdispersion_dataset_independent=kw.get(
            "overdispersion_dataset_independent", True
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
        # Two-state Poisson-Beta quadrature node count; None keeps the
        # PoissonBetaCompound default (60).
        n_quad_nodes=kw.get("n_quad_nodes"),
        correlate_other_column=kw.get("correlate_other_column", False),
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

    # PLN / NBLN: inject data-derived VAE initializers. Both models share
    # the POISSON_LOGNORMAL parameterization (same y_log_rate decoder),
    # so we dispatch by ``base_model`` to pick the right helper.
    if (
        model_config.inference_method.value == "vae"
        and model_config.parameterization.value == "count_lognormal"
    ):
        _base = (
            model_config.base_model.value
            if hasattr(model_config.base_model, "value")
            else str(model_config.base_model)
        )
        if _base == "nbln":
            model_config = _inject_nbln_vae_init(ctx, model_config)
        else:
            model_config = _inject_pln_vae_init(ctx, model_config)

    # TwoState: stash an empirical mu prior loc so SVI starts in the
    # right gene-mean neighborhood. Without this, the default
    # softplus(Normal(0, 1)) prior is many orders of magnitude below
    # realistic gene means and SVI begins with NaN losses that take
    # thousands of steps to recover from.
    _base = (
        model_config.base_model.value
        if hasattr(model_config.base_model, "value")
        else str(model_config.base_model)
    )
    if _base in ("twostate", "twostatevcp"):
        model_config = _inject_twostate_data_init(ctx, model_config)

    # A cross-dataset / multi-factor additive hierarchy (on the mean and/or the
    # dispersion) is additive in the transform-inverse space, so only the exp
    # link yields the documented log-additive (log-fold-change) semantics. Force
    # it (warn on override).
    model_config = _force_exp_for_additive_hierarchy(model_config)

    return model_config


def _force_exp_for_additive_hierarchy(model_config):
    """Force the exp link on additive hierarchical targets (mean and dispersion).

    A cross-dataset / multi-factor hierarchy decomposes the *unconstrained*
    accumulator additively (``log_x^pop + sum_f alpha_f[level]``) and maps it to
    a positive value via ``positive_transform``. Only ``exp`` makes that
    accumulator log-scale, so the per-factor effects are interpretable
    **log-fold-changes** for the mean (``log mu = baseline + treatment + ...``)
    and **log-dispersion-ratios** for r (``log r``; the per-condition effects
    are exactly Delta log r). ``softplus`` would place the effects in
    softplus-inverse space — a different, unstated model. So when an additive
    hierarchy is active we force its target(s) to ``exp``, warning if the
    resolved transform was something else.
    """
    import warnings
    from ...models.config.enums import HierarchicalPriorType, Parameterization

    _NONE = HierarchicalPriorType.NONE

    def _expression_active() -> bool:
        edp = getattr(model_config, "expression_dataset_prior", None)
        if isinstance(edp, dict):
            if any(v not in (_NONE, "none") for v in edp.values()):
                return True
        elif edp is not None and edp != _NONE:
            return True
        gs = getattr(model_config, "grouping_spec", None)
        for f in getattr(gs, "factors", ()) or ():
            if f.family("expression") not in ("none", _NONE):
                return True
        return False

    def _dispersion_active() -> bool:
        gs = getattr(model_config, "grouping_spec", None)
        for f in getattr(gs, "factors", ()) or ():
            if f.family("dispersion") != "none":
                return True
        return False

    if not hasattr(model_config, "resolve_positive_transform"):
        return model_config

    targets = set()
    if _expression_active():
        # The expression mean is ``r`` under canonical/standard, ``mu``
        # otherwise; only that target carries the additive hierarchy.
        param = getattr(model_config, "parameterization", None)
        targets.add(
            "r"
            if param in (Parameterization.CANONICAL, Parameterization.STANDARD)
            else "mu"
        )
    if _dispersion_active():
        targets.add("r")

    targets = {
        t for t in targets if model_config.resolve_positive_transform(t) != "exp"
    }
    if not targets:
        return model_config

    for target in sorted(targets):
        warnings.warn(
            f"An additive hierarchy on {target!r} is active (log-additive): "
            f"forcing positive_transform[{target!r}]='exp' so the per-factor "
            "effects are interpretable log-fold-changes / log-dispersion-ratios "
            "(softplus would place them in softplus-inverse space — a different "
            "model). Pass positive_transform={...: 'exp'} to silence this.",
            stacklevel=2,
        )
    pt = model_config.positive_transform
    new_pt = dict(pt) if isinstance(pt, dict) else {}
    for target in targets:
        new_pt[target] = "exp"
    return model_config.model_copy(update={"positive_transform": new_pt})


def _inject_twostate_data_init(ctx, model_config):
    """Inject an empirical ``mu_prior_loc`` for TwoState models.

    Computes the median per-gene log-mean from the (gene-coverage-
    filtered) count matrix and stashes it as the prior loc for ``mu``
    on the priors extras. The factory reads this when building the
    parameterization's specs.
    """
    from ...core.twostate_data_init import inject_twostate_data_init

    import jax.numpy as _jnp

    # Honour an explicit user override on ``priors['mu']`` (or
    # ``priors['mu_prior_loc']``) by skipping the empirical anchor —
    # otherwise the user-supplied prior would be silently clobbered.
    # The factory always seeds ``model_config.priors.mu`` to its
    # default value, so we check ``ctx.kwargs['priors']`` (the
    # original user-passed dict) rather than the merged config.
    _user_priors = ctx.kwargs.get("priors") or {}
    if isinstance(_user_priors, dict) and (
        "mu" in _user_priors or "mu_prior_loc" in _user_priors
    ):
        _log.info(
            "TwoState: skipping the default data-driven mu_prior_loc "
            "init because priors['mu'] (or priors['mu_prior_loc']) was "
            "provided by the user."
        )
        return model_config

    model_config = inject_twostate_data_init(model_config, ctx.count_data)
    _extra = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    _anchor = _extra.get("mu_prior_loc")
    # ``mu_prior_loc`` is now a per-gene array (post-rev: data-driven
    # anchor); surface the summary range so the user can sanity-check
    # the magnitudes (especially important for high-expression genes
    # like ribosomal markers).
    _anchor_arr = _jnp.asarray(_anchor)
    # The anchor is for ``mu`` specifically; under the dict form of
    # ``positive_transform`` we resolve per-parameter so the log
    # message reports the transform actually applied to ``mu``.
    if hasattr(model_config, "resolve_positive_transform"):
        _transform = model_config.resolve_positive_transform("mu")
    else:
        _transform = getattr(model_config, "positive_transform", "softplus")
    # Surface the mean-capture estimate used to undo per-cell thinning
    # before anchoring (see core/twostate_data_init.py for the
    # closure-under-thinning rationale).
    _mean_capture = _extra.get("_mu_init_mean_capture")
    _capture_tag = (
        f" (pre-capture: observed_mean / {float(_mean_capture):.3f})"
        if _mean_capture is not None and float(_mean_capture) < 0.999
        else ""
    )
    if _anchor_arr.ndim == 0:
        _msg = (
            "TwoState: applied data-driven mu_prior_loc=%.3f from "
            "%d-gene count matrix (transform=%s%s).  Override by "
            "passing priors={'mu': (loc, scale)} to scribe.fit."
        ) % (
            float(_anchor_arr),
            ctx.count_data.shape[1] if ctx.count_data.ndim == 2 else -1,
            _transform,
            _capture_tag,
        )
        _log.info(_msg)
    else:
        _msg = (
            "TwoState: applied per-gene mu_prior_loc to %d genes "
            "(transform=%s%s, unconstrained-space range: min=%.2f, "
            "median=%.2f, max=%.2f).  Each gene's variational mu_loc "
            "starts at its (pre-capture) empirical mean; without this "
            "anchor highly-expressed genes can be slow to recover "
            "under SVI.  Override the data-driven anchor by passing "
            "priors={'mu': (loc, scale)} to scribe.fit."
        ) % (
            int(_anchor_arr.size),
            _transform,
            _capture_tag,
            float(_jnp.min(_anchor_arr)),
            float(_jnp.median(_anchor_arr)),
            float(_jnp.max(_anchor_arr)),
        )
        _log.info(_msg)
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


def _inject_nbln_vae_init(ctx, model_config):
    """Inject NBLN data-derived VAE initializers.

    Reuses the PLN helpers for the decoder bias, PCA loadings, and
    encoder standardization stats (identical between PLN and NBLN
    because both share the y_log_rate decoder), and additionally
    computes a method-of-moments estimator for the gene dispersion
    ``r_g`` and stashes it on the priors extra payload.
    """
    from ...core.nbln_data_init import inject_nbln_vae_data_init

    _latent_dim = model_config.vae.latent_dim
    model_config = inject_nbln_vae_data_init(
        model_config, ctx.count_data, latent_dim=_latent_dim
    )
    _extra = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    _r_init = _extra.get("empirical_r_init")
    _log.info(
        "NBLN: injected empirical log-mean bias init (length %d), "
        "PCA loadings init %s, encoder standardization stats, and "
        "method-of-moments r_g estimator (length %s) into ModelConfig.",
        int(model_config.vae.empirical_log_mean_bias_init.shape[0]),
        (
            model_config.vae.pca_loadings_init.shape
            if model_config.vae.pca_loadings_init is not None
            else "None"
        ),
        int(_r_init.shape[0]) if _r_init is not None else "None",
    )
    return model_config

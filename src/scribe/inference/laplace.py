"""Bridge between the dispatcher and the Laplace engine.

Mirrors the structure of ``inference/vae.py`` but routes to
:class:`LaplaceInferenceEngine`, which runs a custom outer-loop
training (not NumPyro SVI). The engine returns a
:class:`LaplaceRunResult`; this bridge wraps it in a
:class:`ScribeLaplaceResults` for downstream packaging consistency
with the VAE path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import jax.numpy as jnp
import numpy as np

from ..models.config import DataConfig, LaplaceConfig, ModelConfig
from ..laplace import LaplaceInferenceEngine, ScribeLaplaceResults
from ..laplace._global_uncertainty import resolve_positive_fns

if TYPE_CHECKING:
    from anndata import AnnData


def _run_laplace_inference(
    model_config: ModelConfig,
    count_data: jnp.ndarray,
    adata: Optional["AnnData"],
    n_cells: int,
    n_genes: int,
    laplace_config: LaplaceConfig,
    data_config: DataConfig,
    seed: int,
    informative_priors: Optional[Dict[str, Dict[str, Any]]] = None,
    capture_mode_override: Optional[str] = None,
    freeze_values: Optional[Dict[str, Dict[str, Any]]] = None,
    freeze_params: tuple = (),
    cascade_source: Optional[Any] = None,
    cascade_source_counts: Optional[jnp.ndarray] = None,
    w_prior: Optional[Dict[str, Any]] = None,
) -> ScribeLaplaceResults:
    """Run Laplace inference (PLN or LNM) and package results.

    Dispatches on ``model_config.base_model``:

    * ``"pln"`` → PLN Laplace path. Per-cell MAP over log-rates
      ``x_c`` (and optionally capture offset ``eta_c``).
    * ``"lnm"`` → LNM Laplace path. Per-cell MAP over factor
      scores ``z_c`` (when ``d_mode='low_rank'``) or ALR logits
      ``y_alr_c`` (when ``d_mode='learned'``).

    The engine returns a generic ``LaplaceRunResult``; this bridge
    routes the per-cell latent into the right slot of the single
    ``ScribeLaplaceResults`` class based on ``base_model`` + ``d_mode``.

    Parameters
    ----------
    model_config : ModelConfig
        Either a PLN or LNM model.
    count_data, n_cells, n_genes, laplace_config, data_config, seed
        Standard inference-handler signature.

    Returns
    -------
    ScribeLaplaceResults
        Trained globals + per-cell MAP + diagnostics.

    Raises
    ------
    ValueError
        If ``model_config.base_model`` is not in ``{"pln", "lnm"}``.
    """
    base_model = getattr(model_config, "base_model", None)
    # Single source of truth: ``api.constants.LAPLACE_SUPPORTED_BASE_MODELS``.
    # See that constant's docstring for the design rationale.
    from ..api.constants import LAPLACE_SUPPORTED_BASE_MODELS

    if base_model not in LAPLACE_SUPPORTED_BASE_MODELS:
        raise ValueError(
            f"inference_method='laplace' is supported for "
            f"{sorted(LAPLACE_SUPPORTED_BASE_MODELS)} "
            f"(got base_model={base_model!r}). "
            "Use 'svi'/'vae'/'mcmc' for other models."
        )

    # Pull the latent dim from VAEConfig (factories still use the
    # VAE config for the linear-decoder structure). Default 32.
    latent_dim = (
        getattr(getattr(model_config, "vae", None), "latent_dim", None)
        or 32
    )

    # Capture anchor: extracted from the biology-informed prior.
    # PLN reads from priors.eta_capture; LNMVCP reads from the same
    # field (see preset_builder.py — both alias to ``eta_capture``
    # via PRIOR_KEY_ALIASES). Plain LNM has no capture submodel so
    # no anchor is set.
    capture_anchor = None
    if base_model in ("pln", "nbln", "lnmvcp", "twostate_ln_rate"):
        priors_extra = (
            getattr(model_config.priors, "__pydantic_extra__", None) or {}
        )
        eta_capture = priors_extra.get("eta_capture")
        if eta_capture is not None:
            capture_anchor = (float(eta_capture[0]), float(eta_capture[1]))

    # LNMVCP is a *capture-aware* model by definition; running Laplace
    # without a biology-informed capture prior would silently degrade
    # to the plain LNM path and the user would lose the per-cell
    # capture latent without warning. Fail loudly with a constructive
    # error pointing at the right priors invocation.
    if base_model == "lnmvcp" and capture_anchor is None:
        raise ValueError(
            "model='lnmvcp' with inference_method='laplace' requires a "
            "biology-informed capture prior. Pass\n"
            "    priors={'capture_efficiency': (np.log(M_0), sigma_M)}\n"
            "where M_0 is your guess of total mRNA per cell and sigma_M "
            "controls the prior tightness. If you actually want the no-"
            "capture variant, use model='lnm' instead."
        )

    # SVI-informative-prior cascade: when the source provides per-cell
    # eta_capture (capture_mode_override == "eta"), the SVI per-cell
    # prior supersedes the target's scalar (log_M0, sigma_M) anchor.
    # Other modes ("phi_only", "none") leave the target capture
    # configuration intact — see the run_inference stage for the
    # detection logic.
    if capture_mode_override == "eta":
        capture_anchor = None

    run_result = LaplaceInferenceEngine.run_inference(
        model_config=model_config,
        count_data=count_data,
        n_cells=n_cells,
        n_genes=n_genes,
        latent_dim=int(latent_dim),
        laplace_config=laplace_config,
        seed=seed,
        capture_anchor=capture_anchor,
        progress=True,
        progress_backend="auto",
        log_progress_lines=laplace_config.log_progress_lines,
        informative_priors=informative_priors,
        freeze_values=freeze_values,
        freeze_params=freeze_params,
        w_prior=w_prior,
    )

    g = run_result.globals
    # For LNM low_rank d_mode the loss does not optimise ``d`` (the
    # latent prior is N(0, I_k) and the decoder has no diagonal
    # residual); store ``d`` as zeros so downstream PPCs and
    # distribution accessors don't multiply in an un-fitted variance
    # term. For PLN and LNM learned, ``d`` is fitted as usual.
    # Resolve the same positive transform used during training so
    # that constrained globals are derived consistently.  Use the
    # per-parameter resolver to handle dict-form ``positive_transform``
    # (audit round-7: when the user passes
    # ``positive_transform={"mean_expression": "exp"}`` the field is a
    # dict, and the legacy ``resolve_positive_fns(model_config)`` call
    # fails because it tries to use the dict as a hashable key).
    from ..laplace._global_uncertainty import _JAX_POSITIVE_FNS

    if hasattr(model_config, "resolve_positive_transform"):
        _d_transform = model_config.resolve_positive_transform("d")
    else:
        _d_transform = getattr(model_config, "positive_transform", "softplus")
        if not isinstance(_d_transform, str):
            _d_transform = "softplus"
    pos_forward, _ = _JAX_POSITIVE_FNS[_d_transform]

    d_mode = getattr(model_config, "d_mode", "learned") or "learned"
    if base_model in ("lnm", "lnmvcp") and d_mode == "low_rank":
        d_value = jnp.zeros_like(g["mu"])
    else:
        d_value = pos_forward(g["d_loc"])

    common_kwargs = dict(
        model_config=run_result.model_config,
        mu=g["mu"],
        W=g["W"],
        d=d_value,
        final_grad_norms=run_result.final_grad_norms,
        losses=run_result.losses,
        n_genes=int(n_genes),
        n_cells=int(n_cells),
        early_stopped=run_result.early_stopped,
        best_loss=run_result.best_loss,
        stopped_at_step=run_result.stopped_at_step,
        divergence_aborted=run_result.divergence_aborted,
        # Phase-3: propagate W-prior diagnostics dict from the engine.
        # Always present for PLN/NBLN (NoneWPrior populates a minimal
        # dict); None for LNM-family in v1.
        w_prior_diagnostics=run_result.w_prior_diagnostics,
    )

    # Populate NB-on-totals globals when the LNM family fitted them
    # (which is now always the case after the v1.1 audit fixes; PLN
    # has no mu_T_loc / r_T_loc).
    if "mu_T_loc" in g and "r_T_loc" in g:
        common_kwargs["mu_T"] = pos_forward(g["mu_T_loc"])
        common_kwargs["r_T"] = pos_forward(g["r_T_loc"])

    if base_model == "pln":
        # PLN: latent is x_c; eta_c populated when capture anchor on.
        return ScribeLaplaceResults(
            **common_kwargs,
            x_loc=run_result.x_loc,
            eta_loc=run_result.eta_loc,
        )

    if base_model == "nbln":
        # NBLN: same per-cell shape as PLN (x_loc is the log-rate
        # MAP, eta_loc the optional capture offset), plus gene-specific
        # dispersion from the unconstrained r_loc coordinate.
        gu = run_result.global_uncertainty
        r_loc_val = g.get("r_loc")
        r_value = pos_forward(r_loc_val) if r_loc_val is not None else None

        # Phase-2 freeze: when params are frozen, compute_global_uncertainty
        # writes NaN sentinels for r_scale / mu_scale.  Replace with
        # moment-matched values from cascade_source samples so PPC
        # paths and other consumers see a usable Normal posterior in
        # NBLN target coordinate.  Full-fidelity SVI guide is accessed
        # via cascade_source / get_distributions().
        r_scale_val = gu.get("r_scale")
        mu_loc_val = gu.get("mu_loc")
        mu_scale_val = gu.get("mu_scale")
        if cascade_source is not None and run_result.frozen_params:
            r_scale_val, mu_loc_val, mu_scale_val = (
                _moment_match_frozen_for_nbln(
                    cascade_source=cascade_source,
                    cascade_counts=cascade_source_counts,
                    frozen_params=run_result.frozen_params,
                    pos_forward=pos_forward,
                    model_config=run_result.model_config,
                    r_scale_fallback=r_scale_val,
                    mu_loc_fallback=mu_loc_val,
                    mu_scale_fallback=mu_scale_val,
                )
            )

        # Round-5 R5-5: cascade fields live only on the bridge-level
        # result.  `cascade_source` carries the SVI guide for PPC and
        # `get_distributions()` routing; `cascade_source_counts` caches
        # counts for amortized SVI sources (set upstream in
        # api/stages/run_inference.py when freezing is active).
        # `frozen_params` propagates from run_result so downstream
        # consumers can identify cascade-bound parameters.
        return ScribeLaplaceResults(
            **common_kwargs,
            x_loc=run_result.x_loc,
            eta_loc=run_result.eta_loc,
            r=r_value,
            r_loc=r_loc_val,
            r_scale=r_scale_val,
            mu_loc=mu_loc_val,
            mu_scale=mu_scale_val,
            frozen_params=run_result.frozen_params,
            cascade_source=cascade_source,
            cascade_source_counts=cascade_source_counts,
        )

    if base_model == "twostate_ln_rate":
        # TSLN-Rate: same per-cell shape as NBLN (x_loc is the latent
        # log-rate MAP, eta_loc the optional capture offset).  Three
        # gene-level positive globals (gene_mean, burst_size, k_off) plus
        # the derived (alpha, beta, r_hat).
        #
        # CONVENTION (round-5 audit fix): override ``common_kwargs["mu"]``
        # to ``log(r_hat)`` so ``self.mu`` carries the latent log-rate
        # prior center — matching NBLN/PLN where every Laplace mixin
        # treats ``self.mu`` as the loc of the latent log-rate
        # distribution.  The TwoState positive ``mu`` (gene mean) is
        # routed into the new ``gene_mean`` field instead.  This
        # prevents mixins like ``get_distributions``, ``get_map``, and
        # downstream PPC paths from quietly using the wrong coordinate
        # system for ``y_log_rate``.
        gu = run_result.global_uncertainty
        r_hat_val = g.get("r_hat")
        if r_hat_val is not None:
            common_kwargs["mu"] = jnp.log(jnp.maximum(r_hat_val, 1e-30))
        # ``mu_loc / mu_scale`` are NBLN-specific (post-fit Laplace
        # Normal on the latent log-rate prior mean).  For TSLN-Rate we
        # explicitly DO NOT populate them — the analogous quantities
        # for the TwoState parameterization live on
        # ``gene_mean_loc / gene_mean_scale``.
        return ScribeLaplaceResults(
            **common_kwargs,
            x_loc=run_result.x_loc,
            eta_loc=run_result.eta_loc,
            # Constrained positives (computed in pack_result via pos_forward).
            gene_mean=g.get("mu"),  # TwoState positive mu, now routed here
            burst_size=g.get("burst_size"),
            k_off=g.get("k_off"),
            # Derived TSLN quantities.
            alpha=g.get("alpha"),
            beta=g.get("beta"),
            r_hat=r_hat_val,
            # Unconstrained loc/scale fields from the global-uncertainty hook.
            # NB: ``self.mu_loc / self.mu_scale`` left as None for TSLN-Rate.
            gene_mean_loc=g.get("mu_loc"),
            gene_mean_scale=gu.get("mu_scale"),
            burst_size_loc=g.get("burst_size_loc"),
            burst_size_scale=gu.get("burst_size_scale"),
            k_off_loc=g.get("k_off_loc"),
            k_off_scale=gu.get("k_off_scale"),
            # Curvature-clamp diagnostics from pack_result.
            a_raw_min=g.get("a_raw_min"),
            a_raw_negative_fraction=g.get("a_raw_negative_fraction"),
            a_clamp_fraction=g.get("a_clamp_fraction"),
            a_clamp_per_gene=g.get("a_clamp_per_gene"),
            # Cascade plumbing (mirrors NBLN).
            frozen_params=run_result.frozen_params,
            cascade_source=cascade_source,
            cascade_source_counts=cascade_source_counts,
        )

    # LNM / LNMVCP: route the per-cell latent (the engine packed it
    # into ``run_result.x_loc`` for transit) into either ``z_loc``
    # or ``y_alr_loc`` based on ``d_mode``. For LNMVCP, also
    # populate ``p_capture_loc`` from the engine's eta MAP
    # (p_capture = exp(-eta_capture)) and propagate the ALR
    # reference index so PPC sampling and gene subsetting know
    # which gene serves as the gauge fix.
    d_mode = getattr(model_config, "d_mode", "learned") or "learned"
    alr_reference_idx = int(getattr(model_config, "alr_reference_idx", -1))

    # Propagate global uncertainty for the totals block.
    gu = run_result.global_uncertainty
    extras: dict = {}

    # LNMVCP-specific extras: store both the raw eta_capture MAP
    # (the actual latent) and the derived p_capture = exp(-eta).
    if base_model == "lnmvcp" and run_result.eta_loc is not None:
        extras["eta_loc"] = run_result.eta_loc
        extras["p_capture_loc"] = jnp.exp(-run_result.eta_loc)

    # Totals uncertainty fields from the global Laplace hook.
    if "mu_T_loc" in g:
        extras["mu_T_loc"] = g["mu_T_loc"]
    if "r_T_loc" in g:
        extras["r_T_loc"] = g["r_T_loc"]
    if "totals_cov" in gu:
        extras["totals_cov"] = gu["totals_cov"]
    if "mu_T_scale" in gu:
        extras["mu_T_scale"] = gu["mu_T_scale"]
    if "r_T_scale" in gu:
        extras["r_T_scale"] = gu["r_T_scale"]

    if d_mode == "low_rank":
        return ScribeLaplaceResults(
            **common_kwargs,
            z_loc=run_result.x_loc,
            alr_reference_idx=alr_reference_idx,
            **extras,
        )
    return ScribeLaplaceResults(
        **common_kwargs,
        y_alr_loc=run_result.x_loc,
        alr_reference_idx=alr_reference_idx,
        **extras,
    )


def _moment_match_frozen_for_nbln(
    *,
    cascade_source: Any,
    cascade_counts: Optional[jnp.ndarray],
    frozen_params: frozenset,
    pos_forward,
    model_config: ModelConfig,
    r_scale_fallback: Optional[jnp.ndarray],
    mu_loc_fallback: Optional[jnp.ndarray],
    mu_scale_fallback: Optional[jnp.ndarray],
) -> tuple:
    """Moment-match SVI samples into NBLN target coord for r and mu.

    Called at result-packaging time when one or more NBLN parameters
    were frozen at the SVI MAP during the M-step.  Replaces the
    NaN sentinels in ``compute_global_uncertainty`` with usable
    Gaussian summaries so PPC, ``get_distributions``, and other
    downstream consumers see a coherent ``Normal(loc, scale)`` in
    NBLN's target coordinate.

    Full SVI-guide fidelity is still available via
    ``ScribeLaplaceResults.cascade_source.get_distributions()`` /
    ``.get_posterior_samples()``.  This helper provides the simpler
    Gaussian summary the existing PPC paths expect.
    """
    from scribe.laplace._global_uncertainty import resolve_positive_fns

    _pos_fwd, pos_inv = resolve_positive_fns(model_config)

    # Resolve counts for amortized SVI sources.
    counts = cascade_counts
    if counts is None:
        counts = getattr(cascade_source, "_original_counts", None)
    sample_kwargs = {"n_samples": 1000, "store_samples": False}
    if counts is not None:
        sample_kwargs["counts"] = counts
    svi_samples = cascade_source.get_posterior_samples(**sample_kwargs)

    r_scale = r_scale_fallback
    mu_loc = mu_loc_fallback
    mu_scale = mu_scale_fallback

    if "r" in frozen_params and "r" in svi_samples:
        r_pos = jnp.asarray(svi_samples["r"])  # (S, G)
        r_uncon = pos_inv(jnp.maximum(r_pos, 1e-8))
        r_scale = jnp.std(r_uncon, axis=0, ddof=1).astype(jnp.float32)

    if "mu" in frozen_params and "mu" in svi_samples:
        mu_pos = jnp.asarray(svi_samples["mu"])  # (S, G)
        mu_log = jnp.log(jnp.maximum(mu_pos, 1e-8))
        mu_loc = jnp.mean(mu_log, axis=0).astype(jnp.float32)
        mu_scale = jnp.std(mu_log, axis=0, ddof=1).astype(jnp.float32)

    return r_scale, mu_loc, mu_scale


__all__ = ["_run_laplace_inference"]

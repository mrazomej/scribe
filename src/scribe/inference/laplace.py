"""Bridge between the dispatcher and the Laplace engine.

Mirrors the structure of ``inference/vae.py`` but routes to
:class:`LaplaceInferenceEngine`, which runs a custom outer-loop
training (not NumPyro SVI). The engine returns a
:class:`LaplaceRunResult`; this bridge wraps it in a
:class:`ScribeLaplaceResults` for downstream packaging consistency
with the VAE path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np

from ..models.config import DataConfig, LaplaceConfig, ModelConfig
from ..laplace import LaplaceInferenceEngine, ScribeLaplaceResults

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
    if base_model not in ("pln", "lnm", "lnmvcp"):
        raise ValueError(
            f"inference_method='laplace' is currently supported for "
            f"PLN, LNM, and LNMVCP (got base_model={base_model!r}). "
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
    if base_model in ("pln", "lnmvcp"):
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
    )

    g = run_result.globals
    # For LNM low_rank d_mode the loss does not optimise ``d`` (the
    # latent prior is N(0, I_k) and the decoder has no diagonal
    # residual); store ``d`` as zeros so downstream PPCs and
    # distribution accessors don't multiply in an un-fitted variance
    # term. For PLN and LNM learned, ``d`` is fitted as usual.
    d_mode = getattr(model_config, "d_mode", "learned") or "learned"
    if base_model in ("lnm", "lnmvcp") and d_mode == "low_rank":
        d_value = jnp.zeros_like(g["mu"])
    else:
        d_value = jnp.exp(g["d_log"])

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
    )

    # Populate NB-on-totals globals when the LNM family fitted them
    # (which is now always the case after the v1.1 audit fixes; PLN
    # has no log_mu_T / log_r_T).
    if "log_mu_T" in g and "log_r_T" in g:
        common_kwargs["mu_T"] = jnp.exp(g["log_mu_T"])
        common_kwargs["r_T"] = jnp.exp(g["log_r_T"])

    if base_model == "pln":
        # PLN: latent is x_c; eta_c populated when capture anchor on.
        return ScribeLaplaceResults(
            **common_kwargs,
            x_loc=run_result.x_loc,
            eta_loc=run_result.eta_loc,
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

    # LNMVCP-specific extras: store both the raw eta_capture MAP
    # (the actual latent) and the derived p_capture = exp(-eta).
    # ``eta_loc`` is shared with the PLN dataclass slot (it carries
    # the per-cell capture-offset latent in either model); the
    # ``p_capture_loc`` slot is the natural human-readable view.
    extras: dict = {}
    if base_model == "lnmvcp" and run_result.eta_loc is not None:
        extras["eta_loc"] = run_result.eta_loc
        extras["p_capture_loc"] = jnp.exp(-run_result.eta_loc)

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


__all__ = ["_run_laplace_inference"]

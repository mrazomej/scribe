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
    if base_model not in ("pln", "lnm"):
        raise ValueError(
            f"inference_method='laplace' is currently supported for "
            f"PLN and LNM (got base_model={base_model!r}). Use "
            "'svi'/'vae'/'mcmc' for other models."
        )

    # Pull the latent dim from VAEConfig (factories still use the
    # VAE config for the linear-decoder structure). Default 32.
    latent_dim = (
        getattr(getattr(model_config, "vae", None), "latent_dim", None)
        or 32
    )

    # Capture anchor: PLN-only for now. The LNM Laplace v1 does not
    # support LNMVCP capture priors; the user must use VAE inference
    # if they need per-cell capture amortization.
    capture_anchor = None
    if base_model == "pln":
        priors_extra = (
            getattr(model_config.priors, "__pydantic_extra__", None) or {}
        )
        eta_capture = priors_extra.get("eta_capture")
        if eta_capture is not None:
            capture_anchor = (float(eta_capture[0]), float(eta_capture[1]))

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
    common_kwargs = dict(
        model_config=run_result.model_config,
        mu=g["mu"],
        W=g["W"],
        d=jnp.exp(g["d_log"]),
        final_grad_norms=run_result.final_grad_norms,
        losses=run_result.losses,
        n_genes=int(n_genes),
        n_cells=int(n_cells),
        early_stopped=run_result.early_stopped,
        best_loss=run_result.best_loss,
        stopped_at_step=run_result.stopped_at_step,
    )

    if base_model == "pln":
        # PLN: latent is x_c; eta_c populated when capture anchor on.
        return ScribeLaplaceResults(
            **common_kwargs,
            x_loc=run_result.x_loc,
            eta_loc=run_result.eta_loc,
        )

    # LNM: route the per-cell latent (the engine packed it into
    # ``run_result.x_loc`` for transit) into either ``z_loc`` or
    # ``y_alr_loc`` based on ``d_mode``. Also propagate the ALR
    # reference index so PPC sampling and gene subsetting know
    # which gene serves as the gauge fix.
    d_mode = getattr(model_config, "d_mode", "learned") or "learned"
    alr_reference_idx = int(getattr(model_config, "alr_reference_idx", -1))
    if d_mode == "low_rank":
        return ScribeLaplaceResults(
            **common_kwargs,
            z_loc=run_result.x_loc,
            alr_reference_idx=alr_reference_idx,
        )
    return ScribeLaplaceResults(
        **common_kwargs,
        y_alr_loc=run_result.x_loc,
        alr_reference_idx=alr_reference_idx,
    )


__all__ = ["_run_laplace_inference"]

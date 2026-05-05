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
from ..svi.laplace_engine import LaplaceInferenceEngine
from ..svi.laplace_results import ScribeLaplaceResults

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
    """Run PLN Laplace inference and package results.

    Parameters
    ----------
    model_config : ModelConfig
        Must be a PLN model (``base_model="pln"``). Carries the VAE
        latent dimension and any capture-anchor priors.
    count_data : jnp.ndarray
        Filtered count matrix (post ``gene_coverage``, etc.).
    adata : AnnData, optional
        For downstream metadata only; the engine does not consult it.
    n_cells, n_genes : int
        Dataset dimensions.
    laplace_config : LaplaceConfig
        Outer-loop and Newton hyperparameters.
    data_config : DataConfig
        Data-loading config (unused here; kept for signature parity
        with the VAE handler).
    seed : int
        JAX PRNG seed.

    Returns
    -------
    ScribeLaplaceResults
        Trained globals + per-cell MAP + diagnostics.

    Raises
    ------
    ValueError
        If ``model_config`` is not for a PLN model.
    """
    base_model = getattr(model_config, "base_model", None)
    if base_model != "pln":
        raise ValueError(
            f"inference_method='laplace' is currently PLN-only "
            f"(got base_model={base_model!r}). Use 'svi'/'vae'/'mcmc' "
            "for other models."
        )

    # Pull the latent dim from VAEConfig (the PLN factory still uses
    # the VAE config for the linear-decoder structure). Default 32 so
    # we have something sensible if vae is None.
    latent_dim = (
        getattr(getattr(model_config, "vae", None), "latent_dim", None)
        or 32
    )

    # Capture anchor: detect via priors as in the factory.
    capture_anchor = None
    priors_extra = (
        getattr(model_config.priors, "__pydantic_extra__", None) or {}
    )
    eta_capture = priors_extra.get("eta_capture")
    if eta_capture is not None:
        capture_anchor = (float(eta_capture[0]), float(eta_capture[1]))

    # Run the engine. ``progress_backend="auto"`` matches the SVI/VAE
    # paths: rich in terminals, tqdm in notebooks, no-op when stdout
    # isn't a TTY. ``log_progress_lines`` is forwarded from the
    # ``LaplaceConfig`` so users get the same plain-text-line opt-in
    # they have for SVI.
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

    # Pack into ScribeLaplaceResults.
    g = run_result.globals
    return ScribeLaplaceResults(
        model_config=run_result.model_config,
        mu=g["mu"],
        W=g["W"],
        d=jnp.exp(g["d_log"]),
        x_loc=run_result.x_loc,
        eta_loc=run_result.eta_loc,
        final_grad_norms=run_result.final_grad_norms,
        losses=run_result.losses,
        n_genes=int(n_genes),
        n_cells=int(n_cells),
    )


__all__ = ["_run_laplace_inference"]

"""Public Laplace-mode inference entry point.

Thin dispatcher that routes :func:`scribe.api.fit` calls into the
generic Laplace-EM driver in :mod:`scribe.laplace._em`. All the
shared scaffolding (outer Adam, mini-batching, divergence detection,
best-snapshot recording, smoothed-loss patience early stopping, orbax
checkpoint save/resume, final convergence check) lives in the driver;
all the model-specific machinery (initial state, Newton-kernel calls,
per-block ELBO, result packaging) lives in observation-model adapters
under :mod:`scribe.laplace._obs_pln`, :mod:`._obs_nbln`, and
:mod:`._obs_lnm`. Adding a new Laplace-supported model is a matter of
writing one such adapter and adding an ``elif`` here -- there is no
engine-level scaffolding to duplicate.

Public API
----------
- :class:`LaplaceRunResult` -- result container mirroring
  ``SVIRunResult``. Re-exported from :mod:`scribe.laplace._em`.
- :class:`LaplaceInferenceEngine` -- static-method entry point
  analogous to ``SVIInferenceEngine``.

Both are kept signature-compatible with the SVI counterparts so that
``ScribeLaplaceResults`` can reuse most of ``ScribeVAEResults``'s
interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from ..models.config import LaplaceConfig, ModelConfig
from ..svi._progress_backend import ProgressBackendName

# LaplaceRunResult lives in :mod:`scribe.laplace._em` so that the
# generic driver and every observation-model adapter produce the same
# dataclass shape. Re-exported here so existing imports
# ``from scribe.laplace.engine import LaplaceRunResult`` keep working.
from ._em import LaplaceRunResult  # noqa: F401


class LaplaceInferenceEngine:
    """Outer SVI + inner Newton training loop dispatcher.

    Static-method entry point analogous to ``SVIInferenceEngine``.
    Resolves the per-model :class:`LaplaceObservationModel` adapter
    and delegates to :func:`scribe.laplace._em.run_laplace_em` for
    the actual training loop.
    """

    @staticmethod
    def run_inference(
        model_config: ModelConfig,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        latent_dim: int,
        laplace_config: LaplaceConfig,
        seed: int = 42,
        capture_anchor: Optional[Tuple[float, float]] = None,
        progress: bool = True,
        progress_backend: ProgressBackendName = "auto",
        log_progress_lines: bool = False,
        informative_priors: Optional[Dict[str, Dict[str, Any]]] = None,
        freeze_values: Optional[Dict[str, Dict[str, Any]]] = None,
        freeze_params: Tuple[str, ...] = (),
        w_prior: Optional[Dict[str, Any]] = None,
    ) -> LaplaceRunResult:
        """Run Laplace-mode training for any supported observation model.

        Parameters
        ----------
        model_config : ModelConfig
            Model configuration. The ``base_model`` field selects the
            observation-model adapter; the LNM family additionally
            consults ``d_mode`` and ``alr_reference_idx``.
        count_data : jnp.ndarray, shape (n_cells, n_genes)
            Observed UMI count matrix.
        n_cells, n_genes, latent_dim : int
            Dataset and model dimensions.
        laplace_config : LaplaceConfig
            Outer-loop and Newton-step hyperparameters.
        seed : int
            JAX PRNG seed.
        capture_anchor : Optional[Tuple[float, float]]
            ``(log_M_0, sigma_M)``. When ``None``, no capture anchor.
            When set, the joint Newton runs over ``(latent, eta_capture)``
            with the biology-informed TruncN prior on ``eta``.
        progress : bool, default=True
        progress_backend : str, default="auto"
        log_progress_lines : bool, default=False
        informative_priors : Optional[Dict]
            Empirical Gaussian priors derived from a previous SVI fit.
            Keys are a subset of ``{"r", "mu", "eta"}``.  Only consumed
            by the NBLN branch — other base models silently ignore.
            See :mod:`scribe.laplace.priors` for the adapter that builds
            these from a ``ScribeSVIResults`` object.

        Returns
        -------
        LaplaceRunResult

        Raises
        ------
        NotImplementedError
            If ``model_config.base_model`` is not one of the supported
            models (PLN, NBLN, LNM, LNMVCP).
        """
        # Top-level dispatch on the generative model. Every supported
        # base_model routes through the generic Laplace-EM driver
        # (``_em.run_laplace_em``) with a model-specific
        # :class:`LaplaceObservationModel` adapter. Adding a new model
        # is a matter of writing one such adapter and an ``elif``
        # here -- no engine-level scaffolding to duplicate.
        bm = getattr(model_config, "base_model", "pln")

        from ._em import run_laplace_em
        from ._w_priors import build_w_prior_strategy

        # Phase-3: build the W-prior strategy once from the user-facing
        # dict config.  ``None`` and ``{"type": "none"}`` both produce
        # NoneWPrior (no-op).  LNM-family raises NotImplementedError for
        # non-no-op configs since ALR-space W has different shrinkage
        # semantics that need a separate design pass.
        w_prior_strategy = build_w_prior_strategy(w_prior)

        if bm == "pln":
            from ._obs_pln import PLNObservationModel

            obs_model = PLNObservationModel(
                capture_anchor=capture_anchor,
                model_config=model_config,
                w_prior_strategy=w_prior_strategy,
            )
        elif bm == "nbln":
            from ._obs_nbln import NBLNObservationModel

            obs_model = NBLNObservationModel(
                capture_anchor=capture_anchor,
                model_config=model_config,
                informative_priors=informative_priors,
                freeze_values=freeze_values,
                freeze_params=freeze_params,
                w_prior_strategy=w_prior_strategy,
                max_step=float(
                    getattr(laplace_config, "newton_max_step", 5.0)
                ),
            )
        elif bm in ("lnm", "lnmvcp"):
            from ._obs_lnm import LNMObservationModel

            # v1: LNM-family doesn't yet support W-shrinkage priors.
            # Build_w_prior_strategy already normalized {"type": "none"}
            # → NoneWPrior, so any non-NoneWPrior strategy here means
            # the user requested an unsupported config.
            if w_prior_strategy.type_name != "none":
                raise NotImplementedError(
                    "w_prior is currently supported for PLN and NBLN "
                    f"only; got base_model={bm!r} with w_prior type "
                    f"{w_prior_strategy.type_name!r}.  LNM-family W "
                    "lives in ALR compositional coordinates and "
                    "needs a separate design pass for shrinkage."
                )
            d_mode = getattr(model_config, "d_mode", "learned") or "learned"
            alr_reference_idx = int(
                getattr(model_config, "alr_reference_idx", -1)
            )
            obs_model = LNMObservationModel(
                d_mode=d_mode,
                alr_reference_idx=alr_reference_idx,
                capture_anchor=capture_anchor,
                model_config=model_config,
            )
        else:
            raise NotImplementedError(
                f"Laplace inference is supported for PLN, NBLN, LNM, "
                f"and LNMVCP; got base_model={bm!r}."
            )

        return run_laplace_em(
            obs_model=obs_model,
            count_data=count_data,
            n_cells=n_cells,
            n_genes=n_genes,
            latent_dim=latent_dim,
            laplace_config=laplace_config,
            model_config=model_config,
            seed=seed,
            progress=progress,
            progress_backend=progress_backend,
            log_progress_lines=log_progress_lines,
        )


__all__ = ["LaplaceInferenceEngine", "LaplaceRunResult"]

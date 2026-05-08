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

from typing import Optional, Tuple

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

        if bm == "pln":
            from ._obs_pln import PLNObservationModel

            obs_model = PLNObservationModel(capture_anchor=capture_anchor)
        elif bm == "nbln":
            from ._obs_nbln import NBLNObservationModel

            obs_model = NBLNObservationModel(capture_anchor=capture_anchor)
        elif bm in ("lnm", "lnmvcp"):
            from ._obs_lnm import LNMObservationModel

            d_mode = getattr(model_config, "d_mode", "learned") or "learned"
            alr_reference_idx = int(
                getattr(model_config, "alr_reference_idx", -1)
            )
            obs_model = LNMObservationModel(
                d_mode=d_mode,
                alr_reference_idx=alr_reference_idx,
                capture_anchor=capture_anchor,
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

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
        filtered_gene_names: Optional[Any] = None,
        has_pooled_other: Optional[bool] = None,
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
            Keys depend on the target base model:
            - NBLN: ``{"r", "mu", "eta"}`` (from an NBVCP-SVI source).
            - twostate_ln_rate: ``{"mu", "burst_size", "k_off", "eta"}``
              (from a TwoState-SVI source).
            Other base models silently ignore the prior bundle.
            See :mod:`scribe.laplace.priors` for the adapters that build
            these from a ``ScribeSVIResults`` object
            (``priors_from_results`` and ``priors_from_twostate_results``).

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

        # Early-fail check for `correlate_other_column=False` on the
        # decoupled-math models that haven't yet been wired (PLN,
        # TSLN-Rate, TSLN-Logit).  The public API (``scribe.fit`` and
        # ``ModelConfig.correlate_other_column``) advertises decoupling
        # for all four PLN/NBLN/TSLN models, but Commit 2 of the
        # harmonic-hare plan only landed the NBLN scaffolding.  The
        # other three models' obs likelihoods, Newton steps, and
        # global-uncertainty paths land in subsequent commits (3 / 4 /
        # 5).  Until then, catch the inconsistency at the engine
        # **before** the obs model is constructed so the user gets a
        # clear, uniform error rather than a silent legacy-behaviour
        # fall-through that contradicts the API contract.
        #
        # Detection must match ``build_axis_layout``'s priority chain
        # (has_pooled_other > gene_names[-1] == "_other"; auditor
        # finding rev-5 #1) — using only ``has_pooled_other`` would
        # silently miss manually-pre-filtered AnnData whose tail is
        # literally ``"_other"`` (no ``gene_coverage`` stage ran).
        _ccc = bool(
            getattr(model_config, "correlate_other_column", True)
        )
        if (
            (not _ccc)
            and bm in ("pln", "twostate_ln_rate", "twostate_ln_logit")
        ):
            from ._axis_layout import build_axis_layout as _build_layout
            _probe_layout = _build_layout(
                n_genes=int(n_genes),
                correlate_other_column=False,
                gene_names=filtered_gene_names,
                has_pooled_other=has_pooled_other,
            )
            if _probe_layout.decoupled:
                raise NotImplementedError(
                    f"`{bm}` Laplace fit with "
                    "`correlate_other_column=False` and a pooled "
                    "'_other' column is not yet implemented — "
                    "Commit 2 of the harmonic-hare plan landed only "
                    "NBLN scaffolding; PLN / TSLN-Rate / TSLN-Logit "
                    "math lands in Commits 3 / 4 / 5.  Until then, "
                    "pass `correlate_other_column=True` (the current "
                    "default) to use the legacy path with `_other` "
                    "in Σ, or fit without `gene_coverage` filtering."
                )

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
                gene_names=filtered_gene_names,
                has_pooled_other=has_pooled_other,
            )
        elif bm == "twostate_ln_rate":
            from ._obs_twostate_ln_rate import TwoStateLNRateObservationModel

            obs_model = TwoStateLNRateObservationModel(
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
        elif bm == "twostate_ln_logit":
            from ._obs_twostate_ln_logit import (
                TwoStateLNLogitObservationModel,
            )

            # PR-2 capture restriction (Rev 4): the constructor
            # accepts only no-capture and frozen-offset capture.  If
            # the bridge built a non-None ``capture_anchor`` from a
            # biology-informed prior, the constructor will raise
            # NotImplementedError pointing the user at the cascade
            # frozen-eta path or at dropping the anchor.  Validation
            # is delegated to the constructor itself so the error
            # message comes from one place.
            obs_model = TwoStateLNLogitObservationModel(
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
                f"LNMVCP, twostate_ln_rate, and twostate_ln_logit; "
                f"got base_model={bm!r}."
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

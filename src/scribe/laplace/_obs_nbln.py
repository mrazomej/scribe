"""NB-LogNormal implementation of :class:`LaplaceObservationModel`.

Glues :mod:`scribe.laplace._newton_nbln` (per-cell Newton kernel) and
:class:`scribe.stats.distributions.LogMeanNegativeBinomial` (NB log-prob
in pure log-space) into the protocol expected by
:func:`scribe.laplace._em.run_laplace_em`. All shared scaffolding —
outer Adam, mini-batching, divergence detection, progress reporting,
final convergence check — lives in the driver; this module owns only
the model-specific pieces.

State init: empirical log-mean for ``μ``, PCA loadings for ``W``, a
small uniform unconstrained ``d_loc``, and the moment-method-of-moments
estimator for unconstrained ``r_loc``. Per-cell ``x`` warm-starts at
``log(u + 1)`` so the initial Newton iteration sees a near-MAP iterate.

Loss closure: Newton on ``(x[, η])`` with ``stop_gradient`` on the
iterates → ``log det(-H)`` at live globals → NB log-prob through
:class:`LogMeanNegativeBinomial` → MVN prior on ``x`` via the inner
Woodbury → optional truncated-normal prior on ``η``.

Final sweep: ``2 × n_newton`` iterations on the full cell population
to lock in per-cell convergence diagnostics.

Result packaging: ``x_loc = latent_loc`` (since NBLN's per-cell latent
is already the log-rate ``x``); ``globals`` exposes ``μ, W, d_loc,
r_loc``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

if TYPE_CHECKING:
    from ._w_priors import WPriorStrategy

from ..core.nbln_data_init import (
    empirical_dispersion_from_counts,
    empirical_log_mean_from_counts,
    pca_loadings_init,
)
from ..models.config import ModelConfig
from ..stats.distributions import LogMeanNegativeBinomial
from ._em import (
    FinalSweepResult,
    InitState,
    LaplaceObservationModel,
    LaplaceRunResult,
)
from ._global_uncertainty import (
    curvature_to_scale,
    resolve_positive_fns,
    woodbury_inv_diag,
)
from ._newton_nbln import (
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_decoupled,
    laplace_log_det_neg_H_batch_x_only,
    laplace_log_det_neg_H_batch_x_only_decoupled,
    laplace_log_det_neg_H_batch_x_only_offset,
    laplace_log_det_neg_H_batch_x_only_offset_decoupled,
    laplace_newton_batch,
    laplace_newton_batch_decoupled,
    laplace_newton_batch_x_only,
    laplace_newton_batch_x_only_decoupled,
    laplace_newton_batch_x_only_offset,
    laplace_newton_batch_x_only_offset_decoupled,
    nbln_grad_split_batch,
    nbln_grad_split_batch_decoupled,
    nbln_grad_x_only_norm_batch,
    nbln_grad_x_only_norm_batch_decoupled,
    nbln_grad_x_only_offset_norm_batch,
    nbln_grad_x_only_offset_norm_batch_decoupled,
    # Per-cell-W twins for the per-donor correlation hierarchy (Rung 1).
    laplace_log_det_neg_H_batch_percellW,
    laplace_log_det_neg_H_batch_x_only_percellW,
    laplace_log_det_neg_H_batch_x_only_offset_percellW,
    laplace_newton_batch_percellW,
    laplace_newton_batch_x_only_percellW,
    laplace_newton_batch_x_only_offset_percellW,
    nbln_grad_split_batch_percellW,
    nbln_grad_x_only_norm_batch_percellW,
    nbln_grad_x_only_offset_norm_batch_percellW,
    laplace_newton_batch_percellWmu,
    laplace_newton_batch_x_only_percellWmu,
    laplace_newton_batch_x_only_offset_percellWmu,
    nbln_grad_split_batch_percellWmu,
    nbln_grad_x_only_norm_batch_percellWmu,
    nbln_grad_x_only_offset_norm_batch_percellWmu,
    # decoupled per-cell-W (correlation hierarchy)
    laplace_newton_batch_decoupled_percellW,
    laplace_newton_batch_x_only_decoupled_percellW,
    laplace_newton_batch_x_only_offset_decoupled_percellW,
    laplace_log_det_neg_H_batch_decoupled_percellW,
    laplace_log_det_neg_H_batch_x_only_decoupled_percellW,
    laplace_log_det_neg_H_batch_x_only_offset_decoupled_percellW,
    nbln_grad_split_batch_decoupled_percellW,
    nbln_grad_x_only_norm_batch_decoupled_percellW,
    nbln_grad_x_only_offset_norm_batch_decoupled_percellW,
    # decoupled per-cell-W AND per-cell-mu (per-donor marginal cascade)
    laplace_newton_batch_decoupled_percellWmu,
    laplace_newton_batch_x_only_decoupled_percellWmu,
    laplace_newton_batch_x_only_offset_decoupled_percellWmu,
    laplace_log_det_neg_H_batch_decoupled_percellWmu,
    laplace_log_det_neg_H_batch_x_only_decoupled_percellWmu,
    laplace_log_det_neg_H_batch_x_only_offset_decoupled_percellWmu,
    nbln_grad_split_batch_decoupled_percellWmu,
    nbln_grad_x_only_norm_batch_decoupled_percellWmu,
    nbln_grad_x_only_offset_norm_batch_decoupled_percellWmu,
)
from ..models.components.program_scales import (
    program_scales_from_raw,
    _DEFAULT_TAU_LOC,
    _DEFAULT_TAU_SCALE,
)


# =====================================================================
# Woodbury helpers for the MVN prior on x
# =====================================================================


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, diff: jnp.ndarray
) -> jnp.ndarray:
    """Compute ``(diff)ᵀ Σ⁻¹ (diff)`` for ``Σ = W Wᵀ + diag(d)``, batched."""
    inv_d = 1.0 / d
    K_dim = W.shape[1]
    K = jnp.eye(K_dim) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    diff_inv_d = inv_d[None, :] * diff
    rhs = diff_inv_d @ W
    z = jax.scipy.linalg.cho_solve((L_K, True), rhs.T).T
    correction = (diff_inv_d * (z @ W.T)).sum(axis=-1)
    direct = (diff * diff_inv_d).sum(axis=-1)
    return direct - correction


def _woodbury_logdet_sigma(W: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """``log det Σ`` via the matrix-determinant lemma."""
    inv_d = 1.0 / d
    K_dim = W.shape[1]
    K = jnp.eye(K_dim) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
    log_det_d = jnp.sum(jnp.log(d))
    return log_det_d + log_det_K


# Per-cell-W Woodbury twins for the per-donor correlation hierarchy (Rung 1).
# Under the hierarchy each cell's prior covariance is
# ``Σ_{σ(c)} = W_eff,{σ(c)} W_eff,{σ(c)}ᵀ + diag(d)`` with per-cell effective
# loadings, so the quadratic form and log-det of Σ become per-cell. Both are
# ``jax.vmap`` over the cell axis of ``W`` (shape ``(N, G, k)``); ``d`` stays
# shared. The per-cell math is the shared helper above, unchanged.
_woodbury_quadform_percellW = jax.vmap(
    # Each call gets a single cell's W ``(G, k)`` and diff ``(G,)``; reshape
    # diff to ``(1, G)`` so the batched helper returns a length-1 vector, then
    # take the scalar.
    lambda Wc, d_, diffc: _woodbury_quadform(Wc, d_, diffc[None, :])[0],
    in_axes=(0, None, 0),
)
_woodbury_logdet_sigma_percellW = jax.vmap(
    _woodbury_logdet_sigma, in_axes=(0, None)
)


# =====================================================================
# Observation-model class
# =====================================================================


class NBLNObservationModel(LaplaceObservationModel):
    """NB-LogNormal observation channel for the generic Laplace-EM driver.

    Parameters
    ----------
    capture_anchor : Optional[Tuple[float, float]]
        ``(log_M_0, σ_M)`` for the biology-informed truncated-normal
        prior on ``η_capture``. ``None`` disables capture (Newton runs
        on ``x`` only).
    model_config : ModelConfig, optional
        Model configuration.  Used to resolve
        ``positive_transform`` for the dispersion parameter ``r``.
        Falls back to ``softplus`` when ``None``.
    informative_priors : Optional[Dict[str, Dict[str, jnp.ndarray]]]
        Empirical Gaussian priors derived from a previously-fit SVI
        results object.  Keys are a subset of ``{"r", "mu", "eta"}``,
        each mapping to a ``{"loc": ..., "scale": ...}`` dict in the
        target Laplace coordinate space.  See
        :mod:`scribe.laplace.priors` for the adapter that builds these.

        When ``"eta"`` is supplied, capture activates on the target
        even if ``capture_anchor`` is ``None`` — the SVI per-cell
        prior supersedes the scalar anchor.  When ``"r"`` or ``"mu"``
        is supplied, the loss gains a ``Normal(loc, scale).log_prob(...)``
        term that adds prior curvature to the global Hessian during
        both training and post-fit uncertainty extraction.
    freeze_values : Optional[Dict[str, Dict[str, jnp.ndarray]]]
        Point-estimate values for parameters listed in ``freeze_params``.
        Built by :func:`scribe.laplace.priors.freeze_values_from_results`
        from an SVI source's ``get_map()``.  Each value is a dict with a
        single ``"loc"`` key (no scale — frozen parameters are points).
        Coordinates are already in NBLN target space (``log`` for ``mu``,
        ``pos_inverse`` for ``r``, identity for ``eta``).

        Separate from ``informative_priors``: the two dicts can coexist
        (e.g. freeze ``r`` and ``eta``, soft-cascade ``mu``).  A frozen
        key MUST also appear in ``freeze_params``.
    freeze_params : Tuple[str, ...]
        Names of parameters to freeze during the M-step.  Subset of
        ``{"r", "mu", "eta"}``.  Frozen parameters are excluded from the
        optax optimizer's params dict — the optimizer literally cannot
        move what isn't in its state.  Frozen values are spliced back
        into a working ``params_full`` dict inside every method that
        needs them (loss_fn, final_sweep, compute_global_uncertainty,
        pack_result).
    """

    def __init__(
        self,
        capture_anchor: Optional[Tuple[float, float]] = None,
        model_config: Optional[ModelConfig] = None,
        informative_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        freeze_values: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        freeze_params: Tuple[str, ...] = (),
        w_prior_strategy: Optional["WPriorStrategy"] = None,
        max_step: float = 5.0,
        gene_names: Optional[Any] = None,
        has_pooled_other: Optional[bool] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
    ):
        self._max_step = float(max_step)

        # Stash gene_names / has_pooled_other for AxisLayout construction
        # in `init_state` (where we also have `n_genes`).  The layout
        # captures the G_obs vs G_kept split when `correlate_other_column=False`
        # and the data has a trailing `_other` pooled column.  See
        # ``scribe.laplace._axis_layout.build_axis_layout`` for the
        # detection-priority contract and the contradictory-signal raise.
        self._gene_names = gene_names
        self._has_pooled_other = has_pooled_other
        # Fallback default matches the user-facing default flipped in
        # Commit 5b: when ``model_config`` is missing the attribute
        # entirely (mocks, partial configs), behave as if the user
        # accepted the new ``False`` default — decoupled when an
        # ``_other`` column is present, trivial layout otherwise.
        self._correlate_other_column = bool(
            getattr(model_config, "correlate_other_column", False)
        )
        # Built lazily in init_state once `n_genes` is available.
        self._axis_layout = None

        # --- Per-donor correlation hierarchy (NB-LogNormal Rung 1) -----------
        # When active, each cell's latent prior covariance is donor-specific:
        #   Σ_{σ(c)} = W diag(s_{σ(c)}^2) Wᵀ + diag(d).
        # The shared ``program_scales_active`` predicate (used by the SVI path
        # too) gates this on ``correlation_hierarchy == "program_scales"``,
        # ``n_datasets >= 2``, and per-cell ``dataset_indices`` being present.
        # ``s`` (the relative per-donor program activity) is learned as two
        # extra optimizer globals (``program_scale_raw`` and
        # ``program_scale_tau_raw``) added in ``init_state``; ``loss_fn``
        # rebuilds ``s`` via the shared ``program_scales_from_raw`` transform,
        # gathers per-cell effective loadings ``W_eff = W·diag(s_{σ(c)})``, and
        # dispatches to the ``_percellW`` Newton kernels.
        from ..models.components.program_scales import program_scales_active

        self._dataset_indices = (
            None
            if dataset_indices is None
            else jnp.asarray(dataset_indices, dtype=jnp.int32)
        )
        self._n_datasets = int(getattr(model_config, "n_datasets", 0) or 0)
        self._hier_active = program_scales_active(
            model_config, self._dataset_indices
        )

        # Phase-3: pluggable shrinkage prior on W.  Default no-op so
        # existing callers are unaffected.
        from ._w_priors import NoneWPrior

        self._w_prior = (
            w_prior_strategy if w_prior_strategy is not None else NoneWPrior()
        )
        if capture_anchor is None:
            self._capture_anchor = None
            self._sigma_M = 1.0  # placeholder; never read when capture is off
        else:
            log_M0, sigma_M = capture_anchor
            self._capture_anchor = (float(log_M0), float(sigma_M))
            self._sigma_M = float(sigma_M)

        # Resolve the configured positivity map and its inverse once,
        # so loss_fn, final_sweep, and compute_global_uncertainty all
        # share the same coordinate system.
        self._pos_forward, self._pos_inverse = resolve_positive_fns(
            model_config
        )

        # Informative priors — per-parameter storage with key validation.
        # Each key (r/mu/eta) is stored separately so the loss can consult
        # them independently without ambiguity.
        if informative_priors is not None:
            valid = {"r", "mu", "eta"}
            invalid = set(informative_priors) - valid
            if invalid:
                raise ValueError(
                    f"informative_priors has unrecognized keys {invalid}; "
                    f"valid keys are {valid}."
                )
        self._prior_r = (
            informative_priors.get("r")
            if informative_priors is not None
            else None
        )
        self._prior_mu = (
            informative_priors.get("mu")
            if informative_priors is not None
            else None
        )
        self._prior_eta = (
            informative_priors.get("eta")
            if informative_priors is not None
            else None
        )

        # Freeze values — stored SEPARATE from informative_priors so the
        # soft-cascade (Gaussian prior log-density in the loss) and the
        # hard-freeze (exclusion from the optimizer dict) can coexist on
        # different parameters within the same fit (e.g. freeze r+eta,
        # soft-cascade mu).
        valid_frozen = {"r", "mu", "eta"}
        bad_frozen = set(freeze_params) - valid_frozen
        if bad_frozen:
            raise ValueError(
                f"freeze_params has invalid keys {bad_frozen}; "
                f"valid keys are {valid_frozen}."
            )
        # Each frozen key MUST have a corresponding freeze_values entry.
        if freeze_params:
            if freeze_values is None:
                raise ValueError(
                    f"freeze_params={list(freeze_params)} non-empty but "
                    "freeze_values is None.  Each frozen key needs a "
                    "freeze_values[key] entry with a 'loc' field."
                )
            missing = set(freeze_params) - set(freeze_values.keys())
            if missing:
                raise ValueError(
                    f"freeze_params requests {sorted(missing)} but "
                    "freeze_values does not provide those keys.  "
                    "Available freeze_values keys: "
                    f"{sorted(freeze_values.keys())}."
                )
            for k in freeze_params:
                if "loc" not in freeze_values[k]:
                    raise ValueError(
                        f"freeze_values[{k!r}] missing 'loc' field."
                    )
        self._frozen_params = frozenset(freeze_params)
        self._freeze_values = freeze_values if freeze_values is not None else {}

        # --- Per-donor frozen mu (step 4b) -------------------------------
        # The hierarchical-marginal cascade freezes a *leaf-level* mean
        # matrix ``mu`` of shape ``(D, G)`` (one row per dataset/donor)
        # rather than a single pooled ``(G,)`` vector.  Each cell ``c``
        # then uses its donor's prior mean ``mu^{(σ(c))}`` via a per-cell
        # gather on ``self._dataset_indices`` -- the exact mechanism the
        # correlation hierarchy already uses for the program scales.  We
        # detect the per-donor layout from the frozen value's rank.
        self._mu_per_donor = False
        if "mu" in self._frozen_params:
            _mu_loc = jnp.asarray(self._freeze_values["mu"]["loc"])
            if _mu_loc.ndim == 2:
                # Per-donor mu requires the per-cell donor index AND, in
                # v1, the correlation hierarchy (so the per-cell-W Newton
                # path is active and the kernels already vmap over cells).
                if self._dataset_indices is None:
                    raise ValueError(
                        "Per-donor frozen mu has shape "
                        f"{tuple(_mu_loc.shape)} (D, G) but no "
                        "dataset_indices were provided; the per-cell mu "
                        "gather needs the donor assignment of each cell."
                    )
                if not self._hier_active:
                    raise NotImplementedError(
                        "Per-donor frozen mu (shape (D, G)) is supported "
                        "in v1 only together with "
                        'correlation_hierarchy="program_scales" (the '
                        "per-cell-W Newton path).  Pool the cascade mean "
                        "to (G,), or enable the correlation hierarchy."
                    )
                if int(_mu_loc.shape[0]) != self._n_datasets:
                    raise ValueError(
                        "Per-donor frozen mu has "
                        f"{int(_mu_loc.shape[0])} rows but the fit has "
                        f"n_datasets={self._n_datasets}; the row order "
                        "must match the target leaf indexing."
                    )
                self._mu_per_donor = True

    # --- Identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "nbln"

    @property
    def uses_capture(self) -> bool:
        # Capture activates when ANY of:
        # - The user passed a scalar capture_anchor (biology-anchored).
        # - The user passed a soft-cascade prior on eta (SVI-derived).
        # - The user froze eta at SVI's MAP (hard-cascade).
        # The third clause is essential: without it, frozen-eta fits
        # silently degrade to the x-only-no-capture branch in loss_fn /
        # final_sweep.
        return (
            self._capture_anchor is not None
            or self._prior_eta is not None
            or "eta" in self._frozen_params
        )

    @property
    def freezes_eta(self) -> bool:
        """True iff eta is frozen at the SVI MAP (x-only-with-offset path)."""
        return "eta" in self._frozen_params

    # --- Initial state ---------------------------------------------------

    def init_state(
        self,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        latent_dim: int,
        seed: int,
    ) -> InitState:
        counts_np = np.asarray(count_data)

        # Build the AxisLayout now that we know n_genes.  The layout
        # is the single source of truth for whether `_other` is in the
        # latent covariance (legacy: yes; decoupled: no).  All
        # downstream code (loss_fn, Newton, compute_global_uncertainty,
        # pack_result, compositional sampler) branches on
        # `self._axis_layout.decoupled` — when False, every existing
        # code path runs unchanged (bit-equal to legacy).
        from ._axis_layout import build_axis_layout
        self._axis_layout = build_axis_layout(
            n_genes=int(n_genes),
            correlate_other_column=self._correlate_other_column,
            gene_names=self._gene_names,
            has_pooled_other=self._has_pooled_other,
        )
        _layout = self._axis_layout

        # Per-donor correlation hierarchy: supported on BOTH layouts.  The
        # decoupled (correlate_other_column=False) deviation-form kernels have
        # their own per-cell-W / per-cell-mu twins (``*_decoupled_percellW`` /
        # ``*_decoupled_percellWmu`` in ``_newton_nbln``), selected in
        # ``loss_fn`` / ``final_sweep`` exactly as on the legacy layout.

        # Commit 2b: the decoupled path is now fully implemented.  No
        # early guard here — init proceeds for both layouts.  The
        # decoupled branches in ``loss_fn`` / ``final_sweep`` /
        # ``compute_global_uncertainty`` consume ``self._axis_layout``
        # directly and drive the deviation-form Newton kernels in
        # ``_newton_nbln`` (the ``*_decoupled`` family).

        # --- mu init: frozen overrides prior overrides data-driven ---
        # `mu` lives in the OBSERVATION-layer axis (G_obs,) under BOTH
        # layouts — it is the prior centre / baseline per-gene log-mean
        # that the NB likelihood consumes for every gene including
        # `_other`.  Shape is (G_obs,) in the usual case; when the
        # hierarchical-marginal cascade freezes a per-donor mean it is
        # (D, G_obs) (one row per dataset), gathered per cell in loss_fn /
        # final_sweep via ``self._dataset_indices`` (see ``_mu_per_donor``).
        if "mu" in self._frozen_params:
            mu_init = jnp.asarray(
                self._freeze_values["mu"]["loc"], dtype=jnp.float32
            )
        elif self._prior_mu is not None:
            mu_init = jnp.asarray(self._prior_mu["loc"], dtype=jnp.float32)
        else:
            mu_init = jnp.asarray(
                empirical_log_mean_from_counts(counts_np), dtype=jnp.float32
            )

        # W and d live in the LATENT-COVARIANCE axis (G_kept,).  Under
        # the decoupled layout we slice the count matrix to kept genes
        # before PCA so the resulting loadings are (G_kept, K) directly
        # — slicing post-PCA would let `_other` variance leak into
        # kept-gene loadings via the SVD coupling.
        if _layout.decoupled:
            counts_for_pca = counts_np[:, _layout.kept_idx]
        else:
            counts_for_pca = counts_np
        W_init = jnp.asarray(
            pca_loadings_init(counts_for_pca, latent_dim=latent_dim),
            dtype=jnp.float32,
        )
        # d_loc shape matches the latent-covariance axis.
        d_loc_init = self._pos_inverse(
            jnp.full((int(_layout.G_kept),), 0.1, dtype=jnp.float32)
        )

        # --- r init: frozen overrides prior overrides data-driven ---
        if "r" in self._frozen_params:
            r_loc_init = jnp.asarray(
                self._freeze_values["r"]["loc"], dtype=jnp.float32
            )
        elif self._prior_r is not None:
            r_loc_init = jnp.asarray(self._prior_r["loc"], dtype=jnp.float32)
        else:
            r_init = jnp.asarray(
                empirical_dispersion_from_counts(counts_np), dtype=jnp.float32
            )
            r_loc_init = self._pos_inverse(jnp.maximum(r_init, 1e-3)).astype(
                jnp.float32
            )

        # Build the FULL params dict first (covers all keys).
        full_init = {
            "mu": mu_init,
            "W": W_init,
            "d_loc": d_loc_init,
            "r_loc": r_loc_init,
        }

        # --- Per-donor correlation hierarchy: program-scale globals ----------
        # When active, learn the relative per-donor program-activity ``s_d``
        # through two extra optimizer globals (non-centered + sum-to-zero
        # gauge, see ``program_scales_from_raw``):
        #   • ``program_scale_raw``     : (n_datasets, K) NCP raw effects.
        #   • ``program_scale_tau_raw`` : scalar unconstrained between-donor
        #     scale (softplus -> τ_s).
        # Initialized at the prior centre (raw = 0 -> s = 1 for every donor;
        # tau_raw = τ_loc) so the fit starts at "no between-donor difference"
        # and grows ``s_d`` only as the data support it.
        if self._hier_active:
            full_init["program_scale_raw"] = jnp.zeros(
                (self._n_datasets, latent_dim), dtype=jnp.float32
            )
            full_init["program_scale_tau_raw"] = jnp.asarray(
                _DEFAULT_TAU_LOC, dtype=jnp.float32
            )

        # --- Phase-3: W-prior aux params ---
        # The strategy decides which aux params (if any) to register.
        # NoneWPrior returns {} — equivalent to no shrinkage.  Other
        # strategies add per-factor scales (and a global scale) in
        # unconstrained (``raw``) space; they ride through optax's
        # M-step automatically since they're in the params dict.
        # `G` here is the LATENT-COVARIANCE axis (G_kept) — the W prior
        # shrinks `W` rows, and W has shape (G_kept, K).  Under the
        # legacy layout G_kept == G_obs so this is unchanged from today.
        _w_aux_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
        w_aux_init = self._w_prior.init_aux_params(
            G=int(_layout.G_kept),
            k_latent=latent_dim,
            rng_key=_w_aux_key,
        )
        full_init.update(w_aux_init)

        # --- Exclude frozen keys from the optimizer dict (Round-4 R4) ---
        # Stash frozen values on `self._frozen_runtime` for splicing into
        # params_full inside loss_fn / final_sweep / compute_global_uncertainty
        # / pack_result.  The optax optimizer never sees these keys, so
        # they cannot drift regardless of optimizer internals.
        self._frozen_runtime: Dict[str, jnp.ndarray] = {}
        if "r" in self._frozen_params:
            self._frozen_runtime["r_loc"] = full_init.pop("r_loc")
        if "mu" in self._frozen_params:
            self._frozen_runtime["mu"] = full_init.pop("mu")
        # Note: eta is per-cell and never lived in `params` dict — it's
        # in `latent_loc` / `eta_loc` / `aux_data`.  We handle frozen
        # eta below via the offset path.

        params = full_init  # reduced dict for optax

        # --- Four-way capture branch (extends the Round-3 three-way) ---
        # Frozen eta gets its own branch: x-only Newton with fixed
        # eta_offset.  No prior_eta or capture_anchor consulted in this
        # case — the freeze value comes from self._freeze_values["eta"].
        #
        # ``latent_loc`` initial shape (per cell) depends on the layout:
        #   • Legacy (G_kept == G_obs): absolute log-rate ``x``, shape
        #     ``(N, G_obs)`` — matches today's behaviour exactly.
        #   • Decoupled (G_kept < G_obs): deviation ``x_dev``, shape
        #     ``(N, G_kept)`` — initialised at log(u_kept + 1) − μ_kept
        #     so the deviation starts near zero (matches the prior
        #     N(0, Σ_kept) centre).
        def _initial_latent(_with_eta_offset):
            """Build per-cell initial latent in the correct axis."""
            if _layout.decoupled:
                kept_idx_jnp = jnp.asarray(_layout.kept_idx)
                _log_u_kept = jnp.log(count_data[:, kept_idx_jnp] + 1.0)
                _mu_kept = mu_init[kept_idx_jnp]
                _base = _log_u_kept - _mu_kept[None, :]
                if _with_eta_offset is not None:
                    _base = _base + _with_eta_offset[:, None]
                return _base
            _base = jnp.log(count_data + 1.0)
            if _with_eta_offset is not None:
                _base = _base + _with_eta_offset[:, None]
            return _base

        if "eta" in self._frozen_params:
            eta_offset = jnp.asarray(
                self._freeze_values["eta"]["loc"], dtype=jnp.float32
            )
            # eta_anchor / eta_scale_per_cell are unused in this branch
            # but populated for downstream result-storage symmetry.
            eta_anchor = eta_offset
            eta_scale_per_cell = None  # no scale — eta is a fixed point
            eta_loc = eta_offset  # carried on the result so PPC sees it
            latent_loc = _initial_latent(eta_loc)
        elif self._prior_eta is not None:
            # SVI-derived soft-cascade per-cell capture (mode "eta").
            eta_anchor = jnp.asarray(self._prior_eta["loc"], dtype=jnp.float32)
            eta_scale_per_cell = jnp.asarray(
                self._prior_eta["scale"], dtype=jnp.float32
            )
            eta_loc = eta_anchor
            latent_loc = _initial_latent(eta_loc)
        elif self._capture_anchor is not None:
            # Scalar biology-informed capture anchor (legacy path).
            log_M0, _sigma_M = self._capture_anchor
            log_lib = jnp.log(jnp.maximum(jnp.sum(count_data, axis=-1), 1.0))
            eta_anchor = log_M0 - log_lib
            eta_scale_per_cell = jnp.full(
                (n_cells,), self._sigma_M, dtype=jnp.float32
            )
            eta_loc = eta_anchor
            latent_loc = _initial_latent(eta_loc)
        else:
            # No capture at all (x-only Newton path).
            eta_anchor = None
            eta_scale_per_cell = None
            eta_loc = None
            latent_loc = _initial_latent(None)

        # aux_data: per-cell scale for soft-cascade eta path; per-cell
        # offset for frozen-eta path.  Both ride through aux_batch_slice
        # so minibatch indexing is automatic.
        aux_data: Dict[str, jnp.ndarray] = {}
        if eta_scale_per_cell is not None:
            aux_data["eta_scale"] = eta_scale_per_cell
        if "eta" in self._frozen_params:
            aux_data["eta_frozen"] = jnp.asarray(
                self._freeze_values["eta"]["loc"], dtype=jnp.float32
            )
        # Per-cell donor/leaf index for the correlation hierarchy. Riding it
        # on aux_data means ``aux_batch_slice`` gathers it per mini-batch
        # automatically (same as eta_scale/eta_frozen), so ``loss_fn`` can map
        # each cell to its donor's program scales.
        if self._hier_active:
            aux_data["dataset_indices"] = self._dataset_indices

        return InitState(
            params=params,
            latent_loc=latent_loc,
            eta_loc=eta_loc,
            eta_anchor=eta_anchor,
            aux_data=aux_data,
        )

    # --- Splice helper (used by loss_fn, final_sweep, compute_global_uncertainty,
    #     pack_result to reconstruct the full params dict for likelihood / Hessian
    #     computation when frozen keys have been excluded from the optimizer).

    def _splice_frozen(
        self, params: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Return ``{**params, **frozen}`` with stop_gradient on frozen entries.

        The reduced ``params`` dict that the optimizer holds excludes
        any frozen keys; this helper splices their runtime values back
        in for likelihood / Hessian computation.  ``stop_gradient`` on
        frozen entries is belt-and-suspenders on top of the primary
        exclusion-from-optimizer mechanism — autodiff sees them as
        constants either way, but the explicit stop_gradient documents
        intent and protects against any incidental gradient flow.
        """
        if not self._frozen_runtime:
            return params
        out = dict(params)
        for k, v in self._frozen_runtime.items():
            out[k] = jax.lax.stop_gradient(v)
        return out

    # --- Loss closure ----------------------------------------------------

    def loss_fn(
        self,
        params: Dict[str, jnp.ndarray],
        latent_init: jnp.ndarray,
        eta_init: jnp.ndarray,
        counts_batch: jnp.ndarray,
        eta_anchor_batch: jnp.ndarray,
        aux_batch: Dict[str, jnp.ndarray],
        data_scale: float,
        n_newton: int,
        damping: float,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        # Commit 2b: under the decoupled layout
        # (``self._axis_layout.decoupled == True``) the loss runs the
        # deviation-parameterised Newton kernels from ``_newton_nbln``.
        # The legacy (``decoupled is False``) path is untouched and
        # bit-equal to today.  The branch is taken below at the Newton
        # dispatch; here we just resolve the kept_idx into a JAX array
        # once so it can be threaded into the vmapped Newton kernels.
        _layout = self._axis_layout
        _is_decoupled = _layout is not None and _layout.decoupled
        if _is_decoupled:
            _kept_idx_j = jnp.asarray(_layout.kept_idx, dtype=jnp.int32)
        else:
            _kept_idx_j = None  # unused on the legacy path

        # Splice frozen values into a working params_full dict for
        # likelihood computation (Round-4 R4).  The optimizer holds only
        # the reduced params dict (frozen keys excluded).
        params_full = self._splice_frozen(params)

        mu = params_full["mu"]
        W = params_full["W"]
        d = self._pos_forward(params_full["d_loc"])
        # Map unconstrained r_loc to positive dispersion via the
        # configured positive_transform (softplus by default).
        r = self._pos_forward(params_full["r_loc"])

        # Per-cell eta prior scale (Round-1 Finding 4 + Round-2 Finding B).
        # When eta is frozen, we use the x-only-with-offset Newton path
        # which doesn't consume sigma_eta_batch.  When capture is soft
        # (prior_eta or capture_anchor), aux_batch["eta_scale"] provides
        # the per-cell scale.  When no capture, we synthesise a ones
        # array for the vmap signature.
        if self.uses_capture and not self.freezes_eta:
            sigma_eta_batch = aux_batch["eta_scale"]
        else:
            sigma_eta_batch = jnp.ones((counts_batch.shape[0],))

        # Stop-gradient on Newton inputs (variational-EM convention).
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        r_sg = jax.lax.stop_gradient(r)

        # Per-cell mean for the Newton/grad-norm kernels and the MVN-prior
        # diff.  Defaults to the shared ``(G,)`` mu; overridden to a
        # per-cell ``(B, G)`` gather in the per-donor-frozen-mu branch
        # below.  ``mu_cell_live`` is None unless per-donor (signals the
        # broadcast path for the MVN diff).
        mu_newton = mu_sg
        mu_cell_live = None

        # --- Per-donor correlation hierarchy: per-cell effective loadings ----
        # When the hierarchy is active, each cell's prior covariance uses its
        # donor's effective loadings ``W_eff,{σ(c)} = W·diag(s_{σ(c)})``. We
        # build two per-cell ``(B, G, k)`` tensors and select the per-cell-W
        # Newton/log-det/grad-norm twins:
        #   • ``W_eff_sg``   — stop-grad (Newton + grad-norm inputs).
        #   • ``W_eff_live`` — gradient-bearing (log det(-H) and the MVN prior),
        #     the two terms through which ``s_d`` receives its gradient.
        # When inactive, everything aliases to the shared-W path (bit-equal to
        # today). ``d`` stays shared across donors in both cases.
        if self._hier_active:
            s_live = program_scales_from_raw(
                params_full["program_scale_raw"],
                params_full["program_scale_tau_raw"],
            )  # (n_datasets, K), relative per-donor program activity
            ds_batch = aux_batch["dataset_indices"]  # (B,) int
            s_cell = s_live[ds_batch]  # (B, K)
            W_eff_live = W[None, :, :] * s_cell[:, None, :]  # (B, G, K)
            W_eff_sg = W_sg[None, :, :] * jax.lax.stop_gradient(s_cell)[
                :, None, :
            ]
            W_newton, W_logdet = W_eff_sg, W_eff_live
            _woodbury_quad = _woodbury_quadform_percellW
            _woodbury_logdet = _woodbury_logdet_sigma_percellW
            _nt_x_only = laplace_newton_batch_x_only_percellW
            _nt_x_only_offset = laplace_newton_batch_x_only_offset_percellW
            _nt_joint = laplace_newton_batch_percellW
            _ld_x_only = laplace_log_det_neg_H_batch_x_only_percellW
            _ld_x_only_offset = laplace_log_det_neg_H_batch_x_only_offset_percellW
            _ld_joint = laplace_log_det_neg_H_batch_percellW
            _gn_x_only = nbln_grad_x_only_norm_batch_percellW
            _gn_x_only_offset = nbln_grad_x_only_offset_norm_batch_percellW
            _gn_joint = nbln_grad_split_batch_percellW
            # Decoupled-layout per-cell-W twins (step 6).
            _dec_nt_x_only = laplace_newton_batch_x_only_decoupled_percellW
            _dec_nt_x_only_offset = (
                laplace_newton_batch_x_only_offset_decoupled_percellW
            )
            _dec_nt_joint = laplace_newton_batch_decoupled_percellW
            _dec_ld_x_only = (
                laplace_log_det_neg_H_batch_x_only_decoupled_percellW
            )
            _dec_ld_x_only_offset = (
                laplace_log_det_neg_H_batch_x_only_offset_decoupled_percellW
            )
            _dec_ld_joint = laplace_log_det_neg_H_batch_decoupled_percellW
            _dec_gn_x_only = nbln_grad_x_only_norm_batch_decoupled_percellW
            _dec_gn_x_only_offset = (
                nbln_grad_x_only_offset_norm_batch_decoupled_percellW
            )
            _dec_gn_joint = nbln_grad_split_batch_decoupled_percellW
            # Per-donor frozen mu (step 4b): each cell's latent prior mean
            # is mu^{(σ(c))}.  Gather to a per-cell ``(B, G)`` tensor and
            # switch the Newton + grad-norm kernels to their per-cell-mu
            # twins.  On the legacy layout the log-det is mu-free (reused
            # as-is); on the decoupled layout the log-det/grad DO consume mu
            # (log_rate = μ + x_dev), so they switch to per-cell-mu twins too.
            # When mu is pooled, ``mu_newton`` stays the shared ``(G,)``
            # vector and ``mu_cell_live`` is None (broadcast path below).
            if self._mu_per_donor:
                mu_cell_live = mu[ds_batch]          # (B, G), gradient-bearing
                mu_newton = mu_sg[ds_batch]          # (B, G), stop-grad
                _nt_x_only = laplace_newton_batch_x_only_percellWmu
                _nt_x_only_offset = (
                    laplace_newton_batch_x_only_offset_percellWmu
                )
                _nt_joint = laplace_newton_batch_percellWmu
                _gn_x_only = nbln_grad_x_only_norm_batch_percellWmu
                _gn_x_only_offset = (
                    nbln_grad_x_only_offset_norm_batch_percellWmu
                )
                _gn_joint = nbln_grad_split_batch_percellWmu
                # Decoupled per-cell-W AND per-cell-mu twins.
                _dec_nt_x_only = (
                    laplace_newton_batch_x_only_decoupled_percellWmu
                )
                _dec_nt_x_only_offset = (
                    laplace_newton_batch_x_only_offset_decoupled_percellWmu
                )
                _dec_nt_joint = laplace_newton_batch_decoupled_percellWmu
                _dec_ld_x_only = (
                    laplace_log_det_neg_H_batch_x_only_decoupled_percellWmu
                )
                _dec_ld_x_only_offset = (
                    laplace_log_det_neg_H_batch_x_only_offset_decoupled_percellWmu
                )
                _dec_ld_joint = laplace_log_det_neg_H_batch_decoupled_percellWmu
                _dec_gn_x_only = (
                    nbln_grad_x_only_norm_batch_decoupled_percellWmu
                )
                _dec_gn_x_only_offset = (
                    nbln_grad_x_only_offset_norm_batch_decoupled_percellWmu
                )
                _dec_gn_joint = nbln_grad_split_batch_decoupled_percellWmu
        else:
            W_newton, W_logdet = W_sg, W
            _woodbury_quad = _woodbury_quadform
            _woodbury_logdet = _woodbury_logdet_sigma
            _nt_x_only = laplace_newton_batch_x_only
            _nt_x_only_offset = laplace_newton_batch_x_only_offset
            _nt_joint = laplace_newton_batch
            _ld_x_only = laplace_log_det_neg_H_batch_x_only
            _ld_x_only_offset = laplace_log_det_neg_H_batch_x_only_offset
            _ld_joint = laplace_log_det_neg_H_batch
            _gn_x_only = nbln_grad_x_only_norm_batch
            _gn_x_only_offset = nbln_grad_x_only_offset_norm_batch
            _gn_joint = nbln_grad_split_batch
            # Shared-W decoupled kernels (current behaviour; bit-equal).
            _dec_nt_x_only = laplace_newton_batch_x_only_decoupled
            _dec_nt_x_only_offset = laplace_newton_batch_x_only_offset_decoupled
            _dec_nt_joint = laplace_newton_batch_decoupled
            _dec_ld_x_only = laplace_log_det_neg_H_batch_x_only_decoupled
            _dec_ld_x_only_offset = (
                laplace_log_det_neg_H_batch_x_only_offset_decoupled
            )
            _dec_ld_joint = laplace_log_det_neg_H_batch_decoupled
            _dec_gn_x_only = nbln_grad_x_only_norm_batch_decoupled
            _dec_gn_x_only_offset = nbln_grad_x_only_offset_norm_batch_decoupled
            _dec_gn_joint = nbln_grad_split_batch_decoupled

        # Per-cell mean fed to the decoupled log-det / grad kernels (which
        # consume μ via ``log_rate = μ + x_dev``): the per-cell ``(B, G)``
        # gather when mu is frozen per donor, else the shared live ``(G,)``.
        mu_logdet_dec = mu_cell_live if mu_cell_live is not None else mu

        # --- Four-way Newton dispatch (Round-4 R5-2) ---
        # 1. Frozen eta: x-only Newton with offset = freeze value.
        # 2. Soft eta (prior or anchor): joint (x, η) Newton.
        # 3. No capture: plain x-only Newton.
        #
        # Commit 2b: each branch has a decoupled twin selected by
        # ``_is_decoupled``.  The decoupled kernels live on the kept-
        # gene axis: ``latent_init`` (carried through ``x_new``) has
        # shape ``(N, G_kept)`` and represents ``x_dev`` (deviation
        # from μ).  Downstream code builds the full ``(N, G_obs)``
        # ``log_mean`` for the NB log-prob by scattering ``x_dev`` at
        # kept positions and reading ``μ`` directly at ``other_idx``.
        if self.freezes_eta:
            eta_offset_batch = aux_batch["eta_frozen"]
            eta_offset_sg = jax.lax.stop_gradient(eta_offset_batch)
            if _is_decoupled:
                # ``_dec_*`` + ``W_newton``/``W_logdet``/``mu_*`` are the
                # shared-W decoupled kernels when the hierarchy is off and
                # the per-cell-W (+ per-cell-mu) decoupled twins when it is on.
                x_new, _gn, _ = _dec_nt_x_only_offset(
                    latent_init_sg,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    eta_offset_sg,
                    _kept_idx_j,
                    n_newton,
                    damping,
                    self._max_step,
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = eta_offset_batch
                log_det = _dec_ld_x_only_offset(
                    x_new, eta_offset_batch, counts_batch, r, W_logdet, d,
                    mu_logdet_dec, _kept_idx_j,
                )
                gn_x = _dec_gn_x_only_offset(
                    x_new, counts_batch, mu_newton, W_newton, d_sg, r_sg,
                    eta_offset_sg, _kept_idx_j,
                )
            else:
                # ``_nt_*``/``_ld_*``/``_gn_*`` + ``W_newton``/``W_logdet`` are
                # the shared-W kernels (+ shared W) when the hierarchy is off,
                # and the per-cell-W twins (+ per-cell W_eff) when it is on.
                x_new, _gn, _ = _nt_x_only_offset(
                    latent_init_sg,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    eta_offset_sg,
                    n_newton,
                    damping,
                    self._max_step,
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = eta_offset_batch  # carried forward to result/PPC.
                log_det = _ld_x_only_offset(
                    x_new,
                    eta_offset_batch,
                    counts_batch,
                    r,
                    W_logdet,
                    d,
                )
                # Use the *offset-aware* helper: NB factors must be
                # evaluated at ``log_rate = x − η_offset``, not at
                # ``x`` (the latter saturates ``p → 0`` and produces
                # enormous spurious gradients while the Newton MAP is
                # actually fine).
                gn_x = _gn_x_only_offset(
                    x_new,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    eta_offset_sg,
                )
            gn_blocks = {"x": gn_x}
        elif self.uses_capture:
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)
            sigma_eta_sg = jax.lax.stop_gradient(sigma_eta_batch)
            if _is_decoupled:
                x_new, eta_new, _gn, _ = _dec_nt_joint(
                    latent_init_sg,
                    eta_init_sg,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    _kept_idx_j,
                    eta_anchor_sg,
                    sigma_eta_sg,
                    n_newton,
                    damping,
                    self._max_step,
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = jax.lax.stop_gradient(eta_new)
                log_det = _dec_ld_joint(
                    x_new, eta_new, counts_batch, r, W_logdet, d,
                    mu_logdet_dec, _kept_idx_j, sigma_eta_batch,
                )
                gn_x, gn_eta = _dec_gn_joint(
                    x_new, eta_new, counts_batch, mu_newton, W_newton, d_sg,
                    r_sg, _kept_idx_j, eta_anchor_sg, sigma_eta_sg,
                )
            else:
                x_new, eta_new, _gn, _ = _nt_joint(
                    latent_init_sg,
                    eta_init_sg,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    eta_anchor_sg,
                    sigma_eta_sg,
                    n_newton,
                    damping,
                    self._max_step,
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = jax.lax.stop_gradient(eta_new)
                log_det = _ld_joint(
                    x_new, eta_new, counts_batch, r, W_logdet, d,
                    sigma_eta_batch,
                )
                # Per-block grad split for the progress display.
                gn_x, gn_eta = _gn_joint(
                    x_new,
                    eta_new,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    eta_anchor_sg,
                    sigma_eta_sg,
                )
            gn_blocks = {"x": gn_x, "η": gn_eta}
        else:
            if _is_decoupled:
                x_new, _gn, _ = _dec_nt_x_only(
                    latent_init_sg,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    _kept_idx_j,
                    n_newton,
                    damping,
                    self._max_step,
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = eta_init  # placeholder
                log_det = _dec_ld_x_only(
                    x_new, None, counts_batch, r, W_logdet, d, mu_logdet_dec,
                    _kept_idx_j, 1.0,
                )
                gn_x = _dec_gn_x_only(
                    x_new, counts_batch, mu_newton, W_newton, d_sg, r_sg,
                    _kept_idx_j,
                )
            else:
                x_new, _gn, _ = _nt_x_only(
                    latent_init_sg,
                    counts_batch,
                    mu_newton,
                    W_newton,
                    d_sg,
                    r_sg,
                    n_newton,
                    damping,
                    self._max_step,
                )
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = eta_init  # placeholder, not used downstream
                log_det = _ld_x_only(
                    x_new, None, counts_batch, r, W_logdet, d, 1.0
                )
                gn_x = _gn_x_only(
                    x_new, counts_batch, mu_newton, W_newton, d_sg, r_sg
                )
            gn_blocks = {"x": gn_x}

        # Build per-cell, per-gene log_mean (== log_rate = x − η or x
        # under no-capture) for the NB likelihood.  Under decoupling,
        # ``x_new`` holds ``x_dev`` on ``(N, G_kept)`` and we scatter to
        # the full ``(N, G_obs)`` log_mean via μ + x_dev for kept genes
        # and μ for ``_other``.
        if _is_decoupled:
            # Per-cell base = μ minus η when capture is active (frozen-η or
            # soft-η); η is None on the no-capture path.  ``mu_logdet_dec``
            # is the per-cell ``(B, G_obs)`` mean when mu is frozen per donor
            # and the shared ``(G_obs,)`` mean otherwise.
            _mu_base = (
                mu_logdet_dec
                if mu_logdet_dec.ndim == 2
                else mu_logdet_dec[None, :]
            )
            if self.uses_capture or self.freezes_eta:
                _eta_sub = (
                    eta_new[:, None] if eta_new.ndim == 1 else eta_new
                )  # (N, 1)
                base = _mu_base - _eta_sub  # (N, G_obs)
            else:
                base = jnp.broadcast_to(
                    _mu_base, (x_new.shape[0], int(mu.shape[-1]))
                )
            # Scatter x_dev contribution at kept positions.
            log_mean = base.at[:, _kept_idx_j].add(x_new)
        else:
            if self.freezes_eta:
                log_mean = x_new - eta_offset_batch[:, None]
            elif self.uses_capture:
                log_mean = x_new - eta_new[:, None]
            else:
                log_mean = x_new

        # NB log-prob in log-space.  Includes the full PMF (factorial
        # constant and all) -- the absolute scale of the loss is not
        # load-bearing for optimization, only the trend matters.  PLN
        # happens to drop ``-lgamma(u+1)`` and so its losses come out
        # negative on typical data; NBLN keeps it and so losses come
        # out positive.  Both are equivalent for optimization; same
        # convention as the DM-family losses in scribe.
        nb_lp = (
            LogMeanNegativeBinomial(log_mean=log_mean, concentration=r[None, :])
            .log_prob(counts_batch)
            .sum(axis=-1)
        )

        # MVN prior on the per-cell latent.
        #   Legacy: ``x ~ N(μ, Σ)`` on G_obs, with Σ = W Wᵀ + diag(d).
        #     diff = x − μ, MVN dim G = G_obs.
        #   Decoupled: ``x_dev ~ N(0, Σ_kept)`` on G_kept, with
        #     Σ_kept = W Wᵀ + diag(d), W: (G_kept, k), d: (G_kept,).
        #     diff = x_dev (zero-centred), MVN dim G = G_kept.
        # μ has moved into the NB likelihood under decoupling — see the
        # math contract at the top of ``_newton_nbln.py``'s decoupled
        # section.
        if _is_decoupled:
            diff = x_new  # x_dev_new, already zero-centred
            G = int(_layout.G_kept)
        else:
            # ``mu_cell_live`` is a per-cell ``(B, G)`` mean when mu is
            # frozen per donor; otherwise broadcast the shared ``(G,)``.
            if mu_cell_live is not None:
                diff = x_new - mu_cell_live
            else:
                diff = x_new - mu[None, :]
            G = int(mu.shape[-1])
        # ``_woodbury_quad``/``_woodbury_logdet`` + ``W_logdet`` are the shared
        # helpers (+ shared W) when the hierarchy is off, and the per-cell-W
        # twins (+ per-cell W_eff_live) when it is on -- so each cell's MVN
        # prior uses its donor's Σ_{σ(c)}. ``log_det_sigma`` is then per-cell
        # (shape (B,)) under the hierarchy and a scalar otherwise; both
        # broadcast against the per-cell ``quad``.
        quad = _woodbury_quad(W_logdet, d, diff)
        log_det_sigma = _woodbury_logdet(W_logdet, d)
        mvn_lp = (
            -0.5 * quad - 0.5 * log_det_sigma - 0.5 * G * jnp.log(2 * jnp.pi)
        )

        # TruncN prior on η (per-cell scale).
        # Skip when eta is frozen — the prior is a Delta at the freeze
        # value and contributes a constant the optimizer can ignore.
        if self.uses_capture and not self.freezes_eta:
            eta_lp = dist.TruncatedNormal(
                eta_anchor_batch, sigma_eta_batch, low=0.0
            ).log_prob(eta_new)
        else:
            eta_lp = jnp.zeros_like(nb_lp)

        # Laplace correction: -½ log det(-H).
        laplace_corr = -0.5 * log_det

        elbo_per_cell = nb_lp + mvn_lp + eta_lp + laplace_corr
        # Single-cell finite-loss guard (matches PLN engine).
        elbo_per_cell = jnp.where(
            jnp.isfinite(elbo_per_cell),
            elbo_per_cell,
            jnp.zeros_like(elbo_per_cell),
        )

        # Global-parameter priors (NOT scaled by data_scale — these are
        # parameter priors, not per-cell likelihood contributions).
        # Skip frozen parameters: their prior is a Delta at the freeze
        # value, contributing a constant to the loss (no gradient).
        global_prior_lp = jnp.zeros(())
        if self._prior_r is not None and "r" not in self._frozen_params:
            r_loc_prior = self._prior_r
            global_prior_lp = global_prior_lp + jnp.sum(
                dist.Normal(r_loc_prior["loc"], r_loc_prior["scale"]).log_prob(
                    params_full["r_loc"]
                )
            )
        if self._prior_mu is not None and "mu" not in self._frozen_params:
            mu_prior = self._prior_mu
            global_prior_lp = global_prior_lp + jnp.sum(
                dist.Normal(mu_prior["loc"], mu_prior["scale"]).log_prob(
                    params_full["mu"]
                )
            )

        # --- Per-donor correlation hierarchy: program-scale prior ------------
        # The non-centered prior on s_d: the raw effects are standard Normal
        # (the NCP latent), and the shared between-donor scale tau_raw gets the
        # softplus-Normal hyperprior (same defaults as the SVI primitive). The
        # sum-to-zero gauge is enforced by ``program_scales_from_raw`` and adds
        # no prior term. These mirror the ``_prior_r``/``_prior_mu`` terms: a
        # parameter prior, not scaled by ``data_scale``.
        if self._hier_active:
            global_prior_lp = global_prior_lp + jnp.sum(
                dist.Normal(0.0, 1.0).log_prob(
                    params_full["program_scale_raw"]
                )
            )
            global_prior_lp = global_prior_lp + dist.Normal(
                _DEFAULT_TAU_LOC, _DEFAULT_TAU_SCALE
            ).log_prob(params_full["program_scale_tau_raw"])

        # Phase-3: W-prior contribution on the gauge-invariant projection.
        # For NBLN, the rigid-translation gauge means raw W has an
        # unidentified rank-1 all-ones component; shrinking that
        # component is wasted regularization capacity and biases the
        # column norms used for the rank diagnostic.  Project to W_⟂
        # here so the prior targets the biologically meaningful signal.
        # n_constraints=1 tells the strategy that each column lives in
        # a (G-1)-dim subspace (sums to zero) so it uses the correct
        # normalizer ``-(G-1) log σ_k`` instead of ``-G log σ_k``.
        _W_raw = params_full["W"]
        _W_for_prior = _W_raw - jnp.mean(_W_raw, axis=0, keepdims=True)
        w_aux = {
            name: params_full[name] for name in self._w_prior.aux_param_names
        }
        global_prior_lp = global_prior_lp + self._w_prior.log_prior(
            _W_for_prior,
            w_aux,
            n_constraints=1,
        )

        loss = -data_scale * jnp.sum(elbo_per_cell) - global_prior_lp
        return loss, (x_new, eta_new, gn_blocks)

    # --- Final convergence check ----------------------------------------

    def final_sweep(
        self,
        params: Dict[str, jnp.ndarray],
        latent_loc: jnp.ndarray,
        eta_loc: Optional[jnp.ndarray],
        eta_anchor: Optional[jnp.ndarray],
        count_data: jnp.ndarray,
        aux_data: Dict[str, jnp.ndarray],
        n_newton: int,
        damping: float,
    ) -> FinalSweepResult:
        # Round-5 R5-1 fix: splice frozen values into params_full at
        # entry — final_sweep reads params["mu"]/params["r_loc"] which
        # are absent from the reduced optimizer dict when frozen.
        params_full = self._splice_frozen(params)

        mu = jax.lax.stop_gradient(params_full["mu"])
        W = jax.lax.stop_gradient(params_full["W"])
        d = jax.lax.stop_gradient(self._pos_forward(params_full["d_loc"]))
        r = jax.lax.stop_gradient(self._pos_forward(params_full["r_loc"]))

        # Per-cell mean for the Newton kernels.  Defaults to the shared
        # ``(G,)`` mu; gathered to a full-data per-cell ``(N, G)`` tensor
        # when mu is frozen per donor (mirrors ``mu_newton`` in loss_fn).
        mu_fs = mu

        # Per-donor correlation hierarchy: per-cell effective loadings for the
        # full-data final sweep (mirrors loss_fn). All inputs are stop-grad
        # here, so we fold s into a single per-cell ``(N, G, k)`` tensor and
        # select the per-cell-W Newton twins; ``W_fs`` aliases the shared W
        # when the hierarchy is off (bit-equal to today).
        if self._hier_active:
            _s_fs = program_scales_from_raw(
                params_full["program_scale_raw"],
                params_full["program_scale_tau_raw"],
            )
            _ds_fs = aux_data["dataset_indices"]  # (N,)
            W_fs = jax.lax.stop_gradient(
                W[None, :, :] * _s_fs[_ds_fs][:, None, :]
            )  # (N, G, K)
            _fs_nt_x_only = laplace_newton_batch_x_only_percellW
            _fs_nt_x_only_offset = laplace_newton_batch_x_only_offset_percellW
            _fs_nt_joint = laplace_newton_batch_percellW
            _fs_dec_nt_x_only = laplace_newton_batch_x_only_decoupled_percellW
            _fs_dec_nt_x_only_offset = (
                laplace_newton_batch_x_only_offset_decoupled_percellW
            )
            _fs_dec_nt_joint = laplace_newton_batch_decoupled_percellW
            # Per-donor frozen mu: gather each cell's prior mean and use
            # the per-cell-mu Newton twins (mirrors loss_fn).
            if self._mu_per_donor:
                mu_fs = mu[_ds_fs]  # (N, G)
                _fs_nt_x_only = laplace_newton_batch_x_only_percellWmu
                _fs_nt_x_only_offset = (
                    laplace_newton_batch_x_only_offset_percellWmu
                )
                _fs_nt_joint = laplace_newton_batch_percellWmu
                _fs_dec_nt_x_only = (
                    laplace_newton_batch_x_only_decoupled_percellWmu
                )
                _fs_dec_nt_x_only_offset = (
                    laplace_newton_batch_x_only_offset_decoupled_percellWmu
                )
                _fs_dec_nt_joint = laplace_newton_batch_decoupled_percellWmu
        else:
            W_fs = W
            _fs_nt_x_only = laplace_newton_batch_x_only
            _fs_nt_x_only_offset = laplace_newton_batch_x_only_offset
            _fs_nt_joint = laplace_newton_batch
            _fs_dec_nt_x_only = laplace_newton_batch_x_only_decoupled
            _fs_dec_nt_x_only_offset = (
                laplace_newton_batch_x_only_offset_decoupled
            )
            _fs_dec_nt_joint = laplace_newton_batch_decoupled

        # Decoupled-layout branch (Commit 2b): the final sweep mirrors
        # ``loss_fn``'s four-way Newton dispatch but lives on the kept
        # axis.  ``latent_loc`` carries ``x_dev`` shape ``(N, G_kept)``;
        # ``x_final`` returns the same shape.
        _layout = self._axis_layout
        _is_decoupled = _layout is not None and _layout.decoupled
        if _is_decoupled:
            _kept_idx_j = jnp.asarray(_layout.kept_idx, dtype=jnp.int32)

        # Four-way Newton dispatch (parallels loss_fn).
        if self.freezes_eta:
            # x-only Newton with fixed per-cell offset from aux_data.
            eta_offset = aux_data["eta_frozen"]
            if _is_decoupled:
                x_final, gn_final, _ = _fs_dec_nt_x_only_offset(
                    latent_loc, count_data, mu_fs, W_fs, d, r, eta_offset,
                    _kept_idx_j, n_newton, damping, self._max_step,
                )
            else:
                x_final, gn_final, _ = _fs_nt_x_only_offset(
                    latent_loc,
                    count_data,
                    mu_fs,
                    W_fs,
                    d,
                    r,
                    eta_offset,
                    n_newton,
                    damping,
                    self._max_step,
                )
            # Carry the frozen eta forward so the result's `eta_loc` is
            # populated — PPC needs this to compute `log_mean = x − eta`.
            eta_final = eta_offset
        elif self.uses_capture:
            sigma_eta_full = aux_data["eta_scale"]
            if _is_decoupled:
                x_final, eta_final, gn_final, _ = _fs_dec_nt_joint(
                    latent_loc, eta_loc, count_data, mu_fs, W_fs, d, r,
                    _kept_idx_j, eta_anchor, sigma_eta_full, n_newton,
                    damping, self._max_step,
                )
            else:
                x_final, eta_final, gn_final, _ = _fs_nt_joint(
                    latent_loc,
                    eta_loc,
                    count_data,
                    mu_fs,
                    W_fs,
                    d,
                    r,
                    eta_anchor,
                    sigma_eta_full,
                    n_newton,
                    damping,
                    self._max_step,
                )
        else:
            if _is_decoupled:
                x_final, gn_final, _ = _fs_dec_nt_x_only(
                    latent_loc, count_data, mu_fs, W_fs, d, r, _kept_idx_j,
                    n_newton, damping, self._max_step,
                )
            else:
                x_final, gn_final, _ = _fs_nt_x_only(
                    latent_loc,
                    count_data,
                    mu_fs,
                    W_fs,
                    d,
                    r,
                    n_newton,
                    damping,
                    self._max_step,
                )
            eta_final = None

        return FinalSweepResult(
            latent_loc=x_final,
            eta_loc=eta_final,
            final_grad_norms=gn_final,
        )

    # --- Global uncertainty (NBLN r_g and mu_g) --------------------------

    def compute_global_uncertainty(
        self,
        params: Dict[str, jnp.ndarray],
        latent_loc: jnp.ndarray,
        eta_loc: Optional[jnp.ndarray],
        eta_anchor: Optional[jnp.ndarray],
        count_data: jnp.ndarray,
        aux_data: Dict[str, jnp.ndarray],
        model_config: ModelConfig,
    ) -> Dict[str, jnp.ndarray]:
        """Compute diagonal Laplace posteriors for ``r_g`` and ``mu_g``.

        Returns per-gene Gaussian posterior parameters
        ``(r_loc, r_scale)`` and ``(mu_loc, mu_scale)`` in the
        unconstrained coordinate space.  All scales are derived from
        the diagonal of the profiled (marginal) Hessian — the per-cell
        latent uncertainty is integrated out via the Schur complement
        using closed-form Woodbury algebra at the converged MAP.

        For capture-active fits, both the ``r`` and ``mu`` paths use
        the **joint (x, η) inverse**, not just the marginal x-block.
        For ``r_g`` this includes the cross term ``H_{r_g, η_c}`` via
        the identity ``H_{r_g, η_c} = -H_{r_g, x_{c,g}}`` (since NBLN's
        likelihood depends on ``log_mean = x − η``).  For ``mu``, the
        cross with η is zero (μ enters only the latent prior).

        For ``mu``, we use a **diagonal-Σ approximation**: each gene's
        marginal posterior on ``μ_g`` is computed as if ``Σ^{-1}`` were
        gene-diagonal.  The diagonal value ``Σ^{-1}_{gg}`` is exact
        (from ``woodbury_inv_diag``); only the off-diagonal coupling
        between ``μ_g`` and ``μ_{g'}`` through the MVN prior is dropped.
        This is the per-gene marginal-posterior approximation already
        accepted for the ``r`` path.

        Informative priors (when supplied via ``informative_priors=...``)
        contribute their precision ``1/scale**2`` directly to the
        diagonal of the negative-objective Hessian — equivalent to
        what the loss already adds via autodiff during training.

        Returns
        -------
        dict
            ``r_loc, r_scale`` (G,) — NBLN dispersion posterior.
            ``mu_loc, mu_scale`` (G,) — NBLN latent prior mean
            posterior in log-rate coordinates.
            Diagnostics keys: ``r_hessian_diag``, ``r_hessian_min``,
            ``r_hessian_floor_count``, ``r_curvature_floor`` and the
            ``mu_*`` analogues.
        """
        # Round-5 R5-2 fix: splice frozen values into params_full at
        # entry — compute_global_uncertainty reads params["mu"]/["r_loc"]
        # which are absent from the reduced optimizer dict when frozen.
        params_full = self._splice_frozen(params)

        # Commit 2b: under the decoupled layout, the profiled μ Hessian
        # derivation differs because μ no longer enters the MVN prior —
        # it lives only in the NB observation likelihood.  The
        # decoupled branch is implemented inline below at the
        # ``H_mu_profiled_diag`` assembly; the per-cell inverse-block
        # plumbing above produces the right factors for both paths
        # (it always operates on the W/d natural axis, which is
        # G_kept under decoupling and G_obs under legacy).
        _layout = self._axis_layout
        _is_decoupled = _layout is not None and _layout.decoupled

        mu = params_full["mu"]
        W = params_full["W"]
        d = self._pos_forward(params_full["d_loc"])
        r_loc = params_full["r_loc"]
        pos_fwd = self._pos_forward
        r = pos_fwd(r_loc)

        N, G = count_data.shape

        # Compute effective log-rate at each cell, on the FULL G_obs
        # axis regardless of layout.  Under decoupling, ``latent_loc``
        # has shape ``(N, G_kept)`` and carries ``x_dev``; we scatter
        # ``μ + x_dev`` at kept positions and use ``μ`` at ``other_idx``,
        # then subtract η if capture is active.  Under legacy the
        # formula is unchanged.
        if _is_decoupled:
            _kept_idx_j = jnp.asarray(_layout.kept_idx, dtype=jnp.int32)
            # Per-cell mean: gather μ^{(σ(c))} when mu is frozen per donor
            # (shape (D, G_obs)), else the shared (G_obs,).
            if self._mu_per_donor and "dataset_indices" in aux_data:
                _mu_rows = mu[aux_data["dataset_indices"]]  # (N, G_obs)
            else:
                _mu_rows = mu[None, :]  # (1, G_obs)
            # Base = μ across cells minus η when capture is on.
            if self.uses_capture and eta_loc is not None:
                _base = _mu_rows - eta_loc[:, None]
            else:
                _base = jnp.broadcast_to(_mu_rows, (N, G))
            # Scatter ``x_dev`` at kept positions; ``_other`` retains μ − η.
            log_rate_all = _base.at[:, _kept_idx_j].add(latent_loc)
        else:
            if self.uses_capture and eta_loc is not None:
                log_rate_all = latent_loc - eta_loc[:, None]
            else:
                log_rate_all = latent_loc

        # --- Per-cell, per-gene NB Hessian entries via autodiff -------
        # H_{r_loc_g, r_loc_g} and H_{r_loc_g, log_rate_g} per gene per
        # cell.  log_rate = x - η so this autodiff covers both x and η
        # cross-partials via chain rule (H_{r,η} = -H_{r,x}).
        def _nb_lp_gene(r_loc_g, log_rate_g, u_g):
            """NB log-prob for one gene in one cell, as a function of
            unconstrained r_loc_g and log_rate_g = x_g - eta."""
            r_g = pos_fwd(r_loc_g)
            r_g = jnp.maximum(r_g, 1e-6)
            from jax.scipy.special import gammaln

            v = r_g + jnp.exp(log_rate_g)
            return (
                gammaln(u_g + r_g)
                - gammaln(r_g)
                - gammaln(u_g + 1.0)
                + r_g * jnp.log(r_g / v)
                + u_g * jnp.log(jnp.exp(log_rate_g) / v)
            )

        def _neg_nb_rr(r_loc_g, log_rate_g, u_g):
            return -jax.grad(jax.grad(_nb_lp_gene, argnums=0), argnums=0)(
                r_loc_g, log_rate_g, u_g
            )

        def _neg_nb_rx(r_loc_g, log_rate_g, u_g):
            # H_{r_loc_g, x_g}_neg-loss: cross-derivative of the
            # negative log-prob.  For NBLN, H_{r,η} = -H_{r,x} by chain
            # rule on log_rate = x - η, so we compute this once and
            # use the sign relation when assembling capture-aware terms.
            return -jax.grad(jax.grad(_nb_lp_gene, argnums=0), argnums=1)(
                r_loc_g, log_rate_g, u_g
            )

        _neg_nb_rr_genes = jax.vmap(_neg_nb_rr, in_axes=(0, 0, 0))
        _neg_nb_rx_genes = jax.vmap(_neg_nb_rx, in_axes=(0, 0, 0))
        _neg_nb_rr_batch = jax.vmap(_neg_nb_rr_genes, in_axes=(None, 0, 0))
        _neg_nb_rx_batch = jax.vmap(_neg_nb_rx_genes, in_axes=(None, 0, 0))

        # Chunked AD pass over cells to avoid GPU OOM.
        _chunk_size_hess = min(512, N)
        _lr_chunks = [
            log_rate_all[i : i + _chunk_size_hess]
            for i in range(0, N, _chunk_size_hess)
        ]
        _u_chunks = [
            count_data[i : i + _chunk_size_hess]
            for i in range(0, N, _chunk_size_hess)
        ]
        H_rr_all = jnp.concatenate(
            [
                _neg_nb_rr_batch(r_loc, lr, u)
                for lr, u in zip(_lr_chunks, _u_chunks)
            ],
            axis=0,
        )  # (N, G)
        H_rx_all = jnp.concatenate(
            [
                _neg_nb_rx_batch(r_loc, lr, u)
                for lr, u in zip(_lr_chunks, _u_chunks)
            ],
            axis=0,
        )  # (N, G)

        # --- Per-cell joint-inverse Woodbury structure ----------------
        # For each cell we need:
        #   A_inv_diag_c       — diag of (-H_xx_c)^{-1}, shape (G,)
        #   joint_inv_xx_diag  — diag of joint inv top-left block, (G,)
        #   joint_inv_xη_c     — vector (G,) of joint inverse cross-block
        #   joint_inv_ηη_c     — scalar
        #
        # joint_inv_xx = A^{-1} + (A^{-1} a)(A^{-1} a)^T / s_η  (capture-on)
        #              = A^{-1}                                 (capture-off)
        # joint_inv_xη = A^{-1} a / s_η                          (capture-on)
        # joint_inv_ηη = 1 / s_η                                 (capture-on)
        #
        # where s_η = sum(a) + 1/σ_η² − a^T A^{-1} a.
        from ._newton_nbln import (
            _nb_factors,
            _woodbury_factors,
            _solve_A,
        )

        # Frozen-eta acts as infinite-precision capture: η is a fixed
        # constant, so its contribution to the joint (x, η) inverse
        # collapses (s_η → ∞, joint_inv_xη → 0, joint_inv_ηη → 0).  We
        # take the x-only path here, matching the conceptual reduction.
        # For soft-cascade / capture-anchor fits the per-cell σ_η lives
        # in ``aux_data["eta_scale"]`` (populated by ``init_state``
        # whenever ``eta_scale_per_cell is not None``); reading it
        # directly mirrors ``final_sweep`` and avoids the placeholder
        # ``self._sigma_M = 1.0`` that biased the r_scale / mu_scale
        # diagnostics on soft-eta cascade fits.
        capture_on = (
            self.uses_capture
            and eta_loc is not None
            and not self.freezes_eta
        )
        if capture_on:
            sigma_eta_full = aux_data["eta_scale"]
        else:
            sigma_eta_full = jnp.ones((N,))  # unused in capture-off path

        def _A_inv_diag_from_factors(factors: Dict) -> jnp.ndarray:
            """diag(A^{-1}) using the existing Woodbury factors.

            A = M̃ - V K^{-1} V^T where M̃ = diag(a + 1/d), so by Woodbury
            A^{-1} = M̃^{-1} + M̃^{-1} V S^{-1} V^T M̃^{-1} = diag(m_inv) + Q Q^T
            where Q = M̃^{-1} V L_S^{-T}.  diag(Q Q^T)_g = ||Q_g||² is
            computed via one triangular solve at cost O(G·k + k²).
            """
            m_inv = factors["m_inv"]
            V = factors["V"]
            L_S = factors["L_S"]
            V_scaled = m_inv[:, None] * V  # (G, k)
            # Solve L_S Y = V_scaled^T  =>  Y = L_S^{-1} V_scaled^T  (k, G).
            # Then rowsumsq(V_scaled L_S^{-T}) = colsumsq(Y).
            Y = jax.scipy.linalg.solve_triangular(L_S, V_scaled.T, lower=True)
            correction = jnp.sum(Y * Y, axis=0)  # (G,)
            return m_inv + correction

        # Resolve kept_idx once for the decoupled per-cell function.
        if _is_decoupled:
            _kept_idx_inv = jnp.asarray(_layout.kept_idx, dtype=jnp.int32)
        else:
            _kept_idx_inv = None

        def _per_cell_inverse_blocks(
            log_rate_c: jnp.ndarray,
            u_c: jnp.ndarray,
            sigma_eta_c: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Return ``(joint_inv_xx_diag, joint_inv_xη, joint_inv_ηη, a_full)``.

            Under legacy: ``joint_inv_*`` live on ``(G_obs,)``; ``a_full``
            is the same ``a`` already used inside the Woodbury (returned
            for parity with the decoupled path so the caller can use it
            uniformly).

            Under decoupling: ``joint_inv_xx_diag`` and ``joint_inv_xeta``
            live on ``(G_kept,)``; ``joint_inv_etaeta`` is scalar; the
            Woodbury factors operate on ``a_kept = a_full[kept_idx]``,
            while ``a_full`` (G_obs,) is returned for the μ profiled
            formula (which needs the NB curvature at every gene
            including ``_other``).
            """
            nb = _nb_factors(log_rate_c, u_c, r)
            a_full = nb["a"]
            if _is_decoupled:
                a_for_woodbury = a_full[_kept_idx_inv]
            else:
                a_for_woodbury = a_full
            factors = _woodbury_factors(W, d, a_for_woodbury, damping=0.0)
            A_inv_diag = _A_inv_diag_from_factors(factors)
            if capture_on:
                A_inv_a = _solve_A(factors, a_for_woodbury)
                # Schur scalar s_η: η couples to ALL genes' NB curvature
                # under both layouts (the decoupled `_other` still
                # contributes via its NB likelihood term, just not via
                # the latent x_dev).  ``sum(a_full)`` is the right
                # denominator regardless of layout.
                s_eta = jnp.maximum(
                    jnp.sum(a_full)
                    + 1.0 / (sigma_eta_c**2)
                    - jnp.dot(a_for_woodbury, A_inv_a),
                    1e-30,
                )
                joint_inv_xx_diag = A_inv_diag + (A_inv_a**2) / s_eta
                joint_inv_xeta = A_inv_a / s_eta
                joint_inv_etaeta = 1.0 / s_eta
            else:
                joint_inv_xx_diag = A_inv_diag
                joint_inv_xeta = jnp.zeros_like(a_for_woodbury)
                joint_inv_etaeta = jnp.asarray(0.0, dtype=a_for_woodbury.dtype)
            return joint_inv_xx_diag, joint_inv_xeta, joint_inv_etaeta, a_full

        _per_cell_vmap = jax.vmap(_per_cell_inverse_blocks, in_axes=(0, 0, 0))

        # Chunked pass over cells to keep GPU memory bounded.
        _chunk_size = min(256, N)
        _chunks_lr = [
            log_rate_all[i : i + _chunk_size] for i in range(0, N, _chunk_size)
        ]
        _chunks_u = [
            count_data[i : i + _chunk_size] for i in range(0, N, _chunk_size)
        ]
        _chunks_sigma = [
            sigma_eta_full[i : i + _chunk_size]
            for i in range(0, N, _chunk_size)
        ]
        _joint_xx_chunks = []
        _joint_xeta_chunks = []
        _joint_etaeta_chunks = []
        _a_full_chunks = []
        for lr, u, se in zip(_chunks_lr, _chunks_u, _chunks_sigma):
            jx, jxe, jee, af = _per_cell_vmap(lr, u, se)
            _joint_xx_chunks.append(jx)
            _joint_xeta_chunks.append(jxe)
            _joint_etaeta_chunks.append(jee)
            _a_full_chunks.append(af)
        # Under legacy: ``(N, G_obs)``.  Under decoupling: ``(N, G_kept)``.
        joint_inv_xx_diag_all = jnp.concatenate(_joint_xx_chunks, axis=0)
        joint_inv_xeta_all = jnp.concatenate(_joint_xeta_chunks, axis=0)
        joint_inv_etaeta_all = jnp.concatenate(
            _joint_etaeta_chunks, axis=0
        )  # (N,)
        # ``a_full_all`` is always on ``(N, G_obs)`` — the NB curvature
        # at every gene, used by the decoupled μ formula.
        a_full_all = jnp.concatenate(_a_full_chunks, axis=0)

        # --- r profiled Hessian diagonal (Round-3 Finding B) -----------
        # capture-off:  Schur_g = sum_c H_rx_cg² · joint_inv_xx_diag_cg
        # capture-on:   Schur_g = sum_c H_rx_cg² · B_cg
        #               where B_cg = joint_inv_xx_diag_cg
        #                            - 2 · joint_inv_xη_cg
        #                            + joint_inv_ηη_c
        # The bracket comes from H_{r,η}_cg = -H_{r,x}_cg (chain rule on
        # log_rate = x - η).
        H_rr_summed = jnp.sum(H_rr_all, axis=0)  # (G,)
        if _is_decoupled:
            # Decoupled bracket structure (Commit 2b): the joint
            # inverse blocks live on the kept axis, so we assemble the
            # per-gene-on-G_obs Schur correction in two pieces:
            #   • kept gene g at position k_g — bracket combines the
            #     kept-axis ``M_xx``, ``M_xη``, and the scalar ``M_ηη``.
            #   • ``_other`` — no x_dev coupling, so only the η block
            #     contributes (``M_ηη``).
            # Under no-capture / frozen-η the ``M_xη`` and ``M_ηη`` blocks
            # are zero (by construction inside _per_cell_inverse_blocks),
            # so the formulas collapse to "kept gets ``M_xx_diag``, ``_other``
            # gets zero" — both correct.
            kept_idx_np = np.asarray(_layout.kept_idx)
            other_idx_int = int(_layout.other_idx)
            bracket_kept = (
                joint_inv_xx_diag_all
                - 2.0 * joint_inv_xeta_all
                + joint_inv_etaeta_all[:, None]
            )  # (N, G_kept)
            H_rx_kept = H_rx_all[:, kept_idx_np]  # (N, G_kept)
            H_rx_other = H_rx_all[:, other_idx_int]  # (N,)
            r_schur_kept = jnp.sum(
                H_rx_kept ** 2 * bracket_kept, axis=0
            )  # (G_kept,)
            r_schur_other = jnp.sum(
                H_rx_other ** 2 * joint_inv_etaeta_all
            )  # scalar
            r_schur = jnp.zeros((G,), dtype=H_rr_summed.dtype)
            r_schur = r_schur.at[kept_idx_np].set(r_schur_kept)
            r_schur = r_schur.at[other_idx_int].set(r_schur_other)
        else:
            if capture_on:
                bracket = (
                    joint_inv_xx_diag_all
                    - 2.0 * joint_inv_xeta_all
                    + joint_inv_etaeta_all[:, None]
                )  # (N, G)
            else:
                bracket = joint_inv_xx_diag_all
            r_schur = jnp.sum(H_rx_all**2 * bracket, axis=0)  # (G,)
        H_r_profiled_diag = H_rr_summed - r_schur

        # Prior precision injection on r_loc (Round-1 Finding 2).
        if self._prior_r is not None and "r" not in self._frozen_params:
            H_r_profiled_diag = H_r_profiled_diag + 1.0 / (
                self._prior_r["scale"] ** 2
            )

        if "r" in self._frozen_params:
            # Frozen r: post-fit Hessian is meaningless (r doesn't move).
            # The authoritative r posterior lives on the cascade_source
            # and is accessed at get_distributions / PPC time via
            # moment-matching SVI samples in the NBLN target coordinate.
            # Emit NaN sentinels here; downstream code checks
            # `self.frozen_params` to route through cascade_source.
            r_scale = jnp.full_like(r_loc, jnp.nan)
            r_diag = {
                "hessian_min": jnp.asarray(jnp.nan, dtype=jnp.float32),
                "floor_count": jnp.asarray(0, dtype=jnp.int32),
                "curvature_floor": jnp.asarray(jnp.nan, dtype=jnp.float32),
            }
        else:
            r_scale, r_diag = curvature_to_scale(H_r_profiled_diag)

        # --- mu profiled Hessian diagonal ---
        # The legacy and decoupled formulas differ structurally because
        # ``μ`` enters different parts of the per-cell log-density:
        #
        # • Legacy (``correlate_other_column=True``): ``x ~ N(μ, Σ)``;
        #   ``μ`` lives in the MVN prior, NOT the NB likelihood.  The
        #   profiled curvature is the diagonal-Σ approximation:
        #     H_μμ_profile_g = Σ⁻¹_{gg} · [N − Σ⁻¹_{gg} · Σ_c joint_inv_xx_diag_cg]
        #
        # • Decoupled (Commit 2b): ``x_dev ~ N(0, Σ_kept)``; ``μ`` moves
        #   into the NB likelihood (``log_rate_g = μ_g + x_dev[k_g] − η``
        #   for kept genes, ``μ_g − η`` for ``_other``).  The profiled
        #   curvature is then:
        #     For kept gene g at k_g:
        #       H_μμ_profile_g = Σ_c [a_full_cg − a_full_cg² · bracket_kept_{c,k_g}]
        #     For ``_other``:
        #       H_μμ_profile_other = Σ_c [a_full_c,other − a_full_c,other² · M_ηη_c]
        #   where bracket_kept = M_xx − 2·M_xη + M_ηη as for the r path.
        #   See ``paper/_nb_lognormal.qmd`` §sec-nbln-decorrelate-mu-uncertainty.
        if _is_decoupled:
            # ``bracket_kept`` and ``kept_idx_np`` / ``other_idx_int``
            # already built in the r Schur block above (same chunk of
            # data) — reuse them here.
            a_kept_all = a_full_all[:, kept_idx_np]  # (N, G_kept)
            a_other_all = a_full_all[:, other_idx_int]  # (N,)
            H_mu_kept = (
                jnp.sum(a_kept_all, axis=0)
                - jnp.sum(a_kept_all ** 2 * bracket_kept, axis=0)
            )  # (G_kept,)
            H_mu_other = (
                jnp.sum(a_other_all)
                - jnp.sum(a_other_all ** 2 * joint_inv_etaeta_all)
            )  # scalar
            H_mu_profiled_diag = jnp.zeros((G,), dtype=mu.dtype)
            H_mu_profiled_diag = H_mu_profiled_diag.at[kept_idx_np].set(
                H_mu_kept
            )
            H_mu_profiled_diag = H_mu_profiled_diag.at[other_idx_int].set(
                H_mu_other
            )
        else:
            sigma_inv_diag = woodbury_inv_diag(W, d)  # (G,) — exact
            sum_joint_xx_diag = jnp.sum(
                joint_inv_xx_diag_all, axis=0
            )  # (G,)
            H_mu_profiled_diag = sigma_inv_diag * (
                float(N) - sigma_inv_diag * sum_joint_xx_diag
            )

        # Prior precision injection on mu.
        if self._prior_mu is not None and "mu" not in self._frozen_params:
            H_mu_profiled_diag = H_mu_profiled_diag + 1.0 / (
                self._prior_mu["scale"] ** 2
            )

        if "mu" in self._frozen_params:
            # Frozen mu: same sentinel pattern as frozen r.  Authoritative
            # mu posterior is at cascade_source; this field is NaN to
            # signal "do not read directly; route through cascade".
            mu_scale = jnp.full_like(mu, jnp.nan)
            mu_diag = {
                "hessian_min": jnp.asarray(jnp.nan, dtype=jnp.float32),
                "floor_count": jnp.asarray(0, dtype=jnp.int32),
                "curvature_floor": jnp.asarray(jnp.nan, dtype=jnp.float32),
            }
        else:
            mu_scale, mu_diag = curvature_to_scale(H_mu_profiled_diag)

        return {
            "r_loc": r_loc,
            "r_scale": r_scale,
            "r_hessian_diag": H_r_profiled_diag,
            "r_hessian_min": r_diag["hessian_min"],
            "r_hessian_floor_count": r_diag["floor_count"],
            "r_curvature_floor": r_diag["curvature_floor"],
            "mu_loc": mu,
            "mu_scale": mu_scale,
            "mu_hessian_diag": H_mu_profiled_diag,
            "mu_hessian_min": mu_diag["hessian_min"],
            "mu_hessian_floor_count": mu_diag["floor_count"],
            "mu_curvature_floor": mu_diag["curvature_floor"],
        }

    # --- Result packaging -----------------------------------------------

    def pack_result(
        self,
        params: Dict[str, jnp.ndarray],
        final: FinalSweepResult,
        losses: np.ndarray,
        n_steps_run: int,
        model_config: ModelConfig,
        early_stopped: bool,
        best_loss: float,
        stopped_at_step: int,
        divergence_aborted: bool,
        global_uncertainty: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> LaplaceRunResult:
        # Round-5 R5-5 fix: splice frozen values back so run_result.globals
        # carries the FULL params dict (downstream _format_laplace_results
        # and result-packaging consumers expect every key to be present).
        params_full = self._splice_frozen(params)
        # ``stop_gradient``-wrapped entries from _splice_frozen are fine
        # here because we're outside the autodiff graph at result-
        # packaging time; the jax tracer simply unwraps them.

        # Phase-3: W-prior diagnostics.  Same W_⟂ projection convention as
        # loss_fn — the strategy's headline rank lives in compositional
        # coordinates so it aligns with the prior the loss actually used.
        # The obs model adds the ``column_frobenius_raw`` side-channel
        # from the raw W (parallel to the compositional norm).
        _W_raw = params_full["W"]
        _W_for_prior = _W_raw - jnp.mean(_W_raw, axis=0, keepdims=True)
        w_aux = {
            name: params_full[name] for name in self._w_prior.aux_param_names
        }
        strategy_diag = self._w_prior.diagnostics(
            _W_for_prior,
            w_aux,
            n_constraints=1,
        )
        strategy_diag["column_frobenius_raw"] = jnp.linalg.norm(_W_raw, axis=0)
        w_prior_diagnostics = {
            "strategy_type": self._w_prior.type_name,
            **strategy_diag,
        }

        # --- Per-donor frozen mu: pool for the standard accessors --------
        # The per-cell solve used each donor's mean ``mu^{(σ(c))}`` (so the
        # per-cell ``x_loc`` already encodes it).  Every downstream
        # accessor / PPC path expects a single ``(G,)`` ``globals["mu"]``,
        # so we pool the ``(D, G)`` frozen table to its per-gene mean and
        # surface the unpooled table separately on the result.  ``mu`` is
        # frozen here, so pooling does not perturb any learned quantity.
        gene_mean_per_dataset = None
        if self._mu_per_donor:
            _mu_dg = jnp.asarray(params_full["mu"])  # (D, G)
            gene_mean_per_dataset = _mu_dg
            _mu_pooled = jnp.mean(_mu_dg, axis=0)  # (G,)
            params_full = {**params_full, "mu": _mu_pooled}
            # Pool the global-uncertainty mu fields to (G,) too: ``mu_loc``
            # is the per-donor table, ``mu_scale`` a NaN freeze sentinel.
            if global_uncertainty:
                global_uncertainty = dict(global_uncertainty)
                for _k in ("mu_loc", "mu_scale"):
                    _v = global_uncertainty.get(_k)
                    if _v is not None and jnp.asarray(_v).ndim == 2:
                        global_uncertainty[_k] = jnp.mean(
                            jnp.asarray(_v), axis=0
                        )

        return LaplaceRunResult(
            globals=params_full,
            x_loc=final.latent_loc,
            eta_loc=final.eta_loc,
            final_grad_norms=final.final_grad_norms,
            losses=jnp.asarray(losses, dtype=jnp.float32),
            n_steps_run=n_steps_run,
            model_config=model_config,
            early_stopped=early_stopped,
            best_loss=best_loss,
            stopped_at_step=stopped_at_step,
            divergence_aborted=divergence_aborted,
            global_uncertainty=global_uncertainty or {},
            frozen_params=self._frozen_params,
            w_prior_diagnostics=w_prior_diagnostics,
            # Persist the axis layout so the bridge in
            # ``inference/laplace.py`` can attach it to
            # ``ScribeLaplaceResults.axis_layout``.  ``None`` for the
            # legacy trivial layout would also be fine, but storing the
            # layout unconditionally lets downstream tooling rely on
            # ``result.axis_layout.decoupled`` without a None-check.
            axis_layout=self._axis_layout,
            # Pass-through of rescue-pass diagnostics from
            # ``FinalSweepResult``.  ``None`` for v1 (rescue pass not
            # implemented yet); populated by the engine hook in
            # ``_em.py`` once the rescue commit lands.
            pre_rescue_grad_norms=getattr(
                final, "pre_rescue_grad_norms", None
            ),
            rescued_cell_mask=getattr(final, "rescued_cell_mask", None),
            gene_mean_per_dataset=gene_mean_per_dataset,
        )

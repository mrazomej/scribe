"""LNM / LNMVCP implementation of :class:`LaplaceObservationModel`.

Glues :mod:`scribe.laplace._newton_lnm` (per-cell Newton kernels for
the composition and capture blocks) into the protocol expected by
:func:`scribe.laplace._em.run_laplace_em`. All shared scaffolding —
outer Adam, mini-batching, divergence detection, progress reporting,
final convergence check, early stopping, checkpointing — lives in the
driver; this module owns only the model-specific pieces.

LNM is structurally distinct from PLN/NBLN in three ways:

1. **Two latent-space variants** controlled by ``d_mode``:
     * ``"low_rank"`` — Newton on per-cell ``z ∈ ℝ^k``;
       prior is ``z ~ N(0, I_k)``.
     * ``"learned"`` — Newton on per-cell ``y_alr ∈ ℝ^{G-1}``;
       prior is ``y_alr ~ N(μ, W Wᵀ + diag(d))``.
   The per-cell ``latent_loc`` carried by the driver is whichever of
   the two is in use; ``init_state`` returns the right shape.

2. **Block-diagonal Hessian for LNMVCP**. When ``capture_anchor`` is
   supplied, the per-cell joint Newton over ``(composition_latent,
   η)`` factorises because the multinomial composition block is
   independent of the NB totals block conditional on observed counts.
   The composition Newton runs first, then a scalar Newton on η; both
   share the optimised globals but their gradients are decoupled.

3. **Auxiliary data**. The composition Newton needs counts in ALR
   coordinates (``u_alr``, shape ``(n_cells, G-1)``) and per-cell
   totals (``n_total_per_cell``, shape ``(n_cells,)``). These flow
   through the protocol's ``aux_data`` field, sliced per mini-batch
   by the default :meth:`aux_batch_slice`.

Result packaging: ``x_loc`` is whichever composition latent is in
use (``z`` or ``y_alr``); ``eta_loc`` is the η_capture MAP for
LNMVCP and None for plain LNM; ``globals`` exposes
``μ, W, d_loc, mu_T_loc, r_T_loc``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from sklearn.utils.extmath import randomized_svd

from ..models.config import ModelConfig
from ._axis_layout import build_axis_layout
from ._em import (
    FinalSweepResult,
    InitState,
    LaplaceObservationModel,
    LaplaceRunResult,
)
from ._global_uncertainty import (
    invert_hessian_with_jitter,
    resolve_positive_fns,
)
from ._newton_lnm import (
    laplace_log_det_neg_H_batch_eta,
    laplace_log_det_neg_H_batch_y_alr,
    laplace_log_det_neg_H_batch_z,
    laplace_newton_batch_eta,
    laplace_newton_batch_y_alr,
    laplace_newton_batch_z,
)


# =====================================================================
# Woodbury helpers (LNM uses these for the y_alr MVN prior)
# =====================================================================


def _woodbury_logdet_sigma(W: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """``log det(WWᵀ + diag(d))`` via the matrix-determinant lemma."""
    inv_d = 1.0 / d
    K_dim = W.shape[1]
    K = jnp.eye(K_dim) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
    log_det_d = jnp.sum(jnp.log(d))
    return log_det_d + log_det_K


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, diff: jnp.ndarray
) -> jnp.ndarray:
    """Batched ``(diff)ᵀ Σ⁻¹ (diff)`` for ``Σ = WWᵀ + diag(d)``."""
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


# =====================================================================
# LNM ELBO components
# =====================================================================


def _lnm_composition_elbo(
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    counts_batch: jnp.ndarray,
    z_or_y: jnp.ndarray,
    log_det_neg_H: jnp.ndarray,
    alr_reference_idx: int,
    n_genes: int,
    d_mode: str,
) -> jnp.ndarray:
    """Negative composition-block Laplace ELBO summed over a batch.

    Mirrors ``_lnm_laplace_elbo`` from the legacy engine: builds the
    full-G ALR logits from the active latent (z or y_alr), evaluates
    the multinomial log-prob, adds the prior on the latent (Gaussian
    for ``low_rank`` or low-rank-plus-diagonal MVN for ``learned``),
    and applies the Laplace correction ``-½ log det(-H_c)``.
    """
    if d_mode == "low_rank":
        y_alr = mu[None, :] + z_or_y @ W.T
    else:
        y_alr = z_or_y

    leading = y_alr.shape[:-1]
    full_shape = leading + (n_genes,)
    full = jnp.zeros(full_shape, dtype=y_alr.dtype)
    other_idx = jnp.asarray(
        [g for g in range(n_genes) if g != int(alr_reference_idx)]
    )
    full = full.at[..., other_idx].set(y_alr)
    log_p = jax.nn.log_softmax(full, axis=-1)
    multinomial_lp = jnp.sum(counts_batch * log_p, axis=-1)

    if d_mode == "low_rank":
        prior_lp = -0.5 * jnp.sum(z_or_y * z_or_y, axis=-1)
    else:
        diff = y_alr - mu[None, :]
        quad = _woodbury_quadform(W, d, diff)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        g_minus1 = mu.shape[0]
        prior_lp = (
            -0.5 * quad
            - 0.5 * log_det_sigma
            - 0.5 * g_minus1 * jnp.log(2 * jnp.pi)
        )

    laplace_corr = -0.5 * log_det_neg_H
    elbo_per_cell = multinomial_lp + prior_lp + laplace_corr
    elbo_per_cell = jnp.where(
        jnp.isfinite(elbo_per_cell),
        elbo_per_cell,
        jnp.zeros_like(elbo_per_cell),
    )
    return -jnp.sum(elbo_per_cell)


# =====================================================================
# Observation-model class
# =====================================================================


class LNMObservationModel(LaplaceObservationModel):
    """LNM / LNMVCP observation channel for the generic Laplace-EM driver.

    Parameters
    ----------
    d_mode : {"low_rank", "learned"}
        Selects whether the per-cell composition latent is the
        k-dimensional ``z`` (``low_rank``) or the (G-1)-dimensional
        ``y_alr`` (``learned``).
    alr_reference_idx : int
        Index of the ALR reference gene; required to build the
        full-G logits inside the multinomial likelihood.
    capture_anchor : Optional[Tuple[float, float]]
        ``(log_M_0, sigma_M)`` for the biology-informed prior on
        ``eta_capture``. ``None`` disables capture (plain LNM).
    model_config : ModelConfig, optional
        Model configuration.  Used to resolve
        ``positive_transform`` for the totals parameters ``mu_T``
        and ``r_T``, and (harmonic-hare Commit 6) the
        ``correlate_other_column`` flag for the LNM ALR-reference
        consistency check.
    gene_names : Sequence[str], optional
        Gene names in observation-axis order.  Used as a fallback
        signal for the trailing ``"_other"`` detection in the
        ``correlate_other_column=False`` consistency check.  When
        the caller has constructed the obs model via the standard
        engine path (``apply_gene_coverage_and_alr`` →
        ``run_inference``), this is already populated.  Optional
        for direct-construction users.
    has_pooled_other : bool, optional
        Primary signal that the trailing column is the pooled
        ``"_other"`` aggregate (from the ``gene_coverage`` stage).
        Used together with ``gene_names`` to detect ``"_other"``
        for the consistency check; ``None`` defers to the names
        check.  See ``scribe.laplace._axis_layout.build_axis_layout``.
    """

    def __init__(
        self,
        d_mode: str,
        alr_reference_idx: int,
        capture_anchor: Optional[Tuple[float, float]] = None,
        model_config: Optional[ModelConfig] = None,
        gene_names: Optional[Sequence[str]] = None,
        has_pooled_other: Optional[bool] = None,
    ):
        if d_mode not in ("low_rank", "learned"):
            raise ValueError(
                f"d_mode must be 'low_rank' or 'learned'; got {d_mode!r}."
            )
        self._d_mode = d_mode
        self._alr_reference_idx = int(alr_reference_idx)
        if capture_anchor is None:
            self._capture_anchor = None
            self._sigma_M = 1.0
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

        # Harmonic-hare Commit 6: defensive ALR-reference / decoupling
        # consistency check.  Lives at obs-model construction time so
        # callers that bypass ``apply_gene_coverage_and_alr`` (e.g.
        # direct ``ModelConfig`` construction → engine) still get a
        # loud failure rather than silently mis-laying out Σ.  Under
        # legacy (``correlate_other_column=True``), this is a no-op so
        # the bit-equal contract is preserved (auditor rev-3 #8).
        self._model_config = model_config
        self._gene_names_for_layout = gene_names
        self._has_pooled_other_for_layout = (
            None if has_pooled_other is None else bool(has_pooled_other)
        )

    @property
    def name(self) -> str:
        return "lnmvcp" if self.uses_capture else "lnm"

    @property
    def uses_capture(self) -> bool:
        return self._capture_anchor is not None

    @property
    def d_mode(self) -> str:
        return self._d_mode

    # --- Initial state ---------------------------------------------------

    def init_state(
        self,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        latent_dim: int,
        seed: int,
    ) -> InitState:
        if self._alr_reference_idx < 0 or self._alr_reference_idx >= n_genes:
            raise ValueError(
                "LNM Laplace needs alr_reference_idx in [0, n_genes); "
                f"got {self._alr_reference_idx} for n_genes={n_genes}."
            )

        # Harmonic-hare Commit 6: defensive consistency check between
        # ``model_config.correlate_other_column`` and the ALR reference
        # position when the panel has a trailing pooled ``_other``.
        # The standard engine path (``apply_gene_coverage_and_alr``)
        # already pins this correctly, but a user who builds
        # ``ModelConfig`` directly and skips that stage can land here
        # with an inconsistent combination — fail loudly so they don't
        # silently get a fit where ``_other`` is in Σ even though the
        # flag asked for it to be decoupled.  The fallback default
        # mirrors the user-facing default flipped in Commit 5b
        # (``False``): a missing-attribute config triggers the
        # decoupled validator, which is a no-op when no ``_other``
        # column is present (``build_axis_layout`` returns a trivial
        # layout) and catches real misconfigurations when one is.
        # Explicit ``correlate_other_column=True`` (legacy opt-in)
        # produces a trivial layout regardless of the signals, so the
        # check below short-circuits and preserves the bit-equal
        # contract.
        _correlate_other_column = (
            False
            if self._model_config is None
            else bool(
                getattr(self._model_config, "correlate_other_column", False)
            )
        )
        # Reuse the shared detection priority chain (has_pooled_other
        # → gene_names → False) and contradictory-signal validation.
        # Under legacy, the trivial layout is returned regardless of
        # the signals so this is bit-equal-safe.
        _layout = build_axis_layout(
            n_genes=int(n_genes),
            correlate_other_column=_correlate_other_column,
            gene_names=self._gene_names_for_layout,
            has_pooled_other=self._has_pooled_other_for_layout,
        )
        if _layout.decoupled:
            # Layout says the trailing column is the pooled ``_other``
            # and Σ must exclude it.  For LNM, that is realised by
            # pinning the ALR reference to ``_other``'s position
            # (last index in observation order).  Reject any other
            # position with a message pointing the user at both flags
            # so they can decide which to change.
            _other_pos = int(_layout.other_idx)
            if int(self._alr_reference_idx) != _other_pos:
                raise ValueError(
                    "Inconsistent LNM configuration: "
                    "correlate_other_column=False requests that the "
                    "pooled '_other' aggregate be excluded from the "
                    "LNM latent covariance, but the configured "
                    "alr_reference_idx is "
                    f"{int(self._alr_reference_idx)} (not the "
                    f"'_other' position {_other_pos}).  LNM realises "
                    "this decoupling only when the ALR reference is "
                    "pinned to '_other' (the ALR reference gene is "
                    "excluded from Σ by construction).  Either pass "
                    "alr_reference_idx pointing to '_other' (or drop "
                    "the override entirely — the standard "
                    "``apply_gene_coverage_and_alr`` stage will auto-"
                    "pin it for you), or pass "
                    "correlate_other_column=True to keep today's "
                    "real-gene-reference behaviour.  See "
                    "paper/_nb_lognormal.qmd §sec-nbln-decorrelate-"
                    "other for the biophysical rationale."
                )

        n_total_per_cell = count_data.sum(axis=-1)
        keep_mask = jnp.asarray(
            [g for g in range(n_genes) if g != self._alr_reference_idx]
        )
        u_alr = count_data[:, keep_mask]
        g_minus1 = n_genes - 1

        # Data-driven init of ALR-space mu and W via PCA on log-proportions.
        pseudocount = 1.0
        p_full = (count_data + pseudocount) / (
            n_total_per_cell[:, None] + n_genes * pseudocount
        )
        log_p_full_np = np.asarray(jnp.log(p_full))
        log_p_alr_np = (
            log_p_full_np[:, np.asarray(keep_mask)]
            - log_p_full_np[
                :, self._alr_reference_idx : self._alr_reference_idx + 1
            ]
        )
        mu_init = jnp.asarray(log_p_alr_np.mean(axis=0), dtype=jnp.float32)
        centered = log_p_alr_np - log_p_alr_np.mean(axis=0, keepdims=True)
        _U, S, Vt = randomized_svd(
            centered, n_components=int(latent_dim), random_state=int(seed)
        )
        W_init = jnp.asarray(
            Vt.T * (S / np.sqrt(max(n_cells - 1, 1))), dtype=jnp.float32
        )
        # Initialise unconstrained d_loc so positive_transform(d_loc) ≈ 0.01.
        d_loc_init = self._pos_inverse(
            jnp.full((g_minus1,), 0.01, dtype=jnp.float32)
        )

        params = {"mu": mu_init, "W": W_init, "d_loc": d_loc_init}

        # NB-on-totals globals (always present; LNMVCP feeds eta into them).
        # Initialise unconstrained mu_T_loc and r_T_loc so that
        # positive_transform(loc) ≈ desired positive initial value.
        if self.uses_capture:
            log_M0, _sigma_M = self._capture_anchor
            mu_T_pos_init = jnp.exp(jnp.asarray(log_M0, dtype=jnp.float32))
        else:
            mu_T_pos_init = jnp.maximum(jnp.mean(n_total_per_cell), 1.0)
        r_T_pos_init = jnp.asarray(4.0, dtype=jnp.float32)

        params["mu_T_loc"] = self._pos_inverse(mu_T_pos_init).astype(
            jnp.float32
        )
        params["r_T_loc"] = self._pos_inverse(r_T_pos_init).astype(jnp.float32)

        # Per-cell composition latent (z or y_alr).
        if self._d_mode == "low_rank":
            latent_loc = jnp.zeros((n_cells, latent_dim), dtype=jnp.float32)
        else:
            latent_loc = jnp.broadcast_to(mu_init, (n_cells, g_minus1)).copy()

        # Capture-anchor per-cell state.
        if self.uses_capture:
            log_lib_size = jnp.log(jnp.maximum(n_total_per_cell, 1.0))
            eta_anchor = jnp.asarray(
                self._capture_anchor[0] - log_lib_size, dtype=jnp.float32
            )
            eta_loc = eta_anchor
        else:
            eta_anchor = None
            eta_loc = None

        return InitState(
            params=params,
            latent_loc=latent_loc,
            eta_loc=eta_loc,
            eta_anchor=eta_anchor,
            aux_data={"u_alr": u_alr, "n_total": n_total_per_cell},
        )

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
        u_alr_batch = aux_batch["u_alr"]
        n_total_batch = aux_batch["n_total"]

        mu = params["mu"]
        W = params["W"]
        d = self._pos_forward(params["d_loc"])
        n_genes = W.shape[0] + 1  # G = (G-1) + 1

        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)

        # ----- Composition block (z or y_alr) -----
        if self._d_mode == "low_rank":
            z_new, gn_comp = laplace_newton_batch_z(
                latent_init_sg,
                u_alr_batch,
                n_total_batch,
                mu_sg,
                W_sg,
                self._alr_reference_idx,
                n_genes,
                n_newton,
                damping,
            )
            z_new = jax.lax.stop_gradient(z_new)
            log_det_comp = laplace_log_det_neg_H_batch_z(
                z_new,
                u_alr_batch,
                n_total_batch,
                mu,
                W,
                self._alr_reference_idx,
                n_genes,
            )
            comp_loss = data_scale * _lnm_composition_elbo(
                mu,
                W,
                d,
                counts_batch,
                z_new,
                log_det_comp,
                self._alr_reference_idx,
                n_genes,
                self._d_mode,
            )
            latent_new = z_new
        else:
            y_new, gn_comp = laplace_newton_batch_y_alr(
                latent_init_sg,
                u_alr_batch,
                n_total_batch,
                mu_sg,
                W_sg,
                d_sg,
                self._alr_reference_idx,
                n_genes,
                n_newton,
                damping,
            )
            y_new = jax.lax.stop_gradient(y_new)
            log_det_comp = laplace_log_det_neg_H_batch_y_alr(
                y_new,
                u_alr_batch,
                n_total_batch,
                mu,
                W,
                d,
                self._alr_reference_idx,
                n_genes,
            )
            comp_loss = data_scale * _lnm_composition_elbo(
                mu,
                W,
                d,
                counts_batch,
                y_new,
                log_det_comp,
                self._alr_reference_idx,
                n_genes,
                self._d_mode,
            )
            latent_new = y_new

        # ----- Totals block (NB on observed totals) -----
        # Map unconstrained coordinates to positive totals parameters
        # via the configured positive_transform (softplus by default).
        mu_T = self._pos_forward(params["mu_T_loc"])
        r_T = self._pos_forward(params["r_T_loc"])

        if self.uses_capture:
            mu_T_sg = jax.lax.stop_gradient(mu_T)
            r_T_sg = jax.lax.stop_gradient(r_T)
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)

            eta_new, gn_eta = laplace_newton_batch_eta(
                eta_init_sg,
                n_total_batch,
                r_T_sg,
                mu_T_sg,
                eta_anchor_sg,
                self._sigma_M,
                n_newton,
                damping,
            )
            eta_new = jax.lax.stop_gradient(eta_new)

            exp_neg_eta = jnp.exp(jnp.clip(-eta_new, -30.0, 30.0))
            rate_T = mu_T * exp_neg_eta
            v = r_T + rate_T
            nb_lp = (
                gammaln(n_total_batch + r_T)
                - gammaln(r_T)
                + r_T * jnp.log(r_T / v)
                + n_total_batch * jnp.log(rate_T / v)
            )
            eta_diff = eta_new - eta_anchor_batch
            eta_prior_lp = (
                -0.5 * (eta_diff * eta_diff) / (self._sigma_M * self._sigma_M)
            )
            log_det_eta = laplace_log_det_neg_H_batch_eta(
                eta_new,
                n_total_batch,
                r_T,
                mu_T,
                self._sigma_M,
            )
            totals_loss = data_scale * jnp.sum(
                -(nb_lp + eta_prior_lp) + 0.5 * log_det_eta
            )
        else:
            v = r_T + mu_T
            nb_lp = (
                gammaln(n_total_batch + r_T)
                - gammaln(r_T)
                + r_T * jnp.log(r_T / v)
                + n_total_batch * jnp.log(mu_T / v)
            )
            totals_loss = data_scale * jnp.sum(-nb_lp)
            eta_new = eta_init  # passthrough; not used by plain LNM
            gn_eta = None

        loss = comp_loss + totals_loss
        # Per-block grad-norm split for the progress display.  Plain
        # LNM has only the composition Newton block; LNMVCP adds the
        # scalar capture-anchor η block whose Hessian decouples from
        # the composition block (see paper/_logistic_normal_multinomial.qmd).
        if gn_eta is not None:
            gn_blocks = {"comp": gn_comp, "η": gn_eta}
        else:
            gn_blocks = {"comp": gn_comp}
        return loss, (latent_new, eta_new, gn_blocks)

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
        u_alr = aux_data["u_alr"]
        n_total = aux_data["n_total"]
        n_genes = params["W"].shape[0] + 1

        mu = jax.lax.stop_gradient(params["mu"])
        W = jax.lax.stop_gradient(params["W"])
        d = jax.lax.stop_gradient(self._pos_forward(params["d_loc"]))

        if self._d_mode == "low_rank":
            latent_final, gn_final = laplace_newton_batch_z(
                latent_loc,
                u_alr,
                n_total,
                mu,
                W,
                self._alr_reference_idx,
                n_genes,
                n_newton,
                damping,
            )
        else:
            latent_final, gn_final = laplace_newton_batch_y_alr(
                latent_loc,
                u_alr,
                n_total,
                mu,
                W,
                d,
                self._alr_reference_idx,
                n_genes,
                n_newton,
                damping,
            )

        if self.uses_capture:
            mu_T = self._pos_forward(jax.lax.stop_gradient(params["mu_T_loc"]))
            r_T = self._pos_forward(jax.lax.stop_gradient(params["r_T_loc"]))
            eta_final, gn_eta_final = laplace_newton_batch_eta(
                eta_loc,
                n_total,
                r_T,
                mu_T,
                eta_anchor,
                self._sigma_M,
                n_newton,
                damping,
            )
            gn_final = jnp.maximum(gn_final, gn_eta_final)
        else:
            eta_final = None

        return FinalSweepResult(
            latent_loc=latent_final,
            eta_loc=eta_final,
            final_grad_norms=gn_final,
        )

    # --- Global uncertainty (LNM totals) ---------------------------------

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
        """Compute full 2x2 Laplace covariance for LNM totals parameters.

        Builds the negative NB-on-totals objective as a function of
        the unconstrained ``(mu_T_loc, r_T_loc)`` vector, maps through
        ``positive_transform`` to get constrained ``(mu_T, r_T)``,
        evaluates the full-data NB log-prob on observed cell totals,
        and computes the 2x2 Hessian at the optimised MAP.

        For LNMVCP, includes the profiled correction from the scalar
        per-cell ``eta_c*(theta)`` MAP so the global covariance reflects
        eta uncertainty.

        Parameters
        ----------
        params : dict
            Final global parameters including ``mu_T_loc``, ``r_T_loc``.
        latent_loc : jnp.ndarray
            Per-cell composition latent MAPs (not used directly).
        eta_loc : jnp.ndarray or None
            Per-cell eta MAPs for LNMVCP.
        eta_anchor : jnp.ndarray or None
            Per-cell eta anchors for LNMVCP.
        count_data : jnp.ndarray, shape ``(N, G)``
            Full observed count matrix.
        aux_data : dict
            Must contain ``"n_total"`` (per-cell totals).
        model_config : ModelConfig
            Provides ``positive_transform``.

        Returns
        -------
        dict
            ``totals_loc`` (2,), ``totals_cov`` (2, 2),
            ``totals_scale`` (2,), plus marginal aliases
            ``mu_T_loc``, ``mu_T_scale``, ``r_T_loc``, ``r_T_scale``,
            and diagnostics.
        """
        n_total = aux_data["n_total"]
        pos_fwd = self._pos_forward

        mu_T_loc_val = params["mu_T_loc"]
        r_T_loc_val = params["r_T_loc"]
        theta_hat = jnp.stack([mu_T_loc_val, r_T_loc_val])

        if self.uses_capture and eta_loc is not None:
            # LNMVCP: the totals objective includes the eta latent.
            # We build the profiled objective that accounts for the
            # implicit dependence of eta*(theta) on theta.

            def _totals_neg_obj_lnmvcp(theta):
                """Negative NB-on-totals + eta prior + Laplace correction.

                The per-cell eta MAPs are held fixed (conditional
                Hessian). The profiled correction from eta uncertainty
                is added below via the Schur complement.
                """
                mu_T_ = pos_fwd(theta[0])
                r_T_ = pos_fwd(theta[1])
                exp_neg_eta = jnp.exp(jnp.clip(-eta_loc, -30.0, 30.0))
                rate_T = mu_T_ * exp_neg_eta
                v = r_T_ + rate_T
                nb_lp = (
                    gammaln(n_total + r_T_)
                    - gammaln(r_T_)
                    + r_T_ * jnp.log(r_T_ / v)
                    + n_total * jnp.log(rate_T / v)
                )
                # Truncated-normal prior on eta (half-normal from 0).
                eta_diff = eta_loc - eta_anchor
                eta_prior_lp = -0.5 * (eta_diff**2) / (self._sigma_M**2)
                # Laplace correction: -0.5 log det(-H_eta).
                from ._newton_lnm import _nb_eta_grad_and_hessian

                def _neg_H_eta_scalar(eta_c, u_T_c):
                    _, nb_hess = _nb_eta_grad_and_hessian(
                        eta_c, u_T_c, r_T_, mu_T_
                    )
                    return -(nb_hess - 1.0 / (self._sigma_M**2))

                neg_H_eta = jax.vmap(_neg_H_eta_scalar)(eta_loc, n_total)
                log_det_eta = jnp.log(jnp.maximum(neg_H_eta, 1e-30))
                total_lp = jnp.sum(nb_lp + eta_prior_lp - 0.5 * log_det_eta)
                return -total_lp

            # Conditional 2x2 Hessian at the MAP (eta held fixed).
            H_cond = jax.hessian(_totals_neg_obj_lnmvcp)(theta_hat)

            # Schur complement correction from eta uncertainty.
            # For each cell c, the cross-derivative d^2(-obj)/d(theta)d(eta_c)
            # is a 2-vector, and d^2(-obj)/d(eta_c)^2 is a scalar.
            # The profiled correction is:
            #   sum_c  H_{theta,eta_c} * (1/H_{eta_c,eta_c}) * H_{eta_c,theta}
            from ._newton_lnm import _nb_eta_grad_and_hessian

            def _schur_correction_per_cell(eta_c, u_T_c, eta_anch_c):
                """Per-cell Schur correction for the (theta, eta) block."""
                mu_T_ = pos_fwd(theta_hat[0])
                r_T_ = pos_fwd(theta_hat[1])

                # Cross-derivative d^2(-obj)/d(theta_i)d(eta_c).
                # We differentiate the negative per-cell objective
                # w.r.t. theta, evaluated at fixed eta_c.
                def _neg_obj_cell(theta_):
                    mu_T_v = pos_fwd(theta_[0])
                    r_T_v = pos_fwd(theta_[1])
                    exp_neg_eta = jnp.exp(jnp.clip(-eta_c, -30.0, 30.0))
                    rate_T = mu_T_v * exp_neg_eta
                    v = r_T_v + rate_T
                    nb_lp = (
                        gammaln(u_T_c + r_T_v)
                        - gammaln(r_T_v)
                        + r_T_v * jnp.log(r_T_v / v)
                        + u_T_c * jnp.log(rate_T / v)
                    )
                    eta_diff = eta_c - eta_anch_c
                    eta_prior_lp = -0.5 * (eta_diff**2) / (self._sigma_M**2)
                    _, nb_hess = _nb_eta_grad_and_hessian(
                        eta_c, u_T_c, r_T_v, mu_T_v
                    )
                    neg_H_eta = -(nb_hess - 1.0 / (self._sigma_M**2))
                    log_det_eta = jnp.log(jnp.maximum(neg_H_eta, 1e-30))
                    return -(nb_lp + eta_prior_lp - 0.5 * log_det_eta)

                # H_{theta, eta_c}: shape (2,)
                def _neg_obj_joint(theta_, eta_):
                    mu_T_v = pos_fwd(theta_[0])
                    r_T_v = pos_fwd(theta_[1])
                    exp_neg_eta = jnp.exp(jnp.clip(-eta_, -30.0, 30.0))
                    rate_T = mu_T_v * exp_neg_eta
                    v = r_T_v + rate_T
                    nb_lp = (
                        gammaln(u_T_c + r_T_v)
                        - gammaln(r_T_v)
                        + r_T_v * jnp.log(r_T_v / v)
                        + u_T_c * jnp.log(rate_T / v)
                    )
                    eta_diff = eta_ - eta_anch_c
                    eta_prior_lp = -0.5 * (eta_diff**2) / (self._sigma_M**2)
                    return -(nb_lp + eta_prior_lp)

                # Cross-Hessian: d^2(-obj)/d(theta)d(eta) at the MAP
                H_theta_eta = jax.jacobian(
                    jax.grad(_neg_obj_joint, argnums=1), argnums=0
                )(
                    theta_hat, eta_c
                )  # shape (2,)

                # Diagonal Hessian of eta: d^2(-obj)/d(eta)^2
                H_eta_eta = jax.grad(
                    jax.grad(lambda e: _neg_obj_joint(theta_hat, e))
                )(eta_c)

                # Schur term: H_{theta,eta} @ (1/H_{eta,eta}) @ H_{eta,theta}
                # Both are scalars since eta is 1D per cell.
                return jnp.outer(H_theta_eta, H_theta_eta) / H_eta_eta

            schur_total = jnp.sum(
                jax.vmap(_schur_correction_per_cell)(
                    eta_loc, n_total, eta_anchor
                ),
                axis=0,
            )
            H_profiled = H_cond - schur_total
        else:
            # Plain LNM: no eta latent, no profiled correction needed.
            def _totals_neg_obj_plain(theta):
                mu_T_ = pos_fwd(theta[0])
                r_T_ = pos_fwd(theta[1])
                v = r_T_ + mu_T_
                nb_lp = (
                    gammaln(n_total + r_T_)
                    - gammaln(r_T_)
                    + r_T_ * jnp.log(r_T_ / v)
                    + n_total * jnp.log(mu_T_ / v)
                )
                return -jnp.sum(nb_lp)

            H_profiled = jax.hessian(_totals_neg_obj_plain)(theta_hat)

        # Invert the 2x2 profiled Hessian with jitter if needed.
        totals_cov, jitter_used = invert_hessian_with_jitter(H_profiled)
        totals_scale = jnp.sqrt(jnp.diag(totals_cov))

        return {
            "totals_loc": theta_hat,
            "totals_cov": totals_cov,
            "totals_scale": totals_scale,
            "mu_T_loc": theta_hat[0],
            "mu_T_scale": totals_scale[0],
            "r_T_loc": theta_hat[1],
            "r_T_scale": totals_scale[1],
            "totals_hessian_diag": jnp.diag(H_profiled),
            "totals_hessian_jitter": jnp.asarray(
                jitter_used, dtype=jnp.float32
            ),
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
        return LaplaceRunResult(
            globals=params,
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
        )

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
``μ, W, log d, log μ_T, log r_T``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from sklearn.utils.extmath import randomized_svd

from ..models.config import ModelConfig
from ._em import (
    FinalSweepResult,
    InitState,
    LaplaceObservationModel,
    LaplaceRunResult,
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
        k-dimensional ``z`` (``low_rank``) or the (G−1)-dimensional
        ``y_alr`` (``learned``).
    alr_reference_idx : int
        Index of the ALR reference gene; required to build the
        full-G logits inside the multinomial likelihood.
    capture_anchor : Optional[Tuple[float, float]]
        ``(log_M_0, σ_M)`` for the biology-informed prior on
        ``η_capture``. ``None`` disables capture (plain LNM).
    """

    def __init__(
        self,
        d_mode: str,
        alr_reference_idx: int,
        capture_anchor: Optional[Tuple[float, float]] = None,
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
        d_log_init = jnp.full((g_minus1,), jnp.log(0.01), dtype=jnp.float32)

        params = {"mu": mu_init, "W": W_init, "d_log": d_log_init}

        # NB-on-totals globals (always present; LNMVCP feeds eta into them).
        if self.uses_capture:
            log_M0, _sigma_M = self._capture_anchor
            log_mu_T_init = float(log_M0)
        else:
            log_mu_T_init = float(
                jnp.log(jnp.maximum(jnp.mean(n_total_per_cell), 1.0))
            )
        params["log_mu_T"] = jnp.asarray(log_mu_T_init, dtype=jnp.float32)
        params["log_r_T"] = jnp.asarray(jnp.log(4.0), dtype=jnp.float32)

        # Per-cell composition latent (z or y_alr).
        if self._d_mode == "low_rank":
            latent_loc = jnp.zeros(
                (n_cells, latent_dim), dtype=jnp.float32
            )
        else:
            latent_loc = jnp.broadcast_to(
                mu_init, (n_cells, g_minus1)
            ).copy()

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
        d = jnp.exp(params["d_log"])
        n_genes = (
            W.shape[0] + 1  # G = (G-1) + 1
        )

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
                mu, W, d, counts_batch, z_new, log_det_comp,
                self._alr_reference_idx, n_genes, self._d_mode,
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
                mu, W, d, counts_batch, y_new, log_det_comp,
                self._alr_reference_idx, n_genes, self._d_mode,
            )
            latent_new = y_new

        # ----- Totals block (NB on observed totals) -----
        mu_T = jnp.exp(params["log_mu_T"])
        r_T = jnp.exp(params["log_r_T"])

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
            eta_prior_lp = -0.5 * (eta_diff * eta_diff) / (
                self._sigma_M * self._sigma_M
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
            gn_eta = jnp.zeros_like(gn_comp)

        loss = comp_loss + totals_loss
        gn = jnp.maximum(gn_comp, gn_eta)
        return loss, (latent_new, eta_new, gn)

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
        n_genes = (
            params["W"].shape[0] + 1
        )

        mu = jax.lax.stop_gradient(params["mu"])
        W = jax.lax.stop_gradient(params["W"])
        d = jax.lax.stop_gradient(jnp.exp(params["d_log"]))

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
            mu_T = jnp.exp(jax.lax.stop_gradient(params["log_mu_T"]))
            r_T = jnp.exp(jax.lax.stop_gradient(params["log_r_T"]))
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
        )

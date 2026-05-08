"""NB-LogNormal implementation of :class:`LaplaceObservationModel`.

Glues :mod:`scribe.laplace._newton_nbln` (per-cell Newton kernel) and
:class:`scribe.stats.distributions.LogMeanNegativeBinomial` (NB log-prob
in pure log-space) into the protocol expected by
:func:`scribe.laplace._em.run_laplace_em`. All shared scaffolding —
outer Adam, mini-batching, divergence detection, progress reporting,
final convergence check — lives in the driver; this module owns only
the model-specific pieces.

State init: empirical log-mean for ``μ``, PCA loadings for ``W``, a
small uniform ``log d``, and the moment-method-of-moments estimator
for ``log r``. Per-cell ``x`` warm-starts at ``log(u + 1)`` so the
initial Newton iteration sees a near-MAP iterate.

Loss closure: Newton on ``(x[, η])`` with ``stop_gradient`` on the
iterates → ``log det(-H)`` at live globals → NB log-prob through
:class:`LogMeanNegativeBinomial` → MVN prior on ``x`` via the inner
Woodbury → optional truncated-normal prior on ``η``.

Final sweep: ``2 × n_newton`` iterations on the full cell population
to lock in per-cell convergence diagnostics.

Result packaging: ``x_loc = latent_loc`` (since NBLN's per-cell latent
is already the log-rate ``x``); ``globals`` exposes ``μ, W, log d,
log r``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

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
from ._newton_nbln import (
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_x_only,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
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
    """

    def __init__(
        self,
        capture_anchor: Optional[Tuple[float, float]] = None,
    ):
        if capture_anchor is None:
            self._capture_anchor = None
            self._sigma_M = 1.0  # placeholder; never read when capture is off
        else:
            log_M0, sigma_M = capture_anchor
            self._capture_anchor = (float(log_M0), float(sigma_M))
            self._sigma_M = float(sigma_M)

    # --- Identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "nbln"

    @property
    def uses_capture(self) -> bool:
        return self._capture_anchor is not None

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

        mu_init = jnp.asarray(
            empirical_log_mean_from_counts(counts_np), dtype=jnp.float32
        )
        W_init = jnp.asarray(
            pca_loadings_init(counts_np, latent_dim=latent_dim),
            dtype=jnp.float32,
        )
        d_log_init = jnp.full((n_genes,), jnp.log(0.1), dtype=jnp.float32)
        r_init = empirical_dispersion_from_counts(counts_np)
        log_r_init = jnp.log(jnp.maximum(r_init, 1e-3)).astype(jnp.float32)

        params = {
            "mu": mu_init,
            "W": W_init,
            "d_log": d_log_init,
            "log_r": log_r_init,
        }

        # Warm-start per-cell x at log(u + 1) [+ η_anchor when capture is on]
        # so exp(x − η) ≈ u from step 0.
        if self.uses_capture:
            log_M0, _sigma_M = self._capture_anchor
            log_lib = jnp.log(jnp.maximum(jnp.sum(count_data, axis=-1), 1.0))
            eta_anchor = log_M0 - log_lib
            eta_loc = eta_anchor
            latent_loc = jnp.log(count_data + 1.0) + eta_loc[:, None]
        else:
            eta_anchor = None
            eta_loc = None
            latent_loc = jnp.log(count_data + 1.0)

        return InitState(
            params=params,
            latent_loc=latent_loc,
            eta_loc=eta_loc,
            eta_anchor=eta_anchor,
            aux_data={},
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
        del aux_batch  # NBLN has no auxiliary data

        mu = params["mu"]
        W = params["W"]
        d = jnp.exp(params["d_log"])
        r = jnp.exp(params["log_r"])

        # Stop-gradient on Newton inputs (variational-EM convention).
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        r_sg = jax.lax.stop_gradient(r)

        if self.uses_capture:
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)
            x_new, eta_new, gn, _ = laplace_newton_batch(
                latent_init_sg,
                eta_init_sg,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                r_sg,
                eta_anchor_sg,
                self._sigma_M,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = jax.lax.stop_gradient(eta_new)
            log_det = laplace_log_det_neg_H_batch(
                x_new, eta_new, counts_batch, r, W, d, self._sigma_M
            )
            log_mean = x_new - eta_new[:, None]
        else:
            x_new, gn, _ = laplace_newton_batch_x_only(
                latent_init_sg,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                r_sg,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = eta_init  # placeholder, not used downstream
            log_det = laplace_log_det_neg_H_batch_x_only(
                x_new, None, counts_batch, r, W, d, 1.0
            )
            log_mean = x_new

        # NB log-prob in log-space.
        nb_lp = LogMeanNegativeBinomial(
            log_mean=log_mean, concentration=r[None, :]
        ).log_prob(counts_batch).sum(axis=-1)

        # MVN prior on x.
        diff = x_new - mu[None, :]
        quad = _woodbury_quadform(W, d, diff)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        G = mu.shape[0]
        mvn_lp = -0.5 * quad - 0.5 * log_det_sigma - 0.5 * G * jnp.log(
            2 * jnp.pi
        )

        # TruncN prior on η.
        if self.uses_capture:
            eta_lp = dist.TruncatedNormal(
                eta_anchor_batch, self._sigma_M, low=0.0
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
        loss = -data_scale * jnp.sum(elbo_per_cell)
        return loss, (x_new, eta_new, gn)

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
        del aux_data

        mu = jax.lax.stop_gradient(params["mu"])
        W = jax.lax.stop_gradient(params["W"])
        d = jax.lax.stop_gradient(jnp.exp(params["d_log"]))
        r = jax.lax.stop_gradient(jnp.exp(params["log_r"]))

        if self.uses_capture:
            x_final, eta_final, gn_final, _ = laplace_newton_batch(
                latent_loc,
                eta_loc,
                count_data,
                mu,
                W,
                d,
                r,
                eta_anchor,
                self._sigma_M,
                n_newton,
                damping,
            )
        else:
            x_final, gn_final, _ = laplace_newton_batch_x_only(
                latent_loc,
                count_data,
                mu,
                W,
                d,
                r,
                n_newton,
                damping,
            )
            eta_final = None

        return FinalSweepResult(
            latent_loc=x_final,
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

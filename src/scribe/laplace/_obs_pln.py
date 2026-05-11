"""Poisson-LogNormal implementation of :class:`LaplaceObservationModel`.

Glues :mod:`scribe.laplace._newton_pln` (per-cell Newton kernel) into
the protocol expected by :func:`scribe.laplace._em.run_laplace_em`.
All shared scaffolding — outer Adam, mini-batching, divergence
detection, progress reporting, final convergence check — lives in the
driver; this module owns only the model-specific pieces.

State init: empirical log-mean for ``μ``, PCA loadings for ``W``, a
small uniform unconstrained ``d_loc``. Per-cell ``x`` warm-starts at ``log(u + 1)``
(+ ``η_anchor`` when capture is on) so the initial Newton iteration
sees a near-MAP iterate.

Loss closure: PLN Newton on ``(x[, η])`` with ``stop_gradient`` on the
iterates → ``log det(-H)`` at live globals → per-gene Poisson log-prob
on ``exp(x − η)`` → MVN prior on ``x`` via the inner Woodbury →
optional truncated-normal prior on ``η``.

Final sweep: ``2 × n_newton`` iterations on the full cell population
to lock in per-cell convergence diagnostics.

Result packaging: ``x_loc = latent_loc``; ``globals`` exposes
``μ, W, d_loc``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from ..core.pln_data_init import (
    empirical_log_mean_from_counts,
    pca_loadings_init,
)
from ..models.config import ModelConfig
from ._em import (
    FinalSweepResult,
    InitState,
    LaplaceObservationModel,
    LaplaceRunResult,
)
from ._global_uncertainty import resolve_positive_fns
from ._newton_pln import (
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_x_only,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
    pln_grad_split_batch,
    pln_grad_x_only_norm_batch,
)


# Same float32 clamp as the PLN likelihood and Newton kernel.
_LOG_RATE_MIN = -30.0
_LOG_RATE_MAX = 30.0


# =====================================================================
# Woodbury helpers for the MVN prior on x
# =====================================================================


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, diff: jnp.ndarray
) -> jnp.ndarray:
    """``(diff)ᵀ Σ⁻¹ (diff)`` for ``Σ = W Wᵀ + diag(d)``, batched over rows of diff."""
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


class PLNObservationModel(LaplaceObservationModel):
    """Poisson-LogNormal observation channel for the generic Laplace-EM driver.

    Parameters
    ----------
    capture_anchor : Optional[Tuple[float, float]]
        ``(log_M_0, σ_M)`` for the biology-informed truncated-normal
        prior on ``η_capture``. ``None`` disables capture (Newton runs
        on ``x`` only).
    model_config : ModelConfig, optional
        Model configuration.  Used to resolve
        ``positive_transform`` for the diagonal variance ``d``.
        Falls back to ``softplus`` when ``None``.
    """

    def __init__(
        self,
        capture_anchor: Optional[Tuple[float, float]] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        if capture_anchor is None:
            self._capture_anchor = None
            self._sigma_M = 1.0
        else:
            log_M0, sigma_M = capture_anchor
            self._capture_anchor = (float(log_M0), float(sigma_M))
            self._sigma_M = float(sigma_M)

        # Resolve the configured positivity map once so loss_fn,
        # final_sweep, and init_state all share the same coordinate.
        self._pos_forward, self._pos_inverse = resolve_positive_fns(
            model_config
        )

    @property
    def name(self) -> str:
        return "pln"

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
        # Initialise unconstrained d_loc so positive_transform(d_loc) ≈ 0.1.
        d_loc_init = self._pos_inverse(
            jnp.full((n_genes,), 0.1, dtype=jnp.float32)
        )
        params = {"mu": mu_init, "W": W_init, "d_loc": d_loc_init}

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
        del aux_batch  # PLN has no auxiliary data

        mu = params["mu"]
        W = params["W"]
        d = self._pos_forward(params["d_loc"])

        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)

        if self.uses_capture:
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)
            x_new, eta_new, _gn, _ = laplace_newton_batch(
                latent_init_sg,
                eta_init_sg,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                eta_anchor_sg,
                self._sigma_M,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = jax.lax.stop_gradient(eta_new)
            log_det = laplace_log_det_neg_H_batch(
                x_new, eta_new, W, d, self._sigma_M
            )
            log_rate = x_new - eta_new[:, None]
            # Per-block grad split for the progress display.  Same
            # rationale as the NBLN adapter: the Newton kernel returns
            # a joint ``L∞`` over ``(x, η)`` for divergence detection,
            # but the per-block split is what users want for diagnosing
            # which Newton block is the convergence bottleneck.
            gn_x, gn_eta = pln_grad_split_batch(
                x_new,
                eta_new,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                eta_anchor_sg,
                self._sigma_M,
            )
            gn_blocks = {"x": gn_x, "η": gn_eta}
        else:
            x_new, _gn, _ = laplace_newton_batch_x_only(
                latent_init_sg,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = eta_init  # placeholder
            log_det = laplace_log_det_neg_H_batch_x_only(
                x_new, None, W, d, self._sigma_M
            )
            log_rate = x_new
            gn_x = pln_grad_x_only_norm_batch(
                x_new, counts_batch, mu_sg, W_sg, d_sg
            )
            gn_blocks = {"x": gn_x}

        # Poisson log-prob (drops constant ``lgamma(u + 1)``):
        #   log p(u | x, η) = Σ_g [ u_g (x_g − η) − exp(x_g − η) ].
        rate = jnp.exp(jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX))
        if self.uses_capture:
            poisson_lp = jnp.sum(
                counts_batch * (x_new - eta_new[:, None]), axis=-1
            ) - jnp.sum(rate, axis=-1)
        else:
            poisson_lp = jnp.sum(counts_batch * x_new, axis=-1) - jnp.sum(
                rate, axis=-1
            )

        # MVN prior on x.
        diff = x_new - mu[None, :]
        quad = _woodbury_quadform(W, d, diff)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        G = mu.shape[0]
        mvn_lp = -0.5 * quad - 0.5 * log_det_sigma - 0.5 * G * jnp.log(
            2 * jnp.pi
        )

        if self.uses_capture:
            eta_lp = dist.TruncatedNormal(
                eta_anchor_batch, self._sigma_M, low=0.0
            ).log_prob(eta_new)
        else:
            eta_lp = jnp.zeros_like(poisson_lp)

        laplace_corr = -0.5 * log_det
        elbo_per_cell = poisson_lp + mvn_lp + eta_lp + laplace_corr
        elbo_per_cell = jnp.where(
            jnp.isfinite(elbo_per_cell),
            elbo_per_cell,
            jnp.zeros_like(elbo_per_cell),
        )
        loss = -data_scale * jnp.sum(elbo_per_cell)
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
        del aux_data
        mu = jax.lax.stop_gradient(params["mu"])
        W = jax.lax.stop_gradient(params["W"])
        d = jax.lax.stop_gradient(self._pos_forward(params["d_loc"]))

        if self.uses_capture:
            x_final, eta_final, gn_final, _ = laplace_newton_batch(
                latent_loc,
                eta_loc,
                count_data,
                mu,
                W,
                d,
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

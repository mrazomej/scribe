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
from ._global_uncertainty import (
    curvature_to_scale,
    resolve_positive_fns,
    woodbury_inv_diag,
)
from ._newton_nbln import (
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_x_only,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
    nbln_grad_split_batch,
    nbln_grad_x_only_norm_batch,
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
    model_config : ModelConfig, optional
        Model configuration.  Used to resolve
        ``positive_transform`` for the dispersion parameter ``r``.
        Falls back to ``softplus`` when ``None``.
    """

    def __init__(
        self,
        capture_anchor: Optional[Tuple[float, float]] = None,
        model_config: Optional[ModelConfig] = None,
    ):
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
        # Initialise unconstrained d_loc so positive_transform(d_loc) ≈ 0.1.
        d_loc_init = self._pos_inverse(
            jnp.full((n_genes,), 0.1, dtype=jnp.float32)
        )
        # Initialise the unconstrained dispersion coordinate r_loc so
        # that positive_transform(r_loc) ≈ empirical dispersion.
        r_init = jnp.asarray(
            empirical_dispersion_from_counts(counts_np), dtype=jnp.float32
        )
        r_loc_init = self._pos_inverse(jnp.maximum(r_init, 1e-3)).astype(
            jnp.float32
        )

        params = {
            "mu": mu_init,
            "W": W_init,
            "d_loc": d_loc_init,
            "r_loc": r_loc_init,
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
        d = self._pos_forward(params["d_loc"])
        # Map unconstrained r_loc to positive dispersion via the
        # configured positive_transform (softplus by default).
        r = self._pos_forward(params["r_loc"])

        # Stop-gradient on Newton inputs (variational-EM convention).
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        r_sg = jax.lax.stop_gradient(r)

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
            # Per-block grad split for the progress display.  The
            # Newton kernel returns a joint ``L∞`` over ``(x, η)`` --
            # useful for divergence detection but unhelpful for
            # diagnosing which block is stalling.  Re-evaluate ∇f at
            # the post-Newton MAP and split into x- and η-block
            # norms.  Cost: one extra Woodbury solve per cell.
            gn_x, gn_eta = nbln_grad_split_batch(
                x_new,
                eta_new,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                r_sg,
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
            gn_x = nbln_grad_x_only_norm_batch(
                x_new, counts_batch, mu_sg, W_sg, d_sg, r_sg
            )
            gn_blocks = {"x": gn_x}

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

        # MVN prior on x.
        diff = x_new - mu[None, :]
        quad = _woodbury_quadform(W, d, diff)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        G = mu.shape[0]
        mvn_lp = (
            -0.5 * quad - 0.5 * log_det_sigma - 0.5 * G * jnp.log(2 * jnp.pi)
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
        r = jax.lax.stop_gradient(self._pos_forward(params["r_loc"]))

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

    # --- Global uncertainty (NBLN r_g) -----------------------------------

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
        """Compute diagonal Laplace posterior for unconstrained r_g.

        Uses the profiled/marginal observed-information Hessian that
        accounts for per-cell latent uncertainty via the Schur
        complement correction.  The exploitable structure is:

        - ``H_{rr}`` is diagonal across genes (NB factorises per gene).
        - ``H_{rx}`` is diagonal across genes (``r_g`` couples only
          to ``x_{c,g}``).
        - ``diag((-H_{xx,c})^{-1})`` is obtained from Woodbury factors.

        For each gene g the profiled curvature is::

            H_profile_g = sum_c [ H_{r_g,r_g}^(c)
                - (H_{r_g,x_{c,g}}^(c))^2 * [(-H_{xx,c})^{-1}]_{gg} ]

        For capture-anchor NBLN, the joint (x, eta) block inverse is
        used, adding the ``(x_g, eta)`` cross terms.

        Parameters
        ----------
        params : dict
            Final global parameters including ``r_loc``.
        latent_loc : jnp.ndarray, shape ``(N, G)``
            Per-cell log-rate MAPs from the final sweep.
        eta_loc : jnp.ndarray or None, shape ``(N,)``
            Per-cell capture offset MAPs.
        eta_anchor : jnp.ndarray or None, shape ``(N,)``
            Per-cell capture anchors.
        count_data : jnp.ndarray, shape ``(N, G)``
            Observed counts.
        aux_data : dict
            Unused for NBLN.
        model_config : ModelConfig
            Provides ``positive_transform``.

        Returns
        -------
        dict
            ``r_loc`` (G,), ``r_scale`` (G,), plus diagnostics.
        """
        del aux_data

        mu = params["mu"]
        W = params["W"]
        d = self._pos_forward(params["d_loc"])
        r_loc = params["r_loc"]
        pos_fwd = self._pos_forward
        r = pos_fwd(r_loc)

        N, G = count_data.shape

        # Compute effective log-rate at each cell.
        if self.uses_capture and eta_loc is not None:
            log_rate_all = latent_loc - eta_loc[:, None]
        else:
            log_rate_all = latent_loc

        # --- Per-cell, per-gene second derivatives of the NB loss ---
        # We need d^2(-loss)/d(r_loc_g)^2 and d^2(-loss)/d(r_loc_g)d(x_g)
        # for a single cell's NB contribution to gene g.
        #
        # The NB log-prob for gene g in cell c is:
        #   f(r_loc_g, x_g) = log NB(u_g | exp(x_g - eta), pos_fwd(r_loc_g))
        # The negative objective's Hessian entries are the negatives of
        # the log-prob's second derivatives.

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

        # H_{r_loc_g, r_loc_g}: second derivative of (-NB log-prob)
        # w.r.t. r_loc_g.
        def _neg_nb_rr(r_loc_g, log_rate_g, u_g):
            return -jax.grad(jax.grad(_nb_lp_gene, argnums=0), argnums=0)(
                r_loc_g, log_rate_g, u_g
            )

        # H_{r_loc_g, x_g}: cross-derivative of (-NB log-prob).
        def _neg_nb_rx(r_loc_g, log_rate_g, u_g):
            return -jax.grad(jax.grad(_nb_lp_gene, argnums=0), argnums=1)(
                r_loc_g, log_rate_g, u_g
            )

        # Vectorise over genes for a single cell.
        _neg_nb_rr_genes = jax.vmap(_neg_nb_rr, in_axes=(0, 0, 0))
        _neg_nb_rx_genes = jax.vmap(_neg_nb_rx, in_axes=(0, 0, 0))

        # Vectorise over cells (r_loc is shared, log_rate and u vary).
        # Chunk to avoid GPU OOM from large intermediate AD tensors.
        _neg_nb_rr_batch = jax.vmap(_neg_nb_rr_genes, in_axes=(None, 0, 0))
        _neg_nb_rx_batch = jax.vmap(_neg_nb_rx_genes, in_axes=(None, 0, 0))

        _chunk_size_hess = min(512, count_data.shape[0])
        _lr_chunks = [
            log_rate_all[i : i + _chunk_size_hess]
            for i in range(0, count_data.shape[0], _chunk_size_hess)
        ]
        _u_chunks = [
            count_data[i : i + _chunk_size_hess]
            for i in range(0, count_data.shape[0], _chunk_size_hess)
        ]
        H_rr_all = jnp.concatenate(
            [
                _neg_nb_rr_batch(r_loc, lr, u)
                for lr, u in zip(_lr_chunks, _u_chunks)
            ],
            axis=0,
        )
        H_rx_all = jnp.concatenate(
            [
                _neg_nb_rx_batch(r_loc, lr, u)
                for lr, u in zip(_lr_chunks, _u_chunks)
            ],
            axis=0,
        )  # Shape: (N, G) for both.

        # --- Per-cell inverse-Hessian diagonal for the x-block ---
        # The x-block Hessian for cell c is:
        #   -H_{xx,c} = diag(a_c) + Sigma^{-1}
        # where a_c = (u_c + r) * p_c * (1 - p_c) is the NB data Hessian
        # diagonal. We need diag((-H_{xx,c})^{-1}).
        from ._newton_nbln import _nb_factors

        def _inv_hess_diag_per_cell(log_rate_c, u_c):
            """Diagonal of (-H_{xx})^{-1} for one cell using Woodbury."""
            nb = _nb_factors(log_rate_c, u_c, r)
            a_c = nb["a"]
            # (-H_{xx}) = diag(a_c) + Sigma^{-1}
            # = diag(a_c + 1/d) - (1/d) W (I + W^T (1/d) W)^{-1} W^T (1/d)
            # ... actually this is the Woodbury for (diag(a+1/d) + lowrank)
            # The inverse diagonal is obtained from woodbury_inv_diag
            # with d_eff = 1 / (a_c + 1/d) as the "diagonal" and W as loadings.
            d_eff = 1.0 / (a_c + 1.0 / d)
            return woodbury_inv_diag(W, d_eff)

        # Process cells in chunks to avoid GPU OOM from materializing
        # all (N, G, k) intermediates simultaneously during vmap.
        _chunk_size = min(256, log_rate_all.shape[0])
        _n_cells = log_rate_all.shape[0]
        _chunks_lr = [
            log_rate_all[i : i + _chunk_size]
            for i in range(0, _n_cells, _chunk_size)
        ]
        _chunks_u = [
            count_data[i : i + _chunk_size]
            for i in range(0, _n_cells, _chunk_size)
        ]
        _vmap_fn = jax.vmap(_inv_hess_diag_per_cell, in_axes=(0, 0))
        inv_H_xx_diag_all = jnp.concatenate(
            [_vmap_fn(lr, u) for lr, u in zip(_chunks_lr, _chunks_u)],
            axis=0,
        )  # shape (N, G)

        # --- Profiled Hessian diagonal ---
        # diag(H_profile)_g = sum_c H_{r_g,r_g}^(c)
        #   - sum_c (H_{r_g,x_g}^(c))^2 * [(-H_{xx,c})^{-1}]_{gg}
        H_rr_summed = jnp.sum(H_rr_all, axis=0)  # (G,)
        schur_correction = jnp.sum(
            H_rx_all**2 * inv_H_xx_diag_all, axis=0
        )  # (G,)
        H_profiled_diag = H_rr_summed - schur_correction

        # Convert profiled curvature to posterior scales.
        r_scale, diagnostics = curvature_to_scale(H_profiled_diag)

        return {
            "r_loc": r_loc,
            "r_scale": r_scale,
            "r_hessian_diag": H_profiled_diag,
            "r_hessian_min": diagnostics["hessian_min"],
            "r_hessian_floor_count": diagnostics["floor_count"],
            "r_curvature_floor": diagnostics["curvature_floor"],
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

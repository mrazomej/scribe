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
    """

    def __init__(
        self,
        capture_anchor: Optional[Tuple[float, float]] = None,
        model_config: Optional[ModelConfig] = None,
        informative_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
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

        # Informative priors (Round-2/3 audit: split per-parameter
        # storage; validate keys ⊆ {"r", "mu", "eta"}).
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

    # --- Identity --------------------------------------------------------

    @property
    def name(self) -> str:
        return "nbln"

    @property
    def uses_capture(self) -> bool:
        # Round-2 Finding A: per-cell eta prior must also activate
        # capture, otherwise the loss path silently takes the x-only
        # branch and ignores the SVI-supplied per-cell anchors.
        return self._capture_anchor is not None or self._prior_eta is not None

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

        # --- mu init: prior loc overrides data-driven init -----------
        if self._prior_mu is not None:
            mu_init = jnp.asarray(self._prior_mu["loc"], dtype=jnp.float32)
        else:
            mu_init = jnp.asarray(
                empirical_log_mean_from_counts(counts_np), dtype=jnp.float32
            )

        W_init = jnp.asarray(
            pca_loadings_init(counts_np, latent_dim=latent_dim),
            dtype=jnp.float32,
        )
        # Initialise unconstrained d_loc so positive_transform(d_loc) ≈ 0.1.
        # No informative-prior override: NBVCP has no `d` counterpart and
        # the prior would be ill-posed.
        d_loc_init = self._pos_inverse(
            jnp.full((n_genes,), 0.1, dtype=jnp.float32)
        )

        # --- r init: prior loc overrides data-driven init ------------
        if self._prior_r is not None:
            r_loc_init = jnp.asarray(
                self._prior_r["loc"], dtype=jnp.float32
            )
        else:
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

        # --- Three-way capture branch (Round-3 Finding A) ------------
        # Order matters: SVI-derived per-cell prior (if present)
        # supersedes the scalar anchor; the bare ``capture_anchor``
        # branch unpacks the tuple, which crashes when capture_anchor
        # is None but prior_eta activates capture.
        if self._prior_eta is not None:
            # SVI-derived per-cell capture (mode "eta").
            eta_anchor = jnp.asarray(
                self._prior_eta["loc"], dtype=jnp.float32
            )
            eta_scale_per_cell = jnp.asarray(
                self._prior_eta["scale"], dtype=jnp.float32
            )
            eta_loc = eta_anchor
            latent_loc = jnp.log(count_data + 1.0) + eta_loc[:, None]
        elif self._capture_anchor is not None:
            # Scalar biology-informed capture anchor (legacy path).
            log_M0, _sigma_M = self._capture_anchor
            log_lib = jnp.log(jnp.maximum(jnp.sum(count_data, axis=-1), 1.0))
            eta_anchor = log_M0 - log_lib
            eta_scale_per_cell = jnp.full(
                (n_cells,), self._sigma_M, dtype=jnp.float32
            )
            eta_loc = eta_anchor
            latent_loc = jnp.log(count_data + 1.0) + eta_loc[:, None]
        else:
            # No capture at all (x-only Newton path).
            eta_anchor = None
            eta_scale_per_cell = None
            eta_loc = None
            latent_loc = jnp.log(count_data + 1.0)

        # Per-cell eta scale rides through ``aux_data`` so the existing
        # ``aux_batch_slice`` auto-slices it by minibatch index — no
        # InitState schema change required (Round-2 Finding B).
        aux_data: Dict[str, jnp.ndarray] = {}
        if eta_scale_per_cell is not None:
            aux_data["eta_scale"] = eta_scale_per_cell

        return InitState(
            params=params,
            latent_loc=latent_loc,
            eta_loc=eta_loc,
            eta_anchor=eta_anchor,
            aux_data=aux_data,
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
        mu = params["mu"]
        W = params["W"]
        d = self._pos_forward(params["d_loc"])
        # Map unconstrained r_loc to positive dispersion via the
        # configured positive_transform (softplus by default).
        r = self._pos_forward(params["r_loc"])

        # Per-cell eta prior scale (Round-1 Finding 4 + Round-2 Finding B).
        # When capture is on, ``aux_batch["eta_scale"]`` provides one
        # scale per cell — either the SVI-derived per-cell prior scale
        # or a broadcast of the scalar ``self._sigma_M`` for the
        # capture-anchor path.  When capture is off, we synthesise a
        # ones array so the Newton kernel's vmap stays uniform.
        if self.uses_capture:
            sigma_eta_batch = aux_batch["eta_scale"]
        else:
            sigma_eta_batch = jnp.ones((counts_batch.shape[0],))

        # Stop-gradient on Newton inputs (variational-EM convention).
        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        r_sg = jax.lax.stop_gradient(r)

        if self.uses_capture:
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)
            sigma_eta_sg = jax.lax.stop_gradient(sigma_eta_batch)
            x_new, eta_new, _gn, _ = laplace_newton_batch(
                latent_init_sg,
                eta_init_sg,
                counts_batch,
                mu_sg,
                W_sg,
                d_sg,
                r_sg,
                eta_anchor_sg,
                sigma_eta_sg,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = jax.lax.stop_gradient(eta_new)
            log_det = laplace_log_det_neg_H_batch(
                x_new, eta_new, counts_batch, r, W, d, sigma_eta_batch
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
                sigma_eta_sg,
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

        # TruncN prior on η (per-cell scale).
        if self.uses_capture:
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
        # The Normal log-prob terms add prior precision to ``r_loc`` and
        # ``mu``; autodiff carries this through to the training gradient
        # automatically.  ``compute_global_uncertainty`` mirrors the
        # ``1/scale**2`` injection in its closed-form Hessian path.
        global_prior_lp = jnp.zeros(())
        if self._prior_r is not None:
            r_loc_prior = self._prior_r
            global_prior_lp = global_prior_lp + jnp.sum(
                dist.Normal(r_loc_prior["loc"], r_loc_prior["scale"]).log_prob(
                    params["r_loc"]
                )
            )
        if self._prior_mu is not None:
            mu_prior = self._prior_mu
            global_prior_lp = global_prior_lp + jnp.sum(
                dist.Normal(mu_prior["loc"], mu_prior["scale"]).log_prob(
                    params["mu"]
                )
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
        mu = jax.lax.stop_gradient(params["mu"])
        W = jax.lax.stop_gradient(params["W"])
        d = jax.lax.stop_gradient(self._pos_forward(params["d_loc"]))
        r = jax.lax.stop_gradient(self._pos_forward(params["r_loc"]))

        if self.uses_capture:
            sigma_eta_full = aux_data["eta_scale"]
            x_final, eta_final, gn_final, _ = laplace_newton_batch(
                latent_loc,
                eta_loc,
                count_data,
                mu,
                W,
                d,
                r,
                eta_anchor,
                sigma_eta_full,
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

        capture_on = self.uses_capture and eta_loc is not None
        if capture_on:
            # Resolve per-cell sigma_eta — prefer aux_data["eta_scale"]
            # (the per-cell array used by the loss); fall back to scalar
            # ``self._sigma_M`` broadcast for backward compatibility.
            sigma_eta_full = jnp.full((N,), self._sigma_M, dtype=jnp.float32)
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
            Y = jax.scipy.linalg.solve_triangular(
                L_S, V_scaled.T, lower=True
            )
            correction = jnp.sum(Y * Y, axis=0)  # (G,)
            return m_inv + correction

        def _per_cell_inverse_blocks(
            log_rate_c: jnp.ndarray,
            u_c: jnp.ndarray,
            sigma_eta_c: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Return (joint_inv_xx_diag, joint_inv_xη, joint_inv_ηη)."""
            nb = _nb_factors(log_rate_c, u_c, r)
            a_c = nb["a"]
            factors = _woodbury_factors(W, d, a_c, damping=0.0)
            A_inv_diag = _A_inv_diag_from_factors(factors)
            if capture_on:
                A_inv_a = _solve_A(factors, a_c)  # (G,)
                s_eta = jnp.maximum(
                    jnp.sum(a_c) + 1.0 / (sigma_eta_c ** 2)
                    - jnp.dot(a_c, A_inv_a),
                    1e-30,
                )
                joint_inv_xx_diag = A_inv_diag + (A_inv_a ** 2) / s_eta
                joint_inv_xeta = A_inv_a / s_eta
                joint_inv_etaeta = 1.0 / s_eta
            else:
                joint_inv_xx_diag = A_inv_diag
                joint_inv_xeta = jnp.zeros_like(a_c)
                joint_inv_etaeta = jnp.asarray(0.0, dtype=a_c.dtype)
            return joint_inv_xx_diag, joint_inv_xeta, joint_inv_etaeta

        _per_cell_vmap = jax.vmap(
            _per_cell_inverse_blocks, in_axes=(0, 0, 0)
        )

        # Chunked pass over cells to keep GPU memory bounded.
        _chunk_size = min(256, N)
        _chunks_lr = [
            log_rate_all[i : i + _chunk_size]
            for i in range(0, N, _chunk_size)
        ]
        _chunks_u = [
            count_data[i : i + _chunk_size]
            for i in range(0, N, _chunk_size)
        ]
        _chunks_sigma = [
            sigma_eta_full[i : i + _chunk_size]
            for i in range(0, N, _chunk_size)
        ]
        _joint_xx_chunks = []
        _joint_xeta_chunks = []
        _joint_etaeta_chunks = []
        for lr, u, se in zip(_chunks_lr, _chunks_u, _chunks_sigma):
            jx, jxe, jee = _per_cell_vmap(lr, u, se)
            _joint_xx_chunks.append(jx)
            _joint_xeta_chunks.append(jxe)
            _joint_etaeta_chunks.append(jee)
        joint_inv_xx_diag_all = jnp.concatenate(_joint_xx_chunks, axis=0)  # (N, G)
        joint_inv_xeta_all = jnp.concatenate(_joint_xeta_chunks, axis=0)  # (N, G)
        joint_inv_etaeta_all = jnp.concatenate(
            _joint_etaeta_chunks, axis=0
        )  # (N,)

        # --- r profiled Hessian diagonal (Round-3 Finding B) -----------
        # capture-off:  Schur_g = sum_c H_rx_cg² · joint_inv_xx_diag_cg
        # capture-on:   Schur_g = sum_c H_rx_cg² · B_cg
        #               where B_cg = joint_inv_xx_diag_cg
        #                            - 2 · joint_inv_xη_cg
        #                            + joint_inv_ηη_c
        # The bracket comes from H_{r,η}_cg = -H_{r,x}_cg (chain rule on
        # log_rate = x - η).
        H_rr_summed = jnp.sum(H_rr_all, axis=0)  # (G,)
        if capture_on:
            bracket = (
                joint_inv_xx_diag_all
                - 2.0 * joint_inv_xeta_all
                + joint_inv_etaeta_all[:, None]
            )  # (N, G)
        else:
            bracket = joint_inv_xx_diag_all
        r_schur = jnp.sum(H_rx_all ** 2 * bracket, axis=0)  # (G,)
        H_r_profiled_diag = H_rr_summed - r_schur

        # Prior precision injection on r_loc (Round-1 Finding 2).
        if self._prior_r is not None:
            H_r_profiled_diag = (
                H_r_profiled_diag + 1.0 / (self._prior_r["scale"] ** 2)
            )

        r_scale, r_diag = curvature_to_scale(H_r_profiled_diag)

        # --- mu profiled Hessian diagonal (diagonal-Σ approximation) ---
        # Cross-Hessian -H_{μ_g, x_{c,g}} = (Σ^{-1})_{g,g} in the
        # diagonal approximation.  μ does not couple to η_c (NB likelihood
        # depends on x, η only; prior on x couples to μ).  So per cell:
        #   H_μμ_profile_cg ≈ (Σ^{-1})_{gg} − (Σ^{-1})_{gg}² · joint_inv_xx_diag_cg
        # Summed over cells:
        #   H_μμ_profile_g = Σ^{-1}_{gg} · [ N − Σ^{-1}_{gg} · sum_c joint_inv_xx_diag_cg ]
        sigma_inv_diag = woodbury_inv_diag(W, d)  # (G,) — exact
        sum_joint_xx_diag = jnp.sum(joint_inv_xx_diag_all, axis=0)  # (G,)
        H_mu_profiled_diag = sigma_inv_diag * (
            float(N) - sigma_inv_diag * sum_joint_xx_diag
        )

        # Prior precision injection on mu.
        if self._prior_mu is not None:
            H_mu_profiled_diag = (
                H_mu_profiled_diag + 1.0 / (self._prior_mu["scale"] ** 2)
            )

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

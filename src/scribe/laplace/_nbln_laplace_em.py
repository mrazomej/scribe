"""Self-contained Laplace-EM orchestration for the NB-LogNormal model.

This module provides a minimal Laplace-EM driver for ``model="nbln"``
that exercises :mod:`scribe.laplace._newton_nbln` end-to-end without
going through the production engine.  It is intentionally narrower
than :mod:`scribe.laplace.engine`: no checkpointing, no early stopping,
no progress bars, no gene subsetting — just outer Adam on globals
``(μ, W, d, r, [η_anchor params])`` with inner Newton on per-cell ``x``
(and ``η`` when capture is on).

The driver is the recommended way to validate the NB-LogNormal math on
real data: it is direct, fast, and bypasses the encoder VAE entirely,
so any model-spec bug is exposed without amortisation gap masking it.

Mathematical objective
----------------------
Per-cell Laplace marginal:

    log p(u_c) ≈ log p(u_c, x_c*, η_c*)
                + ½(G + 1) log(2π)
                − ½ log det(−H_c).

Summed over cells, dropping constants in the globals, the loss is

    L(θ) = −Σ_c [ log p(u_c | x_c*, η_c*; θ)
                  + log p(x_c*; θ)
                  + log p(η_c*; θ)
                  − ½ log det(−H_c) ],

with ``θ = {μ, W, d, r}`` (and ``η_anchor, σ_M`` if capture is on).
The inner ``(x_c*, η_c*)`` come from the Newton kernel under
``stop_gradient``; gradients flow into ``θ`` through the explicit
appearance of ``θ`` in the joint log-density and the log-determinant
correction (envelope theorem).

Cross-references
----------------
- ``paper/_nb_lognormal.qmd`` §8 (full Laplace-EM derivation).
- :mod:`scribe.laplace._newton_nbln` for the inner Newton kernel.
- :class:`scribe.stats.distributions.LogMeanNegativeBinomial` for the
  NB log-prob computed entirely in log-space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax

from ..core.nbln_data_init import (
    empirical_dispersion_from_counts,
    empirical_log_mean_from_counts,
    pca_loadings_init,
)
from ..stats.distributions import LogMeanNegativeBinomial
from ._newton_nbln import (
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_x_only,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
)


# =====================================================================
# Result container
# =====================================================================


@dataclass
class NBLNLaplaceResult:
    """End-of-fit state from :func:`nbln_laplace_em`."""

    mu: jnp.ndarray  # shape (G,)
    W: jnp.ndarray  # shape (G, k)
    d: jnp.ndarray  # shape (G,)
    r: jnp.ndarray  # shape (G,)
    x_loc: jnp.ndarray  # per-cell MAP, shape (N, G)
    eta_loc: Optional[jnp.ndarray]  # per-cell capture, shape (N,) or None
    losses: np.ndarray  # outer-loop loss trace, shape (n_steps,)
    final_grad_norm: jnp.ndarray  # per-cell |grad|_inf at convergence


# =====================================================================
# Woodbury helpers for the prior MVN log-prob and log-det
# =====================================================================


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, diff: jnp.ndarray
) -> jnp.ndarray:
    """Compute (diff)ᵀ Σ⁻¹ (diff) for Σ = WWᵀ + diag(d), batched over rows of diff.

    Uses the inner Woodbury identity for Σ⁻¹.  ``diff`` shape ``(B, G)``;
    return shape ``(B,)``.
    """
    inv_d = 1.0 / d
    K_dim = W.shape[1]
    K = jnp.eye(K_dim) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    # diff_inv_d = (1/d) ⊙ diff, shape (B, G)
    diff_inv_d = inv_d[None, :] * diff
    # rhs = Wᵀ ((1/d) ⊙ diff), shape (B, K)
    rhs = diff_inv_d @ W  # (B, K)
    z = jax.scipy.linalg.cho_solve((L_K, True), rhs.T).T  # (B, K)
    correction = (diff_inv_d * (z @ W.T)).sum(axis=-1)  # (B,)
    direct = (diff * diff_inv_d).sum(axis=-1)  # (B,)
    return direct - correction


def _woodbury_logdet_sigma(W: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """log det Σ for Σ = WWᵀ + diag(d) via matrix-determinant lemma."""
    inv_d = 1.0 / d
    K_dim = W.shape[1]
    K = jnp.eye(K_dim) + W.T @ (inv_d[:, None] * W)
    L_K = jnp.linalg.cholesky(K)
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))
    log_det_d = jnp.sum(jnp.log(d))
    return log_det_d + log_det_K


# =====================================================================
# Laplace ELBO
# =====================================================================


def nbln_laplace_elbo(
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    x: jnp.ndarray,
    eta: Optional[jnp.ndarray],
    log_det_neg_H: jnp.ndarray,
    counts: jnp.ndarray,
    eta_anchor: Optional[jnp.ndarray] = None,
    sigma_M: Optional[float] = None,
) -> jnp.ndarray:
    """Negative Laplace ELBO summed over a batch of cells.

    Parameters
    ----------
    mu, W, d, r : globals.  ``d > 0``, ``r > 0``.
    x : jnp.ndarray, shape (B, G)
        Per-cell MAP log-rate from the inner Newton (already
        ``stop_gradient``).
    eta : jnp.ndarray or None, shape (B,)
        Per-cell capture-offset MAP.  ``None`` when capture is off.
    log_det_neg_H : jnp.ndarray, shape (B,)
        ``log det(−H_c)`` at the MAP.  Should be evaluated with live
        globals (no damping) so its gradient flows into ``W, d, r, μ``.
    counts : jnp.ndarray, shape (B, G)
    eta_anchor, sigma_M : capture-anchor prior parameters (or ``None``).

    Returns
    -------
    jnp.ndarray, scalar
        Negative Laplace ELBO summed over the batch.
    """
    # Effective log-rate for the NB likelihood.
    if eta is not None:
        log_mean = x - eta[:, None]
    else:
        log_mean = x
    # NB log-prob using the in-house log-mean parameterisation.
    nb_lp = LogMeanNegativeBinomial(
        log_mean=log_mean, concentration=r[None, :]
    ).log_prob(counts).sum(axis=-1)

    # MVN prior log-prob:  log p(x | μ, Σ) = −½ qᵀ Σ⁻¹ q − ½ log|Σ| − G/2 log(2π).
    diff = x - mu[None, :]
    quad = _woodbury_quadform(W, d, diff)
    log_det_sigma = _woodbury_logdet_sigma(W, d)
    G = mu.shape[0]
    mvn_lp = -0.5 * quad - 0.5 * log_det_sigma - 0.5 * G * jnp.log(
        2 * jnp.pi
    )

    # Capture-anchor TruncN prior on η_c (only when capture active).
    if eta is not None and eta_anchor is not None and sigma_M is not None:
        eta_lp = dist.TruncatedNormal(
            eta_anchor, sigma_M, low=0.0
        ).log_prob(eta)
    else:
        eta_lp = jnp.zeros_like(nb_lp)

    laplace_corr = -0.5 * log_det_neg_H

    elbo_per_cell = nb_lp + mvn_lp + eta_lp + laplace_corr
    # Replace any non-finite per-cell contributions with a large but
    # finite penalty so a single pathological cell can't NaN-out the
    # whole gradient. Same defence used by PLN engine.
    elbo_per_cell = jnp.where(
        jnp.isfinite(elbo_per_cell), elbo_per_cell, -1e6
    )
    return -jnp.sum(elbo_per_cell)


# =====================================================================
# Outer driver
# =====================================================================


def nbln_laplace_em(
    counts: jnp.ndarray,
    latent_dim: int,
    *,
    n_outer_steps: int = 200,
    n_newton: int = 12,
    learning_rate: float = 5e-3,
    damping: float = 1e-4,
    capture_anchor: Optional[dict] = None,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 25,
) -> NBLNLaplaceResult:
    """Fit an NB-LogNormal model via Laplace-EM.

    Outer Adam updates ``(μ, W, d, r)`` (and the optional capture-prior
    parameters) using the Laplace-corrected marginal log-likelihood as
    the objective; inner Newton from :mod:`._newton_nbln` finds the
    per-cell MAP each step.

    Parameters
    ----------
    counts : jnp.ndarray, shape (N, G)
        Observed count matrix.
    latent_dim : int
        Rank of the low-rank covariance ``Σ = WWᵀ + diag(d)``.
    n_outer_steps : int, default 200
        Number of Adam steps on the globals.
    n_newton : int, default 12
        Number of inner Newton iterations per outer step.  10–15 is
        almost always enough for a log-concave posterior.
    learning_rate : float, default 5e-3
        Adam learning rate on ``log d`` and ``log r``; ``mu`` and ``W``
        share this rate.  Tune up if the loss plateaus, down if it
        oscillates.
    damping : float, default 1e-4
        Tikhonov damping for the inner Newton.
    capture_anchor : dict or None
        If a dict, must contain keys ``"eta_anchor"`` (shape ``(N,)``)
        and ``"sigma_M"`` (scalar).  When provided, the η_capture
        latent is included in the joint Newton solve.
    seed : int
        Reproducibility seed (used for any internal stochastics; the
        outer loop is currently deterministic).
    verbose : bool, default True
        Whether to print per-step diagnostics.
    log_every : int
        Print frequency.

    Returns
    -------
    NBLNLaplaceResult
        Final globals, per-cell MAPs, loss history, and final
        per-cell Newton gradient norms.
    """
    counts = jnp.asarray(counts, dtype=jnp.float32)
    N, G = counts.shape
    counts_np = np.asarray(counts)

    # ---- Initialise globals from data ----
    mu0 = empirical_log_mean_from_counts(counts_np)
    W0 = pca_loadings_init(counts_np, latent_dim=latent_dim)
    d0 = jnp.full((G,), 0.1, dtype=jnp.float32)
    r0 = empirical_dispersion_from_counts(counts_np)

    # Parameterise positivity-constrained globals via log.
    params = {
        "mu": mu0,
        "W": W0,
        "log_d": jnp.log(d0),
        "log_r": jnp.log(r0),
    }

    # ---- Capture-anchor setup ----
    use_capture = capture_anchor is not None
    if use_capture:
        eta_anchor = jnp.asarray(
            capture_anchor["eta_anchor"], dtype=jnp.float32
        )
        sigma_M = float(capture_anchor["sigma_M"])
    else:
        eta_anchor = None
        sigma_M = None

    # ---- Per-cell Newton state (warm-started across outer iters) ----
    x_loc = jnp.broadcast_to(mu0, (N, G))
    eta_loc = (
        jnp.full((N,), eta_anchor.mean(), dtype=jnp.float32)
        if use_capture
        else None
    )

    # ---- Outer optimiser ----
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x_init, eta_init):
        mu = params["mu"]
        W = params["W"]
        d = jnp.exp(params["log_d"])
        r = jnp.exp(params["log_r"])

        x_init_sg = jax.lax.stop_gradient(x_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)
        r_sg = jax.lax.stop_gradient(r)

        if use_capture:
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            x_new, eta_new, gn, _ = laplace_newton_batch(
                x_init_sg,
                eta_init_sg,
                counts,
                mu_sg,
                W_sg,
                d_sg,
                r_sg,
                eta_anchor,
                sigma_M,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            eta_new = jax.lax.stop_gradient(eta_new)
            log_det = laplace_log_det_neg_H_batch(
                x_new, eta_new, counts, r, W, d, sigma_M
            )
            loss = nbln_laplace_elbo(
                mu, W, d, r, x_new, eta_new, log_det, counts,
                eta_anchor, sigma_M,
            )
            return loss, (x_new, eta_new, gn)
        else:
            x_new, gn, _ = laplace_newton_batch_x_only(
                x_init_sg,
                counts,
                mu_sg,
                W_sg,
                d_sg,
                r_sg,
                n_newton,
                damping,
            )
            x_new = jax.lax.stop_gradient(x_new)
            log_det = laplace_log_det_neg_H_batch_x_only(
                x_new, None, counts, r, W, d, 1.0
            )
            loss = nbln_laplace_elbo(
                mu, W, d, r, x_new, None, log_det, counts,
                None, None,
            )
            return loss, (x_new, None, gn)

    grad_loss = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    losses = []
    for step in range(n_outer_steps):
        (loss, (x_loc, eta_new, gn)), grads = grad_loss(
            params, x_loc, eta_loc
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if eta_new is not None:
            eta_loc = eta_new
        losses.append(float(loss))

        if verbose and (
            step == 0
            or (step + 1) % log_every == 0
            or step == n_outer_steps - 1
        ):
            print(
                f"  step {step + 1:4d}/{n_outer_steps}: "
                f"loss={float(loss):.3f}  "
                f"|grad_inner|_inf median={float(jnp.median(gn)):.2e}  "
                f"max={float(jnp.max(gn)):.2e}"
            )

    # Final Newton pass for diagnostics.
    if use_capture:
        x_loc, eta_loc, gn_final, _ = laplace_newton_batch(
            x_loc, eta_loc, counts,
            params["mu"], params["W"],
            jnp.exp(params["log_d"]), jnp.exp(params["log_r"]),
            eta_anchor, sigma_M, max(2 * n_newton, 20), damping,
        )
    else:
        x_loc, gn_final, _ = laplace_newton_batch_x_only(
            x_loc, counts,
            params["mu"], params["W"],
            jnp.exp(params["log_d"]), jnp.exp(params["log_r"]),
            max(2 * n_newton, 20), damping,
        )

    return NBLNLaplaceResult(
        mu=params["mu"],
        W=params["W"],
        d=jnp.exp(params["log_d"]),
        r=jnp.exp(params["log_r"]),
        x_loc=x_loc,
        eta_loc=eta_loc,
        losses=np.asarray(losses, dtype=np.float64),
        final_grad_norm=gn_final,
    )

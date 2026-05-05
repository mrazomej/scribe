"""Outer training loop for Laplace-mode PLN inference.

Self-contained custom loop (not using ``numpyro.infer.SVI``) that
orchestrates outer Adam on global parameters with inner Newton on
per-cell latents. Variational EM in the spirit of ``PLNmodels`` (R).

The decision to bypass NumPyro's SVI machinery is deliberate. The
Laplace path requires:

1. **Per-cell mutable state** for ``x_loc`` and ``eta_loc`` that
   persists across outer steps but does *not* receive gradients from
   Adam (Newton writes them).
2. **Manual control over the gradient flow**: only globals
   (``mu``, ``W``, ``d``, hyperparameters) get Adam updates; ``x_loc``
   and ``eta_loc`` are updated by Newton with ``stop_gradient`` on
   their inputs, so the implicit-function-theorem corrections are
   intentionally dropped.
3. **A custom loss function** that adds the Laplace correction
   ``-½ log det(-H_c)`` per cell.

All three are most cleanly expressed in a hand-rolled loop. NumPyro's
SVI handles (1) via ``numpyro.mutable`` and (2) via ``optax.multi_transform``,
but threading them through requires either invasive surgery in
``model_builder.py`` or ad-hoc workarounds. The custom loop in this
file is roughly 200 lines and uses standard JAX/optax primitives.

Public API
----------
- :class:`LaplaceRunResult` — result container mirroring ``SVIRunResult``.
- :class:`LaplaceInferenceEngine` — static-method entry point analogous
  to ``SVIInferenceEngine``.

Both are kept signature-compatible with the SVI counterparts so that
``ScribeLaplaceResults`` can reuse most of ``ScribeVAEResults``'s
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax
from jax import random
from sklearn.utils.extmath import randomized_svd

from ..core.lnm_data_init import compute_encoder_standardization  # noqa: F401
from ..core.pln_data_init import (
    empirical_log_mean_from_counts,
    pca_loadings_init,
)
from ..models.config import (
    EarlyStoppingConfig,
    LaplaceConfig,
    ModelConfig,
    SVIConfig,
)
from ._laplace_newton import (
    laplace_newton_batch,
    laplace_newton_batch_x_only,
)
from ._progress_backend import (
    ProgressBackendName,
    build_progress_reporter,
)
from .laplace_checkpoint import (
    laplace_checkpoint_exists,
    load_laplace_checkpoint,
    save_laplace_checkpoint,
)

logger = logging.getLogger(__name__)


def _mean_ignoring_nans(values) -> float:
    """Mean of ``values`` with NaNs filtered out; ``inf`` if all NaN."""
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("inf")
    return float(arr[mask].mean())


# =====================================================================
# Optimizer resolution
# =====================================================================


def _build_optax_from_config(
    optimizer_config: Optional["SVIConfig.OptimizerConfig"],
):
    """Translate a serialized ``OptimizerConfig`` into an ``optax`` optimizer.

    Mirrors :func:`scribe.inference.optimizer_factory.build_optimizer_from_config`
    but returns a native ``optax.GradientTransformation`` for the
    Laplace engine's hand-rolled training loop. NumPyro's optimizers
    wrap ``optax`` internally, so the two factories are
    interoperable in spirit — this one just avoids the NumPyro layer.

    Supported optimiser names match the SVI/VAE side
    (``adam``, ``clipped_adam``, ``adagrad``, ``rmsprop``, ``sgd``,
    ``momentum``).

    Parameters
    ----------
    optimizer_config : SVIConfig.OptimizerConfig or None
        Structured optimizer spec. ``None`` returns scribe's default
        (``optax.adam(1e-3)``) — same default the SVI engine falls
        back to.

    Returns
    -------
    optax.GradientTransformation
        Ready to feed into ``optax.apply_updates`` /
        ``optimizer.init`` / ``optimizer.update``.
    """
    # Conservative default mirrors ``scribe.svi.SVIInferenceEngine``'s
    # built-in fallback (Adam, lr=1e-3) when neither ``optimizer`` nor
    # ``optimizer_config`` is supplied.
    if optimizer_config is None:
        return optax.adam(1e-3)

    name = optimizer_config.name
    lr = (
        optimizer_config.step_size
        if optimizer_config.step_size is not None
        else 1e-3
    )
    b1 = optimizer_config.b1 if optimizer_config.b1 is not None else 0.9
    b2 = optimizer_config.b2 if optimizer_config.b2 is not None else 0.999
    eps = optimizer_config.eps if optimizer_config.eps is not None else 1e-8
    grad_clip_norm = optimizer_config.grad_clip_norm
    weight_decay = optimizer_config.weight_decay

    if name == "adam":
        opt = optax.adam(lr, b1=b1, b2=b2, eps=eps)
    elif name == "clipped_adam":
        # ``clipped_adam`` chains a global-norm clip with Adam — same
        # semantics as ``numpyro.optim.ClippedAdam`` and the user's
        # familiar VAE-mode setup.
        clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else 1.0
        opt = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(lr, b1=b1, b2=b2, eps=eps),
        )
    elif name == "adagrad":
        opt = optax.adagrad(lr, eps=eps)
    elif name == "rmsprop":
        # ``optax.rmsprop`` accepts ``decay`` (= b2 in their API).
        opt = optax.rmsprop(lr, decay=b2, eps=eps)
    elif name == "sgd":
        opt = optax.sgd(lr)
    elif name == "momentum":
        momentum = (
            optimizer_config.momentum
            if optimizer_config.momentum is not None
            else 0.9
        )
        opt = optax.sgd(lr, momentum=float(momentum))
    else:
        # Should never happen — ``OptimizerConfig._normalize_name``
        # validates this. Defensive raise just in case.
        raise ValueError(f"Unsupported optimizer name {name!r} for Laplace.")

    if weight_decay is not None:
        opt = optax.chain(opt, optax.add_decayed_weights(weight_decay))

    if grad_clip_norm is not None and name != "clipped_adam":
        # ``optimizer_factory.build_optimizer_from_config`` only allows
        # ``grad_clip_norm`` with ``clipped_adam`` for SVI — match
        # that behavior here for parity.
        raise ValueError(
            "grad_clip_norm is currently supported only with "
            "optimizer name 'clipped_adam' for Laplace inference."
        )

    return opt


# =====================================================================
# Result container
# =====================================================================


@dataclass
class LaplaceRunResult:
    """Result of a Laplace-mode training run.

    Mirrors :class:`SVIRunResult` so that downstream packaging code can
    treat the two interchangeably.

    Attributes
    ----------
    globals : dict
        Trained global parameters: ``mu`` (G,), ``W`` (G, k),
        ``d_log`` (G,) — the unconstrained ``log d_pln`` so that
        ``d = exp(d_log)``.
    x_loc : jnp.ndarray, shape (n_cells, G)
        Per-cell MAP estimate of the latent log-rate ``x_c``.
    eta_loc : jnp.ndarray or None, shape (n_cells,)
        Per-cell MAP estimate of the capture offset, or ``None`` when
        no capture anchor is active.
    final_grad_norms : jnp.ndarray, shape (n_cells,)
        Per-cell L∞ gradient norm at the final outer step. Used for
        convergence diagnostics.
    losses : jnp.ndarray
        Outer-loop loss history (negative Laplace ELBO).
    n_steps_run : int
        Number of outer steps executed.
    model_config : ModelConfig, optional
        For provenance.
    """

    globals: Dict[str, jnp.ndarray]
    x_loc: jnp.ndarray
    eta_loc: Optional[jnp.ndarray]
    final_grad_norms: jnp.ndarray
    losses: jnp.ndarray
    n_steps_run: int
    model_config: Optional[ModelConfig] = None
    # Early-stopping / best-loss diagnostics. ``early_stopped`` is
    # True when the patience criterion fired before ``n_steps``;
    # ``best_loss`` is the smoothed loss at the best step (``inf`` if
    # no improvement was ever recorded — typical for very short runs
    # that finished inside the warmup window). ``stopped_at_step`` is
    # the actual number of outer iterations executed (which can be
    # less than ``n_steps`` if early-stopping fired).
    early_stopped: bool = False
    best_loss: float = float("inf")
    stopped_at_step: int = 0


# =====================================================================
# Laplace ELBO components
# =====================================================================


def _woodbury_logdet_sigma(W: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """``log det(WW' + diag(d))`` via the matrix-determinant lemma.

    .. math::
        \\log \\det(W W^\\top + \\mathrm{diag}(d))
        = \\sum_g \\log d_g + \\log \\det(I_k + W^\\top \\mathrm{diag}(1/d) W).

    The ``k × k`` Cholesky is the only nonscalar operation, keeping
    this at ``O(G·k + k^3)``.
    """
    k = W.shape[1]
    inv_d = 1.0 / d
    K = jnp.eye(k) + W.T @ (inv_d[:, None] * W)
    L = jnp.linalg.cholesky(K)
    return jnp.sum(jnp.log(d)) + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def _woodbury_quadform(
    W: jnp.ndarray, d: jnp.ndarray, x_minus_mu: jnp.ndarray
) -> jnp.ndarray:
    """``(x − μ)^T Σ⁻¹ (x − μ)`` via Woodbury on ``Σ⁻¹``.

    Vectorised over a leading batch axis: ``x_minus_mu`` may be
    ``(G,)`` or ``(B, G)``.
    """
    k = W.shape[1]
    inv_d = 1.0 / d
    # First term: (x − μ)^T diag(1/d) (x − μ)
    base = jnp.sum((x_minus_mu * inv_d) * x_minus_mu, axis=-1)
    # Correction via Woodbury: − (D y)^T (I_k + W^T D W)⁻¹ (D y) ⊙ W
    # Algebra: Σ⁻¹ = D − D W K⁻¹ W^T D, K = I + W^T D W.
    K = jnp.eye(k) + W.T @ (inv_d[:, None] * W)
    L = jnp.linalg.cholesky(K)
    Dy = inv_d * x_minus_mu  # (..., G)
    WtDy = jnp.einsum("gk,...g->...k", W, Dy)  # (..., k)
    KinvWtDy = jax.scipy.linalg.cho_solve((L, True), WtDy.T).T  # (..., k)
    correction = jnp.sum(WtDy * KinvWtDy, axis=-1)
    return base - correction


def _laplace_elbo(
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    x: jnp.ndarray,
    eta: Optional[jnp.ndarray],
    log_det_neg_H: jnp.ndarray,
    counts: jnp.ndarray,
    eta_anchor: Optional[jnp.ndarray],
    sigma_M: float,
) -> jnp.ndarray:
    """Laplace-corrected ELBO summed over a batch of cells.

    The Laplace approximation to ``log p(u_c)`` is

    .. math::
        \\log p(u_c) \\approx \\log p(u_c, x_c^*, \\eta_c^*)
                            + \\tfrac{1}{2}(G + 1)\\log(2\\pi)
                            - \\tfrac{1}{2} \\log\\det(-H_c)

    Summed over cells, this is the outer-loop objective. We drop the
    ``½(G+1)log(2π)`` constant (it does not depend on parameters).

    Parameters
    ----------
    mu, W, d : jnp.ndarray
        Global parameters at the current outer iterate. ``d > 0``.
    x : jnp.ndarray, shape (B, G)
        Per-cell MAP from the inner Newton (with ``stop_gradient``).
    eta : jnp.ndarray or None, shape (B,)
        Per-cell capture offset MAP. ``None`` when no capture anchor.
    log_det_neg_H : jnp.ndarray, shape (B,)
        Per-cell ``log det(-H_c)`` at the MAP, from the Newton kernel.
    counts : jnp.ndarray, shape (B, G)
        Observed counts (data).
    eta_anchor, sigma_M : capture-anchor prior params (or ``None``).

    Returns
    -------
    jnp.ndarray, scalar
        The negative Laplace ELBO summed over cells (loss to minimise).
    """
    rate = jnp.exp(
        jnp.clip(x - (eta[:, None] if eta is not None else 0.0), -30.0, 30.0)
    )
    # Poisson log-prob per cell (drops constant lgamma(u + 1)):
    # log p(u | x, eta) = sum_g [u_g (x_g - eta) - exp(x_g - eta)]
    if eta is not None:
        poisson_lp = jnp.sum(counts * (x - eta[:, None]), axis=-1) - jnp.sum(
            rate, axis=-1
        )
    else:
        poisson_lp = jnp.sum(counts * x, axis=-1) - jnp.sum(rate, axis=-1)

    # MVN prior log-prob: log p(x | μ, Σ) = -½ (x-μ)' Σ⁻¹ (x-μ) - ½ log|Σ| - G/2 log(2π)
    quad = _woodbury_quadform(W, d, x - mu)  # (B,)
    log_det_sigma = _woodbury_logdet_sigma(W, d)
    G = mu.shape[0]
    mvn_lp = -0.5 * quad - 0.5 * log_det_sigma - 0.5 * G * jnp.log(2 * jnp.pi)

    # Capture-anchor TruncN prior on eta_c. We use the truncated-
    # normal log-prob via numpyro for correctness on the truncation
    # constant.
    if eta is not None and eta_anchor is not None:
        eta_lp = dist.TruncatedNormal(eta_anchor, sigma_M, low=0.0).log_prob(
            eta
        )
    else:
        eta_lp = jnp.zeros_like(poisson_lp)

    # Laplace correction: -½ log det(-H).
    laplace_corr = -0.5 * log_det_neg_H

    elbo_per_cell = poisson_lp + mvn_lp + eta_lp + laplace_corr
    return -jnp.sum(elbo_per_cell)


# =====================================================================
# Engine
# =====================================================================


class LaplaceInferenceEngine:
    """Outer SVI + inner Newton training loop for PLN Laplace inference.

    Static-method entry point analogous to ``SVIInferenceEngine``.
    Builds globals from data initialization, runs a custom training
    loop (not NumPyro SVI), and returns a :class:`LaplaceRunResult`.

    The design intentionally bypasses NumPyro's SVI machinery — see the
    module docstring for rationale.
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
        """Run Laplace-mode training.

        Parameters
        ----------
        model_config : ModelConfig
            Model configuration. Used for provenance and downstream
            results packaging; the engine does not consult it for
            actual training (globals come from data init below).
        count_data : jnp.ndarray, shape (n_cells, n_genes)
            Observed UMI count matrix. May be ``np.ndarray``.
        n_cells, n_genes, latent_dim : int
            Dataset and model dimensions.
        laplace_config : LaplaceConfig
            Outer-loop and Newton-step hyperparameters.
        seed : int
            JAX PRNG seed.
        capture_anchor : Optional[Tuple[float, float]]
            ``(log_M_0, sigma_M)``. When ``None``, no capture anchor —
            Newton runs on ``x_c`` only. When set, joint Newton on
            ``(x_c, eta_c)`` with the biology-informed TruncN prior.
        progress : bool, default=True
            Whether to render an interactive progress bar during
            training. The bar uses scribe's standard backend
            (``rich`` in terminals, ``tqdm`` in notebooks) and
            displays the running average loss and the worst per-cell
            Newton gradient norm — the latter is a Laplace-specific
            diagnostic that flags slow inner-Newton convergence.
        progress_backend : {"auto", "rich", "tqdm", "none"}, default="auto"
            Backend policy for the progress bar. ``"auto"`` matches
            the SVI/VAE engines: ``rich`` in terminals, ``tqdm`` in
            notebooks (Jupyter / marimo / IPython). Override to force
            a specific backend or disable rendering.
        log_progress_lines : bool, default=False
            When ``True``, also emit a plain-text progress line at
            each periodic update (useful for non-interactive runs
            captured to log files).

        Returns
        -------
        LaplaceRunResult
        """
        counts = jnp.asarray(count_data, dtype=jnp.float32)
        rng = random.PRNGKey(seed)

        # ---- Data-driven initialization of globals ----
        # Decoder bias mu = mean(log(u + 1)) per gene.
        mu_init = jnp.asarray(
            empirical_log_mean_from_counts(np.asarray(counts)),
            dtype=jnp.float32,
        )
        # PCA loadings on centered log-counts → init for W (G, k).
        W_init = jnp.asarray(
            pca_loadings_init(np.asarray(counts), latent_dim=latent_dim),
            dtype=jnp.float32,
        )
        # Per-gene residual log-variance, initialized small.
        # Store ``log d`` as the unconstrained variable so the
        # optimizer sees an unconstrained scalar; we exponentiate to
        # get ``d > 0`` everywhere it is used.
        d_log_init = jnp.full((n_genes,), jnp.log(0.1), dtype=jnp.float32)

        params = {"mu": mu_init, "W": W_init, "d_log": d_log_init}

        # ---- Per-cell latent storage ----
        # Warm-start Newton at a near-MAP point so the first iteration
        # is well-conditioned. We initialise ``x_c`` so that the
        # *Poisson part* of the joint log-density already approximately
        # matches the data: ``exp(x_c - η_c) ≈ u_c``. This makes the
        # gradient at step 0 essentially the prior pull
        # ``-Σ⁻¹(x - μ)``, which Newton handles robustly. Without
        # this warm start the first joint Newton step can produce
        # ``η`` corrections of order ``log(L_c)``, propagating to
        # ``x − η`` magnitudes that overflow ``exp`` in float32.
        if capture_anchor is not None:
            log_M0, sigma_M = capture_anchor
            log_M0 = float(log_M0)
            sigma_M = float(sigma_M)
            log_lib = jnp.log(jnp.maximum(jnp.sum(counts, axis=-1), 1.0))
            eta_anchor_per_cell = log_M0 - log_lib  # (n_cells,)
            eta_loc = eta_anchor_per_cell  # warm-start at the anchor
            # x_loc init = log(u + 1) + η_anchor. Then exp(x − η) ≈ u
            # by construction.
            x_loc = jnp.log(counts + 1.0) + eta_loc[:, None]
        else:
            log_M0 = None
            sigma_M = 1.0  # placeholder; unused when capture is off
            eta_anchor_per_cell = None
            eta_loc = None
            # No-capture init: log(u + 1) so exp(x) ≈ u + 1 ≈ u for
            # large counts. Drops the bias term — Newton finds μ
            # itself from the prior gradient.
            x_loc = jnp.log(counts + 1.0)

        # ---- Optimizer ----
        # Resolve the outer-loop optimizer with the same precedence
        # the SVI/VAE paths use:
        #   1. Explicit ``laplace_config.optimizer`` (a pre-built
        #      ``optax.GradientTransformation``) — highest priority,
        #      power-user override.
        #   2. ``laplace_config.optimizer_config`` (serialized name +
        #      kwargs) translated via :func:`_build_optax_from_config`.
        #   3. Default ``optax.adam(1e-3)`` when neither is set.
        # This matches the resolution rules in
        # :func:`scribe.inference.optimizer_factory.resolve_svi_optimizer`
        # so users can use the same ``optimizer_config`` dict (e.g.
        # ``{"name": "clipped_adam", "step_size": 1e-4,
        # "grad_clip_norm": 10.0}``) for SVI, VAE, and Laplace fits.
        if laplace_config.optimizer is not None:
            opt = laplace_config.optimizer
        else:
            opt = _build_optax_from_config(laplace_config.optimizer_config)
        opt_state = opt.init(params)

        # ---- Step function (inner Newton + outer gradient + Adam) ----
        n_newton = int(laplace_config.n_newton_steps)
        damping = float(laplace_config.damping)
        batch_size = laplace_config.batch_size
        if batch_size is None:
            batch_size = n_cells
        batch_size = int(batch_size)
        # SVI subsample-scaling correction (Hoffman et al. 2013). The
        # full-data ELBO is sum_{c=1..N} elbo_c. Under uniform
        # subsampling of B cells, the unbiased estimator is
        # (N / B) * sum_{c in batch} elbo_c. NumPyro applies this
        # automatically via plate(..., subsample_size=B); our hand-
        # rolled loop has to apply it explicitly. The scale is a
        # Python float, so JAX folds it into the JIT trace as a
        # constant (no per-step recompilation).
        # When batch_size == n_cells the factor is 1.0 exactly and
        # the full-batch path is byte-identical to before.
        data_scale = float(n_cells) / float(batch_size)

        def laplace_loss(
            params,
            x_batch_init,
            eta_batch_init,
            counts_batch,
            eta_anchor_batch,
        ):
            """Compute negative Laplace ELBO on a batch.

            Inner Newton runs with ``stop_gradient`` on the iterates
            (we treat the MAP as a function of params with vanishing
            gradient at the MAP — implicit-function-theorem
            argument, the same approximation used by PLNmodels).
            """
            mu, W, d_log = params["mu"], params["W"], params["d_log"]
            d = jnp.exp(d_log)
            # Stop_gradient inside Newton: implementation of the
            # variational-EM gradient flow.
            x_init_sg = jax.lax.stop_gradient(x_batch_init)
            mu_sg = jax.lax.stop_gradient(mu)
            W_sg = jax.lax.stop_gradient(W)
            d_sg = jax.lax.stop_gradient(d)
            if capture_anchor is not None:
                eta_init_sg = jax.lax.stop_gradient(eta_batch_init)
                eta_anch_sg = jax.lax.stop_gradient(eta_anchor_batch)
                x_new, eta_new, _gn, log_det = laplace_newton_batch(
                    x_init_sg,
                    eta_init_sg,
                    counts_batch,
                    mu_sg,
                    W_sg,
                    d_sg,
                    eta_anch_sg,
                    sigma_M,
                    n_newton,
                    damping,
                )
                # The MAP itself is treated as constant w.r.t. params
                # (standard variational-EM approximation).
                x_new = jax.lax.stop_gradient(x_new)
                eta_new = jax.lax.stop_gradient(eta_new)
                log_det = jax.lax.stop_gradient(log_det)
                loss = data_scale * _laplace_elbo(
                    mu,
                    W,
                    d,
                    x_new,
                    eta_new,
                    log_det,
                    counts_batch,
                    eta_anch_sg,
                    sigma_M,
                )
                return loss, (x_new, eta_new, _gn)
            else:
                x_new, _gn, log_det = laplace_newton_batch_x_only(
                    x_init_sg,
                    counts_batch,
                    mu_sg,
                    W_sg,
                    d_sg,
                    n_newton,
                    damping,
                )
                x_new = jax.lax.stop_gradient(x_new)
                log_det = jax.lax.stop_gradient(log_det)
                loss = data_scale * _laplace_elbo(
                    mu,
                    W,
                    d,
                    x_new,
                    None,
                    log_det,
                    counts_batch,
                    None,
                    sigma_M,
                )
                return loss, (x_new, None, _gn)

        loss_grad_fn = jax.jit(jax.value_and_grad(laplace_loss, has_aux=True))

        @jax.jit
        def update_step(params, opt_state, x_loc, eta_loc, idx):
            """One outer-loop step: inner Newton, gradient on globals, Adam."""
            counts_batch = counts[idx]
            x_batch_init = x_loc[idx]
            if eta_loc is not None:
                eta_batch_init = eta_loc[idx]
                eta_anchor_batch = eta_anchor_per_cell[idx]
            else:
                eta_batch_init = jnp.zeros(idx.shape[0])
                eta_anchor_batch = jnp.zeros(idx.shape[0])

            (loss, (x_new, eta_new, gn)), grads = loss_grad_fn(
                params,
                x_batch_init,
                eta_batch_init,
                counts_batch,
                eta_anchor_batch,
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            # Write-back the Newton-updated MAP into the per-cell
            # arrays for warm-starting the next outer step.
            x_loc = x_loc.at[idx].set(x_new)
            if eta_loc is not None:
                eta_loc = eta_loc.at[idx].set(eta_new)
            return params, opt_state, x_loc, eta_loc, gn, loss

        # ---- Outer loop ----
        n_steps = int(laplace_config.n_steps)
        rng_step = rng

        # Periodic-update interval mirrors SVI's ``_progress_display_interval``
        # heuristic: refresh the loss-info string roughly 100 times over
        # the run, with a floor of 1 so very short runs still update.
        display_interval = max(1, n_steps // 100)
        init_loss: Optional[float] = None

        # --------------------------------------------------------------
        # Early-stopping / checkpointing setup. Mirrors the SVI engine.
        # When ``laplace_config.early_stopping`` is None we synthesise a
        # disabled config so the rest of the loop can read fields
        # uniformly without per-call ``if early_stopping is None``
        # branching. checkpoint_dir=None on the disabled config means
        # ``should_checkpoint`` is always False.
        early_stopping: EarlyStoppingConfig = (
            laplace_config.early_stopping
            if laplace_config.early_stopping is not None
            else EarlyStoppingConfig(enabled=False)
        )
        checkpoint_dir = early_stopping.checkpoint_dir

        # Loss history is stored as a list because we mutate it
        # in-place (append per step, slice on smoothed-mean compute).
        # We materialise a numpy array at return time.
        losses: List[float] = []
        start_step = 0
        best_loss = float("inf")
        patience_counter = 0
        resumed = False
        resume_message: Optional[str] = None
        # In-memory snapshot of the best state so far (used when
        # ``restore_best=True`` regardless of whether early-stopping
        # actually fires). None until the first improvement is seen.
        best_state: Optional[Dict[str, Any]] = None
        best_step = 0

        # If a checkpoint dir is set and ``resume=True``, try to
        # restore optax state + per-cell MAPs and continue from where
        # the last run stopped. The orbax restore needs a target
        # pytree with the right shapes and dtypes — we just initialised
        # ``params``, ``opt_state``, ``x_loc``, ``eta_loc`` above, so
        # they double as the target structure here.
        if (
            checkpoint_dir
            and early_stopping.resume
            and laplace_checkpoint_exists(checkpoint_dir)
        ):
            target_state = {
                "params": params,
                "opt_state": opt_state,
                "x_loc": x_loc,
                # Orbax cannot serialise None; the saver substitutes a
                # zero placeholder when capture is off, and ``has_eta_loc``
                # in the metadata controls whether load returns None.
                "eta_loc": (
                    eta_loc
                    if eta_loc is not None
                    else jnp.zeros((1,), dtype=jnp.float32)
                ),
            }
            loaded = load_laplace_checkpoint(checkpoint_dir, target_state)
            if loaded is not None:
                restored_state, md, restored_losses = loaded
                params = restored_state["params"]
                opt_state = restored_state["opt_state"]
                x_loc = restored_state["x_loc"]
                eta_loc = restored_state["eta_loc"]
                start_step = md.step + 1
                best_loss = md.best_loss
                patience_counter = md.patience_counter
                losses = list(restored_losses)
                resumed = True
                best_loss_str = (
                    f"{best_loss:.4e}" if np.isfinite(best_loss) else "N/A"
                )
                resume_message = (
                    f"Resumed Laplace from checkpoint at step "
                    f"{start_step} (best_loss: {best_loss_str})"
                )

        # Track the last-checkpointed step so we save at intervals
        # rather than on every check. Sentinel ``-checkpoint_every``
        # forces a save on the first eligible check when starting
        # fresh (matches SVI semantics).
        last_checkpoint_step = (
            start_step if resumed else -early_stopping.checkpoint_every
        )
        early_stopped = False
        eps = 1e-8  # divide-by-zero guard for percentage-based delta

        # Build the same progress backend used by SVI/VAE so terminal
        # users get rich and notebook users (Jupyter / marimo /
        # IPython) get tqdm. The reporter is a no-op when
        # ``progress=False`` or the backend resolves to ``"none"``.
        progress_reporter = build_progress_reporter(
            progress=progress,
            progress_backend=progress_backend,
        )
        with progress_reporter as reporter:
            init_loss_display = (
                f"{losses[0]:.4e}" if losses else "pending"
            )
            reporter.start(
                description=(
                    "Laplace optimization" + (" (resumed)" if resumed else "")
                ),
                total=n_steps,
                completed=start_step,
                loss_info=f"init loss: {init_loss_display}",
            )
            if resume_message is not None:
                reporter.print_message(resume_message)

            # Track ``init_loss`` for the periodic display string.
            # When resumed, fall back to the first historical loss so
            # the displayed bar carries continuous context across
            # resumes.
            if resumed and losses:
                init_loss = float(losses[0])

            for step in range(start_step, n_steps):
                rng_step, subkey = random.split(rng_step)
                if batch_size >= n_cells:
                    idx = jnp.arange(n_cells)
                else:
                    idx = random.choice(
                        subkey, n_cells, shape=(batch_size,), replace=False
                    )
                params, opt_state, x_loc, eta_loc, gn, loss = update_step(
                    params, opt_state, x_loc, eta_loc, idx
                )
                loss_val = float(loss)
                losses.append(loss_val)
                if init_loss is None:
                    init_loss = loss_val

                # Refresh the displayed loss-info every
                # ``display_interval`` steps, matching SVI's format
                # and appending the worst per-cell Newton gradient
                # norm — a Laplace-specific health check.
                step_completed = step + 1  # 1-based for display
                should_display = (
                    step == start_step
                    or step == n_steps - 1
                    or step_completed % display_interval == 0
                )
                if should_display:
                    window_start = max(0, len(losses) - display_interval)
                    window_end = len(losses)
                    avg_loss = _mean_ignoring_nans(
                        losses[window_start:window_end]
                    )
                    worst_grad = float(jnp.max(gn))
                    loss_info = (
                        f"init loss: {init_loss:.4e}, "
                        f"avg. loss [{window_start + 1}-{window_end}]: "
                        f"{avg_loss:.4e}, "
                        f"worst Newton grad: {worst_grad:.3e}"
                    )
                    reporter.update(advance=1, loss_info=loss_info)
                    if log_progress_lines:
                        print(
                            "Laplace progress "
                            f"[{window_end}/{n_steps}] "
                            f"init loss: {init_loss:.4e}, "
                            f"avg. loss [{window_start + 1}-{window_end}]: "
                            f"{avg_loss:.4e}, "
                            f"worst Newton grad: {worst_grad:.3e}"
                        )
                else:
                    reporter.update(advance=1)

                # ------------------------------------------------------
                # Early-stopping + checkpointing block. Mirrors the SVI
                # engine's ``_run_with_early_stopping`` behaviour:
                # smoothed-loss best-tracking, periodic orbax saves,
                # patience-based stop trigger, optional restore-best.
                # All branches gate on ``len(losses) >= smoothing_window``
                # so the warmup window has time to fill before any of
                # the criteria fire.
                # ------------------------------------------------------
                should_check = (
                    step_completed % early_stopping.check_every == 0
                    and len(losses) >= early_stopping.smoothing_window
                )
                if should_check:
                    window_start = max(
                        0, len(losses) - early_stopping.smoothing_window
                    )
                    smoothed_loss = _mean_ignoring_nans(
                        losses[window_start:]
                    )

                    past_warmup = step >= early_stopping.warmup
                    should_track = past_warmup and (
                        early_stopping.enabled
                        or early_stopping.restore_best
                    )

                    if should_track:
                        if not np.isfinite(best_loss):
                            best_loss = smoothed_loss
                            best_step = step
                            if early_stopping.restore_best:
                                # Snapshot pytree leaves; ``params`` and
                                # ``opt_state`` are pytrees of JAX arrays
                                # (immutable), but ``x_loc``/``eta_loc``
                                # rebind on every update_step so we copy
                                # the references explicitly.
                                best_state = {
                                    "params": params,
                                    "opt_state": opt_state,
                                    "x_loc": x_loc,
                                    "eta_loc": eta_loc,
                                }
                            patience_counter = 0
                        else:
                            improvement = best_loss - smoothed_loss
                            if early_stopping.min_delta_pct is not None:
                                denom = max(abs(best_loss), eps)
                                improvement_pct = (
                                    100.0 * improvement / denom
                                )
                                is_improvement = (
                                    improvement_pct
                                    > early_stopping.min_delta_pct
                                )
                            else:
                                is_improvement = (
                                    improvement > early_stopping.min_delta
                                )

                            if is_improvement:
                                best_loss = smoothed_loss
                                best_step = step
                                if early_stopping.restore_best:
                                    best_state = {
                                        "params": params,
                                        "opt_state": opt_state,
                                        "x_loc": x_loc,
                                        "eta_loc": eta_loc,
                                    }
                                patience_counter = 0
                            else:
                                patience_counter += (
                                    early_stopping.check_every
                                )

                    # Save checkpoint at fixed intervals (independent of
                    # whether the loss improved) so long-running fits
                    # are always resumable. ``checkpoint_dir`` is None
                    # when the user did not configure persistence.
                    should_checkpoint = (
                        checkpoint_dir is not None
                        and (step - last_checkpoint_step)
                        >= early_stopping.checkpoint_every
                    )
                    if should_checkpoint:
                        save_laplace_checkpoint(
                            checkpoint_dir=checkpoint_dir,
                            params=params,
                            opt_state=opt_state,
                            x_loc=x_loc,
                            eta_loc=eta_loc,
                            step=step,
                            best_loss=best_loss,
                            losses=losses,
                            patience_counter=patience_counter,
                        )
                        last_checkpoint_step = step

                    # Trigger early stopping when patience is exceeded.
                    # Past-warmup gating prevents premature stops while
                    # the inner Newton is still finding the MAP basin.
                    if (
                        early_stopping.enabled
                        and past_warmup
                        and patience_counter >= early_stopping.patience
                    ):
                        early_stopped = True
                        reporter.print_message(
                            f"Early stopping triggered at step "
                            f"{step + 1} (no improvement for "
                            f"{patience_counter} steps, best loss "
                            f"{best_loss:.4e} at step {best_step + 1})"
                        )
                        break

        # ---- Restore best state if requested ----
        # ``early_stopping.restore_best`` (default True) restores the
        # snapshot taken at the lowest smoothed loss, regardless of
        # whether early stopping actually triggered. Mirrors the SVI
        # engine's behaviour. When ``best_state is None`` (e.g. the
        # whole run finished inside warmup, no improvement ever
        # tracked), this is a no-op.
        if early_stopping.restore_best and best_state is not None:
            params = best_state["params"]
            opt_state = best_state["opt_state"]
            x_loc = best_state["x_loc"]
            eta_loc = best_state["eta_loc"]
            if not early_stopped:
                logger.info(
                    "Restoring best Laplace params from step %d "
                    "(best smoothed loss: %.4e)",
                    best_step + 1,
                    best_loss,
                )

        # ---- Final convergence check ----
        # Run one more Newton pass over ALL cells to capture the
        # final gradient norms used for diagnostics. Uses whichever
        # ``params``/``x_loc``/``eta_loc`` ended up active after the
        # optional restore-best step above.
        if capture_anchor is not None:
            mu_f = jax.lax.stop_gradient(params["mu"])
            W_f = jax.lax.stop_gradient(params["W"])
            d_f = jax.lax.stop_gradient(jnp.exp(params["d_log"]))
            x_loc_final, eta_loc_final, gn_final, _ = laplace_newton_batch(
                x_loc,
                eta_loc,
                counts,
                mu_f,
                W_f,
                d_f,
                eta_anchor_per_cell,
                sigma_M,
                max(2 * n_newton, 10),
                damping,
            )
        else:
            mu_f = jax.lax.stop_gradient(params["mu"])
            W_f = jax.lax.stop_gradient(params["W"])
            d_f = jax.lax.stop_gradient(jnp.exp(params["d_log"]))
            x_loc_final, gn_final, _ = laplace_newton_batch_x_only(
                x_loc,
                counts,
                mu_f,
                W_f,
                d_f,
                max(2 * n_newton, 10),
                damping,
            )
            eta_loc_final = None

        # Convergence-action handling.
        max_gn = float(jnp.max(gn_final))
        if max_gn > laplace_config.newton_tolerance:
            offending = int(jnp.sum(gn_final > laplace_config.newton_tolerance))
            msg = (
                f"Laplace Newton: {offending}/{n_cells} cells did not "
                f"converge below tolerance={laplace_config.newton_tolerance:.1e} "
                f"(worst grad-norm={max_gn:.3e})."
            )
            if laplace_config.convergence_action == "raise":
                raise RuntimeError(msg)
            elif laplace_config.convergence_action == "warn":
                logger.warning(msg)
            # 'ignore' silently drops the warning.

        return LaplaceRunResult(
            globals=params,
            x_loc=x_loc_final,
            eta_loc=eta_loc_final,
            final_grad_norms=gn_final,
            losses=jnp.asarray(losses, dtype=jnp.float32),
            n_steps_run=len(losses),
            model_config=model_config,
            early_stopped=early_stopped,
            best_loss=best_loss,
            stopped_at_step=len(losses),
        )


__all__ = ["LaplaceInferenceEngine", "LaplaceRunResult"]

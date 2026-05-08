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
from ._newton_pln import (
    laplace_log_det_neg_H_batch,
    laplace_log_det_neg_H_batch_x_only,
    laplace_newton_batch,
    laplace_newton_batch_x_only,
    pln_grad_split_batch,
    pln_grad_x_only_norm_batch,
)
from ._newton_lnm import (
    laplace_log_det_neg_H_batch_eta,
    laplace_log_det_neg_H_batch_y_alr,
    laplace_log_det_neg_H_batch_z,
    laplace_newton_batch_eta,
    laplace_newton_batch_y_alr,
    laplace_newton_batch_z,
)
from ..svi._progress_backend import (
    ProgressBackendName,
    build_progress_reporter,
)
from .checkpoint import (
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


def _grad_summary(gn_arr: jnp.ndarray) -> Tuple[float, float, float]:
    """Return ``(max, p99, median)`` of a per-cell gradient-norm array.

    Used for the progress-bar display so users can distinguish two
    qualitatively different convergence patterns:

    * **Healthy** — most cells converged (``median`` ≪ tolerance),
      a few outliers in the tail (``max`` and ``p99`` larger but
      shrinking over training). The L∞ ``max`` is the most
      conservative reading; ``p99`` says "if you ignore the worst
      1% of cells, this is your tail"; ``median`` reflects the
      typical cell.
    * **Stalled** — ``median`` plateaus *and* ``max`` doesn't trend
      down. Indicates a real convergence problem (Hessian
      conditioning, step-size cap, etc.) rather than a few
      outlier cells.

    Reading per-block: when the LNMVCP engine logs composition and
    η blocks separately, ``η`` should land at ~1e-6 across all
    summaries (scalar Newton on a strictly log-concave 1D problem,
    converges to float precision in a few iterations). Composition
    is where the interesting variability lives.
    """
    return (
        float(jnp.max(gn_arr)),
        float(jnp.percentile(gn_arr, 99)),
        float(jnp.median(gn_arr)),
    )


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


# LaplaceRunResult lives in :mod:`scribe.laplace._em` so that the
# generic driver and the legacy LNM orchestration both produce the
# same dataclass shape. Re-exported here so existing imports
# ``from scribe.laplace.engine import LaplaceRunResult`` keep working.
from ._em import LaplaceRunResult  # noqa: E402, F401


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

        Raises
        ------
        NotImplementedError
            If ``model_config.base_model`` is supported only as a
            stub (currently LNM — the kernel layer is in place at
            :mod:`scribe.laplace._newton_lnm`; the outer training
            loop integration is the remaining piece of the LNM
            Laplace plan).
        """
        # Top-level dispatch on the generative model.  All non-LNM
        # models route through the generic Laplace-EM driver
        # (``_em.run_laplace_em``) with a model-specific
        # :class:`LaplaceObservationModel` adapter.  LNM still uses
        # its parallel orchestration below until it is migrated in
        # the next phase.
        bm = getattr(model_config, "base_model", "pln")
        if bm in ("lnm", "lnmvcp"):
            return _run_lnm_inference(
                model_config=model_config,
                count_data=count_data,
                n_cells=n_cells,
                n_genes=n_genes,
                latent_dim=latent_dim,
                laplace_config=laplace_config,
                seed=seed,
                capture_anchor=capture_anchor,
                progress=progress,
                progress_backend=progress_backend,
                log_progress_lines=log_progress_lines,
            )

        from ._em import run_laplace_em

        if bm == "pln":
            from ._obs_pln import PLNObservationModel

            obs_model = PLNObservationModel(capture_anchor=capture_anchor)
        elif bm == "nbln":
            from ._obs_nbln import NBLNObservationModel

            obs_model = NBLNObservationModel(capture_anchor=capture_anchor)
        else:
            raise NotImplementedError(
                f"Laplace inference is supported for PLN, NBLN, LNM, "
                f"and LNMVCP; got base_model={bm!r}."
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
        )


# =====================================================================
# LNM Laplace orchestration
# =====================================================================


def _lnm_laplace_elbo(
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
    """Negative Laplace ELBO summed over a batch of LNM cells.

    The Laplace marginal-likelihood approximation for cell ``c`` is

    .. math::
        \\log p(u_c) \\approx
            \\log p(u_c, \\xi_c^{*})
            - \\tfrac{1}{2} \\log\\det(-H_c),

    where :math:`\\xi_c^{*}` is the per-cell MAP — :math:`z_c \\in \\mathbb{R}^k`
    when ``d_mode='low_rank'`` and :math:`y_{\\text{alr}, c} \\in \\mathbb{R}^{G-1}`
    when ``d_mode='learned'``. Summed over the batch, this is the
    outer-loop objective.

    The joint log-density splits cleanly: a multinomial likelihood
    on observed counts (with ``log_softmax`` over the augmented full-
    G logits), plus the prior on the chosen latent. For ``low_rank``
    the prior is :math:`z \\sim \\mathcal{N}(0, I_k)` (so
    :math:`-\\tfrac{1}{2} z^\\top z`); for ``learned`` it is
    :math:`y \\sim \\mathcal{N}(\\mu, W W^\\top + \\mathrm{diag}(d))`
    (low-rank-plus-diagonal Gaussian via Woodbury).
    """
    # Build per-cell ALR logits from whichever latent is in use.
    if d_mode == "low_rank":
        # z ∈ ℝ^k → y_alr = mu + W z.
        y_alr = mu[None, :] + z_or_y @ W.T  # (B, G-1)
    else:
        # y_alr ∈ ℝ^{G-1} stored directly.
        y_alr = z_or_y  # (B, G-1)

    # Multinomial likelihood. Augment to full-G with a zero at the
    # reference position, log_softmax, then dot with observed counts.
    leading = y_alr.shape[:-1]
    full_shape = leading + (n_genes,)
    full = jnp.zeros(full_shape, dtype=y_alr.dtype)
    other_idx = jnp.asarray(
        [g for g in range(n_genes) if g != int(alr_reference_idx)]
    )
    full = full.at[..., other_idx].set(y_alr)
    log_p = jax.nn.log_softmax(full, axis=-1)  # (B, G)
    multinomial_lp = jnp.sum(counts_batch * log_p, axis=-1)  # (B,)

    # Prior on the latent.
    if d_mode == "low_rank":
        # -½ z' z per cell.
        prior_lp = -0.5 * jnp.sum(z_or_y * z_or_y, axis=-1)
    else:
        # MVN(mu, WW' + diag(d)) prior on y_alr. Use Woodbury for the
        # quadratic form; the log-det is a constant w.r.t. the per-
        # cell latents but does depend on globals so it goes in.
        diff = y_alr - mu[None, :]  # (B, G-1)
        quad = _woodbury_quadform(W, d, diff)  # (B,)
        log_det_sigma = _woodbury_logdet_sigma(W, d)
        g_minus1 = mu.shape[0]
        prior_lp = (
            -0.5 * quad
            - 0.5 * log_det_sigma
            - 0.5 * g_minus1 * jnp.log(2 * jnp.pi)
        )

    # Laplace correction: -½ log det(-H_c).
    laplace_corr = -0.5 * log_det_neg_H

    elbo_per_cell = multinomial_lp + prior_lp + laplace_corr
    # Per-cell loss-finite guard. A handful of cells whose Newton
    # has wandered off (ρ near a corner of the simplex with the
    # Sherman-Morrison denominator catastrophically cancelling)
    # can produce NaN/Inf in the per-cell ELBO contributions.
    # Without this guard, those NaNs propagate through the JAX
    # autodiff into the gradient on globals and contaminate the
    # entire fit. Replacing NaN/Inf entries with 0 effectively
    # masks the divergent cells from this step's gradient — the
    # outer loop's per-step divergence guard
    # (see ``_run_lnm_inference``) will still detect a sustained
    # blow-up and abort with a constructive error.
    elbo_per_cell = jnp.where(
        jnp.isfinite(elbo_per_cell),
        elbo_per_cell,
        jnp.zeros_like(elbo_per_cell),
    )
    return -jnp.sum(elbo_per_cell)


def _run_lnm_inference(
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
    """LNM Laplace training loop (encoder-free, variational-EM style).

    Parallel structure to the PLN body in
    :meth:`LaplaceInferenceEngine.run_inference`, with two key
    branches selected by ``model_config.d_mode``:

    * ``"low_rank"`` → Newton over per-cell ``z ∈ ℝ^k``. Hessian is
      ``-H_z = I + W^T M(z) W`` with ``M(z) = N (diag(ρ) - ρρ^T)``.
      Cost per cell ``O(k³)``, no Woodbury.
    * ``"learned"`` → Newton over per-cell ``y_alr ∈ ℝ^{G-1}``.
      Hessian is ``-H_y = M_alr + Σ⁻¹``. Reuses scribe's PLN
      Woodbury machinery + one Sherman-Morrison step.

    The outer scaffolding (optimizer, mini-batching, progress bar)
    matches the PLN path. Early stopping and orbax checkpointing
    are not yet wired for LNM Laplace — they're a small follow-up.
    The caller can pass an ``early_stopping`` config but it will be
    silently ignored for now.
    """
    # ---- Resolve d_mode (model dictates the kernel branch) ----
    d_mode = getattr(model_config, "d_mode", "learned") or "learned"
    if d_mode not in ("low_rank", "learned"):
        raise ValueError(
            f"d_mode must be 'low_rank' or 'learned'; got {d_mode!r}."
        )

    # ---- Resolve ALR reference index (must be set on the config) ----
    alr_reference_idx = int(getattr(model_config, "alr_reference_idx", -1))
    if alr_reference_idx < 0 or alr_reference_idx >= n_genes:
        raise ValueError(
            "LNM Laplace needs alr_reference_idx in [0, n_genes); "
            f"got {alr_reference_idx} for n_genes={n_genes}."
        )

    counts = jnp.asarray(count_data, dtype=jnp.float32)
    rng = random.PRNGKey(seed)

    # ---- Total counts per cell (observed) ----
    n_total_per_cell = counts.sum(axis=-1)  # (n_cells,)

    # ---- Counts in ALR coordinates: drop the reference column ----
    # ``u_alr`` excludes the reference gene; we still need the full
    # ``counts`` for the multinomial log-likelihood.
    keep_mask = jnp.asarray(
        [g for g in range(n_genes) if g != alr_reference_idx]
    )
    u_alr = counts[:, keep_mask]  # (n_cells, G-1)
    g_minus1 = n_genes - 1

    # ---- Data-driven init of globals (in ALR space) ----
    # Use per-cell log-proportions to initialise mu (mean ALR logits)
    # and W (PCA loadings on centered log-proportions).
    pseudocount = 1.0
    p_full = (counts + pseudocount) / (
        n_total_per_cell[:, None] + n_genes * pseudocount
    )
    log_p_full_np = np.asarray(jnp.log(p_full))  # (n_cells, G)
    log_p_alr_np = (
        log_p_full_np[:, np.asarray(keep_mask)]
        - log_p_full_np[:, alr_reference_idx : alr_reference_idx + 1]
    )
    mu_init = jnp.asarray(log_p_alr_np.mean(axis=0), dtype=jnp.float32)
    centered = log_p_alr_np - log_p_alr_np.mean(axis=0, keepdims=True)
    # PCA loadings via truncated SVD on centered ALR.
    U, S, Vt = randomized_svd(
        centered, n_components=int(latent_dim), random_state=int(seed)
    )
    # W has shape (G-1, k); rescale so columns have unit norm * sqrt(S).
    W_init = jnp.asarray(
        Vt.T * (S / np.sqrt(max(n_cells - 1, 1))), dtype=jnp.float32
    )
    d_log_init = jnp.full((g_minus1,), jnp.log(0.01), dtype=jnp.float32)

    # Globals dict. Both plain LNM and LNMVCP fit the NB-on-totals
    # parameters (log_mu_T, log_r_T) so the engine fits the FULL
    # generative model documented in
    # paper/_logistic_normal_multinomial.qmd, not just the
    # composition block conditioned on observed totals.
    #
    # For plain LNM:
    #   u_T_c ~ NB(r_T, mu_T)
    # contributes a global log-likelihood at observed u_T (no per-
    # cell latent — Adam fits both scalars directly).
    #
    # For LNMVCP, the NB mean is per-cell:
    #   u_T_c ~ NB(r_T, mu_T · exp(-eta_c))
    # where eta_c is the per-cell capture-offset latent (handled by
    # the η-block Newton). The same (mu_T, r_T) globals are fitted.
    params = {"mu": mu_init, "W": W_init, "d_log": d_log_init}

    # Initial values for the NB globals.
    # mu_T: prior anchor when LNMVCP supplies log_M_0; otherwise the
    #   data-driven empirical mean of observed totals.
    # r_T: moderate-dispersion init; the data will refine quickly.
    if capture_anchor is not None:
        log_M0_user, sigma_M = capture_anchor
        log_mu_T_init = float(log_M0_user)
    else:
        log_M0_user = None
        sigma_M = 1.0  # placeholder; never read in the no-capture branch
        log_mu_T_init = float(
            jnp.log(jnp.maximum(jnp.mean(n_total_per_cell), 1.0))
        )
    params["log_mu_T"] = jnp.asarray(log_mu_T_init, dtype=jnp.float32)
    params["log_r_T"] = jnp.asarray(jnp.log(4.0), dtype=jnp.float32)

    # ---- Per-cell latent state (warm-started near a sensible value) ----
    # For low_rank: warm-start z = 0. For learned: warm-start y_alr = mu
    # (so the prior pull at step 0 is small and Newton focuses on data).
    if d_mode == "low_rank":
        z_loc = jnp.zeros((n_cells, latent_dim), dtype=jnp.float32)
        y_alr_loc = None
    else:
        z_loc = None
        y_alr_loc = jnp.broadcast_to(mu_init, (n_cells, g_minus1)).copy()

    # ---- Capture-anchor per-cell state (LNMVCP only) ----
    # eta_anchor_c = log(M_0) - log(L_c) is the per-cell prior mean
    # for the capture-offset latent. Initialise eta_loc at the
    # anchor so the first Newton step is well-conditioned (η near
    # the prior mode and the data-likelihood mode coincide when
    # mu_T ≈ M_0, which is what the prior says).
    if capture_anchor is not None:
        log_lib_size = jnp.log(jnp.maximum(n_total_per_cell, 1.0))
        eta_anchor_per_cell = jnp.asarray(
            float(log_M0_user) - log_lib_size, dtype=jnp.float32
        )
        eta_loc = eta_anchor_per_cell  # warm-start at the anchor
    else:
        eta_anchor_per_cell = None
        eta_loc = None

    # ---- Optimizer ----
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
    data_scale = float(n_cells) / float(batch_size)

    def lnm_loss(
        params,
        latent_init,
        eta_init,
        eta_anchor_batch,
        u_alr_batch,
        n_total_batch,
        counts_batch,
    ):
        """Compute negative Laplace ELBO on a batch of LNM(VCP) cells.

        Mirrors the PLN ``laplace_loss`` structure but exploits the
        block-diagonal Hessian: the multinomial likelihood (in z or
        y_alr) and the NB-on-totals likelihood (in η) are
        independent, so Newton on each block runs independently.
        For LNMVCP the loss adds the NB log-likelihood + η prior +
        η-block log-det correction; for plain LNM only the
        composition block contributes.
        """
        mu, W, d_log = params["mu"], params["W"], params["d_log"]
        d = jnp.exp(d_log)

        latent_init_sg = jax.lax.stop_gradient(latent_init)
        mu_sg = jax.lax.stop_gradient(mu)
        W_sg = jax.lax.stop_gradient(W)
        d_sg = jax.lax.stop_gradient(d)

        # ----- Composition block (z or y_alr) -----
        if d_mode == "low_rank":
            z_new, _gn_z = laplace_newton_batch_z(
                latent_init_sg,
                u_alr_batch,
                n_total_batch,
                mu_sg,
                W_sg,
                alr_reference_idx,
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
                alr_reference_idx,
                n_genes,
            )
            comp_loss = data_scale * _lnm_laplace_elbo(
                mu,
                W,
                d,
                counts_batch,
                z_new,
                log_det_comp,
                alr_reference_idx,
                n_genes,
                d_mode,
            )
            latent_new = z_new
            gn_comp = _gn_z
        else:
            y_new, _gn_y = laplace_newton_batch_y_alr(
                latent_init_sg,
                u_alr_batch,
                n_total_batch,
                mu_sg,
                W_sg,
                d_sg,
                alr_reference_idx,
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
                alr_reference_idx,
                n_genes,
            )
            comp_loss = data_scale * _lnm_laplace_elbo(
                mu,
                W,
                d,
                counts_batch,
                y_new,
                log_det_comp,
                alr_reference_idx,
                n_genes,
                d_mode,
            )
            latent_new = y_new
            gn_comp = _gn_y

        # ----- Totals block (always present) -----
        # Plain LNM:    u_T_c ~ NB(r_T, mu_T)         — global, no latent
        # LNMVCP:       u_T_c ~ NB(r_T, mu_T·exp(-η_c)) — η_c is per-cell
        # Both contribute a NB log-likelihood term to the loss; only
        # the LNMVCP path additionally adds the η Laplace correction.
        from jax.scipy.special import gammaln

        mu_T = jnp.exp(params["log_mu_T"])
        r_T = jnp.exp(params["log_r_T"])

        if capture_anchor is not None:
            # LNMVCP: scalar Newton on per-cell η, decoupled from the
            # composition block (block-diagonal Hessian).
            mu_T_sg = jax.lax.stop_gradient(mu_T)
            r_T_sg = jax.lax.stop_gradient(r_T)
            eta_init_sg = jax.lax.stop_gradient(eta_init)
            eta_anchor_sg = jax.lax.stop_gradient(eta_anchor_batch)

            eta_new, _gn_eta = laplace_newton_batch_eta(
                eta_init_sg,
                n_total_batch,
                r_T_sg,
                mu_T_sg,
                eta_anchor_sg,
                sigma_M,
                n_newton,
                damping,
            )
            eta_new = jax.lax.stop_gradient(eta_new)

            exp_neg_eta = jnp.exp(jnp.clip(-eta_new, -30.0, 30.0))
            rate_T = mu_T * exp_neg_eta  # per-cell NB mean
            v = r_T + rate_T
            nb_lp = (
                gammaln(n_total_batch + r_T)
                - gammaln(r_T)
                + r_T * jnp.log(r_T / v)
                + n_total_batch * jnp.log(rate_T / v)
            )
            # TruncN(η; eta_anchor, σ_M, low=0) log-prob (normaliser
            # is constant in θ when M_0, σ_M are fixed prior knobs;
            # drops out of the gradient).
            eta_diff = eta_new - eta_anchor_batch
            eta_prior_lp = -0.5 * (eta_diff * eta_diff) / (sigma_M * sigma_M)

            log_det_eta = laplace_log_det_neg_H_batch_eta(
                eta_new,
                n_total_batch,
                r_T,
                mu_T,
                sigma_M,
            )
            totals_loss = data_scale * jnp.sum(
                -(nb_lp + eta_prior_lp) + 0.5 * log_det_eta
            )
        else:
            # Plain LNM: NB mean is the global mu_T (no per-cell
            # variation); the NB log-likelihood at observed totals
            # contributes directly to the loss with no per-cell
            # Laplace correction.
            v = r_T + mu_T
            nb_lp = (
                gammaln(n_total_batch + r_T)
                - gammaln(r_T)
                + r_T * jnp.log(r_T / v)
                + n_total_batch * jnp.log(mu_T / v)
            )
            totals_loss = data_scale * jnp.sum(-nb_lp)
            eta_new = eta_init  # passthrough; not used by plain LNM
            _gn_eta = jnp.zeros_like(gn_comp)

        loss = comp_loss + totals_loss
        # Return per-block grad norms separately so the engine can
        # surface them in the progress bar — when the composition
        # block stalls but η converges (or vice versa), the user
        # sees which one needs more Newton iterations.
        gn = jnp.maximum(gn_comp, _gn_eta)
        return loss, (latent_new, eta_new, gn, gn_comp, _gn_eta)

    loss_grad_fn = jax.jit(jax.value_and_grad(lnm_loss, has_aux=True))

    @jax.jit
    def update_step(params, opt_state, latent_loc, eta_loc_arg, idx):
        u_alr_batch = u_alr[idx]
        n_total_batch = n_total_per_cell[idx]
        counts_batch = counts[idx]
        latent_init_batch = latent_loc[idx]
        # Pull eta state for LNMVCP; placeholder zeros for plain LNM.
        if eta_loc_arg is not None:
            eta_init_batch = eta_loc_arg[idx]
            eta_anchor_batch = eta_anchor_per_cell[idx]
        else:
            eta_init_batch = jnp.zeros(idx.shape[0], dtype=jnp.float32)
            eta_anchor_batch = jnp.zeros(idx.shape[0], dtype=jnp.float32)

        (loss, (latent_new, eta_new, gn, gn_comp, gn_eta)), grads = (
            loss_grad_fn(
                params,
                latent_init_batch,
                eta_init_batch,
                eta_anchor_batch,
                u_alr_batch,
                n_total_batch,
                counts_batch,
            )
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        latent_loc = latent_loc.at[idx].set(latent_new)
        if eta_loc_arg is not None:
            eta_loc_arg = eta_loc_arg.at[idx].set(eta_new)
        return (
            params,
            opt_state,
            latent_loc,
            eta_loc_arg,
            gn,
            gn_comp,
            gn_eta,
            loss,
        )

    # ---- Outer loop ----
    n_steps = int(laplace_config.n_steps)
    losses: List[float] = []
    rng_step = rng
    display_interval = max(1, n_steps // 100)
    init_loss: Optional[float] = None
    # Running minimum loss for the divergence detector. Initialised
    # to +inf so the climb-check is naturally gated off until the
    # first finite loss is observed.
    min_loss_so_far: float = float("inf")
    latent_loc = z_loc if d_mode == "low_rank" else y_alr_loc

    # ---- Best-snapshot tracking ----
    # When the divergence detector aborts the outer loop, we want
    # to return the *best fit found so far*, not the diverged
    # state. Snapshot ``params``, ``latent_loc``, and ``eta_loc``
    # at every new running-minimum loss. Storing references is safe
    # because every update produces NEW JAX arrays (params via
    # optax.apply_updates, latent_loc via .at[idx].set, eta_loc
    # similarly) — the old arrays stay alive for as long as the
    # snapshot dict references them.
    best_params: Dict[str, jnp.ndarray] = {k: v for k, v in params.items()}
    best_latent_loc = latent_loc
    best_eta_loc = eta_loc
    best_step_idx: int = -1
    best_loss_value: float = float("inf")
    divergence_aborted: bool = False

    progress_reporter = build_progress_reporter(
        progress=progress, progress_backend=progress_backend
    )
    with progress_reporter as reporter:
        capture_str = " + capture" if capture_anchor is not None else ""
        reporter.start(
            description=f"LNM Laplace ({d_mode}{capture_str})",
            total=n_steps,
            completed=0,
            loss_info="init loss: pending",
        )
        for step in range(n_steps):
            rng_step, subkey = random.split(rng_step)
            if batch_size >= n_cells:
                idx = jnp.arange(n_cells)
            else:
                idx = random.choice(
                    subkey, n_cells, shape=(batch_size,), replace=False
                )
            (
                params,
                opt_state,
                latent_loc,
                eta_loc,
                gn,
                gn_comp,
                gn_eta,
                loss,
            ) = update_step(params, opt_state, latent_loc, eta_loc, idx)
            loss_val = float(loss)

            # ---- Divergence guard ----
            # Three failure modes worth aborting on, all caused by
            # one or more cells exploding in their per-cell Newton
            # (ρ near a corner of the simplex, Sherman-Morrison
            # denominator catastrophically cancelling, etc.):
            #   1. ``loss`` becomes NaN or Inf → numerical overflow.
            #   2. ``loss`` *climbs* (moves in the wrong direction)
            #      by more than 0.5 × |init_loss| from the running
            #      minimum. Variational EM loss should be
            #      monotonically decreasing modulo small noise;
            #      a sustained climb is the earliest sign of
            #      single-cell divergence contaminating the
            #      gradient on globals. THIS is the trigger that
            #      the previous "1000× absolute growth" check
            #      missed because that bound is far above the
            #      actual divergence regime.
            #   3. |loss| explodes by > 1000× from |init_loss| →
            #      kept as a hard backstop for the case where the
            #      loss crosses 0 and balloons in a single step
            #      faster than the climb-from-min check can react.
            # Helper: gracefully abort by restoring the best
            # snapshot, logging a constructive warning, and
            # breaking out of the loop. Replaces a previous
            # ``raise RuntimeError`` that threw away the best fit
            # found so far — even when training had reached a
            # *better* minimum than the init loss before
            # diverging.
            def _trigger_divergence_abort(reason: str) -> None:
                nonlocal divergence_aborted
                worst_cell_grad = float(jnp.max(gn_comp))
                if best_step_idx >= 0:
                    snapshot_msg = (
                        f"Restoring best snapshot from step "
                        f"{best_step_idx + 1} (best loss="
                        f"{best_loss_value:.3e}; init loss="
                        f"{init_loss:.3e})."
                    )
                else:
                    snapshot_msg = (
                        "No improved snapshot was recorded before "
                        "the divergence — the result reflects the "
                        "data-driven init."
                    )
                logger.warning(
                    f"LNM Laplace divergence detected at step "
                    f"{step + 1}: {reason} (worst comp grad="
                    f"{worst_cell_grad:.3e}). {snapshot_msg} The "
                    "returned result has ``divergence_aborted=True`` "
                    "so downstream code can detect this case. To "
                    "avoid divergence next time, try (a) increasing "
                    "laplace_config['n_newton_steps'] to 20-30, "
                    "(b) tightening laplace_config['damping'] to "
                    "1e-3 or below, or (c) pre-filtering outlier "
                    "cells. ``result.final_grad_norms`` identifies "
                    "the offending cells in the returned result."
                )
                divergence_aborted = True

            if not np.isfinite(loss_val):
                _trigger_divergence_abort(
                    f"loss became non-finite (loss={loss_val})"
                )
                break
            # Climb-from-min check (primary detector).
            if (
                init_loss is not None
                and len(losses) > 50
                and np.isfinite(min_loss_so_far)
                and np.isfinite(init_loss)
                and (loss_val - min_loss_so_far)
                > 0.5 * abs(init_loss)
            ):
                _trigger_divergence_abort(
                    f"loss={loss_val:.3e} climbed by "
                    f"{(loss_val - min_loss_so_far):.3e} from the "
                    f"running minimum {min_loss_so_far:.3e}, which "
                    f"exceeds 0.5 × |init_loss|="
                    f"{0.5 * abs(init_loss):.3e}"
                )
                break
            # Absolute-magnitude backstop.
            if (
                init_loss is not None
                and len(losses) > 50
                and np.isfinite(init_loss)
                and abs(loss_val) > 1e3 * abs(init_loss)
            ):
                _trigger_divergence_abort(
                    f"|loss|={abs(loss_val):.3e} grew >1000× from "
                    f"|init_loss|={abs(init_loss):.3e}"
                )
                break

            losses.append(loss_val)
            if init_loss is None:
                init_loss = loss_val
                min_loss_so_far = loss_val
            else:
                min_loss_so_far = min(min_loss_so_far, loss_val)

            # Best-snapshot update: record post-step references
            # whenever this step pushed the running minimum down.
            # ``params`` / ``latent_loc`` / ``eta_loc`` are
            # immutable JAX-array dicts that are *replaced* (not
            # mutated) by ``optax.apply_updates`` and
            # ``.at[idx].set`` respectively, so saving references
            # is safe.
            if (
                np.isfinite(loss_val)
                and loss_val == min_loss_so_far
                and loss_val < best_loss_value
            ):
                best_loss_value = loss_val
                best_params = {k: v for k, v in params.items()}
                best_latent_loc = latent_loc
                best_eta_loc = eta_loc
                best_step_idx = step

            step_completed = step + 1
            should_display = (
                step == 0
                or step == n_steps - 1
                or step_completed % display_interval == 0
            )
            if should_display:
                window_start = max(0, len(losses) - display_interval)
                avg_loss = _mean_ignoring_nans(losses[window_start:])
                # Per-block max / p99 / median Newton-grad-norm
                # summary. Reading the three numbers together:
                #   * median: the typical cell — should be small
                #     (well below newton_tolerance) on a healthy fit.
                #   * p99: the 1% worst cells — informative because
                #     L∞ ``max`` can be dominated by a single
                #     outlier cell while p99 reflects the broader
                #     tail.
                #   * max: the strict L∞ across cells — the most
                #     conservative reading, used for the
                #     ``convergence_action`` warning at the end.
                comp_max, comp_p99, comp_med = _grad_summary(gn_comp)
                if capture_anchor is not None:
                    eta_max, eta_p99, eta_med = _grad_summary(gn_eta)
                    grad_info = (
                        f"comp max/p99/med "
                        f"{comp_max:.2e}/{comp_p99:.2e}/{comp_med:.2e}; "
                        f"η max/p99/med "
                        f"{eta_max:.2e}/{eta_p99:.2e}/{eta_med:.2e}"
                    )
                else:
                    grad_info = (
                        f"Newton grad max/p99/med "
                        f"{comp_max:.2e}/{comp_p99:.2e}/{comp_med:.2e}"
                    )
                loss_info = (
                    f"init loss: {init_loss:.4e}, "
                    f"avg. loss [{window_start + 1}-{len(losses)}]: "
                    f"{avg_loss:.4e}, "
                    f"{grad_info}"
                )
                reporter.update(advance=1, loss_info=loss_info)
                if log_progress_lines:
                    print(
                        f"LNM Laplace [{len(losses)}/{n_steps}] " f"{loss_info}"
                    )
            else:
                reporter.update(advance=1)

    # ---- Restore best snapshot when divergence aborted the loop ----
    # If the divergence detector fired, the current ``params`` /
    # ``latent_loc`` / ``eta_loc`` are from the diverged state.
    # Roll back to the snapshot taken at the running-minimum loss so
    # the returned result reflects the best fit found before the
    # divergence — typically a meaningful improvement over the data-
    # driven init even when the run aborted early.
    if divergence_aborted and best_step_idx >= 0:
        params = best_params
        latent_loc = best_latent_loc
        eta_loc = best_eta_loc

    # ---- Final convergence check (full-data Newton sweep) ----
    mu_f = jax.lax.stop_gradient(params["mu"])
    W_f = jax.lax.stop_gradient(params["W"])
    d_f = jax.lax.stop_gradient(jnp.exp(params["d_log"]))
    if d_mode == "low_rank":
        latent_final, gn_final = laplace_newton_batch_z(
            latent_loc,
            u_alr,
            n_total_per_cell,
            mu_f,
            W_f,
            alr_reference_idx,
            n_genes,
            max(2 * n_newton, 10),
            damping,
        )
        z_loc_final = latent_final
        y_alr_loc_final = None
    else:
        latent_final, gn_final = laplace_newton_batch_y_alr(
            latent_loc,
            u_alr,
            n_total_per_cell,
            mu_f,
            W_f,
            d_f,
            alr_reference_idx,
            n_genes,
            max(2 * n_newton, 10),
            damping,
        )
        z_loc_final = None
        y_alr_loc_final = latent_final

    # Capture-anchor final Newton sweep (LNMVCP only).
    if capture_anchor is not None:
        mu_T_f = jnp.exp(jax.lax.stop_gradient(params["log_mu_T"]))
        r_T_f = jnp.exp(jax.lax.stop_gradient(params["log_r_T"]))
        eta_loc_final, gn_eta_final = laplace_newton_batch_eta(
            eta_loc,
            n_total_per_cell,
            r_T_f,
            mu_T_f,
            eta_anchor_per_cell,
            sigma_M,
            max(2 * n_newton, 10),
            damping,
        )
        gn_final = jnp.maximum(gn_final, gn_eta_final)
    else:
        eta_loc_final = None

    # Convergence-action handling.
    max_gn = float(jnp.max(gn_final))
    if max_gn > laplace_config.newton_tolerance:
        offending = int(jnp.sum(gn_final > laplace_config.newton_tolerance))
        msg = (
            f"LNM Laplace Newton: {offending}/{n_cells} cells did "
            f"not converge below tolerance="
            f"{laplace_config.newton_tolerance:.1e} "
            f"(worst grad-norm={max_gn:.3e})."
        )
        if laplace_config.convergence_action == "raise":
            raise RuntimeError(msg)
        elif laplace_config.convergence_action == "warn":
            logger.warning(msg)

    # ---- Pack the results. ``globals`` carries (mu, W, d_log) for
    # the composition block, and (log_mu_T, log_r_T) when capture is
    # active. The bridge in inference/laplace.py exponentiates the
    # log-transformed scalars and routes z_loc / y_alr_loc /
    # eta_loc to the right fields on ScribeLaplaceResults.
    return LaplaceRunResult(
        globals=params,
        # Reuse the ``x_loc`` slot for transit of the composition
        # latent; the bridge unpacks based on base_model + d_mode.
        x_loc=z_loc_final if d_mode == "low_rank" else y_alr_loc_final,
        # ``eta_loc`` is the LNMVCP capture-offset MAP (or None for
        # plain LNM); reuses the same slot PLN uses for capture.
        eta_loc=eta_loc_final,
        final_grad_norms=gn_final,
        losses=jnp.asarray(losses, dtype=jnp.float32),
        n_steps_run=len(losses),
        model_config=model_config,
        early_stopped=divergence_aborted,
        best_loss=(
            best_loss_value if best_step_idx >= 0 else float("inf")
        ),
        stopped_at_step=len(losses),
        divergence_aborted=divergence_aborted,
    )


__all__ = ["LaplaceInferenceEngine", "LaplaceRunResult"]

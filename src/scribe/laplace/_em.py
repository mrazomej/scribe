"""Generic Laplace-EM driver and observation-model protocol.

This module factors out the scaffolding shared by every Laplace fit in
scribe — outer Adam, mini-batching, divergence detection, best-snapshot
recording, progress reporting, smoothed-loss-based patience early
stopping, orbax checkpoint save/resume, and the final convergence
check — so that adding a new observation model (NB-LogNormal, NBLN
mixture, future zero-inflated variants, …) reduces to implementing a
small protocol class. The rest of this module is the driver:

    run_laplace_em(obs_model, count_data, latent_dim, ...)

Per-model hooks live in a :class:`LaplaceObservationModel` subclass.
Each subclass owns the four model-specific pieces:

1. ``init_state``: build ``params`` (the optax-optimised globals),
   per-cell ``latent_loc`` (the inner-Newton state), optional
   ``eta_loc`` (capture offset), and any ``aux_data`` (e.g., LNM's
   ALR-coordinate counts).
2. ``loss_fn``: compute the negative Laplace ELBO on a batch of cells.
   Internally calls the model's Newton kernel, computes ``log det(-H)``
   at live globals, returns ``(loss, (latent_new, eta_new, gn))``.
3. ``final_sweep``: run a final Newton pass on **all** cells (not just
   a batch) at convergence to get per-cell gradient norms used for
   diagnostics.
4. ``pack_result``: assemble the final :class:`LaplaceRunResult` —
   the only model-specific touch is selecting which latent goes into
   ``x_loc`` (PLN/NBLN: ``x``; LNM low-rank: ``z``; LNM learned-d:
   ``y_alr``).

Shared scaffolding behaviours
-----------------------------
- **Outer Adam**: optax ``GradientTransformation`` resolved from
  ``laplace_config.optimizer`` then ``laplace_config.optimizer_config``
  then a default ``adam(1e-3)``.
- **Mini-batching**: uniform random subsampling with the standard
  Hoffman-2013 ``N/B`` ELBO scale factor; ``batch_size=None`` means
  full-batch (and ``data_scale=1.0``, byte-identical to no scaling).
- **Divergence guards** (after step 50, the warm-up window):
    1. Loss-finite: abort if ``loss`` is NaN or ±inf.
    2. Climb-from-min: abort if the loss has climbed more than
       ``0.5 × |init_loss|`` above its running minimum.
    3. Absolute magnitude: abort if ``|loss|`` exceeds
       ``1000 × |init_loss|`` — last-line backstop.
    4. Drift-above-initial: when
       ``early_stopping.drift_above_initial_pct`` is set (default
       ``1.0`` = 1%), abort if the *smoothed* loss stays above
       ``init_loss + (pct/100) × |init_loss|`` for
       ``early_stopping.drift_patience`` consecutive steps.  Targets
       the failure mode where the outer Adam wanders into a region
       where the inner Newton's Laplace approximation degrades and
       the loop feeds itself — characteristic of non-amortized M-step
       training.
- **Best-snapshot recording**: independent of early-stopping config,
  the driver always tracks the best ``params``, ``latent_loc``, and
  ``eta_loc`` seen so far. Restores from this snapshot if any
  divergence guard fires, so callers always get a usable result.
- **Patience-based early stopping** (when
  ``laplace_config.early_stopping.enabled``): smoothed-loss
  improvement tracking with ``min_delta`` / ``min_delta_pct`` and
  ``patience`` after a configurable warmup. When
  ``restore_best=True`` (default), the best snapshot is restored at
  the end regardless of whether early stopping fired.
- **Orbax checkpoint save/resume**: when
  ``early_stopping.checkpoint_dir`` is set, the driver saves
  ``(params, opt_state, latent_loc, eta_loc, step, best_loss, losses,
  patience_counter)`` every ``checkpoint_every`` steps and resumes
  from the latest checkpoint on the next call when
  ``early_stopping.resume`` is True.
- **Final convergence check**: ``model.final_sweep`` is called on the
  full cell population at the end with ``2 × n_newton_steps`` Newton
  iterations. The final per-cell gradient-infinity norms drive the
  ``convergence_action`` (``"raise"`` / ``"warn"`` / ``"ignore"``).

What's *not* yet in the generic driver
--------------------------------------
- Per-block grad-norm split for the progress display (the legacy
  engine showed separate composition + η gradient histograms; the
  generic driver shows the joint norm only).

This feature gap will be addressed when LNM is migrated onto the
generic driver — the per-block split is naturally LNM-specific and
will be an optional hook on the protocol.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

from ..models.config import EarlyStoppingConfig, LaplaceConfig, ModelConfig
from ..svi._progress_backend import ProgressBackendName, build_progress_reporter
from .checkpoint import (
    laplace_checkpoint_exists,
    load_laplace_checkpoint,
    save_laplace_checkpoint,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Result and state dataclasses
# =====================================================================


@dataclass
class LaplaceRunResult:
    """Engine-agnostic Laplace-EM result.

    Attributes
    ----------
    globals : dict
        Final optimised global parameters.  Keys are model-specific
        (PLN: ``{"mu", "W", "d_loc"}``; NBLN: ``{"mu", "W", "d_loc",
        "r_loc"}``; LNM: ``{"mu", "W", "d_loc", "mu_T_loc",
        "r_T_loc"}``).  Unconstrained positive-parameter coordinates
        are named ``*_loc`` and mapped to constrained space by the
        configured ``model_config.positive_transform``.
    x_loc : jnp.ndarray
        Per-cell latent MAP. Shape and semantics are model-specific:
        ``(N, G)`` for PLN/NBLN (log-rate), ``(N, k)`` for LNM low-rank
        (latent code), ``(N, G-1)`` for LNM learned (ALR coordinates).
    eta_loc : jnp.ndarray or None
        Per-cell capture offset MAP, ``(N,)``, or None when the
        capture anchor is off.
    final_grad_norms : jnp.ndarray
        Per-cell L-inf Newton gradient norm at the end of training,
        ``(N,)``. Used by ``convergence_action`` and downstream
        diagnostics.
    losses : jnp.ndarray
        Per-step loss history, ``(n_steps_run,)``.
    n_steps_run : int
        Number of outer Adam steps actually executed.
    model_config : ModelConfig
        Provenance.
    early_stopped : bool
        True if the loop terminated before ``n_steps`` (early stopping
        or divergence recovery).
    best_loss : float
        Smallest *smoothed* loss seen over the run (post-warm-up).
    stopped_at_step : int
        Index of the step at which the loop terminated.
    divergence_aborted : bool
        True if the divergence guard restored a best snapshot and
        terminated the loop.
    global_uncertainty : dict
        Post-fit Laplace posterior approximation for selected global
        parameters in unconstrained space.  Populated by each
        observation model's ``compute_global_uncertainty`` hook after
        the final Newton sweep.  Keys are model-specific:

        - NBLN: ``r_loc``, ``r_scale``, and diagnostics.
        - LNM/LNMVCP: ``totals_loc``, ``totals_cov``, ``totals_scale``,
          plus marginal aliases ``mu_T_loc/scale``, ``r_T_loc/scale``.
        - PLN: empty dict (no extra globals to approximate).

        Empty dict when the observation model does not implement the
        hook.
    """

    globals: Dict[str, jnp.ndarray]
    x_loc: jnp.ndarray
    eta_loc: Optional[jnp.ndarray]
    final_grad_norms: jnp.ndarray
    losses: jnp.ndarray
    n_steps_run: int
    model_config: Optional[ModelConfig]
    early_stopped: bool
    best_loss: float
    stopped_at_step: int
    divergence_aborted: bool
    global_uncertainty: Dict[str, jnp.ndarray]
    # Phase-2 freeze: subset of {"r", "mu", "eta"} the obs model
    # excluded from the optimizer.  Read by ``_format_laplace_results``
    # so the final ``ScribeLaplaceResults`` carries the same marker.
    # Empty frozenset for non-NBLN base models and for unfrozen fits.
    frozen_params: frozenset = field(default_factory=frozenset)
    # Phase-3 W-shrinkage prior diagnostics.  Populated by the obs
    # model's ``pack_result`` when a w_prior strategy is configured.
    # Always present (the no-op NoneWPrior populates
    # ``{"strategy_type": "none", "column_frobenius_compositional": …}``);
    # ``None`` only when the obs model itself doesn't run the W-prior
    # integration (e.g. LNM-family in v1).
    w_prior_diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class InitState:
    """Bundle returned by :meth:`LaplaceObservationModel.init_state`."""

    params: Dict[str, jnp.ndarray]
    latent_loc: jnp.ndarray
    eta_loc: Optional[jnp.ndarray]
    eta_anchor: Optional[jnp.ndarray]
    aux_data: Dict[str, jnp.ndarray]


@dataclass
class FinalSweepResult:
    """Bundle returned by :meth:`LaplaceObservationModel.final_sweep`."""

    latent_loc: jnp.ndarray
    eta_loc: Optional[jnp.ndarray]
    final_grad_norms: jnp.ndarray


# =====================================================================
# Observation-model protocol
# =====================================================================


class LaplaceObservationModel(ABC):
    """Per-model hooks plugged into :func:`run_laplace_em`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used in progress messages and error contexts."""

    @abstractmethod
    def init_state(
        self,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        latent_dim: int,
        seed: int,
    ) -> InitState:
        """Build initial ``params``, per-cell latents, and aux data."""

    @abstractmethod
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
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]]:
        """Compute the negative Laplace ELBO on one batch.

        Returns
        -------
        loss : jnp.ndarray, scalar
        aux : tuple
            ``(latent_new, eta_new, gn_blocks)`` where ``gn_blocks`` is
            a dict mapping block name (e.g. ``"x"``, ``"η"``,
            ``"comp"``) to per-cell ``L∞`` gradient norms of shape
            ``(batch,)``.  Single-block models return a one-element
            dict; LNMVCP and PLN/NBLN with capture return two blocks.
            The driver uses the dict for the per-block progress
            display and reduces over blocks for divergence detection.
        """

    @abstractmethod
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
        """Run a final Newton sweep over all cells and return MAP + grad."""

    @abstractmethod
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
        """Pack into the canonical result dataclass."""

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
        """Compute post-fit Laplace uncertainty for model-specific globals.

        Called after the final Newton sweep, at the restored best
        parameters and converged per-cell MAPs.  Returns a dictionary of
        uncertainty quantities in unconstrained space.  The default
        implementation returns an empty dict (no globals to approximate,
        as in PLN).

        Parameters
        ----------
        params : dict
            Final optimised global parameters.
        latent_loc : jnp.ndarray
            Per-cell latent MAPs from the final sweep.
        eta_loc : jnp.ndarray or None
            Per-cell capture offset MAPs, or None.
        eta_anchor : jnp.ndarray or None
            Capture-anchor ``(log_M_0, sigma_M)`` per cell, or None.
        count_data : jnp.ndarray
            Full observed count matrix, shape ``(N, G)``.
        aux_data : dict
            Auxiliary data (e.g. log-library sizes).
        model_config : ModelConfig
            Model configuration, including ``positive_transform``.

        Returns
        -------
        dict
            Model-specific uncertainty fields.  NBLN returns ``r_loc``,
            ``r_scale``, and diagnostics.  LNM returns ``totals_loc``,
            ``totals_cov``, ``totals_scale``, and marginal aliases.
            PLN returns ``{}``.
        """
        return {}

    def aux_batch_slice(
        self, aux_data: Dict[str, jnp.ndarray], idx: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Slice per-cell entries in ``aux_data`` by mini-batch ``idx``.

        Default: slice every value whose first axis matches ``n_cells``
        and pass through anything else unchanged.
        """
        if not aux_data:
            return aux_data
        n_cells = int(idx.shape[0]) if idx.ndim else 0
        out: Dict[str, jnp.ndarray] = {}
        for k, v in aux_data.items():
            if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] >= n_cells:
                out[k] = v[idx]
            else:
                out[k] = v
        return out


# =====================================================================
# Optimizer and helpers
# =====================================================================


def _build_optimizer(laplace_config: LaplaceConfig):
    """Resolve the outer-loop optax optimizer from the config.

    Mirrors :func:`scribe.inference.optimizer_factory.build_optimizer_from_config`
    but returns a native ``optax.GradientTransformation``.  Reads the
    structured Pydantic ``OptimizerConfig`` schema directly (explicit
    fields ``name``, ``step_size``, ``b1``, ``b2``, ``eps``,
    ``grad_clip_norm``, ``weight_decay``, ``momentum``) -- *not* a
    free-form ``kwargs`` dict.

    Resolution order:
        1. ``laplace_config.optimizer`` (a pre-built
           ``optax.GradientTransformation``) -- power-user override.
        2. ``laplace_config.optimizer_config`` (structured spec).
        3. Default ``optax.adam(1e-3)``.
    """
    if laplace_config.optimizer is not None:
        return laplace_config.optimizer

    optim_cfg = laplace_config.optimizer_config
    if optim_cfg is None:
        return optax.adam(1e-3)

    name = optim_cfg.name
    lr = optim_cfg.step_size if optim_cfg.step_size is not None else 1e-3
    b1 = optim_cfg.b1 if optim_cfg.b1 is not None else 0.9
    b2 = optim_cfg.b2 if optim_cfg.b2 is not None else 0.999
    eps = optim_cfg.eps if optim_cfg.eps is not None else 1e-8
    grad_clip_norm = optim_cfg.grad_clip_norm
    weight_decay = optim_cfg.weight_decay

    if name == "adam":
        opt = optax.adam(lr, b1=b1, b2=b2, eps=eps)
    elif name == "clipped_adam":
        # Chain a global-norm clip with Adam -- same semantics as
        # ``numpyro.optim.ClippedAdam`` and the SVI/VAE-mode default.
        clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else 1.0
        opt = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optax.adam(lr, b1=b1, b2=b2, eps=eps),
        )
    elif name == "adagrad":
        opt = optax.adagrad(lr, eps=eps)
    elif name == "rmsprop":
        opt = optax.rmsprop(lr, decay=b2, eps=eps)
    elif name == "sgd":
        opt = optax.sgd(lr)
    elif name == "momentum":
        momentum = (
            optim_cfg.momentum if optim_cfg.momentum is not None else 0.9
        )
        opt = optax.sgd(lr, momentum=float(momentum))
    else:
        raise ValueError(
            f"Unsupported optimizer name {name!r} for Laplace."
        )

    if weight_decay is not None:
        opt = optax.chain(opt, optax.add_decayed_weights(weight_decay))

    if grad_clip_norm is not None and name != "clipped_adam":
        # Match ``optimizer_factory.build_optimizer_from_config``:
        # ``grad_clip_norm`` is currently supported only with
        # ``clipped_adam`` for the Laplace path.
        raise ValueError(
            "grad_clip_norm is currently supported only with "
            "optimizer name 'clipped_adam' for Laplace inference."
        )

    return opt


def _mean_ignoring_nans(values) -> float:
    """Mean of ``values`` with NaNs filtered out; ``inf`` if all NaN."""
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return float("inf")
    return float(arr[mask].mean())


# =====================================================================
# Driver
# =====================================================================


def run_laplace_em(
    obs_model: LaplaceObservationModel,
    count_data: jnp.ndarray,
    n_cells: int,
    n_genes: int,
    latent_dim: int,
    laplace_config: LaplaceConfig,
    model_config: ModelConfig,
    seed: int = 42,
    progress: bool = True,
    progress_backend: ProgressBackendName = "auto",
    log_progress_lines: bool = False,
) -> LaplaceRunResult:
    """Generic Laplace-EM training loop.

    Outer Adam on globals; inner Newton (per-cell) provided by the
    observation model. See module docstring for the full set of
    behaviours.
    """
    counts = jnp.asarray(count_data, dtype=jnp.float32)
    rng = random.PRNGKey(seed)

    # ---- Per-model state (params, per-cell latents, aux data) ----
    init = obs_model.init_state(
        count_data=counts,
        n_cells=n_cells,
        n_genes=n_genes,
        latent_dim=latent_dim,
        seed=seed,
    )
    params = init.params
    latent_loc = init.latent_loc
    eta_loc = init.eta_loc
    eta_anchor = init.eta_anchor
    aux_data = init.aux_data

    # ---- Outer optimiser ----
    opt = _build_optimizer(laplace_config)
    opt_state = opt.init(params)

    # ---- Hyperparameters ----
    n_steps = int(laplace_config.n_steps)
    n_newton = int(laplace_config.n_newton_steps)
    damping = float(laplace_config.damping)
    batch_size = laplace_config.batch_size
    if batch_size is None:
        batch_size = n_cells
    batch_size = int(batch_size)
    data_scale = float(n_cells) / float(batch_size)

    # ---- Early-stopping / checkpointing config ----
    early_stopping: EarlyStoppingConfig = (
        laplace_config.early_stopping
        if laplace_config.early_stopping is not None
        else EarlyStoppingConfig(enabled=False)
    )
    checkpoint_dir = early_stopping.checkpoint_dir

    # ---- JIT the loss + gradient ----
    loss_grad_fn = jax.jit(
        jax.value_and_grad(obs_model.loss_fn, has_aux=True),
        static_argnames=("data_scale", "n_newton", "damping"),
    )

    @jax.jit
    def update_step(params, opt_state, latent_loc, eta_loc, idx):
        counts_batch = counts[idx]
        latent_batch_init = latent_loc[idx]
        if eta_loc is not None:
            eta_batch_init = eta_loc[idx]
            eta_anchor_batch = eta_anchor[idx]
        else:
            eta_batch_init = jnp.zeros(idx.shape[0])
            eta_anchor_batch = jnp.zeros(idx.shape[0])
        aux_batch = obs_model.aux_batch_slice(aux_data, idx)

        (loss, (latent_new, eta_new, gn_blocks)), grads = loss_grad_fn(
            params,
            latent_batch_init,
            eta_batch_init,
            counts_batch,
            eta_anchor_batch,
            aux_batch,
            data_scale,
            n_newton,
            damping,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        latent_loc = latent_loc.at[idx].set(latent_new)
        if eta_loc is not None:
            eta_loc = eta_loc.at[idx].set(eta_new)
        return params, opt_state, latent_loc, eta_loc, gn_blocks, loss

    # ---- Bookkeeping ----
    losses: List[float] = []
    start_step = 0
    init_loss: Optional[float] = None
    init_loss_signed: Optional[float] = None
    min_loss_so_far: float = float("inf")
    display_interval = max(1, n_steps // 100)
    # Drift-above-initial guard: counter accumulates by ``check_every``
    # on each check where ``smoothed_loss`` exceeds the threshold, and
    # resets to zero on any check that drops back below threshold.
    # Same units as ``patience_counter`` so the comparison against
    # ``drift_patience`` mirrors the standard early-stopping semantics.
    drift_counter: int = 0

    # Best-snapshot for divergence recovery (always on).
    best_params_div: Dict[str, jnp.ndarray] = {k: v for k, v in params.items()}
    best_latent_div = latent_loc
    best_eta_div = eta_loc
    best_step_idx_div: int = -1
    best_loss_value_div: float = float("inf")
    divergence_aborted: bool = False

    # Early-stopping snapshot + tracker (smoothed loss).
    best_loss = float("inf")
    patience_counter = 0
    best_state: Optional[Dict[str, Any]] = None
    best_step = 0
    early_stopped = False
    eps_div = 1e-8  # divide-by-zero guard for percentage-based delta

    # ---- Resume from checkpoint if requested ----
    resumed = False
    resume_message: Optional[str] = None
    if (
        checkpoint_dir
        and early_stopping.resume
        and laplace_checkpoint_exists(checkpoint_dir)
    ):
        target_state = {
            "params": params,
            "opt_state": opt_state,
            "x_loc": latent_loc,
            # Orbax cannot serialise None; saver substitutes a zero
            # placeholder when capture is off and the metadata's
            # ``has_eta_loc`` flag controls whether load returns None.
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
            latent_loc = restored_state["x_loc"]
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
            # Restore running minimum from the loaded loss history so
            # the divergence detector operates against the true min,
            # not the post-resume one.  ``init_loss`` is the magnitude
            # used for the climb-from-min threshold and must be
            # ``abs(loss)`` (losses are typically negative; using the
            # signed value would produce a negative threshold and
            # spurious divergence aborts).
            if losses:
                init_loss = abs(float(losses[0]))
                init_loss_signed = float(losses[0])
                _finite_losses = [l for l in losses if np.isfinite(l)]
                if _finite_losses:
                    min_loss_so_far = float(min(_finite_losses))

    last_checkpoint_step = (
        start_step if resumed else -early_stopping.checkpoint_every
    )

    # ---- Progress reporter ----
    progress_reporter = build_progress_reporter(
        progress=progress, progress_backend=progress_backend
    )

    with progress_reporter as reporter:
        init_loss_display = (
            f"{losses[0]:.4e}" if losses else "pending"
        )
        reporter.start(
            description=(
                f"Laplace ({obs_model.name})"
                + (" (resumed)" if resumed else "")
            ),
            total=n_steps,
            completed=start_step,
            loss_info=f"init loss: {init_loss_display}",
        )
        if resume_message is not None:
            reporter.print_message(resume_message)

        for step in range(start_step, n_steps):
            # Mini-batch sampling.
            if batch_size < n_cells:
                rng, sub = random.split(rng)
                idx = random.choice(
                    sub, n_cells, shape=(batch_size,), replace=False
                )
            else:
                idx = jnp.arange(n_cells)

            (
                params,
                opt_state,
                latent_loc,
                eta_loc,
                gn_blocks,
                loss,
            ) = update_step(params, opt_state, latent_loc, eta_loc, idx)

            loss_val = float(loss)
            losses.append(loss_val)
            step_completed = step + 1

            if init_loss is None and np.isfinite(loss_val):
                init_loss = abs(loss_val)
                init_loss_signed = float(loss_val)

            # Best-snapshot (raw loss) for divergence recovery.
            if loss_val < best_loss_value_div:
                best_params_div = {k: v for k, v in params.items()}
                best_latent_div = latent_loc
                best_eta_div = eta_loc
                best_step_idx_div = step
                best_loss_value_div = loss_val
            if loss_val < min_loss_so_far:
                min_loss_so_far = loss_val

            # ---- Divergence guards (warm-up: first 50 steps) ----
            if step >= 50 and init_loss is not None:
                if not np.isfinite(loss_val):
                    logger.warning(
                        "Laplace[%s]: divergence at step %d "
                        "(loss=%s, non-finite). Restoring best snapshot.",
                        obs_model.name, step, loss_val,
                    )
                    divergence_aborted = True
                    early_stopped = True
                    break
                climb = loss_val - min_loss_so_far
                if climb > 0.5 * init_loss:
                    logger.warning(
                        "Laplace[%s]: divergence at step %d "
                        "(loss climbed %.3e above running min, threshold "
                        "%.3e). Restoring best snapshot.",
                        obs_model.name, step, climb, 0.5 * init_loss,
                    )
                    divergence_aborted = True
                    early_stopped = True
                    break
                if abs(loss_val) > 1000.0 * init_loss:
                    logger.warning(
                        "Laplace[%s]: divergence at step %d "
                        "(|loss|=%.3e > 1000x|init_loss|=%.3e). "
                        "Restoring best snapshot.",
                        obs_model.name, step, abs(loss_val),
                        1000.0 * init_loss,
                    )
                    divergence_aborted = True
                    early_stopped = True
                    break

            # ---- Periodic progress update ----
            display_step = (
                step == start_step
                or step_completed % display_interval == 0
                or step == n_steps - 1
            )
            if display_step:
                window_start = max(
                    0,
                    len(losses) - max(1, early_stopping.smoothing_window),
                )
                avg_loss = _mean_ignoring_nans(losses[window_start:])
                init_loss_str = (
                    f"{init_loss:.4e}" if init_loss is not None else "N/A"
                )
                # Per-block ``L∞`` summary across cells -- one
                # ``max/p99/med`` triplet per Newton block (e.g.,
                # ``x``, ``η`` for PLN/NBLN with capture; ``comp``,
                # ``η`` for LNMVCP).  Single-block models show a
                # single triplet.  Mirrors the legacy progress bar so
                # callers can see at a glance which Newton block is
                # the convergence bottleneck.
                grad_info = "; ".join(
                    f"{name} max/p99/med "
                    f"{float(jnp.max(g)):.2e}/"
                    f"{float(jnp.percentile(g, 99)):.2e}/"
                    f"{float(jnp.median(g)):.2e}"
                    for name, g in gn_blocks.items()
                )
                loss_info = (
                    f"init loss: {init_loss_str}, "
                    f"avg. loss [{window_start + 1}-{len(losses)}]: "
                    f"{avg_loss:.4e}, {grad_info}"
                )
                reporter.update(advance=1, loss_info=loss_info)
                if log_progress_lines:
                    print(
                        f"Laplace progress [{len(losses)}/{n_steps}] "
                        f"{loss_info}"
                    )
            else:
                reporter.update(advance=1)

            # ---- Early-stopping + checkpointing block ----
            should_check = (
                step_completed % early_stopping.check_every == 0
                and len(losses) >= early_stopping.smoothing_window
            )
            if should_check:
                window_start = max(
                    0, len(losses) - early_stopping.smoothing_window
                )
                smoothed_loss = _mean_ignoring_nans(losses[window_start:])
                past_warmup = step >= early_stopping.warmup

                # ---- Drift-above-initial guard (orthogonal to the per-
                # step NaN/jump/magnitude detector and to early
                # stopping; runs whenever the smoothed loss is finite
                # and we have an initial reference).  Default
                # ``drift_above_initial_pct=1.0`` means a 1% climb
                # above the *signed* initial loss, sustained for
                # ``drift_patience`` steps, triggers the abort.  The
                # divergence-recovery snapshot (always tracked) is
                # restored after the loop, so the caller still gets
                # the best parameters seen.
                drift_pct = early_stopping.drift_above_initial_pct
                if (
                    drift_pct is not None
                    and past_warmup
                    and init_loss_signed is not None
                    and np.isfinite(smoothed_loss)
                ):
                    drift_threshold = init_loss_signed + (
                        drift_pct / 100.0
                    ) * abs(init_loss_signed)
                    if smoothed_loss > drift_threshold:
                        drift_counter += early_stopping.check_every
                    else:
                        drift_counter = 0
                    if drift_counter >= early_stopping.drift_patience:
                        logger.warning(
                            "Laplace[%s]: drift-above-initial at step %d "
                            "(smoothed_loss=%.4e exceeded threshold=%.4e "
                            "for %d consecutive steps; init_loss=%.4e, "
                            "drift_above_initial_pct=%.3f%%). Restoring "
                            "best snapshot.",
                            obs_model.name,
                            step,
                            smoothed_loss,
                            drift_threshold,
                            drift_counter,
                            init_loss_signed,
                            drift_pct,
                        )
                        reporter.print_message(
                            f"Drift-above-initial guard triggered at step "
                            f"{step + 1}: smoothed loss "
                            f"{smoothed_loss:.4e} > threshold "
                            f"{drift_threshold:.4e} (init "
                            f"{init_loss_signed:.4e}, "
                            f"+{drift_pct:.2f}%) for {drift_counter} "
                            f"steps. Restoring best snapshot."
                        )
                        divergence_aborted = True
                        early_stopped = True
                        break

                should_track = past_warmup and (
                    early_stopping.enabled or early_stopping.restore_best
                )

                if should_track:
                    if not np.isfinite(best_loss):
                        best_loss = smoothed_loss
                        best_step = step
                        if early_stopping.restore_best:
                            best_state = {
                                "params": params,
                                "opt_state": opt_state,
                                "x_loc": latent_loc,
                                "eta_loc": eta_loc,
                            }
                        patience_counter = 0
                    else:
                        improvement = best_loss - smoothed_loss
                        if early_stopping.min_delta_pct is not None:
                            denom = max(abs(best_loss), eps_div)
                            improvement_pct = 100.0 * improvement / denom
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
                                    "x_loc": latent_loc,
                                    "eta_loc": eta_loc,
                                }
                            patience_counter = 0
                        else:
                            patience_counter += early_stopping.check_every

                # Periodic checkpoint save.
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
                        x_loc=latent_loc,
                        eta_loc=eta_loc,
                        step=step,
                        best_loss=best_loss,
                        losses=losses,
                        patience_counter=patience_counter,
                    )
                    last_checkpoint_step = step

                # Trigger early stopping on patience exceeded.
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
    if early_stopping.restore_best and best_state is not None:
        params = best_state["params"]
        opt_state = best_state["opt_state"]
        latent_loc = best_state["x_loc"]
        eta_loc = best_state["eta_loc"]

    # ---- Restore divergence-best if a guard fired ----
    if divergence_aborted and best_step_idx_div >= 0:
        params = best_params_div
        latent_loc = best_latent_div
        eta_loc = best_eta_div
        logger.warning(
            "Laplace[%s]: restored best snapshot from step %d "
            "(best loss: %.4e).",
            obs_model.name, best_step_idx_div + 1, best_loss_value_div,
        )

    # ---- Final convergence check ----
    final = obs_model.final_sweep(
        params=params,
        latent_loc=latent_loc,
        eta_loc=eta_loc,
        eta_anchor=eta_anchor,
        count_data=counts,
        aux_data=aux_data,
        n_newton=max(2 * n_newton, 10),
        damping=damping,
    )

    max_gn = float(jnp.max(final.final_grad_norms))
    if max_gn > laplace_config.newton_tolerance:
        offending = int(
            jnp.sum(final.final_grad_norms > laplace_config.newton_tolerance)
        )
        msg = (
            f"Laplace[{obs_model.name}]: {offending}/{n_cells} cells "
            f"did not converge below tolerance="
            f"{laplace_config.newton_tolerance:.1e} "
            f"(worst grad-norm={max_gn:.3e})."
        )
        action = laplace_config.convergence_action
        if action == "raise":
            raise RuntimeError(msg)
        if action == "warn":
            logger.warning(msg)

    # ---- Post-fit global Laplace uncertainty ----
    # Compute Hessian-based posterior approximations for model-specific
    # global parameters (e.g. NBLN r_g, LNM totals) at the restored
    # best state and converged per-cell MAPs.
    global_uncertainty = obs_model.compute_global_uncertainty(
        params=params,
        latent_loc=final.latent_loc,
        eta_loc=final.eta_loc,
        eta_anchor=eta_anchor,
        count_data=counts,
        aux_data=aux_data,
        model_config=model_config,
    )

    return obs_model.pack_result(
        params=params,
        final=final,
        losses=np.asarray(losses, dtype=np.float64),
        n_steps_run=len(losses),
        model_config=model_config,
        early_stopped=early_stopped or divergence_aborted,
        best_loss=(
            best_loss
            if np.isfinite(best_loss)
            else (float(losses[-1]) if losses else float("inf"))
        ),
        stopped_at_step=len(losses),
        divergence_aborted=divergence_aborted,
        global_uncertainty=global_uncertainty,
    )

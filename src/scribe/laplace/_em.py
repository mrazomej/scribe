"""Generic Laplace-EM driver and observation-model protocol.

This module factors out the scaffolding shared by every Laplace fit in scribe —
outer Adam, mini-batching, divergence detection, best- snapshot recording,
progress reporting, and the final convergence check — so that adding a new
observation model (NB-LogNormal, NBLN mixture, future zero-inflated variants, …)
reduces to implementing a small protocol class. The rest of this module is the
driver:

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
   ``x_loc`` (PLN: ``x``; LNM low-rank: ``z``; LNM learned-d: ``y_alr``;
   NBLN: ``x``).

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
- **Best-snapshot recording**: independent of early-stopping config,
  the driver always tracks the best ``params``, ``latent_loc``, and
  ``eta_loc`` seen so far. Restores from this snapshot if any
  divergence guard fires, so callers always get a usable result.
- **Final convergence check**: ``model.final_sweep`` is called on the
  full cell population at the end with ``2 × n_newton_steps`` Newton
  iterations. The final per-cell gradient-infinity norms drive the
  ``convergence_action`` (``"raise"`` / ``"warn"`` / ``"ignore"``).

What's *not* in this MVP driver
-------------------------------
- Orbax checkpoint save/resume (TODO: lift from ``engine.py``).
- Patience-based early stopping (TODO: lift from ``engine.py``).
- Per-block grad split for the progress display (the existing engine
  shows separate composition + η gradient histograms; the MVP driver
  shows the joint norm only).

These features remain available through the legacy PLN/LNM
orchestrations in ``engine.py`` which haven't been migrated yet. They
will be lifted into the driver in a follow-up.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax

from ..models.config import LaplaceConfig, ModelConfig
from ..svi._progress_backend import ProgressBackendName, build_progress_reporter

logger = logging.getLogger(__name__)


# =====================================================================
# Result and state dataclasses
# =====================================================================


@dataclass
class LaplaceRunResult:
    """Engine-agnostic Laplace-EM result.

    Identical to the legacy :class:`engine.LaplaceRunResult` — kept
    here so the generic driver does not depend on the legacy module.

    Attributes
    ----------
    globals : dict
        Final optimised global parameters. Keys are model-specific
        (PLN: ``{"mu", "W", "d_log"}``; NBLN: ``{"mu", "W", "d_log",
        "log_r"}``; LNM: ``{"mu", "W", "d_log", "log_mu_T",
        "log_r_T"}``).
    x_loc : jnp.ndarray
        Per-cell latent MAP. Shape and semantics are model-specific:
        ``(N, G)`` for PLN/NBLN (log-rate), ``(N, k)`` for LNM low-rank
        (latent code), ``(N, G-1)`` for LNM learned (ALR coordinates).
    eta_loc : jnp.ndarray or None
        Per-cell capture offset MAP, ``(N,)``, or None when the
        capture anchor is off.
    final_grad_norms : jnp.ndarray
        Per-cell L∞ Newton gradient norm at the end of training,
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
        Smallest loss seen over the run (post-warm-up).
    stopped_at_step : int
        Index of the step at which the loop terminated.
    divergence_aborted : bool
        True if the divergence guard restored a best snapshot and
        terminated the loop.
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
    """Per-model hooks plugged into :func:`run_laplace_em`.

    Subclasses describe a specific observation channel (PLN: per-gene
    Poisson on log-normal rates; LNM: ALR Multinomial + NB totals;
    NBLN: per-gene NB on log-normal-modulated means). The driver
    handles all shared scaffolding.

    Subclasses are instantiated once per fit call (typically with
    ``capture_anchor`` and ``model_config`` resolved from the API
    layer), then their hooks are dispatched from inside the driver.
    """

    # --- Identity --------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name used in progress messages and error contexts."""

    # --- State construction ---------------------------------------------

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

    # --- Loss closure ----------------------------------------------------

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
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Compute the negative Laplace ELBO on one batch.

        The implementation calls the model's Newton kernel internally,
        computes ``log det(-H)`` at live globals, sums the joint
        log-density per cell, and scales by ``data_scale`` for the
        SVI subsampling correction.

        Returns
        -------
        loss : jnp.ndarray, scalar
        aux : tuple
            ``(latent_new, eta_new, grad_inf_norm)`` — updated MAP
            iterates and per-cell gradient norms from the inner Newton.
            ``eta_new`` may be a placeholder zero vector when capture
            is off; the driver routes only the real entries back into
            ``eta_loc`` if it is not None.
        """

    # --- Full-data convergence check ------------------------------------

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

    # --- Result packaging -----------------------------------------------

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
    ) -> LaplaceRunResult:
        """Pack into the canonical result dataclass."""

    # --- Optional: aux-data slicing for mini-batches --------------------

    def aux_batch_slice(
        self, aux_data: Dict[str, jnp.ndarray], idx: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Slice per-cell entries in ``aux_data`` by mini-batch ``idx``.

        Default: slice every value whose first axis matches ``n_cells``
        and pass through anything else unchanged. Models with non-cell-
        indexed aux data can override.
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
# Optimizer resolution (mirrors engine.py)
# =====================================================================


def _build_optimizer(laplace_config: LaplaceConfig):
    """Resolve the outer-loop optax optimizer from the config.

    Same precedence rules as the legacy PLN engine:

    1. ``laplace_config.optimizer`` (a pre-built
       ``optax.GradientTransformation``) — power-user override.
    2. ``laplace_config.optimizer_config`` (serialized name + kwargs).
    3. Default ``optax.adam(1e-3)``.
    """
    if laplace_config.optimizer is not None:
        return laplace_config.optimizer
    optim_cfg = laplace_config.optimizer_config
    if optim_cfg is None:
        return optax.adam(1e-3)
    name = optim_cfg.name.lower()
    kwargs = dict(optim_cfg.kwargs or {})
    lr = kwargs.pop("step_size", kwargs.pop("learning_rate", 1e-3))
    if name == "adam":
        return optax.adam(lr, **kwargs)
    if name == "adamw":
        return optax.adamw(lr, **kwargs)
    if name == "sgd":
        return optax.sgd(lr, **kwargs)
    if name in ("clipped_adam", "adam_clipped"):
        clip = kwargs.pop("grad_clip_norm", 10.0)
        return optax.chain(
            optax.clip_by_global_norm(clip), optax.adam(lr, **kwargs)
        )
    raise ValueError(f"Unknown optimizer name: {name!r}")


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
) -> LaplaceRunResult:
    """Generic Laplace-EM training loop.

    Outer Adam on globals; inner Newton (per-cell) provided by the
    observation model. See module docstring for the full set of
    behaviours.

    Parameters
    ----------
    obs_model : LaplaceObservationModel
        Per-model hooks. See subclasses in
        :mod:`scribe.laplace._obs_nbln` etc.
    count_data : jnp.ndarray, shape (n_cells, n_genes)
    n_cells, n_genes, latent_dim : int
    laplace_config : LaplaceConfig
        Outer-loop and Newton-step hyperparameters.
    model_config : ModelConfig
    seed : int
    progress : bool, default True
    progress_backend : str, default "auto"
    """
    counts = jnp.asarray(count_data, dtype=jnp.float32)
    rng = random.PRNGKey(seed)

    # Per-model state (params, per-cell latents, aux data).
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

    # Outer optimiser.
    opt = _build_optimizer(laplace_config)
    opt_state = opt.init(params)

    # Hyperparameters.
    n_newton = int(laplace_config.n_newton_steps)
    damping = float(laplace_config.damping)
    batch_size = laplace_config.batch_size
    if batch_size is None:
        batch_size = n_cells
    batch_size = int(batch_size)
    data_scale = float(n_cells) / float(batch_size)

    # JIT the loss + gradient.
    loss_grad_fn = jax.jit(
        jax.value_and_grad(obs_model.loss_fn, has_aux=True),
        static_argnames=("data_scale", "n_newton", "damping"),
    )

    @jax.jit
    def update_step(params, opt_state, latent_loc, eta_loc, idx):
        """One outer-loop step: inner Newton + Adam on globals."""
        counts_batch = counts[idx]
        latent_batch_init = latent_loc[idx]
        if eta_loc is not None:
            eta_batch_init = eta_loc[idx]
            eta_anchor_batch = eta_anchor[idx]
        else:
            eta_batch_init = jnp.zeros(idx.shape[0])
            eta_anchor_batch = jnp.zeros(idx.shape[0])
        aux_batch = obs_model.aux_batch_slice(aux_data, idx)

        (loss, (latent_new, eta_new, gn)), grads = loss_grad_fn(
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
        return params, opt_state, latent_loc, eta_loc, gn, loss

    # Outer loop bookkeeping.
    n_steps = int(laplace_config.n_steps)
    display_interval = max(1, n_steps // 100)
    init_loss: Optional[float] = None
    min_loss_so_far: float = float("inf")
    losses: List[float] = []

    # Best-snapshot tracking for divergence recovery (always on).
    best_params: Dict[str, jnp.ndarray] = {k: v for k, v in params.items()}
    best_latent = latent_loc
    best_eta = eta_loc
    best_step_idx: int = -1
    best_loss_value: float = float("inf")
    divergence_aborted: bool = False
    early_stopped: bool = False

    def _capture_best_snapshot(p, x, e, step, loss_val):
        nonlocal best_params, best_latent, best_eta, best_step_idx, best_loss_value
        best_params = {k: v for k, v in p.items()}
        best_latent = x
        best_eta = e
        best_step_idx = step
        best_loss_value = loss_val

    # Progress reporter.
    progress_reporter = build_progress_reporter(
        progress=progress, progress_backend=progress_backend
    )

    with progress_reporter as reporter:
        reporter.start(
            description=f"Laplace ({obs_model.name})",
            total=n_steps,
            completed=0,
            loss_info="init loss: pending",
        )

        for step in range(n_steps):
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
                gn,
                loss,
            ) = update_step(params, opt_state, latent_loc, eta_loc, idx)

            loss_val = float(loss)
            losses.append(loss_val)

            if init_loss is None and np.isfinite(loss_val):
                init_loss = abs(loss_val)

            # Best-snapshot tracking.
            if loss_val < best_loss_value:
                _capture_best_snapshot(
                    params, latent_loc, eta_loc, step, loss_val
                )
            if loss_val < min_loss_so_far:
                min_loss_so_far = loss_val

            # Divergence guards (warm-up: first 50 steps).
            if step >= 50 and init_loss is not None:
                # 1. Loss-finite.
                if not np.isfinite(loss_val):
                    logger.warning(
                        "Laplace[%s]: divergence detected at step %d "
                        "(loss=%s, non-finite). Restoring best snapshot.",
                        obs_model.name,
                        step,
                        loss_val,
                    )
                    divergence_aborted = True
                    early_stopped = True
                    break
                # 2. Climb-from-min.
                climb = loss_val - min_loss_so_far
                if climb > 0.5 * init_loss:
                    logger.warning(
                        "Laplace[%s]: divergence detected at step %d "
                        "(loss climbed %.3e above running min, threshold "
                        "%.3e). Restoring best snapshot.",
                        obs_model.name,
                        step,
                        climb,
                        0.5 * init_loss,
                    )
                    divergence_aborted = True
                    early_stopped = True
                    break
                # 3. Absolute-magnitude.
                if abs(loss_val) > 1000.0 * init_loss:
                    logger.warning(
                        "Laplace[%s]: divergence detected at step %d "
                        "(|loss|=%.3e > 1000x|init_loss|=%.3e). "
                        "Restoring best snapshot.",
                        obs_model.name,
                        step,
                        abs(loss_val),
                        1000.0 * init_loss,
                    )
                    divergence_aborted = True
                    early_stopped = True
                    break

            # Periodic progress update.  ``reporter.update`` accepts
            # ``advance`` to step the bar by one tick on every call,
            # with ``loss_info`` only on the periodic-display steps so
            # the rendered string isn't churned every iteration.
            if (
                step == 0
                or (step + 1) % display_interval == 0
                or step == n_steps - 1
            ):
                reporter.update(
                    advance=1,
                    loss_info=(
                        f"loss={loss_val:.4e}  "
                        f"|grad_inner| max={float(jnp.max(gn)):.2e}"
                    ),
                )
            else:
                reporter.update(advance=1)

    # Restore best snapshot if divergence aborted.
    if divergence_aborted and best_step_idx >= 0:
        params = best_params
        latent_loc = best_latent
        eta_loc = best_eta
        logger.warning(
            "Laplace[%s]: restored best snapshot from step %d "
            "(best loss: %.4e).",
            obs_model.name,
            best_step_idx + 1,
            best_loss_value,
        )

    # Final convergence check.
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

    # Convergence-action handling.
    max_gn = float(jnp.max(final.final_grad_norms))
    if max_gn > laplace_config.newton_tolerance:
        offending = int(
            jnp.sum(final.final_grad_norms > laplace_config.newton_tolerance)
        )
        msg = (
            f"Laplace[{obs_model.name}]: {offending}/{n_cells} cells did "
            f"not converge below tolerance="
            f"{laplace_config.newton_tolerance:.1e} "
            f"(worst grad-norm={max_gn:.3e})."
        )
        action = laplace_config.convergence_action
        if action == "raise":
            raise RuntimeError(msg)
        if action == "warn":
            logger.warning(msg)

    return obs_model.pack_result(
        params=params,
        final=final,
        losses=np.asarray(losses, dtype=np.float64),
        n_steps_run=len(losses),
        model_config=model_config,
        early_stopped=early_stopped,
        best_loss=(
            best_loss_value
            if best_loss_value != float("inf")
            else float(losses[-1]) if losses else float("inf")
        ),
        stopped_at_step=len(losses),
        divergence_aborted=divergence_aborted,
    )

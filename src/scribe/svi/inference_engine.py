"""
Inference engine for SVI.

This module handles the execution of SVI inference including setting up the
SVI instance and running the optimization. Supports early stopping based on
loss convergence and Orbax checkpointing for resumable training.
"""

from dataclasses import dataclass
import sys
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import jax.numpy as jnp
from jax import random, jit
import numpyro
from numpyro.infer import SVI, TraceMeanField_ELBO
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich import print as rich_print
from ..models.model_registry import get_model_and_guide
from ..models.config import ModelConfig, EarlyStoppingConfig
from .checkpoint import (
    checkpoint_exists,
    save_svi_checkpoint,
    load_svi_checkpoint,
)

# ==============================================================================
# SVIRunResult class
# ==============================================================================


@dataclass
class SVIRunResult:
    """Result container for SVI inference.

    This mirrors numpyro's SVIRunResult to maintain API compatibility while
    adding early stopping metadata.

    Attributes
    ----------
    params : Dict[str, Any]
        Optimized variational parameters.
    losses : jnp.ndarray
        Loss values at each optimization step.
    state : Any
        Final SVI state (contains optimizer state).
    early_stopped : bool
        Whether training was stopped early due to convergence.
    stopped_at_step : int
        The step at which training stopped (equals len(losses)).
    best_loss : float
        The best smoothed loss achieved during training.
    """

    params: Dict[str, Any]
    losses: jnp.ndarray
    state: Any = None
    early_stopped: bool = False
    stopped_at_step: int = 0
    best_loss: float = float("inf")
    model_config: Optional[ModelConfig] = None


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def _progress_display_interval(n_steps: int) -> int:
    """
    Compute the periodic display/logging interval for SVI progress.

    Parameters
    ----------
    n_steps : int
        Total number of optimization steps.

    Returns
    -------
    int
        Display interval in steps, computed as ``max(1, n_steps // 20)``.
        This yields approximately 20 updates over a full run.
    """
    return max(1, n_steps // 20)


def _progress_render_interval(progress_update_every: int) -> int:
    """
    Compute the interactive progress-bar render interval in steps.

    Parameters
    ----------
    progress_update_every : int
        Requested redraw cadence for Rich progress updates.

    Returns
    -------
    int
        A safe positive render interval. Values less than 1 are coerced
        to ``1`` so callers never divide by zero.
    """
    return max(1, int(progress_update_every))


def _should_emit_progress_update(step: int, n_steps: int) -> bool:
    """
    Determine whether to emit a periodic progress update at ``step``.

    Parameters
    ----------
    step : int
        Zero-based optimization step index.
    n_steps : int
        Total number of optimization steps.

    Returns
    -------
    bool
        ``True`` when ``step`` falls on the configured periodic interval.
    """
    interval = _progress_display_interval(n_steps)
    return step % interval == 0


def _display_window_from_last_update(
    last_display_end: int, batch_end: int
) -> Tuple[int, int]:
    """
    Compute the 1-based inclusive loss window since the last displayed update.

    Parameters
    ----------
    last_display_end : int
        Last 1-based loss index already shown in the progress display.
    batch_end : int
        Current 1-based loss index corresponding to the latest optimization step.

    Returns
    -------
    Tuple[int, int]
        Inclusive ``(start, end)`` indices representing the losses newly
        accumulated since the last display event.
    """
    start = max(1, last_display_end + 1)
    end = max(start, batch_end)
    return start, end


def _should_render_progress_update(
    step: int,
    start_step: int,
    n_steps: int,
    pending_advance: int,
    progress_update_every: int,
) -> bool:
    """
    Determine whether to redraw the interactive Rich progress bar.

    Parameters
    ----------
    step : int
        Zero-based optimization step index.
    start_step : int
        Initial step index when training starts or resumes.
    n_steps : int
        Total number of optimization steps.
    pending_advance : int
        Number of completed optimization steps not yet flushed to the bar.
    progress_update_every : int
        Requested redraw cadence for interactive progress rendering.

    Returns
    -------
    bool
        ``True`` when the bar should be redrawn at this step. The function
        redraws periodically and always flushes on the final step.
    """
    # Skip no-op redraws so terminal rendering stays lightweight.
    if pending_advance <= 0:
        return False

    completed_steps = step + 1
    interval = _progress_render_interval(progress_update_every)

    # Force a final flush so progress reaches the exact final total even when
    # n_steps is not divisible by the render interval.
    is_final_step = completed_steps == n_steps
    is_periodic_tick = (completed_steps - start_step) % interval == 0
    return is_periodic_tick or is_final_step


def _is_interactive_terminal() -> bool:
    """
    Check whether stdout is an interactive TTY terminal.

    Returns
    -------
    bool
        ``True`` when stdout supports TTY-style live rendering. Non-interactive
        contexts (e.g. redirected logs, many batch runners) return ``False``.
    """
    isatty_fn = getattr(sys.stdout, "isatty", None)
    if not callable(isatty_fn):
        return False
    return bool(isatty_fn())


def _mean_ignoring_nans(values: List[float]) -> float:
    """
    Compute a mean while excluding NaN values.

    Parameters
    ----------
    values : List[float]
        Sequence of scalar loss values.

    Returns
    -------
    float
        Arithmetic mean over finite, non-NaN values. Returns ``nan`` when
        there are no finite values to average.

    Notes
    -----
    This helper is used for progress reporting only so that transient NaNs
    from unstable minibatches do not contaminate the displayed rolling mean
    when stable updates are enabled.
    """
    # Convert once to a NumPy array so masking is vectorized and warning-free.
    array_values = np.asarray(values, dtype=float)
    # Keep only finite values; this naturally excludes NaN and +/-Inf.
    finite_values = array_values[np.isfinite(array_values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def _run_with_early_stopping(
    svi: SVI,
    rng_key: random.PRNGKey,
    model_args: Dict[str, Any],
    n_steps: int,
    early_stopping: EarlyStoppingConfig,
    stable_update: bool = True,
    progress: bool = True,
    progress_update_every: int = 100,
    log_progress_lines: bool = False,
    model_config: Optional[ModelConfig] = None,
) -> SVIRunResult:
    """Run SVI with a custom training loop, checkpointing, and optional early stopping.

    This function implements a JIT-compiled training loop that monitors the
    loss, saves periodic Orbax checkpoints for resumable training, and
    optionally stops when convergence is detected.

    The loop is used whenever an ``EarlyStoppingConfig`` is provided,
    **regardless** of ``early_stopping.enabled``:

    * ``enabled=True`` — full early-stopping behaviour (convergence checks,
      patience counter, restore-best).
    * ``enabled=False`` — the loop still runs for all ``n_steps`` but
      checkpoints are saved periodically and training can be resumed after
      interruption.

    Parameters
    ----------
    svi : SVI
        NumPyro SVI instance.
    rng_key : random.PRNGKey
        JAX random key for reproducibility.
    model_args : Dict[str, Any]
        Arguments to pass to the model/guide.
    n_steps : int
        Maximum number of optimization steps.
    early_stopping : EarlyStoppingConfig
        Early stopping and checkpointing configuration, including
        ``checkpoint_dir`` and ``resume``.
    stable_update : bool, default=True
        Whether to use numerically stable updates.
    progress : bool, default=True
        Whether to show progress bar.
    progress_update_every : int, default=100
        Step cadence for interactive progress-bar redraws. A larger value
        reduces terminal render pressure in IDE terminals.
    log_progress_lines : bool, default=False
        Whether to emit periodic plain-text progress lines. When enabled,
        one log line is emitted every ``max(1, n_steps // 20)`` steps
        (approximately 20 updates per run).
    model_config : Optional[ModelConfig], default=None
        Model configuration to attach to the result for provenance.

    Returns
    -------
    SVIRunResult
        Results containing optimized parameters, loss history, and
        early stopping metadata.

    Notes
    -----
    When ``checkpoint_dir`` is set in early_stopping config:

    - Saves checkpoints at regular intervals (every ``checkpoint_every``
      steps) regardless of whether the loss improved.
    - If ``resume=True`` and a checkpoint exists, resumes from it.
    - Checkpoint includes optimiser state, step, best_loss, and loss
      history.
    """
    # Check for existing checkpoint and restore if requested
    checkpoint_dir = early_stopping.checkpoint_dir
    start_step = 0
    losses: List[float] = []
    best_loss = float("inf")
    patience_counter = 0
    resumed = False

    # Always initialize fresh state first (needed for target structure)
    svi_state = svi.init(rng_key, **model_args)

    if (
        checkpoint_dir
        and early_stopping.resume
        and checkpoint_exists(checkpoint_dir)
    ):
        # Restore from checkpoint - pass target structure for proper restoration
        checkpoint_data = load_svi_checkpoint(
            checkpoint_dir, target_optim_state=svi_state.optim_state
        )
        if checkpoint_data is not None:
            restored_optim_state, metadata, restored_losses = checkpoint_data
            start_step = metadata.step + 1
            best_loss = metadata.best_loss
            losses = restored_losses
            patience_counter = metadata.patience_counter
            resumed = True

            # Create SVIState with restored optimizer state
            from numpyro.infer.svi import SVIState

            svi_state = SVIState(
                restored_optim_state, svi_state.mutable_state, rng_key
            )

            if progress:
                # Display best_loss only when a meaningful finite value is
                # available; a NaN here means the checkpoint was saved before
                # the post-warmup baseline was established.
                best_loss_str = (
                    f"{best_loss:.4f}" if np.isfinite(best_loss) else "N/A"
                )
                rich_print(
                    f"[bold cyan]Resumed from checkpoint at step {start_step}"
                    f"[/bold cyan] (best_loss: {best_loss_str})"
                )

    # Track best state in memory
    best_state = None
    best_step = start_step if resumed else 0
    last_checkpoint_step = (
        start_step if resumed else -early_stopping.checkpoint_every
    )
    early_stopped = False

    # Calculate remaining steps
    remaining_steps = n_steps - start_step
    if remaining_steps <= 0:
        # Already completed
        params = svi.get_params(svi_state)
        return SVIRunResult(
            params=params,
            losses=jnp.array(losses),
            state=svi_state,
            early_stopped=False,
            stopped_at_step=len(losses),
            best_loss=best_loss,
            model_config=model_config,
        )

    # Rich live rendering can overwhelm some IDE terminals when redraws happen
    # at every optimization step, so only enable it for true interactive TTYs.
    use_interactive_progress = progress and _is_interactive_terminal()

    # Progress bar setup - matches NumPyro's format showing loss range.
    progress_ctx = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss_info]}"),
        disable=not use_interactive_progress,
    )

    with progress_ctx as pbar:
        # Track initial loss for display (like NumPyro)
        init_loss = losses[0] if losses else 0.0
        # Track the last 1-based loss index already shown so display ranges
        # report exactly "since previous display" instead of a rolling window.
        last_display_end = start_step

        task = pbar.add_task(
            "SVI optimization" + (" (resumed)" if resumed else ""),
            total=n_steps,
            completed=start_step,
            loss_info=f"init loss: {init_loss:.4e}",
        )
        pending_advance = 0

        eps = 1e-8  # Small constant to avoid division by zero

        # Separate large JAX arrays from lightweight metadata so that
        # the JIT-compiled update function receives them as dynamic
        # inputs rather than baking them into the compiled program as
        # traced constants (which would duplicate gigabytes of data).
        _dynamic_keys = ("counts", "dataset_indices", "annotation_prior_logits")
        dynamic_arrays = {
            k: model_args.pop(k)
            for k in _dynamic_keys
            if model_args.get(k) is not None
        }

        def body_fn(svi_state, dynamic):
            all_args = {**model_args, **dynamic}
            if stable_update:
                return svi.stable_update(svi_state, **all_args)
            else:
                return svi.update(svi_state, **all_args)

        jit_body_fn = jit(body_fn)

        # Run the update loop
        for step in range(start_step, n_steps):
            svi_state, loss = jit_body_fn(svi_state, dynamic_arrays)

            # Store loss every step (like NumPyro's svi.run)
            loss_val = float(loss)
            losses.append(loss_val)

            # Track initial loss on first step
            if step == start_step:
                init_loss = loss_val

            # Accumulate completed steps and flush them at a throttled cadence.
            # This keeps terminal redraw overhead low while preserving accurate
            # progress counts.
            pending_advance += 1

            # Compute summary loss info and plain-text logging cadence
            # independently from redraw cadence so SLURM logs remain concise.
            should_display_log = _should_emit_progress_update(step, n_steps)
            should_render = (
                use_interactive_progress
                and _should_render_progress_update(
                    step=step,
                    start_step=start_step,
                    n_steps=n_steps,
                    pending_advance=pending_advance,
                    progress_update_every=progress_update_every,
                )
            )

            if should_display_log or should_render:
                batch_end = len(losses)
                display_start, display_end = _display_window_from_last_update(
                    last_display_end=last_display_end, batch_end=batch_end
                )
                avg_loss = _mean_ignoring_nans(
                    losses[display_start - 1 : display_end]
                )
                loss_info = (
                    f"init loss: {init_loss:.4e}, "
                    f"avg. loss [{display_start}-{display_end}]: {avg_loss:.4e}"
                )
                if should_render:
                    pbar.update(
                        task, advance=pending_advance, loss_info=loss_info
                    )
                    pending_advance = 0
                if should_display_log and log_progress_lines:
                    print(
                        "SVI progress "
                        f"[{batch_end}/{n_steps}] "
                        f"init loss: {init_loss:.4e}, "
                        f"avg. loss [{display_start}-{display_end}]: "
                        f"{avg_loss:.4e}"
                    )
                # Mark this range as displayed for the next update interval.
                last_display_end = display_end
            # Periodically monitor loss and save checkpoints.
            # This runs regardless of early_stopping.enabled so that
            # checkpoints are always saved for resumability.
            should_check = (
                step % early_stopping.check_every == 0
                and len(losses) >= early_stopping.smoothing_window
            )
            if should_check:
                # Compute smoothed loss (moving average), ignoring NaN
                # values from numerically unstable steps so that transient
                # instability does not corrupt best_loss or the checkpoint.
                window_start = max(
                    0, len(losses) - early_stopping.smoothing_window
                )
                smoothed_loss = _mean_ignoring_nans(losses[window_start:])

                # Apply convergence logic only after warmup when enabled. During
                # warmup, we intentionally skip best-loss/patience updates so
                # early transient improvements do not set an unrealistically
                # strict baseline.
                if early_stopping.enabled and step >= early_stopping.warmup:
                    # Initialize the post-warmup baseline on first eligible
                    # check.
                    if not np.isfinite(best_loss):
                        best_loss = smoothed_loss
                        best_step = step
                        if early_stopping.restore_best:
                            best_state = svi_state
                        patience_counter = 0
                    else:
                        # Compute improvement (absolute difference)
                        improvement = best_loss - smoothed_loss

                        # Check for improvement using either relative (%) or
                        # absolute threshold. min_delta_pct takes precedence.
                        if early_stopping.min_delta_pct is not None:
                            # Use a finite denominator to avoid NaN/Inf on the
                            # first comparison or with very small losses.
                            denom = max(abs(best_loss), eps)
                            improvement_pct = 100.0 * improvement / denom
                            is_improvement = (
                                improvement_pct > early_stopping.min_delta_pct
                            )
                        else:
                            # Use absolute threshold
                            is_improvement = (
                                improvement > early_stopping.min_delta
                            )

                        if is_improvement:
                            # Improvement detected
                            best_loss = smoothed_loss
                            if early_stopping.restore_best:
                                best_state = svi_state
                            best_step = step
                            patience_counter = 0
                        else:
                            # No improvement - count towards patience
                            patience_counter += early_stopping.check_every

                # Save checkpoint if checkpoint_dir is set and enough steps
                # have passed since the last checkpoint.  Checkpoints are
                # saved at regular intervals regardless of whether the loss
                # improved, so that long runs can always be resumed.
                should_checkpoint = (
                    checkpoint_dir
                    and (step - last_checkpoint_step)
                    >= early_stopping.checkpoint_every
                )
                if should_checkpoint:
                    save_svi_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        optim_state=svi_state.optim_state,
                        step=step,
                        best_loss=best_loss,
                        losses=losses,
                        patience_counter=patience_counter,
                    )
                    last_checkpoint_step = step

                # Check if patience exceeded (only when early stopping is
                # enabled and past warmup)
                if (
                    early_stopping.enabled
                    and step >= early_stopping.warmup
                    and patience_counter >= early_stopping.patience
                ):
                    early_stopped = True
                    # Flush any pending bar steps before emitting the stopping
                    # message so terminal state is consistent.
                    if use_interactive_progress and pending_advance > 0:
                        pbar.update(task, advance=pending_advance)
                        pending_advance = 0
                    pbar.console.print(
                        f"[bold green]Early stopping triggered at step "
                        f"{step + 1}[/bold green] "
                        f"(no improvement for {patience_counter} steps, "
                        f"best loss: {best_loss:.4e} at step {best_step + 1})"
                    )
                    break

    # Restore best state if requested and early stopping was triggered
    if early_stopped and early_stopping.restore_best and best_state is not None:
        svi_state = best_state

    # Get final parameters
    params = svi.get_params(svi_state)

    return SVIRunResult(
        params=params,
        losses=jnp.array(losses),
        state=svi_state,
        early_stopped=early_stopped,
        stopped_at_step=len(losses),
        best_loss=best_loss,
        model_config=model_config,
    )


# ------------------------------------------------------------------------------


def _run_standard(
    svi: SVI,
    rng_key: random.PRNGKey,
    model_args: Dict[str, Any],
    n_steps: int,
    stable_update: bool = True,
    progress: bool = True,
    model_config: Optional[ModelConfig] = None,
) -> SVIRunResult:
    """Run SVI using NumPyro's built-in run method (no early stopping).

    Parameters
    ----------
    svi : SVI
        NumPyro SVI instance.
    rng_key : random.PRNGKey
        JAX random key for reproducibility.
    model_args : Dict[str, Any]
        Arguments to pass to the model/guide.
    n_steps : int
        Number of optimization steps.
    stable_update : bool, default=True
        Whether to use numerically stable updates.
    progress : bool, default=True
        Whether to show an interactive progress bar in TTY environments.

    Returns
    -------
    SVIRunResult
        Results containing optimized parameters and loss history.
    """
    # Match custom-loop behavior: only enable live progress rendering on
    # interactive terminals unless the caller disables progress explicitly.
    use_interactive_progress = progress and _is_interactive_terminal()

    # Use NumPyro's built-in run method with progress bar
    result = svi.run(
        rng_key,
        n_steps,
        stable_update=stable_update,
        progress_bar=use_interactive_progress,
        **model_args,
    )

    return SVIRunResult(
        params=result.params,
        losses=result.losses,
        state=result.state,
        early_stopped=False,
        stopped_at_step=n_steps,
        best_loss=(
            float(result.losses[-1]) if len(result.losses) > 0 else float("inf")
        ),
        model_config=model_config,
    )


# ==============================================================================
# SVIInferenceEngine class
# ==============================================================================


class SVIInferenceEngine:
    """Handles SVI inference execution with checkpointing and optional early stopping.

    This engine supports two modes of operation:

    1. **Custom loop** (when ``early_stopping`` config is provided): Uses a
       JIT-compiled training loop with periodic checkpoint saving and optional
       convergence-based early stopping.  Checkpoints are always saved so
       that interrupted runs can be resumed, even when
       ``early_stopping.enabled=False``.
    2. **Standard mode** (when ``early_stopping`` is ``None``): Falls back to
       NumPyro's built-in ``SVI.run()`` method with no checkpointing.

    Examples
    --------
    >>> from scribe.svi import SVIInferenceEngine
    >>> from scribe.models.config import ModelConfig, EarlyStoppingConfig
    >>>
    >>> # Standard inference (no early stopping)
    >>> results = SVIInferenceEngine.run_inference(
    ...     model_config=config,
    ...     count_data=counts,
    ...     n_cells=1000,
    ...     n_genes=2000,
    ...     n_steps=50000,
    ... )
    >>>
    >>> # With early stopping
    >>> early_stop = EarlyStoppingConfig(patience=500, min_delta=1e-4)
    >>> results = SVIInferenceEngine.run_inference(
    ...     model_config=config,
    ...     count_data=counts,
    ...     n_cells=1000,
    ...     n_genes=2000,
    ...     n_steps=50000,
    ...     early_stopping=early_stop,
    ... )
    >>> if results.early_stopped:
    ...     print(f"Stopped early at step {results.stopped_at_step}")
    """

    @staticmethod
    def run_inference(
        model_config: ModelConfig,
        count_data: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(
            step_size=0.001
        ),
        loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
        n_steps: int = 100_000,
        batch_size: Optional[int] = None,
        seed: int = 42,
        stable_update: bool = True,
        progress_update_every: int = 100,
        log_progress_lines: bool = False,
        early_stopping: Optional[EarlyStoppingConfig] = None,
        annotation_prior_logits: Optional[jnp.ndarray] = None,
        dataset_indices: Optional[jnp.ndarray] = None,
        progress: bool = True,
    ) -> SVIRunResult:
        """
        Execute SVI inference with optional early stopping.

        Parameters
        ----------
        model_config : ModelConfig
            Model configuration object specifying the model architecture.
        count_data : jnp.ndarray
            Processed count data (cells as rows, genes as columns).
        n_cells : int
            Number of cells in the dataset.
        n_genes : int
            Number of genes in the dataset.
        optimizer : numpyro.optim.optimizers, default=Adam(step_size=0.001)
            Optimizer for variational inference.
        loss : numpyro.infer.elbo, default=TraceMeanField_ELBO()
            Loss function for variational inference.
        n_steps : int, default=100_000
            Maximum number of optimization steps. Training may stop earlier
            if early stopping is enabled and convergence is detected.
        batch_size : Optional[int], default=None
            Mini-batch size for stochastic optimization. If None, uses
            the full dataset (batch gradient descent).
        seed : int, default=42
            Random seed for reproducibility.
        stable_update : bool, default=True
            Whether to use numerically stable parameter updates. When True,
            uses `svi.stable_update()` which handles NaN/Inf gracefully.
        progress_update_every : int, default=100
            Step cadence for interactive progress-bar redraws in the custom
            loop used by SVI early-stopping/checkpoint execution.
        log_progress_lines : bool, default=False
            Whether to emit periodic plain-text progress lines in addition to
            the interactive progress bar. When enabled, one line is emitted
            every ``max(1, n_steps // 20)`` steps.
        early_stopping : Optional[EarlyStoppingConfig], default=None
            Configuration for early stopping and checkpointing. When provided
            (even with ``enabled=False``), the custom training loop is used
            so that checkpoints are saved and training can be resumed. Only
            falls back to NumPyro's built-in ``svi.run()`` when ``None``.
        dataset_indices : Optional[jnp.ndarray], default=None
            Per-cell integer array mapping each cell to its dataset index
            ``{0, ..., n_datasets-1}``.  Required when
            ``model_config.n_datasets`` is set.
        progress : bool, default=True
            Whether to show progress bar during training.

        Returns
        -------
        SVIRunResult
            Results object containing:
            - params: Optimized variational parameters
            - losses: Loss values at each step
            - state: Final SVI state
            - early_stopped: Whether early stopping was triggered
            - stopped_at_step: Step at which training stopped
            - best_loss: Best smoothed loss achieved

        Examples
        --------
        >>> results = SVIInferenceEngine.run_inference(
        ...     model_config=config,
        ...     count_data=counts,
        ...     n_cells=1000,
        ...     n_genes=2000,
        ...     n_steps=50000,
        ...     early_stopping=EarlyStoppingConfig(patience=500),
        ... )
        >>> print(f"Final loss: {results.losses[-1]:.4f}")
        >>> if results.early_stopped:
        ...     print(f"Converged at step {results.stopped_at_step}")

        Notes
        -----
        When early stopping is enabled, the training loop monitors a smoothed
        (moving average) loss. If the loss doesn't improve by at least
        `min_delta` for `patience` steps, training stops and optionally
        restores the parameters from the best checkpoint.

        See Also
        --------
        EarlyStoppingConfig : Configuration for early stopping criteria.
        SVIRunResult : Container for inference results.
        """
        # Get model and guide functions using the builder-based API
        # Pass n_genes for VAE (required for encoder/decoder sizing)
        model, guide, model_config_for_results = get_model_and_guide(
            model_config, n_genes=n_genes
        )

        # Create SVI instance
        svi = SVI(model, guide, optimizer, loss=loss)

        # Create random key
        rng_key = random.PRNGKey(seed)

        # Prepare model arguments.
        # Use model_config_for_results (which has param_specs populated)
        # so that index_dataset_params in the likelihood can reliably
        # identify which parameters carry a dataset axis.  Without
        # param_specs the function falls back to a shape[0]==n_datasets
        # heuristic that fails for mixture+dataset params (K, D, G)
        # when K != D.
        model_args = {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "counts": count_data,
            "batch_size": batch_size,
            "model_config": model_config_for_results,
            "annotation_prior_logits": annotation_prior_logits,
            "dataset_indices": dataset_indices,
        }

        # Choose execution mode based on early stopping configuration.
        # Use the custom loop whenever an early_stopping config is provided
        # (even when enabled=False) so that checkpoint saving and resume
        # are always available.  Only fall back to NumPyro's built-in
        # svi.run() when no early_stopping config is supplied at all.
        if early_stopping is not None:
            return _run_with_early_stopping(
                svi=svi,
                rng_key=rng_key,
                model_args=model_args,
                n_steps=n_steps,
                early_stopping=early_stopping,
                stable_update=stable_update,
                progress=progress,
                progress_update_every=progress_update_every,
                log_progress_lines=log_progress_lines,
                model_config=model_config_for_results,
            )
        else:
            return _run_standard(
                svi=svi,
                rng_key=rng_key,
                model_args=model_args,
                n_steps=n_steps,
                stable_update=stable_update,
                progress=progress,
                model_config=model_config_for_results,
            )

"""
Inference engine for SVI.

This module handles the execution of SVI inference including setting up the
SVI instance and running the optimization. Supports early stopping based on
loss convergence and Orbax checkpointing for resumable training.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
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


def _run_with_early_stopping(
    svi: SVI,
    rng_key: random.PRNGKey,
    model_args: Dict[str, Any],
    n_steps: int,
    early_stopping: EarlyStoppingConfig,
    stable_update: bool = True,
    progress: bool = True,
    model_config: Optional[ModelConfig] = None,
) -> SVIRunResult:
    """Run SVI with early stopping based on loss convergence.

    This function implements a custom training loop that monitors the loss and
    stops when convergence is detected. Supports Orbax checkpointing for
    resumable training.

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
        Early stopping configuration, including checkpoint_dir and resume.
    stable_update : bool, default=True
        Whether to use numerically stable updates.
    progress : bool, default=True
        Whether to show progress bar.

    Returns
    -------
    SVIRunResult
        Results containing optimized parameters, loss history, and
        early stopping metadata.

    Notes
    -----
    When `checkpoint_dir` is set in early_stopping config:
    - Saves checkpoint whenever loss improves
    - If `resume=True` and checkpoint exists, resumes from it
    - Checkpoint includes params, step, best_loss, and loss history
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
                rich_print(
                    f"[bold cyan]Resumed from checkpoint at step {start_step}"
                    f"[/bold cyan] (best_loss: {best_loss:.4f})"
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

    # Progress bar setup - matches NumPyro's format showing loss range
    progress_ctx = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss_info]}"),
        disable=not progress,
    )

    with progress_ctx as pbar:
        # Track initial loss for display (like NumPyro)
        init_loss = losses[0] if losses else 0.0
        loss_display_interval = max(1, n_steps // 20)

        task = pbar.add_task(
            "SVI optimization" + (" (resumed)" if resumed else ""),
            total=n_steps,
            completed=start_step,
            loss_info=f"init loss: {init_loss:.4e}",
        )

        eps = 1e-8  # Small constant to avoid division by zero

        # JIT compile the update function (critical for GPU performance!)
        # This matches how NumPyro's svi.run() works internally
        # 1. Define the update function (distinguishing between stable and
        #    unstable updates)
        def body_fn(svi_state):
            if stable_update:
                return svi.stable_update(svi_state, **model_args)
            else:
                return svi.update(svi_state, **model_args)

        # 2. JIT compile the update function
        jit_body_fn = jit(body_fn)

        # 3. Run the update loop
        for step in range(start_step, n_steps):
            # Perform update step (JIT compiled)
            svi_state, loss = jit_body_fn(svi_state)

            # Store loss every step (like NumPyro's svi.run)
            loss_val = float(loss)
            losses.append(loss_val)

            # Track initial loss on first step
            if step == start_step:
                init_loss = loss_val

            # Update progress bar periodically with avg loss over recent batch
            # Format matches NumPyro: "init loss: X, avg. loss [start-end]: X"
            should_display = step % loss_display_interval == 0
            if should_display:
                batch_start = max(0, len(losses) - loss_display_interval)
                batch_end = len(losses)
                avg_loss = np.mean(losses[batch_start:batch_end])
                loss_info = (
                    f"init loss: {init_loss:.4e}, "
                    f"avg. loss [{batch_start + 1}-{batch_end}]: {avg_loss:.4e}"
                )
                pbar.update(task, advance=1, loss_info=loss_info)
            else:
                pbar.update(task, advance=1)

            # Check for improvement every check_every steps (for checkpointing)
            # This happens even during warmup - we still track best loss and save
            should_check_improvement = (
                early_stopping.enabled
                and step % early_stopping.check_every == 0
                and len(losses) >= early_stopping.smoothing_window
            )
            if should_check_improvement:
                # Compute smoothed loss (moving average)
                window_start = max(
                    0, len(losses) - early_stopping.smoothing_window
                )
                smoothed_loss = np.mean(losses[window_start:])

                # Compute improvement (absolute difference)
                improvement = best_loss - smoothed_loss

                # Check for improvement using either relative (%) or absolute
                # threshold min_delta_pct takes precedence if specified
                if early_stopping.min_delta_pct is not None:
                    # Use relative (percentage) threshold
                    improvement_pct = 100.0 * improvement / (best_loss + eps)
                    is_improvement = (
                        improvement_pct > early_stopping.min_delta_pct
                    )
                else:
                    # Use absolute threshold
                    is_improvement = improvement > early_stopping.min_delta

                if is_improvement:
                    # Improvement detected
                    best_loss = smoothed_loss
                    if early_stopping.restore_best:
                        best_state = svi_state
                    best_step = step
                    patience_counter = 0

                    # Save checkpoint if checkpoint_dir is set and enough steps
                    # have passed since the last checkpoint
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
                else:
                    # No improvement - only count towards patience after warmup
                    if step >= early_stopping.warmup:
                        patience_counter += early_stopping.check_every

                # Check if patience exceeded (only after warmup)
                if (
                    step >= early_stopping.warmup
                    and patience_counter >= early_stopping.patience
                ):
                    early_stopped = True
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

    Returns
    -------
    SVIRunResult
        Results containing optimized parameters and loss history.
    """
    # Use NumPyro's built-in run method with progress bar
    result = svi.run(
        rng_key,
        n_steps,
        stable_update=stable_update,
        progress_bar=True,
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
    """Handles SVI inference execution with optional early stopping.

    This engine supports two modes of operation:
    1. Standard mode: Uses NumPyro's built-in SVI.run() method
    2. Early stopping mode: Uses a custom training loop that monitors
       loss convergence and stops when improvement plateaus

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
        early_stopping: Optional[EarlyStoppingConfig] = None,
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
        early_stopping : Optional[EarlyStoppingConfig], default=None
            Configuration for early stopping. If None or if
            `early_stopping.enabled=False`, runs for the full `n_steps`.
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
        model, guide, model_config_for_results = get_model_and_guide(
            model_config
        )

        # Create SVI instance
        svi = SVI(model, guide, optimizer, loss=loss)

        # Create random key
        rng_key = random.PRNGKey(seed)

        # Prepare model arguments
        model_args = {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "counts": count_data,
            "batch_size": batch_size,
            "model_config": model_config,
        }

        # Choose execution mode based on early stopping configuration
        if early_stopping is not None and early_stopping.enabled:
            return _run_with_early_stopping(
                svi=svi,
                rng_key=rng_key,
                model_args=model_args,
                n_steps=n_steps,
                early_stopping=early_stopping,
                stable_update=stable_update,
                progress=progress,
                model_config=model_config_for_results,
            )
        else:
            return _run_standard(
                svi=svi,
                rng_key=rng_key,
                model_args=model_args,
                n_steps=n_steps,
                stable_update=stable_update,
                model_config=model_config_for_results,
            )

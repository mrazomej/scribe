"""
Utility functions for SCRIBE.
"""

import os
import jax
import warnings
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Callable
import jax.numpy as jnp
from collections import deque
from threading import Lock
from tqdm.auto import tqdm as tqdm_auto
from jax import lax
from functools import partial
from jax.experimental import io_callback
import time

# ------------------------------------------------------------------------------

def git_root(current_path=None):
    """
    Finds the root directory of a Git repository.
    
    Args:
        current_path (str, optional): The starting path. If None, uses the current working directory.
    
    Returns:
        str: The path to the Git root directory, or None if not found.
    """
    if current_path is None:
        current_path = os.getcwd()
    
    while current_path and current_path != os.path.dirname(current_path):
        if os.path.isdir(os.path.join(current_path, '.git')):
            return current_path
        current_path = os.path.dirname(current_path)  # Move up one directory level
    
    return None  # Git root not found

# ------------------------------------------------------------------------------

@contextmanager
def use_cpu():
    """
    Context manager to temporarily force JAX computations to run on CPU.
    
    This is useful when you want to ensure specific computations run on CPU
    rather than GPU/TPU, for example when running out of GPU memory.
    
    Returns
    -------
    None
        Yields control back to the context block
        
    Example
    -------
    >>> # Force posterior sampling to run on CPU
    >>> with use_cpu():
    ...     results.ppc_samples(n_samples=100)
    """
    # Store the current default device to restore it later
    original_device = jax.default_device()
    
    # Get the first available CPU device
    cpu_device = jax.devices('cpu')[0]
    
    # Set CPU as the default device for JAX computations
    jax.default_device(cpu_device)
    
    try:
        # Yield control to the context block
        yield
    finally:
        # Restore the original default device when exiting the context
        jax.default_device(original_device)

# ------------------------------------------------------------------------------

class EarlyStoppingCallback:
    """
    Callback for early stopping in SCRIBE inference.
    
    This class implements early stopping based on the following criteria:
        1. Loss becomes NaN or infinite
        2. Loss hasn't improved significantly over a patience window
        3. Loss starts diverging (ratio between consecutive losses too large)
    
    The callback is designed to minimize computational overhead by only checking
    conditions every N steps and maintaining a small window of loss values.
    
    Parameters
    ----------
    patience : int, optional
        Number of checks to wait for improvement before stopping (default: 50)
    min_delta : float, optional
        Minimum change in loss to qualify as an improvement (default: 10)
    check_every : int, optional
        Number of iterations between checks (default: 100)
    window_size : int, optional
        Size of the moving window for loss averaging (default: 5)
    max_ratio : float, optional
        Maximum allowed ratio between consecutive losses (default: 2.0)
    """
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 10,
        check_every: int = 100,
        window_size: int = 5,
        max_ratio: float = 2.0
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.check_every = check_every
        self.window_size = window_size
        self.max_ratio = max_ratio
        
        # Initialize tracking variables
        self.best_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
        self.step = 0
        
        # Use deque for efficient window operations
        self.loss_window = deque(maxlen=window_size)
    
    def __call__(
        self,
        step: int,
        loss: float,
        params: Dict,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if training should stop based on current state.
        
        Parameters
        ----------
        step : int
            Current training step
        loss : float
            Current loss value
        params : Dict
            Current parameter values
            
        Returns
        -------
        tuple[bool, Optional[str]]
            (should_stop, reason)
                - should_stop: True if training should stop
                - reason: String explaining why training stopped, or None if it
                  should continue
        """
        self.step = step
        
        # Only check every N steps to reduce overhead
        if step % self.check_every != 0:
            return False, None
            
        # Check for NaN or infinite loss
        if not jnp.isfinite(loss):
            return True, "Loss is NaN or infinite"
        
        # Update loss window
        self.loss_window.append(loss)
        
        # Only proceed with other checks if we have enough values
        if len(self.loss_window) < self.window_size:
            return False, None
            
        # Compute average loss over window
        avg_loss = jnp.mean(jnp.array(list(self.loss_window)))
        
        # Check for divergence using ratio of consecutive window averages
        if len(self.loss_window) == self.window_size and step >= self.check_every:
            prev_losses = jnp.array(list(self.loss_window)[:-1])
            curr_losses = jnp.array(list(self.loss_window)[1:])
            ratios = curr_losses / prev_losses
            if jnp.any(ratios > self.max_ratio):
                return True, "Loss is diverging"
        
        # Update best loss and parameters if improved
        if avg_loss < self.best_loss - self.min_delta:
            self.best_loss = avg_loss
            self.best_params = params
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check patience
        if self.patience_counter >= self.patience:
            return True, f"No improvement for {self.patience} checks"
        
        return False, None

# ------------------------------------------------------------------------------

@contextmanager
def suppress_warnings():
    """Context manager to temporarily suppress warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

# ------------------------------------------------------------------------------
# Progress bar with early stopping
# ------------------------------------------------------------------------------

def create_progress_bar_with_early_stopping(
    num_steps: int,
    early_stopping=None,
    check_every: int = 100,
) -> Tuple[Callable, Callable]:
    """
    Creates progress bar infrastructure for SVI training with early stopping
    support.
    
    Parameters
    ----------
    num_steps : int
        Total number of optimization steps to run
    early_stopping : Optional[EarlyStoppingCallback]
        Callback object for early stopping criteria
    check_every : int
        Number of steps between early stopping checks
        
    Returns
    -------
    Tuple[Callable, Callable]
        Returns:
            - progress_wrapper: Function to wrap optimization step with progress
            - should_stop_early: Function to check if early stopping was
              triggered
    """
    # Calculate update rate for ~20 total updates
    if num_steps > 20:
        print_rate = int(num_steps / 20)
    else:
        print_rate = 1
    remainder = num_steps % print_rate
    
    # Initialize progress bar and early stopping state
    progress_bar = tqdm_auto(
        range(num_steps), 
        desc="Initializing SVI...",
        leave=True
    )
    early_stop_info = {'triggered': False, 'reason': None}
    
    def _update_bar(increment, loss=None):
        """Updates progress bar display"""
        if loss is not None:
            progress_bar.set_description(
                f"Training - Loss: {loss:.3f}", refresh=False
            )
        progress_bar.update(increment)
    
    def _close_bar(increment, early_stop=False, reason=None):
        """Closes progress bar"""
        progress_bar.update(increment)
        if early_stop and reason:
            progress_bar.write(f"\nStopping early: {reason}")
        progress_bar.close()
    
    def _check_early_stopping(step, loss, params):
        """Checks early stopping criteria"""
        if early_stopping is None:
            return False
        
        should_stop, reason = early_stopping(step, loss, params)
        if should_stop:
            early_stop_info['triggered'] = True
            early_stop_info['reason'] = reason
        return should_stop
    
    def update_progress(iter_num, state, loss):
        """Updates progress and checks early stopping"""
        # Initialize progress bar
        _ = lax.cond(
            iter_num == 1,
            lambda _: io_callback(_update_bar, jnp.array(0), loss),
            lambda _: None,
            operand=None,
        )
        
        # Regular progress updates
        _ = lax.cond(
            iter_num % print_rate == 0,
            lambda _: io_callback(_update_bar, jnp.array(print_rate), loss),
            lambda _: None,
            operand=None,
        )
        
        # Early stopping check
        if early_stopping is not None and iter_num % check_every == 0:
            should_stop = io_callback(
                _check_early_stopping,
                None,
                iter_num,
                loss,
                state
            )
            if should_stop:
                io_callback(
                    _close_bar,
                    None,
                    remainder,
                    True,
                    early_stop_info['reason']
                )
                return True
        
        # Final update
        _ = lax.cond(
            iter_num == num_steps,
            lambda _: io_callback(_close_bar, None, remainder),
            lambda _: None,
            operand=None,
        )
        
        return False
    
    def progress_bar_wrapper(func):
        """Wraps optimization step with progress updates"""
        def wrapper_progress(i, state):
            new_state, loss = func(i, state)
            should_stop = update_progress(i + 1, new_state, loss)
            return new_state, loss, should_stop
        return wrapper_progress
    
    def should_stop_early():
        """Checks if early stopping was triggered"""
        return early_stop_info['triggered'], early_stop_info['reason']
    
    return progress_bar_wrapper, should_stop_early

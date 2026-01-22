"""
Orbax-based checkpointing utilities for SVI inference.

This module provides functions to save and restore SVI training state,
enabling resumable training and persistence of best parameters.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orbax.checkpoint as ocp

# Suppress verbose Orbax/absl logs (INFO and WARNING clutter the output)
logging.getLogger("absl").setLevel(logging.ERROR)

# ==============================================================================
# CheckpointMetadata class
# ==============================================================================


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside the SVI state checkpoint.

    Attributes
    ----------
    step : int
        Training step at which checkpoint was saved.
    best_loss : float
        Best smoothed loss achieved at checkpoint time.
    n_losses : int
        Number of loss values in the history.
    patience_counter : int
        Current patience counter value.
    """

    step: int
    best_loss: float
    n_losses: int
    patience_counter: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step": self.step,
            "best_loss": self.best_loss,
            "n_losses": self.n_losses,
            "patience_counter": self.patience_counter,
        }

    # --------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Create from dictionary."""
        return cls(
            step=data["step"],
            best_loss=data["best_loss"],
            n_losses=data["n_losses"],
            patience_counter=data["patience_counter"],
        )


# ------------------------------------------------------------------------------


def _get_checkpoint_path(checkpoint_dir: str) -> Path:
    """Get the path to the checkpoint subdirectory."""
    return Path(checkpoint_dir) / "best"


# ------------------------------------------------------------------------------


def _get_metadata_path(checkpoint_dir: str) -> Path:
    """Get the path to the metadata JSON file."""
    return Path(checkpoint_dir) / "metadata.json"


# ------------------------------------------------------------------------------


def _get_losses_path(checkpoint_dir: str) -> Path:
    """Get the path to the losses numpy file."""
    return Path(checkpoint_dir) / "losses.npy"


# ------------------------------------------------------------------------------


def checkpoint_exists(checkpoint_dir: str) -> bool:
    """Check if a valid checkpoint exists in the directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to check for checkpoints.

    Returns
    -------
    bool
        True if a valid checkpoint exists, False otherwise.
    """
    if not checkpoint_dir:
        return False

    checkpoint_path = _get_checkpoint_path(checkpoint_dir)
    metadata_path = _get_metadata_path(checkpoint_dir)

    return checkpoint_path.exists() and metadata_path.exists()


# ------------------------------------------------------------------------------


def save_svi_checkpoint(
    checkpoint_dir: str,
    params: Dict[str, Any],
    step: int,
    best_loss: float,
    losses: List[float],
    patience_counter: int = 0,
) -> None:
    """Save SVI parameters and training state to checkpoint directory.

    This saves:
    - Variational parameters (using Orbax)
    - Training metadata (step, best_loss, etc.)
    - Loss history

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoint to.
    params : Dict[str, Any]
        Variational parameters from SVI (svi.get_params(state)).
    step : int
        Current training step.
    best_loss : float
        Best smoothed loss achieved.
    losses : List[float]
        Loss history up to this point.
    patience_counter : int, default=0
        Current patience counter value.

    Notes
    -----
    The checkpoint directory structure:
    ```
    checkpoint_dir/
      best/           # Orbax checkpoint of params
      metadata.json   # Training metadata
      losses.npy      # Loss history as numpy array
    ```
    """
    checkpoint_path = _get_checkpoint_path(checkpoint_dir)
    metadata_path = _get_metadata_path(checkpoint_dir)
    losses_path = _get_losses_path(checkpoint_dir)

    # Create directory if needed
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save parameters using Orbax
    # Use synchronous checkpointer to ensure save completes before returning
    checkpointer = ocp.StandardCheckpointer()

    # Remove old checkpoint if exists (Orbax doesn't overwrite by default)
    if checkpoint_path.exists():
        import shutil

        shutil.rmtree(checkpoint_path)

    checkpointer.save(checkpoint_path, params)
    # Wait for async save to complete
    checkpointer.wait_until_finished()

    # Save metadata as JSON
    metadata = CheckpointMetadata(
        step=step,
        best_loss=best_loss,
        n_losses=len(losses),
        patience_counter=patience_counter,
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    # Save losses as numpy array
    np.save(losses_path, np.array(losses))


# ------------------------------------------------------------------------------


def load_svi_checkpoint(
    checkpoint_dir: str,
) -> Optional[Tuple[Dict[str, Any], CheckpointMetadata, List[float]]]:
    """Load SVI parameters and training state from checkpoint.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing the checkpoint.

    Returns
    -------
    Optional[Tuple[Dict[str, Any], CheckpointMetadata, List[float]]]
        Tuple of (params, metadata, losses) if checkpoint exists,
        None otherwise.

        - params: Variational parameters dictionary
        - metadata: CheckpointMetadata with step, best_loss, etc.
        - losses: Loss history as list of floats
    """
    if not checkpoint_exists(checkpoint_dir):
        return None

    checkpoint_path = _get_checkpoint_path(checkpoint_dir)
    metadata_path = _get_metadata_path(checkpoint_dir)
    losses_path = _get_losses_path(checkpoint_dir)

    # Load parameters using Orbax
    checkpointer = ocp.StandardCheckpointer()

    # We need to restore without a target structure
    # Use abstract restore to get the structure
    params = checkpointer.restore(checkpoint_path)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = CheckpointMetadata.from_dict(json.load(f))

    # Load losses
    losses = np.load(losses_path).tolist()

    return params, metadata, losses


# ------------------------------------------------------------------------------


def remove_checkpoint(checkpoint_dir: str) -> None:
    """Remove checkpoint directory and all its contents.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to remove.
    """
    if checkpoint_dir and Path(checkpoint_dir).exists():
        import shutil

        shutil.rmtree(checkpoint_dir)

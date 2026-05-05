"""Orbax-based checkpointing utilities for Laplace-mode inference.

Mirrors :mod:`scribe.svi.checkpoint` but saves the additional state
that the Laplace path carries (per-cell ``x_loc``/``eta_loc`` arrays
and the optax optimizer state for globals). The SVI checkpoint saves
only ``svi_state.optim_state`` because everything else lives inside
that NumPyro state object; the hand-rolled Laplace loop has its
state spread across four pieces and so saves them as a single
pytree under one orbax handle.

Layout on disk::

    checkpoint_dir/
      best/                   # Orbax StandardCheckpointer pytree
        manifest.ocdbt        # OCDBT data (or ocdbt.process_0/manifest.ocdbt
        _METADATA             #  for paths with brackets)
        ...
      metadata.json           # step, best_loss, n_losses, patience_counter,
                              # has_eta_loc
      losses.npy              # per-step loss history (np.ndarray)

The orbax pytree itself contains four named entries: ``params``,
``opt_state``, ``x_loc``, and ``eta_loc`` (the last is a 1-element
zero array when capture anchor is off — orbax cannot serialize
``None`` directly, and we recover the original by dropping it during
``load`` based on the ``has_eta_loc`` metadata flag).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

# Suppress verbose Orbax/absl logs (INFO and WARNING clutter the
# output) — same convention scribe's SVI checkpoint module uses.
logging.getLogger("absl").setLevel(logging.ERROR)


# ==============================================================================
# Metadata
# ==============================================================================


@dataclass
class LaplaceCheckpointMetadata:
    """Metadata stored alongside the Laplace state checkpoint.

    Attributes
    ----------
    step : int
        Outer-loop step at which the checkpoint was saved.
    best_loss : float
        Best smoothed loss observed at checkpoint time.
    n_losses : int
        Number of loss values in the history.
    patience_counter : int
        Current early-stopping patience counter value.
    has_eta_loc : bool
        Whether ``eta_loc`` was active when the checkpoint was saved.
        Used to restore ``None`` rather than the placeholder zero
        array when the capture anchor is off.
    """

    step: int
    best_loss: float
    n_losses: int
    patience_counter: int
    has_eta_loc: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "best_loss": self.best_loss,
            "n_losses": self.n_losses,
            "patience_counter": self.patience_counter,
            "has_eta_loc": self.has_eta_loc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LaplaceCheckpointMetadata":
        return cls(
            step=int(data["step"]),
            best_loss=float(data["best_loss"]),
            n_losses=int(data["n_losses"]),
            patience_counter=int(data["patience_counter"]),
            # Default True for backwards compat in case an older
            # checkpoint lacks the field.
            has_eta_loc=bool(data.get("has_eta_loc", True)),
        )


# ==============================================================================
# Path utilities
# ==============================================================================


def _sanitize_checkpoint_path(path: str) -> str:
    """Strip bracket characters that orbax OCDBT interprets as glob.

    Same shim as in :mod:`scribe.svi.checkpoint` — exists because
    Hydra overrides like ``[mu,phi]`` end up in checkpoint paths and
    OCDBT silently breaks. We mirror the behaviour so SVI and Laplace
    checkpoints sit side-by-side under the same parent.
    """
    return path.replace("[", "").replace("]", "")


def _get_checkpoint_path(checkpoint_dir: str) -> Path:
    return Path(_sanitize_checkpoint_path(checkpoint_dir)) / "best"


def _get_metadata_path(checkpoint_dir: str) -> Path:
    return Path(_sanitize_checkpoint_path(checkpoint_dir)) / "metadata.json"


def _get_losses_path(checkpoint_dir: str) -> Path:
    return Path(_sanitize_checkpoint_path(checkpoint_dir)) / "losses.npy"


def laplace_checkpoint_exists(checkpoint_dir: str) -> bool:
    """Return ``True`` when a Laplace checkpoint directory looks complete.

    A complete checkpoint has the orbax ``_METADATA`` file plus the
    JSON metadata and the loss-history numpy array on disk. This
    function does not validate the orbax state pytree itself — that
    happens lazily in :func:`load_laplace_checkpoint`.
    """
    sanitized = _sanitize_checkpoint_path(checkpoint_dir)
    if not Path(sanitized).exists():
        return False
    cp = _get_checkpoint_path(checkpoint_dir)
    md = _get_metadata_path(checkpoint_dir)
    ls = _get_losses_path(checkpoint_dir)
    return (cp / "_METADATA").exists() and md.exists() and ls.exists()


# ==============================================================================
# Save / load
# ==============================================================================


def save_laplace_checkpoint(
    checkpoint_dir: str,
    params: Dict[str, jnp.ndarray],
    opt_state: Any,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    step: int,
    best_loss: float,
    losses: List[float],
    patience_counter: int = 0,
) -> None:
    """Save Laplace state and metadata to ``checkpoint_dir``.

    Parameters
    ----------
    checkpoint_dir : str
        Output directory. Created if missing. Existing ``best/``
        subdirectory is removed before writing (orbax does not
        overwrite by default).
    params : dict
        Globals dict (``mu``, ``W``, ``d_log``).
    opt_state : Any
        optax optimizer state for the globals. Saved as part of the
        same orbax pytree so resumed training keeps Adam's first /
        second moment estimates.
    x_loc : jnp.ndarray, shape (n_cells, G)
        Per-cell MAP estimate of the latent log-rates.
    eta_loc : jnp.ndarray or None, shape (n_cells,)
        Per-cell capture-offset MAP, or ``None`` when capture anchor
        is off. Persisted as a placeholder zero array when ``None``;
        ``has_eta_loc=False`` is recorded in the metadata so load
        recovers ``None`` correctly.
    step : int
        Current outer step.
    best_loss : float
        Best smoothed loss so far.
    losses : list of float
        Loss history (full).
    patience_counter : int, default=0
        Current patience counter.

    Raises
    ------
    RuntimeError
        If the orbax save did not produce the expected ``_METADATA``
        / ``manifest.ocdbt`` files (indicates an interrupted I/O).
    """
    sanitized_dir = _sanitize_checkpoint_path(checkpoint_dir)
    cp_path = _get_checkpoint_path(checkpoint_dir)
    md_path = _get_metadata_path(checkpoint_dir)
    ls_path = _get_losses_path(checkpoint_dir)

    os.makedirs(sanitized_dir, exist_ok=True)

    # Build the pytree to save. Orbax cannot serialize Python `None`,
    # so substitute a placeholder zero array when no capture anchor is
    # active and record the original status in the metadata.
    has_eta_loc = eta_loc is not None
    eta_to_save = (
        jnp.asarray(eta_loc)
        if has_eta_loc
        else jnp.zeros((1,), dtype=jnp.float32)
    )
    state = {
        "params": params,
        "opt_state": opt_state,
        "x_loc": jnp.asarray(x_loc),
        "eta_loc": eta_to_save,
    }

    # Orbax does not overwrite by default; remove any prior checkpoint
    # under ``best/`` first. We do *not* remove ``metadata.json`` or
    # ``losses.npy`` until after the orbax save succeeds, so an
    # interrupted save leaves the prior on-disk metadata intact rather
    # than corrupted.
    if cp_path.exists():
        shutil.rmtree(cp_path)

    checkpointer = ocp.StandardCheckpointer()
    try:
        checkpointer.save(cp_path, state)
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()

    # Verify the orbax write produced the files we expect. OCDBT may
    # place ``manifest.ocdbt`` either at the checkpoint root or inside
    # ``ocdbt.process_0/`` depending on path characters; accept either.
    metadata_ok = (cp_path / "_METADATA").exists()
    manifest_at_root = (cp_path / "manifest.ocdbt").exists()
    manifest_in_ocdbt = (
        cp_path / "ocdbt.process_0" / "manifest.ocdbt"
    ).exists()
    if not metadata_ok:
        raise RuntimeError(
            f"Laplace checkpoint save incomplete: missing _METADATA at "
            f"{cp_path}. Disk I/O issue or interrupted save?"
        )
    if not (manifest_at_root or manifest_in_ocdbt):
        raise RuntimeError(
            f"Laplace checkpoint save incomplete: missing manifest.ocdbt "
            f"under {cp_path}. Disk I/O issue or interrupted save?"
        )

    # Now write the metadata + loss history. Doing this *after* the
    # orbax save means a partial run never produces a metadata file
    # that points at a missing pytree.
    metadata = LaplaceCheckpointMetadata(
        step=int(step),
        best_loss=float(best_loss),
        n_losses=len(losses),
        patience_counter=int(patience_counter),
        has_eta_loc=bool(has_eta_loc),
    )
    with open(md_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    np.save(ls_path, np.asarray(losses, dtype=np.float32))


def load_laplace_checkpoint(
    checkpoint_dir: str,
    target_state: Dict[str, Any],
) -> Optional[
    Tuple[Dict[str, Any], LaplaceCheckpointMetadata, List[float]]
]:
    """Load a Laplace checkpoint, returning ``(state, metadata, losses)``.

    Parameters
    ----------
    checkpoint_dir : str
        Directory previously written by :func:`save_laplace_checkpoint`.
    target_state : dict
        A pytree with the same shape as the saved state, used by
        orbax to re-construct the JAX/optax leaves at the right
        dtype and shape. Must contain at minimum ``params``,
        ``opt_state``, ``x_loc``, and ``eta_loc`` with the right
        leaf shapes (the placeholder-zero ``eta_loc`` for the
        capture-off case is fine).

    Returns
    -------
    tuple or None
        ``(state, metadata, losses)`` on success, or ``None`` when
        the checkpoint directory is missing/incomplete (the caller
        falls back to fresh initialization). The returned ``state``
        has its ``eta_loc`` set to ``None`` when the metadata flag
        ``has_eta_loc=False``.
    """
    if not laplace_checkpoint_exists(checkpoint_dir):
        return None

    cp_path = _get_checkpoint_path(checkpoint_dir)
    md_path = _get_metadata_path(checkpoint_dir)
    ls_path = _get_losses_path(checkpoint_dir)

    # Match the SVI module: pass the target structure via the
    # legacy ``target=`` kwarg so orbax preserves tuples / namedtuples
    # / dict ordering on restore. The ``args=ocp.args.StandardRestore``
    # form is from a newer orbax that scribe doesn't pin to.
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(cp_path, target=target_state)
    finally:
        checkpointer.close()

    with open(md_path, "r") as f:
        metadata = LaplaceCheckpointMetadata.from_dict(json.load(f))

    # Strip the placeholder eta_loc when the saved run had no capture
    # anchor — keeps the engine's internal state typed as
    # Optional[jnp.ndarray] rather than always-array.
    if not metadata.has_eta_loc:
        restored = {**restored, "eta_loc": None}

    losses = np.load(ls_path).astype(np.float32).tolist()
    return restored, metadata, losses


def remove_laplace_checkpoint(checkpoint_dir: str) -> None:
    """Remove all files written by :func:`save_laplace_checkpoint`.

    Idempotent — silently no-ops when the directory does not exist.
    """
    sanitized = _sanitize_checkpoint_path(checkpoint_dir)
    if not Path(sanitized).exists():
        return
    cp_path = _get_checkpoint_path(checkpoint_dir)
    md_path = _get_metadata_path(checkpoint_dir)
    ls_path = _get_losses_path(checkpoint_dir)
    if cp_path.exists():
        shutil.rmtree(cp_path)
    if md_path.exists():
        md_path.unlink()
    if ls_path.exists():
        ls_path.unlink()


__all__ = [
    "LaplaceCheckpointMetadata",
    "laplace_checkpoint_exists",
    "save_laplace_checkpoint",
    "load_laplace_checkpoint",
    "remove_laplace_checkpoint",
]

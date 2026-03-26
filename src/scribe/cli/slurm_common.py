"""Shared SLURM configuration helpers for CLI commands.

This module exposes a stable shared surface for profile loading, interactive
prompting, and ``--slurm-set`` parsing so multiple CLIs (infer/visualize) can
reuse identical behavior.
"""

from __future__ import annotations

from .slurm import (
    SlurmPromptConfig,
    load_slurm_profile,
    parse_slurm_set_entries,
    prompt_slurm_config,
)

__all__ = [
    "SlurmPromptConfig",
    "load_slurm_profile",
    "parse_slurm_set_entries",
    "prompt_slurm_config",
]


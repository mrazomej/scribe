"""Tests for optional Hydra/OmegaConf dependency boundaries.

These checks ensure the core package import path stays usable when optional
CLI dependencies are absent.
"""

from __future__ import annotations

import subprocess
import sys


def test_core_imports_do_not_require_hydra_or_omegaconf() -> None:
    """Core package import should succeed when optional deps are unavailable.

    Returns
    -------
    None
        Asserts that ``import scribe`` plus lazy access to core-adjacent
        exports does not import Hydra/OmegaConf at import time.
    """
    # Execute in a subprocess so the import hook does not affect other tests.
    script = """
import builtins
import importlib

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"hydra", "omegaconf"}:
        raise ModuleNotFoundError(f"No module named '{name}'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
scribe = importlib.import_module("scribe")
_ = scribe.data_loader
_ = scribe.ExperimentCatalog
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr

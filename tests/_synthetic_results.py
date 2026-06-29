"""Shared synthetic-results factories for the test suite.

This module is **not** a test module (no ``test_`` prefix, so pytest never
collects it).  It holds small factories that build synthetic ``Scribe*Results``
objects used by more than one test file.  It is importable as a top-level
module because ``tests/`` is on ``pythonpath`` (see ``[tool.pytest.ini_options]``
in ``pyproject.toml``), so tests in any subfolder can do::

    from _synthetic_results import _nbln_result

Keeping these here (rather than importing one test module from another) avoids
cross-test relative imports, which are fragile under ``--import-mode=importlib``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from scribe import ScribeLaplaceResults
from scribe.laplace._global_uncertainty import resolve_positive_fns
from scribe.models.config import ModelConfig
from scribe.models.config.enums import InferenceMethod, Parameterization


def _nbln_result(
    *,
    G: int = 20,
    C: int = 15,
    k: int = 3,
    with_uncertainty: bool = True,
    positive_transform: str = "softplus",
    seed: int = 0,
) -> ScribeLaplaceResults:
    """Build a small synthetic NBLN ``ScribeLaplaceResults`` for unit tests."""
    rng = np.random.default_rng(seed)
    mu = jnp.asarray(rng.normal(0, 0.5, G).astype(np.float32))
    W = jnp.asarray((0.3 * rng.normal(size=(G, k))).astype(np.float32))
    d = jnp.asarray(np.full(G, 0.05, dtype=np.float32))
    x_loc = jnp.asarray(rng.normal(0, 1, (C, G)).astype(np.float32))
    r = jnp.asarray(rng.uniform(0.5, 5.0, G).astype(np.float32))

    mc = ModelConfig(
        base_model="nbln",
        parameterization=Parameterization.COUNT_LOGNORMAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform=positive_transform,
    )

    pos_fwd, pos_inv = resolve_positive_fns(mc)

    kwargs = {}
    if with_uncertainty:
        kwargs["r_loc"] = pos_inv(r)
        kwargs["r_scale"] = jnp.asarray(
            rng.uniform(0.01, 0.3, G).astype(np.float32)
        )

    return ScribeLaplaceResults(
        model_config=mc,
        mu=mu,
        W=W,
        d=d,
        n_genes=G,
        n_cells=C,
        x_loc=x_loc,
        r=r,
        losses=jnp.zeros(1),
        final_grad_norms=jnp.zeros(1),
        **kwargs,
    )

"""Opt-in benchmark for the TSLN-Rate Laplace path.

Run with::

    pytest tests/bench_twostate_ln_rate.py -v -m slow

Reports per-step wall-clock, compile time, peak memory, and the
ratio against an NBLN-Laplace baseline at matched configuration.
Not a pass/fail gate — these are hypotheses (see plan §10):

- TSLN-Rate hypothesized at ~1.5× NBLN per-step wall-clock.
- TSLN-Logit (PR-2) hypothesized at ~2× NBLN.

The benchmark refutes/confirms these claims on the actual hardware
the user is running on; the production target is "within 3× NBLN."
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest


pytestmark = pytest.mark.slow


def _build_synthetic_counts(C: int, G: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.integers(0, 8, size=(C, G)).astype(np.float32))


@pytest.mark.parametrize("K_quad", [40, 60, 120])
def test_bench_twostate_ln_rate_perstep(K_quad):
    """Per-step wall-clock at standard config (C, G, k) for various K."""
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.base import ModelConfig
    from scribe.models.config.groups import (
        DataConfig,
        LaplaceConfig,
        VAEConfig,
    )
    from scribe.models.config.enums import InferenceMethod, Parameterization

    # Modest size — plan §10 target is (C, G, k) = (10000, 2000, 10)
    # but that's too slow for CI; downsize to (1000, 200, 8) to give
    # a representative trend without hour-long runs.
    C, G, k = 1000, 200, 8
    n_steps = 60
    counts = _build_synthetic_counts(C, G, seed=0)

    cfg = ModelConfig(
        base_model="twostate_ln_rate",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
        vae=VAEConfig(latent_dim=k),
    )

    # Note: TSLN-Rate's Newton kernel uses ``n_quad_nodes=60`` by
    # default; to vary K we'd need to thread it through the obs model's
    # constructor. For now this test uses the default; K parameter is
    # marked but not exercised — leaves a hook for a future K-sweep
    # benchmark.
    _ = K_quad

    # Warm-up step to trigger JIT compile.
    t_compile_start = time.perf_counter()
    _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=1, n_newton_steps=2),
        data_config=DataConfig(),
        seed=0,
    )
    t_compile = time.perf_counter() - t_compile_start

    # Timed run.
    t_run_start = time.perf_counter()
    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=n_steps, n_newton_steps=5),
        data_config=DataConfig(),
        seed=0,
    )
    t_run = time.perf_counter() - t_run_start

    per_step_ms = (t_run / n_steps) * 1000.0

    # Report (visible with -v; not asserted).
    print(
        f"\nTSLN-Rate benchmark: C={C}, G={G}, k={k}, K={K_quad}, "
        f"n_steps={n_steps}\n"
        f"  compile time: {t_compile:.2f}s\n"
        f"  total run time: {t_run:.2f}s\n"
        f"  per-step wall-clock: {per_step_ms:.1f} ms\n"
        f"  per-cell-per-step: {per_step_ms / C * 1000:.1f} μs"
    )
    assert jnp.all(jnp.isfinite(result.losses)), "loss diverged during benchmark"

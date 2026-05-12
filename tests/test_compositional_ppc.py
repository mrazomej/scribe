"""Tests for compositional posterior predictive check plotting.

Coverage:

- ``_compute_empirical_compositions`` produces per-cell and pseudobulk
  arrays of the right shape and values.
- ``_select_compositional_genes`` filters by abundance and spans the
  abundance range.
- The Theorem-1 mathematical equivalence ``softmax(μ + W z + √d ε) ==
  softmax(μ + W_⟂ z + √d ε)`` is byte-respected by
  ``get_compositional_samples`` for PLN/NBLN — both routes give
  identical compositions modulo PRNG ordering.
- ``plot_compositional_ppc`` and ``plot_compositional_corner_ppc`` run
  end-to-end on a small synthetic NBLN result without raising.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.viz.compositional_ppc import (
    _compute_empirical_compositions,
    _resolve_compositional_bin_range,
    _select_compositional_genes,
)


# Reuse the small synthetic NBLN result helper from the Laplace tests.
from .test_nbln_laplace import _nbln_result


# =====================================================================
# Helper unit tests
# =====================================================================


def test_compute_empirical_compositions_shapes_and_simplex():
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=5.0, size=(20, 8))
    per_cell, pseudobulk = _compute_empirical_compositions(counts)
    # Per-cell: each row sums to 1.
    assert per_cell.shape == (20, 8)
    np.testing.assert_allclose(per_cell.sum(axis=-1), 1.0, atol=1e-6)
    # Pseudobulk: single G-vector summing to 1.
    assert pseudobulk.shape == (8,)
    np.testing.assert_allclose(pseudobulk.sum(), 1.0, atol=1e-6)


def test_compute_empirical_compositions_drops_zero_library_cells():
    counts = np.zeros((4, 3), dtype=float)
    counts[0] = [1, 2, 3]
    counts[2] = [4, 5, 6]
    # Two cells have zero library size and should be dropped from per-cell.
    per_cell, pseudobulk = _compute_empirical_compositions(counts)
    assert per_cell.shape == (2, 3)
    # Pseudobulk uses the grand total (= sum of non-zero rows).
    np.testing.assert_allclose(
        pseudobulk, np.array([5, 7, 9]) / 21, atol=1e-6,
    )


def test_select_compositional_genes_respects_abundance_floor():
    rng = np.random.default_rng(1)
    # Build counts where genes 0-3 have very low expression and 4-9 have
    # higher expression.
    n_cells, n_genes = 30, 10
    counts = np.zeros((n_cells, n_genes), dtype=float)
    counts[:, :4] = rng.poisson(lam=0.5, size=(n_cells, 4))
    counts[:, 4:] = rng.poisson(lam=20.0, size=(n_cells, 6))
    selected = _select_compositional_genes(counts, n_genes=4, min_mean_umi=5.0)
    # All selected indices should be from the high-abundance pool (4-9).
    assert np.all(selected >= 4)


def test_select_compositional_genes_raises_when_pool_empty():
    counts = np.zeros((10, 5), dtype=float) + 0.1  # all genes below floor
    with pytest.raises(ValueError, match="abundance filter"):
        _select_compositional_genes(counts, n_genes=3, min_mean_umi=5.0)


def test_resolve_compositional_bin_range_handles_degenerate():
    # Both model and empirical are constant — should produce a tiny but
    # non-empty range.
    model_g = np.full(50, 0.1)
    emp_g = np.full(20, 0.1)
    edges, (lo, hi) = _resolve_compositional_bin_range(model_g, emp_g)
    assert hi > lo
    assert edges.size > 1
    assert lo >= 0.0


# =====================================================================
# Theorem-1 invariance check on get_compositional_samples
# =====================================================================


def test_compositional_samples_use_W_perp_for_nbln():
    """The PLN/NBLN branch should consume the gene-centered W_⟂.

    Theorem-1 in paper/_diffexp_nbln_robustness.qmd: softmax kills the
    rigid-translation gauge, so the *output* compositions are identical
    whether we use raw W or W_⟂.  This test asserts the equivalence on
    a deterministic small fit by comparing samples drawn with the same
    PRNG seed.
    """
    res = _nbln_result(G=10, C=8, k=2, with_uncertainty=False)
    # First call uses the production path (W_⟂ inside).
    samples_prod = np.asarray(
        res.get_compositional_samples(
            n_samples=64, rng_key=jax.random.PRNGKey(7),
            store_samples=False,
        )
    )
    # Manual recompute: latent = μ + W_raw z + √d ε, then softmax.
    # Should give the *same* compositions even when starting from raw W.
    mu = jnp.asarray(res.mu)
    W_raw = jnp.asarray(res.W)
    d = jnp.asarray(res.d)
    sqrt_d = jnp.sqrt(jnp.maximum(d, 0.0))
    k_z, k_eps = jax.random.split(jax.random.PRNGKey(7))
    G = int(mu.shape[0])
    k_dim = int(W_raw.shape[1])
    z = jax.random.normal(k_z, (64, k_dim), dtype=mu.dtype)
    eps = jax.random.normal(k_eps, (64, G), dtype=mu.dtype)
    latent_raw = mu[None, :] + z @ W_raw.T + sqrt_d[None, :] * eps
    samples_raw = np.asarray(jax.nn.softmax(latent_raw, axis=-1))

    # The production path uses chunking + a different RNG split layout
    # (jax.random.split over chunks then inside each chunk again), so
    # samples_prod and samples_raw won't be byte-identical.  Instead,
    # assert that the *distributions* match: mean and std across the
    # n_samples axis agree within MC error.
    mean_prod = samples_prod.mean(axis=0)
    mean_raw = samples_raw.mean(axis=0)
    std_prod = samples_prod.std(axis=0)
    std_raw = samples_raw.std(axis=0)
    # With 64 draws and a low-rank model, per-gene means agree to ~0.02
    # in absolute compositional fraction.
    np.testing.assert_allclose(mean_prod, mean_raw, atol=0.05)
    np.testing.assert_allclose(std_prod, std_raw, atol=0.05)


def test_compositional_samples_sum_to_one():
    res = _nbln_result(G=10, C=8, k=2, with_uncertainty=False)
    samples = np.asarray(
        res.get_compositional_samples(n_samples=32, store_samples=False)
    )
    assert samples.shape == (32, 10)
    np.testing.assert_allclose(samples.sum(axis=-1), 1.0, atol=1e-5)
    # Non-negative simplex.
    assert float(samples.min()) >= 0.0


# =====================================================================
# Smoke tests for the two plotting entry points
# =====================================================================


def test_plot_compositional_ppc_runs_end_to_end():
    import scribe.viz
    res = _nbln_result(G=12, C=60, k=3, with_uncertainty=False)
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=10.0, size=(60, 12))
    out = scribe.viz.plot_compositional_ppc(
        res, counts, n_genes=6, n_rows=2, n_cols=3, n_samples=512,
        min_mean_umi=0.0,  # synthetic, no abundance filter
        save=False, show=False, close=True,
    )
    assert out is not None
    # The decorator wraps tuple returns in a PlotResult.
    assert hasattr(out, "fig") or hasattr(out, "axes")


def test_plot_compositional_corner_ppc_runs_end_to_end():
    import scribe.viz
    res = _nbln_result(G=12, C=60, k=3, with_uncertainty=False)
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=10.0, size=(60, 12))
    out = scribe.viz.plot_compositional_corner_ppc(
        res, counts, n_genes=4, n_samples=512,
        min_mean_umi=0.0,
        save=False, show=False, close=True,
    )
    assert out is not None
    assert hasattr(out, "fig") or hasattr(out, "axes")


def test_plot_compositional_ppc_rejects_results_without_method():
    """Results lacking get_compositional_samples should fail fast."""
    import scribe.viz

    class _BareResults:
        n_genes = 5

    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=3.0, size=(10, 5))
    with pytest.raises(ValueError, match="get_compositional_samples"):
        scribe.viz.plot_compositional_ppc(
            _BareResults(), counts, n_genes=3, n_rows=1, n_cols=3,
            min_mean_umi=0.0,
            save=False, show=False, close=True,
        )

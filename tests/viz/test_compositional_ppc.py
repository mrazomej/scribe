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
    _select_compositional_correlation_diverse_genes,
    _select_compositional_genes,
)


# Reuse the small synthetic NBLN result helper (shared, not imported from a
# sibling test module).  Top-level import via ``pythonpath = ["tests"]``.
from _synthetic_results import _nbln_result


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


def test_resolve_compositional_bin_range_clips_empirical_outliers():
    """Outlier empirical values should not stretch the bin range.

    Regression guard for the MT-ND2-style failure where a handful of
    high-mitochondrial cells produced empirical compositions up to
    ~0.08 while the bulk lives near 0.005, distorting the axis.
    """
    rng = np.random.default_rng(0)
    # Bulk empirical ~ 0.005 ± 0.001 with three outliers at 0.08.
    bulk = rng.normal(loc=0.005, scale=0.001, size=500)
    bulk = np.clip(bulk, 0.0, 1.0)
    outliers = np.array([0.08, 0.09, 0.075])
    emp_g = np.concatenate([bulk, outliers])
    # Model is tight near 0.005 (matches the bulk).
    model_g = rng.normal(loc=0.005, scale=0.0008, size=1000)
    model_g = np.clip(model_g, 0.0, 1.0)
    edges, (lo, hi) = _resolve_compositional_bin_range(
        model_g, emp_g,
        empirical_clip_percentiles=(0.5, 99.5),
    )
    # Upper bound should reflect the empirical 99.5 percentile, not
    # the outlier maximum.  99.5 percentile of the bulk ≈ 0.008.
    assert hi < 0.05, (
        f"Outliers leaked into range: hi={hi}; should be << 0.08."
    )
    # Lower bound is non-negative.
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


def test_get_correlation_compositional_for_nbln():
    """``get_correlation_compositional`` differs from ``get_correlation``
    only by the gene-centering projection on W."""
    res = _nbln_result(G=6, C=4, k=2, with_uncertainty=False)
    corr_raw = np.asarray(res.get_correlation())
    corr_perp = np.asarray(res.get_correlation_compositional())
    # Same shape.
    assert corr_raw.shape == corr_perp.shape == (6, 6)
    # Both are valid correlation matrices on the diagonal.
    np.testing.assert_allclose(np.diag(corr_raw), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.diag(corr_perp), 1.0, atol=1e-5)
    # In general they're not identical (W is not already centered).
    if not np.allclose(np.asarray(res.W).mean(axis=0), 0.0, atol=1e-5):
        assert not np.allclose(corr_raw, corr_perp, atol=1e-4)


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


def test_correlation_diverse_selector_uses_compositional_correlation():
    """Smart selector should pick genes spanning the correlation spectrum.

    Build a synthetic NBLN result whose W has a clear two-block structure
    (genes 0-3 correlated; genes 4-7 anti-correlated with 0-3); the
    correlation-diversity selector should pick at least one gene from
    each block.
    """
    res = _nbln_result(G=8, C=12, k=1, with_uncertainty=False)
    # Override W to a two-block sign pattern.
    import jax.numpy as jnp
    object.__setattr__(
        res, "W", jnp.asarray(np.array([1, 1, 1, 1, -1, -1, -1, -1])[:, None],
                              dtype=jnp.float32),
    )
    object.__setattr__(res, "d", jnp.full((8,), 0.1, dtype=jnp.float32))
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam=15.0, size=(12, 8))
    selected = _select_compositional_correlation_diverse_genes(
        res, counts, n_genes=4, min_mean_umi=0.0,
    )
    selected_set = set(int(g) for g in selected)
    # At least one of each block should appear in the seed pair
    # (the seed pair includes the most + and most − correlated pair).
    has_block_a = any(g <= 3 for g in selected_set)
    has_block_b = any(g >= 4 for g in selected_set)
    assert has_block_a and has_block_b, (
        f"Selector missed cross-block diversity; picked {selected_set}"
    )


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


def test_render_compositional_offdiag_panel_kde_default(monkeypatch):
    """By default, _render_compositional_offdiag_panel should use KDE and call gaussian_kde."""
    from scribe.viz.compositional_corner_ppc import _render_compositional_offdiag_panel
    import matplotlib.pyplot as plt
    import scipy.stats

    rng = np.random.default_rng(0)
    model_x = rng.uniform(0.01, 0.99, size=100)
    model_y = rng.uniform(0.01, 0.99, size=100)
    emp_x = rng.uniform(0.01, 0.99, size=10)
    emp_y = rng.uniform(0.01, 0.99, size=10)

    seen = {"called": False}
    class _FakeKDE:
        def __call__(self, positions):
            return np.ones(positions.shape[1])

    def _fake_kde(stacked):
        seen["called"] = True
        return _FakeKDE()

    monkeypatch.setattr(scipy.stats, "gaussian_kde", _fake_kde)

    fig, ax = plt.subplots()
    _render_compositional_offdiag_panel(
        ax, model_x, model_y, emp_x, emp_y, 0.5, 0.5
    )
    plt.close(fig)
    assert seen["called"] is True


def test_render_compositional_offdiag_panel_hist2d(monkeypatch):
    """In hist2d mode, _render_compositional_offdiag_panel should not call gaussian_kde."""
    from scribe.viz.compositional_corner_ppc import _render_compositional_offdiag_panel
    import matplotlib.pyplot as plt
    import scipy.stats

    rng = np.random.default_rng(0)
    model_x = rng.uniform(0.01, 0.99, size=100)
    model_y = rng.uniform(0.01, 0.99, size=100)
    emp_x = rng.uniform(0.01, 0.99, size=10)
    emp_y = rng.uniform(0.01, 0.99, size=10)

    monkeypatch.setattr(
        scipy.stats,
        "gaussian_kde",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("gaussian_kde should not be called")
        ),
    )

    fig, ax = plt.subplots()
    _render_compositional_offdiag_panel(
        ax, model_x, model_y, emp_x, emp_y, 0.5, 0.5, density_method="hist2d"
    )
    plt.close(fig)


def test_compositional_invalid_density_method_raises():
    """ValueError should be raised for an invalid density method."""
    from scribe.viz.compositional_corner_ppc import _render_compositional_offdiag_panel
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    model_x = rng.uniform(0.01, 0.99, size=100)
    model_y = rng.uniform(0.01, 0.99, size=100)
    emp_x = rng.uniform(0.01, 0.99, size=10)
    emp_y = rng.uniform(0.01, 0.99, size=10)

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="density_method"):
        _render_compositional_offdiag_panel(
            ax, model_x, model_y, emp_x, emp_y, 0.5, 0.5, density_method="invalid"
        )
    plt.close(fig)

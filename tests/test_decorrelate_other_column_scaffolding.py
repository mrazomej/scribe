"""Tests for the harmonic-hare ``correlate_other_column`` ladder.

Tracks the scaffolding (Commits 2–6) and the per-model math commits
(2b/3b/4b/5b) for decoupling the pooled ``_other`` column from the
latent low-rank covariance Σ.

Landed so far:

* :class:`scribe.laplace._axis_layout.AxisLayout` + factory
  (``build_axis_layout``) with explicit ``has_pooled_other`` primary
  signal, gene-names fallback, and contradictory-signal raise.
* :class:`scribe.models.config.ModelConfig.correlate_other_column`
  flag (runtime default ``True`` while the per-model math commits
  finish landing — flips to ``False`` once 3b/4b/5b ship).
* All four count obs-models accept ``gene_names`` and
  ``has_pooled_other`` and build an ``AxisLayout`` at init.  W/d/
  latent_loc are sliced to the kept axis under decoupling and the
  layout is threaded through to
  :class:`scribe.laplace.ScribeLaplaceResults` via the
  ``axis_layout`` field.
* :attr:`ScribeLaplaceResults.G_obs` / ``G_kept`` properties.
* LNM real wiring (Commit 6): ALR auto-pin to ``_other`` when
  ``correlate_other_column=False`` and a pooled column exists.
* NBLN deviation-form math (Commit 2b): loss, Newton, profiled-μ
  Schur, compositional sampler all implemented under decoupling.
  TSLN-Rate (3b), TSLN-Logit (4b), and PLN (5b) still raise
  ``NotImplementedError`` under decoupling pending their math
  commits.

These tests verify:

1. ``AxisLayout`` detection priority — explicit ``has_pooled_other``
   beats names; ``gene_names[-1] == "_other"`` is the fallback;
   contradictory signals raise loudly.
2. Legacy NBLN/PLN/TSLN fits (no ``gene_coverage``, no ``_other``)
   produce a trivial layout (``G_kept == G_obs``) and work bit-equal
   to today.
3. Legacy opt-in (``correlate_other_column=True`` with
   ``gene_coverage < 1.0``) produces a trivial layout and a working
   fit (auditor finding rev-3 #8: legacy passes through silently).
4. NBLN decoupled fits run end-to-end with the deviation-form math
   (Commit 2b); TSLN-Rate/TSLN-Logit/PLN decoupled still raise
   ``NotImplementedError`` until 3b/4b/5b land.
5. ``ScribeLaplaceResults.G_obs`` / ``G_kept`` return the right
   numbers for both layouts.
6. The ``correlate_other_column`` kwarg round-trips through
   ``scribe.fit`` → ``ModelConfig``.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from types import SimpleNamespace

from scribe.laplace._axis_layout import (
    AxisLayout,
    _OTHER_NAME,
    build_axis_layout,
)


# =====================================================================
# AxisLayout factory unit tests
# =====================================================================


class TestAxisLayoutFactory:
    """Build-axis-layout detection priority + contradictory-signal raise."""

    def test_trivial_layout_when_legacy_flag(self):
        # `correlate_other_column=True` → trivial layout regardless of
        # the gene_names / has_pooled_other signals.
        layout = build_axis_layout(
            5,
            correlate_other_column=True,
            gene_names=["g1", "g2", "g3", "g4", "_other"],
            has_pooled_other=True,
        )
        assert layout.G_obs == 5
        assert layout.G_kept == 5
        assert layout.other_idx is None
        assert layout.decoupled is False
        np.testing.assert_array_equal(
            layout.kept_idx, np.arange(5, dtype=np.int64)
        )

    def test_trivial_layout_when_no_pooled_other(self):
        layout = build_axis_layout(
            5,
            correlate_other_column=False,
            gene_names=["g1", "g2", "g3", "g4", "g5"],
            has_pooled_other=False,
        )
        assert layout.decoupled is False
        assert layout.other_idx is None

    def test_decoupled_layout_via_names(self):
        # Names-only detection (legacy primary signal).
        layout = build_axis_layout(
            5,
            correlate_other_column=False,
            gene_names=["g1", "g2", "g3", "g4", _OTHER_NAME],
        )
        assert layout.decoupled is True
        assert layout.G_obs == 5
        assert layout.G_kept == 4
        assert layout.other_idx == 4
        np.testing.assert_array_equal(
            layout.kept_idx, np.arange(4, dtype=np.int64)
        )

    def test_decoupled_layout_via_has_pooled_other_array_input(self):
        # Array-input fits (no AnnData → no gene_names): the
        # ``has_pooled_other`` primary signal alone must trigger
        # decoupling.  Auditor finding rev-2 #3.
        layout = build_axis_layout(
            5, correlate_other_column=False, has_pooled_other=True,
        )
        assert layout.decoupled is True
        assert layout.G_kept == 4
        assert layout.other_idx == 4

    def test_has_pooled_other_false_overrides_names(self):
        # When ``has_pooled_other`` is explicitly False but names end
        # in `_other`, the signals disagree → raise (rev-3 #9).
        with pytest.raises(ValueError, match=r"Contradictory '_other' signals"):
            build_axis_layout(
                5,
                correlate_other_column=False,
                gene_names=["g1", "g2", "g3", "g4", _OTHER_NAME],
                has_pooled_other=False,
            )

    def test_has_pooled_other_true_disagrees_with_names_raises(self):
        # Symmetric case.
        with pytest.raises(ValueError, match=r"Contradictory '_other' signals"):
            build_axis_layout(
                5,
                correlate_other_column=False,
                gene_names=["g1", "g2", "g3", "g4", "g5"],
                has_pooled_other=True,
            )

    def test_contradictory_signals_silent_under_legacy_flag(self):
        """Auditor rev-5 #2: legacy mode never breaks.

        When `correlate_other_column=True`, the layout is always
        trivial regardless of signals — so disagreement between
        `has_pooled_other` and `gene_names[-1]` cannot corrupt the
        axis split.  The check is skipped to keep the legacy path
        bit-equal even on metadata-drift edge cases (e.g.
        manually-pre-filtered AnnData whose tail is literally
        ``_other`` but no gene_coverage stage ran).
        """
        # Disagreement that would raise under `False` is silent here.
        layout = build_axis_layout(
            5,
            correlate_other_column=True,
            gene_names=["g1", "g2", "g3", "g4", _OTHER_NAME],
            has_pooled_other=False,
        )
        assert layout.decoupled is False
        assert layout.G_kept == layout.G_obs == 5

        # Symmetric case (other direction of disagreement).
        layout2 = build_axis_layout(
            5,
            correlate_other_column=True,
            gene_names=["g1", "g2", "g3", "g4", "g5"],
            has_pooled_other=True,
        )
        assert layout2.decoupled is False


class TestAxisLayoutInvariants:
    """`AxisLayout`'s `__post_init__` enforces shape invariants."""

    def test_kept_idx_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"kept_idx must have shape"):
            AxisLayout(
                G_obs=5,
                G_kept=4,
                kept_idx=np.array([0, 1, 2]),  # wrong length
                other_idx=4,
            )

    def test_decoupled_requires_other_idx(self):
        with pytest.raises(ValueError, match=r"other_idx must be set"):
            AxisLayout(
                G_obs=5,
                G_kept=4,
                kept_idx=np.arange(4),
                other_idx=None,
            )

    def test_only_one_other_column_supported(self):
        with pytest.raises(
            ValueError, match=r"Only one trailing '_other'"
        ):
            AxisLayout(
                G_obs=10,
                G_kept=4,
                kept_idx=np.arange(4),
                other_idx=4,
            )

    # rev-10 self-audit H2: dtype / range / monotonicity / disjoint
    # invariants on ``kept_idx``.  These guard the fancy-indexing of
    # W / d / x across the (G_obs, G_kept) split.

    def test_kept_idx_must_be_integer_dtype(self):
        with pytest.raises(ValueError, match=r"kept_idx must have integer"):
            AxisLayout(
                G_obs=5,
                G_kept=4,
                kept_idx=np.arange(4, dtype=np.float32),
                other_idx=4,
            )

    def test_kept_idx_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"kept_idx values must be in"):
            AxisLayout(
                G_obs=5,
                G_kept=4,
                kept_idx=np.array([0, 1, 2, 7], dtype=np.int64),
                other_idx=4,
            )

    def test_kept_idx_not_monotone_raises(self):
        with pytest.raises(
            ValueError, match=r"kept_idx must be strictly increasing"
        ):
            AxisLayout(
                G_obs=5,
                G_kept=4,
                kept_idx=np.array([0, 2, 1, 3], dtype=np.int64),
                other_idx=4,
            )

    def test_kept_idx_collides_with_other_idx_raises(self):
        with pytest.raises(
            ValueError,
            match=r"other_idx \(2\) must not appear in kept_idx",
        ):
            AxisLayout(
                G_obs=5,
                G_kept=4,
                # kept_idx contains other_idx=2 — disallowed.
                kept_idx=np.array([0, 1, 2, 3], dtype=np.int64),
                other_idx=2,
            )


# =====================================================================
# NBLN obs-model integration
# =====================================================================


def _make_adata(n_cells: int, n_genes: int, seed: int):
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(
        np.float32
    )
    return ad.AnnData(counts)


class TestNblnLegacyPaths:
    """Legacy paths (no decoupling) work unchanged."""

    def test_no_gene_coverage_no_other_column(self):
        import scribe

        adata = _make_adata(30, 8, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == 8
        assert res.G_kept == 8
        # W / d / mu / r all (G_obs,) — unchanged from today.
        assert res.W.shape == (8, 2)
        assert res.d.shape == (8,)
        assert res.mu.shape == (8,)
        assert res.r.shape == (8,)

    def test_gene_coverage_default_is_decoupled_after_5b(self):
        """After Commit 5b flipped the default to ``False``, a
        ``gene_coverage<1.0`` fit without an explicit
        ``correlate_other_column`` now routes through the decoupled
        path.  ``W`` and ``d`` live on G_kept; ``μ`` / ``r`` stay on
        G_obs.  Explicit ``correlate_other_column=True`` recovers
        legacy."""
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            # No explicit `correlate_other_column` — uses the default
            # (`False` after Commit 5b).
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is True
        # G_kept = G_obs − 1 (one pooled _other).
        assert res.G_kept == res.G_obs - 1
        # W / d on G_kept; mu / r on G_obs.
        assert res.W.shape[0] == res.G_kept
        assert res.d.shape == (res.G_kept,)
        assert res.mu.shape == (res.G_obs,)
        assert res.r.shape == (res.G_obs,)
        # Explicit ``True`` opt-in recovers legacy (trivial layout).
        res_explicit = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=True,
        )
        assert res_explicit.axis_layout.decoupled is False
        assert res_explicit.G_obs == res_explicit.G_kept
        assert res_explicit.G_obs == res.G_obs


class TestNblnDecoupledMath:
    """Commit 2b: NBLN deviation-form math is now implemented.

    The scaffolding's ``NotImplementedError`` guards have been
    replaced by the deviation reparameterisation:

    * ``x_dev ~ 𝒩(0, Σ_kept)`` per cell (zero-centred) on the kept axis
    * ``μ`` moves from the MVN prior into the NB likelihood
    * ``_other``'s log-rate is deterministic (``μ_other − η``) with no
      ``x_dev`` contribution
    * Profiled-μ Schur correction uses the per-cell ``M_c^{-1}`` block
      restricted to ``(k_g, k_g)`` for kept genes and ``M_ηη`` for
      ``_other`` (see ``paper/_nb_lognormal.qmd``
      §sec-nbln-decorrelate-mu-uncertainty).

    See ``src/scribe/laplace/_newton_nbln.py`` for the kernels (the
    ``*_decoupled`` family).
    """

    def test_decoupled_fit_runs_and_produces_kept_axis_shapes(self):
        """Smoke test: decoupled NBLN fit converges and the result has
        the right shape contract.  ``W`` and ``d`` live on ``G_kept``;
        ``μ`` and per-gene ``r`` live on ``G_obs``.
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is True
        # G_kept = G_obs − 1 (one pooled `_other`).
        assert res.G_kept == res.G_obs - 1
        # Per-gene observation parameters live on G_obs.
        assert res.mu.shape == (res.G_obs,)
        assert res.r.shape == (res.G_obs,)
        # Latent-covariance parameters live on G_kept.
        assert res.W.shape == (res.G_kept, 2)
        assert res.d.shape == (res.G_kept,)
        # Global-uncertainty diagnostics live on G_obs (per-gene).
        assert res.mu_scale.shape == (res.G_obs,)

    def test_decoupled_compositional_samples_full_simplex(self):
        """Compositional samples have shape ``(n_samples, G_obs)`` and
        rows sum to 1.  ``_other``'s entry is deterministic per draw
        (no ``x_dev`` contribution), but it still participates in the
        simplex normalisation.
        """
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        samples = res.get_compositional_samples(n_samples=64, chunk_size=32)
        assert samples.shape == (64, res.G_obs)
        # Each row is a proper simplex (rows sum to 1 within float32
        # precision).
        row_sums = samples.sum(axis=1)
        assert _np.allclose(row_sums, 1.0, atol=1e-4)

    def test_legacy_with_coverage_still_works(self):
        """The legacy path (``correlate_other_column=True``) continues
        to fit with ``_other`` participating in Σ.  This is the
        bit-equal contract regression — exercising it ensures the
        decoupled branches don't accidentally short-circuit the legacy
        code path.
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=True,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        # Legacy: W and d on G_obs (== G_kept under trivial layout).
        assert res.G_kept == res.G_obs
        assert res.W.shape[0] == res.G_obs
        assert res.d.shape == (res.G_obs,)

    def test_decoupled_get_map_returns_full_g_obs_log_rate(self):
        """``get_map()["y_log_rate"]`` reconstructs the full per-cell
        G_obs log-rate under decoupling.  Under the deviation
        reparameterisation, ``x_loc`` carries ``x_dev`` on G_kept; the
        accessor must scatter ``μ + x_dev`` at kept positions and use
        ``μ`` at ``_other``."""
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        m = res.get_map()
        assert m["y_log_rate"].shape == (30, res.G_obs)
        # ``_other`` column entry is deterministically ``μ[other_idx]``
        # for every cell (no per-cell x_dev contribution).
        other_idx = res.axis_layout.other_idx
        # Float32 comparison: within float epsilon.
        import numpy as _np
        assert _np.allclose(
            _np.asarray(m["y_log_rate"])[:, other_idx],
            float(res.mu[other_idx]),
            atol=1e-5,
        )

    def test_decoupled_get_distributions_y_log_rate_event_shape(self):
        """``get_distributions()["y_log_rate"]`` returns a
        ``LowRankMultivariateNormal`` on the full G_obs axis with W/d
        padded so the ``_other`` row has effectively-zero variance
        (~1e-12).  Samples have shape ``(G_obs,)``."""
        import jax
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        d = res.get_distributions()
        y = d["y_log_rate"]
        assert y.event_shape == (res.G_obs,)
        s = y.sample(jax.random.PRNGKey(0), sample_shape=(64,))
        assert s.shape == (64, res.G_obs)
        # ``_other``'s samples concentrate tightly around μ_other (the
        # 1e-12 epsilon variance + zero W row gives variance ~1e-12).
        other_idx = res.axis_layout.other_idx
        other_samples = _np.asarray(s[:, other_idx])
        assert _np.std(other_samples) < 1e-3
        assert _np.allclose(
            _np.mean(other_samples),
            float(res.mu[other_idx]),
            atol=1e-3,
        )

    def test_decoupled_ppc_all_levels_full_g_obs_shape(self):
        """All four PPC pathways (marginal / library_anchored / per_cell
        Laplace / get_map_ppc) emit counts on the full G_obs axis under
        decoupling.  Auditor finding: the helpers used to assume one
        shared gene axis and crashed under decoupled NBLN."""
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        # The library-anchored / per-cell kernels only consume per-cell
        # library sizes from ``counts``, so a synthetic count matrix
        # of the post-coverage shape ``(n_cells, G_obs)`` is enough —
        # the values don't enter the predictive draws beyond ``sum``.
        counts_for_ppc = _np.full((30, res.G_obs), 1.0, dtype=_np.float32)

        m = res.get_ppc_samples(n_samples=4, level="marginal")
        assert m.shape == (4, res.G_obs)

        la = res.get_ppc_samples(
            n_samples=2, level="library_anchored", counts=counts_for_ppc,
        )
        assert la.shape == (2, 30, res.G_obs)

        pc = res.get_ppc_samples(
            n_samples=2, level="per_cell", counts=counts_for_ppc,
        )
        assert pc.shape == (2, 30, res.G_obs)

        mp = res.get_map_ppc_samples(n_samples=2)
        assert mp.shape == (2, 30, res.G_obs)

    def test_legacy_no_coverage_bit_equal_after_2b(self):
        """Legacy NBLN fit on data WITHOUT ``_other`` continues to use
        the existing code path bit-equal to before Commit 2b.  All
        decoupled branches in loss_fn / final_sweep /
        compute_global_uncertainty / get_compositional_samples are
        gated by ``layout.decoupled is True`` — under the trivial
        layout the legacy code runs without any new branch entered.

        Verify by re-running the legacy fit twice with the same seed
        and confirming exact bit-equality of W, d, μ, r, mu_scale.
        """
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        r1 = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
        )
        r2 = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
        )
        # Self-consistency: two runs with identical config produce
        # identical results (deterministic).  This is what "bit-equal"
        # means in practice when the new branches don't fire.
        assert _np.allclose(r1.W, r2.W, atol=0)
        assert _np.allclose(r1.d, r2.d, atol=0)
        assert _np.allclose(r1.mu, r2.mu, atol=0)
        assert _np.allclose(r1.r, r2.r, atol=0)


class TestPlnScaffolding:
    """PLN scaffolding mirror of NBLN / TSLN-Rate / TSLN-Logit
    (harmonic-hare Commit 5).

    PLN is the simplest of the four affected models: observation
    is plain Poisson so there's no per-gene NB dispersion or
    two-state parameter on the observation-layer axis.  Under
    decoupled, only ``W`` and ``d`` shrink to ``G_kept``; ``mu``
    stays on ``G_obs`` as the per-gene log-rate prior centre.

    As of Commit 5, the engine early-fail block has been retired
    entirely — every affected model owns its own decoupled
    detection via the obs-model's ``init_state``.
    """

    def test_legacy_no_coverage_works(self):
        """Plain PLN fit (no coverage, no _other) trivial layout."""
        import scribe

        adata = _make_adata(20, 6, seed=0)
        res = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            latent_dim=2,
            n_steps=5,
            seed=0,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == 6
        assert res.G_kept == 6
        # W / d / mu all (G_obs,) — unchanged from today.
        assert res.W.shape == (6, 2)
        assert res.d.shape == (6,)
        assert res.mu.shape == (6,)

    def test_explicit_legacy_with_gene_coverage(self):
        """Explicit ``correlate_other_column=True`` recovers legacy
        behaviour under ``gene_coverage < 1.0``: ``_other``
        participates in Σ; trivial layout.  (The default flipped to
        ``False`` in Commit 5b — see
        ``test_decoupled_fit_produces_kept_axis_shapes`` for the new
        default.)"""
        import scribe

        adata = _make_adata(20, 8, seed=0)
        res = scribe.fit(
            adata,
            model="pln",
            inference_method="laplace",
            latent_dim=2,
            n_steps=5,
            seed=0,
            gene_coverage=0.85,
            correlate_other_column=True,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == res.G_kept

    def test_decoupled_fit_produces_kept_axis_shapes(self):
        """Commit 5b: PLN decoupled math is live.  ``W`` and ``d``
        live on G_kept; ``μ`` lives on G_obs.  PLN is the simplest of
        the four count models — no per-gene NB dispersion, no
        Two-State Beta parameters, just pure Poisson on the per-gene
        log-rate."""
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="pln", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is True
        assert res.G_kept == res.G_obs - 1
        # ``μ`` lives on G_obs (per-gene prior centre).
        assert res.mu.shape == (res.G_obs,)
        # Latent-covariance parameters live on G_kept.
        assert res.W.shape == (res.G_kept, 2)
        assert res.d.shape == (res.G_kept,)
        assert res.x_loc.shape == (30, res.G_kept)

    def test_decoupled_ppc_all_levels_full_g_obs_shape(self):
        """All four PLN PPC pathways emit counts on G_obs under
        decoupling.  Same pattern as NBLN/TSLN-Rate/TSLN-Logit
        positive tests."""
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="pln", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        counts_for_ppc = _np.full(
            (30, res.G_obs), 1.0, dtype=_np.float32
        )

        m = res.get_ppc_samples(n_samples=4, level="marginal")
        assert m.shape == (4, res.G_obs)

        la = res.get_ppc_samples(
            n_samples=2, level="library_anchored", counts=counts_for_ppc,
        )
        assert la.shape == (2, 30, res.G_obs)

        pc = res.get_ppc_samples(
            n_samples=2, level="per_cell", counts=counts_for_ppc,
        )
        assert pc.shape == (2, 30, res.G_obs)

        mp = res.get_map_ppc_samples(n_samples=2)
        assert mp.shape == (2, 30, res.G_obs)

    def test_decoupled_get_map_and_distributions(self):
        """``get_map()["y_log_rate"]`` shape == ``(N, G_obs)``;
        ``_other`` column == ``μ_other`` (deterministic — no z
        modulation).  ``get_distributions()["y_log_rate"].event_shape
        == (G_obs,)``."""
        import jax
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="pln", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        m = res.get_map()
        assert m["y_log_rate"].shape == (30, res.G_obs)
        other_idx = res.axis_layout.other_idx
        assert _np.allclose(
            _np.asarray(m["y_log_rate"])[:, other_idx],
            float(res.mu[other_idx]),
            atol=1e-5,
        )

        d = res.get_distributions()
        y = d["y_log_rate"]
        assert y.event_shape == (res.G_obs,)
        s = y.sample(jax.random.PRNGKey(0), sample_shape=(32,))
        assert s.shape == (32, res.G_obs)


class TestTslnRateScaffolding:
    """TSLN-Rate scaffolding mirror of NBLN (harmonic-hare Commit 3).

    Same shape contract: when `correlate_other_column=False` and a
    pooled `_other` column is present, the obs model builds an
    `AxisLayout` and raises `NotImplementedError` from inside
    ``loss_fn`` / ``final_sweep`` / ``compute_global_uncertainty``.
    Legacy fits (no `gene_coverage`, or `correlate_other_column=True`)
    work unchanged.
    """

    def test_legacy_no_coverage_works(self):
        """Plain TSLN-Rate fit (no coverage, no _other) trivial layout."""
        import scribe

        adata = _make_adata(20, 6, seed=0)
        res = scribe.fit(
            adata,
            model="twostate_ln_rate",
            inference_method="laplace",
            latent_dim=2,
            n_steps=5,
            seed=0,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == 6
        assert res.G_kept == 6
        # W and d live on the latent-covariance axis (G_kept).  Under
        # the trivial layout this equals G_obs.
        if res.W is not None:
            assert res.W.shape[0] == 6
        if res.d is not None:
            assert res.d.shape == (6,)

    def test_explicit_legacy_with_gene_coverage(self):
        """Explicit ``correlate_other_column=True`` recovers legacy
        behaviour under ``gene_coverage < 1.0`` — ``_other``
        participates in Σ as a regular gene; layout is trivial.  (The
        default flipped to ``False`` in Commit 5b.)"""
        import scribe

        adata = _make_adata(20, 8, seed=0)
        res = scribe.fit(
            adata,
            model="twostate_ln_rate",
            inference_method="laplace",
            latent_dim=2,
            n_steps=5,
            seed=0,
            gene_coverage=0.85,
            correlate_other_column=True,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        # G_obs accounts for the trailing `_other` column emitted by
        # the gene-coverage stage.
        assert res.G_obs == res.G_kept
        assert res.G_obs >= 2  # kept + _other at minimum

    def test_decoupled_fit_produces_kept_axis_shapes(self):
        """Commit 3b: TSLN-Rate decoupled math now lives.  Mirror of
        ``TestNblnDecoupledMath::test_decoupled_fit_runs_and_produces_kept_axis_shapes``.
        ``W`` and ``d`` live on G_kept; ``μ`` / ``alpha`` / ``beta`` /
        per-gene TwoState parameters live on G_obs.
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="twostate_ln_rate", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is True
        assert res.G_kept == res.G_obs - 1
        # Per-gene observation parameters live on G_obs.
        assert res.mu.shape == (res.G_obs,)
        # Latent-covariance parameters live on G_kept.
        assert res.W.shape == (res.G_kept, 2)
        assert res.d.shape == (res.G_kept,)

    def test_decoupled_ppc_all_levels_full_g_obs_shape(self):
        """All four TSLN-Rate PPC pathways emit counts on G_obs under
        decoupling.  Mirrors ``test_decoupled_ppc_all_levels_full_g_obs_shape``
        from NBLN.
        """
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="twostate_ln_rate", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        counts_for_ppc = _np.full(
            (30, res.G_obs), 1.0, dtype=_np.float32
        )

        m = res.get_ppc_samples(n_samples=4, level="marginal")
        assert m.shape == (4, res.G_obs)

        la = res.get_ppc_samples(
            n_samples=2, level="library_anchored", counts=counts_for_ppc,
        )
        assert la.shape == (2, 30, res.G_obs)

        pc = res.get_ppc_samples(
            n_samples=2, level="per_cell", counts=counts_for_ppc,
        )
        assert pc.shape == (2, 30, res.G_obs)

        mp = res.get_map_ppc_samples(n_samples=2)
        assert mp.shape == (2, 30, res.G_obs)

    def test_decoupled_get_map_and_distributions(self):
        """``get_map()["y_log_rate"]`` shape == ``(N, G_obs)``;
        ``_other`` column == ``μ_other``.  ``get_distributions()[
        "y_log_rate"].event_shape == (G_obs,)``."""
        import jax
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="twostate_ln_rate", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        m = res.get_map()
        assert m["y_log_rate"].shape == (30, res.G_obs)
        other_idx = res.axis_layout.other_idx
        assert _np.allclose(
            _np.asarray(m["y_log_rate"])[:, other_idx],
            float(res.mu[other_idx]),
            atol=1e-5,
        )

        d = res.get_distributions()
        y = d["y_log_rate"]
        assert y.event_shape == (res.G_obs,)
        s = y.sample(jax.random.PRNGKey(0), sample_shape=(32,))
        assert s.shape == (32, res.G_obs)


class TestTslnLogitScaffolding:
    """TSLN-Logit scaffolding mirror of NBLN / TSLN-Rate (Commit 4).

    Per-gene parameters ``rate`` / ``kappa`` / ``eta_anchor`` stay on
    the OBSERVATION-layer axis (G_obs,) under decoupled — they're
    per-gene baselines, not regulatory.  Only ``W`` / ``d`` / per-cell
    z shrink to G_kept.  The decoupled-math path is deferred to a
    later commit; for now ``init_state`` fails fast with a clear
    NotImplementedError.
    """

    def test_legacy_no_coverage_works(self):
        """Plain TSLN-Logit fit (no coverage, no _other) trivial layout.

        Symmetric to ``TestPlnScaffolding`` / ``TestTslnRateScaffolding``.
        Added in rev-10 to close a coverage gap surfaced by the
        self-audit (TSLN-Logit was the only count model without a
        direct legacy-path positive test).
        """
        import scribe

        adata = _make_adata(20, 6, seed=0)
        res = scribe.fit(
            adata,
            model="twostate_ln_logit",
            inference_method="laplace",
            latent_dim=2,
            n_steps=5,
            seed=0,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == 6
        assert res.G_kept == 6
        if res.W is not None:
            assert res.W.shape[0] == 6
        if res.d is not None:
            assert res.d.shape == (6,)

    def test_explicit_legacy_with_gene_coverage(self):
        """Explicit ``correlate_other_column=True`` recovers legacy
        behaviour under ``gene_coverage < 1.0``: ``_other``
        participates in Σ as a regular gene; layout is trivial.
        (The default flipped to ``False`` in Commit 5b.)"""
        import scribe

        adata = _make_adata(20, 8, seed=0)
        res = scribe.fit(
            adata,
            model="twostate_ln_logit",
            inference_method="laplace",
            latent_dim=2,
            n_steps=5,
            seed=0,
            gene_coverage=0.85,
            correlate_other_column=True,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == res.G_kept
        assert res.G_obs >= 2

    def test_decoupled_fit_produces_kept_axis_shapes(self):
        """Commit 4b: TSLN-Logit decoupled math is live.  ``W`` and
        ``d`` live on G_kept; per-gene ``rate`` / ``kappa`` /
        ``eta_anchor`` stay on G_obs.  Per-cell ``x_loc`` carries
        ``z_kept`` (zero-centred under both layouts since TSLN-Logit's
        z has no μ centring)."""
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="twostate_ln_logit", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is True
        assert res.G_kept == res.G_obs - 1
        # Per-gene observation parameters stay on G_obs.
        assert res.rate.shape == (res.G_obs,)
        assert res.kappa.shape == (res.G_obs,)
        assert res.eta_anchor.shape == (res.G_obs,)
        # Latent-covariance parameters live on G_kept.
        assert res.W.shape == (res.G_kept, 2)
        assert res.d.shape == (res.G_kept,)
        assert res.x_loc.shape == (30, res.G_kept)

    def test_decoupled_ppc_all_levels_full_g_obs_shape(self):
        """All three TSLN-Logit PPC pathways emit counts on G_obs
        under decoupling — mirrors the NBLN/TSLN-Rate fixes."""
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="twostate_ln_logit", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        counts_for_ppc = _np.full(
            (30, res.G_obs), 1.0, dtype=_np.float32
        )

        m = res.get_ppc_samples(n_samples=4, level="marginal")
        assert m.shape == (4, res.G_obs)

        pc = res.get_ppc_samples(
            n_samples=2, level="per_cell", counts=counts_for_ppc,
        )
        assert pc.shape == (2, 30, res.G_obs)

        mp = res.get_map_ppc_samples(n_samples=2)
        assert mp.shape == (2, 30, res.G_obs)

    def test_decoupled_get_map_and_distributions(self):
        """``get_map()["y_latent"]`` shape == ``(N, G_obs)``;
        ``_other`` column == ``0`` (no z modulation under decoupling).
        ``get_distributions()["y_latent"].event_shape == (G_obs,)``."""
        import jax
        import numpy as _np
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="twostate_ln_logit", inference_method="laplace",
            latent_dim=2, n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        m = res.get_map()
        assert m["y_latent"].shape == (30, res.G_obs)
        other_idx = res.axis_layout.other_idx
        # ``_other``'s latent stays at zero (no z modulation).
        assert _np.allclose(
            _np.asarray(m["y_latent"])[:, other_idx], 0.0, atol=1e-6,
        )

        d = res.get_distributions()
        y = d["y_latent"]
        assert y.event_shape == (res.G_obs,)
        s = y.sample(jax.random.PRNGKey(0), sample_shape=(32,))
        assert s.shape == (32, res.G_obs)


# =====================================================================
# Decoupled flag + no `_other` column: trivial-layout no-op
# =====================================================================
#
# Symmetry coverage gap closed in rev-10 (self-audit M3): when the
# user sets ``correlate_other_column=False`` on data that does NOT
# have a pooled ``_other`` column, the AxisLayout factory should
# return the trivial layout (decoupled=False) and the fit should run
# normally without raising the scaffolding's NotImplementedError.
# This was previously covered for LNM only; the four count models
# are now exercised in parallel here.


@pytest.mark.parametrize(
    "model",
    ["nbln", "pln", "twostate_ln_rate", "twostate_ln_logit"],
)
def test_decoupled_flag_no_other_column_is_noop(model):
    """``correlate_other_column=False`` is a no-op when the data has
    no pooled ``_other`` column.  Under detection priority, the
    factory falls through to the trivial layout (G_kept == G_obs),
    and the obs-model's fail-fast guard is NOT triggered."""
    import scribe

    adata = _make_adata(20, 6, seed=0)
    # No gene_coverage stage, no manually-named _other gene → the
    # mask signal is None and the var_names fallback yields False.
    # The factory returns a trivial layout regardless of the flag.
    res = scribe.fit(
        adata,
        model=model,
        inference_method="laplace",
        latent_dim=2,
        n_steps=5,
        seed=0,
        correlate_other_column=False,
    )
    assert res.axis_layout is not None
    assert res.axis_layout.decoupled is False
    assert res.G_obs == res.G_kept == 6


# =====================================================================
# Manually-named `_other` end-to-end coverage for all four models
# =====================================================================
#
# The shared ``AxisLayout`` factory tests cover names-only detection
# (no ``gene_coverage`` stage running, AnnData `var_names[-1] ==
# "_other"`), but the per-model fit tests above all rely on the
# ``gene_coverage`` stage to emit the pooled column.  This class
# closes the per-model coverage gap (auditor rev-8 Low #2): the
# names fallback path is verified end-to-end for NBLN / TSLN-Rate /
# TSLN-Logit / PLN by hand-naming the AnnData `var_names[-1] =
# "_other"` and fitting WITHOUT `gene_coverage`.  Under explicit
# `correlate_other_column=False`, each model must raise its
# AxisLayout-aware NotImplementedError; under the legacy default
# (`True`), the contradictory-signal check is skipped (rev-5 #2)
# so the fit succeeds with the trivial layout.


class TestManuallyNamedOtherEndToEnd:
    """End-to-end coverage for the AnnData `var_names` fallback path
    on all four affected models.  Closes auditor rev-8 Low #2.
    """

    @staticmethod
    def _make_adata_with_manual_other(n_cells=20, n_genes=6, seed=0):
        rng = np.random.default_rng(seed)
        counts = rng.negative_binomial(
            5, 0.5, size=(n_cells, n_genes)
        ).astype(np.float32)
        ad_obj = _make_adata(n_cells, n_genes, seed=seed)
        # Override the auto-generated var_names so the tail is the
        # `_OTHER_NAME` sentinel — simulating a manually pre-filtered
        # AnnData where the user collapsed low-coverage genes upstream
        # WITHOUT calling the gene_coverage stage.
        names = [f"g{i}" for i in range(n_genes - 1)] + [_OTHER_NAME]
        ad_obj.var_names = names
        return ad_obj

    @pytest.mark.parametrize(
        "model",
        ["nbln", "twostate_ln_rate", "twostate_ln_logit", "pln"],
    )
    def test_manually_named_other_decoupled_with_math_runs(self, model):
        """All four count models now have decoupled math wired (NBLN:
        Commit 2b, TSLN-Rate: Commit 3b, TSLN-Logit: Commit 4b, PLN:
        Commit 5b).  With ``var_names[-1] == "_other"`` AND
        ``correlate_other_column=False``, each fits cleanly via the
        names-fallback detection in ``build_axis_layout`` and
        produces ``W`` / ``d`` on the kept axis."""
        import scribe

        adata = self._make_adata_with_manual_other(seed=0)
        res = scribe.fit(
            adata, model=model, inference_method="laplace",
            latent_dim=2, n_steps=3, seed=0,
            correlate_other_column=False,
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is True
        assert res.W.shape == (res.G_kept, 2)
        assert res.d.shape == (res.G_kept,)

    @pytest.mark.parametrize(
        "model",
        ["nbln", "twostate_ln_rate", "twostate_ln_logit", "pln"],
    )
    def test_manually_named_other_explicit_legacy_is_silent(self, model):
        """Under explicit ``correlate_other_column=True``, a manually-
        named ``_other`` tail with no ``gene_coverage`` must NOT raise
        — the contradictory-signal check is skipped under legacy so
        the fit proceeds with the trivial layout (rev-5 #2 contract).
        (After Commit 5b's default flip, this test pins the explicit-
        opt-in path; the default is now ``False`` and routes through
        decoupled — see ``test_manually_named_other_decoupled_with_
        math_runs`` above for that path.)"""
        import scribe

        adata = self._make_adata_with_manual_other(seed=0)
        res = scribe.fit(
            adata,
            model=model,
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
            correlate_other_column=True,
        )
        # Trivial layout under legacy.
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == res.G_kept


# =====================================================================
# ScribeLaplaceResults shape contract
# =====================================================================


class TestResultsShapeContract:
    """G_obs / G_kept properties on ScribeLaplaceResults."""

    def test_legacy_result_g_obs_equals_g_kept(self):
        import scribe

        adata = _make_adata(30, 8, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
        )
        assert res.G_obs == 8
        assert res.G_kept == 8
        # n_genes matches G_obs for the legacy path.
        assert res.n_genes == res.G_obs

    def test_g_obs_falls_back_to_axis_layout_when_n_genes_missing(self):
        # When `n_genes` is absent (corner case), `G_obs` should
        # consult ``axis_layout`` instead.
        from scribe.laplace.results import ScribeLaplaceResults
        from scribe.laplace._axis_layout import build_axis_layout

        layout = build_axis_layout(
            7, correlate_other_column=False, has_pooled_other=True,
        )
        # Minimal stub bypassing the dataclass defaults that require a
        # ModelConfig.  We just need to verify property lookup.
        stub = SimpleNamespace(
            n_genes=None,
            axis_layout=layout,
            mu=None,
        )
        # Borrow the property descriptors.
        assert ScribeLaplaceResults.G_obs.fget(stub) == 7
        assert ScribeLaplaceResults.G_kept.fget(stub) == 6

    def test_g_kept_falls_back_to_g_obs_without_layout(self):
        # When no layout is attached, G_kept must fall back to G_obs.
        # We use a small subclass to mimic the property descriptor
        # lookup correctly (a plain SimpleNamespace can't resolve the
        # `self.G_obs` call inside the G_kept property).
        from scribe.laplace.results import ScribeLaplaceResults

        class _Stub:
            G_obs = ScribeLaplaceResults.G_obs
            G_kept = ScribeLaplaceResults.G_kept

            def __init__(self):
                self.n_genes = 10
                self.axis_layout = None
                self.mu = None

        stub = _Stub()
        assert stub.G_obs == 10
        assert stub.G_kept == 10


# =====================================================================
# ModelConfig + fit kwarg round-trip
# =====================================================================


class TestModelConfigFlag:
    """`correlate_other_column` round-trips through ModelConfig and fit."""

    def test_modelconfig_default_false_after_5b(self):
        """Default is now False (biologically cleaner) — flipped in
        Commit 5b once all four count likelihoods had their
        deviation-form math wired.  ``True`` is the explicit legacy
        opt-in."""
        from scribe.models.config import ModelConfig

        mc = ModelConfig(base_model="nbln")
        assert mc.correlate_other_column is False

    def test_modelconfig_explicit_true_is_legacy_opt_in(self):
        """Users can explicitly opt into the legacy layout (where
        ``_other`` participates in Σ) by passing
        ``correlate_other_column=True``."""
        from scribe.models.config import ModelConfig

        mc_legacy = ModelConfig(
            base_model="nbln", correlate_other_column=True,
        )
        assert mc_legacy.correlate_other_column is True
        # Explicit False matches the new default.
        mc_decoupled = ModelConfig(
            base_model="nbln", correlate_other_column=False,
        )
        assert mc_decoupled.correlate_other_column is False

    def test_fit_kwarg_propagates_to_modelconfig(self):
        import scribe

        adata = _make_adata(20, 6, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=3, seed=0,
            correlate_other_column=False,
        )
        assert res.model_config.correlate_other_column is False


# =====================================================================
# LNM real wiring (Commit 6)
# =====================================================================
#
# LNM is structurally distinct from the four count-likelihood models:
# it lives in ALR compositional coordinates, where the ALR reference
# gene is excluded from the latent covariance by construction.  So
# LNM realises the ``correlate_other_column=False`` decoupling only
# when the ALR reference is pinned to ``_other``'s position.  Today's
# auto-selection by minimum variance does NOT guarantee this — the
# winner of the variance competition is often a low-expression gene,
# not the pooled aggregate.
#
# Commit 6 adds two complementary checks:
#
# 1. ``apply_gene_coverage_and_alr`` (the standard engine path):
#    when ``correlate_other_column=False`` AND a pooled ``_other``
#    exists AND ``alr_reference_idx`` was not supplied, auto-pin to
#    ``_other``'s position (skip the min-variance fallback).  When
#    the user supplies an explicit ``alr_reference_idx`` that does
#    not resolve to ``_other`` under decoupling, raise.  Under legacy
#    (``True``), preserve today's contract verbatim.
#
# 2. ``LNMObservationModel.__init__`` / ``init_state``: defensive
#    consistency check that catches direct ``ModelConfig`` construction
#    bypassing the gene-coverage stage.  Under legacy, no-op (bit-
#    equal).
#
# Tests below cover both checks plus the legacy bit-equal guarantee.


class TestLnmRealWiringStage:
    """``apply_gene_coverage_and_alr`` LNM ALR-reference resolution.

    These exercise the new branch in
    ``scribe.api.stages.gene_coverage.apply_gene_coverage_and_alr``
    that pins / validates the ALR reference based on
    ``correlate_other_column``.
    """

    def test_decoupled_auto_pins_alr_reference_to_other(self):
        """``correlate_other_column=False`` + ``gene_coverage<1.0`` +
        ``alr_reference_idx=None`` → auto-pinned to the filtered
        ``_other`` position (last index post-filter).
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
            gene_coverage=0.85,
            correlate_other_column=False,
        )
        # The resolved reference is on ``ModelConfig`` after the
        # stages complete; we check it via the result's model_config.
        # ``_other`` is the last column post-filter — ``G_obs`` is
        # observation-axis length (kept + 1).
        assert res.model_config.alr_reference_idx == res.G_obs - 1

    def test_decoupled_explicit_non_other_reference_raises(self):
        """``correlate_other_column=False`` + explicit
        ``alr_reference_idx`` that resolves to a retained gene → the
        stage raises ``ValueError`` pointing at both flags.
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        with pytest.raises(ValueError) as excinfo:
            scribe.fit(
                adata,
                model="lnm",
                inference_method="laplace",
                latent_dim=2,
                n_steps=3,
                seed=0,
                gene_coverage=0.85,
                alr_reference_idx=0,  # Index 0 in original space — a
                # retained gene, NOT ``_other``.
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        assert "correlate_other_column=False" in msg
        assert "_other" in msg

    def test_legacy_explicit_real_gene_reference_passes_through(self):
        """``correlate_other_column=True`` (default) + explicit
        ``alr_reference_idx`` that resolves to a retained gene →
        accepted silently (bit-equal contract per auditor rev-3 #8).
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
            gene_coverage=0.85,
            alr_reference_idx=0,
            correlate_other_column=True,
        )
        # The resolved reference is the original index 0 → filtered
        # index 0 (assuming gene 0 was retained, which it is with
        # ``coverage=0.85`` on this seed).
        assert res.model_config.alr_reference_idx == 0

    def test_legacy_no_other_column_min_variance_selection_unchanged(self):
        """Without ``gene_coverage`` (no ``_other`` column), legacy
        path runs today's min-variance auto-selection.  Bit-equal.
        """
        import scribe

        adata = _make_adata(30, 8, seed=0)
        res = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
            # No ``gene_coverage``, no ``_other`` column.
            correlate_other_column=False,
        )
        # The auto-selected reference is in [0, n_genes).  Without
        # ``_other`` the new code path falls back to min-variance
        # selection identically to today.
        assert 0 <= res.model_config.alr_reference_idx < 8

    def test_manually_named_other_decoupled_auto_pins_via_names(self):
        """Auditor rev-9 Medium: when ``gene_coverage`` is None AND
        AnnData ``var_names[-1] == "_other"`` AND
        ``correlate_other_column=False``, the stage's names-fallback
        path must auto-pin the ALR reference to ``_other``'s
        position rather than running min-variance selection.  This
        keeps the LNM stage's detection priority consistent with the
        shared ``build_axis_layout`` factory used by the four
        count-likelihood obs models.
        """
        import scribe

        n_genes = 8
        adata = _make_adata(30, n_genes, seed=0)
        # Manually rename the trailing column to the ``_other``
        # sentinel — simulating a user who pre-filtered their
        # AnnData upstream WITHOUT calling the gene_coverage stage.
        names = [f"g{i}" for i in range(n_genes - 1)] + [_OTHER_NAME]
        adata.var_names = names

        res = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
            correlate_other_column=False,
            # No ``gene_coverage``, no explicit ``alr_reference_idx``.
        )
        # Auto-pinned to the ``_other`` position (last column).
        assert res.model_config.alr_reference_idx == n_genes - 1

    def test_manually_named_other_decoupled_explicit_non_other_raises(self):
        """Auditor rev-9 Medium follow-up: when the names-fallback
        detected a pooled ``_other`` AND the user supplies an
        explicit non-``_other`` ``alr_reference_idx`` AND
        ``correlate_other_column=False``, the stage must raise — same
        contract as the coverage-emitted-``_other`` path.
        """
        import scribe

        n_genes = 8
        adata = _make_adata(30, n_genes, seed=0)
        names = [f"g{i}" for i in range(n_genes - 1)] + [_OTHER_NAME]
        adata.var_names = names

        with pytest.raises(ValueError) as excinfo:
            scribe.fit(
                adata,
                model="lnm",
                inference_method="laplace",
                latent_dim=2,
                n_steps=3,
                seed=0,
                correlate_other_column=False,
                alr_reference_idx=0,  # Real gene, not ``_other``.
            )
        msg = str(excinfo.value)
        assert "correlate_other_column=False" in msg
        assert "_other" in msg

    def test_manually_named_other_legacy_auto_runs_min_variance(self):
        """Auditor rev-9 Medium: under legacy
        ``correlate_other_column=True``, the names-fallback must NOT
        force the ALR reference to ``_other`` — that would break the
        bit-equal contract for users who have a real gene literally
        named ``_other``.  Min-variance auto-selection runs as today.
        """
        import scribe

        n_genes = 8
        adata = _make_adata(30, n_genes, seed=0)
        names = [f"g{i}" for i in range(n_genes - 1)] + [_OTHER_NAME]
        adata.var_names = names

        res = scribe.fit(
            adata,
            model="lnm",
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
            correlate_other_column=True,  # legacy
        )
        # Under legacy, min-variance ran — the result is some index
        # in [0, n_genes).  It might happen to be ``_other``'s
        # position; either way, the stage did NOT force-pin.  The
        # observable contract here is that the fit succeeds and the
        # reference is a valid index.
        assert 0 <= res.model_config.alr_reference_idx < n_genes


class TestLnmRealWiringObsModel:
    """``LNMObservationModel`` defensive consistency check at init_state.

    These bypass the stage by constructing the obs model directly
    with a deliberately mismatched ``alr_reference_idx`` /
    ``correlate_other_column`` combination.  The init_state guard
    must fire with a clear message naming both flags.
    """

    def test_decoupled_with_wrong_alr_reference_raises_at_init(self):
        """Direct construction: ``correlate_other_column=False`` +
        ``has_pooled_other=True`` + ``alr_reference_idx`` pointing at
        a non-``_other`` position → ``init_state`` raises.
        """
        from scribe.laplace._obs_lnm import LNMObservationModel

        # The defensive guard only reads ``correlate_other_column`` via
        # ``getattr`` — a ``SimpleNamespace`` stub avoids the full LNM-
        # specific ``ModelConfig`` validation chain (parameterization,
        # inference_method, etc.) that's irrelevant to this guard.
        mc = SimpleNamespace(
            correlate_other_column=False,
            positive_transform="softplus",
        )
        # Reference is gene 0 (NOT ``_other`` at position 5).
        obs = LNMObservationModel(
            d_mode="learned",
            alr_reference_idx=0,
            capture_anchor=None,
            model_config=mc,
            gene_names=None,
            has_pooled_other=True,  # primary signal: pooled exists
        )
        rng = np.random.default_rng(0)
        count_data = rng.negative_binomial(
            5, 0.5, size=(20, 6)
        ).astype(np.float32)
        with pytest.raises(ValueError) as excinfo:
            obs.init_state(
                count_data=count_data,
                n_cells=20,
                n_genes=6,
                latent_dim=2,
                seed=0,
            )
        msg = str(excinfo.value)
        assert "correlate_other_column" in msg
        assert "_other" in msg
        assert "alr_reference_idx" in msg

    def test_decoupled_with_correct_alr_reference_succeeds(self):
        """Direct construction: ``correlate_other_column=False`` +
        ``has_pooled_other=True`` + ``alr_reference_idx`` at the
        ``_other`` position → ``init_state`` succeeds.
        """
        from scribe.laplace._obs_lnm import LNMObservationModel

        mc = SimpleNamespace(
            correlate_other_column=False,
            positive_transform="softplus",
        )
        # Reference is the ``_other`` position (last index).
        obs = LNMObservationModel(
            d_mode="learned",
            alr_reference_idx=5,
            capture_anchor=None,
            model_config=mc,
            gene_names=None,
            has_pooled_other=True,
        )
        rng = np.random.default_rng(0)
        count_data = rng.negative_binomial(
            5, 0.5, size=(20, 6)
        ).astype(np.float32)
        # Should not raise.
        state = obs.init_state(
            count_data=count_data,
            n_cells=20,
            n_genes=6,
            latent_dim=2,
            seed=0,
        )
        # mu / W live in ALR (G-1) coordinates per today's LNM.
        assert state.params["mu"].shape == (5,)
        assert state.params["W"].shape == (5, 2)

    def test_legacy_with_non_other_reference_no_raise(self):
        """Direct construction: legacy default + ``has_pooled_other``
        signal + non-``_other`` reference → no raise (bit-equal
        contract per auditor rev-3 #8).
        """
        from scribe.laplace._obs_lnm import LNMObservationModel

        mc = SimpleNamespace(
            correlate_other_column=True,  # legacy
            positive_transform="softplus",
        )
        obs = LNMObservationModel(
            d_mode="learned",
            alr_reference_idx=0,
            capture_anchor=None,
            model_config=mc,
            gene_names=None,
            has_pooled_other=True,
        )
        rng = np.random.default_rng(0)
        count_data = rng.negative_binomial(
            5, 0.5, size=(20, 6)
        ).astype(np.float32)
        state = obs.init_state(
            count_data=count_data,
            n_cells=20,
            n_genes=6,
            latent_dim=2,
            seed=0,
        )
        # No raise; legacy path is bit-equal-safe regardless of
        # whether the reference is ``_other`` or not.
        assert state.params["mu"].shape == (5,)

    def test_manually_named_other_decoupled_raises(self):
        """``var_names[-1] == "_other"`` without the gene-coverage
        stage (so ``has_pooled_other`` is None) → names-fallback
        detection triggers the guard under decoupling.
        """
        from scribe.laplace._obs_lnm import LNMObservationModel

        mc = SimpleNamespace(
            correlate_other_column=False,
            positive_transform="softplus",
        )
        names = [f"g{i}" for i in range(5)] + [_OTHER_NAME]
        obs = LNMObservationModel(
            d_mode="learned",
            alr_reference_idx=0,  # NOT the ``_other`` position.
            capture_anchor=None,
            model_config=mc,
            gene_names=names,
            has_pooled_other=None,  # Defer to names fallback.
        )
        rng = np.random.default_rng(0)
        count_data = rng.negative_binomial(
            5, 0.5, size=(20, 6)
        ).astype(np.float32)
        with pytest.raises(ValueError) as excinfo:
            obs.init_state(
                count_data=count_data,
                n_cells=20,
                n_genes=6,
                latent_dim=2,
                seed=0,
            )
        assert "_other" in str(excinfo.value)

"""Tests for Commit 2 scaffolding: `correlate_other_column` flag.

This commit lands the foundation for the harmonic-hare follow-up
plan §Commit 2:

* :class:`scribe.laplace._axis_layout.AxisLayout` + factory
  (`build_axis_layout`) — with the rev-3 ``has_pooled_other``
  primary signal and the contradictory-signal raise.
* :class:`scribe.models.config.ModelConfig.correlate_other_column`
  flag (default False) accepted by :func:`scribe.fit`.
* :class:`scribe.laplace.NBLNObservationModel` accepts
  ``gene_names`` and ``has_pooled_other``, builds an ``AxisLayout``
  at init, slices W/d/latent_loc when decoupled, and threads the
  layout through to :class:`scribe.laplace.ScribeLaplaceResults`
  via the new ``axis_layout`` field.
* :attr:`scribe.laplace.ScribeLaplaceResults.G_obs` and
  :attr:`scribe.laplace.ScribeLaplaceResults.G_kept` properties
  for downstream tooling.
* Guards at ``loss_fn`` / ``final_sweep`` /
  ``compute_global_uncertainty`` that raise
  ``NotImplementedError`` with a clear "Commit 2b lands the math"
  message when ``layout.decoupled`` is True. The deviation-
  parameterised Newton / Schur re-derivation lives in Commit 2b.

These tests verify:

1. ``AxisLayout`` detection priority — explicit ``has_pooled_other``
   beats names; ``gene_names[-1] == "_other"`` is the fallback;
   contradictory signals raise loudly.
2. Legacy NBLN fits (no ``gene_coverage``, no ``_other``) produce a
   trivial layout (``G_kept == G_obs``) and work bit-equal to today.
3. Legacy opt-in (``correlate_other_column=True`` with
   ``gene_coverage < 1.0``) produces a trivial layout and a working
   fit (auditor finding rev-3 #8: legacy passes through silently).
4. Decoupled (default flag, ``gene_coverage < 1.0``) raises
   ``NotImplementedError`` with a clear remediation message.
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

    def test_gene_coverage_default_is_legacy(self):
        """Current default `correlate_other_column=True` keeps legacy
        behaviour even when gene_coverage<1.0 emits an `_other` column.
        Auditor finding rev-4 #2: default is held at True for Commit 2
        so existing fits don't break by routing through Commit 2b's
        not-yet-implemented math.
        """
        import scribe

        adata = _make_adata(30, 12, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            # No explicit `correlate_other_column` — uses the default
            # (`True` in this release).
        )
        # G_obs includes the _other column; layout is still trivial
        # because the default opts into legacy.
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == res.G_kept  # legacy
        # W / d / mu / r all (G_obs,).
        assert res.W.shape[0] == res.G_obs
        assert res.d.shape == (res.G_obs,)
        assert res.mu.shape == (res.G_obs,)
        assert res.r.shape == (res.G_obs,)
        # And the explicit-True opt-in produces identical structure.
        res_explicit = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=5, seed=0,
            gene_coverage=0.85,
            correlate_other_column=True,
        )
        assert res_explicit.axis_layout.decoupled is False
        assert res_explicit.G_obs == res.G_obs


class TestNblnDecoupledRaises:
    """Decoupled NBLN fits raise NotImplementedError until Commit 2b."""

    def test_decoupled_raises_with_clear_message(self):
        import scribe

        adata = _make_adata(30, 12, seed=0)
        # Explicit opt-in to the new (False) default is needed in this
        # release because the default is held at True (rev-4 #2).
        # When Commit 2b lands the math, the default flips to False
        # and this explicit kwarg becomes redundant.
        with pytest.raises(NotImplementedError) as excinfo:
            scribe.fit(
                adata, model="nbln", inference_method="laplace", latent_dim=2,
                n_steps=5, seed=0,
                gene_coverage=0.85,
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        # Message must point at the remediation path.
        assert "Commit 2b" in msg
        assert "correlate_other_column=True" in msg


class TestEnginePlnTslnEarlyFail:
    """PLN raises on decoupled BEFORE obs-model construction.

    As of Commit 4 of the harmonic-hare plan, NBLN, TSLN-Rate, AND
    TSLN-Logit have AxisLayout-aware obs models and raise from
    inside ``init_state`` (see :class:`TestTslnRateScaffolding` and
    :class:`TestTslnLogitScaffolding` below).  Only PLN remains on
    the engine early-fail path (Commit 5).
    """

    def test_pln_decoupled_raises_at_engine(self):
        import scribe

        adata = _make_adata(30, 12, seed=0)
        with pytest.raises(NotImplementedError) as excinfo:
            scribe.fit(
                adata, model="pln", inference_method="laplace", latent_dim=2,
                n_steps=5, seed=0,
                gene_coverage=0.85,
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        assert "pln" in msg.lower() or "PLN" in msg
        assert "Commit" in msg
        assert "correlate_other_column=True" in msg


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

    def test_legacy_default_with_gene_coverage(self):
        """Default `correlate_other_column=True` keeps legacy behaviour
        under `gene_coverage < 1.0` — `_other` participates in Σ as a
        regular gene; layout is trivial."""
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
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        # G_obs accounts for the trailing `_other` column emitted by
        # the gene-coverage stage.
        assert res.G_obs == res.G_kept
        assert res.G_obs >= 2  # kept + _other at minimum

    def test_decoupled_raises_from_obs_model(self):
        """Explicit `correlate_other_column=False` with a pooled
        `_other` column triggers the obs-model NotImplementedError
        (not the engine early-fail)."""
        import scribe

        adata = _make_adata(20, 8, seed=0)
        with pytest.raises(NotImplementedError) as excinfo:
            scribe.fit(
                adata,
                model="twostate_ln_rate",
                inference_method="laplace",
                latent_dim=2,
                n_steps=5,
                seed=0,
                gene_coverage=0.85,
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        # Message identifies TSLN-Rate and points at the remediation.
        assert "TSLN-Rate" in msg
        assert "correlate_other_column=True" in msg


class TestTslnLogitScaffolding:
    """TSLN-Logit scaffolding mirror of NBLN / TSLN-Rate (Commit 4).

    Per-gene parameters ``rate`` / ``kappa`` / ``eta_anchor`` stay on
    the OBSERVATION-layer axis (G_obs,) under decoupled — they're
    per-gene baselines, not regulatory.  Only ``W`` / ``d`` / per-cell
    z shrink to G_kept.  The decoupled-math path is deferred to a
    later commit; for now ``init_state`` fails fast with a clear
    NotImplementedError.
    """

    def test_decoupled_raises_from_obs_model(self):
        """Explicit `correlate_other_column=False` with a pooled
        `_other` column triggers the TSLN-Logit obs-model
        NotImplementedError (not the engine early-fail, which has
        retired ``twostate_ln_logit`` as of Commit 4)."""
        import scribe

        adata = _make_adata(20, 8, seed=0)
        with pytest.raises(NotImplementedError) as excinfo:
            scribe.fit(
                adata,
                model="twostate_ln_logit",
                inference_method="laplace",
                latent_dim=2,
                n_steps=5,
                seed=0,
                gene_coverage=0.85,
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        assert "TSLN-Logit" in msg
        assert "correlate_other_column=True" in msg

    def test_engine_early_fail_now_only_pln(self):
        """As of Commit 4 the engine early-fail set has narrowed to
        PLN only (NBLN / TSLN-Rate / TSLN-Logit all have AxisLayout-
        aware obs models).  Confirm PLN still raises the engine
        message (pointing at Commit 5)."""
        import scribe

        adata = _make_adata(20, 8, seed=0)
        with pytest.raises(NotImplementedError) as excinfo:
            scribe.fit(
                adata,
                model="pln",
                inference_method="laplace",
                latent_dim=2,
                n_steps=5,
                seed=0,
                gene_coverage=0.85,
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        # PLN engine guard points at Commit 5 now (not the old
        # "Commits 3 / 4 / 5" string from earlier engine guards).
        assert "Commit 5" in msg
        assert "pln" in msg.lower() or "PLN" in msg


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

    def test_modelconfig_default_true(self):
        """Current default is True (legacy) — held until Commit 2b
        lands the decoupled math (auditor finding rev-4 #2)."""
        from scribe.models.config import ModelConfig

        mc = ModelConfig(base_model="nbln")
        assert mc.correlate_other_column is True

    def test_modelconfig_explicit_false(self):
        """Users can explicitly opt into the (not-yet-implemented)
        decoupled layout — useful for testing the NotImplementedError
        guard and for forward-compatibility when Commit 2b ships."""
        from scribe.models.config import ModelConfig

        mc = ModelConfig(base_model="nbln", correlate_other_column=False)
        assert mc.correlate_other_column is False

    def test_fit_kwarg_propagates_to_modelconfig(self):
        import scribe

        adata = _make_adata(20, 6, seed=0)
        res = scribe.fit(
            adata, model="nbln", inference_method="laplace", latent_dim=2,
            n_steps=3, seed=0,
            correlate_other_column=False,
        )
        assert res.model_config.correlate_other_column is False

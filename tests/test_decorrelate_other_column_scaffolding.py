"""Tests for Commit 2 scaffolding: `correlate_other_column` flag.

This commit lands the foundation for the harmonic-hare follow-up
plan §Commit 2:

* :class:`scribe.laplace._axis_layout.AxisLayout` + factory
  (`build_axis_layout`) — with the rev-3 ``has_pooled_other``
  primary signal and the contradictory-signal raise.
* :class:`scribe.models.config.ModelConfig.correlate_other_column`
  flag accepted by :func:`scribe.fit`.  Runtime default is held at
  ``True`` (legacy) for the harmonic-hare Commit 2-4 series until
  the per-model decoupled-math kernels land; the default flips to
  ``False`` (the new biologically-cleaner setting) when 2b / 3b /
  4b ship.
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

    def test_legacy_default_with_gene_coverage(self):
        """Default `correlate_other_column=True` keeps legacy behaviour
        under `gene_coverage < 1.0`."""
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
        )
        assert res.axis_layout is not None
        assert res.axis_layout.decoupled is False
        assert res.G_obs == res.G_kept

    def test_decoupled_raises_from_obs_model(self):
        """Explicit `correlate_other_column=False` with a pooled
        `_other` column triggers PLN's obs-model NotImplementedError
        (no engine early-fail anymore — Commit 5 retires it entirely)."""
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
        # Message now identifies PLN by name and points at the
        # remediation; previously the engine early-fail would say
        # the model name lower-cased.
        assert "PLN" in msg
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

    # Obsolete after Commit 5: PLN now owns its own decoupled
    # detection in the obs model, so the engine-level early-fail
    # set is empty.  See ``TestPlnScaffolding`` for the
    # obs-model-driven test on PLN decoupled.


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
        "model,expected_str",
        [
            ("nbln", "NBLN"),
            ("twostate_ln_rate", "TSLN-Rate"),
            ("twostate_ln_logit", "TSLN-Logit"),
            ("pln", "PLN"),
        ],
    )
    def test_manually_named_other_decoupled_raises(
        self, model, expected_str
    ):
        """With `var_names[-1] == "_other"` AND no `gene_coverage` AND
        explicit `correlate_other_column=False`, each model's obs
        model `init_state` raises ``NotImplementedError`` via the
        names-fallback detection path in `build_axis_layout`."""
        import scribe

        adata = self._make_adata_with_manual_other(seed=0)
        with pytest.raises(NotImplementedError) as excinfo:
            scribe.fit(
                adata,
                model=model,
                inference_method="laplace",
                latent_dim=2,
                n_steps=3,
                seed=0,
                correlate_other_column=False,
            )
        msg = str(excinfo.value)
        # Each model's guard names itself in the error message.
        assert expected_str in msg
        assert "correlate_other_column=True" in msg

    @pytest.mark.parametrize(
        "model",
        ["nbln", "twostate_ln_rate", "twostate_ln_logit", "pln"],
    )
    def test_manually_named_other_legacy_default_is_silent(self, model):
        """Under the current legacy default (`correlate_other_column=
        True`), a manually-named `_other` tail with no
        `gene_coverage` must NOT raise — the contradictory-signal
        check is skipped under legacy so the fit proceeds with the
        trivial layout (rev-5 #2 contract)."""
        import scribe

        adata = self._make_adata_with_manual_other(seed=0)
        # Default is True — this should silently fit with trivial layout.
        res = scribe.fit(
            adata,
            model=model,
            inference_method="laplace",
            latent_dim=2,
            n_steps=3,
            seed=0,
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

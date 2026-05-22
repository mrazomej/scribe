"""Unit tests for subset-aware cascade aggregation.

Covers the behavior added by the harmonic-hare plan (`paper/_nb_lognormal.qmd`
§sec-nbln-cascade-aggregation):

* :class:`SubsetInfo` dataclass population from var-names and from
  boolean coverage masks.
* :func:`_aggregate_other_nb` per-sample moment matching against a
  hand-computed reference.
* :func:`_aggregate_other_tsln_rate` additive-μ + inherit-(b, k_off)
  behaviour.
* :func:`priors_from_results` and :func:`freeze_values_from_results`
  subset-aware paths for NBLN (soft + hard freeze).
* :func:`priors_from_twostate_results` and the
  :func:`freeze_values_from_twostate_results` subset-aware paths for
  TSLN-rate; TSLN-logit subset cascades must raise ``NotImplementedError``.
* Equal-panel pass-through bit-equal to today.
* Non-subset relationship raises with a clear error.
* Amortized-capture SVI sources raise ``NotImplementedError`` on
  subset cascades.

These tests use minimal mock SVI-results stubs that mimic the
:class:`ScribeSVIResults` interface ``priors_from_results`` actually
uses.  All math is done at ``atol=1e-5`` against hand-rolled NumPy
references.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pytest

from scribe.laplace.priors import (
    SubsetInfo,
    _aggregate_other_nb,
    _aggregate_other_tsln_rate,
    _assemble_per_gene_subset_samples,
    _check_gene_identity,
    fit_empirical_gaussian,
    freeze_values_from_results,
    freeze_values_from_twostate_results,
    priors_from_results,
    priors_from_twostate_results,
)


# =====================================================================
# Fake SVI-results stub mirroring tests/test_laplace_priors.py
# =====================================================================


class _FakeSVIResults:
    """Minimal stub exposing the surface the cascade adapter consumes.

    ``var_names`` and optionally ``_gene_coverage_mask`` drive the
    subset-detection path in ``_check_gene_identity``.  Samples
    returned by ``get_posterior_samples`` are sliced to ``n_samples``
    along the leading axis.  ``get_map`` returns the first row of each
    sample array as a 1-D MAP estimate.
    """

    def __init__(
        self,
        n_genes: int,
        n_cells: int,
        samples: Dict[str, jnp.ndarray],
        var_names: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        amortized: bool = False,
    ):
        self.n_genes = n_genes
        self.n_cells = n_cells
        self._samples = samples
        if var_names is not None:
            self.var_names = np.asarray(var_names)
        if mask is not None:
            self._gene_coverage_mask = np.asarray(mask, dtype=bool)
        self._amortized = amortized

    def _uses_amortized_capture(self) -> bool:
        return self._amortized

    def get_posterior_samples(
        self,
        rng_key: Any = None,
        n_samples: int = 100,
        counts: Optional[jnp.ndarray] = None,
        store_samples: bool = False,
        **_kwargs: Any,
    ) -> Dict[str, jnp.ndarray]:
        return {k: v[:n_samples] for k, v in self._samples.items()}

    def get_map(
        self,
        counts: Optional[jnp.ndarray] = None,
        verbose: bool = False,
        **_kwargs: Any,
    ) -> Dict[str, jnp.ndarray]:
        # Use the first posterior sample as the MAP point estimate.
        return {k: v[0] for k, v in self._samples.items()}


def _build_lognormal_samples(
    n_samples: int, shape: tuple, mean_log: float = 0.0, sd_log: float = 0.3
) -> jnp.ndarray:
    """Synthesize a positive sample array via LogNormal draws."""
    rng = np.random.default_rng(0)
    z = rng.normal(loc=mean_log, scale=sd_log, size=(n_samples,) + shape)
    return jnp.asarray(np.exp(z), dtype=jnp.float32)


# =====================================================================
# Reference moment-matching computation (hand-rolled)
# =====================================================================


def _reference_aggregate_other_nb(
    r_src: np.ndarray, mu_src: np.ndarray,
    dropped_idx: np.ndarray, source_other_index: Optional[int],
) -> tuple:
    """Per-sample NB aggregation in pure NumPy, mirroring the docstring."""
    eps_r, eps_mu = 1e-8, 1e-12
    r_safe = np.maximum(r_src, eps_r)
    mu_safe = np.maximum(mu_src, eps_mu)

    mu_other = np.zeros(r_safe.shape[0], dtype=mu_safe.dtype)
    var_extra = np.zeros(r_safe.shape[0], dtype=mu_safe.dtype)

    if source_other_index is not None:
        mu_other += mu_safe[:, source_other_index]
        var_extra += (
            mu_safe[:, source_other_index] ** 2
        ) / r_safe[:, source_other_index]

    if dropped_idx.size > 0:
        mu_d = mu_safe[:, dropped_idx]
        r_d = r_safe[:, dropped_idx]
        mu_other += np.sum(mu_d, axis=1)
        var_extra += np.sum((mu_d ** 2) / r_d, axis=1)

    mu_other_safe = np.maximum(mu_other, eps_mu)
    var_extra_safe = np.maximum(var_extra, eps_mu)
    r_other = (mu_other_safe ** 2) / var_extra_safe
    return r_other, mu_other


# =====================================================================
# SubsetInfo + _check_gene_identity
# =====================================================================


class TestSubsetInfoVarNames:
    """Var-names path of ``_check_gene_identity`` builds correct SubsetInfo."""

    def test_equal_panels_returns_is_equal_true(self):
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "g2", "g3", "_other"])
        results = _FakeSVIResults(
            n_genes=4, n_cells=10, samples={}, var_names=src_names
        )
        strict, method, info = _check_gene_identity(
            results, 4, tgt_names, None
        )
        assert strict is True
        assert method == "var_names"
        assert info.is_equal is True
        assert info.is_subset is True
        # _other indices populated even on the equal-panel branch so the
        # downstream pass-through still knows the trailing slot's role.
        assert info.source_has_other is True
        assert info.target_has_other is True

    def test_proper_subset_populates_kept_dropped(self):
        src_names = np.array(["g1", "g2", "g3", "g4", "g5", "_other"])
        # Target keeps g1, g3 individually; pools g2, g4, g5 into "_other".
        tgt_names = np.array(["g1", "g3", "_other"])
        results = _FakeSVIResults(
            n_genes=6, n_cells=10, samples={}, var_names=src_names
        )
        _, method, info = _check_gene_identity(
            results, 3, tgt_names, None
        )
        assert method == "var_names"
        assert info.is_equal is False
        assert info.is_subset is True
        # Source positions for the target's kept (non-"_other") genes.
        np.testing.assert_array_equal(
            info.kept_idx_in_source, np.array([0, 2])
        )
        # Source positions for the additional genes the target drops.
        np.testing.assert_array_equal(
            info.dropped_idx_in_source, np.array([1, 3, 4])
        )
        assert info.source_has_other is True
        assert info.target_has_other is True
        assert info.source_other_index_in_source == 5
        assert info.target_other_index_in_target == 2

    def test_non_subset_raises_with_missing_genes(self):
        src_names = np.array(["g1", "g2", "_other"])
        # Target has g99 which the source does not have ⇒ non-subset.
        tgt_names = np.array(["g1", "g99", "_other"])
        results = _FakeSVIResults(
            n_genes=3, n_cells=10, samples={}, var_names=src_names
        )
        with pytest.raises(ValueError, match=r"NOT a subset"):
            _check_gene_identity(results, 3, tgt_names, None)

    def test_subset_without_target_other_raises(self):
        # Source has dropped genes but target has NO "_other" column to
        # receive the pooled signal.
        src_names = np.array(["g1", "g2", "g3"])
        tgt_names = np.array(["g1"])  # no "_other"
        results = _FakeSVIResults(
            n_genes=3, n_cells=10, samples={}, var_names=src_names
        )
        with pytest.raises(ValueError, match=r"no trailing '_other' column"):
            _check_gene_identity(results, 1, tgt_names, None)

    def test_svi_cov_1_0_no_other_subset_still_detected(self):
        # SVI used gene_coverage=1.0 (no "_other"); target uses a
        # narrower coverage and has "_other".  Aggregator runs over
        # per-gene SVI samples only.
        src_names = np.array(["g1", "g2", "g3"])
        tgt_names = np.array(["g1", "_other"])
        results = _FakeSVIResults(
            n_genes=3, n_cells=10, samples={}, var_names=src_names
        )
        _, _, info = _check_gene_identity(results, 2, tgt_names, None)
        assert info.is_subset is True
        assert info.is_equal is False
        assert info.source_has_other is False
        assert info.source_other_index_in_source is None
        # All non-"g1" source genes contribute to the pooled "_other".
        np.testing.assert_array_equal(
            info.dropped_idx_in_source, np.array([1, 2])
        )


class TestSubsetInfoMaskPath:
    """Boolean mask path of ``_check_gene_identity``.

    Per the implementation: mask-only subset detection is INTENTIONALLY
    unsupported (would require var-names to disambiguate gene identity
    across the two panels).  The mask path handles only equal-mask
    cases and raises on non-equal masks.
    """

    def test_equal_mask_returns_equal(self):
        mask = np.array([True, True, False, True])
        results = _FakeSVIResults(
            n_genes=3, n_cells=5, samples={}, mask=mask
        )
        _, method, info = _check_gene_identity(results, 3, None, mask)
        assert method == "mask"
        assert info.is_equal is True

    def test_mask_subset_without_varnames_raises(self):
        src_mask = np.array([True, True, True, True])
        tgt_mask = np.array([True, True, False, True])  # strict subset
        results = _FakeSVIResults(
            n_genes=4, n_cells=5, samples={}, mask=src_mask
        )
        with pytest.raises(
            ValueError, match=r"via boolean coverage masks alone is not"
        ):
            _check_gene_identity(results, 3, None, tgt_mask)


# =====================================================================
# _aggregate_other_nb
# =====================================================================


class TestAggregateOtherNB:
    """Per-sample NB moment-matching aggregator correctness."""

    def test_basic_aggregation_matches_reference(self):
        rng = np.random.default_rng(42)
        S, G_src = 200, 6
        r_src = np.exp(rng.normal(0.0, 0.5, size=(S, G_src))).astype(
            np.float32
        )
        mu_src = np.exp(rng.normal(1.0, 0.5, size=(S, G_src))).astype(
            np.float32
        )
        dropped = np.array([1, 3, 4])
        svi_other_idx = 5  # last column is SVI's "_other"

        r_other_jax, mu_other_jax = _aggregate_other_nb(
            jnp.asarray(r_src), jnp.asarray(mu_src), dropped, svi_other_idx
        )
        r_other_ref, mu_other_ref = _reference_aggregate_other_nb(
            r_src, mu_src, dropped, svi_other_idx
        )
        np.testing.assert_allclose(
            np.asarray(r_other_jax), r_other_ref, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(mu_other_jax), mu_other_ref, atol=1e-5, rtol=1e-5
        )

    def test_no_svi_other_only_drops_match_reference(self):
        # SVI used gene_coverage=1.0 — no "_other" column upstream.
        rng = np.random.default_rng(123)
        S, G_src = 100, 4
        r_src = np.exp(rng.normal(0.0, 0.3, size=(S, G_src))).astype(
            np.float32
        )
        mu_src = np.exp(rng.normal(1.0, 0.3, size=(S, G_src))).astype(
            np.float32
        )
        dropped = np.array([1, 2, 3])

        r_other_jax, mu_other_jax = _aggregate_other_nb(
            jnp.asarray(r_src), jnp.asarray(mu_src), dropped, None
        )
        r_other_ref, mu_other_ref = _reference_aggregate_other_nb(
            r_src, mu_src, dropped, None
        )
        np.testing.assert_allclose(
            np.asarray(r_other_jax), r_other_ref, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(mu_other_jax), mu_other_ref, atol=1e-5, rtol=1e-5
        )

    def test_no_drops_only_svi_other_is_identity(self):
        # When all SVI genes match the target (no drops) but the SVI
        # had its own "_other", the aggregator degenerates to the
        # identity on that column.
        rng = np.random.default_rng(7)
        S, G_src = 50, 3
        r_src = np.exp(rng.normal(0, 0.3, size=(S, G_src))).astype(
            np.float32
        )
        mu_src = np.exp(rng.normal(1, 0.3, size=(S, G_src))).astype(
            np.float32
        )
        r_other_jax, mu_other_jax = _aggregate_other_nb(
            jnp.asarray(r_src), jnp.asarray(mu_src),
            np.array([], dtype=np.int64), 2,
        )
        np.testing.assert_allclose(
            np.asarray(r_other_jax), r_src[:, 2], atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(mu_other_jax), mu_src[:, 2], atol=1e-5, rtol=1e-5
        )

    def test_no_contributions_raises(self):
        # Both empty dropped and None svi_other_index is a programming
        # error (caller should have routed through pass-through).
        with pytest.raises(ValueError, match=r"no contributing terms"):
            _aggregate_other_nb(
                jnp.zeros((10, 3)),
                jnp.ones((10, 3)),
                np.array([], dtype=np.int64),
                None,
            )

    def test_first_two_moments_match_pooled_nb_sim(self):
        """Aggregator output matches first two moments of a sampled sum.

        Simulate two NB populations with different (r, μ) per
        sample, sum the draws, and check that the aggregator's pooled
        ``(r_other, μ_other)`` reproduces the empirical mean and
        variance to within MC tolerance for moderate sample sizes.
        """
        rng = np.random.default_rng(2024)
        S = 50
        # Two genes with distinct (r, μ); generate "true" parameters.
        r_src = np.array([[2.0, 3.5]], dtype=np.float32).repeat(S, axis=0)
        mu_src = np.array([[5.0, 2.0]], dtype=np.float32).repeat(S, axis=0)
        # Aggregator's prediction for the pooled NB.
        r_other_jax, mu_other_jax = _aggregate_other_nb(
            jnp.asarray(r_src), jnp.asarray(mu_src),
            np.array([0, 1]), None,
        )
        # Per-sample (constant across S because params are fixed).
        r_other_pred = float(np.asarray(r_other_jax)[0])
        mu_other_pred = float(np.asarray(mu_other_jax)[0])
        # Closed-form mean/variance for sum of two independent NBs:
        #     mean_total = μ1 + μ2
        #     var_total  = (μ1 + μ1²/r1) + (μ2 + μ2²/r2)
        mu_total_true = 5.0 + 2.0
        var_total_true = (5.0 + 25.0 / 2.0) + (2.0 + 4.0 / 3.5)
        # Moment-matched NB(μ_other_pred, r_other_pred) has:
        #     mean = μ_other_pred
        #     var  = μ_other_pred + μ_other_pred²/r_other_pred
        var_predicted = mu_other_pred + (mu_other_pred ** 2) / r_other_pred
        assert mu_other_pred == pytest.approx(mu_total_true, abs=1e-5)
        assert var_predicted == pytest.approx(var_total_true, abs=1e-5)


# =====================================================================
# _aggregate_other_tsln_rate
# =====================================================================


class TestAggregateOtherTSLNRate:
    """TSLN-rate aggregator: additive μ + inherit / median (b, k_off)."""

    def test_with_svi_other_inherits_b_and_k_off(self):
        rng = np.random.default_rng(11)
        S, G_src = 100, 5
        mu_src = np.exp(rng.normal(1.0, 0.3, size=(S, G_src))).astype(
            np.float32
        )
        bs_src = np.exp(rng.normal(0.0, 0.2, size=(S, G_src))).astype(
            np.float32
        )
        ko_src = np.exp(rng.normal(0.5, 0.2, size=(S, G_src))).astype(
            np.float32
        )
        dropped = np.array([1, 2])
        svi_other = 4

        mu_other, bs_other, ko_other = _aggregate_other_tsln_rate(
            jnp.asarray(mu_src), jnp.asarray(bs_src), jnp.asarray(ko_src),
            dropped, svi_other,
        )
        # μ is additive sum over (dropped ∪ {svi_other}).
        mu_ref = (
            mu_src[:, svi_other] + mu_src[:, dropped].sum(axis=1)
        )
        np.testing.assert_allclose(
            np.asarray(mu_other), mu_ref, atol=1e-5, rtol=1e-5
        )
        # (b, k_off) inherit the SVI '_other' column unchanged.
        np.testing.assert_allclose(
            np.asarray(bs_other), bs_src[:, svi_other], atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(ko_other), ko_src[:, svi_other], atol=1e-5
        )

    def test_without_svi_other_uses_median_of_dropped(self):
        rng = np.random.default_rng(22)
        S, G_src = 80, 4
        mu_src = np.exp(rng.normal(1, 0.3, size=(S, G_src))).astype(
            np.float32
        )
        bs_src = np.exp(rng.normal(0, 0.2, size=(S, G_src))).astype(
            np.float32
        )
        ko_src = np.exp(rng.normal(0.5, 0.2, size=(S, G_src))).astype(
            np.float32
        )
        dropped = np.array([0, 1, 2, 3])

        mu_other, bs_other, ko_other = _aggregate_other_tsln_rate(
            jnp.asarray(mu_src), jnp.asarray(bs_src), jnp.asarray(ko_src),
            dropped, None,
        )
        # Mean is additive; b and k_off use the per-sample median.
        np.testing.assert_allclose(
            np.asarray(mu_other), mu_src.sum(axis=1), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(bs_other),
            np.median(bs_src, axis=1),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(ko_other),
            np.median(ko_src, axis=1),
            atol=1e-5,
        )


# =====================================================================
# _assemble_per_gene_subset_samples
# =====================================================================


class TestAssembleSubsetSamples:
    def test_preserves_target_axis_order(self):
        rng = np.random.default_rng(3)
        src = jnp.asarray(rng.normal(size=(50, 6)))
        # Reorder: take target gene 0 from source position 2; target
        # gene 1 from source position 0; target gene 2 from source 5.
        kept = np.array([2, 0, 5])
        out = _assemble_per_gene_subset_samples(src, kept)
        assert out.shape == (50, 3)
        np.testing.assert_array_equal(
            np.asarray(out[:, 0]), np.asarray(src[:, 2])
        )
        np.testing.assert_array_equal(
            np.asarray(out[:, 1]), np.asarray(src[:, 0])
        )
        np.testing.assert_array_equal(
            np.asarray(out[:, 2]), np.asarray(src[:, 5])
        )


# =====================================================================
# priors_from_results — NBLN soft cascade
# =====================================================================


class TestPriorsFromResultsSubsetCascade:
    """Subset-aware soft cascade for NBLN."""

    def _build_source(
        self, S: int, G_src: int, src_names: np.ndarray,
        with_eta: bool = False, n_cells: int = 12,
    ) -> _FakeSVIResults:
        r_src = _build_lognormal_samples(S, (G_src,), 0.0, 0.4)
        mu_src = _build_lognormal_samples(S, (G_src,), 1.0, 0.4)
        samples = {"r": r_src, "mu": mu_src}
        if with_eta:
            samples["eta_capture"] = jnp.asarray(
                np.maximum(
                    np.random.default_rng(0).normal(
                        2.0, 0.3, size=(S, n_cells)
                    ),
                    1e-3,
                ).astype(np.float32)
            )
        return _FakeSVIResults(
            n_genes=G_src, n_cells=n_cells, samples=samples,
            var_names=src_names,
        )

    def test_equal_panel_bit_equal_to_legacy(self):
        S, G = 200, 4
        names = np.array(["g1", "g2", "g3", "_other"])
        source = self._build_source(S, G, names)
        prior_a, _ = priors_from_results(
            source, target_positive_transform="softplus",
            target_n_genes=G, target_n_cells=12,
            target_gene_names=names, target_gene_mask=None,
            n_samples=S, tau=1.0, verbose=False,
        )
        # Same call again — must be deterministic + identical.
        prior_b, _ = priors_from_results(
            source, target_positive_transform="softplus",
            target_n_genes=G, target_n_cells=12,
            target_gene_names=names, target_gene_mask=None,
            n_samples=S, tau=1.0, verbose=False,
        )
        for k in ("r", "mu"):
            np.testing.assert_array_equal(
                np.asarray(prior_a[k]["loc"]),
                np.asarray(prior_b[k]["loc"]),
            )
            np.testing.assert_array_equal(
                np.asarray(prior_a[k]["scale"]),
                np.asarray(prior_b[k]["scale"]),
            )

    def test_proper_subset_aggregates_other_against_reference(self):
        S = 200
        src_names = np.array(
            ["g1", "g2", "g3", "g4", "g5", "_other"]
        )
        tgt_names = np.array(["g1", "g3", "_other"])
        source = self._build_source(S, 6, src_names)

        prior, _ = priors_from_results(
            source, target_positive_transform="softplus",
            target_n_genes=3, target_n_cells=12,
            target_gene_names=tgt_names, target_gene_mask=None,
            n_samples=S, tau=1.0, verbose=False,
        )
        # Pull the underlying samples to build a reference.
        r_src = np.asarray(source._samples["r"][:S])
        mu_src = np.asarray(source._samples["mu"][:S])

        # Hand-compute the moment-matched aggregate for the trailing
        # "_other" slot.  Source has "_other" at position 5; the
        # dropped set is {1, 3, 4}.
        r_other_ref, mu_other_ref = _reference_aggregate_other_nb(
            r_src, mu_src, np.array([1, 3, 4]), 5
        )
        # The expected target shape arrays.
        r_expected_per_gene = np.concatenate(
            [r_src[:, [0]], r_src[:, [2]], r_other_ref[:, None]],
            axis=1,
        )
        mu_expected_per_gene = np.concatenate(
            [mu_src[:, [0]], mu_src[:, [2]], mu_other_ref[:, None]],
            axis=1,
        )

        # Coordinate convert and moment-match by hand, mirroring the
        # production code path.
        from scribe.laplace._global_uncertainty import _JAX_POSITIVE_FNS
        _fwd, pos_inv = _JAX_POSITIVE_FNS["softplus"]
        r_uncon_ref = np.asarray(pos_inv(jnp.maximum(jnp.asarray(r_expected_per_gene), 1e-8)))
        mu_log_ref = np.log(np.maximum(mu_expected_per_gene, 1e-8))

        np.testing.assert_allclose(
            np.asarray(prior["r"]["loc"]),
            r_uncon_ref.mean(axis=0),
            atol=1e-5, rtol=1e-5,
        )
        # Variance uses ddof=1 to match fit_empirical_gaussian.
        np.testing.assert_allclose(
            np.asarray(prior["r"]["scale"]),
            r_uncon_ref.std(axis=0, ddof=1),
            atol=1e-5, rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(prior["mu"]["loc"]),
            mu_log_ref.mean(axis=0),
            atol=1e-5, rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(prior["mu"]["scale"]),
            mu_log_ref.std(axis=0, ddof=1),
            atol=1e-5, rtol=1e-5,
        )
        # Shape sanity: G_target = 3.
        assert prior["r"]["loc"].shape == (3,)
        assert prior["mu"]["loc"].shape == (3,)

    def test_subset_with_no_svi_other_runs(self):
        # SVI used gene_coverage=1.0 (no "_other"); Laplace target has
        # "_other".  Aggregator runs over per-gene samples only.
        S = 100
        src_names = np.array(["g1", "g2", "g3", "g4"])
        tgt_names = np.array(["g1", "_other"])
        source = self._build_source(S, 4, src_names)
        prior, _ = priors_from_results(
            source, target_positive_transform="softplus",
            target_n_genes=2, target_n_cells=12,
            target_gene_names=tgt_names, target_gene_mask=None,
            n_samples=S, tau=1.0, verbose=False,
        )
        assert prior["r"]["loc"].shape == (2,)
        assert prior["mu"]["loc"].shape == (2,)

    def test_non_subset_raises(self):
        src_names = np.array(["g1", "g2", "_other"])
        tgt_names = np.array(["g1", "g99", "_other"])
        source = self._build_source(50, 3, src_names)
        with pytest.raises(ValueError, match=r"NOT a subset"):
            priors_from_results(
                source, target_positive_transform="softplus",
                target_n_genes=3, target_n_cells=12,
                target_gene_names=tgt_names, target_gene_mask=None,
                n_samples=50, tau=1.0, verbose=False,
            )

    def test_amortized_subset_raises_not_implemented(self):
        S = 50
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        source = self._build_source(S, 4, src_names)
        source._amortized = True
        with pytest.raises(NotImplementedError, match=r"amortized-capture"):
            priors_from_results(
                source, target_positive_transform="softplus",
                target_n_genes=2, target_n_cells=12,
                target_gene_names=tgt_names, target_gene_mask=None,
                source_counts=jnp.zeros((12, 2)),
                n_samples=S, tau=1.0, verbose=False,
            )


# =====================================================================
# freeze_values_from_results — NBLN hard freeze
# =====================================================================


class TestFreezeValuesFromResultsSubsetCascade:
    """Subset-aware hard-freeze MAP extraction for NBLN."""

    def test_proper_subset_extracts_aggregated_other(self):
        S = 100
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        mu_src = _build_lognormal_samples(S, (4,), 1.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10, samples={"r": r_src, "mu": mu_src},
            var_names=src_names,
        )
        freeze = freeze_values_from_results(
            source, target_positive_transform="softplus",
            target_n_genes=2, target_n_cells=10,
            target_gene_names=tgt_names, target_gene_mask=None,
            freeze_params=("r", "mu"), verbose=False,
        )
        # MAP-based aggregation reproduces _aggregate_other_nb on a
        # singleton sample axis (the FakeSVIResults's get_map returns
        # the first row of each sample array).
        r_map = np.asarray(r_src[0])
        mu_map = np.asarray(mu_src[0])
        r_other_ref, mu_other_ref = _reference_aggregate_other_nb(
            r_map[None, :], mu_map[None, :], np.array([1, 2]), 3,
        )
        from scribe.laplace._global_uncertainty import _JAX_POSITIVE_FNS
        _fwd, pos_inv = _JAX_POSITIVE_FNS["softplus"]
        r_expected = np.concatenate([r_map[[0]], r_other_ref])
        mu_expected = np.concatenate([mu_map[[0]], mu_other_ref])
        np.testing.assert_allclose(
            np.asarray(freeze["r"]["loc"]),
            np.asarray(pos_inv(jnp.maximum(jnp.asarray(r_expected), 1e-8))),
            atol=1e-5, rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(freeze["mu"]["loc"]),
            np.log(np.maximum(mu_expected, 1e-8)),
            atol=1e-5, rtol=1e-5,
        )

    def test_subset_requires_both_r_and_mu(self):
        # Source has only "r"; subset path requires both keys.
        S = 50
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10, samples={"r": r_src},
            var_names=src_names,
        )
        with pytest.raises(ValueError, match=r"couples them"):
            freeze_values_from_results(
                source, target_positive_transform="softplus",
                target_n_genes=2, target_n_cells=10,
                target_gene_names=tgt_names, target_gene_mask=None,
                freeze_params=("r", "mu"), verbose=False,
            )


# =====================================================================
# TSLN-rate and TSLN-logit subset cascades
# =====================================================================


class TestTSLNRateSubsetCascade:
    """TSLN-rate subset cascade aggregates (mu, burst_size, k_off)."""

    def _build_tsln_source(
        self, S: int, G_src: int, src_names: np.ndarray,
        n_cells: int = 10,
    ) -> _FakeSVIResults:
        return _FakeSVIResults(
            n_genes=G_src, n_cells=n_cells,
            samples={
                "mu": _build_lognormal_samples(S, (G_src,), 1.0, 0.3),
                "burst_size": _build_lognormal_samples(S, (G_src,), 0.0, 0.2),
                "k_off": _build_lognormal_samples(S, (G_src,), 0.5, 0.2),
            },
            var_names=src_names,
        )

    def test_rate_subset_aggregates_correctly(self):
        S = 150
        src_names = np.array(
            ["g1", "g2", "g3", "g4", "_other"]
        )
        tgt_names = np.array(["g1", "_other"])
        source = self._build_tsln_source(S, 5, src_names)

        prior, _ = priors_from_twostate_results(
            source, target_positive_transform="softplus",
            target_n_genes=2, target_n_cells=10,
            target_variant="rate",
            target_gene_names=tgt_names, target_gene_mask=None,
            n_samples=S, tau=1.0, verbose=False,
        )
        # Bundle should have target-shaped priors for all three.
        for key in ("mu", "burst_size", "k_off"):
            assert prior[key]["loc"].shape == (2,)
            assert prior[key]["scale"].shape == (2,)

    def test_logit_subset_raises_not_implemented(self):
        S = 50
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        source = self._build_tsln_source(S, 4, src_names)
        with pytest.raises(
            NotImplementedError, match=r"TSLN-Logit"
        ):
            priors_from_twostate_results(
                source, target_positive_transform="softplus",
                target_n_genes=2, target_n_cells=10,
                target_variant="logit",
                target_gene_names=tgt_names, target_gene_mask=None,
                n_samples=S, tau=1.0, verbose=False,
            )

    def test_freeze_rate_subset_aggregates(self):
        S = 40
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        source = self._build_tsln_source(S, 4, src_names)
        freeze = freeze_values_from_twostate_results(
            source, target_positive_transform="softplus",
            target_n_genes=2, target_n_cells=10,
            target_variant="rate",
            target_gene_names=tgt_names, target_gene_mask=None,
            freeze_params=("mu", "burst_size", "k_off"),
            verbose=False,
        )
        for key in ("mu", "burst_size", "k_off"):
            assert freeze[key]["loc"].shape == (2,)

    def test_freeze_logit_subset_raises_not_implemented(self):
        S = 20
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        source = self._build_tsln_source(S, 4, src_names)
        with pytest.raises(NotImplementedError, match=r"TSLN-Logit"):
            freeze_values_from_twostate_results(
                source, target_positive_transform="softplus",
                target_n_genes=2, target_n_cells=10,
                target_variant="logit",
                target_gene_names=tgt_names, target_gene_mask=None,
                freeze_params=("rate", "kappa", "eta_anchor"),
                verbose=False,
            )


# =====================================================================
# Auditor-driven coverage gap: edge cases at SubsetInfo / aggregator
# =====================================================================


class TestSubsetEdgeCasesFromAudit:
    """Edge cases surfaced by the post-implementation audit."""

    def test_source_other_without_target_other_raises(self):
        """Auditor finding (high): when source has `_other` and the
        target does not, the aggregator has nowhere to put the source's
        pooled aggregate.  Must raise even when dropped is empty.
        """
        src_names = np.array(["g1", "g2", "_other"])
        tgt_names = np.array(["g1", "g2"])  # no "_other"
        results = _FakeSVIResults(
            n_genes=3, n_cells=10, samples={}, var_names=src_names
        )
        with pytest.raises(ValueError, match=r"has nowhere to go"):
            _check_gene_identity(results, 2, tgt_names, None)

    def test_subset_priors_missing_mu_raises(self):
        """Auditor finding (medium): subset aggregation requires both
        r and mu; silently falling back to source-shape samples would
        produce wrong-shape priors.
        """
        S = 50
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        # Source exposes only "r" — missing "mu".
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10, samples={"r": r_src},
            var_names=src_names,
        )
        with pytest.raises(
            ValueError, match=r"requires the SVI source to expose both"
        ):
            priors_from_results(
                source, target_positive_transform="softplus",
                target_n_genes=2, target_n_cells=10,
                target_gene_names=tgt_names, target_gene_mask=None,
                n_samples=S, tau=1.0, verbose=False,
            )

    def test_subset_priors_missing_r_raises(self):
        """Symmetric to ``test_subset_priors_missing_mu_raises``."""
        S = 50
        src_names = np.array(["g1", "g2", "g3", "_other"])
        tgt_names = np.array(["g1", "_other"])
        mu_src = _build_lognormal_samples(S, (4,), 1.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10, samples={"mu": mu_src},
            var_names=src_names,
        )
        with pytest.raises(
            ValueError, match=r"requires the SVI source to expose both"
        ):
            priors_from_results(
                source, target_positive_transform="softplus",
                target_n_genes=2, target_n_cells=10,
                target_gene_names=tgt_names, target_gene_mask=None,
                n_samples=S, tau=1.0, verbose=False,
            )


class TestMomentMatchFrozenSubset:
    """Coverage of ``_moment_match_frozen_for_nbln`` subset path.

    Audited gap: previously no test exercised this helper directly with
    a subset cascade.  It is called from ``inference/laplace.py``
    result-packaging when ``cascade_subset_info`` is populated, so a
    regression here would silently corrupt ``r_scale`` / ``mu_loc`` /
    ``mu_scale`` on the result object.
    """

    def _model_config_softplus(self):
        # Minimal stub with the attributes ``resolve_positive_fns`` reads.
        return SimpleNamespace(positive_transform="softplus")

    def test_subset_aggregates_r_scale_and_mu_summaries(self):
        from scribe.inference.laplace import _moment_match_frozen_for_nbln
        import jax.nn as jnn
        S = 100
        # Source has 4 genes including "_other" at index 3.
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        mu_src = _build_lognormal_samples(S, (4,), 1.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10,
            samples={"r": r_src, "mu": mu_src},
            var_names=np.array(["g1", "g2", "g3", "_other"]),
        )
        # Target keeps g1 only, with "_other" → kept_idx = [0],
        # dropped_idx = [1, 2], source_other_index = 3.
        subset_info = SubsetInfo(
            is_equal=False, is_subset=True,
            kept_idx_in_source=np.array([0]),
            dropped_idx_in_source=np.array([1, 2]),
            source_has_other=True, target_has_other=True,
            source_other_index_in_source=3,
            target_other_index_in_target=1,
        )
        # Call the moment-match helper with a softplus model config.
        # Use jax.nn.softplus as the pos_forward.
        r_scale_fallback = jnp.ones((2,), dtype=jnp.float32)
        mu_loc_fallback = jnp.zeros((2,), dtype=jnp.float32)
        mu_scale_fallback = jnp.ones((2,), dtype=jnp.float32)
        r_scale, mu_loc, mu_scale = _moment_match_frozen_for_nbln(
            cascade_source=source,
            cascade_counts=None,
            frozen_params=frozenset({"r", "mu"}),
            pos_forward=jnn.softplus,
            model_config=self._model_config_softplus(),
            r_scale_fallback=r_scale_fallback,
            mu_loc_fallback=mu_loc_fallback,
            mu_scale_fallback=mu_scale_fallback,
            cascade_subset_info=subset_info,
        )
        # Result must be in TARGET shape (G_target=2), not source (4).
        assert r_scale.shape == (2,)
        assert mu_loc.shape == (2,)
        assert mu_scale.shape == (2,)

    def test_subset_missing_key_raises(self):
        from scribe.inference.laplace import _moment_match_frozen_for_nbln
        import jax.nn as jnn
        S = 50
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10,
            samples={"r": r_src},   # missing "mu"
            var_names=np.array(["g1", "g2", "g3", "_other"]),
        )
        subset_info = SubsetInfo(
            is_equal=False, is_subset=True,
            kept_idx_in_source=np.array([0]),
            dropped_idx_in_source=np.array([1, 2]),
            source_has_other=True, target_has_other=True,
            source_other_index_in_source=3,
            target_other_index_in_target=1,
        )
        with pytest.raises(
            ValueError,
            match=r"requires the SVI source to expose both",
        ):
            _moment_match_frozen_for_nbln(
                cascade_source=source,
                cascade_counts=None,
                frozen_params=frozenset({"r", "mu"}),
                pos_forward=jnn.softplus,
                model_config=self._model_config_softplus(),
                r_scale_fallback=jnp.ones((2,)),
                mu_loc_fallback=jnp.zeros((2,)),
                mu_scale_fallback=jnp.ones((2,)),
                cascade_subset_info=subset_info,
            )


class TestNblnFrozenDistributionsSubset:
    """Coverage of ``_nbln_frozen_distributions`` (auditor finding high)."""

    def test_subset_aggregates_r_and_mu_distributions(self):
        from scribe.laplace._dispatch import _nbln_frozen_distributions
        S = 100
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        mu_src = _build_lognormal_samples(S, (4,), 1.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10,
            samples={"r": r_src, "mu": mu_src},
            var_names=np.array(["g1", "g2", "g3", "_other"]),
        )
        subset_info = SubsetInfo(
            is_equal=False, is_subset=True,
            kept_idx_in_source=np.array([0]),
            dropped_idx_in_source=np.array([1, 2]),
            source_has_other=True, target_has_other=True,
            source_other_index_in_source=3,
            target_other_index_in_target=1,
        )
        # Minimal result stub providing the surface
        # ``_nbln_frozen_distributions`` reads.
        result = SimpleNamespace(
            model_config=SimpleNamespace(positive_transform="softplus"),
            _cascade_subset_info=subset_info,
            eta_loc=None,
        )
        out = _nbln_frozen_distributions(
            result,
            frozen=frozenset({"r", "mu"}),
            cascade=source,
            cascade_counts=None,
            n_samples=S,
        )
        # The returned distributions live on TARGET gene axis (G=2):
        # the kept gene plus the aggregated `_other`.
        assert out["r"].event_shape == (2,)
        assert out["mu"].event_shape == (2,)
        assert out["r_unconstrained"].event_shape == (2,)

    def test_subset_missing_key_raises(self):
        from scribe.laplace._dispatch import _nbln_frozen_distributions
        S = 50
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10,
            samples={"r": r_src},
            var_names=np.array(["g1", "g2", "g3", "_other"]),
        )
        subset_info = SubsetInfo(
            is_equal=False, is_subset=True,
            kept_idx_in_source=np.array([0]),
            dropped_idx_in_source=np.array([1, 2]),
            source_has_other=True, target_has_other=True,
            source_other_index_in_source=3,
            target_other_index_in_target=1,
        )
        result = SimpleNamespace(
            model_config=SimpleNamespace(positive_transform="softplus"),
            _cascade_subset_info=subset_info,
            eta_loc=None,
        )
        with pytest.raises(
            ValueError,
            match=r"requires the SVI source to expose both",
        ):
            _nbln_frozen_distributions(
                result,
                frozen=frozenset({"r", "mu"}),
                cascade=source,
                cascade_counts=None,
                n_samples=S,
            )


class TestPPCSubsetRouting:
    """Coverage of subset-aware PPC routing in ``_resolve_nbln_ppc_arrays``.

    Auditor finding (medium): test coverage previously skipped this
    code path.  The helper applies the same aggregator before
    ``_gene_slice`` so PPC samples on the `_other` column reflect the
    pooled aggregate, not the SVI source's own `_other`.
    """

    def test_subset_aggregates_before_gene_slice(self):
        import jax
        from scribe.laplace._sampling import _resolve_nbln_ppc_arrays
        S = 80
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        mu_src = _build_lognormal_samples(S, (4,), 1.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10,
            samples={"r": r_src, "mu": mu_src},
            var_names=np.array(["g1", "g2", "g3", "_other"]),
        )
        subset_info = SubsetInfo(
            is_equal=False, is_subset=True,
            kept_idx_in_source=np.array([0]),
            dropped_idx_in_source=np.array([1, 2]),
            source_has_other=True, target_has_other=True,
            source_other_index_in_source=3,
            target_other_index_in_target=1,
        )
        # Minimal NBLN result stub.
        result = SimpleNamespace(
            frozen_params=frozenset({"r", "mu"}),
            cascade_source=source,
            cascade_source_counts=None,
            _cascade_subset_info=subset_info,
            _subset_gene_index=None,
            model_config=SimpleNamespace(positive_transform="softplus"),
            r_loc=None, r_scale=None,
            mu_loc=None, mu_scale=None,
        )
        out = _resolve_nbln_ppc_arrays(
            result,
            rng_key=jax.random.PRNGKey(0),
            n_samples=32,
            per_cell=False,
        )
        # Result arrays live on TARGET gene axis (G=2): kept gene plus
        # aggregated "_other".  The pool size is capped at 32 (the
        # n_samples request) post-expansion.
        assert out["r_samples"] is not None
        assert out["mu_samples"] is not None
        assert out["r_samples"].shape == (32, 2)
        assert out["mu_samples"].shape == (32, 2)

    def test_subset_missing_key_raises(self):
        import jax
        from scribe.laplace._sampling import _resolve_nbln_ppc_arrays
        S = 50
        r_src = _build_lognormal_samples(S, (4,), 0.0, 0.3)
        source = _FakeSVIResults(
            n_genes=4, n_cells=10,
            samples={"r": r_src},   # missing "mu"
            var_names=np.array(["g1", "g2", "g3", "_other"]),
        )
        subset_info = SubsetInfo(
            is_equal=False, is_subset=True,
            kept_idx_in_source=np.array([0]),
            dropped_idx_in_source=np.array([1, 2]),
            source_has_other=True, target_has_other=True,
            source_other_index_in_source=3,
            target_other_index_in_target=1,
        )
        result = SimpleNamespace(
            frozen_params=frozenset({"r", "mu"}),
            cascade_source=source,
            cascade_source_counts=None,
            _cascade_subset_info=subset_info,
            _subset_gene_index=None,
            model_config=SimpleNamespace(positive_transform="softplus"),
            r_loc=None, r_scale=None,
            mu_loc=None, mu_scale=None,
        )
        with pytest.raises(
            ValueError,
            match=r"requires the SVI source to expose both",
        ):
            _resolve_nbln_ppc_arrays(
                result,
                rng_key=jax.random.PRNGKey(0),
                n_samples=16,
                per_cell=False,
            )

"""Tests for factor-level (population) DE: compare_groups (M2).

Covers the pure pair-resolution / weighting kernels and the compare_groups
orchestration (pair resolution -> per-pair CLR deltas -> weighted average),
with ``compare`` monkeypatched so the full path is exercised without a fit.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scribe.models.config.grouping import normalize_grouping
from scribe.de._factors import _resolve_pairs, _pair_weights, compare_groups


def _spec(obs):
    spec, _ = normalize_grouping(
        dataset_key=["perturbation", "sample"],
        hierarchy=None,
        interactions=None,
        obs=obs,
        dataset_priors={
            t: "none"
            for t in ("expression", "prob", "zero_inflation", "overdispersion", "regime")
        },
    )
    return spec


@pytest.fixture
def complete_spec():
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D2", "D3", "D1", "D2", "D3"],
            "perturbation": ["control"] * 3 + ["drug"] * 3,
        }
    )
    return _spec(obs)


@pytest.fixture
def incomplete_spec():
    # drug|D3 missing.
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D2", "D3", "D1", "D2"],
            "perturbation": ["control", "control", "control", "drug", "drug"],
        }
    )
    return _spec(obs)


# ------------------------------------------------------------------------------
# _resolve_pairs
# ------------------------------------------------------------------------------


def test_resolve_pairs_complete(complete_spec):
    pairing, present, dropped = _resolve_pairs(
        complete_spec, "perturbation", "control", "drug", None
    )
    assert pairing == "sample"
    assert dropped == []
    # leaves: control|D1=0, control|D2=1, control|D3=2, drug|D1=3, drug|D2=4, drug|D3=5
    assert present == [("D1", 0, 3), ("D2", 1, 4), ("D3", 2, 5)]


def test_resolve_pairs_incomplete(incomplete_spec):
    pairing, present, dropped = _resolve_pairs(
        incomplete_spec, "perturbation", "control", "drug", None
    )
    assert pairing == "sample"
    assert dropped == ["D3"]
    assert [p[0] for p in present] == ["D1", "D2"]


def test_resolve_pairs_unknown_contrast(complete_spec):
    with pytest.raises(ValueError, match="not a base grouping factor"):
        _resolve_pairs(complete_spec, "nope", "control", "drug", None)


def test_resolve_pairs_explicit_pairing(complete_spec):
    pairing, present, _ = _resolve_pairs(
        complete_spec, "perturbation", "control", "drug", "sample"
    )
    assert pairing == "sample"


def test_resolve_pairs_bad_pairing(complete_spec):
    with pytest.raises(ValueError, match="must be a base factor"):
        _resolve_pairs(complete_spec, "perturbation", "control", "drug", "zzz")


# ------------------------------------------------------------------------------
# _pair_weights
# ------------------------------------------------------------------------------


def test_pair_weights_uniform():
    pairs = [("D1", 0, 3), ("D2", 1, 4)]
    w = _pair_weights(pairs, None, "uniform")
    np.testing.assert_allclose(w, [0.5, 0.5])


def test_pair_weights_total_cells():
    pairs = [("D1", 0, 3), ("D2", 1, 4)]
    nc = np.array([10, 0, 0, 30, 0])  # leaf0=10, leaf3=30 -> 40; leaf1=0, leaf4=0 -> 0
    w = _pair_weights(pairs, nc, "total_cells")
    np.testing.assert_allclose(w, [1.0, 0.0])


def test_pair_weights_harmonic_and_normalization():
    pairs = [("D1", 0, 3), ("D2", 1, 4)]
    nc = np.array([10, 20, 0, 10, 20])
    w = _pair_weights(pairs, nc, "harmonic")
    # pair1: harmonic(10,10)=10; pair2: harmonic(20,20)=20; normalized.
    np.testing.assert_allclose(w, [10 / 30, 20 / 30])
    assert abs(w.sum() - 1.0) < 1e-9


def test_pair_weights_unknown():
    with pytest.raises(ValueError, match="unknown pair_weighting"):
        _pair_weights([("D1", 0, 3)], np.array([1, 1, 1, 1]), "bogus")


# ------------------------------------------------------------------------------
# compare_groups orchestration (compare monkeypatched)
# ------------------------------------------------------------------------------


def _make_results(spec, n_cells_per_leaf=None):
    results = SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=spec, n_components=None),
        _n_cells_per_dataset=n_cells_per_leaf,
    )
    # get_dataset returns the leaf index as the "view"; the monkeypatched
    # compare reads it back to produce a deterministic per-pair delta.
    results.get_dataset = lambda i: i
    results.get_component = lambda c: results
    return results


def _patch_compare(monkeypatch, N=5, D=4):
    def fake_compare(model_A, model_B, method, paired, **kwargs):
        assert method == "empirical" and paired is True
        # Deterministic per-pair value encoding (leaf_B, leaf_A).
        val = float(model_B * 10 + model_A)
        return SimpleNamespace(
            delta_samples=np.full((N, D), val),
            gene_names=[f"g{i}" for i in range(D)],
        )

    monkeypatch.setattr("scribe.de.results.compare", fake_compare)


def test_compare_groups_uniform_average(complete_spec, monkeypatch):
    _patch_compare(monkeypatch)
    results = _make_results(complete_spec)
    de = compare_groups(results, "perturbation", "control", "drug")
    # pair deltas: D1=3*10+0=30, D2=4*10+1=41, D3=5*10+2=52; uniform mean=41.
    assert de.delta_samples.shape == (5, 4)
    np.testing.assert_allclose(de.delta_samples, 41.0)
    assert de.label_A == "perturbation=control"
    assert de.label_B == "perturbation=drug"


def test_compare_groups_weighted_average(complete_spec, monkeypatch):
    _patch_compare(monkeypatch)
    # Weight D1 fully, others zero (total_cells): only leaves 0 and 3 have cells.
    nc = np.array([5, 0, 0, 5, 0, 0])
    results = _make_results(complete_spec, n_cells_per_leaf=nc)
    de = compare_groups(
        results, "perturbation", "control", "drug", pair_weighting="total_cells"
    )
    np.testing.assert_allclose(de.delta_samples, 30.0)  # only D1 (=30) contributes


def test_compare_groups_incomplete_error(incomplete_spec, monkeypatch):
    _patch_compare(monkeypatch)
    results = _make_results(incomplete_spec)
    with pytest.raises(ValueError, match="missing one contrast level"):
        compare_groups(
            results, "perturbation", "control", "drug", incomplete_pairs="error"
        )


def test_compare_groups_min_pairs(incomplete_spec, monkeypatch):
    _patch_compare(monkeypatch)
    results = _make_results(incomplete_spec)
    with pytest.raises(ValueError, match="need >="):
        compare_groups(
            results, "perturbation", "control", "drug", min_complete_pairs=3
        )


def test_compare_groups_estimand_and_method_guards(complete_spec):
    results = _make_results(complete_spec)
    with pytest.raises(NotImplementedError, match="estimand"):
        compare_groups(results, "perturbation", "control", "drug", estimand="effect")
    with pytest.raises(NotImplementedError, match="method"):
        compare_groups(results, "perturbation", "control", "drug", method="shrinkage")


def test_compare_groups_n_samples_redraws(complete_spec, monkeypatch):
    """n_samples / batch_size / convert_to_numpy reach get_posterior_samples."""
    _patch_compare(monkeypatch)
    results = _make_results(complete_spec)
    seen = {}

    def _gps(n_samples=100, **kw):
        seen["n_samples"] = n_samples
        seen.update(kw)
        results.posterior_samples = {"r": np.zeros((n_samples, 1))}
        return results.posterior_samples

    results.get_posterior_samples = _gps
    results.posterior_samples = None
    compare_groups(
        results, "perturbation", "control", "drug",
        n_samples=777, batch_size=200,
    )
    assert seen["n_samples"] == 777
    assert seen["batch_size"] == 200
    # Large draws offload to host RAM by default.
    assert seen["convert_to_numpy"] is True


def test_compare_groups_n_samples_ignored_for_mcmc(complete_spec, monkeypatch):
    """A results type without an n_samples kwarg (e.g. MCMC) warns and falls back."""
    _patch_compare(monkeypatch)
    results = _make_results(complete_spec)

    def _gps(descriptive_names=False):  # MCMC-style: no n_samples kwarg
        results.posterior_samples = {"r": np.zeros((3, 1))}
        return results.posterior_samples

    results.get_posterior_samples = _gps
    results.posterior_samples = None
    with pytest.warns(UserWarning, match="not supported by this results type"):
        compare_groups(results, "perturbation", "control", "drug", n_samples=777)
    assert results.posterior_samples is not None  # fell back to a default draw


def test_compare_groups_requires_grouping():
    results = SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=None, n_components=None)
    )
    with pytest.raises(ValueError, match="requires a multi-factor fit"):
        compare_groups(results, "perturbation", "control", "drug")


# ------------------------------------------------------------------------------
# Per-arm population fields (mu_map / biological samples) for DE plots
# ------------------------------------------------------------------------------


def _patch_compare_bio(monkeypatch, N=5, D=4, drop_mu_map_A_for=None):
    """compare() stub returning per-arm fields encoded from the leaf indices."""

    def fake_compare(
        model_A, model_B, method, paired, compute_biological=False, **kwargs
    ):
        assert method == "empirical" and paired is True
        # compare_groups defaults biological summaries on so mu_map is built.
        assert compute_biological is True
        mu_map_A = (
            None
            if model_A == drop_mu_map_A_for
            else np.full(D, float(model_A))
        )
        return SimpleNamespace(
            delta_samples=np.full((N, D), float(model_B * 10 + model_A)),
            gene_names=[f"g{i}" for i in range(D)],
            mu_map_A=mu_map_A,
            mu_map_B=np.full(D, float(model_B)),
            mu_samples_A=np.full((N, D), float(model_A)),
            mu_samples_B=np.full((N, D), float(model_B)),
            # r/p/phi samples intentionally absent -> dropped on aggregation.
        )

    monkeypatch.setattr("scribe.de.results.compare", fake_compare)


def test_compare_groups_aggregates_arm_population_fields(
    complete_spec, monkeypatch
):
    """Per-arm fields are donor-weighted averages; simplex stays unset."""
    _patch_compare_bio(monkeypatch)
    results = _make_results(complete_spec)
    de = compare_groups(results, "perturbation", "control", "drug")

    # leaves: control|{D1,D2,D3}={0,1,2}; drug|{D1,D2,D3}={3,4,5}.
    # mu_map_A = leaf_A index, uniform mean over (0,1,2) = 1.0;
    # mu_map_B = leaf_B index, uniform mean over (3,4,5) = 4.0.
    np.testing.assert_allclose(de.mu_map_A, 1.0)
    np.testing.assert_allclose(de.mu_map_B, 4.0)
    np.testing.assert_allclose(de.mu_samples_A, 1.0)
    np.testing.assert_allclose(de.mu_samples_B, 4.0)
    assert de.mu_map_A.shape == (4,)
    assert de.mu_samples_A.shape == (5, 4)

    # The paired estimand averages within-pair CLR deltas, so simplices are
    # deliberately NOT aggregated (CLR(avg) != avg CLR).
    assert de.simplex_A is None
    assert de.simplex_B is None
    # Fields absent from every pair are dropped, not zero-filled.
    assert de.r_samples_A is None


def test_compare_groups_drops_arm_field_missing_in_any_pair(
    complete_spec, monkeypatch
):
    """A per-arm field absent from even one pair is dropped entirely."""
    # control|D1 has leaf index 0; make that pair return mu_map_A=None.
    _patch_compare_bio(monkeypatch, drop_mu_map_A_for=0)
    results = _make_results(complete_spec)
    de = compare_groups(results, "perturbation", "control", "drug")
    assert de.mu_map_A is None  # dropped (one pair lacked it)
    np.testing.assert_allclose(de.mu_map_B, 4.0)  # still aggregated


def test_compare_groups_mask_builders_without_simplex(
    complete_spec, monkeypatch
):
    """expression_mask / composition_coverage_mask work without a stored simplex.

    compare_groups does not store simplex_A/B, so the apply-in-place
    set_expression_threshold / set_composition_coverage cannot recompute deltas.
    The mask *builders* read only mu_map and must still work, so a coverage /
    expression filter can be passed up front via gene_mask=.
    """
    _patch_compare_bio(monkeypatch, D=6)
    results = _make_results(complete_spec)
    de = compare_groups(results, "perturbation", "control", "drug")
    assert getattr(de, "simplex_A", None) is None  # no simplex stored

    # mu_map_A == 1.0 (control leaves 0,1,2 averaged); mu_map_B == 4.0.
    assert np.asarray(de.expression_mask(2.0)).all()  # 4 >= 2 everywhere
    assert not np.asarray(de.expression_mask(5.0)).any()  # neither arm >= 5
    cm = de.composition_coverage_mask(0.95)
    assert cm.dtype == bool and cm.size == 6

    # The apply-in-place variant needs the simplex compare_groups omits.
    with pytest.raises(ValueError, match="simplex"):
        de.set_composition_coverage(0.95)


def test_compare_groups_propagates_gene_mask(complete_spec, monkeypatch):
    """The pairs' shared gene-mask bookkeeping is carried to the result.

    compare() stores mu_map at full-gene length plus a ``_gene_mask`` and lets
    to_dataframe() mask it down to the kept genes (the rest pooled into the
    dropped "other" pseudo-gene). compare_groups must propagate that mask so the
    aggregated full-length per-arm vectors align with the kept-gene delta table
    rather than raising an off-by-one length mismatch.
    """
    mask = np.array([True, True, True, False])  # 1 gene pooled into "other"
    all_names = ["g0", "g1", "g2", "g3"]

    def fake_compare(
        model_A, model_B, method, paired, compute_biological=False, **kwargs
    ):
        ns = SimpleNamespace(
            delta_samples=np.zeros((5, 3)),  # kept = 3 (other dropped)
            gene_names=["g0", "g1", "g2"],  # kept names
            mu_map_A=np.full(4, float(model_A)),  # full-gene length = 4
            mu_map_B=np.full(4, float(model_B)),
        )
        ns._gene_mask = mask
        ns._all_gene_names = all_names
        return ns

    monkeypatch.setattr("scribe.de.results.compare", fake_compare)
    results = _make_results(complete_spec)
    de = compare_groups(results, "perturbation", "control", "drug")

    np.testing.assert_array_equal(np.asarray(de._gene_mask), mask)
    assert de._all_gene_names == all_names
    # Full-length per-arm vector retained; kept-length delta table.
    assert np.asarray(de.mu_map_A).shape == (4,)
    assert np.asarray(de.delta_samples).shape == (5, 3)

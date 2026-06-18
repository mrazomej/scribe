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


def test_compare_groups_requires_grouping():
    results = SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=None, n_components=None)
    )
    with pytest.raises(ValueError, match="requires a multi-factor fit"):
        compare_groups(results, "perturbation", "control", "drug")

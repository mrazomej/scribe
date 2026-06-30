"""Tests for the NB-family gate on context-aware DE posterior narrowing.

Narrowing the posterior draw to the DE keep-set is an NB-family optimisation.
These tests assert:

- ``_supports_de_narrowing`` is an explicit allowlist — True for NB roots and
  their parameterization/mixture variants, False for two-state, the marginal
  families (LNM/PLN/NBLN/TSLN), and any unknown/future family;
- ``compare_groups`` requests ``purpose="de_paired"`` only for NB-family fits
  and bypasses (no ``purpose``) for everything else.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scribe.models.config.grouping import normalize_grouping
from scribe.de._factors import (
    _supports_de_narrowing,
    _ensure_posterior_draw,
    compare_groups,
)


# ---------------------------------------------------------------------------
# _supports_de_narrowing — allowlist semantics
# ---------------------------------------------------------------------------


def _results_with_base_model(base_model):
    # No get_compositional_samples attr -> has_compositional_marginal is False,
    # so the base_model deny-set / allowlist drive the decision.
    return SimpleNamespace(model_config=SimpleNamespace(base_model=base_model))


@pytest.mark.parametrize(
    "base_model",
    ["nbdm", "zinb", "nbvcp", "zinbvcp", "bnb",
     # parameterization/mixture variants collapse to the same root:
     "nbdm_standard", "zinb_odds_ratio", "zinbvcp_mix"],
)
def test_nb_family_supported(base_model):
    assert _supports_de_narrowing(_results_with_base_model(base_model)) is True


@pytest.mark.parametrize(
    "base_model",
    ["twostate", "twostatevcp", "twostate_ln_rate", "twostate_ln_logit",
     "nbln", "pln", "lnm", "lnmvcp",
     # unknown / future family must NOT narrow by default (allowlist):
     "bogusfuture", "", "poisson"],
)
def test_non_nb_family_bypassed(base_model):
    assert _supports_de_narrowing(_results_with_base_model(base_model)) is False


def test_no_model_config_bypassed():
    assert _supports_de_narrowing(SimpleNamespace(model_config=None)) is False


def test_compositional_marginal_bypassed_even_if_root_unknown():
    # A result that exposes a compositional marginal is bypassed up front,
    # regardless of base_model (defense-in-depth, evaluated first).
    r = SimpleNamespace(
        model_config=SimpleNamespace(base_model="nbln"),
        get_compositional_samples=lambda *a, **k: None,
    )
    assert _supports_de_narrowing(r) is False


# ---------------------------------------------------------------------------
# compare_groups wiring — purpose requested only for NB family
# ---------------------------------------------------------------------------


def _spec(obs):
    spec, _ = normalize_grouping(
        dataset_key=["perturbation", "sample"],
        hierarchy=None,
        interactions=None,
        obs=obs,
        dataset_priors={
            t: "none"
            for t in (
                "expression", "prob", "zero_inflation",
                "overdispersion", "regime",
            )
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


def _patch_compare(monkeypatch, N=5, D=4):
    def fake_compare(model_A, model_B, method, paired, **kwargs):
        return SimpleNamespace(
            delta_samples=np.full((N, D), float(model_B * 10 + model_A)),
            gene_names=[f"g{i}" for i in range(D)],
        )

    monkeypatch.setattr("scribe.de.results.compare", fake_compare)


def _make_results(spec, base_model):
    results = SimpleNamespace(
        model_config=SimpleNamespace(
            grouping_spec=spec, n_components=None, base_model=base_model
        ),
        _n_cells_per_dataset=None,
    )
    results.get_dataset = lambda i: i
    results.get_component = lambda c: results
    results.posterior_samples = None
    return results


def _run_capturing_purpose(monkeypatch, results):
    seen = {}

    def _gps(n_samples=100, **kw):
        seen["called"] = True
        seen.update(kw)
        results.posterior_samples = {"r": np.zeros((n_samples, 1))}
        return results.posterior_samples

    results.get_posterior_samples = _gps
    compare_groups(results, "perturbation", "control", "drug", n_samples=64)
    return seen


def test_compare_groups_requests_de_paired_for_nb(complete_spec, monkeypatch):
    _patch_compare(monkeypatch)
    results = _make_results(complete_spec, base_model="nbdm")
    seen = _run_capturing_purpose(monkeypatch, results)
    assert seen.get("purpose") == "de_paired"


def test_compare_groups_bypasses_for_non_nb(complete_spec, monkeypatch):
    _patch_compare(monkeypatch)
    for bm in ("twostate", "nbln", "lnm"):
        results = _make_results(complete_spec, base_model=bm)
        seen = _run_capturing_purpose(monkeypatch, results)
        # No narrowing -> no purpose kwarg reaches get_posterior_samples.
        assert "purpose" not in seen, bm
        assert seen.get("called") is True, bm


# ---------------------------------------------------------------------------
# _ensure_posterior_draw — site-aware cache reuse (superset rule)
# ---------------------------------------------------------------------------


def _cache_results(*, n, is_full, sites):
    r = SimpleNamespace(
        model_config=SimpleNamespace(n_components=None, base_model="nbdm"),
        posterior_samples={"r": np.zeros((n, 1))},
        _posterior_is_full=is_full,
        _posterior_sites=sites,
    )
    calls = {"n": 0}

    def _gps(n_samples=100, **kw):
        calls["n"] += 1
        r.posterior_samples = {"r": np.zeros((n_samples, 1))}
        r._posterior_is_full = (
            kw.get("purpose") is None and kw.get("return_sites") is None
        )
        return r.posterior_samples

    r.get_posterior_samples = _gps
    return r, calls


def test_full_cache_is_reused_for_de_paired():
    r, calls = _cache_results(n=64, is_full=True, sites=None)
    _ensure_posterior_draw(r, 64, None, None, purpose="de_paired")
    assert calls["n"] == 0  # full cache covers any narrowed request -> reuse


def test_matching_de_paired_cache_is_reused():
    r, calls = _cache_results(
        n=64, is_full=False, sites=frozenset({"r", "p", "mu", "phi"})
    )
    _ensure_posterior_draw(r, 64, None, None, purpose="de_paired")
    assert calls["n"] == 0  # requested keep-set already covered -> reuse


def test_effect_cache_not_reused_for_paired():
    # A de_effect-narrow cache (effect site only) lacks r -> a de_paired call
    # (needs r) must force a redraw.
    r, calls = _cache_results(
        n=64, is_full=False,
        sites=frozenset({"mu_perturbation__sample_effect"}),
    )
    _ensure_posterior_draw(r, 64, None, None, purpose="de_paired")
    assert calls["n"] == 1  # superset check fails -> redraw


def test_narrowed_cache_not_reused_for_full_request():
    # A full (purpose=None) request must not reuse a narrowed cache.
    r, calls = _cache_results(
        n=64, is_full=False, sites=frozenset({"r", "p", "mu", "phi"})
    )
    _ensure_posterior_draw(r, 64, None, None, purpose=None)
    assert calls["n"] == 1  # narrowed cache cannot satisfy a full draw -> redraw


def test_count_mismatch_forces_redraw_even_if_sites_ok():
    r, calls = _cache_results(n=64, is_full=True, sites=None)
    _ensure_posterior_draw(r, 128, None, None, purpose="de_paired")
    assert calls["n"] == 1  # different n_samples -> redraw despite full cache

"""Unit tests for context-aware posterior-site selection.

Covers the pure-logic pieces of the site-selection feature (no model fit
required):

- ``_as_site_set`` string-vs-iterable normalization (the ``set("mu")`` trap);
- ``resolve_keep_set`` / ``_canonical_param_keys`` policy resolution, including
  the regression guards the auditor flagged (scalar ``p`` kept, ``gate``
  excluded, explicit list wins, unknown purpose raises);
- the shared ``_require_full_posterior_cache`` guard.
"""

from types import SimpleNamespace

import pytest

from scribe.svi._posterior_policy import (
    _as_site_set,
    _canonical_param_keys,
    resolve_keep_set,
)
from scribe.sampling._predictive import _as_site_set as _sampling_as_site_set
from scribe.svi._sampling_posterior_predictive import (
    PosteriorPredictiveSamplingMixin,
)


# ---------------------------------------------------------------------------
# _as_site_set normalization
# ---------------------------------------------------------------------------


def test_as_site_set_single_string_is_one_name():
    # The classic trap: set("mu") == {"m", "u"}. A bare string must be one name.
    assert _as_site_set("mu") == {"mu"}
    assert _as_site_set("r") == {"r"}


def test_as_site_set_iterable_and_none():
    assert _as_site_set(["r", "p"]) == {"r", "p"}
    assert _as_site_set(("r", "mu", "phi")) == {"r", "mu", "phi"}
    assert _as_site_set(None) is None


def test_as_site_set_is_single_shared_definition():
    # svi re-exports the sampling-layer definition (single source of truth).
    assert _as_site_set is _sampling_as_site_set


# ---------------------------------------------------------------------------
# resolve_keep_set / _canonical_param_keys
# ---------------------------------------------------------------------------


def _mc(n_components=None):
    return SimpleNamespace(n_components=n_components)


def test_purpose_none_keeps_everything():
    assert resolve_keep_set(_mc(), purpose=None) is None


def test_purpose_ppc_and_all_keep_everything():
    assert resolve_keep_set(_mc(), purpose="ppc") is None
    assert resolve_keep_set(_mc(), purpose="all") is None


def test_de_paired_is_canonical_contract_and_excludes_gate():
    keep = resolve_keep_set(_mc(), purpose="de_paired")
    assert keep == {"r", "p", "mu", "phi"}
    # gate is NOT a DE input -> excluded (regression guard).
    assert "gate" not in keep
    # capture + per-factor effects fall out (whitelist).
    assert "p_capture" not in keep
    assert "mu_perturbation__sample_effect" not in keep


def test_de_paired_keeps_scalar_shared_p():
    # The keep-set is defined by consumption, not by an is_gene_specific proxy;
    # ``p`` (often scalar/shared) must always be present.
    assert "p" in resolve_keep_set(_mc(), purpose="de_paired")


def test_de_paired_includes_mixing_weights_for_mixture():
    assert resolve_keep_set(_mc(n_components=3), purpose="de_paired") == {
        "r", "p", "mu", "phi", "mixing_weights",
    }
    # ... and via the explicit flag even if n_components is unset.
    assert "mixing_weights" in resolve_keep_set(
        _mc(), purpose="de_paired", mixture_weighted=True
    )


def test_de_paired_always_contains_r():
    assert "r" in resolve_keep_set(_mc(), purpose="de_paired")


def test_de_effect_returns_effect_and_scale_candidates():
    keep = resolve_keep_set(
        _mc(), purpose="de_effect", factor_name="perturbation:sample"
    )
    # interaction ":" -> "__"; both mu/r targets + scale candidates included.
    assert keep == {
        "mu_perturbation__sample_effect",
        "r_perturbation__sample_effect",
        "mu_perturbation__sample_scale",
        "r_perturbation__sample_scale",
    }


def test_de_effect_requires_factor_name():
    with pytest.raises(ValueError, match="factor_name"):
        resolve_keep_set(_mc(), purpose="de_effect")


def test_explicit_return_sites_wins_over_purpose():
    # Escape hatch: an explicit list beats the named policy.
    assert resolve_keep_set(
        _mc(), purpose="de_paired", return_sites=["r"]
    ) == {"r"}
    # A bare string is one site name, not characters.
    assert resolve_keep_set(_mc(), return_sites="mu") == {"mu"}


def test_unknown_purpose_raises():
    with pytest.raises(ValueError, match="unknown purpose"):
        resolve_keep_set(_mc(), purpose="bogus")


def test_canonical_param_keys_fixed_contract():
    assert _canonical_param_keys(_mc()) == {"r", "p", "mu", "phi"}


# ---------------------------------------------------------------------------
# _require_full_posterior_cache guard
# ---------------------------------------------------------------------------


class _GuardStub(PosteriorPredictiveSamplingMixin):
    """Minimal carrier of the cache flags + the guard method."""

    def __init__(self, *, posterior_samples, is_full, sites=None):
        self.posterior_samples = posterior_samples
        self._posterior_is_full = is_full
        self._posterior_sites = sites


def test_guard_noop_when_full():
    stub = _GuardStub(posterior_samples={"r": 1}, is_full=True)
    # Should not raise.
    stub._require_full_posterior_cache(method="m")


def test_guard_noop_when_no_cache():
    # No cache yet -> the consumer draws a fresh full cache; guard is a no-op.
    stub = _GuardStub(posterior_samples=None, is_full=True)
    stub._require_full_posterior_cache(method="m")
    # Even if a stale flag says not-full, a None cache must not raise.
    stub2 = _GuardStub(posterior_samples=None, is_full=False)
    stub2._require_full_posterior_cache(method="m")


def test_guard_raises_on_narrowed_cache():
    stub = _GuardStub(
        posterior_samples={"r": 1},
        is_full=False,
        sites=frozenset({"r", "p", "mu", "phi"}),
    )
    with pytest.raises(RuntimeError, match="narrowed for differential expression"):
        stub._require_full_posterior_cache(method="get_predictive_samples")

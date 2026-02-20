"""Tests for the PSIS-LOO module (scribe.mc._psis_loo).

Validates the Pareto tail fitting, single-observation smoothing, and the
full PSIS-LOO computation including k̂ diagnostics and elpd estimates.
"""

import pytest
import numpy as np

from scribe.mc._psis_loo import (
    _n_tail,
    _fit_gpd,
    _pareto_smooth_single,
    compute_psis_loo,
    psis_loo_summary,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def well_behaved_log_liks():
    """Log-likelihood matrix with well-behaved IS weights.

    All log-liks drawn from a moderate-variance normal → k̂ should be low.
    """
    rng = np.random.default_rng(42)
    S, n = 500, 100
    return rng.normal(-3.0, 0.3, size=(S, n))


@pytest.fixture
def constant_log_liks():
    """All samples agree → no tail smoothing needed."""
    S, n = 200, 50
    return np.full((S, n), -2.0)


# --------------------------------------------------------------------------
# _n_tail
# --------------------------------------------------------------------------


def test_n_tail_minimum():
    """Should be at least 5."""
    assert _n_tail(10) >= 5


def test_n_tail_small_S():
    """For small S, min(S//5, ceil(3*sqrt(S))) should be correct."""
    S = 50
    expected = min(50 // 5, int(np.ceil(3.0 * np.sqrt(50))))
    assert _n_tail(S) == max(5, expected)


def test_n_tail_large_S():
    """For large S=1000, M should be S//5 = 200."""
    S = 1000
    result = _n_tail(S)
    # min(200, ceil(3*31.6)) = min(200, 95) = 95
    expected = max(5, min(200, int(np.ceil(3.0 * np.sqrt(1000)))))
    assert result == expected


# --------------------------------------------------------------------------
# _fit_gpd
# --------------------------------------------------------------------------


def test_fit_gpd_returns_two_floats():
    """_fit_gpd should return (k_hat, sigma_hat) as floats."""
    rng = np.random.default_rng(0)
    x = rng.exponential(1.0, size=50)
    k, sigma = _fit_gpd(x)
    assert isinstance(k, float)
    assert isinstance(sigma, float)


def test_fit_gpd_sigma_positive():
    """Scale sigma must be positive."""
    rng = np.random.default_rng(1)
    x = rng.exponential(2.0, size=100)
    _, sigma = _fit_gpd(x)
    assert sigma > 0


def test_fit_gpd_k_in_reasonable_range():
    """k̂ should be in [-2, 2] (clamped)."""
    rng = np.random.default_rng(2)
    x = rng.exponential(1.0, size=30)
    k, _ = _fit_gpd(x)
    assert -2.0 <= k <= 2.0


def test_fit_gpd_constant_input():
    """Should not raise for constant (degenerate) input; sigma must be positive."""
    x = np.zeros(20)
    k, sigma = _fit_gpd(x)
    assert isinstance(k, float)
    # After filtering zero values, the array is empty → fallback returns sigma=1e-8
    assert sigma > 0 and np.isfinite(sigma)


# --------------------------------------------------------------------------
# _pareto_smooth_single
# --------------------------------------------------------------------------


def test_pareto_smooth_single_output_shape():
    """Smoothed weights should have same shape as input."""
    rng = np.random.default_rng(10)
    log_w = rng.normal(0.0, 1.0, size=500)
    smoothed, k_hat = _pareto_smooth_single(log_w)
    assert smoothed.shape == log_w.shape


def test_pareto_smooth_single_k_is_float():
    """k_hat should be a scalar float."""
    rng = np.random.default_rng(11)
    log_w = rng.normal(0.0, 1.0, size=300)
    _, k_hat = _pareto_smooth_single(log_w)
    assert isinstance(k_hat, float)


def test_pareto_smooth_does_not_change_bulk():
    """Only the tail (M largest) should change; bulk should be unchanged."""
    rng = np.random.default_rng(12)
    S = 300
    log_w = rng.normal(0.0, 0.5, size=S)
    M = _n_tail(S)
    sort_idx = np.argsort(log_w)
    bulk_idx = sort_idx[: S - M]

    smoothed, _ = _pareto_smooth_single(log_w)
    # Bulk positions should be identical
    np.testing.assert_array_equal(smoothed[bulk_idx], log_w[bulk_idx])


# --------------------------------------------------------------------------
# compute_psis_loo
# --------------------------------------------------------------------------


def test_psis_loo_output_keys(well_behaved_log_liks):
    """compute_psis_loo should return all required keys."""
    result = compute_psis_loo(well_behaved_log_liks)
    expected = {"elpd_loo", "p_loo", "looic", "elpd_loo_i", "k_hat", "lppd", "n_bad"}
    assert expected == set(result.keys())


def test_psis_loo_elpd_loo_i_shape(well_behaved_log_liks):
    """elpd_loo_i should have shape (n,)."""
    n = well_behaved_log_liks.shape[1]
    result = compute_psis_loo(well_behaved_log_liks)
    assert result["elpd_loo_i"].shape == (n,)


def test_psis_loo_k_hat_shape(well_behaved_log_liks):
    """k_hat should have shape (n,)."""
    n = well_behaved_log_liks.shape[1]
    result = compute_psis_loo(well_behaved_log_liks)
    assert result["k_hat"].shape == (n,)


def test_psis_loo_elpd_loo_finite(well_behaved_log_liks):
    """elpd_loo should be a finite scalar."""
    result = compute_psis_loo(well_behaved_log_liks)
    assert np.isfinite(result["elpd_loo"])


def test_psis_loo_looic_formula(well_behaved_log_liks):
    """LOO-IC should equal -2 * elpd_loo."""
    result = compute_psis_loo(well_behaved_log_liks)
    assert abs(result["looic"] - (-2.0 * result["elpd_loo"])) < 1e-8


def test_psis_loo_p_loo_formula(well_behaved_log_liks):
    """p_loo should equal lppd - elpd_loo."""
    result = compute_psis_loo(well_behaved_log_liks)
    assert abs(result["p_loo"] - (result["lppd"] - result["elpd_loo"])) < 1e-8


def test_psis_loo_constant_logliks(constant_log_liks):
    """With constant log-liks, elpd_loo should be finite.

    When all S samples produce the same log-likelihood (zero posterior
    variance), PSIS detects zero spread, skips tail smoothing (k̂=0),
    and the IS-weighted LOO estimate equals the in-sample lppd.
    """
    result = compute_psis_loo(constant_log_liks)
    assert np.isfinite(result["elpd_loo"]), "elpd_loo must be finite for constant log-liks"
    # lppd = n * ell and elpd_loo should agree closely (p_loo ≈ 0)
    assert abs(result["p_loo"]) < 0.1


def test_psis_loo_k_hat_well_behaved(well_behaved_log_liks):
    """With low-variance log-liks, most k̂ should be < 0.7."""
    result = compute_psis_loo(well_behaved_log_liks)
    frac_bad = result["n_bad"] / well_behaved_log_liks.shape[1]
    assert frac_bad < 0.1, f"Too many bad k̂: {result['n_bad']} / {well_behaved_log_liks.shape[1]}"


def test_psis_loo_elpd_leq_lppd(well_behaved_log_liks):
    """elpd_loo <= lppd (in-sample is always at least as good as LOO)."""
    result = compute_psis_loo(well_behaved_log_liks)
    assert result["elpd_loo"] <= result["lppd"] + 1e-6


def test_psis_loo_accepts_jax_array(well_behaved_log_liks):
    """compute_psis_loo should work with JAX arrays (auto-converted)."""
    import jax.numpy as jnp
    log_liks_jax = jnp.array(well_behaved_log_liks)
    result = compute_psis_loo(log_liks_jax)
    assert np.isfinite(result["elpd_loo"])


# --------------------------------------------------------------------------
# psis_loo_summary
# --------------------------------------------------------------------------


def test_psis_loo_summary_returns_string(well_behaved_log_liks):
    """psis_loo_summary should return a non-empty string."""
    result = compute_psis_loo(well_behaved_log_liks)
    s = psis_loo_summary(result)
    assert isinstance(s, str) and len(s) > 0


def test_psis_loo_summary_contains_elpd(well_behaved_log_liks):
    """Summary should contain the elpd_loo value."""
    result = compute_psis_loo(well_behaved_log_liks)
    s = psis_loo_summary(result)
    assert "elpd_loo" in s

"""Tests for the WAIC module (scribe.mc._waic).

Validates correctness of lppd, p_waic_1, p_waic_2, and WAIC values against
analytically known results and internal consistency checks.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from scribe.mc._waic import (
    _lppd,
    _p_waic_1,
    _p_waic_2,
    compute_waic_stats,
    waic,
    pseudo_bma_weights,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def constant_log_liks():
    """Log-likelihood matrix where all samples agree exactly.

    When all S samples give the same log-likelihood ℓ, then:
      lppd   = n * ℓ
      p_waic_1 = 0   (no gap between log-mean and mean-log when all equal)
      p_waic_2 = 0   (zero variance across samples)
    """
    S, n = 100, 50
    # All samples give log-lik = -3.0
    return jnp.full((S, n), -3.0)


@pytest.fixture
def random_log_liks():
    """Random log-likelihood matrix for shape/consistency checks."""
    rng = np.random.default_rng(42)
    S, n = 200, 100
    return jnp.array(rng.normal(-3.0, 0.5, size=(S, n)), dtype=jnp.float32)


# --------------------------------------------------------------------------
# lppd
# --------------------------------------------------------------------------


def test_lppd_constant(constant_log_liks):
    """With constant log-liks, lppd = n * ℓ exactly."""
    n = constant_log_liks.shape[1]
    expected = n * (-3.0)
    result = _lppd(constant_log_liks, aggregate=True)
    assert jnp.allclose(result, expected, atol=1e-4), f"lppd={result} expected={expected}"


def test_lppd_pointwise_shape(random_log_liks):
    """Pointwise lppd should have shape (n,) when aggregate=False."""
    lppd_pw = _lppd(random_log_liks, aggregate=False)
    assert lppd_pw.shape == (random_log_liks.shape[1],)


def test_lppd_aggregate_equals_sum_of_pointwise(random_log_liks):
    """Aggregated lppd should equal sum of pointwise values."""
    lppd_agg = float(_lppd(random_log_liks, aggregate=True))
    lppd_pw = _lppd(random_log_liks, aggregate=False)
    assert abs(lppd_agg - float(jnp.sum(lppd_pw))) < 1e-3


def test_lppd_numerical_stability():
    """lppd should be finite even for very large negative log-likelihoods."""
    log_liks = jnp.full((100, 50), -1000.0)
    result = _lppd(log_liks)
    assert jnp.isfinite(result)


# --------------------------------------------------------------------------
# p_waic_1 and p_waic_2
# --------------------------------------------------------------------------


def test_p_waic_1_constant_is_zero(constant_log_liks):
    """p_waic_1 = 0 when all samples agree (no uncertainty)."""
    result = float(_p_waic_1(constant_log_liks, aggregate=True))
    assert abs(result) < 1e-4, f"p_waic_1={result} expected 0"


def test_p_waic_2_constant_is_zero(constant_log_liks):
    """p_waic_2 = 0 when all samples agree (zero posterior variance)."""
    result = float(_p_waic_2(constant_log_liks, aggregate=True))
    assert abs(result) < 1e-4, f"p_waic_2={result} expected 0"


def test_p_waic_1_nonnegative(random_log_liks):
    """p_waic_1 should be non-negative (lppd >= mean log-lik by Jensen)."""
    pw1 = float(_p_waic_1(random_log_liks, aggregate=True))
    assert pw1 >= -1e-4, f"p_waic_1={pw1} is unexpectedly negative"


def test_p_waic_2_nonnegative(random_log_liks):
    """p_waic_2 should be non-negative (it is a sum of variances)."""
    pw2 = float(_p_waic_2(random_log_liks, aggregate=True))
    assert pw2 >= -1e-4, f"p_waic_2={pw2} is unexpectedly negative"


def test_p_waic_pointwise_shapes(random_log_liks):
    """Pointwise p_waic values should have shape (n,)."""
    n = random_log_liks.shape[1]
    pw1 = _p_waic_1(random_log_liks, aggregate=False)
    pw2 = _p_waic_2(random_log_liks, aggregate=False)
    assert pw1.shape == (n,)
    assert pw2.shape == (n,)


# --------------------------------------------------------------------------
# compute_waic_stats / waic
# --------------------------------------------------------------------------


def test_waic_stats_keys(random_log_liks):
    """compute_waic_stats should return all required keys."""
    stats = compute_waic_stats(random_log_liks)
    expected_keys = {"lppd", "p_waic_1", "p_waic_2", "elppd_waic_1", "elppd_waic_2", "waic_1", "waic_2"}
    assert expected_keys == set(stats.keys())


def test_waic_formula_consistency(random_log_liks):
    """WAIC = -2 * (lppd - p_waic) should hold for both versions."""
    stats = compute_waic_stats(random_log_liks)
    lppd = float(stats["lppd"])
    pw1 = float(stats["p_waic_1"])
    pw2 = float(stats["p_waic_2"])
    waic1 = float(stats["waic_1"])
    waic2 = float(stats["waic_2"])
    assert abs(waic1 - (-2.0 * (lppd - pw1))) < 1e-3, "WAIC1 formula violated"
    assert abs(waic2 - (-2.0 * (lppd - pw2))) < 1e-3, "WAIC2 formula violated"


def test_waic_pointwise_aggregate_consistency(random_log_liks):
    """Aggregate WAIC should equal sum of pointwise contributions."""
    agg = compute_waic_stats(random_log_liks, aggregate=True)
    pw = compute_waic_stats(random_log_liks, aggregate=False)
    for key in ["lppd", "p_waic_2"]:
        total = float(jnp.sum(pw[key]))
        scalar = float(agg[key])
        assert abs(total - scalar) < 1e-2, f"{key}: sum of pointwise ({total}) != aggregate ({scalar})"


def test_waic_constant_logliks(constant_log_liks):
    """With constant log-liks: WAIC = -2 * n * ℓ."""
    n = constant_log_liks.shape[1]
    ell = -3.0
    stats = waic(constant_log_liks)
    expected_waic = -2.0 * n * ell  # = 300.0
    assert abs(float(stats["waic_2"]) - expected_waic) < 1e-3


def test_waic_two_constant_models():
    """A model with higher lppd (lower WAIC) should be preferred."""
    S, n = 100, 50
    # Model A: log-lik = -2 per obs (better)
    ll_A = jnp.full((S, n), -2.0)
    # Model B: log-lik = -3 per obs (worse)
    ll_B = jnp.full((S, n), -3.0)
    waic_A = float(waic(ll_A)["waic_2"])
    waic_B = float(waic(ll_B)["waic_2"])
    assert waic_A < waic_B, f"WAIC_A={waic_A} should be < WAIC_B={waic_B}"


# --------------------------------------------------------------------------
# pseudo_bma_weights
# --------------------------------------------------------------------------


def test_pseudo_bma_weights_sum_to_one():
    """Pseudo-BMA weights must sum to 1."""
    w = pseudo_bma_weights(jnp.array([200.0, 210.0, 215.0]))
    assert abs(float(jnp.sum(w)) - 1.0) < 1e-5


def test_pseudo_bma_weights_best_model_has_highest_weight():
    """The model with the lowest WAIC should receive the highest weight."""
    waic_vals = jnp.array([100.0, 200.0, 300.0])
    w = pseudo_bma_weights(waic_vals)
    assert int(jnp.argmax(w)) == 0, "Best model (lowest WAIC) should have highest weight"


def test_pseudo_bma_weights_equal_waic_gives_uniform():
    """Equal WAICs should give equal weights."""
    w = pseudo_bma_weights(jnp.array([100.0, 100.0, 100.0]))
    expected = 1.0 / 3.0
    assert jnp.allclose(w, expected, atol=1e-5)


def test_pseudo_bma_weights_length():
    """Output should have same length as input."""
    w = pseudo_bma_weights(jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert w.shape == (4,)

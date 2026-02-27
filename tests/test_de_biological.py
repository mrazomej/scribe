"""Tests for biological-level differential expression metrics.

Tests the LFC (mean shift), log-variance ratio (dispersion shift), and
Gamma Jeffreys divergence (distributional shift) computed from posterior
NB parameter samples.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.de._biological import (
    biological_differential_expression,
    _summarise_signed_metric,
    _summarise_nonneg_metric,
)
from scribe.de.results import ScribeEmpiricalDEResults


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    return random.PRNGKey(42)


@pytest.fixture
def identical_params():
    """Two conditions with identical NB parameters."""
    N, D = 500, 10
    r = jnp.ones((N, D)) * 5.0
    p = jnp.ones((N, D)) * 0.3
    return r, p, r, p


@pytest.fixture
def known_mean_shift():
    """Condition A has 2x the mean of condition B (same dispersion r)."""
    N, D = 500, 10
    r_A = jnp.ones((N, D)) * 5.0
    r_B = jnp.ones((N, D)) * 5.0
    # mu = r * (1-p) / p.  Halving p doubles (1-p)/p.
    # p_A = 0.2 → mu_A = 5*0.8/0.2 = 20
    # p_B = 1/3  → mu_B = 5*2/3/(1/3) = 10
    # LFC = log(20/10) = log(2)
    p_A = jnp.ones((N, D)) * 0.2
    p_B = jnp.ones((N, D)) * (1.0 / 3.0)
    return r_A, p_A, r_B, p_B


# --------------------------------------------------------------------------
# Test: identical conditions
# --------------------------------------------------------------------------


class TestIdenticalConditions:
    """When both conditions are identical, all metrics should be ~0."""

    def test_lfc_zero(self, identical_params):
        r, p, r2, p2 = identical_params
        result = biological_differential_expression(r, r2, p, p2)
        np.testing.assert_allclose(result["lfc_mean"], 0.0, atol=1e-5)

    def test_lvr_zero(self, identical_params):
        r, p, r2, p2 = identical_params
        result = biological_differential_expression(r, r2, p, p2)
        np.testing.assert_allclose(result["lvr_mean"], 0.0, atol=1e-5)

    def test_kl_zero(self, identical_params):
        r, p, r2, p2 = identical_params
        result = biological_differential_expression(r, r2, p, p2)
        np.testing.assert_allclose(result["kl_mean"], 0.0, atol=1e-5)

    def test_lfsr_with_noise(self, rng_key):
        """lfsr should be ~0.5 when there's no true signal but noise."""
        N, D = 2000, 10
        k1, k2 = random.split(rng_key)
        # Independent noise around the same mean for both conditions
        r_A = jnp.ones((N, D)) * 5.0 + 0.1 * random.normal(k1, (N, D))
        r_B = jnp.ones((N, D)) * 5.0 + 0.1 * random.normal(k2, (N, D))
        p = jnp.ones((N, D)) * 0.3
        result = biological_differential_expression(r_A, r_B, p, p)
        np.testing.assert_allclose(result["lfc_lfsr"], 0.5, atol=0.05)


# --------------------------------------------------------------------------
# Test: known mean shift
# --------------------------------------------------------------------------


class TestKnownMeanShift:
    """r_A = r_B = 5, p_A = 0.2, p_B = 1/3 → LFC = log(2)."""

    def test_lfc_log2(self, known_mean_shift):
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(r_A, r_B, p_A, p_B)
        np.testing.assert_allclose(
            result["lfc_mean"], np.log(2), atol=1e-4
        )

    def test_lfc_lfsr_near_zero(self, known_mean_shift):
        """Strong signal → lfsr should be very small."""
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(r_A, r_B, p_A, p_B)
        assert float(jnp.max(result["lfc_lfsr"])) < 0.01

    def test_kl_positive(self, known_mean_shift):
        """Jeffreys divergence should be positive for different params."""
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(r_A, r_B, p_A, p_B)
        assert jnp.all(result["kl_mean"] > 0)

    def test_variance_ratio(self, known_mean_shift):
        """var = r*(1-p)/p^2. Check ratio is consistent with params."""
        r_A, p_A, r_B, p_B = known_mean_shift
        # var_A = 5*0.8/0.04 = 100, var_B = 5*(2/3)/(1/9) = 30
        expected_lvr = np.log(100.0 / 30.0)
        result = biological_differential_expression(r_A, r_B, p_A, p_B)
        np.testing.assert_allclose(
            result["lvr_mean"][0], expected_lvr, atol=1e-4
        )


# --------------------------------------------------------------------------
# Test: practical significance thresholds
# --------------------------------------------------------------------------


class TestPracticalSignificance:

    def test_prob_effect_with_tau(self, known_mean_shift):
        """With tau_lfc < log(2), prob_effect should be ~1."""
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(
            r_A, r_B, p_A, p_B, tau_lfc=0.5
        )
        np.testing.assert_allclose(
            result["lfc_prob_effect"], 1.0, atol=0.01
        )

    def test_prob_effect_with_large_tau(self, known_mean_shift):
        """With tau_lfc > log(2), prob_effect should be ~0."""
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(
            r_A, r_B, p_A, p_B, tau_lfc=1.0
        )
        np.testing.assert_allclose(
            result["lfc_prob_effect"], 0.0, atol=0.01
        )


# --------------------------------------------------------------------------
# Test: auxiliary outputs
# --------------------------------------------------------------------------


class TestAuxiliaryOutputs:

    def test_mu_means(self, known_mean_shift):
        """Check posterior mean biological expression is correct."""
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(r_A, r_B, p_A, p_B)
        np.testing.assert_allclose(result["mu_A_mean"], 20.0, atol=1e-3)
        np.testing.assert_allclose(result["mu_B_mean"], 10.0, atol=1e-3)

    def test_var_means(self, known_mean_shift):
        r_A, p_A, r_B, p_B = known_mean_shift
        result = biological_differential_expression(r_A, r_B, p_A, p_B)
        np.testing.assert_allclose(result["var_A_mean"], 100.0, atol=1e-2)
        np.testing.assert_allclose(result["var_B_mean"], 30.0, atol=1e-2)

    def test_gene_names_auto(self):
        """Auto-generated gene names when not provided."""
        N, D = 50, 5
        r = jnp.ones((N, D))
        p = jnp.ones((N, D)) * 0.5
        result = biological_differential_expression(r, r, p, p)
        assert result["gene_names"] == [f"gene_{i}" for i in range(D)]

    def test_gene_names_provided(self):
        N, D = 50, 3
        r = jnp.ones((N, D))
        p = jnp.ones((N, D)) * 0.5
        names = ["A", "B", "C"]
        result = biological_differential_expression(
            r, r, p, p, gene_names=names
        )
        assert result["gene_names"] == names


# --------------------------------------------------------------------------
# Test: broadcasting for non-hierarchical (shared p)
# --------------------------------------------------------------------------


class TestSharedP:
    """Non-hierarchical models have p shape (N,) instead of (N, D)."""

    def test_shared_p_broadcast(self):
        N, D = 200, 8
        r_A = jnp.ones((N, D)) * 4.0
        r_B = jnp.ones((N, D)) * 2.0
        p_shared = jnp.ones((N,)) * 0.4

        result = biological_differential_expression(
            r_A, r_B, p_shared, p_shared
        )
        # mu = r*(1-p)/p, same p → LFC = log(r_A/r_B) = log(2)
        np.testing.assert_allclose(
            result["lfc_mean"], np.log(2), atol=1e-4
        )

    def test_shared_p_kl(self):
        """Same r, same shared p → KL = 0."""
        N, D = 200, 5
        r = jnp.ones((N, D)) * 3.0
        p_shared = jnp.ones((N,)) * 0.5
        result = biological_differential_expression(r, r, p_shared, p_shared)
        np.testing.assert_allclose(result["kl_mean"], 0.0, atol=1e-5)


# --------------------------------------------------------------------------
# Test: integration with results object
# --------------------------------------------------------------------------


class TestResultsObjectIntegration:

    def test_biological_level_from_results(self):
        """ScribeEmpiricalDEResults.biological_level() round-trip."""
        N, D = 200, 5
        r_A = jnp.ones((N, D)) * 6.0
        r_B = jnp.ones((N, D)) * 3.0
        p = jnp.ones((N, D)) * 0.4

        results = ScribeEmpiricalDEResults(
            delta_samples=jnp.zeros((N, D)),
            gene_names=[f"g{i}" for i in range(D)],
            label_A="treated",
            label_B="control",
            r_samples_A=r_A,
            r_samples_B=r_B,
            p_samples_A=p,
            p_samples_B=p,
        )

        assert results.has_biological
        bio = results.biological_level(tau_lfc=0.5)
        np.testing.assert_allclose(bio["lfc_mean"], np.log(2), atol=1e-4)

    def test_no_biological_raises(self):
        """RuntimeError when biological samples are not stored."""
        N, D = 50, 5
        results = ScribeEmpiricalDEResults(
            delta_samples=jnp.zeros((N, D)),
            gene_names=[f"g{i}" for i in range(D)],
            label_A="A",
            label_B="B",
        )
        assert not results.has_biological
        with pytest.raises(RuntimeError, match="Biological-level DE"):
            results.biological_level()

    def test_caching(self):
        """Calling biological_level with same taus returns cached result."""
        N, D = 100, 3
        r = jnp.ones((N, D)) * 5.0
        p = jnp.ones((N, D)) * 0.3
        results = ScribeEmpiricalDEResults(
            delta_samples=jnp.zeros((N, D)),
            gene_names=["a", "b", "c"],
            label_A="A",
            label_B="B",
            r_samples_A=r,
            r_samples_B=r,
            p_samples_A=p,
            p_samples_B=p,
        )
        bio1 = results.biological_level(tau_lfc=0.1)
        bio2 = results.biological_level(tau_lfc=0.1)
        assert bio1 is bio2

    def test_cache_invalidation(self):
        """Different taus trigger recomputation."""
        N, D = 100, 3
        r = jnp.ones((N, D)) * 5.0
        p = jnp.ones((N, D)) * 0.3
        results = ScribeEmpiricalDEResults(
            delta_samples=jnp.zeros((N, D)),
            gene_names=["a", "b", "c"],
            label_A="A",
            label_B="B",
            r_samples_A=r,
            r_samples_B=r,
            p_samples_A=p,
            p_samples_B=p,
        )
        bio1 = results.biological_level(tau_lfc=0.1)
        bio2 = results.biological_level(tau_lfc=0.5)
        assert bio1 is not bio2


# --------------------------------------------------------------------------
# Test: internal helpers
# --------------------------------------------------------------------------


class TestHelpers:

    def test_summarise_signed_metric(self):
        samples = jnp.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
        stats = _summarise_signed_metric(samples, tau=0.5)
        np.testing.assert_allclose(stats["mean"], [1.0, -1.0], atol=1e-5)
        np.testing.assert_allclose(stats["lfsr"], [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(
            stats["prob_effect"], [1.0, 1.0], atol=1e-5
        )

    def test_summarise_nonneg_metric(self):
        samples = jnp.array([[2.0, 0.1], [3.0, 0.2], [4.0, 0.05]])
        stats = _summarise_nonneg_metric(samples, tau=1.0)
        np.testing.assert_allclose(stats["mean"], [3.0, 0.1167], atol=0.01)
        np.testing.assert_allclose(stats["prob_effect"], [1.0, 0.0])


# --------------------------------------------------------------------------
# Test: parameterization-aware computation (phi path = mean_odds)
# --------------------------------------------------------------------------


class TestPhiPath:
    """Verify that supplying mu and phi uses the numerically stable path."""

    @pytest.fixture
    def mean_odds_params(self):
        """Construct consistent (r, p, mu, phi) for mean_odds."""
        N, D = 500, 10
        phi_A = jnp.ones((N, D)) * 4.0
        phi_B = jnp.ones((N, D)) * 4.0
        mu_A = jnp.ones((N, D)) * 20.0
        mu_B = jnp.ones((N, D)) * 10.0
        # Derived: r = mu * phi, p = 1 / (1 + phi)
        r_A = mu_A * phi_A
        r_B = mu_B * phi_B
        p_A = 1.0 / (1.0 + phi_A)
        p_B = 1.0 / (1.0 + phi_B)
        return r_A, p_A, r_B, p_B, mu_A, mu_B, phi_A, phi_B

    def test_lfc_matches_fallback(self, mean_odds_params):
        """phi path and fallback should agree for well-conditioned params."""
        r_A, p_A, r_B, p_B, mu_A, mu_B, phi_A, phi_B = mean_odds_params
        res_phi = biological_differential_expression(
            r_A, r_B, p_A, p_B,
            mu_samples_A=mu_A, mu_samples_B=mu_B,
            phi_samples_A=phi_A, phi_samples_B=phi_B,
        )
        res_fallback = biological_differential_expression(r_A, r_B, p_A, p_B)
        np.testing.assert_allclose(
            res_phi["lfc_mean"], res_fallback["lfc_mean"], atol=1e-4
        )

    def test_kl_matches_fallback(self, mean_odds_params):
        """KL via beta=1/phi should match beta=p/(1-p) for good params."""
        r_A, p_A, r_B, p_B, mu_A, mu_B, phi_A, phi_B = mean_odds_params
        res_phi = biological_differential_expression(
            r_A, r_B, p_A, p_B,
            mu_samples_A=mu_A, mu_samples_B=mu_B,
            phi_samples_A=phi_A, phi_samples_B=phi_B,
        )
        res_fallback = biological_differential_expression(r_A, r_B, p_A, p_B)
        np.testing.assert_allclose(
            res_phi["kl_mean"], res_fallback["kl_mean"], rtol=1e-3
        )

    def test_variance_via_phi(self, mean_odds_params):
        """var = mu * (1 + phi) should match var = mu / p."""
        r_A, p_A, r_B, p_B, mu_A, mu_B, phi_A, phi_B = mean_odds_params
        res_phi = biological_differential_expression(
            r_A, r_B, p_A, p_B,
            mu_samples_A=mu_A, mu_samples_B=mu_B,
            phi_samples_A=phi_A, phi_samples_B=phi_B,
        )
        # var_A = 20 * (1 + 4) = 100
        np.testing.assert_allclose(res_phi["var_A_mean"], 100.0, atol=1e-2)
        # var_B = 10 * (1 + 4) = 50
        np.testing.assert_allclose(res_phi["var_B_mean"], 50.0, atol=1e-2)

    def test_phi_broadcast(self):
        """Shared phi (shape (N,)) broadcasts correctly."""
        N, D = 200, 5
        phi_shared = jnp.ones((N,)) * 2.0
        mu_A = jnp.ones((N, D)) * 10.0
        mu_B = jnp.ones((N, D)) * 5.0
        r_A = mu_A * phi_shared[:, None]
        r_B = mu_B * phi_shared[:, None]
        p = 1.0 / (1.0 + phi_shared)
        result = biological_differential_expression(
            r_A, r_B, p, p,
            mu_samples_A=mu_A, mu_samples_B=mu_B,
            phi_samples_A=phi_shared, phi_samples_B=phi_shared,
        )
        np.testing.assert_allclose(
            result["lfc_mean"], np.log(2), atol=1e-4
        )


# --------------------------------------------------------------------------
# Test: parameterization-aware computation (mu path = mean_prob)
# --------------------------------------------------------------------------


class TestMuPath:
    """Verify that supplying only mu (no phi) uses the mu path."""

    def test_lfc_from_mu(self):
        """LFC should use mu directly when provided."""
        N, D = 300, 5
        mu_A = jnp.ones((N, D)) * 20.0
        mu_B = jnp.ones((N, D)) * 10.0
        p = jnp.ones((N, D)) * 0.3
        r_A = mu_A * (1.0 - p) / p
        r_B = mu_B * (1.0 - p) / p

        result = biological_differential_expression(
            r_A, r_B, p, p,
            mu_samples_A=mu_A, mu_samples_B=mu_B,
        )
        np.testing.assert_allclose(
            result["lfc_mean"], np.log(2), atol=1e-4
        )

    def test_kl_uses_p_for_beta(self):
        """Without phi, beta should still come from p/(1-p)."""
        N, D = 300, 5
        mu = jnp.ones((N, D)) * 10.0
        p = jnp.ones((N, D)) * 0.3
        r = mu * (1.0 - p) / p

        # Identical conditions → KL = 0
        result = biological_differential_expression(
            r, r, p, p, mu_samples_A=mu, mu_samples_B=mu,
        )
        np.testing.assert_allclose(result["kl_mean"], 0.0, atol=1e-5)


# --------------------------------------------------------------------------
# Test: near-zero expression regression (phi path vs fallback)
# --------------------------------------------------------------------------


class TestNearZeroExpression:
    """Regression test: phi path should be stable where fallback blows up.

    For genes with near-zero expression, phi -> 0 and p -> 1.  The
    fallback path computes beta = p / (1 - p) which diverges, while
    the phi path computes beta = 1 / phi which is large but doesn't
    suffer from catastrophic cancellation in the (1 - p) subtraction.

    Both paths give large metrics for truly-zero genes (beta is huge
    either way), but the phi path avoids *infinite* or *NaN* values
    that arise from float32 precision loss in (1 - p) when p ≈ 1.
    """

    def test_phi_path_finite_at_low_expression(self):
        """phi path produces finite KL even when phi is very small."""
        N, D = 100, 5
        # phi ~ 1e-6 → p ≈ 1 - 1e-6, mu ~ 0.001
        phi_small = jnp.ones((N, D)) * 1e-6
        mu_tiny = jnp.ones((N, D)) * 0.001
        r = mu_tiny * phi_small
        p = 1.0 / (1.0 + phi_small)

        # Small perturbation in condition B
        phi_small_B = jnp.ones((N, D)) * 2e-6
        mu_tiny_B = jnp.ones((N, D)) * 0.002
        r_B = mu_tiny_B * phi_small_B
        p_B = 1.0 / (1.0 + phi_small_B)

        result = biological_differential_expression(
            r, r_B, p, p_B,
            mu_samples_A=mu_tiny, mu_samples_B=mu_tiny_B,
            phi_samples_A=phi_small, phi_samples_B=phi_small_B,
        )

        # All values should be finite (no inf/nan)
        assert jnp.all(jnp.isfinite(result["lfc_mean"]))
        assert jnp.all(jnp.isfinite(result["kl_mean"]))
        assert jnp.all(jnp.isfinite(result["lvr_mean"]))

    def test_lfc_correct_at_low_expression(self):
        """LFC from direct mu should be log(2) regardless of expression."""
        N, D = 100, 3
        phi = jnp.ones((N, D)) * 1e-6
        mu_A = jnp.ones((N, D)) * 2e-4
        mu_B = jnp.ones((N, D)) * 1e-4
        r_A = mu_A * phi
        r_B = mu_B * phi
        p = 1.0 / (1.0 + phi)

        result = biological_differential_expression(
            r_A, r_B, p, p,
            mu_samples_A=mu_A, mu_samples_B=mu_B,
            phi_samples_A=phi, phi_samples_B=phi,
        )
        np.testing.assert_allclose(
            result["lfc_mean"], np.log(2), atol=1e-4
        )

    def test_phi_path_identical_gives_zero_kl(self):
        """Identical small-phi conditions → KL = 0 (no spurious signal)."""
        N, D = 100, 5
        phi = jnp.ones((N, D)) * 1e-5
        mu = jnp.ones((N, D)) * 0.01
        r = mu * phi
        p = 1.0 / (1.0 + phi)

        result = biological_differential_expression(
            r, r, p, p,
            mu_samples_A=mu, mu_samples_B=mu,
            phi_samples_A=phi, phi_samples_B=phi,
        )
        np.testing.assert_allclose(result["kl_mean"], 0.0, atol=1e-5)
        np.testing.assert_allclose(result["lfc_mean"], 0.0, atol=1e-5)


# --------------------------------------------------------------------------
# Test: results object integration with native params
# --------------------------------------------------------------------------


class TestResultsObjectWithNativeParams:
    """ScribeEmpiricalDEResults should pass mu/phi to biological_level."""

    def test_biological_level_with_phi(self):
        N, D = 200, 5
        phi = jnp.ones((N, D)) * 3.0
        mu_A = jnp.ones((N, D)) * 15.0
        mu_B = jnp.ones((N, D)) * 15.0
        r = mu_A * phi
        p = 1.0 / (1.0 + phi)

        results = ScribeEmpiricalDEResults(
            delta_samples=jnp.zeros((N, D)),
            gene_names=[f"g{i}" for i in range(D)],
            label_A="A",
            label_B="B",
            r_samples_A=r,
            r_samples_B=r,
            p_samples_A=p,
            p_samples_B=p,
            mu_samples_A=mu_A,
            mu_samples_B=mu_B,
            phi_samples_A=phi,
            phi_samples_B=phi,
        )

        bio = results.biological_level()
        # Identical conditions → all metrics ~ 0
        np.testing.assert_allclose(bio["lfc_mean"], 0.0, atol=1e-5)
        np.testing.assert_allclose(bio["kl_mean"], 0.0, atol=1e-5)

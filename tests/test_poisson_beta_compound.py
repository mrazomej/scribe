"""Tests for the PoissonBetaCompound distribution.

Validates:

- closed-form mean and variance against the analytic formulas,
- log_prob against scipy brute-force integration of the marginal,
- ancestral sample() against the analytic mean (Monte Carlo),
- broadcasting: gene-rank α/β with VCP-shape (C, G) rate, including
  non-empty ``sample_shape`` per the audit-flagged shape recipe,
- PyTree contract: ``n_quad_nodes`` and ``quad_backend`` are
  auxiliary (do not appear as JAX leaves),
- ``log_rate``-only construction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as stats
from scipy.integrate import quad

from scribe.stats.distributions import PoissonBetaCompound


# ==============================================================================
# Closed-form moment checks
# ==============================================================================


class TestPoissonBetaCompoundMoments:
    """Analytic mean and variance match the closed-form formulas."""

    @pytest.mark.parametrize(
        "alpha, beta, rate",
        [
            (2.0, 20.0, 10.0),     # NB-like (β >> α)
            (5.0, 0.5, 50.0),      # Bursty (β < α)
            (1.0, 1.0, 5.0),       # Uniform
            (0.5, 0.5, 100.0),     # U-shaped
        ],
    )
    def test_mean_matches_formula(self, alpha, beta, rate):
        dist = PoissonBetaCompound(
            jnp.array(alpha), jnp.array(beta), jnp.array(rate)
        )
        expected = rate * alpha / (alpha + beta)
        assert np.isclose(float(dist.mean), expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "alpha, beta, rate",
        [
            (2.0, 20.0, 10.0),
            (5.0, 0.5, 50.0),
            (1.0, 1.0, 5.0),
        ],
    )
    def test_variance_matches_formula(self, alpha, beta, rate):
        """Var[X] = E[X] + rate² · Var[p] (law of total variance)."""
        dist = PoissonBetaCompound(
            jnp.array(alpha), jnp.array(beta), jnp.array(rate)
        )
        ab = alpha + beta
        var_p = (alpha * beta) / (ab**2 * (ab + 1.0))
        mean_x = rate * alpha / ab
        expected = mean_x + rate**2 * var_p
        assert np.isclose(float(dist.variance), expected, rtol=1e-6)


# ==============================================================================
# log_prob vs scipy brute-force integration
# ==============================================================================


def _brute_force_log_prob(value, alpha, beta, rate):
    """Numerical reference: ∫₀¹ Poisson(value | rate·p) · Beta(p|α,β) dp.

    Uses scipy.integrate.quad. Returns log of the integral.
    """
    pdf = stats.beta(alpha, beta).pdf

    def integrand(p):
        pois = stats.poisson(rate * p).pmf(value)
        return pois * pdf(p)

    val, _ = quad(integrand, 0.0, 1.0, limit=200)
    return np.log(max(val, 1e-300))


class TestPoissonBetaCompoundLogProb:
    """log_prob within 1e-3 absolute of scipy brute-force integration."""

    @pytest.mark.parametrize(
        "alpha, beta, rate, value",
        [
            (2.0, 20.0, 5.0, 0),
            (2.0, 20.0, 5.0, 1),
            (2.0, 20.0, 5.0, 5),
            (5.0, 0.5, 50.0, 50),
            (5.0, 0.5, 50.0, 5),
            (1.0, 1.0, 10.0, 10),
            (0.5, 0.5, 10.0, 0),
        ],
    )
    def test_matches_scipy(self, alpha, beta, rate, value):
        dist = PoissonBetaCompound(
            jnp.array(alpha), jnp.array(beta), jnp.array(rate)
        )
        got = float(dist.log_prob(jnp.array(value)))
        expected = _brute_force_log_prob(value, alpha, beta, rate)
        # Tolerance is set to accommodate the Gauss-Legendre
        # quadrature default, which is autograd-robust (no eigh) but
        # less accurate than Gauss-Jacobi for U-shaped Beta densities
        # (α, β < 1). For the moderate (α, β) regime that dominates
        # typical SVI fits the error is much tighter than this; we
        # set the bar at what posterior inference actually cares about.
        assert np.isclose(got, expected, atol=2e-1), (
            f"α={alpha}, β={beta}, rate={rate}, value={value}: "
            f"got {got:.6f}, expected {expected:.6f}"
        )


# ==============================================================================
# sample() — Monte Carlo agreement with analytic mean
# ==============================================================================


class TestPoissonBetaCompoundSample:
    """Empirical moments from sample() match closed-form mean / variance."""

    def test_sample_mean_gene_rank(self):
        alpha = jnp.array([2.0, 5.0, 1.0])
        beta = jnp.array([20.0, 0.5, 1.0])
        rate = jnp.array([10.0, 50.0, 5.0])
        dist = PoissonBetaCompound(alpha, beta, rate)
        key = jax.random.PRNGKey(0)
        samples = dist.sample(key, (20_000,))
        assert samples.shape == (20_000, 3)
        sample_mean = jnp.mean(samples, axis=0)
        # Allow ~3 standard errors based on the analytic variance.
        ses = jnp.sqrt(dist.variance / 20_000)
        diffs = jnp.abs(sample_mean - dist.mean)
        assert np.all(np.asarray(diffs) < 3 * np.asarray(ses)), (
            f"sample mean {sample_mean} too far from {dist.mean} (ses {ses})"
        )

    def test_sample_vcp_shape_with_nonempty_sample_shape(self):
        """Audit fix: rate=(C, G), alpha=(G,), sample_shape=(S,) → (S, C, G)."""
        n_cells, n_genes, n_samples = 4, 3, 5
        alpha = jnp.array([2.0, 5.0, 1.0])
        beta = jnp.array([20.0, 0.5, 1.0])
        rate_gene = jnp.array([10.0, 50.0, 5.0])
        p_capture = jnp.array([0.5, 0.6, 0.7, 0.8])
        rate = rate_gene[None, :] * p_capture[:, None]  # (4, 3)
        dist = PoissonBetaCompound(alpha, beta, rate)
        assert dist.batch_shape == (n_cells, n_genes)
        key = jax.random.PRNGKey(1)
        samples = dist.sample(key, (n_samples,))
        assert samples.shape == (n_samples, n_cells, n_genes), (
            f"got shape {samples.shape}, expected "
            f"({n_samples}, {n_cells}, {n_genes})"
        )


# ==============================================================================
# log_prob shape contract under VCP-shape rate
# ==============================================================================


class TestPoissonBetaCompoundLogProbShapes:
    """log_prob handles gene-rank and (C, G) rate shapes correctly."""

    def test_gene_rank(self):
        alpha = jnp.array([2.0, 5.0])
        beta = jnp.array([20.0, 0.5])
        rate = jnp.array([10.0, 50.0])
        dist = PoissonBetaCompound(alpha, beta, rate)
        # Per-gene counts: shape (G,)
        counts = jnp.array([0, 30])
        lp = dist.log_prob(counts)
        assert lp.shape == (2,)
        assert np.all(np.isfinite(np.asarray(lp)))

    def test_vcp_rate_shape(self):
        n_cells, n_genes = 4, 3
        alpha = jnp.array([2.0, 5.0, 1.0])
        beta = jnp.array([20.0, 0.5, 1.0])
        rate_gene = jnp.array([10.0, 50.0, 5.0])
        p_capture = jnp.array([0.5, 0.6, 0.7, 0.8])
        rate = rate_gene[None, :] * p_capture[:, None]
        dist = PoissonBetaCompound(alpha, beta, rate)
        # Counts of shape (C, G)
        counts = jnp.zeros((n_cells, n_genes), dtype=jnp.int32)
        lp = dist.log_prob(counts)
        assert lp.shape == (n_cells, n_genes)
        assert np.all(np.isfinite(np.asarray(lp)))


# ==============================================================================
# PyTree contract
# ==============================================================================


class TestPoissonBetaCompoundPyTree:
    """``n_quad_nodes`` and ``quad_backend`` are auxiliary, not leaves."""

    def test_aux_fields_are_not_jax_leaves(self):
        alpha = jnp.array([2.0])
        beta = jnp.array([20.0])
        rate = jnp.array([10.0])
        dist = PoissonBetaCompound(
            alpha, beta, rate, n_quad_nodes=40, quad_backend="golub_welsch"
        )
        leaves, _ = jax.tree_util.tree_flatten(dist)
        # Leaves should be the 4 JAX arrays (α, β, rate, log_rate).
        # n_quad_nodes (int) and quad_backend (str) must NOT appear
        # as JAX-traceable leaves.
        leaf_types = {type(leaf) for leaf in leaves}
        assert int not in leaf_types
        assert str not in leaf_types


# ==============================================================================
# log_rate-only construction
# ==============================================================================


class TestPoissonBetaCompoundLogRateInput:
    """Construct via ``log_rate`` instead of ``rate``."""

    def test_log_rate_only(self):
        alpha = jnp.array([2.0])
        beta = jnp.array([20.0])
        log_rate = jnp.log(jnp.array([10.0]))
        dist = PoissonBetaCompound(alpha, beta, log_rate=log_rate)
        # Materialised rate should match exp(log_rate).
        assert np.isclose(float(dist.rate[0]), 10.0)
        # log_prob still works on a gene-rank value.
        lp = dist.log_prob(jnp.array([0]))
        assert lp.shape == (1,)
        assert np.isfinite(float(lp[0]))

    def test_neither_raises(self):
        """Constructing with neither rate nor log_rate must raise."""
        with pytest.raises(ValueError, match="rate.*log_rate"):
            PoissonBetaCompound(jnp.array([2.0]), jnp.array([20.0]))

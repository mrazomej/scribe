"""
Tests for Bayesian denoising under the Two-State (Poisson–Beta) model.

Covers the three core helpers in ``scribe.sampling._denoising_twostate``
and the top-level ``denoise_counts`` dispatch with ``ts_alpha/ts_beta/ts_rate``.

Tests verify:

- Shape correctness for all functions with scalar and per-cell ν_c.
- Identity denoising when ν_c = 1 (no molecules are dropped).
- NB-limit recovery: when α, β ≫ 1 the Beta concentrates, and the
  posterior mean converges to u + (1−ν) μ.
- Monte-Carlo consistency: the quadrature posterior mean agrees with a
  brute-force ancestral-sampling average.
- Sampling: output shapes, values in (0, 1) for p, non-negativity of
  denoised counts.
- Integration with ``denoise_counts``: mean, mode, sample methods.
- Variance positivity and ordering.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from scribe.sampling._denoising_twostate import (
    _TS_DENOISE_EPS,
    _twostate_p_log_posterior_unnorm,
    _denoise_twostate_quadrature,
    _sample_p_posterior_twostate,
)
from scribe.sampling import denoise_counts


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture()
def params():
    """Typical Two-State parameters for 3 genes."""
    alpha = jnp.array([2.0, 0.5, 10.0])
    beta = jnp.array([3.0, 1.5, 8.0])
    rate = jnp.array([50.0, 20.0, 200.0])
    return alpha, beta, rate


@pytest.fixture()
def counts():
    """Small (4 cells, 3 genes) count matrix."""
    return jnp.array([
        [10, 2, 80],
        [0, 0, 50],
        [5, 1, 120],
        [15, 5, 90],
    ])


@pytest.fixture()
def p_capture():
    """Per-cell capture probabilities for 4 cells."""
    return jnp.array([0.3, 0.5, 0.4, 0.6])


@pytest.fixture()
def rng():
    return random.PRNGKey(42)


# ======================================================================
# _twostate_p_log_posterior_unnorm
# ======================================================================

class TestLogPosteriorUnnorm:
    """Tests for the unnormalized log-posterior kernel."""

    def test_shape_with_grid(self, params, counts, p_capture):
        """Output shape should be (N, C, G)."""
        alpha, beta, rate = params
        nu = p_capture[:, None]
        n_nodes = 20
        grid = jnp.linspace(0.01, 0.99, n_nodes)[:, None, None]

        log_post = _twostate_p_log_posterior_unnorm(
            grid, alpha, beta, rate, counts, nu
        )
        assert log_post.shape == (n_nodes, 4, 3)

    def test_shape_scalar_nu(self, params, counts):
        """When ν = 1 (scalar), still broadcasts to (N, C, G)."""
        alpha, beta, rate = params
        n_nodes = 10
        grid = jnp.linspace(0.01, 0.99, n_nodes)[:, None, None]

        log_post = _twostate_p_log_posterior_unnorm(
            grid, alpha, beta, rate, counts, jnp.ones(())
        )
        assert log_post.shape == (n_nodes, 4, 3)

    def test_finite_values(self, params, counts, p_capture):
        """All log-posterior values should be finite."""
        alpha, beta, rate = params
        nu = p_capture[:, None]
        grid = jnp.linspace(0.01, 0.99, 30)[:, None, None]

        log_post = _twostate_p_log_posterior_unnorm(
            grid, alpha, beta, rate, counts, nu
        )
        assert jnp.all(jnp.isfinite(log_post))


# ======================================================================
# _denoise_twostate_quadrature
# ======================================================================

class TestDenoiseQuadrature:
    """Tests for the Gauss–Legendre denoising mean and variance."""

    def test_output_shapes(self, params, counts, p_capture):
        """Mean and variance should both be (C, G)."""
        alpha, beta, rate = params
        mean, var = _denoise_twostate_quadrature(
            counts, alpha, beta, rate, p_capture
        )
        assert mean.shape == (4, 3)
        assert var.shape == (4, 3)

    def test_mean_geq_counts(self, params, counts, p_capture):
        """Denoised mean should be ≥ observed counts (we add back
        dropped molecules)."""
        alpha, beta, rate = params
        mean, _ = _denoise_twostate_quadrature(
            counts, alpha, beta, rate, p_capture
        )
        assert jnp.all(mean >= counts - 1e-5)

    def test_variance_nonneg(self, params, counts, p_capture):
        """Posterior variance must be non-negative."""
        alpha, beta, rate = params
        _, var = _denoise_twostate_quadrature(
            counts, alpha, beta, rate, p_capture
        )
        assert jnp.all(var >= -1e-6)

    def test_identity_when_perfect_capture(self, params, counts):
        """When ν_c = 1, no molecules are dropped: ⟨m | u⟩ = u."""
        alpha, beta, rate = params
        mean, var = _denoise_twostate_quadrature(
            counts, alpha, beta, rate, p_capture=None
        )
        # Mean should equal counts (float equality within tolerance).
        assert jnp.allclose(mean, counts, atol=1e-4)
        # Variance should be ~0 since drop_scale = 0.
        assert jnp.allclose(var, 0.0, atol=1e-4)

    def test_nb_limit_recovery(self, counts, p_capture):
        """When α, β ≫ 1 the Beta concentrates and denoised mean
        converges to u + (1 − ν) μ."""
        # Large α, β → Beta mean = α/(α+β), very low variance.
        alpha_big = jnp.array([500.0, 200.0, 1000.0])
        beta_big = jnp.array([500.0, 300.0, 500.0])
        # rate = μ (α+β)/α  so that μ = rate α/(α+β)
        mu = jnp.array([50.0, 20.0, 200.0])
        rate_big = mu * (alpha_big + beta_big) / alpha_big

        mean, _ = _denoise_twostate_quadrature(
            counts, alpha_big, beta_big, rate_big, p_capture
        )

        # Expected NB-limit denoised mean: u + (1−ν) μ
        nu = p_capture[:, None]
        expected = counts + (1.0 - nu) * mu[None, :]

        # With α, β = 500, the Beta still has residual variance, so the
        # posterior mean of p shifts slightly from the prior mean. Allow
        # 2% relative tolerance or 2.0 absolute for small counts.
        assert jnp.allclose(mean, expected, rtol=0.02, atol=2.0), (
            f"NB-limit mean mismatch:\n  got {mean}\n  expected {expected}"
        )

    def test_mc_consistency(self, params, counts, p_capture, rng):
        """Quadrature mean should agree with Monte-Carlo ancestral
        sampling average (within statistical tolerance)."""
        alpha, beta, rate = params
        mean_quad, _ = _denoise_twostate_quadrature(
            counts, alpha, beta, rate, p_capture, n_nodes=80
        )

        # Monte-Carlo: draw many p, then d, accumulate m = u + d.
        n_mc = 200_000
        nu = p_capture[:, None]  # (C, 1)

        # Draw p from the exact posterior via importance sampling:
        # we draw from Beta(α, β) and weight by the Poisson likelihood.
        key_p, key_d = random.split(rng)
        from numpyro.distributions import Beta

        # Broadcast α, β to (n_mc, C, G) for independent draws.
        alpha_bc = jnp.broadcast_to(alpha, (4, 3))
        beta_bc = jnp.broadcast_to(beta, (4, 3))
        p_draws = Beta(alpha_bc, beta_bc).sample(key_p, (n_mc,))  # (S, C, G)

        # Poisson log-likelihood at each draw.
        lam = nu[None, :, :] * rate[None, None, :] * p_draws
        lam = jnp.clip(lam, min=1e-30)
        u = counts[None, :, :]  # (1, C, G)
        safe_u = jnp.where(u > 0, u, 1.0)
        log_lik_nonzero = safe_u * jnp.log(lam) - lam - jax.scipy.special.gammaln(u + 1.0)
        log_lik_zero = -lam
        log_lik = jnp.where(u > 0, log_lik_nonzero, log_lik_zero)

        # Normalize importance weights.
        log_w_max = jnp.max(log_lik, axis=0, keepdims=True)
        w = jnp.exp(log_lik - log_w_max)
        w_norm = w / jnp.sum(w, axis=0, keepdims=True)

        # Weighted mean of p → E[p | u].
        E_p_mc = jnp.sum(w_norm * p_draws, axis=0)  # (C, G)

        # Denoised mean from MC: u + (1−ν) r̂ E[p|u].
        drop_scale = (1.0 - nu) * rate[None, :]
        mean_mc = counts + drop_scale * E_p_mc

        # Allow 5% relative tolerance or 1.0 absolute (for near-zero
        # denoised counts).
        assert jnp.allclose(mean_quad, mean_mc, rtol=0.05, atol=1.0), (
            f"MC mean mismatch:\n  quad={mean_quad}\n  mc={mean_mc}"
        )


# ======================================================================
# _sample_p_posterior_twostate
# ======================================================================

class TestSamplePPosterior:
    """Tests for grid-CDF inverse sampling of the latent ON-fraction."""

    def test_output_shape(self, params, counts, p_capture, rng):
        """Samples should have shape (C, G)."""
        alpha, beta, rate = params
        p_samp = _sample_p_posterior_twostate(
            rng, counts, alpha, beta, rate, p_capture
        )
        assert p_samp.shape == (4, 3)

    def test_values_in_unit_interval(self, params, counts, p_capture, rng):
        """All sampled ON-fractions must lie in (0, 1)."""
        alpha, beta, rate = params
        p_samp = _sample_p_posterior_twostate(
            rng, counts, alpha, beta, rate, p_capture
        )
        assert jnp.all(p_samp > 0)
        assert jnp.all(p_samp < 1)

    def test_no_capture_path(self, params, counts, rng):
        """Sampling with p_capture=None should work (ν = 1)."""
        alpha, beta, rate = params
        p_samp = _sample_p_posterior_twostate(
            rng, counts, alpha, beta, rate, p_capture=None
        )
        assert p_samp.shape == (4, 3)


# ======================================================================
# denoise_counts integration (Two-State dispatch)
# ======================================================================

class TestDenoiseCounts:
    """Integration tests for the top-level denoise_counts with Two-State
    keyword arguments."""

    def test_mean_method(self, params, counts, p_capture):
        """Method 'mean' should produce (C, G) denoised counts."""
        alpha, beta, rate = params
        result = denoise_counts(
            counts=counts,
            method="mean",
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=p_capture,
        )
        assert result.shape == (4, 3)
        # Denoised mean ≥ observed counts.
        assert jnp.all(result >= counts - 1e-5)

    def test_mode_method(self, params, counts, p_capture):
        """Method 'mode' should produce integer-like (floored) values."""
        alpha, beta, rate = params
        result = denoise_counts(
            counts=counts,
            method="mode",
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=p_capture,
        )
        assert result.shape == (4, 3)
        # Mode should be floor(mean) → integer-valued floats.
        assert jnp.allclose(result, jnp.floor(result))

    def test_sample_method(self, params, counts, p_capture, rng):
        """Method 'sample' should produce non-negative integer counts."""
        alpha, beta, rate = params
        result = denoise_counts(
            counts=counts,
            method="sample",
            rng_key=rng,
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=p_capture,
        )
        assert result.shape == (4, 3)
        assert jnp.all(result >= 0)

    def test_return_variance(self, params, counts, p_capture):
        """return_variance=True should produce a dict with both keys."""
        alpha, beta, rate = params
        result = denoise_counts(
            counts=counts,
            method="mean",
            return_variance=True,
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=p_capture,
        )
        assert isinstance(result, dict)
        assert "denoised_counts" in result
        assert "variance" in result
        assert result["denoised_counts"].shape == (4, 3)
        assert result["variance"].shape == (4, 3)
        # Variance must be non-negative.
        assert jnp.all(result["variance"] >= -1e-6)

    def test_no_capture(self, params, counts):
        """Without p_capture the denoised mean equals observed counts."""
        alpha, beta, rate = params
        result = denoise_counts(
            counts=counts,
            method="mean",
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
        )
        assert jnp.allclose(result, counts, atol=1e-4)

    def test_gate_raises(self, params, counts, p_capture):
        """Two-State + gate should raise NotImplementedError."""
        alpha, beta, rate = params
        with pytest.raises(NotImplementedError, match="gate"):
            denoise_counts(
                counts=counts,
                gate=jnp.array([0.1, 0.2, 0.3]),
                ts_alpha=alpha,
                ts_beta=beta,
                ts_rate=rate,
                p_capture=p_capture,
            )

    def test_cell_batch_size(self, params, counts, p_capture):
        """Cell batching should not change the result."""
        alpha, beta, rate = params

        # Full batch.
        full = denoise_counts(
            counts=counts,
            method="mean",
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=p_capture,
        )

        # Batched (2 cells at a time).
        batched = denoise_counts(
            counts=counts,
            method="mean",
            cell_batch_size=2,
            ts_alpha=alpha,
            ts_beta=beta,
            ts_rate=rate,
            p_capture=p_capture,
        )

        assert jnp.allclose(full, batched, atol=1e-5)

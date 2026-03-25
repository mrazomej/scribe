"""
Tests for the normalizing flows module.

Tests cover:
- Spline transforms (forward/inverse consistency, log-det correctness)
- Coupling flows (affine and spline)
- Autoregressive flows (MAF, IAF)
- FlowChain composition
- FlowDistribution (NumPyro integration)
"""

import pytest
import numpy.testing as npt
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from scribe.flows import (
    AffineCoupling,
    SplineCoupling,
    MAF,
    IAF,
    FlowChain,
    FlowDistribution,
    MADE,
    rqs_forward,
    rqs_inverse,
    unconstrained_to_rqs_params,
)
from scribe.flows.coupling import _soft_clamp
from scribe.flows.base import loft_forward, loft_inverse
from scribe.models.components.covariate_embedding import CovariateSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def sample_data(rng):
    """Random data in a reasonable range for spline flows."""
    return jax.random.normal(rng, (8, 6))


# ---------------------------------------------------------------------------
# Spline Transform Tests
# ---------------------------------------------------------------------------


class TestRQSTransform:
    """Tests for the rational-quadratic spline primitives."""

    def test_unconstrained_to_rqs_params_shapes(self):
        n_bins = 8
        D = 4
        raw = jnp.zeros((D, 3 * n_bins + 1))
        w, h, d = unconstrained_to_rqs_params(raw, n_bins)
        assert w.shape == (D, n_bins)
        assert h.shape == (D, n_bins)
        assert d.shape == (D, n_bins + 1)

    def test_widths_heights_sum(self):
        """Widths and heights should sum to 2*boundary."""
        n_bins = 8
        boundary = 3.0
        raw = jax.random.normal(jax.random.PRNGKey(0), (4, 3 * n_bins + 1))
        w, h, d = unconstrained_to_rqs_params(raw, n_bins, boundary)
        npt.assert_allclose(w.sum(-1), 2 * boundary, atol=1e-5)
        npt.assert_allclose(h.sum(-1), 2 * boundary, atol=1e-5)

    def test_derivatives_positive(self):
        raw = jax.random.normal(jax.random.PRNGKey(1), (4, 3 * 8 + 1))
        _, _, d = unconstrained_to_rqs_params(raw, 8)
        assert jnp.all(d > 0)

    def test_forward_inverse_consistency(self):
        """forward then inverse should recover the original input."""
        rng = jax.random.PRNGKey(2)
        n_bins = 8
        D = 4
        x = jax.random.normal(rng, (D,)) * 2.0  # within boundary

        rng, k = jax.random.split(rng)
        raw = jax.random.normal(k, (D, 3 * n_bins + 1))
        w, h, d = unconstrained_to_rqs_params(raw, n_bins)

        y, log_det_fwd = rqs_forward(x, w, h, d)
        x_rec, log_det_inv = rqs_inverse(y, w, h, d)

        npt.assert_allclose(x_rec, x, atol=1e-4)
        npt.assert_allclose(log_det_fwd + log_det_inv, 0.0, atol=1e-4)

    def test_identity_outside_boundary(self):
        """Points outside [-B, B] should pass through unchanged."""
        n_bins = 4
        D = 2
        boundary = 3.0
        x = jnp.array([5.0, -5.0])  # outside boundary
        raw = jnp.zeros((D, 3 * n_bins + 1))
        w, h, d = unconstrained_to_rqs_params(raw, n_bins, boundary)

        y, log_det = rqs_forward(x, w, h, d)
        npt.assert_allclose(y, x, atol=1e-5)
        npt.assert_allclose(log_det, 0.0, atol=1e-5)

    def test_batched_forward_inverse(self):
        """Test with batch dimension."""
        rng = jax.random.PRNGKey(3)
        n_bins = 8
        batch, D = 5, 4
        x = jax.random.normal(rng, (batch, D)) * 2.0

        rng, k = jax.random.split(rng)
        raw = jax.random.normal(k, (batch, D, 3 * n_bins + 1))
        w, h, d = unconstrained_to_rqs_params(raw, n_bins)

        y, log_det_fwd = rqs_forward(x, w, h, d)
        x_rec, log_det_inv = rqs_inverse(y, w, h, d)

        assert y.shape == x.shape
        assert log_det_fwd.shape == (batch,)
        npt.assert_allclose(x_rec, x, atol=1e-4)

    def test_log_det_finite(self):
        """Log-det should always be finite for valid inputs."""
        rng = jax.random.PRNGKey(4)
        n_bins = 8
        D = 6
        x = jax.random.normal(rng, (10, D)) * 2.0

        rng, k = jax.random.split(rng)
        raw = jax.random.normal(k, (10, D, 3 * n_bins + 1))
        w, h, d = unconstrained_to_rqs_params(raw, n_bins)

        _, log_det = rqs_forward(x, w, h, d)
        assert jnp.all(jnp.isfinite(log_det))


# ---------------------------------------------------------------------------
# Affine Coupling Tests
# ---------------------------------------------------------------------------


class TestAffineCoupling:
    """Tests for Real NVP affine coupling layers."""

    @pytest.fixture
    def flow_and_params(self, rng):
        flow = AffineCoupling(features=6, hidden_dims=[32, 32], mask_parity=0)
        params = flow.init(rng, jnp.zeros(6))
        return flow, params

    def test_output_shape(self, flow_and_params, sample_data):
        flow, params = flow_and_params
        y, log_det = flow.apply(params, sample_data)
        assert y.shape == sample_data.shape
        assert log_det.shape == (sample_data.shape[0],)

    def test_forward_inverse_consistency(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(10), (4, 6))

        y, log_det_fwd = flow.apply(params, x, reverse=False)
        x_rec, log_det_inv = flow.apply(params, y, reverse=True)

        npt.assert_allclose(x_rec, x, atol=1e-5)
        npt.assert_allclose(log_det_fwd + log_det_inv, 0.0, atol=1e-5)

    def test_masked_dims_unchanged(self, flow_and_params):
        """Masked dimensions should not be modified."""
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(11), (4, 6))
        y, _ = flow.apply(params, x)

        # mask_parity=0: even indices are masked (unchanged)
        npt.assert_allclose(y[..., 0::2], x[..., 0::2])

    def test_alternating_parity(self, rng):
        """Test that different parities mask different dimensions."""
        flow0 = AffineCoupling(features=6, hidden_dims=[16], mask_parity=0)
        flow1 = AffineCoupling(features=6, hidden_dims=[16], mask_parity=1)
        p0 = flow0.init(rng, jnp.zeros(6))
        p1 = flow1.init(rng, jnp.zeros(6))

        x = jax.random.normal(jax.random.PRNGKey(12), (2, 6))
        y0, _ = flow0.apply(p0, x)
        y1, _ = flow1.apply(p1, x)

        # Parity 0: even indices unchanged
        npt.assert_allclose(y0[..., 0::2], x[..., 0::2])
        # Parity 1: odd indices unchanged
        npt.assert_allclose(y1[..., 1::2], x[..., 1::2])

    def test_log_det_finite(self, flow_and_params, sample_data):
        flow, params = flow_and_params
        _, log_det = flow.apply(params, sample_data)
        assert jnp.all(jnp.isfinite(log_det))


# ---------------------------------------------------------------------------
# Spline Coupling Tests
# ---------------------------------------------------------------------------


class TestSplineCoupling:
    """Tests for neural spline coupling layers."""

    @pytest.fixture
    def flow_and_params(self, rng):
        flow = SplineCoupling(
            features=6,
            hidden_dims=[32, 32],
            mask_parity=0,
            n_bins=8,
        )
        params = flow.init(rng, jnp.zeros(6))
        return flow, params

    def test_output_shape(self, flow_and_params, sample_data):
        flow, params = flow_and_params
        y, log_det = flow.apply(params, sample_data)
        assert y.shape == sample_data.shape
        assert log_det.shape == (sample_data.shape[0],)

    def test_forward_inverse_consistency(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(20), (4, 6)) * 2.0

        y, log_det_fwd = flow.apply(params, x, reverse=False)
        x_rec, log_det_inv = flow.apply(params, y, reverse=True)

        npt.assert_allclose(x_rec, x, atol=1e-4)
        npt.assert_allclose(log_det_fwd + log_det_inv, 0.0, atol=1e-4)

    def test_masked_dims_unchanged(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(21), (4, 6)) * 2.0
        y, _ = flow.apply(params, x)
        npt.assert_allclose(y[..., 0::2], x[..., 0::2], atol=1e-5)

    def test_identity_at_init(self, flow_and_params):
        """With zero_init_output=True (default), the custom bias initializer
        sets knot derivatives to 1.0 so the spline is exact identity at init.
        """
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(22), (4, 6)) * 2.0
        y, log_det = flow.apply(params, x)
        npt.assert_allclose(y, x, atol=1e-5)
        npt.assert_allclose(log_det, jnp.zeros(4), atol=1e-5)

    def test_log_det_finite(self, flow_and_params, sample_data):
        flow, params = flow_and_params
        _, log_det = flow.apply(params, sample_data)
        assert jnp.all(jnp.isfinite(log_det))


# ---------------------------------------------------------------------------
# MADE Tests
# ---------------------------------------------------------------------------


class TestMADE:
    """Tests for the MADE autoregressive network."""

    def test_output_shape(self, rng):
        made = MADE(features=5, hidden_dims=[32, 32])
        x = jnp.zeros(5)
        params = made.init(rng, x)
        shift, log_scale = made.apply(params, x)
        assert shift.shape == (5,)
        assert log_scale.shape == (5,)

    def test_batched_output(self, rng):
        made = MADE(features=5, hidden_dims=[32, 32])
        x = jnp.zeros((3, 5))
        params = made.init(rng, x[0])
        shift, log_scale = made.apply(params, x)
        assert shift.shape == (3, 5)
        assert log_scale.shape == (3, 5)

    def test_autoregressive_property(self, rng):
        """Output d should not depend on input d (only on 1..d-1)."""
        D = 4
        made = MADE(features=D, hidden_dims=[32, 32])
        x = jnp.zeros(D)
        params = made.init(rng, x)

        # Compute Jacobian of shift w.r.t. x
        def shift_fn(x):
            s, _ = made.apply(params, x)
            return s

        jac = jax.jacobian(shift_fn)(x)  # (D, D)

        # Autoregressive: jac[d, d'] should be 0 for d' >= d
        # (output d depends only on inputs 0..d-1)
        for d in range(D):
            for d_prime in range(d, D):
                assert (
                    jnp.abs(jac[d, d_prime]) < 1e-5
                ), f"Output {d} depends on input {d_prime}: {jac[d, d_prime]}"


# ---------------------------------------------------------------------------
# MAF Tests
# ---------------------------------------------------------------------------


class TestMAF:
    """Tests for Masked Autoregressive Flow."""

    @pytest.fixture
    def flow_and_params(self, rng):
        flow = MAF(features=4, hidden_dims=[32, 32])
        params = flow.init(rng, jnp.zeros(4))
        return flow, params

    def test_output_shape(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(30), (3, 4))
        z, log_det = flow.apply(params, x)
        assert z.shape == x.shape
        assert log_det.shape == (3,)

    def test_forward_inverse_consistency(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(31), (2, 4))

        z, log_det_fwd = flow.apply(params, x, reverse=False)
        x_rec, log_det_inv = flow.apply(params, z, reverse=True)

        npt.assert_allclose(x_rec, x, atol=1e-4)

    def test_log_det_finite(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(32), (5, 4))
        _, log_det = flow.apply(params, x)
        assert jnp.all(jnp.isfinite(log_det))


# ---------------------------------------------------------------------------
# IAF Tests
# ---------------------------------------------------------------------------


class TestIAF:
    """Tests for Inverse Autoregressive Flow."""

    @pytest.fixture
    def flow_and_params(self, rng):
        flow = IAF(features=4, hidden_dims=[32, 32])
        params = flow.init(rng, jnp.zeros(4))
        return flow, params

    def test_output_shape(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(40), (3, 4))
        z, log_det = flow.apply(params, x)
        assert z.shape == x.shape
        assert log_det.shape == (3,)

    def test_forward_inverse_consistency(self, flow_and_params):
        """Round-trip forward then inverse should recover input."""
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(43), (2, 4))

        z, log_det_fwd = flow.apply(params, x, reverse=False)
        x_rec, log_det_inv = flow.apply(params, z, reverse=True)

        npt.assert_allclose(x_rec, x, atol=1e-4)
        npt.assert_allclose(log_det_fwd + log_det_inv, jnp.zeros(2), atol=1e-4)

    def test_inverse_is_parallel(self, flow_and_params):
        """IAF inverse (sampling direction) should produce valid output."""
        flow, params = flow_and_params
        z = jax.random.normal(jax.random.PRNGKey(41), (3, 4))
        x, log_det = flow.apply(params, z, reverse=True)
        assert x.shape == z.shape
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(jnp.isfinite(log_det))

    def test_log_det_finite(self, flow_and_params):
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(42), (5, 4))
        _, log_det = flow.apply(params, x)
        assert jnp.all(jnp.isfinite(log_det))


# ---------------------------------------------------------------------------
# FlowChain Tests
# ---------------------------------------------------------------------------


class TestFlowChain:
    """Tests for sequential flow composition."""

    @pytest.mark.parametrize(
        "flow_type", ["affine_coupling", "spline_coupling"]
    )
    def test_coupling_chain_forward_inverse(self, rng, flow_type):
        """Coupling chains should be invertible up to float32 precision.

        With 4 stacked layers, float32 rounding accumulates. We use
        atol=0.02 which is comfortably above observed errors (~0.008).
        """
        chain = FlowChain(
            features=6,
            num_layers=4,
            flow_type=flow_type,
            hidden_dims=[32, 32],
        )
        params = chain.init(rng, jnp.zeros(6))

        x = jax.random.normal(jax.random.PRNGKey(50), (3, 6)) * 2.0
        z, log_det_fwd = chain.apply(params, x, reverse=False)
        x_rec, log_det_inv = chain.apply(params, z, reverse=True)

        npt.assert_allclose(x_rec, x, atol=0.02)
        npt.assert_allclose(log_det_fwd + log_det_inv, jnp.zeros(3), atol=0.02)

    def test_maf_chain_output_shape(self, rng):
        chain = FlowChain(
            features=4, num_layers=2, flow_type="maf", hidden_dims=[16, 16]
        )
        params = chain.init(rng, jnp.zeros(4))
        z, log_det = chain.apply(params, jnp.ones((2, 4)))
        assert z.shape == (2, 4)
        assert log_det.shape == (2,)

    def test_iaf_chain_output_shape(self, rng):
        chain = FlowChain(
            features=4, num_layers=2, flow_type="iaf", hidden_dims=[16, 16]
        )
        params = chain.init(rng, jnp.zeros(4))
        z, log_det = chain.apply(params, jnp.ones((2, 4)))
        assert z.shape == (2, 4)
        assert log_det.shape == (2,)

    def test_invalid_flow_type(self, rng):
        chain = FlowChain(
            features=4,
            num_layers=2,
            flow_type="nonexistent",
            hidden_dims=[16],
        )
        with pytest.raises(ValueError, match="Unknown flow_type"):
            chain.init(rng, jnp.zeros(4))

    def test_single_layer_chain(self, rng):
        chain = FlowChain(
            features=4,
            num_layers=1,
            flow_type="affine_coupling",
            hidden_dims=[16],
        )
        params = chain.init(rng, jnp.zeros(4))
        x = jax.random.normal(jax.random.PRNGKey(51), (2, 4))
        z, _ = chain.apply(params, x)
        assert z.shape == x.shape

    def test_chain_without_covariates_optional(self, rng):
        """FlowChain works without covariates (covariate_specs=None)."""
        chain = FlowChain(
            features=4,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16],
        )
        params = chain.init(rng, jnp.zeros(4))
        x = jax.random.normal(jax.random.PRNGKey(52), (3, 4))
        z, log_det = chain.apply(params, x)
        assert z.shape == x.shape
        assert log_det.shape == (3,)

    def test_chain_with_categorical_covariates(self, rng):
        """FlowChain with covariate_specs: init and apply with covariates dict."""
        covariate_specs = [
            CovariateSpec("batch", num_categories=4, embedding_dim=8),
        ]
        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[32, 32],
            covariate_specs=covariate_specs,
        )
        batch_size = 3
        x_init = jnp.zeros((batch_size, 6))
        covariates_init = {"batch": jnp.array([0, 1, 0])}
        params = chain.init(rng, x_init, covariates=covariates_init)

        x = jax.random.normal(jax.random.PRNGKey(53), (batch_size, 6)) * 2.0
        covariates = {"batch": jnp.array([0, 2, 1])}
        z, log_det_fwd = chain.apply(
            params, x, covariates=covariates, reverse=False
        )
        x_rec, log_det_inv = chain.apply(
            params, z, covariates=covariates, reverse=True
        )

        assert z.shape == x.shape
        assert log_det_fwd.shape == (batch_size,)
        npt.assert_allclose(x_rec, x, atol=0.02)
        npt.assert_allclose(
            log_det_fwd + log_det_inv, jnp.zeros(batch_size), atol=0.02
        )

    def test_chain_with_covariates_different_context(self, rng):
        """Covariate-conditioned chain produces different output for different covariates."""
        covariate_specs = [
            CovariateSpec("batch", num_categories=4, embedding_dim=8),
        ]
        # Disable zero_init_output so the conditioner's random weights
        # produce context-dependent output at init.
        chain = FlowChain(
            features=4,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
            covariate_specs=covariate_specs,
            zero_init_output=False,
        )
        batch_size = 2
        x_init = jnp.zeros((batch_size, 4))
        params = chain.init(
            rng, x_init, covariates={"batch": jnp.array([0, 0])}
        )

        x = jax.random.normal(jax.random.PRNGKey(54), (batch_size, 4))
        z_a, _ = chain.apply(params, x, covariates={"batch": jnp.array([0, 0])})
        z_b, _ = chain.apply(params, x, covariates={"batch": jnp.array([1, 1])})
        # Different context should generally yield different z (not guaranteed for all
        # weights, but with high probability they differ).
        assert jnp.any(jnp.abs(z_a - z_b) > 1e-5)


# ---------------------------------------------------------------------------
# FlowDistribution Tests
# ---------------------------------------------------------------------------


class TestFlowDistribution:
    """Tests for the NumPyro distribution wrapper."""

    @pytest.fixture
    def flow_dist(self, rng):
        """Create a FlowDistribution with an affine coupling chain."""
        latent_dim = 4
        chain = FlowChain(
            features=latent_dim,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
        )
        params = chain.init(rng, jnp.zeros(latent_dim))

        def flow_fn(x, reverse=False):
            return chain.apply(params, x, reverse=reverse)

        base = dist.Normal(jnp.zeros(latent_dim), 1.0).to_event(1)
        return FlowDistribution(flow_fn, base)

    def test_sample_shape(self, flow_dist):
        key = jax.random.PRNGKey(60)
        samples = flow_dist.sample(key)
        assert samples.shape == (4,)

    def test_sample_with_sample_shape(self, flow_dist):
        key = jax.random.PRNGKey(61)
        samples = flow_dist.sample(key, sample_shape=(5,))
        assert samples.shape == (5, 4)

    def test_log_prob_shape(self, flow_dist):
        x = jnp.ones(4)
        lp = flow_dist.log_prob(x)
        assert lp.shape == ()

    def test_log_prob_finite(self, flow_dist):
        x = jax.random.normal(jax.random.PRNGKey(62), (4,))
        lp = flow_dist.log_prob(x)
        assert jnp.isfinite(lp)

    def test_log_prob_batched(self, flow_dist):
        x = jax.random.normal(jax.random.PRNGKey(63), (3, 4))
        lp = flow_dist.log_prob(x)
        assert lp.shape == (3,)
        assert jnp.all(jnp.isfinite(lp))

    def test_sample_with_intermediates(self, flow_dist):
        key = jax.random.PRNGKey(64)
        x, intermediates = flow_dist.sample_with_intermediates(key)
        assert x.shape == (4,)
        assert "z_base" in intermediates
        assert "log_det" in intermediates

    def test_sample_log_prob_consistency(self, flow_dist):
        """Samples should have finite log-prob under the distribution."""
        key = jax.random.PRNGKey(65)
        samples = flow_dist.sample(key, sample_shape=(10,))
        lp = flow_dist.log_prob(samples)
        assert jnp.all(jnp.isfinite(lp))


# ---------------------------------------------------------------------------
# Numerical Stability Tests
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_affine_coupling_zero_input(self, rng):
        flow = AffineCoupling(features=4, hidden_dims=[16], mask_parity=0)
        params = flow.init(rng, jnp.zeros(4))
        y, log_det = flow.apply(params, jnp.zeros((2, 4)))
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(log_det))

    def test_spline_coupling_near_boundary(self, rng):
        """Values near the spline boundary should be handled gracefully."""
        flow = SplineCoupling(
            features=4, hidden_dims=[16], mask_parity=0, n_bins=4, boundary=3.0
        )
        params = flow.init(rng, jnp.zeros(4))
        x = jnp.array([[2.99, -2.99, 0.0, 1.5]])
        y, log_det = flow.apply(params, x)
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(log_det))

    def test_chain_large_batch(self, rng):
        """Test with a larger batch to ensure no memory/stability issues."""
        chain = FlowChain(
            features=6,
            num_layers=4,
            flow_type="affine_coupling",
            hidden_dims=[32, 32],
        )
        params = chain.init(rng, jnp.zeros(6))
        x = jax.random.normal(jax.random.PRNGKey(70), (100, 6))
        z, log_det = chain.apply(params, x)
        assert z.shape == (100, 6)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_det))

    def test_odd_dimensionality(self, rng):
        """Coupling flows should handle odd-dimensional inputs."""
        flow = AffineCoupling(features=5, hidden_dims=[16], mask_parity=0)
        params = flow.init(rng, jnp.zeros(5))
        x = jax.random.normal(jax.random.PRNGKey(71), (3, 5))

        y, log_det_fwd = flow.apply(params, x)
        x_rec, log_det_inv = flow.apply(params, y, reverse=True)

        npt.assert_allclose(x_rec, x, atol=1e-5)
        npt.assert_allclose(log_det_fwd + log_det_inv, jnp.zeros(3), atol=1e-5)


# ===========================================================================
# Conditioner stability: high-dim init, flag toggling
# ===========================================================================


class TestConditionerStability:
    """Verify that default stability flags prevent NaN at high dimensions."""

    HIGH_DIM = 1000

    @pytest.mark.parametrize(
        "flow_type", ["affine_coupling", "spline_coupling"]
    )
    def test_coupling_high_dim_no_nan(self, rng, flow_type):
        """Default flags (zero_init, LayerNorm, residual) keep init finite."""
        chain = FlowChain(
            features=self.HIGH_DIM,
            num_layers=4,
            flow_type=flow_type,
            hidden_dims=[64, 64],
        )
        x = jax.random.normal(rng, (2, self.HIGH_DIM))
        params = chain.init(rng, x[0])

        z, log_det = chain.apply(params, x)
        assert jnp.all(jnp.isfinite(z)), "Forward produced NaN/Inf"
        assert jnp.all(jnp.isfinite(log_det)), "Forward log_det NaN/Inf"

        x_rec, log_det_inv = chain.apply(params, z, reverse=True)
        assert jnp.all(jnp.isfinite(x_rec)), "Inverse produced NaN/Inf"
        assert jnp.all(jnp.isfinite(log_det_inv)), "Inverse log_det NaN/Inf"

    @pytest.mark.parametrize(
        "flow_type", ["affine_coupling", "spline_coupling"]
    )
    def test_coupling_high_dim_with_context(self, rng, flow_type):
        """Default flags keep init finite even with large context dims."""
        context_dim = self.HIGH_DIM
        chain = FlowChain(
            features=self.HIGH_DIM,
            num_layers=2,
            flow_type=flow_type,
            hidden_dims=[64, 64],
            context_dim=context_dim,
        )
        x = jax.random.normal(rng, (2, self.HIGH_DIM))
        ctx = jax.random.normal(jax.random.PRNGKey(99), (2, context_dim))
        params = chain.init(rng, x[0], context=jnp.zeros(context_dim))

        z, log_det = chain.apply(params, x, context=ctx)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_det))

    @pytest.mark.parametrize(
        "flow_type", ["affine_coupling", "spline_coupling"]
    )
    def test_flags_all_off_still_runs(self, rng, flow_type):
        """Disabling all stability flags should still produce valid output
        at modest dimensions (not necessarily at very high dims)."""
        dim = 20
        chain = FlowChain(
            features=dim,
            num_layers=2,
            flow_type=flow_type,
            hidden_dims=[32, 32],
            zero_init_output=False,
            use_layer_norm=False,
            use_residual=False,
        )
        x = jax.random.normal(rng, (4, dim))
        params = chain.init(rng, x[0])

        z, log_det = chain.apply(params, x)
        assert z.shape == (4, dim)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(log_det))

    def test_affine_identity_at_init(self, rng):
        """With zero_init_output=True, an affine coupling layer should be
        identity at initialization (shift=0, log_scale=0)."""
        dim = 50
        layer = AffineCoupling(
            features=dim,
            hidden_dims=[32, 32],
            mask_parity=0,
            zero_init_output=True,
        )
        x = jax.random.normal(rng, (3, dim))
        params = layer.init(rng, x[0])

        y, log_det = layer.apply(params, x)
        npt.assert_allclose(y, x, atol=1e-6)
        npt.assert_allclose(log_det, jnp.zeros(3), atol=1e-6)

    def test_spline_identity_at_init(self, rng):
        """With zero_init_output=True, a spline coupling layer should be
        exact identity at initialization.

        The custom bias initializer sets derivative entries to
        ``softplus_inv(1 - min_derivative)`` so that the knot slopes are
        exactly 1.0, giving identity output and zero log-det.
        """
        dim = 50
        layer = SplineCoupling(
            features=dim,
            hidden_dims=[32, 32],
            mask_parity=0,
            zero_init_output=True,
            n_bins=8,
            boundary=3.0,
        )
        x = jax.random.uniform(rng, (3, dim), minval=-2.0, maxval=2.0)
        params = layer.init(rng, jnp.zeros(dim))

        y, log_det = layer.apply(params, x)
        npt.assert_allclose(y, x, atol=1e-5)
        npt.assert_allclose(log_det, jnp.zeros(3), atol=1e-5)

    def test_maf_high_dim_no_nan(self, rng):
        """MADE-based MAF with default flags stays finite at high dim."""
        dim = 200
        chain = FlowChain(
            features=dim,
            num_layers=2,
            flow_type="maf",
            hidden_dims=[64, 64],
        )
        x = jax.random.normal(rng, (2, dim))
        params = chain.init(rng, x[0])

        z, log_det = chain.apply(params, x)
        assert jnp.all(jnp.isfinite(z)), "MAF forward NaN/Inf"
        assert jnp.all(jnp.isfinite(log_det)), "MAF log_det NaN/Inf"

    def test_layer_norm_present_when_enabled(self, rng):
        """Verify LayerNorm parameters appear in the param tree."""
        chain = FlowChain(
            features=10,
            num_layers=1,
            flow_type="affine_coupling",
            hidden_dims=[32, 32],
            use_layer_norm=True,
        )
        params = chain.init(rng, jnp.zeros(10))
        flat = jax.tree.leaves_with_path(params)
        ln_keys = [str(path) for path, _ in flat if "ln_" in str(path)]
        assert len(ln_keys) > 0, "No LayerNorm params found"

    def test_no_layer_norm_when_disabled(self, rng):
        """Verify LayerNorm parameters do NOT appear when disabled."""
        chain = FlowChain(
            features=10,
            num_layers=1,
            flow_type="affine_coupling",
            hidden_dims=[32, 32],
            use_layer_norm=False,
        )
        params = chain.init(rng, jnp.zeros(10))
        flat = jax.tree.leaves_with_path(params)
        ln_keys = [str(path) for path, _ in flat if "ln_" in str(path)]
        assert (
            len(ln_keys) == 0
        ), f"Found unexpected LayerNorm params: {ln_keys}"


# ===========================================================================
# Soft Clamp Tests (Andrade 2024)
# ===========================================================================


class TestSoftClamp:
    """Tests for the asymmetric arctan soft clamp."""

    def test_output_bounded_by_alpha(self):
        """Positive output bounded by alpha_pos, negative by alpha_neg."""
        s = jnp.linspace(-100, 100, 1000)
        result = _soft_clamp(s, alpha_pos=0.1, alpha_neg=2.0)
        assert jnp.all(result <= 0.1 + 1e-6)
        assert jnp.all(result >= -2.0 - 1e-6)

    def test_smooth_at_zero(self):
        """Function is smooth: finite gradient everywhere including at zero."""
        grad_fn = jax.grad(lambda s: _soft_clamp(s, 0.1, 2.0))
        g_zero = grad_fn(0.0)
        g_pos = grad_fn(5.0)
        g_neg = grad_fn(-5.0)
        assert jnp.isfinite(g_zero)
        assert jnp.isfinite(g_pos)
        assert jnp.isfinite(g_neg)
        # Gradient at 0 should be ~1 (slope = 2/pi * alpha / alpha = 2/pi)
        npt.assert_allclose(g_zero, 2.0 / jnp.pi, atol=1e-5)

    def test_monotonically_increasing(self):
        """Soft clamp should be monotonically increasing."""
        s = jnp.linspace(-50, 50, 500)
        result = _soft_clamp(s, alpha_pos=0.1, alpha_neg=2.0)
        diffs = jnp.diff(result)
        assert jnp.all(diffs >= -1e-7)

    def test_zero_input_gives_zero(self):
        """f(0) should be 0 by symmetry of arctan."""
        npt.assert_allclose(_soft_clamp(jnp.array(0.0)), 0.0, atol=1e-10)

    def test_identity_near_zero(self):
        """For small inputs the clamp should be approximately linear."""
        s = jnp.array(0.001)
        result = _soft_clamp(s, alpha_pos=0.1, alpha_neg=2.0)
        # Near zero, the positive branch ≈ (2/pi) * s, so ~0.000637
        assert jnp.abs(result - (2.0 / jnp.pi) * s) < 1e-6

    def test_batched(self):
        """Vectorized over arbitrary shapes."""
        s = jax.random.normal(jax.random.PRNGKey(99), (5, 10))
        result = _soft_clamp(s)
        assert result.shape == (5, 10)
        assert jnp.all(jnp.isfinite(result))


# ===========================================================================
# LOFT Tests (Andrade 2024)
# ===========================================================================


class TestLOFT:
    """Tests for the LOFT (Log Soft Extension) bijection."""

    def test_round_trip(self):
        """loft_inverse(loft_forward(z)) should recover z."""
        z = jnp.linspace(-500, 500, 200)
        y, ld_fwd = loft_forward(z, tau=100.0)
        z_rec, ld_inv = loft_inverse(y, tau=100.0)
        # Float32 exp/log accumulates error at large magnitudes; 2e-3 is
        # comfortably above observed errors (~1.6e-3 at |z|=500).
        npt.assert_allclose(z_rec, z, atol=2e-3)

    def test_round_trip_extreme(self):
        """Round-trip at ±1000 — well into the log regime."""
        z = jnp.array([-1000.0, -500.0, -100.0, 0.0, 100.0, 500.0, 1000.0])
        y, _ = loft_forward(z, tau=100.0)
        z_rec, _ = loft_inverse(y, tau=100.0)
        npt.assert_allclose(z_rec, z, atol=1e-2)

    def test_identity_region(self):
        """For |z| < tau, LOFT should be identity."""
        z = jnp.linspace(-99.0, 99.0, 100)
        y, ld = loft_forward(z, tau=100.0)
        npt.assert_allclose(y, z, atol=1e-5)
        npt.assert_allclose(ld, 0.0, atol=1e-5)

    def test_compression_beyond_tau(self):
        """Beyond tau, output magnitude should be smaller than input magnitude."""
        z = jnp.array([200.0, -200.0, 500.0, -500.0])
        y, _ = loft_forward(z, tau=100.0)
        assert jnp.all(jnp.abs(y) < jnp.abs(z))

    def test_log_det_forward_finite(self):
        z = jax.random.normal(jax.random.PRNGKey(0), (50,)) * 200
        _, ld = loft_forward(z, tau=100.0)
        assert jnp.isfinite(ld)

    def test_log_det_consistency(self):
        """Forward + inverse log-dets should cancel (up to float precision)."""
        z = jnp.array([-300.0, -50.0, 0.0, 50.0, 300.0])
        y, ld_fwd = loft_forward(z, tau=100.0)
        _, ld_inv = loft_inverse(y, tau=100.0)
        npt.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-3)

    def test_log_det_numerical(self):
        """Verify log-det via finite differences on a scalar."""
        z_scalar = jnp.array(150.0)
        _, ld_analytic = loft_forward(z_scalar.reshape(1), tau=100.0)

        def fwd_scalar(z_):
            y, _ = loft_forward(z_.reshape(1), tau=100.0)
            return y[0]

        # log|dy/dz| via jax.grad
        grad_val = jax.grad(fwd_scalar)(z_scalar)
        ld_numerical = jnp.log(jnp.abs(grad_val))
        npt.assert_allclose(ld_analytic, ld_numerical, atol=1e-4)

    def test_batched_shapes(self):
        """LOFT should handle batched inputs."""
        z = jax.random.normal(jax.random.PRNGKey(1), (8, 20)) * 200
        y, ld = loft_forward(z, tau=100.0)
        assert y.shape == (8, 20)
        assert ld.shape == (8,)


# ===========================================================================
# FlowChain LOFT integration
# ===========================================================================


class TestFlowChainLOFT:
    """Tests for LOFT + final affine integrated into FlowChain."""

    def test_chain_with_loft_forward_inverse(self, rng):
        """FlowChain with use_loft=True should be invertible."""
        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[32, 32],
            use_loft=True,
        )
        params = chain.init(rng, jnp.zeros(6))
        x = jax.random.normal(jax.random.PRNGKey(80), (4, 6))

        z, ld_fwd = chain.apply(params, x, reverse=False)
        x_rec, ld_inv = chain.apply(params, z, reverse=True)

        npt.assert_allclose(x_rec, x, atol=0.05)
        npt.assert_allclose(ld_fwd + ld_inv, jnp.zeros(4), atol=0.05)

    def test_chain_without_loft_no_final_params(self, rng):
        """When use_loft=False, no final_mu/final_log_sigma in params."""
        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16],
            use_loft=False,
        )
        params = chain.init(rng, jnp.zeros(6))
        flat = jax.tree.leaves_with_path(params)
        final_keys = [
            str(p) for p, _ in flat if "final_mu" in str(p) or "final_log_sigma" in str(p)
        ]
        assert len(final_keys) == 0, f"Unexpected final affine params: {final_keys}"

    def test_chain_with_loft_has_final_params(self, rng):
        """When use_loft=True, final_mu and final_log_sigma are present."""
        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16],
            use_loft=True,
        )
        params = chain.init(rng, jnp.zeros(6))
        flat = jax.tree.leaves_with_path(params)
        final_keys = [
            str(p) for p, _ in flat if "final_mu" in str(p) or "final_log_sigma" in str(p)
        ]
        assert len(final_keys) == 2, f"Expected 2 final affine params, got: {final_keys}"

    def test_loft_chain_no_nan_high_dim(self, rng):
        """LOFT + soft_clamp keep high-dim flow output finite."""
        dim = 500
        chain = FlowChain(
            features=dim,
            num_layers=4,
            flow_type="affine_coupling",
            hidden_dims=[64, 64],
            use_loft=True,
            soft_clamp=True,
        )
        x = jax.random.normal(rng, (2, dim))
        params = chain.init(rng, x[0])

        z, ld = chain.apply(params, x)
        assert jnp.all(jnp.isfinite(z)), "LOFT chain forward NaN/Inf"
        assert jnp.all(jnp.isfinite(ld)), "LOFT chain log_det NaN/Inf"

    def test_spline_with_loft(self, rng):
        """Spline coupling + LOFT round-trip."""
        chain = FlowChain(
            features=6,
            num_layers=2,
            flow_type="spline_coupling",
            hidden_dims=[32, 32],
            use_loft=True,
        )
        params = chain.init(rng, jnp.zeros(6))
        x = jax.random.normal(jax.random.PRNGKey(81), (3, 6)) * 2.0

        z, ld_fwd = chain.apply(params, x, reverse=False)
        x_rec, ld_inv = chain.apply(params, z, reverse=True)

        npt.assert_allclose(x_rec, x, atol=0.05)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(ld_fwd))


# ===========================================================================
# Soft clamp integration (AffineCoupling)
# ===========================================================================


class TestAffineSoftClamp:
    """Verify soft_clamp toggle on AffineCoupling."""

    def test_soft_clamp_on_forward_inverse(self, rng):
        """Affine coupling with soft_clamp=True should be invertible."""
        flow = AffineCoupling(
            features=6, hidden_dims=[32, 32], soft_clamp=True
        )
        params = flow.init(rng, jnp.zeros(6))
        x = jax.random.normal(jax.random.PRNGKey(90), (4, 6))

        y, ld_fwd = flow.apply(params, x)
        x_rec, ld_inv = flow.apply(params, y, reverse=True)
        npt.assert_allclose(x_rec, x, atol=1e-5)
        npt.assert_allclose(ld_fwd + ld_inv, jnp.zeros(4), atol=1e-5)

    def test_soft_clamp_off_forward_inverse(self, rng):
        """Affine coupling with soft_clamp=False (hard clip) should also work."""
        flow = AffineCoupling(
            features=6, hidden_dims=[32, 32], soft_clamp=False
        )
        params = flow.init(rng, jnp.zeros(6))
        x = jax.random.normal(jax.random.PRNGKey(91), (4, 6))

        y, ld_fwd = flow.apply(params, x)
        x_rec, ld_inv = flow.apply(params, y, reverse=True)
        npt.assert_allclose(x_rec, x, atol=1e-5)

    def test_soft_clamp_identity_at_init(self, rng):
        """With zero_init_output + soft_clamp, flow is identity at init."""
        flow = AffineCoupling(
            features=10,
            hidden_dims=[32, 32],
            zero_init_output=True,
            soft_clamp=True,
        )
        x = jax.random.normal(rng, (3, 10))
        params = flow.init(rng, x[0])

        y, ld = flow.apply(params, x)
        npt.assert_allclose(y, x, atol=1e-6)
        npt.assert_allclose(ld, jnp.zeros(3), atol=1e-6)


# ===========================================================================
# Float64 log-det accumulation
# ===========================================================================


class TestLogDetF64:
    """Verify ``log_det_f64`` flag controls the dtype of the log-det output."""

    def test_log_det_f64_produces_float64(self, rng):
        """With log_det_f64=True and x64 enabled, log_det should be float64."""
        with jax.enable_x64(True):
            chain = FlowChain(
                features=8,
                num_layers=2,
                flow_type="affine_coupling",
                hidden_dims=[16, 16],
                log_det_f64=True,
            )
            x = jax.random.normal(rng, (4, 8))
            params = chain.init(rng, x[0])

            # Forward
            _, ld_fwd = chain.apply(params, x)
            assert ld_fwd.dtype == jnp.float64, (
                f"Expected float64, got {ld_fwd.dtype}"
            )

            # Inverse
            _, ld_inv = chain.apply(params, x, reverse=True)
            assert ld_inv.dtype == jnp.float64, (
                f"Expected float64, got {ld_inv.dtype}"
            )

    def test_log_det_f32_by_default(self, rng):
        """With log_det_f64=False (default), log_det should be float32."""
        chain = FlowChain(
            features=8,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
            log_det_f64=False,
        )
        x = jax.random.normal(rng, (4, 8))
        params = chain.init(rng, x[0])

        _, ld_fwd = chain.apply(params, x)
        assert ld_fwd.dtype == jnp.float32, (
            f"Expected float32, got {ld_fwd.dtype}"
        )

    def test_log_det_f64_with_loft(self, rng):
        """Float64 log-det should work when LOFT is enabled."""
        with jax.enable_x64(True):
            chain = FlowChain(
                features=8,
                num_layers=2,
                flow_type="affine_coupling",
                hidden_dims=[16, 16],
                use_loft=True,
                log_det_f64=True,
            )
            x = jax.random.normal(rng, (4, 8))
            params = chain.init(rng, x[0])

            _, ld_fwd = chain.apply(params, x)
            assert ld_fwd.dtype == jnp.float64

            _, ld_inv = chain.apply(params, x, reverse=True)
            assert ld_inv.dtype == jnp.float64

    def test_log_det_f64_invertibility(self, rng):
        """Forward + inverse log-dets should cancel even in float64."""
        with jax.enable_x64(True):
            chain = FlowChain(
                features=8,
                num_layers=2,
                flow_type="affine_coupling",
                hidden_dims=[16, 16],
                log_det_f64=True,
            )
            x = jax.random.normal(rng, (4, 8))
            params = chain.init(rng, x[0])

            y, ld_fwd = chain.apply(params, x)
            x_rec, ld_inv = chain.apply(params, y, reverse=True)
            npt.assert_allclose(x_rec, x, atol=1e-5)
            npt.assert_allclose(ld_fwd + ld_inv, jnp.zeros(4), atol=1e-5)

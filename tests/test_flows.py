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

    def test_near_identity_at_init(self, flow_and_params):
        """With zero-initialized conditioner, transform should be near-identity.

        Not exact because softplus(0)+eps derivative != bin slope. But the
        deviation should be small (< 0.1 per dimension).
        """
        flow, params = flow_and_params
        x = jax.random.normal(jax.random.PRNGKey(22), (4, 6)) * 2.0
        y, log_det = flow.apply(params, x)
        npt.assert_allclose(y, x, atol=0.1)
        npt.assert_allclose(log_det, jnp.zeros(4), atol=0.5)

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
        chain = FlowChain(
            features=4,
            num_layers=2,
            flow_type="affine_coupling",
            hidden_dims=[16, 16],
            covariate_specs=covariate_specs,
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

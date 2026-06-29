"""Tests for the Gauss-Jacobi quadrature backend.

Validates the new ``scribe.stats._jacobi_quad`` module against:

- closed-form Beta moments (E[p], E[p²]) over a grid of (α, β),
- the floor values used by the two-state likelihood reparam,
- JAX differentiability through the backend,
- the precomputed-grid backend stub.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scribe.stats._jacobi_quad import (
    GOLUB_WELSCH,
    PRECOMPUTED_GRID,
    gauss_jacobi_nodes_weights,
)


# Floor values used by the two-state reparam helper. The quadrature
# must remain accurate when the Beta shapes touch these.
_ALPHA_FLOOR = 0.05
_K_OFF_FLOOR = 0.05


def _beta_moment_via_quadrature(alpha, beta, n_nodes, power):
    """Approximate ∫₀¹ p^power · Beta(p|α, β) dp via Gauss-Jacobi.

    Runs at whatever dtype JAX is configured for (float32 by default).
    """
    nodes, log_w = gauss_jacobi_nodes_weights(
        jnp.asarray(alpha),
        jnp.asarray(beta),
        n_nodes,
    )
    return float(jnp.sum(jnp.exp(log_w) * nodes**power))


def _beta_moment_closed_form(alpha, beta, power):
    """Closed-form: E[p^k] under Beta(α, β) = ∏ (α + j) / (α + β + j)."""
    out = 1.0
    for j in range(power):
        out *= (alpha + j) / (alpha + beta + j)
    return out


# ==============================================================================
# Closed-form Beta-moment recovery
# ==============================================================================


class TestQuadratureBetaMomentRecovery:
    """E[p^k] should be recovered within tight absolute tolerance."""

    @pytest.mark.parametrize(
        "alpha, beta",
        [
            # The floor values themselves — explicit per audit feedback.
            (_ALPHA_FLOOR, _ALPHA_FLOOR),
            (_ALPHA_FLOOR, 50.0),
            (50.0, _K_OFF_FLOOR),
            # A spread covering the bursty (< 1), intermediate, and
            # concentrated (>> 1) regimes.
            (0.1, 0.1),
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 2.0),
            (5.0, 1.0),
            (1.0, 5.0),
            (50.0, 50.0),
            (100.0, 0.5),
        ],
    )
    def test_first_moment(self, alpha, beta):
        """E[p] = α / (α + β) within 5e-4 absolute (float32 default).

        Tolerance accommodates the singularity-avoidance nudge of 1e-3
        in (a, b) inside the recurrence, which shifts the effective
        Beta parameters by 1e-3. For asymmetric (α ≠ β) cases this
        translates to a moment error scaling as |α − β| · NUDGE /
        (α + β)², bounded by ~3e-4 across the test grid.
        """
        expected = _beta_moment_closed_form(alpha, beta, 1)
        got = _beta_moment_via_quadrature(alpha, beta, 60, 1)
        assert np.isclose(got, expected, atol=5e-4), (
            f"alpha={alpha}, beta={beta}: got {got}, expected {expected}"
        )

    @pytest.mark.parametrize(
        "alpha, beta",
        [
            (_ALPHA_FLOOR, _ALPHA_FLOOR),
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 5.0),
            (50.0, 0.5),
        ],
    )
    def test_second_moment(self, alpha, beta):
        """E[p²] within 1e-3 absolute (float32 default).

        The singularity-avoidance nudge perturbs the effective Beta
        shapes by ~1e-3, which dominates the E[p²] error in the
        bursty regime (α, β → 0). For α, β ≥ 0.5 the error is < 1e-4.
        """
        expected = _beta_moment_closed_form(alpha, beta, 2)
        got = _beta_moment_via_quadrature(alpha, beta, 60, 2)
        assert np.isclose(got, expected, atol=1e-3), (
            f"alpha={alpha}, beta={beta}: got {got}, expected {expected}"
        )


# ==============================================================================
# Output shape / normalisation
# ==============================================================================


class TestQuadratureShapes:
    """Shape contract: nodes and weights at ``broadcast(alpha, beta) + (K,)``."""

    def test_scalar_inputs(self):
        nodes, log_w = gauss_jacobi_nodes_weights(
            jnp.array(2.0), jnp.array(2.0), 8
        )
        assert nodes.shape == (8,)
        assert log_w.shape == (8,)
        # Weights sum to 1 (Beta integrates to 1), with float32
        # round-off tolerance.
        assert np.isclose(float(jnp.exp(log_w).sum()), 1.0, atol=1e-6)

    def test_gene_rank_inputs(self):
        alpha = jnp.array([0.5, 2.0, 50.0])
        beta = jnp.array([0.5, 2.0, 50.0])
        nodes, log_w = gauss_jacobi_nodes_weights(alpha, beta, 16)
        assert nodes.shape == (3, 16)
        assert log_w.shape == (3, 16)
        sums = jnp.exp(log_w).sum(axis=-1)
        np.testing.assert_allclose(np.asarray(sums), 1.0, atol=1e-6)

    def test_nodes_lie_in_unit_interval(self):
        alpha = jnp.array([0.05, 1.0, 50.0])
        beta = jnp.array([0.05, 1.0, 50.0])
        nodes, _ = gauss_jacobi_nodes_weights(alpha, beta, 40)
        assert float(nodes.min()) >= 0.0
        assert float(nodes.max()) <= 1.0


# ==============================================================================
# Differentiability through Golub-Welsch
# ==============================================================================


class TestQuadratureDifferentiability:
    """``jax.grad`` through the backend should produce finite gradients."""

    def test_grad_through_first_moment(self):
        """∂ E[p] / ∂α should be finite and have the right sign."""

        def first_moment(alpha):
            nodes, log_w = gauss_jacobi_nodes_weights(
                alpha, jnp.array(2.0), 40
            )
            return jnp.sum(jnp.exp(log_w) * nodes)

        grad_fn = jax.grad(first_moment)
        g = float(grad_fn(jnp.array(2.0)))
        assert np.isfinite(g)
        # dE[p]/dα = β / (α + β)² > 0 → at α=β=2 this is 2/16 = 0.125
        assert np.isclose(g, 0.125, atol=1e-2)


# ==============================================================================
# Backend dispatch
# ==============================================================================


class TestBackendDispatch:
    """Public API rejects unknown backends and stubs the phase-2 grid."""

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown Gauss-Jacobi backend"):
            gauss_jacobi_nodes_weights(
                jnp.array(1.0), jnp.array(1.0), 8, backend="not_a_backend"
            )

    def test_precomputed_grid_is_not_implemented(self):
        with pytest.raises(NotImplementedError, match="phase 2"):
            gauss_jacobi_nodes_weights(
                jnp.array(1.0), jnp.array(1.0), 8, backend=PRECOMPUTED_GRID
            )

    def test_golub_welsch_constant_is_default(self):
        assert GOLUB_WELSCH == "golub_welsch"
